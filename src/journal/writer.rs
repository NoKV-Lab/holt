//! `WalWriter` — append-only WAL file writer.
//!
//! Lifecycle:
//!
//! 1. [`WalWriter::create`] for a fresh file or
//!    [`WalWriter::open_existing`] to resume an existing one.
//! 2. [`WalWriter::append`] for each `TxnOp` — bytes land in an
//!    in-memory buffer. When the buffer crosses
//!    [`AUTO_FLUSH_THRESHOLD`] (64 KB), `append` transparently
//!    drains it to the OS via `write_all` (no `sync_data`). This
//!    is group-commit: the WAL bytes reach the OS page cache in
//!    big batches without the user-space buffer growing
//!    unboundedly, but the per-record cost is still just an
//!    in-memory copy.
//! 3. [`WalWriter::flush`] drains whatever is still pending and
//!    runs `sync_data` so every record so far is durable past a
//!    power failure. This is the **durability boundary**.
//! 4. Drop is a no-op — callers are responsible for the final
//!    `flush` (the WAL semantic is "what's flushed is durable;
//!    what's been auto-drained to page cache survives a process
//!    crash but not a power loss until you `flush`").
//!
//! `Tree::checkpoint` calls [`WalWriter::truncate`] to trim
//! records past the last durable blob commit.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::api::errors::{Error, Result};

use super::codec::{
    decode_file_header, encode_erase_record, encode_file_header, encode_insert_record,
    encode_record, encode_rename_object_record, BatchEncoder, FileHeader, FILE_HEADER_SIZE,
};
use super::txn_op::TxnOp;

/// Append's in-memory buffer is auto-drained to the OS page
/// cache once it crosses this many bytes. Drops user-space
/// buffering pressure without forcing a `sync_data` per record.
///
/// 64 KB is a coarse pick: large enough that the per-record
/// syscall overhead is amortised across hundreds of records,
/// small enough that the worst-case in-flight loss on a crash
/// is bounded.
pub const AUTO_FLUSH_THRESHOLD: usize = 64 * 1024;

/// Append-only WAL writer with explicit `flush`-for-durability.
#[derive(Debug)]
pub struct WalWriter {
    /// Path of the underlying file — needed by `truncate` to
    /// atomically replace the live log on checkpoint.
    path: PathBuf,
    /// Underlying file handle, opened in append mode.
    file: File,
    /// Buffered bytes not yet handed to the OS.
    pending: Vec<u8>,
    /// Sum of `pending.len()` over the lifetime of this writer
    /// (durable + in-flight). Useful for stats / tests.
    bytes_written: u64,
    /// File-header info recovered on open.
    header: FileHeader,
}

impl WalWriter {
    /// Create a fresh WAL file at `path` and write the file header.
    ///
    /// Returns an error if `path` already exists — use
    /// [`WalWriter::open_existing`] to append to an existing log.
    pub fn create(path: &Path, tree_id: u64) -> Result<Self> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let header = FileHeader::now(tree_id);
        let mut buf = Vec::with_capacity(FILE_HEADER_SIZE);
        encode_file_header(&header, &mut buf);
        file.write_all(&buf)?;
        file.sync_data()?;
        // Reopen in append mode so subsequent writes go to the end
        // even if other code seeks around.
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            pending: Vec::with_capacity(4096),
            bytes_written: FILE_HEADER_SIZE as u64,
            header,
        })
    }

    /// Open an existing WAL file for append. Validates the file
    /// header and returns the parsed `FileHeader` via
    /// [`WalWriter::header`].
    pub fn open_existing(path: &Path) -> Result<Self> {
        let mut header_bytes = [0u8; FILE_HEADER_SIZE];
        {
            let mut f = File::open(path)?;
            use std::io::Read;
            f.read_exact(&mut header_bytes)?;
        }
        let header = decode_file_header(&header_bytes)?;
        let file = OpenOptions::new().append(true).open(path)?;
        let bytes_written = file.metadata()?.len();
        Ok(Self {
            path: path.to_path_buf(),
            file,
            pending: Vec::with_capacity(4096),
            bytes_written,
            header,
        })
    }

    /// Open existing or create fresh. The file's recorded
    /// `tree_id` is **not** rewritten when opening — pass the
    /// expected `tree_id` so a mismatch (a wrong tree's WAL) can
    /// surface as an error.
    pub fn open_or_create(path: &Path, tree_id: u64) -> Result<Self> {
        if path.exists() {
            let w = Self::open_existing(path)?;
            if w.header.tree_id != tree_id {
                return Err(Error::ReplaySanityFailed {
                    context: "WAL file tree_id mismatch on open",
                    record_offset: 0,
                });
            }
            Ok(w)
        } else {
            Self::create(path, tree_id)
        }
    }

    /// Header recovered on open, including the embedded tree id.
    ///
    /// Diagnostic — verified by codec round-trip tests but not
    /// directly read on the `Tree::open` hot path.
    #[allow(dead_code)]
    #[must_use]
    pub fn header(&self) -> FileHeader {
        self.header
    }

    /// Bytes written (durable + buffered) since the file was
    /// created — useful as a stand-in offset for telemetry.
    #[allow(dead_code)]
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written + self.pending.len() as u64
    }

    /// Stage a single `TxnOp` for the next flush.
    ///
    /// The record is encoded into the pending buffer in memory.
    /// If the buffer crosses [`AUTO_FLUSH_THRESHOLD`] the writer
    /// transparently drains it to the OS via `write_all` (no
    /// `sync_data`) — bounded user-space buffering, but per-op
    /// cost stays at an in-memory copy.
    ///
    /// All four user-facing hot paths
    /// (`Tree::put` / `delete` / `rename` / `txn`) use the
    /// per-variant fast paths instead ([`Self::append_insert`] /
    /// [`Self::append_erase`] / [`Self::append_rename_object`] /
    /// [`Self::append_batch`]). Those skip the `TxnOp` enum's
    /// `Vec` clones — measurable against the bench's hot path.
    /// `append` stays as a generic test-time entry point for
    /// exercising structural variants (Split / Merge / Compact /
    /// MemMarker / NewTree / RmTree) end-to-end through the
    /// writer + replay path; production code never goes through
    /// it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn append(&mut self, op: &TxnOp, seq: u64) -> Result<()> {
        encode_record(op, seq, &mut self.pending);
        self.maybe_drain()
    }

    /// Fast-path: append an `Insert` record directly from refs.
    /// Skips the `TxnOp::Insert` enum + the two `Vec::to_vec`
    /// clones (key + value).
    pub fn append_insert(
        &mut self,
        seq: u64,
        tree_id: u64,
        key: &[u8],
        value: &[u8],
    ) -> Result<()> {
        encode_insert_record(&mut self.pending, seq, tree_id, key, value);
        self.maybe_drain()
    }

    /// Fast-path: append an `Erase` record directly from refs.
    /// Carries key only; the prior value (if the caller wanted it)
    /// is delivered through `Tree::remove`'s return path, not
    /// through the WAL.
    pub fn append_erase(&mut self, seq: u64, tree_id: u64, key: &[u8]) -> Result<()> {
        encode_erase_record(&mut self.pending, seq, tree_id, key);
        self.maybe_drain()
    }

    /// Fast-path: append a `RenameObject` record directly from
    /// refs.
    pub fn append_rename_object(
        &mut self,
        seq: u64,
        tree_id: u64,
        src_key: &[u8],
        dst_key: &[u8],
        force: bool,
    ) -> Result<()> {
        encode_rename_object_record(&mut self.pending, seq, tree_id, src_key, dst_key, force);
        self.maybe_drain()
    }

    /// Fast-path: stream a `Batch` envelope record directly from
    /// refs. Hands the caller a [`BatchEncoder`] to push primitive
    /// inner ops (`Insert` / `Erase` / `RenameObject`) without
    /// constructing intermediate `TxnOp::Insert { key: Vec, value:
    /// Vec, .. }` values — the per-op `Vec` clones that
    /// `encode_record` would force are skipped.
    ///
    /// The closure is `FnOnce(&mut BatchEncoder<'_>) -> Result<()>`:
    /// returning `Err` cleanly aborts the batch (the encoder's
    /// `Drop` rolls back the partial record bytes so the WAL
    /// buffer ends up exactly where `begin` found it). On `Ok`
    /// the encoder is `finish`ed (backpatch + CRC) and the
    /// auto-drain threshold is re-checked.
    ///
    /// Returns the final inner op count (= the number of
    /// `push_*` calls that ran inside the closure). The caller
    /// can use this to skip emitting the record at all when the
    /// effective inner count is 0 (e.g. a batch of pure no-op
    /// deletes) — though calling `append_batch` with zero ops
    /// still writes a well-formed empty-Batch record, matching
    /// the generic `encode_record(&TxnOp::Batch { ops: vec![] })`
    /// shape.
    pub fn append_batch<F>(&mut self, base_seq: u64, tree_id: u64, build: F) -> Result<u32>
    where
        F: FnOnce(&mut BatchEncoder<'_>) -> Result<()>,
    {
        let mut enc = BatchEncoder::begin(&mut self.pending, base_seq, tree_id);
        build(&mut enc)?;
        let n = enc.finish();
        self.maybe_drain()?;
        Ok(n)
    }

    fn maybe_drain(&mut self) -> Result<()> {
        if self.pending.len() >= AUTO_FLUSH_THRESHOLD {
            self.drain_to_os()?;
        }
        Ok(())
    }

    /// Drain pending bytes to the OS without forcing a sync.
    /// After this call the bytes are in the page cache; survive
    /// a process crash but not a power loss until `sync_data`
    /// (i.e. `flush`) runs.
    fn drain_to_os(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.file.write_all(&self.pending)?;
        self.bytes_written += self.pending.len() as u64;
        self.pending.clear();
        Ok(())
    }

    /// Drain pending bytes to the OS and `sync_data` so every
    /// record appended so far is durable past a power loss.
    ///
    /// On platforms where `sync_data` is a no-op (memory-only
    /// filesystems used in CI / tests), durability falls back to
    /// whatever the OS provides — the bytes still land in the
    /// page cache.
    pub fn flush(&mut self) -> Result<()> {
        self.drain_to_os()?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Drop pending records without writing them. Useful when a
    /// caller decides mid-batch to bail out (e.g. precondition
    /// check failed). Records already `flush`ed or auto-drained
    /// are unaffected — `discard_pending` only touches the
    /// in-memory tail since the last drain.
    ///
    /// Currently exercised by journal unit tests only; the
    /// `Tree::txn` mid-batch bail path doesn't call this yet.
    #[allow(dead_code)]
    pub fn discard_pending(&mut self) {
        self.pending.clear();
    }

    /// Atomically reset the log to a fresh, header-only file.
    ///
    /// Used by `Tree::checkpoint` once every record up through the
    /// last `flush` is reflected in a durable blob commit: the WAL
    /// has nothing the blob doesn't already have, so we drop the
    /// records and start growing the log again from zero.
    ///
    /// Implementation: write a fresh header to a sibling temp file
    /// (`<path>.tmp`), `sync_data` it, atomically `rename` over
    /// the live log, and reopen the file handle. The temp file's
    /// pre-rename header has the **same** tree_id; `created_at`
    /// gets refreshed to the current wall clock.
    ///
    /// Any `pending` records buffered since the last `flush` are
    /// dropped — call `flush()` first if they matter.
    pub fn truncate(&mut self) -> Result<()> {
        self.pending.clear();

        let tmp_path = self.path.with_extension("wal.tmp");
        {
            let mut tmp = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            let header = FileHeader::now(self.header.tree_id);
            let mut buf = Vec::with_capacity(FILE_HEADER_SIZE);
            encode_file_header(&header, &mut buf);
            tmp.write_all(&buf)?;
            tmp.sync_data()?;
            self.header = header;
        }

        // Atomic replace — on POSIX, `rename` is atomic, and the
        // new fd inherits the durable state of the temp file.
        std::fs::rename(&tmp_path, &self.path)?;

        // The previous append-mode handle is now pointing at the
        // unlinked old inode; reopen against the new file.
        self.file = OpenOptions::new().append(true).open(&self.path)?;
        self.bytes_written = FILE_HEADER_SIZE as u64;

        #[cfg(feature = "tracing")]
        tracing::info!(target: "holt::wal", "wal truncated to header-only");

        Ok(())
    }
}
