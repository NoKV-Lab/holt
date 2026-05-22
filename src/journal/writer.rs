//! `WalWriter` — append-only WAL file writer.
//!
//! Lifecycle:
//!
//! 1. [`WalWriter::create`] for a fresh file or
//!    [`WalWriter::open_existing`] to resume an existing one.
//! 2. Encoded records are appended into an in-memory buffer.
//!    When the buffer crosses [`AUTO_FLUSH_THRESHOLD`] (64 KB),
//!    the writer transparently drains it to the OS via `write_all`
//!    (no `sync_data`). The higher-level group-commit worker
//!    controls which append waiters share a durability flush.
//! 3. [`WalWriter::flush`] drains whatever is still pending and
//!    runs `sync_data` so every record so far is durable past a
//!    power failure. This is the **durability boundary**.
//! 4. Drop is a no-op — callers are responsible for the final
//!    `flush` (the WAL semantic is "what's flushed is durable;
//!    what's been auto-drained to page cache survives a process
//!    crash but not a power loss until you `flush`").
//!
//! The group-commit worker calls [`WalWriter::truncate`] when
//! `Tree::checkpoint` proves every WAL record is reflected in the
//! durable blob image.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use crate::api::errors::{Error, Result};

#[cfg(test)]
use super::codec::encode_record;
use super::codec::{decode_file_header, encode_file_header, FileHeader, FILE_HEADER_SIZE};
#[cfg(test)]
use super::wal_op::WalOp;

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
    #[cfg(test)]
    #[must_use]
    pub fn header(&self) -> FileHeader {
        self.header
    }

    /// Bytes written (durable + buffered) since the file was created.
    #[cfg(test)]
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written + self.pending.len() as u64
    }

    /// True when this WAL contains bytes beyond the fixed file
    /// header. Used by the group-commit worker to distinguish a
    /// genuinely clean WAL from an opened log that replay already
    /// consumed but checkpoint has not truncated yet.
    #[must_use]
    pub(crate) fn has_records(&self) -> bool {
        self.bytes_written + self.pending.len() as u64 > FILE_HEADER_SIZE as u64
    }

    /// Stage a single `WalOp` for the next flush.
    ///
    /// The record is encoded into the pending buffer in memory.
    /// If the buffer crosses [`AUTO_FLUSH_THRESHOLD`] the writer
    /// transparently drains it to the OS via `write_all` (no
    /// `sync_data`) — bounded user-space buffering, but per-op
    /// cost stays at an in-memory copy.
    ///
    /// Generic test-time entry point for exercising enum encoding
    /// end-to-end through the writer + replay path. Production
    /// hot paths encode records before handing them to the
    /// group-commit journal worker.
    #[cfg(test)]
    pub fn append(&mut self, op: &WalOp, seq: u64) -> Result<()> {
        encode_record(op, seq, &mut self.pending);
        self.maybe_drain()
    }

    /// Append one already-encoded WAL record.
    ///
    /// Used by the group-commit worker: foreground threads encode
    /// into owned buffers, then the worker serially appends those
    /// bytes to this writer and optionally flushes the whole batch.
    pub(crate) fn append_encoded(&mut self, record: &[u8]) -> Result<()> {
        self.pending.extend_from_slice(record);
        self.maybe_drain()
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
    pub(crate) fn drain_to_os(&mut self) -> Result<()> {
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
    /// Test helper for rollback semantics.
    #[cfg(test)]
    pub fn discard_pending(&mut self) {
        self.pending.clear();
    }

    /// Reset the log to the existing header-only prefix.
    ///
    /// Used by `Tree::checkpoint` once every record up through the
    /// last `flush` is reflected in a durable blob commit: the WAL
    /// has nothing the blob doesn't already have, so we drop the
    /// records and start growing the log again from zero.
    ///
    /// Implementation: drop the in-memory tail, `ftruncate` the
    /// live WAL back to [`FILE_HEADER_SIZE`], then `sync_data`.
    /// After checkpoint the old records are redundant: if a crash
    /// leaves the old file length in place, replay is safe; if it
    /// leaves the truncated length, store is already durable.
    /// Avoiding temp-file rename keeps the checkpoint tail cheap.
    ///
    /// Any `pending` records buffered since the last `flush` are
    /// dropped — call `flush()` first if they matter.
    pub fn truncate(&mut self) -> Result<()> {
        self.pending.clear();
        self.file.set_len(FILE_HEADER_SIZE as u64)?;
        self.file.sync_data()?;
        self.bytes_written = FILE_HEADER_SIZE as u64;

        #[cfg(feature = "tracing")]
        tracing::info!(target: "holt::wal", "wal truncated to header-only");

        Ok(())
    }
}
