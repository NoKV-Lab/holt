//! Logical WAL record codec — binary encoding for [`WalOp`].
//!
//! Each record on disk has the shape
//!
//! ```text
//! +------+------+------+----+-----------+------+------+
//! | MAGIC| LEN  | SEQ  | TY |   BODY    | CRC32| LEN2 |
//! | u32  | u32  | u64  | u8 |  varlen   | u32  | u32  |
//! +------+------+------+----+-----------+------+------+
//!
//!  ^                                       ^
//!  |--------- CRC32 covers everything -----|
//!  |    from MAGIC through end of BODY     |
//! ```
//!
//! - `MAGIC` (`0x5243_4552`, ASCII `"RECR"` little-endian) marks
//!   the start of every record. Lets replay resync after a torn
//!   write at the end of the log.
//! - `LEN` = byte length of `BODY` only (not header, not footer).
//! - `SEQ` = monotonic sequence stamped by the engine. Replay
//!   uses it to skip ops already reflected in the last checkpoint
//!   and to resume `next_seq` after restart.
//! - `TY` = one-byte variant tag (stable on disk; see the
//!   `TY_*` constants).
//! - `BODY` = variant-specific bytes; see the per-variant encoder
//!   functions and `decode_body` for the exact layout per variant.
//! - `CRC32` (IEEE 802.3 polynomial `0xEDB8_8320`) detects torn
//!   writes and silent disk corruption.
//! - `LEN2` repeats the body length. Replay uses this independent framing
//!   witness to distinguish a torn tail from a corrupted header length that
//!   would otherwise swallow later acknowledged records.
//!
//! All integers are little-endian. All length-prefixed byte
//! strings (keys, values, tree names) use a `u32` LE length
//! followed by raw bytes.

use super::wal_op::WalOp;
use crate::api::errors::{Error, Result};

/// Start-of-record magic — `"RECR"` little-endian.
pub const RECORD_MAGIC: u32 = 0x5243_4552;

/// Fixed-size header bytes: `magic | len | seq | ty`.
pub const RECORD_HEADER_SIZE: usize = 4 + 4 + 8 + 1;

/// Fixed-size footer bytes: `crc32 | repeated_body_len`.
pub const RECORD_FOOTER_SIZE: usize = 4 + 4;

/// Native ceiling for one encoded atomic WAL record.
///
/// The WAL body length field could address more, but Holt deliberately caps a
/// single commit at 256 MiB so the on-demand oversized-record lane has a hard
/// memory bound. The ordinary journal ring remains much smaller.
pub const MAX_ATOMIC_WAL_RECORD_BYTES: usize = 256 * 1024 * 1024;
/// Maximum logical primitive operations carried by one atomic WAL record.
///
/// Compact runs can encode zero-length keys/values without consuming one byte
/// per logical op, so the byte ceiling alone is not a CPU bound for replay.
pub const MAX_ATOMIC_WAL_OPS: usize = 65_536;

// ---------- File header ----------

/// Top-of-file magic — `"WALA"` little-endian. Sits at offset 0 of
/// every WAL file and is checked on open. Mismatch = "this isn't
/// one of our WAL files".
pub const FILE_MAGIC: u32 = 0x414C_4157;

/// Format version stored in the file header. New format revisions
/// bump this and grow the header (in the reserved tail) rather
/// than moving existing fields.
///
/// v0.8.2 ships format `4`: `RenameObject` now carries the source value
/// captured by the committing operation, and every record footer repeats its
/// body length. Format 3 WALs fail closed on startup: Holt intentionally does
/// not dual-read a redo shape whose rename semantics were not deterministic.
///
/// Format `3` dropped the dead `prev_value` field
/// from `WalOp::Insert` and the dead `value` field from
/// `WalOp::Erase`. Both were "for replay reversibility" but
/// replay never undoes — it's an idempotent forward redo that
/// only consumes `key, value` (Insert) / `key` (Erase). The
/// trailing `optional_bytes` slot is gone from both record bodies.
///
/// Older internal v0.3 draft binaries that still wrote format `2`
/// would mis-parse the absent slot as a length prefix; the
/// file-header check rejects that upgrade with "format version
/// unsupported" rather than silently corrupting state on replay.
/// Upgrade path for any local draft data: checkpoint the old tree
/// first so the WAL is truncated before opening it with v0.3.0.
/// The v0.2 → v0.3.0 public upgrade follows the same
/// "checkpoint before upgrade" rule.
pub const FORMAT_VERSION: u32 = 4;

/// File-header byte size. The record stream starts at this offset.
pub const FILE_HEADER_SIZE: usize = 32;

/// Top-of-file layout:
///
/// ```text
/// +------+------+------+--------+--------+
/// | MAGIC|  VER | TREE | CREATED|  RSVD  |
/// |  u32 |  u32 |  u64 |   u64  |  u64   |
/// +------+------+------+--------+--------+
/// ```
///
/// - `MAGIC` = [`FILE_MAGIC`] (`"WALA"` LE).
/// - `VER`   = [`FORMAT_VERSION`].
/// - `TREE`  = tree owner identifier; `0` for the single-tree API.
/// - `CREATED` = unix epoch seconds; `0` when the writer chose
///   not to stamp a time (e.g. tests).
/// - `RSVD`  = reserved for a future version bump, must be `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    /// Tree owner identifier.
    pub tree_id: u64,
    /// Unix-epoch seconds when the file was created. `0` if the
    /// writer didn't stamp one.
    pub created_at: u64,
}

impl FileHeader {
    /// Build a header with the current wall clock.
    #[must_use]
    pub fn now(tree_id: u64) -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            tree_id,
            created_at,
        }
    }
}

/// Encode the file header into the first [`FILE_HEADER_SIZE`] bytes
/// of `out` (the buffer is resized as needed).
pub fn encode_file_header(h: &FileHeader, out: &mut Vec<u8>) {
    out.extend_from_slice(&FILE_MAGIC.to_le_bytes());
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&h.tree_id.to_le_bytes());
    out.extend_from_slice(&h.created_at.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    debug_assert_eq!(out.len(), FILE_HEADER_SIZE);
}

/// Decode a file header from the first [`FILE_HEADER_SIZE`] bytes
/// of `buf`. Returns the header on success and a sanity-failed
/// error (with `record_offset = 0`) on mismatch.
pub fn decode_file_header(buf: &[u8]) -> Result<FileHeader> {
    if buf.len() < FILE_HEADER_SIZE {
        return Err(sanity("WAL file header truncated"));
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != FILE_MAGIC {
        return Err(sanity("WAL file magic mismatch"));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(sanity("WAL file format version unsupported"));
    }
    let tree_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let created_at = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    // bytes 24..32 reserved; ignore for forward-compatibility.
    Ok(FileHeader {
        tree_id,
        created_at,
    })
}

// On-disk variant tags. Stable within format v4; only ever add new
// tags, never renumber existing ones. Tags 2..4 and 6..9 are
// intentionally unassigned in production: an internal v0.3 draft
// had non-emitted structural / multi-tree variants there, but
// Holt's recovery contract is logical redo plus checkpointed blob
// images, not standalone structural WAL replay.
const TY_INSERT: u8 = 0;
const TY_ERASE: u8 = 1;
const TY_RENAME_OBJECT: u8 = 5;
const TY_BATCH: u8 = 10;
const TY_BATCH_INSERT_RUN: u8 = 11;
const TY_BATCH_INSERT_PREFIX_RUN: u8 = 12;

// ---------- CRC32 (IEEE 802.3) ----------

/// CRC32 — IEEE 802.3 polynomial `0xEDB8_8320`, reflected
/// (i.e. the variant `gzip` / `PNG` / RocksDB block-checksum
/// use). Used as the record-level `sanity_info`.
///
/// Routes to [`crc32fast`], which auto-detects PCLMULQDQ on
/// x86_64 and the `CRC32` instruction on AArch64 at first call
/// and dispatches via function pointer afterwards. On supported
/// hardware (≈Skylake+, Apple Silicon, recent ARM cores) that's
/// ≈8-12 GB/s; the fallback `slice-by-16` table-driven path on
/// older cores is still well ahead of a byte-at-a-time loop.
pub fn crc32(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

// ---------- encode ----------

/// Test-only generic encoder for `WalOp` variants.
///
/// Production hot paths use the per-variant encoders below. Keeping
/// this generic enum path out of release builds prevents it from
/// becoming a second supported mutation surface.
#[cfg(test)]
pub fn encode_record(op: &WalOp, seq: u64, out: &mut Vec<u8>) {
    write_record(out, seq, variant_tag(op), |buf| encode_body(op, buf));
}

/// Internal: lay down the fixed record header, run the
/// variant-specific body writer, backpatch the body length, and
/// append the CRC32 footer.
fn write_record<F>(out: &mut Vec<u8>, seq: u64, ty: u8, write_body: F)
where
    F: FnOnce(&mut Vec<u8>),
{
    let start = out.len();
    out.extend_from_slice(&RECORD_MAGIC.to_le_bytes());
    let len_pos = out.len();
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&seq.to_le_bytes());
    out.push(ty);

    let body_start = out.len();
    write_body(out);
    let body_end = out.len();
    let body_len = u32::try_from(body_end - body_start).expect("body fits in u32");
    out[len_pos..len_pos + 4].copy_from_slice(&body_len.to_le_bytes());

    let crc = crc32(&out[start..body_end]);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&body_len.to_le_bytes());
}

// ---------- per-variant fast-path encoders ----------
//
// These mirror the variants `Tree::put` / `delete` / `rename`
// hit on the hot path. They take borrowed bytes rather than
// constructing a `WalOp` enum, so callers don't pay for the
// `Vec` clones that enum construction forces.

/// Encode an `Insert` record directly from refs. Equivalent to
/// `encode_record(&WalOp::Insert { ... }, seq, out)` but without
/// the intermediate enum.
pub fn encode_insert_record(out: &mut Vec<u8>, seq: u64, tree_id: u64, key: &[u8], value: &[u8]) {
    out.reserve(encoded_insert_record_len(key.len(), value.len()));
    write_record(out, seq, TY_INSERT, |buf| {
        buf.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(buf, key);
        write_bytes(buf, value);
    });
}

#[inline]
pub(crate) const fn encoded_insert_record_len(key_len: usize, value_len: usize) -> usize {
    RECORD_HEADER_SIZE + 8 + 4 + key_len + 4 + value_len + RECORD_FOOTER_SIZE
}

/// Encode an `Erase` record directly from refs. Carries key only
/// because replay redoes from `key` alone.
pub fn encode_erase_record(out: &mut Vec<u8>, seq: u64, tree_id: u64, key: &[u8]) {
    out.reserve(encoded_erase_record_len(key.len()));
    write_record(out, seq, TY_ERASE, |buf| {
        buf.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(buf, key);
    });
}

#[inline]
pub(crate) const fn encoded_erase_record_len(key_len: usize) -> usize {
    RECORD_HEADER_SIZE + 8 + 4 + key_len + RECORD_FOOTER_SIZE
}

/// Encode a `RenameObject` record directly from refs.
pub fn encode_rename_object_record(
    out: &mut Vec<u8>,
    seq: u64,
    tree_id: u64,
    src_key: &[u8],
    dst_key: &[u8],
    value: &[u8],
    force: bool,
) {
    out.reserve(encoded_rename_object_record_len(
        src_key.len(),
        dst_key.len(),
        value.len(),
    ));
    write_record(out, seq, TY_RENAME_OBJECT, |buf| {
        buf.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(buf, src_key);
        write_bytes(buf, dst_key);
        write_bytes(buf, value);
        buf.push(u8::from(force));
    });
}

#[inline]
pub(crate) const fn encoded_rename_object_record_len(
    src_key_len: usize,
    dst_key_len: usize,
    value_len: usize,
) -> usize {
    RECORD_HEADER_SIZE
        + 8
        + 4
        + src_key_len
        + 4
        + dst_key_len
        + 4
        + value_len
        + 1
        + RECORD_FOOTER_SIZE
}

/// Streaming `Batch` record builder. Encodes inner primitive ops
/// directly from `&[u8]` refs into the WAL pending buffer, skipping
/// the intermediate `WalOp::Insert` / `WalOp::Erase` /
/// `WalOp::RenameObject` enum constructions and their `Vec` clones
/// that [`encode_record`] would force on the caller.
///
/// Lifecycle:
///
/// 1. [`BatchEncoder::begin`] writes the record header and the
///    batch body prefix (`tree_id` + zero-placeholder inner-count).
/// 2. The caller interleaves walker mutations with
///    [`Self::push_insert`] / [`Self::push_insert_run`] /
///    [`Self::push_erase`] / [`Self::push_rename_object`] calls.
///    Each push appends one logical inner op or one compact run
///    of logical inner ops to the body.
/// 3. [`Self::finish`] backpatches the inner count + body length
///    and appends the CRC. On a successful finish the record is
///    fully formed in the underlying buffer.
///
/// If the encoder is dropped without `finish` (e.g. the caller
/// bailed mid-batch with `?`), the partial bytes appended so far
/// are truncated back to the encoder's start position — leaving
/// the buffer in the same shape as if `begin` had never run.
pub struct BatchEncoder<'buf> {
    out: &'buf mut Vec<u8>,
    /// Buffer offset of the record's `MAGIC` byte — used by the
    /// `Drop` rollback path.
    start: usize,
    /// Buffer offset of the record-header `body_len` slot.
    len_pos: usize,
    /// Buffer offset where the body starts (immediately after the
    /// record header). CRC covers `start..body_end`.
    body_start: usize,
    /// Buffer offset of the batch body's `count` slot (a `u32` that
    /// holds the number of inner ops pushed).
    count_pos: usize,
    inner_count: u32,
    finished: bool,
}

impl<'buf> BatchEncoder<'buf> {
    /// Open a new `Batch` record on `out`. The header + body prefix
    /// (tree_id, zero-placeholder count) are written immediately;
    /// subsequent `push_*` calls extend the body.
    pub fn begin(out: &'buf mut Vec<u8>, seq: u64, tree_id: u64) -> Self {
        out.reserve(RECORD_HEADER_SIZE + 8 + 4 + RECORD_FOOTER_SIZE);
        let start = out.len();
        out.extend_from_slice(&RECORD_MAGIC.to_le_bytes());
        let len_pos = out.len();
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&seq.to_le_bytes());
        out.push(TY_BATCH);
        let body_start = out.len();
        out.extend_from_slice(&tree_id.to_le_bytes());
        let count_pos = out.len();
        out.extend_from_slice(&[0u8; 4]);
        Self {
            out,
            start,
            len_pos,
            body_start,
            count_pos,
            inner_count: 0,
            finished: false,
        }
    }

    /// Append one `Insert` inner op. Mirrors the wire shape that
    /// `encode_body` writes for `WalOp::Insert` (sans the leading
    /// type tag, which we prepend here for batch framing).
    pub fn push_insert(&mut self, tree_id: u64, key: &[u8], value: &[u8]) {
        self.out.reserve(1 + 8 + 4 + key.len() + 4 + value.len());
        self.out.push(TY_INSERT);
        self.out.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(self.out, key);
        write_bytes(self.out, value);
        self.inner_count += 1;
    }

    /// Append a compact run of consecutive `Insert` inner ops
    /// where every key and value has the same byte length.
    ///
    /// This is still logically `count` primitive insert records:
    /// replay expands the run back into `Insert` ops with seqs
    /// `base + logical_index`. The compact wire frame only removes
    /// repeated inner tags, tree ids, and per-item length prefixes.
    pub fn push_insert_run<'a, I>(
        &mut self,
        tree_id: u64,
        count: usize,
        key_len: usize,
        value_len: usize,
        items: I,
    ) where
        I: IntoIterator<Item = (&'a [u8], &'a [u8])>,
    {
        if count == 1 {
            let mut iter = items.into_iter();
            let (key, value) = iter.next().expect("single insert run has one item");
            debug_assert!(iter.next().is_none());
            self.push_insert(tree_id, key, value);
            return;
        }

        let count_u32 = u32::try_from(count).expect("insert run count fits in u32");
        let key_len_u32 = u32::try_from(key_len).expect("key length fits in u32");
        let value_len_u32 = u32::try_from(value_len).expect("value length fits in u32");
        self.out
            .reserve(1 + 8 + 4 + 4 + 4 + count.saturating_mul(key_len.saturating_add(value_len)));

        self.out.push(TY_BATCH_INSERT_RUN);
        self.out.extend_from_slice(&tree_id.to_le_bytes());
        self.out.extend_from_slice(&count_u32.to_le_bytes());
        self.out.extend_from_slice(&key_len_u32.to_le_bytes());
        self.out.extend_from_slice(&value_len_u32.to_le_bytes());

        let mut actual = 0usize;
        for (key, value) in items {
            debug_assert_eq!(key.len(), key_len);
            debug_assert_eq!(value.len(), value_len);
            self.out.extend_from_slice(key);
            self.out.extend_from_slice(value);
            actual += 1;
        }
        assert_eq!(actual, count, "insert run item count mismatch");
        self.inner_count += count_u32;
    }

    /// Append a compact run of consecutive `Insert` inner ops whose
    /// keys share a byte prefix. This keeps logical replay identical
    /// while removing the repeated path prefix from metadata bulk
    /// creates under one directory/object prefix.
    pub fn push_insert_prefix_run<'a, I>(
        &mut self,
        tree_id: u64,
        prefix: &[u8],
        count: usize,
        items: I,
    ) where
        I: IntoIterator<Item = (&'a [u8], &'a [u8])>,
    {
        let mut iter = items.into_iter();
        if count == 1 {
            let (key, value) = iter.next().expect("single prefix run has one item");
            debug_assert!(iter.next().is_none());
            self.push_insert(tree_id, key, value);
            return;
        }

        let count_u32 = u32::try_from(count).expect("insert prefix run count fits in u32");
        self.out.reserve(1 + 8 + 4 + 4 + prefix.len());
        self.out.push(TY_BATCH_INSERT_PREFIX_RUN);
        self.out.extend_from_slice(&tree_id.to_le_bytes());
        self.out.extend_from_slice(&count_u32.to_le_bytes());
        write_bytes(self.out, prefix);

        let mut actual = 0usize;
        for (key, value) in iter {
            debug_assert!(key.starts_with(prefix));
            write_bytes(self.out, &key[prefix.len()..]);
            write_bytes(self.out, value);
            actual += 1;
        }
        assert_eq!(actual, count, "insert prefix run item count mismatch");
        self.inner_count += count_u32;
    }

    /// Append one `Erase` inner op.
    pub fn push_erase(&mut self, tree_id: u64, key: &[u8]) {
        self.out.reserve(1 + 8 + 4 + key.len());
        self.out.push(TY_ERASE);
        self.out.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(self.out, key);
        self.inner_count += 1;
    }

    /// Append one `RenameObject` inner op.
    pub fn push_rename_object(
        &mut self,
        tree_id: u64,
        src: &[u8],
        dst: &[u8],
        value: &[u8],
        force: bool,
    ) {
        self.out
            .reserve(1 + 8 + 4 + src.len() + 4 + dst.len() + 4 + value.len() + 1);
        self.out.push(TY_RENAME_OBJECT);
        self.out.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(self.out, src);
        write_bytes(self.out, dst);
        write_bytes(self.out, value);
        self.out.push(u8::from(force));
        self.inner_count += 1;
    }

    /// Backpatch the inner count + record-header `body_len` and
    /// append the CRC footer. Returns the final inner count.
    ///
    /// Consumes `self` to make the "fully-formed record" state
    /// type-enforced — after this returns, the record is committed
    /// to the buffer and the `Drop` rollback path is suppressed.
    pub fn finish(mut self) -> u32 {
        let body_end = self.out.len();
        let body_len = u32::try_from(body_end - self.body_start).expect("batch body fits in u32");
        self.out[self.count_pos..self.count_pos + 4]
            .copy_from_slice(&self.inner_count.to_le_bytes());
        self.out[self.len_pos..self.len_pos + 4].copy_from_slice(&body_len.to_le_bytes());
        let crc = crc32(&self.out[self.start..body_end]);
        self.out.extend_from_slice(&crc.to_le_bytes());
        self.out.extend_from_slice(&body_len.to_le_bytes());
        self.finished = true;
        self.inner_count
    }
}

impl Drop for BatchEncoder<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // Caller bailed mid-batch (e.g. a walker `?` propagated
            // out). Roll back the partial record so the WAL buffer
            // looks exactly like it did before `begin`.
            self.out.truncate(self.start);
        }
    }
}

#[cfg(test)]
fn variant_tag(op: &WalOp) -> u8 {
    match op {
        WalOp::Insert { .. } => TY_INSERT,
        WalOp::Erase { .. } => TY_ERASE,
        WalOp::RenameObject { .. } => TY_RENAME_OBJECT,
        WalOp::Batch { .. } => TY_BATCH,
    }
}

#[cfg(test)]
fn encode_body(op: &WalOp, out: &mut Vec<u8>) {
    match op {
        WalOp::Insert {
            tree_id,
            key,
            value,
        } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, key);
            write_bytes(out, value);
        }
        WalOp::Erase { tree_id, key } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, key);
        }
        WalOp::RenameObject {
            tree_id,
            src_key,
            dst_key,
            value,
            force,
        } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, src_key);
            write_bytes(out, dst_key);
            write_bytes(out, value);
            out.push(u8::from(*force));
        }
        WalOp::Batch { ops } => {
            out.extend_from_slice(&0u64.to_le_bytes());
            let count = u32::try_from(ops.len()).expect("batch ops fit in u32");
            out.extend_from_slice(&count.to_le_bytes());
            for inner in ops {
                let inner_ty = variant_tag(inner);
                assert!(
                    inner_ty != TY_BATCH,
                    "nested Batch is rejected — Tree::atomic must flatten",
                );
                out.push(inner_ty);
                encode_body(inner, out);
            }
        }
    }
}

fn write_bytes(out: &mut Vec<u8>, b: &[u8]) {
    let len = u32::try_from(b.len()).expect("byte string fits in u32");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(b);
}

// ---------- decode ----------

/// Maximum user-key bytes accepted by the public tree API. The ART search key
/// adds one virtual terminator byte before reaching the `u16` walker limit.
pub(crate) const MAX_WAL_KEY_BYTES: usize = u16::MAX as usize - 1;
/// Maximum value bytes accepted by the public tree API.
pub(crate) const MAX_WAL_VALUE_BYTES: usize = u16::MAX as usize;

/// Header fields needed by the bounded replay scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecordHeader {
    pub(crate) body_len: u32,
    pub(crate) seq: u64,
    pub(crate) ty: u8,
    pub(crate) total_len: usize,
}

/// Decode and limit-check a fixed record header without reading its body.
pub(crate) fn decode_record_header(header: &[u8]) -> Result<RecordHeader> {
    if header.len() < RECORD_HEADER_SIZE {
        return Err(sanity("record header truncated"));
    }
    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    if magic != RECORD_MAGIC {
        return Err(sanity("record magic mismatch"));
    }
    let body_len = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let total_len = RECORD_HEADER_SIZE
        .checked_add(body_len as usize)
        .and_then(|len| len.checked_add(RECORD_FOOTER_SIZE))
        .ok_or_else(|| sanity("record length overflow"))?;
    if total_len > MAX_ATOMIC_WAL_RECORD_BYTES {
        return Err(sanity("record exceeds native atomic WAL ceiling"));
    }
    Ok(RecordHeader {
        body_len,
        seq: u64::from_le_bytes(header[8..16].try_into().unwrap()),
        ty: header[16],
        total_len,
    })
}

/// Minimal cursor contract shared by slice replay and bounded file `pread`.
/// Implementations must never expose bytes past the declared record body.
pub(crate) trait RecordBodyCursor {
    fn remaining(&self) -> u64;
    fn read_exact_bytes(&mut self, out: &mut [u8]) -> Result<()>;
    fn skip_bytes(&mut self, len: u64) -> Result<()>;
}

fn cursor_take(cursor: &mut impl RecordBodyCursor, out: &mut [u8]) -> Result<()> {
    let len = u64::try_from(out.len()).map_err(|_| sanity("body length overflow"))?;
    if cursor.remaining() < len {
        return Err(sanity("body truncated"));
    }
    cursor.read_exact_bytes(out)
}

fn cursor_skip(cursor: &mut impl RecordBodyCursor, len: u64) -> Result<()> {
    if cursor.remaining() < len {
        return Err(sanity("body truncated"));
    }
    cursor.skip_bytes(len)
}

fn cursor_u8(cursor: &mut impl RecordBodyCursor) -> Result<u8> {
    let mut bytes = [0u8; 1];
    cursor_take(cursor, &mut bytes)?;
    Ok(bytes[0])
}

fn cursor_u32(cursor: &mut impl RecordBodyCursor) -> Result<u32> {
    let mut bytes = [0u8; 4];
    cursor_take(cursor, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn cursor_u64(cursor: &mut impl RecordBodyCursor) -> Result<u64> {
    let mut bytes = [0u8; 8];
    cursor_take(cursor, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn validate_key_len(len: usize) -> Result<()> {
    if len > MAX_WAL_KEY_BYTES {
        return Err(sanity("WAL key exceeds public API limit"));
    }
    Ok(())
}

fn validate_value_len(len: usize) -> Result<()> {
    if len > MAX_WAL_VALUE_BYTES {
        return Err(sanity("WAL value exceeds public API limit"));
    }
    Ok(())
}

fn validate_sized_bytes(
    cursor: &mut impl RecordBodyCursor,
    validate_len: impl FnOnce(usize) -> Result<()>,
) -> Result<usize> {
    let len = cursor_u32(cursor)? as usize;
    validate_len(len)?;
    cursor_skip(cursor, len as u64)?;
    Ok(len)
}

fn validate_record_tree_id(tree_id: u64, expected_tree_id: Option<u64>) -> Result<()> {
    if expected_tree_id.is_some_and(|expected| tree_id != expected) {
        return Err(sanity("WAL record tree_id does not belong to this Tree"));
    }
    Ok(())
}

fn validate_primitive_body(
    ty: u8,
    cursor: &mut impl RecordBodyCursor,
    expected_tree_id: Option<u64>,
) -> Result<()> {
    match ty {
        TY_INSERT => {
            validate_record_tree_id(cursor_u64(cursor)?, expected_tree_id)?;
            validate_sized_bytes(cursor, validate_key_len)?;
            validate_sized_bytes(cursor, validate_value_len)?;
        }
        TY_ERASE => {
            validate_record_tree_id(cursor_u64(cursor)?, expected_tree_id)?;
            validate_sized_bytes(cursor, validate_key_len)?;
        }
        TY_RENAME_OBJECT => {
            validate_record_tree_id(cursor_u64(cursor)?, expected_tree_id)?;
            validate_sized_bytes(cursor, validate_key_len)?;
            validate_sized_bytes(cursor, validate_key_len)?;
            validate_sized_bytes(cursor, validate_value_len)?;
            if cursor_u8(cursor)? > 1 {
                return Err(sanity("RenameObject force flag is not canonical"));
            }
        }
        TY_BATCH => return Err(sanity("nested Batch is rejected")),
        _ => return Err(sanity("unknown WalOp variant tag")),
    }
    Ok(())
}

fn validate_insert_run_body(
    cursor: &mut impl RecordBodyCursor,
    logical: u64,
    batch_count: u64,
    expected_tree_id: Option<u64>,
) -> Result<u64> {
    validate_record_tree_id(cursor_u64(cursor)?, expected_tree_id)?;
    let count = u64::from(cursor_u32(cursor)?);
    if count == 0 {
        return Err(sanity("empty BatchInsertRun is rejected"));
    }
    let end = logical
        .checked_add(count)
        .ok_or_else(|| sanity("batch inner count overflow"))?;
    if end > batch_count {
        return Err(sanity("BatchInsertRun exceeds batch inner count"));
    }
    let key_len = cursor_u32(cursor)? as usize;
    let value_len = cursor_u32(cursor)? as usize;
    validate_key_len(key_len)?;
    validate_value_len(value_len)?;
    let item_len = key_len
        .checked_add(value_len)
        .ok_or_else(|| sanity("BatchInsertRun item length overflow"))?;
    let payload_len = u64::try_from(item_len)
        .ok()
        .and_then(|len| len.checked_mul(count))
        .ok_or_else(|| sanity("BatchInsertRun payload length overflow"))?;
    cursor_skip(cursor, payload_len)?;
    Ok(end)
}

fn validate_insert_prefix_run_body(
    cursor: &mut impl RecordBodyCursor,
    logical: u64,
    batch_count: u64,
    expected_tree_id: Option<u64>,
) -> Result<u64> {
    validate_record_tree_id(cursor_u64(cursor)?, expected_tree_id)?;
    let count = u64::from(cursor_u32(cursor)?);
    if count == 0 {
        return Err(sanity("empty BatchInsertPrefixRun is rejected"));
    }
    let end = logical
        .checked_add(count)
        .ok_or_else(|| sanity("batch inner count overflow"))?;
    if end > batch_count {
        return Err(sanity("BatchInsertPrefixRun exceeds batch inner count"));
    }
    let prefix_len = cursor_u32(cursor)? as usize;
    validate_key_len(prefix_len)?;
    cursor_skip(cursor, prefix_len as u64)?;
    for _ in 0..count {
        let suffix_len = cursor_u32(cursor)? as usize;
        let key_len = prefix_len
            .checked_add(suffix_len)
            .ok_or_else(|| sanity("BatchInsertPrefixRun key length overflow"))?;
        validate_key_len(key_len)?;
        cursor_skip(cursor, suffix_len as u64)?;
        validate_sized_bytes(cursor, validate_value_len)?;
    }
    Ok(end)
}

/// Validate one record body without allocating payloads or invoking replay.
/// The caller separately validates the record CRC/footer framing.
pub(crate) fn validate_record_body(
    header: RecordHeader,
    cursor: &mut impl RecordBodyCursor,
    file_tree_id: u64,
    expected_primitive_tree_id: Option<u64>,
) -> Result<()> {
    if header.ty == TY_BATCH {
        let outer_tree_id = cursor_u64(cursor)?;
        if outer_tree_id != file_tree_id {
            return Err(sanity("WAL batch owner does not match file header"));
        }
        let count = u64::from(cursor_u32(cursor)?);
        if count > MAX_ATOMIC_WAL_OPS as u64 {
            return Err(sanity("batch exceeds native logical-operation ceiling"));
        }
        if count != 0 {
            header
                .seq
                .checked_add(count)
                .ok_or_else(|| sanity("batch sequence range has no representable successor"))?;
        }
        let mut logical = 0u64;
        while logical < count {
            let inner_ty = cursor_u8(cursor)?;
            logical = match inner_ty {
                TY_BATCH_INSERT_RUN => {
                    validate_insert_run_body(cursor, logical, count, expected_primitive_tree_id)?
                }
                TY_BATCH_INSERT_PREFIX_RUN => validate_insert_prefix_run_body(
                    cursor,
                    logical,
                    count,
                    expected_primitive_tree_id,
                )?,
                _ => {
                    validate_primitive_body(inner_ty, cursor, expected_primitive_tree_id)?;
                    logical
                        .checked_add(1)
                        .ok_or_else(|| sanity("batch inner count overflow"))?
                }
            };
        }
    } else {
        header
            .seq
            .checked_add(1)
            .ok_or_else(|| sanity("record sequence has no representable successor"))?;
        validate_primitive_body(header.ty, cursor, expected_primitive_tree_id)?;
    }
    if cursor.remaining() != 0 {
        return Err(sanity("trailing bytes after variant body"));
    }
    Ok(())
}

fn cursor_owned_bytes(
    cursor: &mut impl RecordBodyCursor,
    len: usize,
    allocation_context: &'static str,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| Error::Internal(allocation_context))?;
    bytes.resize(len, 0);
    cursor_take(cursor, &mut bytes)?;
    Ok(bytes)
}

fn cursor_key(cursor: &mut impl RecordBodyCursor) -> Result<Vec<u8>> {
    let len = cursor_u32(cursor)? as usize;
    validate_key_len(len)?;
    cursor_owned_bytes(cursor, len, "WAL replay key allocation failed")
}

fn cursor_value(cursor: &mut impl RecordBodyCursor) -> Result<Vec<u8>> {
    let len = cursor_u32(cursor)? as usize;
    validate_value_len(len)?;
    cursor_owned_bytes(cursor, len, "WAL replay value allocation failed")
}

fn visit_primitive_body<F>(
    ty: u8,
    cursor: &mut impl RecordBodyCursor,
    seq: u64,
    callback: &mut F,
) -> Result<()>
where
    F: FnMut(&WalOp, u64) -> Result<()>,
{
    let op = match ty {
        TY_INSERT => WalOp::Insert {
            tree_id: cursor_u64(cursor)?,
            key: cursor_key(cursor)?,
            value: cursor_value(cursor)?,
        },
        TY_ERASE => WalOp::Erase {
            tree_id: cursor_u64(cursor)?,
            key: cursor_key(cursor)?,
        },
        TY_RENAME_OBJECT => WalOp::RenameObject {
            tree_id: cursor_u64(cursor)?,
            src_key: cursor_key(cursor)?,
            dst_key: cursor_key(cursor)?,
            value: cursor_value(cursor)?,
            force: cursor_u8(cursor)? != 0,
        },
        TY_BATCH => return Err(sanity("nested Batch is rejected")),
        _ => return Err(sanity("unknown WalOp variant tag")),
    };
    callback(&op, seq)
}

fn visit_insert_run_body<F>(
    cursor: &mut impl RecordBodyCursor,
    base_seq: u64,
    logical: u64,
    callback: &mut F,
) -> Result<u64>
where
    F: FnMut(&WalOp, u64) -> Result<()>,
{
    let tree_id = cursor_u64(cursor)?;
    let count = u64::from(cursor_u32(cursor)?);
    let key_len = cursor_u32(cursor)? as usize;
    let value_len = cursor_u32(cursor)? as usize;
    let end = logical
        .checked_add(count)
        .ok_or_else(|| sanity("batch inner count overflow"))?;
    for index in logical..end {
        let key = cursor_owned_bytes(cursor, key_len, "WAL replay key allocation failed")?;
        let value = cursor_owned_bytes(cursor, value_len, "WAL replay value allocation failed")?;
        let seq = base_seq
            .checked_add(index)
            .ok_or_else(|| sanity("batch sequence range overflows u64"))?;
        callback(
            &WalOp::Insert {
                tree_id,
                key,
                value,
            },
            seq,
        )?;
    }
    Ok(end)
}

fn visit_insert_prefix_run_body<F>(
    cursor: &mut impl RecordBodyCursor,
    base_seq: u64,
    logical: u64,
    callback: &mut F,
) -> Result<u64>
where
    F: FnMut(&WalOp, u64) -> Result<()>,
{
    let tree_id = cursor_u64(cursor)?;
    let count = u64::from(cursor_u32(cursor)?);
    let prefix_len = cursor_u32(cursor)? as usize;
    let prefix = cursor_owned_bytes(
        cursor,
        prefix_len,
        "WAL replay key-prefix allocation failed",
    )?;
    let end = logical
        .checked_add(count)
        .ok_or_else(|| sanity("batch inner count overflow"))?;
    for index in logical..end {
        let suffix_len = cursor_u32(cursor)? as usize;
        let key_len = prefix_len
            .checked_add(suffix_len)
            .ok_or_else(|| sanity("BatchInsertPrefixRun key length overflow"))?;
        let mut key = Vec::new();
        key.try_reserve_exact(key_len)
            .map_err(|_| Error::Internal("WAL replay key allocation failed"))?;
        key.extend_from_slice(&prefix);
        key.resize(key_len, 0);
        cursor_take(cursor, &mut key[prefix_len..])?;
        let value = cursor_value(cursor)?;
        let seq = base_seq
            .checked_add(index)
            .ok_or_else(|| sanity("batch sequence range overflows u64"))?;
        callback(
            &WalOp::Insert {
                tree_id,
                key,
                value,
            },
            seq,
        )?;
    }
    Ok(end)
}

/// Apply one body that already passed [`validate_record_body`], materializing
/// only the current primitive op. Batch payloads are never accumulated.
pub(crate) fn visit_record_body<F>(
    header: RecordHeader,
    cursor: &mut impl RecordBodyCursor,
    callback: &mut F,
) -> Result<()>
where
    F: FnMut(&WalOp, u64) -> Result<()>,
{
    if header.ty == TY_BATCH {
        let _outer_tree_id = cursor_u64(cursor)?;
        let count = u64::from(cursor_u32(cursor)?);
        let mut logical = 0u64;
        while logical < count {
            let inner_ty = cursor_u8(cursor)?;
            logical = match inner_ty {
                TY_BATCH_INSERT_RUN => {
                    visit_insert_run_body(cursor, header.seq, logical, callback)?
                }
                TY_BATCH_INSERT_PREFIX_RUN => {
                    visit_insert_prefix_run_body(cursor, header.seq, logical, callback)?
                }
                _ => {
                    let seq = header
                        .seq
                        .checked_add(logical)
                        .ok_or_else(|| sanity("batch sequence range overflows u64"))?;
                    visit_primitive_body(inner_ty, cursor, seq, callback)?;
                    logical
                        .checked_add(1)
                        .ok_or_else(|| sanity("batch inner count overflow"))?
                }
            };
        }
    } else {
        visit_primitive_body(header.ty, cursor, header.seq, callback)?;
    }
    if cursor.remaining() != 0 {
        return Err(sanity("trailing bytes after variant body"));
    }
    Ok(())
}

/// Outcome of [`decode_record`].
#[cfg(test)]
#[derive(Debug)]
pub struct DecodedRecord {
    /// Parsed op.
    pub op: WalOp,
    /// Sequence carried in the record header.
    pub seq: u64,
    /// Total bytes consumed from the input slice.
    pub bytes_consumed: usize,
}

/// Decode a single record from the start of `buf`.
///
/// The codec doesn't know its file-level offset; the caller (the
/// WAL replay scanner) is responsible for setting `record_offset`
/// on any returned [`Error::ReplaySanityFailed`] before surfacing
/// it to the user.
#[cfg(test)]
pub fn decode_record(buf: &[u8]) -> Result<DecodedRecord> {
    if buf.len() < RECORD_HEADER_SIZE {
        return Err(sanity("record header truncated"));
    }

    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != RECORD_MAGIC {
        return Err(sanity("record magic mismatch"));
    }
    let body_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let seq = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let ty = buf[16];

    let total = RECORD_HEADER_SIZE + body_len + RECORD_FOOTER_SIZE;
    if buf.len() < total {
        return Err(sanity("record body truncated"));
    }

    let body_end = RECORD_HEADER_SIZE + body_len;
    let crc_expected = u32::from_le_bytes(buf[body_end..body_end + 4].try_into().unwrap());
    let crc_computed = crc32(&buf[..body_end]);
    if crc_computed != crc_expected {
        return Err(sanity("record CRC mismatch"));
    }
    let repeated_body_len =
        u32::from_le_bytes(buf[body_end + 4..body_end + 8].try_into().unwrap()) as usize;
    if repeated_body_len != body_len {
        return Err(sanity("record footer body length mismatch"));
    }

    let body = &buf[RECORD_HEADER_SIZE..body_end];
    let op = decode_body(ty, body)?;

    Ok(DecodedRecord {
        op,
        seq,
        bytes_consumed: total,
    })
}

#[cfg(test)]
fn decode_body(ty: u8, body: &[u8]) -> Result<WalOp> {
    let mut cursor = body;
    let op = decode_body_into(ty, &mut cursor)?;
    if !cursor.is_empty() {
        return Err(sanity("trailing bytes after variant body"));
    }
    Ok(op)
}

/// Internal: decode one variant body from `cursor`, advancing it.
/// Doesn't enforce body-exhaustion — `decode_body` wraps with
/// that check, and `TY_BATCH` re-enters this for each inner op
/// (sharing the parent's cursor as the inner-frame stream).
#[cfg(test)]
fn decode_body_into(ty: u8, body: &mut &[u8]) -> Result<WalOp> {
    let op = match ty {
        TY_INSERT => {
            let tree_id = read_u64(body)?;
            let key = read_bytes(body)?;
            let value = read_bytes(body)?;
            WalOp::Insert {
                tree_id,
                key,
                value,
            }
        }
        TY_ERASE => {
            let tree_id = read_u64(body)?;
            let key = read_bytes(body)?;
            WalOp::Erase { tree_id, key }
        }
        TY_RENAME_OBJECT => {
            let tree_id = read_u64(body)?;
            let src_key = read_bytes(body)?;
            let dst_key = read_bytes(body)?;
            let value = read_bytes(body)?;
            let force = read_u8(body)? != 0;
            WalOp::RenameObject {
                tree_id,
                src_key,
                dst_key,
                value,
                force,
            }
        }
        TY_BATCH => {
            let _tree_id = read_u64(body)?;
            let count = read_u32(body)? as usize;
            let mut ops = Vec::new();
            ops.try_reserve_exact(count)
                .map_err(|_| Error::Internal("WAL batch decode allocation failed"))?;
            while ops.len() < count {
                let inner_ty = read_u8(body)?;
                if inner_ty == TY_BATCH {
                    return Err(sanity("nested Batch is rejected"));
                }
                match inner_ty {
                    TY_BATCH_INSERT_RUN => decode_insert_run(body, count, &mut ops)?,
                    TY_BATCH_INSERT_PREFIX_RUN => decode_insert_prefix_run(body, count, &mut ops)?,
                    _ => {
                        let inner = decode_body_into(inner_ty, body)?;
                        ops.push(inner);
                    }
                }
            }
            WalOp::Batch { ops }
        }
        _ => return Err(sanity("unknown WalOp variant tag")),
    };
    Ok(op)
}

#[cfg(test)]
fn decode_insert_run(body: &mut &[u8], batch_count: usize, ops: &mut Vec<WalOp>) -> Result<()> {
    let tree_id = read_u64(body)?;
    let count = read_u32(body)? as usize;
    if count == 0 {
        return Err(sanity("empty BatchInsertRun is rejected"));
    }
    if ops.len().saturating_add(count) > batch_count {
        return Err(sanity("BatchInsertRun exceeds batch inner count"));
    }
    let key_len = read_u32(body)? as usize;
    let value_len = read_u32(body)? as usize;
    for _ in 0..count {
        let (key, rest) = take(body, key_len)?;
        *body = rest;
        let (value, rest) = take(body, value_len)?;
        *body = rest;
        ops.push(WalOp::Insert {
            tree_id,
            key: key.to_vec(),
            value: value.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn decode_insert_prefix_run(
    body: &mut &[u8],
    batch_count: usize,
    ops: &mut Vec<WalOp>,
) -> Result<()> {
    let tree_id = read_u64(body)?;
    let count = read_u32(body)? as usize;
    if count == 0 {
        return Err(sanity("empty BatchInsertPrefixRun is rejected"));
    }
    if ops.len().saturating_add(count) > batch_count {
        return Err(sanity("BatchInsertPrefixRun exceeds batch inner count"));
    }
    let prefix = read_bytes(body)?;
    for _ in 0..count {
        let suffix = read_bytes(body)?;
        let value = read_bytes(body)?;
        let mut key = Vec::with_capacity(prefix.len() + suffix.len());
        key.extend_from_slice(&prefix);
        key.extend_from_slice(&suffix);
        ops.push(WalOp::Insert {
            tree_id,
            key,
            value,
        });
    }
    Ok(())
}

#[cfg(test)]
fn read_u8(body: &mut &[u8]) -> Result<u8> {
    let (front, rest) = take(body, 1)?;
    *body = rest;
    Ok(front[0])
}

#[cfg(test)]
fn read_u32(body: &mut &[u8]) -> Result<u32> {
    let (front, rest) = take(body, 4)?;
    *body = rest;
    Ok(u32::from_le_bytes(front.try_into().unwrap()))
}

#[cfg(test)]
fn read_u64(body: &mut &[u8]) -> Result<u64> {
    let (front, rest) = take(body, 8)?;
    *body = rest;
    Ok(u64::from_le_bytes(front.try_into().unwrap()))
}

#[cfg(test)]
fn read_bytes(body: &mut &[u8]) -> Result<Vec<u8>> {
    let len = read_u32(body)? as usize;
    let (front, rest) = take(body, len)?;
    *body = rest;
    Ok(front.to_vec())
}

#[cfg(test)]
fn take(buf: &[u8], n: usize) -> Result<(&[u8], &[u8])> {
    if buf.len() < n {
        return Err(sanity("body truncated"));
    }
    Ok(buf.split_at(n))
}

fn sanity(context: &'static str) -> Error {
    Error::ReplaySanityFailed {
        context,
        record_offset: 0,
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(op: WalOp, seq: u64) {
        let mut buf = Vec::new();
        encode_record(&op, seq, &mut buf);

        let r = decode_record(&buf).unwrap();
        assert_eq!(r.seq, seq);
        assert_eq!(r.bytes_consumed, buf.len());
        assert_eq!(format!("{:?}", r.op), format!("{op:?}"));
    }

    #[test]
    fn crc32_matches_known_vector() {
        // "123456789" → 0xCBF43926 (standard CRC-32/IEEE).
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn roundtrip_insert_small() {
        roundtrip(
            WalOp::Insert {
                tree_id: 0,
                key: b"img/01.jpg".to_vec(),
                value: b"v-new".to_vec(),
            },
            42,
        );
    }

    #[test]
    fn roundtrip_insert_large_value() {
        roundtrip(
            WalOp::Insert {
                tree_id: 0,
                key: b"new/key".to_vec(),
                value: vec![0xAB; 200],
            },
            7,
        );
    }

    #[test]
    fn roundtrip_erase() {
        roundtrip(
            WalOp::Erase {
                tree_id: 0,
                key: b"img/02.jpg".to_vec(),
            },
            99,
        );
    }

    #[test]
    fn roundtrip_rename_object() {
        roundtrip(
            WalOp::RenameObject {
                tree_id: 0,
                src_key: b"a/b".to_vec(),
                dst_key: b"a/c".to_vec(),
                value: b"rename-value".to_vec(),
                force: true,
            },
            10,
        );
    }

    #[test]
    fn removed_cross_tree_rename_tag_is_rejected() {
        let mut buf = Vec::new();
        write_record(&mut buf, 11, 6, |body| {
            body.extend_from_slice(&1u64.to_le_bytes());
            body.extend_from_slice(&2u64.to_le_bytes());
            write_bytes(body, b"x");
            write_bytes(body, b"y");
            body.push(0);
        });

        assert!(matches!(
            decode_record(&buf),
            Err(Error::ReplaySanityFailed {
                context: "unknown WalOp variant tag",
                ..
            })
        ));
    }

    #[test]
    fn removed_structural_tags_are_rejected() {
        for ty in [2, 3, 4] {
            let mut buf = Vec::new();
            write_record(&mut buf, 500 + u64::from(ty), ty, |_| {});
            assert!(
                matches!(
                    decode_record(&buf),
                    Err(Error::ReplaySanityFailed {
                        context: "unknown WalOp variant tag",
                        ..
                    })
                ),
                "removed structural tag {ty} should not decode",
            );
        }
    }

    #[test]
    fn record_length_breakdown_is_predictable() {
        let op = WalOp::Insert {
            tree_id: 0,
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        };
        let mut buf = Vec::new();
        encode_record(&op, 0, &mut buf);
        // tree_id (8) + key_len (4) + key (1) + val_len (4) + val (1)
        //   = 18 byte body. Header (17) + body (18) + footer (8) = 43.
        assert_eq!(buf.len(), 43);
    }

    #[test]
    fn corrupt_crc_is_caught() {
        let op = WalOp::Insert {
            tree_id: 0,
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        };
        let mut buf = Vec::new();
        encode_record(&op, 1, &mut buf);
        let body_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let crc_offset = RECORD_HEADER_SIZE + body_len;
        buf[crc_offset] ^= 0x01;
        match decode_record(&buf) {
            Err(Error::ReplaySanityFailed { context, .. }) => {
                assert!(context.contains("CRC"));
            }
            other => panic!("expected CRC sanity failure, got {other:?}"),
        }
    }

    #[test]
    fn corrupt_magic_is_caught() {
        let op = WalOp::Erase {
            tree_id: 0,
            key: b"k".to_vec(),
        };
        let mut buf = Vec::new();
        encode_record(&op, 5, &mut buf);
        buf[0] ^= 0xFF;
        match decode_record(&buf) {
            Err(Error::ReplaySanityFailed { context, .. }) => {
                assert!(context.contains("magic"));
            }
            other => panic!("expected magic sanity failure, got {other:?}"),
        }
    }

    #[test]
    fn truncated_record_is_caught() {
        let op = WalOp::Insert {
            tree_id: 0,
            key: vec![0xAB; 100],
            value: vec![0xCD; 100],
        };
        let mut buf = Vec::new();
        encode_record(&op, 1, &mut buf);
        // Drop the last 10 bytes — simulates a torn write at EOF.
        let len = buf.len();
        buf.truncate(len - 10);
        match decode_record(&buf) {
            Err(Error::ReplaySanityFailed { context, .. }) => {
                assert!(context.contains("truncated"));
            }
            other => panic!("expected truncation sanity failure, got {other:?}"),
        }
    }

    #[test]
    fn unknown_variant_tag_is_caught() {
        let op = WalOp::Erase {
            tree_id: 0,
            key: b"k".to_vec(),
        };
        let mut buf = Vec::new();
        encode_record(&op, 1, &mut buf);
        // Overwrite the ty byte (header offset 16) with a bogus value.
        buf[16] = 0xFF;
        // Recompute the CRC so the corruption looks plausible
        // — confirms the "unknown tag" path triggers (and not "CRC").
        let body_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let body_end = RECORD_HEADER_SIZE + body_len;
        let crc = crc32(&buf[..body_end]);
        buf[body_end..body_end + 4].copy_from_slice(&crc.to_le_bytes());

        match decode_record(&buf) {
            Err(Error::ReplaySanityFailed { context, .. }) => {
                assert!(context.contains("variant"));
            }
            other => panic!("expected unknown-variant sanity failure, got {other:?}"),
        }
    }

    #[test]
    fn back_to_back_records_concatenate_cleanly() {
        let mut buf = Vec::new();
        encode_record(
            &WalOp::Insert {
                tree_id: 0,
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
            },
            1,
            &mut buf,
        );
        encode_record(
            &WalOp::Erase {
                tree_id: 0,
                key: b"k1".to_vec(),
            },
            2,
            &mut buf,
        );

        let r1 = decode_record(&buf).unwrap();
        assert_eq!(r1.seq, 1);
        let r2 = decode_record(&buf[r1.bytes_consumed..]).unwrap();
        assert_eq!(r2.seq, 2);
        assert_eq!(r1.bytes_consumed + r2.bytes_consumed, buf.len());
    }

    #[test]
    fn roundtrip_batch_three_inner_ops() {
        // Insert + Erase + RenameObject under one Batch envelope.
        // Inner seqs are derived from `base + index`, so the encoder
        // should not need explicit per-inner seq storage.
        let base = 100u64;
        let batch = WalOp::Batch {
            ops: vec![
                WalOp::Insert {
                    tree_id: 0,
                    key: b"a".to_vec(),
                    value: b"v-a".to_vec(),
                },
                WalOp::Erase {
                    tree_id: 0,
                    key: b"b".to_vec(),
                },
                WalOp::RenameObject {
                    tree_id: 0,
                    src_key: b"c".to_vec(),
                    dst_key: b"d".to_vec(),
                    value: b"v-c".to_vec(),
                    force: false,
                },
            ],
        };
        let mut buf = Vec::new();
        encode_record(&batch, base, &mut buf);

        let r = decode_record(&buf).unwrap();
        assert_eq!(r.seq, base);
        assert_eq!(r.bytes_consumed, buf.len());
        match r.op {
            WalOp::Batch { ops } => {
                assert_eq!(ops.len(), 3);
                match &ops[0] {
                    WalOp::Insert { key, .. } => {
                        assert_eq!(key, b"a");
                    }
                    other => panic!("expected Insert, got {other:?}"),
                }
                match &ops[1] {
                    WalOp::Erase { key, .. } => {
                        assert_eq!(key, b"b");
                    }
                    other => panic!("expected Erase, got {other:?}"),
                }
                match &ops[2] {
                    WalOp::RenameObject {
                        src_key,
                        dst_key,
                        force,
                        ..
                    } => {
                        assert_eq!(src_key, b"c");
                        assert_eq!(dst_key, b"d");
                        assert!(!force);
                    }
                    other => panic!("expected RenameObject, got {other:?}"),
                }
            }
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_batch_empty() {
        let batch = WalOp::Batch { ops: vec![] };
        let mut buf = Vec::new();
        encode_record(&batch, 7, &mut buf);
        let r = decode_record(&buf).unwrap();
        assert_eq!(r.seq, 7);
        match r.op {
            WalOp::Batch { ops } => assert!(ops.is_empty()),
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn batch_encoder_wire_matches_encode_record() {
        // The streaming `BatchEncoder` and the generic
        // `encode_record(&WalOp::Batch { .. })` path must produce
        // byte-identical records — that's what lets `Tree::atomic`
        // bypass the enum without breaking replay.
        let base = 200u64;

        // Path A: streaming encoder.
        let mut buf_streaming = Vec::new();
        {
            let mut enc = BatchEncoder::begin(&mut buf_streaming, base, 0);
            enc.push_insert(0, b"a", b"v-a");
            enc.push_erase(0, b"b");
            enc.push_rename_object(0, b"c", b"d", b"v-c", false);
            let n = enc.finish();
            assert_eq!(n, 3);
        }

        // Path B: enum-and-encode.
        let mut buf_enum = Vec::new();
        let batch = WalOp::Batch {
            ops: vec![
                WalOp::Insert {
                    tree_id: 0,
                    key: b"a".to_vec(),
                    value: b"v-a".to_vec(),
                },
                WalOp::Erase {
                    tree_id: 0,
                    key: b"b".to_vec(),
                },
                WalOp::RenameObject {
                    tree_id: 0,
                    src_key: b"c".to_vec(),
                    dst_key: b"d".to_vec(),
                    value: b"v-c".to_vec(),
                    force: false,
                },
            ],
        };
        encode_record(&batch, base, &mut buf_enum);

        assert_eq!(buf_streaming, buf_enum);

        // Round-trips cleanly via the standard decoder.
        let r = decode_record(&buf_streaming).unwrap();
        assert_eq!(r.seq, base);
        match r.op {
            WalOp::Batch { ops } => assert_eq!(ops.len(), 3),
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn batch_insert_run_round_trips_and_saves_wire_bytes() {
        let base = 300u64;

        let mut compact = Vec::new();
        {
            let mut enc = BatchEncoder::begin(&mut compact, base, 0);
            enc.push_insert_run(
                0,
                3,
                4,
                2,
                [
                    (&b"k001"[..], &b"v1"[..]),
                    (&b"k002"[..], &b"v2"[..]),
                    (&b"k003"[..], &b"v3"[..]),
                ],
            );
            assert_eq!(enc.finish(), 3);
        }

        let mut individual = Vec::new();
        {
            let mut enc = BatchEncoder::begin(&mut individual, base, 0);
            enc.push_insert(0, b"k001", b"v1");
            enc.push_insert(0, b"k002", b"v2");
            enc.push_insert(0, b"k003", b"v3");
            assert_eq!(enc.finish(), 3);
        }

        assert!(
            compact.len() < individual.len(),
            "compact insert run should be smaller: compact={}, individual={}",
            compact.len(),
            individual.len(),
        );

        let r = decode_record(&compact).unwrap();
        match r.op {
            WalOp::Batch { ops } => {
                assert_eq!(ops.len(), 3);
                for (idx, op) in ops.iter().enumerate() {
                    let WalOp::Insert { key, value, .. } = op else {
                        panic!("expected insert, got {op:?}");
                    };
                    assert_eq!(key, format!("k{:03}", idx + 1).as_bytes());
                    assert_eq!(value, format!("v{}", idx + 1).as_bytes());
                }
            }
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn batch_insert_prefix_run_round_trips_and_saves_wire_bytes() {
        let base = 301u64;
        let prefix = b"bucket/table/date=2026-06-26/";

        let mut compact = Vec::new();
        {
            let mut enc = BatchEncoder::begin(&mut compact, base, 0);
            enc.push_insert_prefix_run(
                0,
                prefix,
                3,
                [
                    (
                        &b"bucket/table/date=2026-06-26/000001.parquet"[..],
                        &b"m1"[..],
                    ),
                    (
                        &b"bucket/table/date=2026-06-26/000002.parquet"[..],
                        &b"metadata-2"[..],
                    ),
                    (
                        &b"bucket/table/date=2026-06-26/part-000003.parquet"[..],
                        &b"m3"[..],
                    ),
                ],
            );
            assert_eq!(enc.finish(), 3);
        }

        let mut individual = Vec::new();
        {
            let mut enc = BatchEncoder::begin(&mut individual, base, 0);
            enc.push_insert(0, b"bucket/table/date=2026-06-26/000001.parquet", b"m1");
            enc.push_insert(
                0,
                b"bucket/table/date=2026-06-26/000002.parquet",
                b"metadata-2",
            );
            enc.push_insert(
                0,
                b"bucket/table/date=2026-06-26/part-000003.parquet",
                b"m3",
            );
            assert_eq!(enc.finish(), 3);
        }

        assert!(
            compact.len() < individual.len(),
            "prefix insert run should be smaller: compact={}, individual={}",
            compact.len(),
            individual.len(),
        );

        let r = decode_record(&compact).unwrap();
        match r.op {
            WalOp::Batch { ops } => {
                assert_eq!(ops.len(), 3);
                assert!(matches!(
                    &ops[0],
                    WalOp::Insert { key, value, .. }
                        if key == b"bucket/table/date=2026-06-26/000001.parquet"
                            && value == b"m1"
                ));
                assert!(matches!(
                    &ops[1],
                    WalOp::Insert { key, value, .. }
                        if key == b"bucket/table/date=2026-06-26/000002.parquet"
                            && value == b"metadata-2"
                ));
                assert!(matches!(
                    &ops[2],
                    WalOp::Insert { key, value, .. }
                        if key == b"bucket/table/date=2026-06-26/part-000003.parquet"
                            && value == b"m3"
                ));
            }
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn batch_encoder_empty_round_trips() {
        let mut buf = Vec::new();
        {
            let enc = BatchEncoder::begin(&mut buf, 9, 0);
            assert_eq!(enc.finish(), 0);
        }
        let r = decode_record(&buf).unwrap();
        assert_eq!(r.seq, 9);
        match r.op {
            WalOp::Batch { ops } => assert!(ops.is_empty()),
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn batch_encoder_drop_without_finish_rolls_back() {
        // Caller bails mid-batch (e.g. `?` propagated out of the
        // closure). The encoder's `Drop` must truncate the partial
        // record so the WAL buffer ends up exactly where it was.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"pre-existing bytes");
        let before = buf.len();
        {
            let mut enc = BatchEncoder::begin(&mut buf, 1, 0);
            enc.push_insert(0, b"would-be-rolled-back", b"v");
            // Drop without calling finish().
        }
        assert_eq!(buf.len(), before, "Drop should truncate the partial record");
        assert_eq!(&buf[..], b"pre-existing bytes");
    }

    #[test]
    fn batch_encoder_finish_commits_record() {
        // Confirm the happy path: after finish() the encoder's
        // bytes are committed — a subsequent Drop is a no-op.
        let mut buf = Vec::new();
        {
            let mut enc = BatchEncoder::begin(&mut buf, 5, 0);
            enc.push_insert(0, b"k", b"v");
            let _ = enc.finish();
        }
        assert!(!buf.is_empty());
        let r = decode_record(&buf).unwrap();
        assert_eq!(r.seq, 5);
    }

    #[test]
    #[should_panic(expected = "nested Batch is rejected")]
    fn nested_batch_encode_panics() {
        let inner = WalOp::Batch { ops: vec![] };
        let outer = WalOp::Batch { ops: vec![inner] };
        let mut buf = Vec::new();
        encode_record(&outer, 0, &mut buf);
    }
}
