//! Logical WAL record codec — binary encoding for [`WalOp`].
//!
//! Each record on disk has the shape
//!
//! ```text
//! +------+------+------+----+-----------+------+
//! | MAGIC| LEN  | SEQ  | TY |   BODY    | CRC32|
//! | u32  | u32  | u64  | u8 |  varlen   | u32  |
//! +------+------+------+----+-----------+------+
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
//!
//! All integers are little-endian. All length-prefixed byte
//! strings (keys, values, tree names) use a `u32` LE length
//! followed by raw bytes.

use super::wal_op::WalOp;
use crate::api::errors::{Error, Result};
use crate::api::journal::{JournalAnchor, JournalEnvelope, JOURNAL_DIGEST_BYTES};

/// Start-of-record magic — `"RECR"` little-endian.
pub const RECORD_MAGIC: u32 = 0x5243_4552;

/// Fixed-size header bytes: `magic | len | seq | ty`.
pub const RECORD_HEADER_SIZE: usize = 4 + 4 + 8 + 1;

/// Fixed-size footer bytes: `crc32`.
pub const RECORD_FOOTER_SIZE: usize = 4;

// ---------- File header ----------

/// Top-of-file magic — `"WALA"` little-endian. Sits at offset 0 of
/// every WAL file and is checked on open. Mismatch = "this isn't
/// one of our WAL files".
pub const FILE_MAGIC: u32 = 0x414C_4157;

/// Previous public format accepted by the replay reader.
///
/// Format 3 contains only the original logical redo tags. It is replay-only
/// under a format-4 writer: callers must checkpoint it with a format-3 Holt
/// binary before reopening for new writes.
pub const LEGACY_FORMAT_VERSION: u32 = 3;

/// Format version emitted by new WAL writers.
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
///
/// Format `4` adds `DbBatchWithEnvelope` (tag 13). Its application recovery
/// bytes and logical DB operations are covered by one record CRC.
pub const FORMAT_VERSION: u32 = 4;

/// Legacy format-3 header byte size and record-stream offset.
pub const LEGACY_FILE_HEADER_SIZE: usize = 32;

/// Format-4 page header byte size and record-stream offset.
///
/// Keeping records page-aligned leaves room for two independently checksummed
/// checkpoint-anchor slots without rewriting the append stream.
pub const FILE_HEADER_SIZE: usize = 4096;

const ANCHOR_SLOT_MAGIC: u32 = 0x5248_4341; // "ACHR" little-endian.
const ANCHOR_SLOT_VERSION: u32 = 1;
pub(crate) const ANCHOR_SLOT_SIZE: usize = 128;
pub(crate) const ANCHOR_SLOT_OFFSETS: [usize; 2] = [64, 64 + ANCHOR_SLOT_SIZE];
const ANCHOR_SLOT_CHECKSUM_OFFSET: usize = 64;
const ANCHOR_SLOT_FLAG_INITIALIZED: u64 = 1;

/// Format-4 top-of-file layout:
///
/// ```text
/// base[32] | padding[32] | anchor_slot_a[128] | anchor_slot_b[128]
///           |                         zero padding to 4096 bytes            |
///
/// base = MAGIC:u32 | VER:u32 | TREE:u64 | CREATED:u64 | HEADER_SIZE:u64
/// slot = MAGIC:u32 | VER:u32 | GENERATION:u64 | FLAGS:u64
///        | SEQUENCE:u64 | DIGEST:[u8;32] | CRC32:u32 | padding[60]
/// ```
///
/// - `MAGIC` = [`FILE_MAGIC`] (`"WALA"` LE).
/// - `VER`   = [`FORMAT_VERSION`] for new files; replay also accepts
///   [`LEGACY_FORMAT_VERSION`].
/// - `TREE`  = tree owner identifier; `0` for the single-tree API.
/// - `CREATED` = unix epoch seconds; `0` when the writer chose
///   not to stamp a time (e.g. tests).
/// - `HEADER_SIZE` = [`FILE_HEADER_SIZE`].
/// - Each initialized anchor slot has an independent generation and CRC.
///   Decode selects the highest valid generation. Checkpoint updates write and
///   sync both slots in sequence before they truncate the record stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    /// On-disk WAL format version.
    pub version: u32,
    /// Tree owner identifier.
    pub tree_id: u64,
    /// Unix-epoch seconds when the file was created. `0` if the
    /// writer didn't stamp one.
    pub created_at: u64,
    /// Latest checkpoint anchor recovered from the two format-4 slots.
    pub checkpoint_anchor: Option<JournalAnchor>,
    /// Generation of `checkpoint_anchor`; zero when uninitialized.
    pub anchor_generation: u64,
}

impl FileHeader {
    /// Build a header with the current wall clock.
    #[must_use]
    pub fn now(tree_id: u64) -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            version: FORMAT_VERSION,
            tree_id,
            created_at,
            checkpoint_anchor: None,
            anchor_generation: 0,
        }
    }

    /// Byte offset at which this format's record stream starts.
    #[must_use]
    pub const fn record_offset(self) -> usize {
        if self.version == LEGACY_FORMAT_VERSION {
            LEGACY_FILE_HEADER_SIZE
        } else {
            FILE_HEADER_SIZE
        }
    }
}

/// Encode the file header into the first [`FILE_HEADER_SIZE`] bytes
/// of `out` (the buffer is resized as needed).
pub fn encode_file_header(h: &FileHeader, out: &mut Vec<u8>) {
    let start = out.len();
    out.extend_from_slice(&FILE_MAGIC.to_le_bytes());
    out.extend_from_slice(&h.version.to_le_bytes());
    out.extend_from_slice(&h.tree_id.to_le_bytes());
    out.extend_from_slice(&h.created_at.to_le_bytes());
    if h.version == LEGACY_FORMAT_VERSION {
        out.extend_from_slice(&0u64.to_le_bytes());
        debug_assert_eq!(out.len() - start, LEGACY_FILE_HEADER_SIZE);
        return;
    }

    out.extend_from_slice(&(FILE_HEADER_SIZE as u64).to_le_bytes());
    out.resize(start + FILE_HEADER_SIZE, 0);
    if let Some(anchor) = h.checkpoint_anchor {
        debug_assert_ne!(h.anchor_generation, 0);
        let slot_index = ((h.anchor_generation - 1) & 1) as usize;
        let slot = encode_anchor_slot(anchor, h.anchor_generation);
        let offset = start + ANCHOR_SLOT_OFFSETS[slot_index];
        out[offset..offset + ANCHOR_SLOT_SIZE].copy_from_slice(&slot);
    }
    debug_assert_eq!(out.len() - start, FILE_HEADER_SIZE);
}

/// Decode a file header from the first [`FILE_HEADER_SIZE`] bytes
/// of `buf`. Returns the header on success and a sanity-failed
/// error (with `record_offset = 0`) on mismatch.
pub fn decode_file_header(buf: &[u8]) -> Result<FileHeader> {
    if buf.len() < LEGACY_FILE_HEADER_SIZE {
        return Err(sanity("WAL file header truncated"));
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != FILE_MAGIC {
        return Err(sanity("WAL file magic mismatch"));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != LEGACY_FORMAT_VERSION && version != FORMAT_VERSION {
        return Err(sanity("WAL file format version unsupported"));
    }
    let required = if version == LEGACY_FORMAT_VERSION {
        LEGACY_FILE_HEADER_SIZE
    } else {
        FILE_HEADER_SIZE
    };
    if buf.len() < required {
        return Err(sanity("WAL file header truncated"));
    }
    let tree_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let created_at = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    if version == FORMAT_VERSION {
        let encoded_size = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        if encoded_size != FILE_HEADER_SIZE as u64 {
            return Err(sanity("WAL format-4 header size mismatch"));
        }
    }
    let (checkpoint_anchor, anchor_generation) = if version == FORMAT_VERSION {
        decode_checkpoint_anchor(buf)?
    } else {
        (None, 0)
    };
    Ok(FileHeader {
        version,
        tree_id,
        created_at,
        checkpoint_anchor,
        anchor_generation,
    })
}

/// Return the header size named by the prefix, without requiring the full
/// format-4 page to be present yet.
pub(crate) fn file_header_size_from_prefix(prefix: &[u8]) -> Result<usize> {
    if prefix.len() < 8 {
        return Err(sanity("WAL file header truncated"));
    }
    let magic = u32::from_le_bytes(prefix[0..4].try_into().unwrap());
    if magic != FILE_MAGIC {
        return Err(sanity("WAL file magic mismatch"));
    }
    match u32::from_le_bytes(prefix[4..8].try_into().unwrap()) {
        LEGACY_FORMAT_VERSION => Ok(LEGACY_FILE_HEADER_SIZE),
        FORMAT_VERSION => Ok(FILE_HEADER_SIZE),
        _ => Err(sanity("WAL file format version unsupported")),
    }
}

pub(crate) fn encode_anchor_slot(anchor: JournalAnchor, generation: u64) -> [u8; ANCHOR_SLOT_SIZE] {
    debug_assert_ne!(generation, 0);
    let mut slot = [0u8; ANCHOR_SLOT_SIZE];
    slot[0..4].copy_from_slice(&ANCHOR_SLOT_MAGIC.to_le_bytes());
    slot[4..8].copy_from_slice(&ANCHOR_SLOT_VERSION.to_le_bytes());
    slot[8..16].copy_from_slice(&generation.to_le_bytes());
    slot[16..24].copy_from_slice(&ANCHOR_SLOT_FLAG_INITIALIZED.to_le_bytes());
    slot[24..32].copy_from_slice(&anchor.sequence().to_le_bytes());
    slot[32..64].copy_from_slice(&anchor.digest());
    let checksum = crc32(&slot[..ANCHOR_SLOT_CHECKSUM_OFFSET]);
    slot[ANCHOR_SLOT_CHECKSUM_OFFSET..ANCHOR_SLOT_CHECKSUM_OFFSET + 4]
        .copy_from_slice(&checksum.to_le_bytes());
    slot
}

fn decode_checkpoint_anchor(buf: &[u8]) -> Result<(Option<JournalAnchor>, u64)> {
    let mut valid = Vec::with_capacity(2);
    let mut nonzero_slots = 0;
    for &offset in &ANCHOR_SLOT_OFFSETS {
        let slot = &buf[offset..offset + ANCHOR_SLOT_SIZE];
        if slot.iter().any(|byte| *byte != 0) {
            nonzero_slots += 1;
        }
        if let Some(decoded) = decode_anchor_slot(slot) {
            valid.push(decoded);
        }
    }
    if valid.is_empty() {
        // One non-zero invalid slot can only be a torn first initialization;
        // the all-zero sibling is still the authoritative uninitialized state.
        if nonzero_slots > 1 {
            return Err(sanity("WAL checkpoint anchor slots corrupt"));
        }
        return Ok((None, 0));
    }
    valid.sort_unstable_by_key(|(_, generation)| *generation);
    if valid.len() == 2 && valid[0].1 == valid[1].1 && valid[0].0 != valid[1].0 {
        return Err(sanity("WAL checkpoint anchor generations conflict"));
    }
    let (anchor, generation) = *valid.last().unwrap();
    Ok((Some(anchor), generation))
}

fn decode_anchor_slot(slot: &[u8]) -> Option<(JournalAnchor, u64)> {
    if slot.len() != ANCHOR_SLOT_SIZE {
        return None;
    }
    let magic = u32::from_le_bytes(slot[0..4].try_into().ok()?);
    let version = u32::from_le_bytes(slot[4..8].try_into().ok()?);
    let generation = u64::from_le_bytes(slot[8..16].try_into().ok()?);
    let flags = u64::from_le_bytes(slot[16..24].try_into().ok()?);
    if magic != ANCHOR_SLOT_MAGIC
        || version != ANCHOR_SLOT_VERSION
        || generation == 0
        || flags != ANCHOR_SLOT_FLAG_INITIALIZED
    {
        return None;
    }
    let expected = u32::from_le_bytes(
        slot[ANCHOR_SLOT_CHECKSUM_OFFSET..ANCHOR_SLOT_CHECKSUM_OFFSET + 4]
            .try_into()
            .ok()?,
    );
    if crc32(&slot[..ANCHOR_SLOT_CHECKSUM_OFFSET]) != expected {
        return None;
    }
    let sequence = u64::from_le_bytes(slot[24..32].try_into().ok()?);
    let digest = slot[32..64].try_into().ok()?;
    Some((JournalAnchor::new(sequence, digest), generation))
}

// On-disk variant tags. Stable through format v4; only ever add new
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
const TY_DB_BATCH_WITH_ENVELOPE: u8 = 13;

/// Version of the envelope sub-frame carried by tag 13.
const JOURNAL_ENVELOPE_WIRE_VERSION: u8 = 1;

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
    force: bool,
) {
    out.reserve(encoded_rename_object_record_len(
        src_key.len(),
        dst_key.len(),
    ));
    write_record(out, seq, TY_RENAME_OBJECT, |buf| {
        buf.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(buf, src_key);
        write_bytes(buf, dst_key);
        buf.push(u8::from(force));
    });
}

#[inline]
pub(crate) const fn encoded_rename_object_record_len(
    src_key_len: usize,
    dst_key_len: usize,
) -> usize {
    RECORD_HEADER_SIZE + 8 + 4 + src_key_len + 4 + dst_key_len + 1 + RECORD_FOOTER_SIZE
}

/// Streaming `Batch` record builder. Encodes inner primitive ops
/// directly from `&[u8]` refs into the WAL pending buffer, skipping
/// the intermediate `WalOp::Insert` / `WalOp::Erase` /
/// `WalOp::RenameObject` enum constructions and their `Vec` clones
/// that [`encode_record`] would force on the caller.
///
/// Lifecycle:
///
/// 1. [`BatchEncoder::begin`] or [`BatchEncoder::begin_with_envelope`]
///    writes the record header and batch body prefix (`tree_id` plus a
///    zero-placeholder inner count). The attached form then writes its
///    validated application envelope before the operation stream.
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
        Self::begin_record(out, seq, tree_id, TY_BATCH, None)
    }

    /// Open a format-4 DB `Batch` record with one recovery envelope attached.
    ///
    /// The body layout is:
    ///
    /// ```text
    /// tree_id:u64 | count:u32 | envelope_version:u8
    /// previous_sequence:u64 | previous_digest:[u8;32]
    /// current_sequence:u64  | current_digest:[u8;32]
    /// payload_len:u32 | payload:[u8;payload_len] | inner_ops...
    /// ```
    ///
    /// `finish` computes one CRC over the record header, the full envelope,
    /// and all inner operations. The application payload remains opaque.
    // This is the format-4 producer seam. The DB API calls it in the next
    // integration slice; keeping the allow local avoids masking other dead code.
    #[allow(dead_code)]
    pub fn begin_with_envelope(
        out: &'buf mut Vec<u8>,
        seq: u64,
        tree_id: u64,
        envelope: &JournalEnvelope,
    ) -> Self {
        Self::begin_record(out, seq, tree_id, TY_DB_BATCH_WITH_ENVELOPE, Some(envelope))
    }

    fn begin_record(
        out: &'buf mut Vec<u8>,
        seq: u64,
        tree_id: u64,
        ty: u8,
        envelope: Option<&JournalEnvelope>,
    ) -> Self {
        let envelope_len =
            envelope.map_or(0, |value| encoded_journal_envelope_len(value.payload()));
        out.reserve(RECORD_HEADER_SIZE + 8 + 4 + envelope_len + RECORD_FOOTER_SIZE);
        let start = out.len();
        out.extend_from_slice(&RECORD_MAGIC.to_le_bytes());
        let len_pos = out.len();
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&seq.to_le_bytes());
        out.push(ty);
        let body_start = out.len();
        out.extend_from_slice(&tree_id.to_le_bytes());
        let count_pos = out.len();
        out.extend_from_slice(&[0u8; 4]);
        if let Some(envelope) = envelope {
            encode_journal_envelope(envelope, out);
        }
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
    pub fn push_rename_object(&mut self, tree_id: u64, src: &[u8], dst: &[u8], force: bool) {
        self.out.reserve(1 + 8 + 4 + src.len() + 4 + dst.len() + 1);
        self.out.push(TY_RENAME_OBJECT);
        self.out.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(self.out, src);
        write_bytes(self.out, dst);
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
        WalOp::DbBatchWithEnvelope { .. } => TY_DB_BATCH_WITH_ENVELOPE,
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
            force,
        } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, src_key);
            write_bytes(out, dst_key);
            out.push(u8::from(*force));
        }
        WalOp::Batch { ops } => {
            out.extend_from_slice(&0u64.to_le_bytes());
            let count = u32::try_from(ops.len()).expect("batch ops fit in u32");
            out.extend_from_slice(&count.to_le_bytes());
            for inner in ops {
                let inner_ty = variant_tag(inner);
                assert!(
                    inner_ty != TY_BATCH && inner_ty != TY_DB_BATCH_WITH_ENVELOPE,
                    "nested Batch is rejected — Tree::atomic must flatten",
                );
                out.push(inner_ty);
                encode_body(inner, out);
            }
        }
        WalOp::DbBatchWithEnvelope { envelope, ops } => {
            out.extend_from_slice(&0u64.to_le_bytes());
            let count = u32::try_from(ops.len()).expect("batch ops fit in u32");
            out.extend_from_slice(&count.to_le_bytes());
            encode_journal_envelope(envelope, out);
            for inner in ops {
                let inner_ty = variant_tag(inner);
                assert!(
                    inner_ty != TY_BATCH && inner_ty != TY_DB_BATCH_WITH_ENVELOPE,
                    "nested Batch is rejected — DB::atomic must flatten",
                );
                out.push(inner_ty);
                encode_body(inner, out);
            }
        }
    }
}

fn encoded_journal_envelope_len(payload: &[u8]) -> usize {
    1 + 8 + JOURNAL_DIGEST_BYTES + 8 + JOURNAL_DIGEST_BYTES + 4 + payload.len()
}

fn encode_journal_envelope(envelope: &JournalEnvelope, out: &mut Vec<u8>) {
    out.push(JOURNAL_ENVELOPE_WIRE_VERSION);
    let previous = envelope.previous();
    out.extend_from_slice(&previous.sequence().to_le_bytes());
    out.extend_from_slice(&previous.digest());
    let current = envelope.current();
    out.extend_from_slice(&current.sequence().to_le_bytes());
    out.extend_from_slice(&current.digest());
    write_bytes(out, envelope.payload());
}

fn write_bytes(out: &mut Vec<u8>, b: &[u8]) {
    let len = u32::try_from(b.len()).expect("byte string fits in u32");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(b);
}

// ---------- decode ----------

/// Outcome of [`decode_record`].
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

    let body = &buf[RECORD_HEADER_SIZE..body_end];
    let op = decode_body(ty, body)?;

    Ok(DecodedRecord {
        op,
        seq,
        bytes_consumed: total,
    })
}

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
            let force = read_u8(body)? != 0;
            WalOp::RenameObject {
                tree_id,
                src_key,
                dst_key,
                force,
            }
        }
        TY_BATCH | TY_DB_BATCH_WITH_ENVELOPE => {
            let _tree_id = read_u64(body)?;
            let count = read_u32(body)? as usize;
            let envelope = if ty == TY_DB_BATCH_WITH_ENVELOPE {
                Some(decode_journal_envelope(body)?)
            } else {
                None
            };
            let mut ops = Vec::with_capacity(count);
            while ops.len() < count {
                let inner_ty = read_u8(body)?;
                if inner_ty == TY_BATCH || inner_ty == TY_DB_BATCH_WITH_ENVELOPE {
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
            if let Some(envelope) = envelope {
                WalOp::DbBatchWithEnvelope { envelope, ops }
            } else {
                WalOp::Batch { ops }
            }
        }
        _ => return Err(sanity("unknown WalOp variant tag")),
    };
    Ok(op)
}

fn decode_journal_envelope(body: &mut &[u8]) -> Result<JournalEnvelope> {
    let wire_version = read_u8(body)?;
    if wire_version != JOURNAL_ENVELOPE_WIRE_VERSION {
        return Err(sanity("journal envelope wire version unsupported"));
    }

    let previous_sequence = read_u64(body)?;
    let previous_digest = read_digest(body)?;
    let current_sequence = read_u64(body)?;
    let current_digest = read_digest(body)?;
    let payload = read_bytes(body)?;
    JournalEnvelope::new(
        JournalAnchor::new(previous_sequence, previous_digest),
        JournalAnchor::new(current_sequence, current_digest),
        payload,
    )
    .map_err(|error| match error {
        crate::JournalEnvelopeError::NonContiguousSequence { .. } => {
            sanity("journal envelope sequence is not contiguous")
        }
        crate::JournalEnvelopeError::EmptyPayload => sanity("journal envelope payload is empty"),
        crate::JournalEnvelopeError::PayloadTooLarge { .. } => {
            sanity("journal envelope payload is too large")
        }
    })
}

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

fn read_u8(body: &mut &[u8]) -> Result<u8> {
    let (front, rest) = take(body, 1)?;
    *body = rest;
    Ok(front[0])
}

fn read_u32(body: &mut &[u8]) -> Result<u32> {
    let (front, rest) = take(body, 4)?;
    *body = rest;
    Ok(u32::from_le_bytes(front.try_into().unwrap()))
}

fn read_u64(body: &mut &[u8]) -> Result<u64> {
    let (front, rest) = take(body, 8)?;
    *body = rest;
    Ok(u64::from_le_bytes(front.try_into().unwrap()))
}

fn read_digest(body: &mut &[u8]) -> Result<[u8; JOURNAL_DIGEST_BYTES]> {
    let (front, rest) = take(body, JOURNAL_DIGEST_BYTES)?;
    *body = rest;
    Ok(front.try_into().unwrap())
}

fn read_bytes(body: &mut &[u8]) -> Result<Vec<u8>> {
    let len = read_u32(body)? as usize;
    let (front, rest) = take(body, len)?;
    *body = rest;
    Ok(front.to_vec())
}

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
    use std::fmt::Write;

    fn sample_envelope() -> JournalEnvelope {
        JournalEnvelope::new(
            JournalAnchor::new(41, [0x11; JOURNAL_DIGEST_BYTES]),
            JournalAnchor::new(42, [0x22; JOURNAL_DIGEST_BYTES]),
            b"rv1".to_vec(),
        )
        .unwrap()
    }

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
        //   = 18 byte body. Header (17) + body (18) + footer (4) = 39.
        assert_eq!(buf.len(), 39);
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
        let last = buf.len() - 1;
        buf[last] ^= 0x01;
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
    fn attached_batch_streaming_and_enum_encoders_match() {
        let envelope = sample_envelope();
        let mut streaming = Vec::new();
        {
            let mut encoder =
                BatchEncoder::begin_with_envelope(&mut streaming, 1_000, 0, &envelope);
            encoder.push_insert(7, b"k", b"v");
            assert_eq!(encoder.finish(), 1);
        }

        let attached = WalOp::DbBatchWithEnvelope {
            envelope: envelope.clone(),
            ops: vec![WalOp::Insert {
                tree_id: 7,
                key: b"k".to_vec(),
                value: b"v".to_vec(),
            }],
        };
        let mut generic = Vec::new();
        encode_record(&attached, 1_000, &mut generic);
        assert_eq!(streaming, generic);

        let decoded = decode_record(&streaming).unwrap();
        assert_eq!(decoded.seq, 1_000);
        assert_eq!(decoded.bytes_consumed, streaming.len());
        match decoded.op {
            WalOp::DbBatchWithEnvelope {
                envelope: decoded_envelope,
                ops,
            } => {
                assert_eq!(decoded_envelope, envelope);
                assert!(matches!(
                    ops.as_slice(),
                    [WalOp::Insert {
                        tree_id: 7,
                        key,
                        value,
                    }] if key == b"k" && value == b"v"
                ));
            }
            other => panic!("expected attached DB batch, got {other:?}"),
        }
    }

    #[test]
    fn attached_batch_has_stable_format_4_golden_bytes() {
        let envelope = sample_envelope();
        let mut encoded = Vec::new();
        {
            let mut encoder = BatchEncoder::begin_with_envelope(&mut encoded, 1_000, 0, &envelope);
            encoder.push_insert(7, b"k", b"v");
            encoder.finish();
        }

        let actual = encoded.iter().fold(String::new(), |mut output, byte| {
            write!(output, "{byte:02x}").unwrap();
            output
        });
        assert_eq!(
            actual,
            concat!(
                "5245435277000000e8030000000000000d0000000000000000010000000129",
                "0000000000000011111111111111111111111111111111111111111111111111",
                "111111111111112a000000000000002222222222222222222222222222222222",
                "2222222222222222222222222222220300000072763100070000000000000001",
                "0000006b0100000076ca022951"
            )
        );
    }

    #[test]
    fn attached_batch_crc_covers_envelope_and_inner_ops() {
        let mut encoded = Vec::new();
        {
            let mut encoder =
                BatchEncoder::begin_with_envelope(&mut encoded, 1_000, 0, &sample_envelope());
            encoder.push_insert(7, b"k", b"v");
            encoder.finish();
        }

        let payload_offset = encoded
            .windows(3)
            .position(|window| window == b"rv1")
            .unwrap();
        let mut envelope_corrupt = encoded.clone();
        envelope_corrupt[payload_offset] ^= 0x01;
        assert!(matches!(
            decode_record(&envelope_corrupt),
            Err(Error::ReplaySanityFailed {
                context: "record CRC mismatch",
                ..
            })
        ));

        let mut op_corrupt = encoded;
        let value_offset = op_corrupt.len() - RECORD_FOOTER_SIZE - 1;
        op_corrupt[value_offset] ^= 0x01;
        assert!(matches!(
            decode_record(&op_corrupt),
            Err(Error::ReplaySanityFailed {
                context: "record CRC mismatch",
                ..
            })
        ));
    }

    #[test]
    fn attached_batch_rejects_bad_envelope_version_and_sequence() {
        let mut encoded = Vec::new();
        {
            let encoder =
                BatchEncoder::begin_with_envelope(&mut encoded, 1_000, 0, &sample_envelope());
            encoder.finish();
        }

        let envelope_version_offset = RECORD_HEADER_SIZE + 8 + 4;
        let mut bad_version = encoded.clone();
        bad_version[envelope_version_offset] = 2;
        rewrite_record_crc(&mut bad_version);
        assert!(matches!(
            decode_record(&bad_version),
            Err(Error::ReplaySanityFailed {
                context: "journal envelope wire version unsupported",
                ..
            })
        ));

        let current_sequence_offset = envelope_version_offset + 1 + 8 + JOURNAL_DIGEST_BYTES;
        let mut bad_sequence = encoded;
        bad_sequence[current_sequence_offset..current_sequence_offset + 8]
            .copy_from_slice(&43u64.to_le_bytes());
        rewrite_record_crc(&mut bad_sequence);
        assert!(matches!(
            decode_record(&bad_sequence),
            Err(Error::ReplaySanityFailed {
                context: "journal envelope sequence is not contiguous",
                ..
            })
        ));
    }

    fn rewrite_record_crc(encoded: &mut [u8]) {
        let body_len = u32::from_le_bytes(encoded[4..8].try_into().unwrap()) as usize;
        let body_end = RECORD_HEADER_SIZE + body_len;
        let checksum = crc32(&encoded[..body_end]);
        encoded[body_end..body_end + RECORD_FOOTER_SIZE].copy_from_slice(&checksum.to_le_bytes());
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
            enc.push_rename_object(0, b"c", b"d", false);
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
