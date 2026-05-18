//! Physiological WAL record codec — binary encoding for [`TxnOp`].
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
//! - `BODY` = variant-specific bytes; see `encode_body` /
//!   `decode_body` for the exact layout per variant.
//! - `CRC32` (IEEE 802.3 polynomial `0xEDB8_8320`) detects torn
//!   writes and silent disk corruption.
//!
//! All integers are little-endian. All length-prefixed byte
//! strings (keys, values, tree names) use a `u32` LE length
//! followed by raw bytes.

use crate::api::errors::{Error, Result};
use crate::engine::compact::CompactReason;
use crate::layout::BlobGuid;

use super::txn_op::TxnOp;

/// Start-of-record magic — `"RECR"` little-endian.
pub const RECORD_MAGIC: u32 = 0x5243_4552;

/// Fixed-size header bytes: `magic | len | seq | ty`.
pub const RECORD_HEADER_SIZE: usize = 4 + 4 + 8 + 1;

/// Fixed-size footer bytes: `crc32`.
pub const RECORD_FOOTER_SIZE: usize = 4;

/// Overhead per record (header + footer, excluding variable body).
pub const RECORD_OVERHEAD: usize = RECORD_HEADER_SIZE + RECORD_FOOTER_SIZE;

// ---------- File header ----------

/// Top-of-file magic — `"WALA"` little-endian. Sits at offset 0 of
/// every WAL file and is checked on open. Mismatch = "this isn't
/// one of our WAL files".
pub const FILE_MAGIC: u32 = 0x414C_4157;

/// Format version stored in the file header. New format revisions
/// bump this and grow the header (in the reserved tail) rather
/// than moving existing fields.
pub const FORMAT_VERSION: u32 = 1;

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
/// - `TREE`  = tree owner identifier; `0` for the single-tree v0.1.
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
            .map(|d| d.as_secs())
            .unwrap_or(0);
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

// On-disk variant tags. Stable; only ever add new tags, never
// renumber existing ones.
const TY_INSERT: u8 = 0;
const TY_ERASE: u8 = 1;
const TY_SPLIT: u8 = 2;
const TY_MERGE: u8 = 3;
const TY_COMPACT: u8 = 4;
const TY_RENAME_OBJECT: u8 = 5;
const TY_RENAME: u8 = 6;
const TY_NEW_TREE: u8 = 7;
const TY_RM_TREE: u8 = 8;
const TY_MEM_MARKER: u8 = 9;

// CompactReason on-disk tags (stable).
const REASON_SPLIT_TOMBSTONE: u8 = 0;
const REASON_SPLIT_GAP_SPACE: u8 = 1;
const REASON_OUT_OF_BLOB_FRAME: u8 = 2;

// ---------- CRC32 (IEEE 802.3) ----------

/// 256-entry CRC32 lookup table for the IEEE 802.3 reflected
/// polynomial `0xEDB8_8320`. Built once at compile time via
/// [`build_crc32_table`]; ships zero runtime initialisation cost.
const CRC32_TABLE: [u32; 256] = build_crc32_table();

const fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// CRC32 — IEEE 802.3 polynomial `0xEDB8_8320`, reflected
/// (i.e. the variant `gzip` / `PNG` / RocksDB block-checksum
/// use). Used as the record-level `sanity_info`.
///
/// Table-driven byte-at-a-time impl: ≈1.5 GB/s on a modern core.
/// For a typical 175-byte WAL record that's ~110 ns of per-record
/// overhead; SIMD-accelerated variants (PCLMULQDQ on x86,
/// CRC32 intrinsic on AArch64) would push that under 30 ns but
/// are out of scope for v0.1.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    !crc
}

// ---------- encode ----------

/// Append the binary record for `op` (sequence `seq`) to `out`.
///
/// On return, `out` has grown by exactly
/// `RECORD_HEADER_SIZE + body_len + RECORD_FOOTER_SIZE` bytes.
/// `body_len` is variant-dependent — see `encode_body`.
///
/// Hot mutation paths in `Tree` use the per-variant
/// [`encode_insert_record`] / [`encode_erase_record`] /
/// [`encode_rename_object_record`] entry points instead — those
/// skip the `TxnOp` enum construction and the three `Vec` clones
/// it forces on the caller.
pub fn encode_record(op: &TxnOp, seq: u64, out: &mut Vec<u8>) -> Result<()> {
    write_record(out, seq, variant_tag(op), |buf| encode_body(op, buf));
    Ok(())
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
// constructing a `TxnOp` enum, so callers don't pay for the
// three `Vec` clones (key + value + prev_value) that enum
// construction forces.

/// Encode an `Insert` record directly from refs. Equivalent to
/// `encode_record(&TxnOp::Insert { ... }, seq, out)` but without
/// the intermediate enum.
pub fn encode_insert_record(
    out: &mut Vec<u8>,
    seq: u64,
    tree_id: u64,
    key: &[u8],
    value: &[u8],
    prev_value: Option<&[u8]>,
) {
    write_record(out, seq, TY_INSERT, |buf| {
        buf.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(buf, key);
        write_bytes(buf, value);
        write_optional_bytes(buf, prev_value);
    });
}

/// Encode an `Erase` record directly from refs.
pub fn encode_erase_record(out: &mut Vec<u8>, seq: u64, tree_id: u64, key: &[u8], value: &[u8]) {
    write_record(out, seq, TY_ERASE, |buf| {
        buf.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(buf, key);
        write_bytes(buf, value);
    });
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
    write_record(out, seq, TY_RENAME_OBJECT, |buf| {
        buf.extend_from_slice(&tree_id.to_le_bytes());
        write_bytes(buf, src_key);
        write_bytes(buf, dst_key);
        buf.push(u8::from(force));
    });
}

fn variant_tag(op: &TxnOp) -> u8 {
    match op {
        TxnOp::Insert { .. } => TY_INSERT,
        TxnOp::Erase { .. } => TY_ERASE,
        TxnOp::Split { .. } => TY_SPLIT,
        TxnOp::Merge { .. } => TY_MERGE,
        TxnOp::Compact { .. } => TY_COMPACT,
        TxnOp::RenameObject { .. } => TY_RENAME_OBJECT,
        TxnOp::Rename { .. } => TY_RENAME,
        TxnOp::NewTree { .. } => TY_NEW_TREE,
        TxnOp::RmTree { .. } => TY_RM_TREE,
        TxnOp::MemMarker { .. } => TY_MEM_MARKER,
    }
}

fn encode_body(op: &TxnOp, out: &mut Vec<u8>) {
    match op {
        TxnOp::Insert {
            tree_id,
            seq: _,
            key,
            value,
            prev_value,
        } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, key);
            write_bytes(out, value);
            write_optional_bytes(out, prev_value.as_deref());
        }
        TxnOp::Erase {
            tree_id,
            seq: _,
            key,
            value,
        } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, key);
            write_bytes(out, value);
        }
        TxnOp::Split {
            parent_blob,
            pre_split_node,
            new_child_blob,
            new_child_entry,
        } => {
            out.extend_from_slice(parent_blob);
            out.extend_from_slice(&pre_split_node.to_le_bytes());
            out.extend_from_slice(new_child_blob);
            out.extend_from_slice(&new_child_entry.to_le_bytes());
        }
        TxnOp::Merge {
            parent_blob,
            pre_merge_node,
            child_blob,
        } => {
            out.extend_from_slice(parent_blob);
            out.extend_from_slice(&pre_merge_node.to_le_bytes());
            out.extend_from_slice(child_blob);
        }
        TxnOp::Compact { blob, reason } => {
            out.extend_from_slice(blob);
            out.push(encode_reason(*reason));
        }
        TxnOp::RenameObject {
            tree_id,
            seq: _,
            src_key,
            dst_key,
            force,
        } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, src_key);
            write_bytes(out, dst_key);
            out.push(u8::from(*force));
        }
        TxnOp::Rename {
            src_tree_id,
            dst_tree_id,
            seq: _,
            src_key,
            dst_key,
            force,
        } => {
            out.extend_from_slice(&src_tree_id.to_le_bytes());
            out.extend_from_slice(&dst_tree_id.to_le_bytes());
            write_bytes(out, src_key);
            write_bytes(out, dst_key);
            out.push(u8::from(*force));
        }
        TxnOp::NewTree { tree_id, name } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
            write_bytes(out, name);
        }
        TxnOp::RmTree { tree_id } => {
            out.extend_from_slice(&tree_id.to_le_bytes());
        }
        TxnOp::MemMarker { seq: _ } => {
            // Body is empty — seq travels in the record header.
        }
    }
}

fn encode_reason(r: CompactReason) -> u8 {
    match r {
        CompactReason::SplitTombstone => REASON_SPLIT_TOMBSTONE,
        CompactReason::SplitGapSpace => REASON_SPLIT_GAP_SPACE,
        CompactReason::OutOfBlobFrame => REASON_OUT_OF_BLOB_FRAME,
    }
}

fn write_bytes(out: &mut Vec<u8>, b: &[u8]) {
    let len = u32::try_from(b.len()).expect("byte string fits in u32");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(b);
}

fn write_optional_bytes(out: &mut Vec<u8>, b: Option<&[u8]>) {
    match b {
        None => out.push(0),
        Some(x) => {
            out.push(1);
            write_bytes(out, x);
        }
    }
}

// ---------- decode ----------

/// Outcome of [`decode_record`].
#[derive(Debug)]
pub struct DecodedRecord {
    /// Parsed op.
    pub op: TxnOp,
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
    let op = decode_body(ty, body, seq)?;

    Ok(DecodedRecord {
        op,
        seq,
        bytes_consumed: total,
    })
}

fn decode_body(ty: u8, mut body: &[u8], seq: u64) -> Result<TxnOp> {
    let op = match ty {
        TY_INSERT => {
            let tree_id = read_u64(&mut body)?;
            let key = read_bytes(&mut body)?;
            let value = read_bytes(&mut body)?;
            let prev_value = read_optional_bytes(&mut body)?;
            TxnOp::Insert {
                tree_id,
                seq,
                key,
                value,
                prev_value,
            }
        }
        TY_ERASE => {
            let tree_id = read_u64(&mut body)?;
            let key = read_bytes(&mut body)?;
            let value = read_bytes(&mut body)?;
            TxnOp::Erase {
                tree_id,
                seq,
                key,
                value,
            }
        }
        TY_SPLIT => {
            let parent_blob = read_guid(&mut body)?;
            let pre_split_node = read_u16(&mut body)?;
            let new_child_blob = read_guid(&mut body)?;
            let new_child_entry = read_u16(&mut body)?;
            TxnOp::Split {
                parent_blob,
                pre_split_node,
                new_child_blob,
                new_child_entry,
            }
        }
        TY_MERGE => {
            let parent_blob = read_guid(&mut body)?;
            let pre_merge_node = read_u16(&mut body)?;
            let child_blob = read_guid(&mut body)?;
            TxnOp::Merge {
                parent_blob,
                pre_merge_node,
                child_blob,
            }
        }
        TY_COMPACT => {
            let blob = read_guid(&mut body)?;
            let reason = decode_reason(read_u8(&mut body)?)?;
            TxnOp::Compact { blob, reason }
        }
        TY_RENAME_OBJECT => {
            let tree_id = read_u64(&mut body)?;
            let src_key = read_bytes(&mut body)?;
            let dst_key = read_bytes(&mut body)?;
            let force = read_u8(&mut body)? != 0;
            TxnOp::RenameObject {
                tree_id,
                seq,
                src_key,
                dst_key,
                force,
            }
        }
        TY_RENAME => {
            let src_tree_id = read_u64(&mut body)?;
            let dst_tree_id = read_u64(&mut body)?;
            let src_key = read_bytes(&mut body)?;
            let dst_key = read_bytes(&mut body)?;
            let force = read_u8(&mut body)? != 0;
            TxnOp::Rename {
                src_tree_id,
                dst_tree_id,
                seq,
                src_key,
                dst_key,
                force,
            }
        }
        TY_NEW_TREE => {
            let tree_id = read_u64(&mut body)?;
            let name = read_bytes(&mut body)?;
            TxnOp::NewTree { tree_id, name }
        }
        TY_RM_TREE => {
            let tree_id = read_u64(&mut body)?;
            TxnOp::RmTree { tree_id }
        }
        TY_MEM_MARKER => TxnOp::MemMarker { seq },
        _ => return Err(sanity("unknown TxnOp variant tag")),
    };

    if !body.is_empty() {
        return Err(sanity("trailing bytes after variant body"));
    }
    Ok(op)
}

fn decode_reason(t: u8) -> Result<CompactReason> {
    match t {
        REASON_SPLIT_TOMBSTONE => Ok(CompactReason::SplitTombstone),
        REASON_SPLIT_GAP_SPACE => Ok(CompactReason::SplitGapSpace),
        REASON_OUT_OF_BLOB_FRAME => Ok(CompactReason::OutOfBlobFrame),
        _ => Err(sanity("unknown CompactReason tag")),
    }
}

fn read_u8(body: &mut &[u8]) -> Result<u8> {
    let (front, rest) = take(body, 1)?;
    *body = rest;
    Ok(front[0])
}

fn read_u16(body: &mut &[u8]) -> Result<u16> {
    let (front, rest) = take(body, 2)?;
    *body = rest;
    Ok(u16::from_le_bytes(front.try_into().unwrap()))
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

fn read_guid(body: &mut &[u8]) -> Result<BlobGuid> {
    let (front, rest) = take(body, 16)?;
    *body = rest;
    let mut g = [0u8; 16];
    g.copy_from_slice(front);
    Ok(g)
}

fn read_bytes(body: &mut &[u8]) -> Result<Vec<u8>> {
    let len = read_u32(body)? as usize;
    let (front, rest) = take(body, len)?;
    *body = rest;
    Ok(front.to_vec())
}

fn read_optional_bytes(body: &mut &[u8]) -> Result<Option<Vec<u8>>> {
    match read_u8(body)? {
        0 => Ok(None),
        1 => Ok(Some(read_bytes(body)?)),
        _ => Err(sanity("optional-bytes presence byte out of range")),
    }
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

    fn roundtrip(op: TxnOp, seq: u64) {
        let mut buf = Vec::new();
        encode_record(&op, seq, &mut buf).unwrap();

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
    fn roundtrip_insert_with_prev_value() {
        roundtrip(
            TxnOp::Insert {
                tree_id: 1,
                seq: 42,
                key: b"img/01.jpg".to_vec(),
                value: b"v-new".to_vec(),
                prev_value: Some(b"v-old".to_vec()),
            },
            42,
        );
    }

    #[test]
    fn roundtrip_insert_no_prev_value() {
        roundtrip(
            TxnOp::Insert {
                tree_id: 0,
                seq: 7,
                key: b"new/key".to_vec(),
                value: vec![0xAB; 200],
                prev_value: None,
            },
            7,
        );
    }

    #[test]
    fn roundtrip_erase() {
        roundtrip(
            TxnOp::Erase {
                tree_id: 3,
                seq: 99,
                key: b"img/02.jpg".to_vec(),
                value: b"v".to_vec(),
            },
            99,
        );
    }

    #[test]
    fn roundtrip_split() {
        roundtrip(
            TxnOp::Split {
                parent_blob: [0xAA; 16],
                pre_split_node: 123,
                new_child_blob: [0xBB; 16],
                new_child_entry: 7,
            },
            500,
        );
    }

    #[test]
    fn roundtrip_merge() {
        roundtrip(
            TxnOp::Merge {
                parent_blob: [0x33; 16],
                pre_merge_node: 200,
                child_blob: [0x44; 16],
            },
            501,
        );
    }

    #[test]
    fn roundtrip_compact_each_reason() {
        for reason in [
            CompactReason::SplitTombstone,
            CompactReason::SplitGapSpace,
            CompactReason::OutOfBlobFrame,
        ] {
            roundtrip(
                TxnOp::Compact {
                    blob: [0x77; 16],
                    reason,
                },
                700,
            );
        }
    }

    #[test]
    fn roundtrip_rename_object() {
        roundtrip(
            TxnOp::RenameObject {
                tree_id: 2,
                seq: 10,
                src_key: b"a/b".to_vec(),
                dst_key: b"a/c".to_vec(),
                force: true,
            },
            10,
        );
    }

    #[test]
    fn roundtrip_cross_tree_rename() {
        roundtrip(
            TxnOp::Rename {
                src_tree_id: 1,
                dst_tree_id: 2,
                seq: 11,
                src_key: b"x".to_vec(),
                dst_key: b"y".to_vec(),
                force: false,
            },
            11,
        );
    }

    #[test]
    fn roundtrip_new_and_rm_tree() {
        roundtrip(
            TxnOp::NewTree {
                tree_id: 5,
                name: b"bucket-images".to_vec(),
            },
            1,
        );
        roundtrip(TxnOp::RmTree { tree_id: 5 }, 2);
    }

    #[test]
    fn roundtrip_mem_marker() {
        roundtrip(TxnOp::MemMarker { seq: 999 }, 999);
    }

    #[test]
    fn record_length_breakdown_is_predictable() {
        let op = TxnOp::Insert {
            tree_id: 0,
            seq: 0,
            key: b"k".to_vec(),
            value: b"v".to_vec(),
            prev_value: None,
        };
        let mut buf = Vec::new();
        encode_record(&op, 0, &mut buf).unwrap();
        // tree_id (8) + key_len (4) + key (1) + val_len (4) + val (1)
        //   + prev_present (1) = 19 byte body
        // Header (17) + body (19) + footer (4) = 40 bytes.
        assert_eq!(buf.len(), 40);
    }

    #[test]
    fn corrupt_crc_is_caught() {
        let op = TxnOp::Insert {
            tree_id: 0,
            seq: 1,
            key: b"k".to_vec(),
            value: b"v".to_vec(),
            prev_value: None,
        };
        let mut buf = Vec::new();
        encode_record(&op, 1, &mut buf).unwrap();
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
        let op = TxnOp::MemMarker { seq: 5 };
        let mut buf = Vec::new();
        encode_record(&op, 5, &mut buf).unwrap();
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
        let op = TxnOp::Insert {
            tree_id: 0,
            seq: 1,
            key: vec![0xAB; 100],
            value: vec![0xCD; 100],
            prev_value: None,
        };
        let mut buf = Vec::new();
        encode_record(&op, 1, &mut buf).unwrap();
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
        let op = TxnOp::MemMarker { seq: 1 };
        let mut buf = Vec::new();
        encode_record(&op, 1, &mut buf).unwrap();
        // Overwrite the ty byte (header offset 16) with a bogus value.
        buf[16] = 0xFF;
        // Recompute the CRC so the corruption looks plausible
        // — confirms the "unknown tag" path triggers (and not "CRC").
        let body_end = RECORD_HEADER_SIZE; // MemMarker has empty body
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
            &TxnOp::Insert {
                tree_id: 0,
                seq: 1,
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
                prev_value: None,
            },
            1,
            &mut buf,
        )
        .unwrap();
        encode_record(
            &TxnOp::Erase {
                tree_id: 0,
                seq: 2,
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
            },
            2,
            &mut buf,
        )
        .unwrap();

        let r1 = decode_record(&buf).unwrap();
        assert_eq!(r1.seq, 1);
        let r2 = decode_record(&buf[r1.bytes_consumed..]).unwrap();
        assert_eq!(r2.seq, 2);
        assert_eq!(r1.bytes_consumed + r2.bytes_consumed, buf.len());
    }
}
