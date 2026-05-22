//! Forward WAL scanner — read every record in a WAL file in order
//! and yield them to a callback.
//!
//! The scanner is **torn-tail-tolerant**: a partially written
//! record at the end of the file is the expected outcome of a
//! crash during a buffered write. We stop cleanly when we hit one
//! and report its offset in [`ReplayStats::torn_tail_at`].
//!
//! Failures earlier in the file (a record whose CRC mismatches,
//! whose magic is wrong, or whose body parses with a trailing
//! variant tag, etc.) propagate as
//! [`Error::ReplaySanityFailed`] with the byte offset of the bad
//! record patched in — the caller can no longer trust the log
//! and should not continue replay.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::api::errors::{Error, Result};

use super::codec::{decode_file_header, decode_record, FileHeader, FILE_HEADER_SIZE};
use super::wal_op::WalOp;

/// Outcome of a successful scan.
///
/// The callback receives the sequence for each record it handles;
/// this summary is the file-level replay boundary used by reopen
/// and tests.
#[derive(Debug, Clone, Copy)]
pub struct ReplayStats {
    /// Number of records the callback was invoked for.
    pub records_seen: u64,
    /// Largest `seq` observed across all records, or `None` if the
    /// file had no records past the header.
    pub highest_seq: Option<u64>,
    /// Byte offset where the scan stopped due to a torn tail, or
    /// `None` if the file ended cleanly on a record boundary.
    pub torn_tail_at: Option<u64>,
}

/// Open `path`, validate its file header, and yield every record
/// to `callback`. The callback receives `(op, seq, record_offset)`
/// where `record_offset` is the byte position the record starts at
/// inside the file.
///
/// The callback may return an error to abort replay — the function
/// then propagates that error verbatim with the current file
/// offset patched onto any sanity-failure variant it carries.
pub fn replay<F>(path: &Path, mut callback: F) -> Result<(FileHeader, ReplayStats)>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    replay_bytes(&bytes, &mut callback)
}

/// Same as [`replay`] but reads from an in-memory buffer. Splitting
/// the I/O out makes unit tests trivially exercise both paths
/// (file vs. raw buffer) with the same logic.
pub fn replay_bytes<F>(bytes: &[u8], callback: &mut F) -> Result<(FileHeader, ReplayStats)>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    if bytes.len() < FILE_HEADER_SIZE {
        return Err(Error::ReplaySanityFailed {
            context: "WAL too short — missing file header",
            record_offset: 0,
        });
    }
    let header = decode_file_header(&bytes[..FILE_HEADER_SIZE])?;

    let mut offset = FILE_HEADER_SIZE;
    let mut records_seen = 0u64;
    let mut highest_seq: Option<u64> = None;
    let mut torn_tail_at: Option<u64> = None;

    while offset < bytes.len() {
        match decode_record(&bytes[offset..]) {
            Ok(r) => {
                // Flatten Batch transparently: the callback never
                // sees a `WalOp::Batch`, just the inner primitive
                // ops with derived seqs (`base + i`, mirroring the
                // encoder's contiguous seq reservation).
                if let WalOp::Batch { ops } = &r.op {
                    for (i, inner) in ops.iter().enumerate() {
                        let inner_seq = r.seq.wrapping_add(i as u64);
                        callback(inner, inner_seq, offset as u64)
                            .map_err(|e| patch_offset(e, offset))?;
                        highest_seq = Some(match highest_seq {
                            None => inner_seq,
                            Some(s) => s.max(inner_seq),
                        });
                    }
                } else {
                    callback(&r.op, r.seq, offset as u64).map_err(|e| patch_offset(e, offset))?;
                    highest_seq = Some(match highest_seq {
                        None => r.seq,
                        Some(s) => s.max(r.seq),
                    });
                }
                records_seen += 1;
                offset += r.bytes_consumed;
            }
            Err(Error::ReplaySanityFailed { context, .. }) if is_torn_tail(context) => {
                // Partial record at EOF — the expected outcome of
                // a crash during a buffered write. Stop here.
                torn_tail_at = Some(offset as u64);
                break;
            }
            Err(e) => {
                return Err(patch_offset(e, offset));
            }
        }
    }

    Ok((
        header,
        ReplayStats {
            records_seen,
            highest_seq,
            torn_tail_at,
        },
    ))
}

fn is_torn_tail(context: &'static str) -> bool {
    // Two codec sanity-failure cases are consistent with a torn
    // tail at EOF and not a corrupted middle: header / body
    // truncation. CRC mismatch / magic mismatch / unknown variant
    // tag etc. mean the bytes are *present* but invalid, which is
    // real corruption, not a torn write.
    context == "record header truncated" || context == "record body truncated"
}

fn patch_offset(e: Error, offset: usize) -> Error {
    match e {
        Error::ReplaySanityFailed { context, .. } => Error::ReplaySanityFailed {
            context,
            record_offset: offset as u64,
        },
        other => other,
    }
}
