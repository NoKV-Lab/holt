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
use std::os::unix::fs::FileExt;
#[cfg(test)]
use std::path::Path;

use crate::api::errors::{Error, Result};

use crc32fast::Hasher;

use super::codec::{
    decode_file_header, decode_record_header, validate_record_body, visit_record_body, FileHeader,
    RecordBodyCursor, RecordHeader, FILE_HEADER_SIZE, MAX_ATOMIC_WAL_RECORD_BYTES,
    RECORD_FOOTER_SIZE, RECORD_HEADER_SIZE, RECORD_MAGIC,
};
use super::wal_op::WalOp;

/// Fixed replay scratch. Neither one record nor the whole WAL is materialized.
const REPLAY_CHUNK_BYTES: usize = 16 * 1024;
/// A malicious torn payload can contain many fake magic strings. Both the
/// byte scan and candidate validation remain bounded; exhausting either
/// budget fails closed as corruption instead of classifying the tail as
/// safely truncatable.
const MAX_SUFFIX_CANDIDATES: usize = 4096;
const MAX_SUFFIX_VERIFICATION_BYTES: u64 = (MAX_ATOMIC_WAL_RECORD_BYTES as u64) * 2;

/// Outcome of a successful scan.
///
/// The callback receives the sequence for each record it handles;
/// this summary is the file-level replay boundary used by reopen
/// and tests.
#[derive(Debug, Clone, Copy)]
pub struct ReplayStats {
    /// Number of physical WAL records validated and applied. A Batch record
    /// may invoke the callback once per logical inner operation.
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
#[cfg(test)]
pub fn replay<F>(path: &Path, callback: F) -> Result<(FileHeader, ReplayStats)>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    let file = File::open(path)?;
    replay_file_with_owner(&file, None, None, callback)
}

/// Replay from an already-open file object, checking its file owner and, for
/// a standalone Tree, every primitive record owner before any callback.
pub(crate) fn replay_file<F>(
    file: &File,
    expected_file_tree_id: u64,
    expected_primitive_tree_id: Option<u64>,
    callback: F,
) -> Result<(FileHeader, ReplayStats)>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    replay_file_with_owner(
        file,
        Some(expected_file_tree_id),
        expected_primitive_tree_id,
        callback,
    )
}

fn replay_file_with_owner<F>(
    file: &File,
    expected_file_tree_id: Option<u64>,
    expected_primitive_tree_id: Option<u64>,
    mut callback: F,
) -> Result<(FileHeader, ReplayStats)>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    let file_len = file.metadata()?.len();
    if file_len < FILE_HEADER_SIZE as u64 {
        return Err(Error::ReplaySanityFailed {
            context: "WAL too short — missing file header",
            record_offset: 0,
        });
    }
    let mut file_header_bytes = [0u8; FILE_HEADER_SIZE];
    file.read_exact_at(&mut file_header_bytes, 0)?;
    let file_header = decode_file_header(&file_header_bytes)?;
    if expected_file_tree_id.is_some_and(|expected| file_header.tree_id != expected) {
        return Err(Error::ReplaySanityFailed {
            context: "WAL file tree_id mismatch on open",
            record_offset: 0,
        });
    }

    // Validate the entire visible WAL before the first callback. A corrupt
    // later record must not let an earlier callback create/flush a root or
    // otherwise leave persistent recovery side effects.
    let validated = validate_file_stream(
        file,
        file_len,
        file_header.tree_id,
        expected_primitive_tree_id,
    )?;
    let mut highest_seq = None;
    apply_file_stream(file, validated.valid_end, &mut callback, &mut highest_seq)?;

    Ok((
        file_header,
        ReplayStats {
            records_seen: validated.records_seen,
            highest_seq,
            torn_tail_at: validated.torn_tail_at,
        },
    ))
}

/// Same as [`replay`] but reads from an in-memory buffer. Splitting
/// the I/O out makes unit tests trivially exercise both paths
/// (file vs. raw buffer) with the same logic.
#[cfg(test)]
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
    let validated = validate_slice_stream(bytes, header.tree_id, None)?;
    let mut highest_seq = None;
    apply_slice_stream(bytes, validated.valid_end, callback, &mut highest_seq)?;

    Ok((
        header,
        ReplayStats {
            records_seen: validated.records_seen,
            highest_seq,
            torn_tail_at: validated.torn_tail_at,
        },
    ))
}

#[derive(Debug, Clone, Copy)]
struct ValidatedStream {
    records_seen: u64,
    valid_end: u64,
    torn_tail_at: Option<u64>,
}

struct FileBodyCursor<'file, 'hash> {
    file: &'file File,
    position: u64,
    end: u64,
    buffer: [u8; REPLAY_CHUNK_BYTES],
    buffered: usize,
    consumed: usize,
    hasher: Option<&'hash mut Hasher>,
}

impl<'file, 'hash> FileBodyCursor<'file, 'hash> {
    fn new(file: &'file File, start: u64, end: u64, hasher: Option<&'hash mut Hasher>) -> Self {
        Self {
            file,
            position: start,
            end,
            buffer: [0; REPLAY_CHUNK_BYTES],
            buffered: 0,
            consumed: 0,
            hasher,
        }
    }

    fn refill(&mut self) -> Result<()> {
        let remaining = self.end.saturating_sub(self.position);
        let wanted = usize::try_from(remaining.min(REPLAY_CHUNK_BYTES as u64))
            .map_err(|_| Error::Internal("WAL replay chunk length conversion failed"))?;
        if wanted == 0 {
            return Err(Error::ReplaySanityFailed {
                context: "body truncated",
                record_offset: 0,
            });
        }
        self.file
            .read_exact_at(&mut self.buffer[..wanted], self.position)?;
        self.buffered = wanted;
        self.consumed = 0;
        Ok(())
    }

    fn consume(&mut self, mut len: u64) -> Result<()> {
        while len != 0 {
            if self.consumed == self.buffered {
                self.refill()?;
            }
            let available = self.buffered - self.consumed;
            let take = usize::try_from(len.min(available as u64))
                .map_err(|_| Error::Internal("WAL replay chunk length conversion failed"))?;
            let chunk = &self.buffer[self.consumed..self.consumed + take];
            if let Some(hasher) = self.hasher.as_deref_mut() {
                hasher.update(chunk);
            }
            self.consumed += take;
            self.position = self
                .position
                .checked_add(take as u64)
                .ok_or(Error::Internal("WAL replay file offset overflow"))?;
            len -= take as u64;
        }
        Ok(())
    }
}

impl RecordBodyCursor for FileBodyCursor<'_, '_> {
    fn remaining(&self) -> u64 {
        self.end.saturating_sub(self.position)
    }

    fn read_exact_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        let mut written = 0usize;
        while written < out.len() {
            if self.consumed == self.buffered {
                self.refill()?;
            }
            let take = (out.len() - written).min(self.buffered - self.consumed);
            let chunk = &self.buffer[self.consumed..self.consumed + take];
            out[written..written + take].copy_from_slice(chunk);
            if let Some(hasher) = self.hasher.as_deref_mut() {
                hasher.update(chunk);
            }
            self.consumed += take;
            self.position = self
                .position
                .checked_add(take as u64)
                .ok_or(Error::Internal("WAL replay file offset overflow"))?;
            written += take;
        }
        Ok(())
    }

    fn skip_bytes(&mut self, len: u64) -> Result<()> {
        self.consume(len)
    }
}

#[cfg(test)]
struct SliceBodyCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

#[cfg(test)]
impl<'a> SliceBodyCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
}

#[cfg(test)]
impl RecordBodyCursor for SliceBodyCursor<'_> {
    fn remaining(&self) -> u64 {
        (self.bytes.len() - self.position) as u64
    }

    fn read_exact_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        let end = self
            .position
            .checked_add(out.len())
            .ok_or(Error::Internal("WAL replay slice offset overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::ReplaySanityFailed {
                context: "body truncated",
                record_offset: 0,
            });
        }
        out.copy_from_slice(&self.bytes[self.position..end]);
        self.position = end;
        Ok(())
    }

    fn skip_bytes(&mut self, len: u64) -> Result<()> {
        let len = usize::try_from(len)
            .map_err(|_| Error::Internal("WAL replay slice length conversion failed"))?;
        let end = self
            .position
            .checked_add(len)
            .ok_or(Error::Internal("WAL replay slice offset overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::ReplaySanityFailed {
                context: "body truncated",
                record_offset: 0,
            });
        }
        self.position = end;
        Ok(())
    }
}

fn checked_add_offset(offset: u64, len: usize) -> Result<u64> {
    offset
        .checked_add(
            u64::try_from(len).map_err(|_| Error::Internal("WAL length conversion failed"))?,
        )
        .ok_or(Error::Internal("WAL file offset overflow"))
}

fn read_record_header_at(
    file: &File,
    offset: u64,
) -> Result<([u8; RECORD_HEADER_SIZE], RecordHeader)> {
    let mut bytes = [0u8; RECORD_HEADER_SIZE];
    file.read_exact_at(&mut bytes, offset)?;
    let header = decode_record_header(&bytes)?;
    Ok((bytes, header))
}

fn validate_file_record(
    file: &File,
    offset: u64,
    raw_header: &[u8; RECORD_HEADER_SIZE],
    header: RecordHeader,
    file_tree_id: u64,
    expected_primitive_tree_id: Option<u64>,
) -> Result<()> {
    let body_start = checked_add_offset(offset, RECORD_HEADER_SIZE)?;
    let body_end = checked_add_offset(body_start, header.body_len as usize)?;
    let mut hasher = Hasher::new();
    hasher.update(raw_header);
    {
        let mut cursor = FileBodyCursor::new(file, body_start, body_end, Some(&mut hasher));
        validate_record_body(
            header,
            &mut cursor,
            file_tree_id,
            expected_primitive_tree_id,
        )?;
    }

    let mut footer = [0u8; RECORD_FOOTER_SIZE];
    file.read_exact_at(&mut footer, body_end)?;
    let expected_crc = u32::from_le_bytes(footer[..4].try_into().unwrap());
    let repeated_len = u32::from_le_bytes(footer[4..8].try_into().unwrap());
    if repeated_len != header.body_len {
        return Err(Error::ReplaySanityFailed {
            context: "record footer body length mismatch",
            record_offset: 0,
        });
    }
    if hasher.finalize() != expected_crc {
        return Err(Error::ReplaySanityFailed {
            context: "record CRC mismatch",
            record_offset: 0,
        });
    }
    Ok(())
}

/// Validate only the redundant v4 frame boundary and CRC. Torn-tail
/// classification deliberately uses this weaker check: a complete frame with
/// an invalid semantic body is still corruption, never a safely truncatable
/// partial write.
fn validate_file_frame(
    file: &File,
    offset: u64,
    raw_header: &[u8; RECORD_HEADER_SIZE],
    header: RecordHeader,
) -> Result<()> {
    let body_start = checked_add_offset(offset, RECORD_HEADER_SIZE)?;
    let body_end = checked_add_offset(body_start, header.body_len as usize)?;
    let mut hasher = Hasher::new();
    hasher.update(raw_header);
    {
        let mut cursor = FileBodyCursor::new(file, body_start, body_end, Some(&mut hasher));
        cursor.consume(u64::from(header.body_len))?;
    }

    let mut footer = [0u8; RECORD_FOOTER_SIZE];
    file.read_exact_at(&mut footer, body_end)?;
    let expected_crc = u32::from_le_bytes(footer[..4].try_into().unwrap());
    let repeated_len = u32::from_le_bytes(footer[4..8].try_into().unwrap());
    if repeated_len != header.body_len {
        return Err(Error::ReplaySanityFailed {
            context: "record footer body length mismatch",
            record_offset: 0,
        });
    }
    if hasher.finalize() != expected_crc {
        return Err(Error::ReplaySanityFailed {
            context: "record CRC mismatch",
            record_offset: 0,
        });
    }
    Ok(())
}

#[cfg(test)]
fn validate_slice_record(
    record: &[u8],
    header: RecordHeader,
    file_tree_id: u64,
    expected_primitive_tree_id: Option<u64>,
) -> Result<()> {
    let body_end = RECORD_HEADER_SIZE
        .checked_add(header.body_len as usize)
        .ok_or(Error::Internal("WAL body offset overflow"))?;
    let footer = &record[body_end..body_end + RECORD_FOOTER_SIZE];
    let repeated_len = u32::from_le_bytes(footer[4..8].try_into().unwrap());
    if repeated_len != header.body_len {
        return Err(Error::ReplaySanityFailed {
            context: "record footer body length mismatch",
            record_offset: 0,
        });
    }
    let expected_crc = u32::from_le_bytes(footer[..4].try_into().unwrap());
    if super::codec::crc32(&record[..body_end]) != expected_crc {
        return Err(Error::ReplaySanityFailed {
            context: "record CRC mismatch",
            record_offset: 0,
        });
    }
    let mut cursor = SliceBodyCursor::new(&record[RECORD_HEADER_SIZE..body_end]);
    validate_record_body(
        header,
        &mut cursor,
        file_tree_id,
        expected_primitive_tree_id,
    )
}

#[cfg(test)]
fn validate_slice_frame(
    record: &[u8],
    raw_header: &[u8; RECORD_HEADER_SIZE],
    header: RecordHeader,
) -> Result<()> {
    let body_end = RECORD_HEADER_SIZE
        .checked_add(header.body_len as usize)
        .ok_or(Error::Internal("WAL body offset overflow"))?;
    let footer = &record[body_end..body_end + RECORD_FOOTER_SIZE];
    let repeated_len = u32::from_le_bytes(footer[4..8].try_into().unwrap());
    if repeated_len != header.body_len {
        return Err(Error::ReplaySanityFailed {
            context: "record footer body length mismatch",
            record_offset: 0,
        });
    }
    let expected_crc = u32::from_le_bytes(footer[..4].try_into().unwrap());
    let mut hasher = Hasher::new();
    hasher.update(raw_header);
    hasher.update(&record[RECORD_HEADER_SIZE..body_end]);
    if hasher.finalize() != expected_crc {
        return Err(Error::ReplaySanityFailed {
            context: "record CRC mismatch",
            record_offset: 0,
        });
    }
    Ok(())
}

#[derive(Default)]
struct CandidateBudget {
    candidates: usize,
    work_bytes: u64,
}

impl CandidateBudget {
    fn charge_scan(&mut self, bytes: usize) -> bool {
        self.work_bytes = self.work_bytes.saturating_add(bytes as u64);
        self.work_bytes <= MAX_SUFFIX_VERIFICATION_BYTES
    }

    fn charge(&mut self, bytes: usize) -> bool {
        self.candidates = self.candidates.saturating_add(1);
        self.candidates <= MAX_SUFFIX_CANDIDATES && self.charge_scan(bytes)
    }
}

fn sanity_candidate(result: Result<()>) -> Result<bool> {
    match result {
        Ok(()) => Ok(true),
        Err(Error::ReplaySanityFailed { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

fn corrected_header_length_exists_file(
    file: &File,
    record_offset: u64,
    file_len: u64,
    raw_header: &[u8; RECORD_HEADER_SIZE],
    budget: &mut CandidateBudget,
) -> Result<bool> {
    let first_repeat = checked_add_offset(record_offset, RECORD_HEADER_SIZE + 4)?;
    if file_len.saturating_sub(first_repeat) < 4 {
        return Ok(false);
    }
    if !budget.charge_scan(4) {
        return Ok(true);
    }
    let mut cursor = FileBodyCursor::new(file, first_repeat, file_len, None);
    let mut window = [0u8; 4];
    cursor.read_exact_bytes(&mut window)?;
    let mut window_offset = first_repeat;
    loop {
        let expected_len = window_offset.saturating_sub(first_repeat);
        if u32::try_from(expected_len).is_ok()
            && u64::from(u32::from_le_bytes(window)) == expected_len
        {
            let mut corrected = *raw_header;
            corrected[4..8].copy_from_slice(&(expected_len as u32).to_le_bytes());
            if let Ok(header) = decode_record_header(&corrected) {
                // A payload may coincidentally contain four bytes equal to
                // their distance from the footer slot. It is a candidate only
                // when the reconstructed header itself satisfies all native
                // framing limits.
                if !budget.charge(header.total_len) {
                    return Ok(true);
                }
                if sanity_candidate(validate_file_frame(file, record_offset, &corrected, header))? {
                    return Ok(true);
                }
            }
        }
        if cursor.remaining() == 0 {
            break;
        }
        if !budget.charge_scan(1) {
            return Ok(true);
        }
        window.rotate_left(1);
        cursor.read_exact_bytes(&mut window[3..4])?;
        window_offset = window_offset
            .checked_add(1)
            .ok_or(Error::Internal("WAL suffix scan offset overflow"))?;
    }
    Ok(false)
}

fn valid_suffix_exists_file(
    file: &File,
    record_offset: u64,
    file_len: u64,
    budget: &mut CandidateBudget,
) -> Result<bool> {
    let start = record_offset
        .checked_add(1)
        .ok_or(Error::Internal("WAL suffix scan offset overflow"))?;
    if file_len.saturating_sub(start) < RECORD_HEADER_SIZE as u64 {
        return Ok(false);
    }
    if !budget.charge_scan(4) {
        return Ok(true);
    }
    let mut cursor = FileBodyCursor::new(file, start, file_len, None);
    let mut window = [0u8; 4];
    cursor.read_exact_bytes(&mut window)?;
    let magic = RECORD_MAGIC.to_le_bytes();
    let mut candidate = start;
    loop {
        if window == magic && file_len.saturating_sub(candidate) >= RECORD_HEADER_SIZE as u64 {
            match read_record_header_at(file, candidate) {
                Ok((raw_header, header)) => {
                    let end = checked_add_offset(candidate, header.total_len)?;
                    if end <= file_len {
                        let mut repeated = [0u8; 4];
                        file.read_exact_at(&mut repeated, end - 4)?;
                        if u32::from_le_bytes(repeated) == header.body_len {
                            if !budget.charge(header.total_len) {
                                return Ok(true);
                            }
                            if sanity_candidate(validate_file_frame(
                                file,
                                candidate,
                                &raw_header,
                                header,
                            ))? {
                                return Ok(true);
                            }
                        }
                    }
                }
                Err(Error::ReplaySanityFailed { .. }) => {
                    // Payload magic is not a frame unless the complete
                    // header/footer/LEN2/CRC checks below accept it.
                }
                Err(error) => return Err(error),
            }
        }
        if cursor.remaining() == 0 {
            break;
        }
        if !budget.charge_scan(1) {
            return Ok(true);
        }
        window.rotate_left(1);
        cursor.read_exact_bytes(&mut window[3..4])?;
        candidate = candidate
            .checked_add(1)
            .ok_or(Error::Internal("WAL suffix scan offset overflow"))?;
    }
    Ok(false)
}

fn truncated_file_record_is_corruption(
    file: &File,
    record_offset: u64,
    file_len: u64,
    raw_header: &[u8; RECORD_HEADER_SIZE],
) -> Result<bool> {
    let mut budget = CandidateBudget::default();
    if corrected_header_length_exists_file(file, record_offset, file_len, raw_header, &mut budget)?
    {
        return Ok(true);
    }
    valid_suffix_exists_file(file, record_offset, file_len, &mut budget)
}

fn validate_file_stream(
    file: &File,
    file_len: u64,
    file_tree_id: u64,
    expected_primitive_tree_id: Option<u64>,
) -> Result<ValidatedStream> {
    let mut offset = FILE_HEADER_SIZE as u64;
    let mut records_seen = 0u64;
    while offset < file_len {
        if file_len - offset < RECORD_HEADER_SIZE as u64 {
            return Ok(ValidatedStream {
                records_seen,
                valid_end: offset,
                torn_tail_at: Some(offset),
            });
        }
        let (raw_header, header) =
            read_record_header_at(file, offset).map_err(|error| patch_offset(error, offset))?;
        let end = checked_add_offset(offset, header.total_len)?;
        if end > file_len {
            if truncated_file_record_is_corruption(file, offset, file_len, &raw_header)? {
                return Err(Error::ReplaySanityFailed {
                    context: "truncated record precedes validated WAL data",
                    record_offset: offset,
                });
            }
            return Ok(ValidatedStream {
                records_seen,
                valid_end: offset,
                torn_tail_at: Some(offset),
            });
        }
        validate_file_record(
            file,
            offset,
            &raw_header,
            header,
            file_tree_id,
            expected_primitive_tree_id,
        )
        .map_err(|error| patch_offset(error, offset))?;
        records_seen = records_seen
            .checked_add(1)
            .ok_or(Error::Internal("WAL physical record count overflow"))?;
        offset = end;
    }
    Ok(ValidatedStream {
        records_seen,
        valid_end: offset,
        torn_tail_at: None,
    })
}

fn apply_file_stream<F>(
    file: &File,
    valid_end: u64,
    callback: &mut F,
    highest_seq: &mut Option<u64>,
) -> Result<()>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    let mut offset = FILE_HEADER_SIZE as u64;
    while offset < valid_end {
        let (_raw_header, header) =
            read_record_header_at(file, offset).map_err(|error| patch_offset(error, offset))?;
        let body_start = checked_add_offset(offset, RECORD_HEADER_SIZE)?;
        let body_end = checked_add_offset(body_start, header.body_len as usize)?;
        let mut cursor = FileBodyCursor::new(file, body_start, body_end, None);
        visit_record_body(header, &mut cursor, &mut |op, seq| {
            callback(op, seq, offset).map_err(|error| patch_offset(error, offset))?;
            *highest_seq = Some(highest_seq.map_or(seq, |highest| highest.max(seq)));
            Ok(())
        })
        .map_err(|error| patch_offset(error, offset))?;
        offset = checked_add_offset(offset, header.total_len)?;
    }
    Ok(())
}

#[cfg(test)]
fn corrected_header_length_exists_slice(
    bytes: &[u8],
    record_offset: usize,
    raw_header: &[u8; RECORD_HEADER_SIZE],
    budget: &mut CandidateBudget,
) -> Result<bool> {
    let first_repeat = record_offset + RECORD_HEADER_SIZE + 4;
    if bytes.len().saturating_sub(first_repeat) < 4 {
        return Ok(false);
    }
    for position in first_repeat..=bytes.len() - 4 {
        if !budget.charge_scan(1) {
            return Ok(true);
        }
        let expected_len = position - first_repeat;
        if expected_len > u32::MAX as usize {
            break;
        }
        if u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap()) as usize
            != expected_len
        {
            continue;
        }
        let mut corrected = *raw_header;
        corrected[4..8].copy_from_slice(&(expected_len as u32).to_le_bytes());
        let Ok(header) = decode_record_header(&corrected) else {
            continue;
        };
        let end = record_offset + header.total_len;
        if end > bytes.len() {
            continue;
        }
        if !budget.charge(header.total_len) {
            return Ok(true);
        }
        if sanity_candidate(validate_slice_frame(
            &bytes[record_offset..end],
            &corrected,
            header,
        ))? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
fn valid_suffix_exists_slice(
    bytes: &[u8],
    record_offset: usize,
    budget: &mut CandidateBudget,
) -> Result<bool> {
    let magic = RECORD_MAGIC.to_le_bytes();
    if bytes.len().saturating_sub(record_offset + 1) < RECORD_HEADER_SIZE {
        return Ok(false);
    }
    for candidate in record_offset + 1..=bytes.len() - RECORD_HEADER_SIZE {
        if !budget.charge_scan(1) {
            return Ok(true);
        }
        if bytes[candidate..candidate + 4] != magic {
            continue;
        }
        let raw_header: [u8; RECORD_HEADER_SIZE] = bytes[candidate..candidate + RECORD_HEADER_SIZE]
            .try_into()
            .unwrap();
        let Ok(header) = decode_record_header(&raw_header) else {
            continue;
        };
        let Some(end) = candidate.checked_add(header.total_len) else {
            continue;
        };
        if end > bytes.len()
            || u32::from_le_bytes(bytes[end - 4..end].try_into().unwrap()) != header.body_len
        {
            continue;
        }
        if !budget.charge(header.total_len) {
            return Ok(true);
        }
        if sanity_candidate(validate_slice_frame(
            &bytes[candidate..end],
            &raw_header,
            header,
        ))? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
fn validate_slice_stream(
    bytes: &[u8],
    file_tree_id: u64,
    expected_primitive_tree_id: Option<u64>,
) -> Result<ValidatedStream> {
    let mut offset = FILE_HEADER_SIZE;
    let mut records_seen = 0u64;
    while offset < bytes.len() {
        if bytes.len() - offset < RECORD_HEADER_SIZE {
            return Ok(ValidatedStream {
                records_seen,
                valid_end: offset as u64,
                torn_tail_at: Some(offset as u64),
            });
        }
        let raw_header: [u8; RECORD_HEADER_SIZE] = bytes[offset..offset + RECORD_HEADER_SIZE]
            .try_into()
            .unwrap();
        let header = decode_record_header(&raw_header)
            .map_err(|error| patch_offset(error, offset as u64))?;
        let Some(end) = offset.checked_add(header.total_len) else {
            return Err(Error::ReplaySanityFailed {
                context: "record length overflow",
                record_offset: offset as u64,
            });
        };
        if end > bytes.len() {
            let mut budget = CandidateBudget::default();
            if corrected_header_length_exists_slice(bytes, offset, &raw_header, &mut budget)?
                || valid_suffix_exists_slice(bytes, offset, &mut budget)?
            {
                return Err(Error::ReplaySanityFailed {
                    context: "truncated record precedes validated WAL data",
                    record_offset: offset as u64,
                });
            }
            return Ok(ValidatedStream {
                records_seen,
                valid_end: offset as u64,
                torn_tail_at: Some(offset as u64),
            });
        }
        validate_slice_record(
            &bytes[offset..end],
            header,
            file_tree_id,
            expected_primitive_tree_id,
        )
        .map_err(|error| patch_offset(error, offset as u64))?;
        records_seen = records_seen
            .checked_add(1)
            .ok_or(Error::Internal("WAL physical record count overflow"))?;
        offset = end;
    }
    Ok(ValidatedStream {
        records_seen,
        valid_end: offset as u64,
        torn_tail_at: None,
    })
}

#[cfg(test)]
fn apply_slice_stream<F>(
    bytes: &[u8],
    valid_end: u64,
    callback: &mut F,
    highest_seq: &mut Option<u64>,
) -> Result<()>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    let valid_end = usize::try_from(valid_end)
        .map_err(|_| Error::Internal("WAL replay slice end conversion failed"))?;
    let mut offset = FILE_HEADER_SIZE;
    while offset < valid_end {
        let header = decode_record_header(&bytes[offset..offset + RECORD_HEADER_SIZE])?;
        let body_start = offset + RECORD_HEADER_SIZE;
        let body_end = body_start + header.body_len as usize;
        let mut cursor = SliceBodyCursor::new(&bytes[body_start..body_end]);
        visit_record_body(header, &mut cursor, &mut |op, seq| {
            callback(op, seq, offset as u64).map_err(|error| patch_offset(error, offset as u64))?;
            *highest_seq = Some(highest_seq.map_or(seq, |highest| highest.max(seq)));
            Ok(())
        })
        .map_err(|error| patch_offset(error, offset as u64))?;
        offset += header.total_len;
    }
    Ok(())
}

fn patch_offset(e: Error, offset: u64) -> Error {
    match e {
        Error::ReplaySanityFailed { context, .. } => Error::ReplaySanityFailed {
            context,
            record_offset: offset,
        },
        other => other,
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    #[test]
    fn exhausted_suffix_scan_budget_fails_closed() {
        let bytes = vec![0; RECORD_HEADER_SIZE + 8];
        let raw_header = [0; RECORD_HEADER_SIZE];
        let mut budget = CandidateBudget {
            candidates: 0,
            work_bytes: MAX_SUFFIX_VERIFICATION_BYTES,
        };

        assert!(
            corrected_header_length_exists_slice(&bytes, 0, &raw_header, &mut budget).unwrap(),
            "budget exhaustion must classify the tail as corruption"
        );
    }
}
