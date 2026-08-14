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
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::api::errors::{Error, Result};
use crate::api::journal::{
    JournalAnchor, JournalEnvelope, JournalEnvelopePage, JournalState, MAX_JOURNAL_RECORD_BYTES,
};

use super::codec::{
    decode_file_header, decode_record, file_header_size_from_prefix, FileHeader,
    LEGACY_FILE_HEADER_SIZE, LEGACY_FORMAT_VERSION, RECORD_FOOTER_SIZE, RECORD_HEADER_SIZE,
    RECORD_MAGIC,
};
use super::wal_op::WalOp;

#[cfg(test)]
thread_local! {
    static STREAMING_RECORDS_DECODED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

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

/// Validated attached-envelope suffix retained after the checkpoint anchor.
#[cfg(test)]
pub(crate) struct AttachedEnvelopeScan {
    pub(crate) checkpoint: Option<JournalAnchor>,
    pub(crate) tail: Option<JournalAnchor>,
    pub(crate) envelopes: Vec<JournalEnvelope>,
}

/// A validated place from which a sequential caller can continue scanning.
///
/// This is process-local only. The checkpoint is re-read and compared before
/// the offset is trusted, and checkpoint/truncate clears the coordinator's
/// cached value.
#[derive(Clone, Copy)]
pub(crate) struct AttachedEnvelopeResume {
    checkpoint: JournalAnchor,
    cursor: JournalAnchor,
    record_offset: u64,
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

/// Reject a nonempty format-3 WAL before a writable open mutates database
/// state. A header-only format-3 file remains eligible for the writer's
/// in-place format-4 header upgrade.
pub(crate) fn preflight_writable_wal(path: &Path) -> Result<()> {
    let mut records = StreamingRecords::open(path)?;
    if records.header.version == LEGACY_FORMAT_VERSION
        && records.file_len != LEGACY_FILE_HEADER_SIZE as u64
    {
        return Err(Error::ReplaySanityFailed {
            context: "nonempty WAL format 3 is replay-only; checkpoint it with a format-3 Holt binary before v4 writes",
            record_offset: 0,
        });
    }
    if records.header.checkpoint_anchor.is_some() {
        while records.next_record()?.is_some() {}
    }
    Ok(())
}

/// Same as [`replay`] but reads from an in-memory buffer. Splitting
/// the I/O out makes unit tests trivially exercise both paths
/// (file vs. raw buffer) with the same logic.
pub fn replay_bytes<F>(bytes: &[u8], callback: &mut F) -> Result<(FileHeader, ReplayStats)>
where
    F: FnMut(&WalOp, u64, u64) -> Result<()>,
{
    if bytes.len() < LEGACY_FILE_HEADER_SIZE {
        return Err(Error::ReplaySanityFailed {
            context: "WAL too short — missing file header",
            record_offset: 0,
        });
    }
    let header = decode_file_header(bytes)?;

    let mut offset = header.record_offset();
    let mut records_seen = 0u64;
    let mut highest_seq: Option<u64> = None;
    let mut torn_tail_at: Option<u64> = None;

    while offset < bytes.len() {
        match decode_record(&bytes[offset..]) {
            Ok(r) => {
                if header.version == LEGACY_FORMAT_VERSION
                    && matches!(&r.op, WalOp::DbBatchWithEnvelope { .. })
                {
                    return Err(Error::ReplaySanityFailed {
                        context: "attached batch requires WAL format version 4",
                        record_offset: offset as u64,
                    });
                }
                validate_record_mode(&header, &r.op, offset as u64)?;

                // Flatten both batch variants transparently: the callback sees
                // the inner primitive ops with derived seqs (`base + i`). The
                // direct record decoder retains an attached envelope for the
                // record-level recovery scanner added by the DB API layer.
                let batch_ops = match &r.op {
                    WalOp::Batch { ops } => Some(ops),
                    WalOp::DbBatchWithEnvelope { envelope, ops } => {
                        debug_assert!(!envelope.payload().is_empty());
                        Some(ops)
                    }
                    _ => None,
                };
                if let Some(ops) = batch_ops {
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

/// Validate an attached journal and return its durable and live anchors.
///
/// The scanner retains at most one encoded WAL record. It validates every
/// record CRC and the full attached-envelope chain, but does not retain the
/// application payload suffix.
pub(crate) fn validate_attached_journal(path: &Path) -> Result<Option<JournalState>> {
    let mut records = StreamingRecords::open(path)?;
    let mut chain = AttachedChain::new(records.header.checkpoint_anchor);
    while let Some(record) = records.next_record()? {
        if let WalOp::DbBatchWithEnvelope { envelope, .. } = record.op {
            chain.observe(records.header.version, &envelope, record.offset)?;
        }
    }
    Ok(chain.state())
}

/// Return the initialized attached-stream state for a read-only caller.
pub(crate) fn attached_journal_state(path: &Path) -> Result<JournalState> {
    validate_attached_journal(path)?.ok_or(Error::JournalStreamUnavailable {
        reason: "stream has not been initialized",
    })
}

/// Read one bounded page after `cursor` without materializing the retained WAL
/// suffix. This stateless entry point is also used by read-only databases.
pub(crate) fn scan_attached_envelope_page(
    path: &Path,
    cursor: JournalAnchor,
    row_limit: usize,
    payload_byte_limit: usize,
) -> Result<JournalEnvelopePage> {
    scan_attached_envelope_page_from(path, cursor, row_limit, payload_byte_limit, None)
        .map(|(page, _)| page)
}

/// Stateful form used by the writable journal coordinator. A matching resume
/// avoids rescanning an already validated prefix during sequential paging.
/// The scan validates through the first envelope after the page so `has_more`
/// never names an unvalidated record. Corruption farther into the suffix is
/// reported by the page that reaches it; open/state validation still scans the
/// complete file. Validating the entire suffix on every page would restore the
/// quadratic I/O this path is designed to avoid.
pub(crate) fn scan_attached_envelope_page_from(
    path: &Path,
    cursor: JournalAnchor,
    row_limit: usize,
    payload_byte_limit: usize,
    resume: Option<AttachedEnvelopeResume>,
) -> Result<(JournalEnvelopePage, AttachedEnvelopeResume)> {
    if row_limit == 0 {
        return Err(Error::InvalidJournalScanLimit {
            reason: "row limit must be non-zero",
        });
    }

    let mut records = StreamingRecords::open(path)?;
    let Some(checkpoint) = records.header.checkpoint_anchor else {
        // Preserve open-time validation for ordinary/uninitialized WALs. An
        // attached record without a durable genesis is rejected by observe.
        let mut chain = AttachedChain::new(None);
        while let Some(record) = records.next_record()? {
            if let WalOp::DbBatchWithEnvelope { envelope, .. } = record.op {
                chain.observe(records.header.version, &envelope, record.offset)?;
            }
        }
        return Err(Error::JournalStreamUnavailable {
            reason: "stream has not been initialized",
        });
    };
    if cursor.sequence() < checkpoint.sequence() {
        return Err(Error::JournalPositionExpired {
            requested: cursor.sequence(),
            checkpoint: checkpoint.sequence(),
        });
    }
    if cursor.sequence() == checkpoint.sequence() && cursor != checkpoint {
        return Err(Error::JournalAnchorMismatch {
            requested: cursor.sequence(),
            expected: checkpoint.sequence(),
        });
    }

    scan_initialized_envelope_page(
        records,
        checkpoint,
        cursor,
        row_limit,
        payload_byte_limit,
        resume,
    )
}

fn scan_initialized_envelope_page(
    mut records: StreamingRecords,
    checkpoint: JournalAnchor,
    cursor: JournalAnchor,
    row_limit: usize,
    payload_byte_limit: usize,
    resume: Option<AttachedEnvelopeResume>,
) -> Result<(JournalEnvelopePage, AttachedEnvelopeResume)> {
    let usable_resume = resume.filter(|value| {
        value.checkpoint == checkpoint
            && value.cursor == cursor
            && value.record_offset >= records.header.record_offset() as u64
            && value.record_offset <= records.file_len
    });
    let mut chain = if let Some(value) = usable_resume {
        records.seek_to(value.record_offset)?;
        AttachedChain::resume(checkpoint, cursor)
    } else {
        AttachedChain::new(Some(checkpoint))
    };
    let mut cursor_found = usable_resume.is_some() || cursor == checkpoint;
    let mut envelopes = Vec::new();
    let mut payload_bytes = 0usize;

    while let Some(record) = records.next_record()? {
        let record_offset = record.offset;
        let WalOp::DbBatchWithEnvelope { envelope, .. } = record.op else {
            continue;
        };
        let retained = chain.observe(records.header.version, &envelope, record_offset)?;
        if !retained {
            continue;
        }

        if !cursor_found {
            match envelope.current().sequence().cmp(&cursor.sequence()) {
                std::cmp::Ordering::Less => continue,
                std::cmp::Ordering::Equal => {
                    if envelope.current() != cursor {
                        return Err(Error::JournalAnchorMismatch {
                            requested: cursor.sequence(),
                            expected: envelope.current().sequence(),
                        });
                    }
                    cursor_found = true;
                    continue;
                }
                std::cmp::Ordering::Greater => {
                    return Err(Error::JournalAnchorMismatch {
                        requested: cursor.sequence(),
                        expected: envelope.current().sequence(),
                    });
                }
            }
        }

        let next_bytes = payload_bytes.saturating_add(envelope.payload().len());
        let page_full = envelopes.len() == row_limit
            || (!envelopes.is_empty() && next_bytes > payload_byte_limit);
        if page_full {
            let next = envelopes.last().map_or(cursor, JournalEnvelope::current);
            return Ok((
                JournalEnvelopePage::new(envelopes, next, true),
                AttachedEnvelopeResume {
                    checkpoint,
                    cursor: next,
                    record_offset,
                },
            ));
        }
        payload_bytes = next_bytes;
        envelopes.push(envelope);
    }

    if !cursor_found {
        return Err(Error::JournalAnchorMismatch {
            requested: cursor.sequence(),
            expected: chain.tail.expect("initialized chain has a tail").sequence(),
        });
    }
    let next = envelopes.last().map_or(cursor, JournalEnvelope::current);
    Ok((
        JournalEnvelopePage::new(envelopes, next, false),
        AttachedEnvelopeResume {
            checkpoint,
            cursor: next,
            record_offset: records.offset,
        },
    ))
}

/// Read and retain the suffix for focused legacy tests only. Production open,
/// state, and page paths use the bounded scanners above.
#[cfg(test)]
pub(crate) fn scan_attached_envelopes(path: &Path) -> Result<AttachedEnvelopeScan> {
    let mut records = StreamingRecords::open(path)?;
    let checkpoint = records.header.checkpoint_anchor;
    let mut chain = AttachedChain::new(checkpoint);
    let mut envelopes = Vec::new();
    while let Some(record) = records.next_record()? {
        if let WalOp::DbBatchWithEnvelope { envelope, .. } = record.op {
            if chain.observe(records.header.version, &envelope, record.offset)? {
                envelopes.push(envelope);
            }
        }
    }
    Ok(AttachedEnvelopeScan {
        checkpoint,
        tail: chain.tail,
        envelopes,
    })
}

struct AttachedChain {
    checkpoint: Option<JournalAnchor>,
    tail: Option<JournalAnchor>,
    previous_record: Option<JournalAnchor>,
}

impl AttachedChain {
    fn new(checkpoint: Option<JournalAnchor>) -> Self {
        Self {
            checkpoint,
            tail: checkpoint,
            previous_record: None,
        }
    }

    fn resume(checkpoint: JournalAnchor, cursor: JournalAnchor) -> Self {
        Self {
            checkpoint: Some(checkpoint),
            tail: Some(cursor),
            previous_record: Some(cursor),
        }
    }

    /// Validate one envelope and return whether it is retained after the
    /// checkpoint.
    fn observe(
        &mut self,
        format_version: u32,
        envelope: &JournalEnvelope,
        record_offset: u64,
    ) -> Result<bool> {
        if format_version == LEGACY_FORMAT_VERSION {
            return Err(Error::ReplaySanityFailed {
                context: "attached batch requires WAL format version 4",
                record_offset,
            });
        }
        let Some(checkpoint) = self.checkpoint else {
            return Err(Error::JournalStreamUnavailable {
                reason: "WAL contains attached envelopes but its stream anchor is uninitialized",
            });
        };
        if let Some(previous) = self.previous_record {
            if envelope.previous() != previous {
                return Err(Error::ReplaySanityFailed {
                    context: "attached journal records do not form a contiguous chain",
                    record_offset,
                });
            }
        }
        self.previous_record = Some(envelope.current());

        match envelope.current().sequence().cmp(&checkpoint.sequence()) {
            std::cmp::Ordering::Less => Ok(false),
            std::cmp::Ordering::Equal => {
                if envelope.current() != checkpoint {
                    return Err(Error::ReplaySanityFailed {
                        context: "attached journal record conflicts with checkpoint anchor",
                        record_offset,
                    });
                }
                Ok(false)
            }
            std::cmp::Ordering::Greater => {
                let expected = self.tail.expect("checkpoint exists");
                if envelope.previous() != expected {
                    return Err(Error::ReplaySanityFailed {
                        context: "attached journal suffix does not continue checkpoint anchor",
                        record_offset,
                    });
                }
                self.tail = Some(envelope.current());
                Ok(true)
            }
        }
    }

    fn state(&self) -> Option<JournalState> {
        Some(JournalState::new(self.checkpoint?, self.tail?))
    }
}

struct RecordAt {
    op: WalOp,
    offset: u64,
}

/// Forward record reader with memory bounded by one accepted WAL record.
struct StreamingRecords {
    reader: BufReader<File>,
    header: FileHeader,
    file_len: u64,
    offset: u64,
}

impl StreamingRecords {
    fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut prefix = [0u8; LEGACY_FILE_HEADER_SIZE];
        file.read_exact(&mut prefix).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::ReplaySanityFailed {
                    context: "WAL too short — missing file header",
                    record_offset: 0,
                }
            } else {
                Error::BlobStoreIo(error)
            }
        })?;
        let header_size = file_header_size_from_prefix(&prefix)?;
        let mut header_bytes = Vec::with_capacity(header_size);
        header_bytes.extend_from_slice(&prefix);
        if header_size > LEGACY_FILE_HEADER_SIZE {
            header_bytes.resize(header_size, 0);
            file.read_exact(&mut header_bytes[LEGACY_FILE_HEADER_SIZE..])
                .map_err(|error| {
                    if error.kind() == std::io::ErrorKind::UnexpectedEof {
                        Error::ReplaySanityFailed {
                            context: "WAL file header truncated",
                            record_offset: 0,
                        }
                    } else {
                        Error::BlobStoreIo(error)
                    }
                })?;
        }
        let header = decode_file_header(&header_bytes)?;
        Ok(Self {
            reader: BufReader::new(file),
            header,
            file_len,
            offset: header_size as u64,
        })
    }

    fn seek_to(&mut self, offset: u64) -> Result<()> {
        self.reader.seek(SeekFrom::Start(offset))?;
        self.offset = offset;
        Ok(())
    }

    fn next_record(&mut self) -> Result<Option<RecordAt>> {
        if self.offset >= self.file_len {
            return Ok(None);
        }
        let record_offset = self.offset;
        let remaining = self.file_len - record_offset;
        if remaining < RECORD_HEADER_SIZE as u64 {
            // A partial header at EOF is a tolerated torn tail.
            return Ok(None);
        }

        let mut header = [0u8; RECORD_HEADER_SIZE];
        self.reader.read_exact(&mut header)?;
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != RECORD_MAGIC {
            return Err(Error::ReplaySanityFailed {
                context: "record magic mismatch",
                record_offset,
            });
        }
        let body_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let total = RECORD_HEADER_SIZE
            .checked_add(body_len)
            .and_then(|value| value.checked_add(RECORD_FOOTER_SIZE))
            .ok_or(Error::ReplaySanityFailed {
                context: "record size overflow",
                record_offset,
            })?;
        if total as u64 > remaining {
            // A valid record header followed by an incomplete body/footer at
            // EOF has the same torn-tail semantics as replay_bytes.
            return Ok(None);
        }
        if total > MAX_JOURNAL_RECORD_BYTES {
            return Err(Error::ReplaySanityFailed {
                context: "record exceeds journal size limit",
                record_offset,
            });
        }

        let mut bytes = Vec::with_capacity(total);
        bytes.extend_from_slice(&header);
        bytes.resize(total, 0);
        self.reader.read_exact(&mut bytes[RECORD_HEADER_SIZE..])?;
        let decoded =
            decode_record(&bytes).map_err(|error| patch_offset(error, record_offset as usize))?;
        validate_record_mode(&self.header, &decoded.op, record_offset)?;
        #[cfg(test)]
        STREAMING_RECORDS_DECODED.with(|count| count.set(count.get() + 1));
        self.offset = self
            .offset
            .checked_add(decoded.bytes_consumed as u64)
            .expect("WAL offset fits in u64");
        Ok(Some(RecordAt {
            op: decoded.op,
            offset: record_offset,
        }))
    }
}

fn validate_record_mode(header: &FileHeader, op: &WalOp, record_offset: u64) -> Result<()> {
    if header.checkpoint_anchor.is_some() && !matches!(op, WalOp::DbBatchWithEnvelope { .. }) {
        return Err(Error::ReplaySanityFailed {
            context: "initialized attached WAL contains an ordinary record",
            record_offset,
        });
    }
    Ok(())
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

#[cfg(test)]
mod attached_streaming_tests {
    use std::fs;
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    use tempfile::tempdir;

    use super::*;
    use crate::journal::codec::{
        encode_file_header, BatchEncoder, FileHeader, FILE_HEADER_SIZE, LEGACY_FORMAT_VERSION,
    };
    use crate::journal::writer::WalWriter;

    const ORDINARY_RECORD_IN_ATTACHED_WAL: &str =
        "initialized attached WAL contains an ordinary record";

    fn anchor(sequence: u64) -> JournalAnchor {
        let mut digest = [0u8; 32];
        digest[..8].copy_from_slice(&sequence.to_le_bytes());
        JournalAnchor::new(sequence, digest)
    }

    fn write_envelopes(
        count: u64,
        payload_len: usize,
    ) -> (tempfile::TempDir, std::path::PathBuf, Vec<u64>) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let mut writer = WalWriter::create(&path, 0).unwrap();
        writer.persist_checkpoint_anchor(anchor(0)).unwrap();
        let mut offsets = Vec::new();
        let mut offset = FILE_HEADER_SIZE as u64;
        for sequence in 1..=count {
            let envelope = JournalEnvelope::new(
                anchor(sequence - 1),
                anchor(sequence),
                vec![sequence as u8; payload_len],
            )
            .unwrap();
            let mut record = Vec::new();
            BatchEncoder::begin_with_envelope(&mut record, sequence, 0, &envelope).finish();
            offsets.push(offset);
            offset += record.len() as u64;
            writer.append_encoded(&record).unwrap();
        }
        writer.flush().unwrap();
        (dir, path, offsets)
    }

    fn write_initialized_ordinary_record(op: &WalOp) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let mut writer = WalWriter::create(&path, 0).unwrap();
        writer.append(op, 1).unwrap();
        writer.flush().unwrap();
        writer.persist_checkpoint_anchor(anchor(0)).unwrap();
        drop(writer);
        (dir, path)
    }

    fn assert_ordinary_record_rejected<T>(result: Result<T>, record_offset: u64) {
        match result {
            Err(Error::ReplaySanityFailed {
                context: ORDINARY_RECORD_IN_ATTACHED_WAL,
                record_offset: actual,
            }) if actual == record_offset => {}
            Err(error) => panic!("expected ordinary-record rejection, got {error:?}"),
            Ok(_) => panic!("expected ordinary-record rejection"),
        }
    }

    #[test]
    fn initialized_wal_rejects_insert_before_each_reader_accepts_it() {
        let insert = WalOp::Insert {
            tree_id: 0,
            key: b"key".to_vec(),
            value: b"value".to_vec(),
        };
        let (_dir, path) = write_initialized_ordinary_record(&insert);
        let bytes = fs::read(&path).unwrap();
        let record_offset = FILE_HEADER_SIZE as u64;

        let raw_callbacks = std::cell::Cell::new(0usize);
        let mut raw_callback = |_: &WalOp, _: u64, _: u64| {
            raw_callbacks.set(raw_callbacks.get() + 1);
            Ok(())
        };
        assert_ordinary_record_rejected(replay_bytes(&bytes, &mut raw_callback), record_offset);
        assert_eq!(raw_callbacks.get(), 0);

        let file_callbacks = std::cell::Cell::new(0usize);
        assert_ordinary_record_rejected(
            replay(&path, |_, _, _| {
                file_callbacks.set(file_callbacks.get() + 1);
                Ok(())
            }),
            record_offset,
        );
        assert_eq!(file_callbacks.get(), 0);

        assert_ordinary_record_rejected(validate_attached_journal(&path), record_offset);
        assert_ordinary_record_rejected(
            scan_attached_envelope_page(&path, anchor(0), 1, usize::MAX),
            record_offset,
        );
        assert_ordinary_record_rejected(
            scan_attached_envelope_page_from(&path, anchor(0), 1, usize::MAX, None),
            record_offset,
        );
        assert_ordinary_record_rejected(scan_attached_envelopes(&path), record_offset);
        assert_ordinary_record_rejected(preflight_writable_wal(&path), record_offset);
    }

    #[test]
    fn initialized_wal_rejects_an_ordinary_batch_before_flattening() {
        let batch = WalOp::Batch {
            ops: vec![WalOp::Insert {
                tree_id: 0,
                key: b"key".to_vec(),
                value: b"value".to_vec(),
            }],
        };
        let (_dir, path) = write_initialized_ordinary_record(&batch);
        let bytes = fs::read(&path).unwrap();
        let callbacks = std::cell::Cell::new(0usize);
        let mut callback = |_: &WalOp, _: u64, _: u64| {
            callbacks.set(callbacks.get() + 1);
            Ok(())
        };

        assert_ordinary_record_rejected(
            replay_bytes(&bytes, &mut callback),
            FILE_HEADER_SIZE as u64,
        );
        assert_eq!(callbacks.get(), 0);
        assert_ordinary_record_rejected(preflight_writable_wal(&path), FILE_HEADER_SIZE as u64);
    }

    #[test]
    fn uninitialized_wal_keeps_ordinary_records_valid() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let mut writer = WalWriter::create(&path, 0).unwrap();
        writer
            .append(
                &WalOp::Insert {
                    tree_id: 0,
                    key: b"first".to_vec(),
                    value: b"value".to_vec(),
                },
                1,
            )
            .unwrap();
        writer
            .append(
                &WalOp::Batch {
                    ops: vec![WalOp::Insert {
                        tree_id: 0,
                        key: b"second".to_vec(),
                        value: b"value".to_vec(),
                    }],
                },
                2,
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        preflight_writable_wal(&path).unwrap();
        let mut callbacks = 0usize;
        let (_, stats) = replay(&path, |_, _, _| {
            callbacks += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(callbacks, 2);
        assert_eq!(stats.records_seen, 2);
        assert_eq!(validate_attached_journal(&path).unwrap(), None);
        assert!(matches!(
            scan_attached_envelope_page(&path, anchor(0), 1, usize::MAX),
            Err(Error::JournalStreamUnavailable {
                reason: "stream has not been initialized",
            })
        ));
        let scan = scan_attached_envelopes(&path).unwrap();
        assert_eq!(scan.checkpoint, None);
        assert_eq!(scan.tail, None);
        assert!(scan.envelopes.is_empty());
    }

    #[test]
    fn initialized_wal_keeps_attached_records_valid() {
        let (_dir, path, _offsets) = write_envelopes(1, 16);

        preflight_writable_wal(&path).unwrap();
        let state = validate_attached_journal(&path).unwrap().unwrap();
        assert_eq!(state.checkpoint(), anchor(0));
        assert_eq!(state.tail(), anchor(1));

        let mut callbacks = 0usize;
        let (_, stats) = replay(&path, |_, _, _| {
            callbacks += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(callbacks, 0);
        assert_eq!(stats.records_seen, 1);

        let page = scan_attached_envelope_page(&path, anchor(0), 1, usize::MAX).unwrap();
        assert_eq!(page.envelopes().len(), 1);
        assert_eq!(page.next(), anchor(1));
        assert!(!page.has_more());

        let scan = scan_attached_envelopes(&path).unwrap();
        assert_eq!(scan.checkpoint, Some(anchor(0)));
        assert_eq!(scan.tail, Some(anchor(1)));
        assert_eq!(scan.envelopes.len(), 1);
    }

    #[test]
    fn resumed_pages_decode_only_one_lookahead_per_page() {
        const RECORDS: u64 = 96;
        const PAYLOAD_BYTES: usize = 64 * 1024;
        let (_dir, path, _offsets) = write_envelopes(RECORDS, PAYLOAD_BYTES);
        STREAMING_RECORDS_DECODED.with(|count| count.set(0));

        let mut cursor = anchor(0);
        let mut resume = None;
        let mut seen = 0u64;
        let mut pages = 0usize;
        loop {
            let (page, next_resume) =
                scan_attached_envelope_page_from(&path, cursor, 16, PAYLOAD_BYTES * 2, resume)
                    .unwrap();
            assert!(!page.envelopes().is_empty());
            assert!(page.envelopes().len() <= 2);
            seen += page.envelopes().len() as u64;
            pages += 1;
            cursor = page.next();
            resume = Some(next_resume);
            if !page.has_more() {
                break;
            }
        }

        assert_eq!(seen, RECORDS);
        assert_eq!(cursor, anchor(RECORDS));
        assert_eq!(pages, RECORDS as usize / 2);
        let decoded = STREAMING_RECORDS_DECODED.with(std::cell::Cell::get);
        assert_eq!(decoded, RECORDS as usize + pages - 1);
    }

    #[test]
    fn page_validates_one_lookahead_and_defers_later_corruption() {
        let (_dir, path, offsets) = write_envelopes(3, 32);
        let corrupt_at = offsets[2] + RECORD_HEADER_SIZE as u64 + 1;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(corrupt_at)).unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= 0x80;
        file.seek(SeekFrom::Start(corrupt_at)).unwrap();
        file.write_all(&byte).unwrap();
        file.sync_data().unwrap();

        let (first, resume) =
            scan_attached_envelope_page_from(&path, anchor(0), 1, usize::MAX, None).unwrap();
        assert_eq!(first.envelopes()[0].current(), anchor(1));
        assert!(first.has_more());

        assert!(matches!(
            scan_attached_envelope_page_from(
                &path,
                first.next(),
                1,
                usize::MAX,
                Some(resume),
            ),
            Err(Error::ReplaySanityFailed {
                context: "record CRC mismatch",
                record_offset,
            }) if record_offset == offsets[2]
        ));
        assert!(matches!(
            validate_attached_journal(&path),
            Err(Error::ReplaySanityFailed {
                context: "record CRC mismatch",
                record_offset,
            }) if record_offset == offsets[2]
        ));
    }

    #[test]
    fn writable_preflight_accepts_header_only_v3_but_rejects_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy.wal");
        let mut bytes = Vec::new();
        encode_file_header(
            &FileHeader {
                version: LEGACY_FORMAT_VERSION,
                tree_id: 0,
                created_at: 0,
                checkpoint_anchor: None,
                anchor_generation: 0,
            },
            &mut bytes,
        );
        fs::write(&path, &bytes).unwrap();
        preflight_writable_wal(&path).unwrap();

        bytes.push(0);
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            preflight_writable_wal(&path),
            Err(Error::ReplaySanityFailed {
                context: "nonempty WAL format 3 is replay-only; checkpoint it with a format-3 Holt binary before v4 writes",
                record_offset: 0,
            })
        ));
    }
}
