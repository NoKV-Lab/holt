//! Application-owned recovery material attached to one atomic WAL batch.
//!
//! Holt treats the payload as opaque bytes. The application owns its schema,
//! canonical encoding, and digest calculation; Holt only keeps the envelope in
//! the same checksummed WAL record as the batch that mutates the database.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use super::errors::{Error, Result};

/// Byte width of a journal anchor digest.
pub const JOURNAL_DIGEST_BYTES: usize = 32;

/// Maximum encoded size of one WAL record accepted by Holt.
///
/// Applications that attach canonical recovery material must keep the full
/// envelope plus physical DB batch within this bound.
pub const MAX_JOURNAL_RECORD_BYTES: usize = 16 * 1024 * 1024;

/// One position in an application's hash-chained recovery stream.
///
/// The digest also binds the stream identity: applications should derive their
/// genesis digest from the logical stream identity before constructing the
/// first attached envelope.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct JournalAnchor {
    sequence: u64,
    digest: [u8; JOURNAL_DIGEST_BYTES],
}

impl JournalAnchor {
    /// Construct an anchor from an application sequence and digest.
    #[must_use]
    pub const fn new(sequence: u64, digest: [u8; JOURNAL_DIGEST_BYTES]) -> Self {
        Self { sequence, digest }
    }

    /// Return the application sequence represented by this anchor.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    /// Return the application-provided digest represented by this anchor.
    #[must_use]
    pub const fn digest(self) -> [u8; JOURNAL_DIGEST_BYTES] {
        self.digest
    }
}

/// Validation failure while constructing an attached journal envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum JournalEnvelopeError {
    /// The current anchor is not the immediate successor of the previous one.
    NonContiguousSequence {
        /// Sequence in the previous anchor.
        previous: u64,
        /// Sequence in the current anchor.
        current: u64,
    },
    /// An attached envelope must carry canonical recovery bytes.
    EmptyPayload,
    /// The WAL envelope frame uses a 32-bit payload length.
    PayloadTooLarge {
        /// Caller-supplied payload length.
        len: usize,
    },
}

impl fmt::Display for JournalEnvelopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonContiguousSequence { previous, current } => write!(
                formatter,
                "journal sequence {current} is not the immediate successor of {previous}"
            ),
            Self::EmptyPayload => formatter.write_str("journal envelope payload must not be empty"),
            Self::PayloadTooLarge { len } => {
                write!(
                    formatter,
                    "journal envelope payload is too large ({len} bytes)"
                )
            }
        }
    }
}

impl std::error::Error for JournalEnvelopeError {}

/// Opaque application recovery bytes anchored before and after one DB batch.
///
/// The current sequence must equal `previous.sequence() + 1`, and `payload`
/// must not be empty. Holt does not recompute either digest; the enclosing WAL
/// record CRC protects both anchors, the payload, and the database operations
/// against torn writes and accidental corruption.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JournalEnvelope {
    previous: JournalAnchor,
    current: JournalAnchor,
    payload: Vec<u8>,
}

/// Checkpointed and live positions of one attached recovery stream.
///
/// `checkpoint` is the oldest position retained by the local stream.
/// Envelopes at or before it have already been folded into the checkpointed
/// database image. `tail` includes every attached envelope accepted after
/// that checkpoint. File-backed images are durable; memory images are not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JournalState {
    checkpoint: JournalAnchor,
    tail: JournalAnchor,
}

impl JournalState {
    pub(crate) const fn new(checkpoint: JournalAnchor, tail: JournalAnchor) -> Self {
        Self { checkpoint, tail }
    }

    /// Return the oldest stream position retained by the local WAL.
    #[must_use]
    pub const fn checkpoint(self) -> JournalAnchor {
        self.checkpoint
    }

    /// Return the latest attached stream position accepted by this database.
    #[must_use]
    pub const fn tail(self) -> JournalAnchor {
        self.tail
    }
}

/// One bounded page of attached recovery envelopes.
///
/// `next` is the cursor for the next call. It equals the input cursor when
/// the page is empty. `has_more` reports whether another retained envelope
/// follows `next`. The payload-byte limit is a soft page boundary: when at
/// least one row is requested, an oversized first envelope is returned by
/// itself so a caller can always advance its cursor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JournalEnvelopePage {
    envelopes: Vec<JournalEnvelope>,
    next: JournalAnchor,
    has_more: bool,
}

impl JournalEnvelopePage {
    pub(crate) fn new(
        envelopes: Vec<JournalEnvelope>,
        next: JournalAnchor,
        has_more: bool,
    ) -> Self {
        Self {
            envelopes,
            next,
            has_more,
        }
    }

    /// Return the retained envelopes after the requested cursor.
    #[must_use]
    pub fn envelopes(&self) -> &[JournalEnvelope] {
        &self.envelopes
    }

    /// Consume the page and return its retained envelopes.
    #[must_use]
    pub fn into_envelopes(self) -> Vec<JournalEnvelope> {
        self.envelopes
    }

    /// Return the cursor to pass to the next scan call.
    #[must_use]
    pub const fn next(&self) -> JournalAnchor {
        self.next
    }

    /// Return whether another retained envelope follows this page.
    #[must_use]
    pub const fn has_more(&self) -> bool {
        self.has_more
    }
}

pub(crate) fn bounded_envelope_page(
    retained: &[JournalEnvelope],
    start: usize,
    cursor: JournalAnchor,
    row_limit: usize,
    payload_byte_limit: usize,
) -> JournalEnvelopePage {
    debug_assert!(row_limit > 0);
    let mut envelopes = Vec::new();
    let mut payload_bytes = 0usize;
    for envelope in &retained[start..] {
        if envelopes.len() == row_limit {
            break;
        }
        let next_bytes = payload_bytes.saturating_add(envelope.payload().len());
        if !envelopes.is_empty() && next_bytes > payload_byte_limit {
            break;
        }
        payload_bytes = next_bytes;
        envelopes.push(envelope.clone());
    }
    let consumed = envelopes.len();
    let next = envelopes.last().map_or(cursor, JournalEnvelope::current);
    let has_more = start + consumed < retained.len();
    JournalEnvelopePage::new(envelopes, next, has_more)
}

impl JournalEnvelope {
    /// Construct a validated owned envelope.
    pub fn new(
        previous: JournalAnchor,
        current: JournalAnchor,
        payload: Vec<u8>,
    ) -> Result<Self, JournalEnvelopeError> {
        if previous.sequence.checked_add(1) != Some(current.sequence) {
            return Err(JournalEnvelopeError::NonContiguousSequence {
                previous: previous.sequence,
                current: current.sequence,
            });
        }
        if payload.is_empty() {
            return Err(JournalEnvelopeError::EmptyPayload);
        }
        if u32::try_from(payload.len()).is_err() {
            return Err(JournalEnvelopeError::PayloadTooLarge { len: payload.len() });
        }
        Ok(Self {
            previous,
            current,
            payload,
        })
    }

    /// Return the durable anchor that must precede this envelope.
    #[must_use]
    pub const fn previous(&self) -> JournalAnchor {
        self.previous
    }

    /// Return the anchor produced by this envelope.
    #[must_use]
    pub const fn current(&self) -> JournalAnchor {
        self.current
    }

    /// Return the opaque canonical recovery bytes.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Consume the envelope and return its opaque canonical recovery bytes.
    #[must_use]
    pub fn into_payload(self) -> Vec<u8> {
        self.payload
    }
}

/// Process-local attached stream used by [`crate::DB`] in memory mode.
///
/// This profile preserves the same ordering, fencing, paging, and checkpoint
/// semantics as the file-backed stream. It deliberately provides no reopen or
/// crash durability: dropping the memory DB drops both its data and envelopes.
pub(crate) struct VolatileJournal {
    ordinary_fenced: AtomicBool,
    state: Mutex<VolatileJournalState>,
}

#[derive(Default)]
struct VolatileJournalState {
    checkpoint: Option<JournalAnchor>,
    tail: Option<JournalAnchor>,
    envelopes: Vec<JournalEnvelope>,
}

pub(crate) struct VolatileJournalAppendGuard<'a> {
    state: MutexGuard<'a, VolatileJournalState>,
}

impl VolatileJournal {
    pub(crate) fn new() -> Self {
        Self {
            ordinary_fenced: AtomicBool::new(false),
            state: Mutex::new(VolatileJournalState::default()),
        }
    }

    pub(crate) fn initialize(&self, genesis: JournalAnchor) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        match state.checkpoint {
            Some(existing) if existing == genesis => Ok(()),
            Some(existing) => Err(Error::JournalAnchorMismatch {
                requested: genesis.sequence(),
                expected: existing.sequence(),
            }),
            None => {
                self.ordinary_fenced.store(true, Ordering::Release);
                state.checkpoint = Some(genesis);
                state.tail = Some(genesis);
                Ok(())
            }
        }
    }

    pub(crate) fn state(&self) -> Option<JournalState> {
        let state = self.state.lock().unwrap();
        Some(JournalState::new(state.checkpoint?, state.tail?))
    }

    pub(crate) fn ensure_ordinary_writes_allowed(&self) -> Result<()> {
        if self.ordinary_fenced.load(Ordering::Acquire) {
            Err(Error::JournalStreamUnavailable {
                reason: "ordinary writes are disabled after attached stream initialization",
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn begin_attached(
        &self,
        expected_previous: JournalAnchor,
        record_len: usize,
    ) -> Result<VolatileJournalAppendGuard<'_>> {
        if record_len == 0 {
            return Err(Error::Internal("journal record must not be empty"));
        }
        if record_len > MAX_JOURNAL_RECORD_BYTES {
            return Err(Error::WalRecordTooLarge {
                bytes: record_len,
                maximum: MAX_JOURNAL_RECORD_BYTES,
            });
        }
        let state = self.state.lock().unwrap();
        let expected = state.tail.ok_or(Error::JournalStreamUnavailable {
            reason: "stream has not been initialized",
        })?;
        if expected_previous != expected {
            return Err(Error::JournalAnchorMismatch {
                requested: expected_previous.sequence(),
                expected: expected.sequence(),
            });
        }
        Ok(VolatileJournalAppendGuard { state })
    }

    pub(crate) fn checkpoint(&self) {
        let mut state = self.state.lock().unwrap();
        if let Some(tail) = state.tail {
            state.checkpoint = Some(tail);
            state.envelopes.clear();
        }
    }

    pub(crate) fn envelopes_after(
        &self,
        cursor: JournalAnchor,
        row_limit: usize,
        payload_byte_limit: usize,
    ) -> Result<JournalEnvelopePage> {
        if row_limit == 0 {
            return Err(Error::InvalidJournalScanLimit {
                reason: "row limit must be non-zero",
            });
        }
        let state = self.state.lock().unwrap();
        let checkpoint = state.checkpoint.ok_or(Error::JournalStreamUnavailable {
            reason: "stream has not been initialized",
        })?;
        if cursor.sequence() < checkpoint.sequence() {
            return Err(Error::JournalPositionExpired {
                requested: cursor.sequence(),
                checkpoint: checkpoint.sequence(),
            });
        }
        let start = if cursor == checkpoint {
            0
        } else if cursor.sequence() == checkpoint.sequence() {
            return Err(Error::JournalAnchorMismatch {
                requested: cursor.sequence(),
                expected: cursor.sequence(),
            });
        } else if let Some(index) = state
            .envelopes
            .iter()
            .position(|envelope| envelope.current().sequence() == cursor.sequence())
        {
            if state.envelopes[index].current() != cursor {
                return Err(Error::JournalAnchorMismatch {
                    requested: cursor.sequence(),
                    expected: cursor.sequence(),
                });
            }
            index + 1
        } else {
            return Err(Error::JournalAnchorMismatch {
                requested: cursor.sequence(),
                expected: state.tail.unwrap_or(checkpoint).sequence(),
            });
        };
        Ok(bounded_envelope_page(
            &state.envelopes,
            start,
            cursor,
            row_limit,
            payload_byte_limit,
        ))
    }
}

impl VolatileJournalAppendGuard<'_> {
    pub(crate) fn submit(mut self, envelope: JournalEnvelope) -> Result<()> {
        let expected = self
            .state
            .tail
            .expect("begin_attached initialized volatile tail");
        if envelope.previous() != expected {
            return Err(Error::JournalAnchorMismatch {
                requested: envelope.previous().sequence(),
                expected: expected.sequence(),
            });
        }
        self.state.tail = Some(envelope.current());
        self.state.envelopes.push(envelope);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_requires_one_sequence_step_and_payload() {
        let previous = JournalAnchor::new(7, [0x11; JOURNAL_DIGEST_BYTES]);
        let current = JournalAnchor::new(8, [0x22; JOURNAL_DIGEST_BYTES]);
        let envelope = JournalEnvelope::new(previous, current, b"canonical".to_vec()).unwrap();
        assert_eq!(envelope.previous(), previous);
        assert_eq!(envelope.current(), current);
        assert_eq!(envelope.payload(), b"canonical");

        assert!(matches!(
            JournalEnvelope::new(previous, JournalAnchor::new(9, [0x33; 32]), vec![1]),
            Err(JournalEnvelopeError::NonContiguousSequence { .. })
        ));
        assert_eq!(
            JournalEnvelope::new(previous, current, Vec::new()),
            Err(JournalEnvelopeError::EmptyPayload)
        );
    }

    #[test]
    fn envelope_rejects_sequence_overflow() {
        let previous = JournalAnchor::new(u64::MAX, [0x11; JOURNAL_DIGEST_BYTES]);
        let current = JournalAnchor::new(0, [0x22; JOURNAL_DIGEST_BYTES]);
        assert!(matches!(
            JournalEnvelope::new(previous, current, vec![1]),
            Err(JournalEnvelopeError::NonContiguousSequence { .. })
        ));
    }
}
