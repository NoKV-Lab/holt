//! WAL append coordinator — lock-free shared ring + single flusher.
//!
//! Concurrent writers reserve a byte range in the lock-free [`WalRing`]
//! (one `tail.fetch_add`), memcpy their encoded record in parallel, and
//! publish by folding the contiguous published prefix; a single background
//! **flusher** drains that prefix into the [`WalWriter`] (unchanged on-disk
//! format + replay) and fsyncs on the sync path. This replaced an earlier
//! per-record `Vec` + single crossbeam channel + single batching worker,
//! which serialized concurrent durable writes (see
//! `PERF_FINDINGS.md`: ~5–6× faster concurrent durable write, beating
//! RocksDB 2.8–5.5× at 1/4/8/16 threads).
//!
//! ## Watermarks live in the record-count domain
//!
//! `queued`/`written`/`flushed`/`checkpointed` count RECORDS (dense,
//! monotone), exactly mirroring the legacy work-id watermarks — so
//! `needs_checkpoint`, the reopen signal, and the checkpoint round-trip are
//! preserved bit-for-bit. `record_base` is the reopen offset (1 when the
//! file already had records, else 0). `written/flushed = record_base +
//! ring.committed_records() + oversized_records` at drain/fsync time;
//! `queued = record_base + records submitted this process`. They reconcile
//! because each admitted record is published through exactly one ordered lane.
//!
//! ## Durability: the flusher drains PROMPTLY
//!
//! Async (`wal_sync=false`) records must reach the OS page cache promptly so
//! they survive a *process* crash (as the legacy worker guarantees). The
//! flusher polls every `FLUSH_POLL` and on every wake drains the committed
//! prefix into `WalWriter` (whose 64 KB auto-drain reaches the page cache).
//! fsync happens only when a sync target is outstanding (sync write or
//! checkpoint barrier), exactly like legacy group commit.

use std::fs::File;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::api::errors::{CommitPhase, Error, Result};
use crate::store::blob_store::file::{FileBlobStore, FileStoreHealth, FileStoreHealthPermit};

use super::codec::MAX_ATOMIC_WAL_RECORD_BYTES;
use super::ring::{CommittedPrefix, ReserveTicket, WalRing};
use super::writer::WalWriter;

/// Production journal counters surfaced through `Tree::stats`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JournalStats {
    pub(crate) appends: u64,
    pub(crate) batches: u64,
    pub(crate) syncs: u64,
    pub(crate) queued_work: u64,
    pub(crate) written_work: u64,
    pub(crate) flushed_work: u64,
    pub(crate) checkpointed_work: u64,
    pub(crate) pending_work: u64,
    pub(crate) checkpoint_debt: u64,
}

/// In-RAM ring capacity. Records larger than this use the bounded oversized
/// lane instead of permanently multiplying every store's resident WAL RAM.
const RING_CAPACITY_BYTES: usize = 16 * 1024 * 1024;
/// Flusher idle poll. Bounds the async RAM→page-cache window (process-crash
/// durability) and the latency a sync waiter adds if a wake is ever missed.
const FLUSH_POLL: Duration = Duration::from_micros(50);
const ORDERED_CONTROL_CAPACITY: usize = 1;
const WAKE_CAPACITY: usize = 1;
const RECORD_BUFFER_POOL_LIMIT: usize = 1024;
const RECORD_BUFFER_RETAIN_MAX: usize = 64 * 1024;

/// Strictly ordered control messages to the flusher.
///
/// Prompt drain wakes use a separate coalesced one-token channel; dropping a
/// redundant wake can never drop, reorder, or make one of these commands
/// unreachable.
enum Control {
    /// Drain, then truncate the WAL to its header and reset the ring.
    Truncate(Sender<Result<()>>),
    /// Drain the preceding ring prefix, append one oversized record directly,
    /// and acknowledge only after its bytes have reached the OS page cache.
    AppendOversized {
        record: Vec<u8>,
        target: u64,
        ack: Sender<std::result::Result<(), JournalFailure>>,
    },
    Stop,
}

#[derive(Debug, Clone, Copy)]
struct JournalFailure {
    phase: CommitPhase,
    context: &'static str,
}

impl JournalFailure {
    const fn unknown(self) -> Error {
        Error::CommitOutcomeUnknown {
            phase: self.phase,
            context: self.context,
        }
    }
}

const RECORD_LANE_WRITE_BIT: usize = 1usize << (usize::BITS - 1);
const RECORD_LANE_COUNT_MASK: usize = RECORD_LANE_WRITE_BIT - 1;

/// Writer-preferred admission gate joining the ring and oversized lanes.
/// Small records hold a shared permit only through publication; an oversized
/// waiter blocks new small admissions, drains existing publishers, then owns
/// the direct-append position exclusively.
#[derive(Debug)]
struct RecordLane {
    /// High bit = oversized pending/active; low bits = active small permits.
    /// Uncontended small admission is one lock-free CAS.
    state: AtomicUsize,
    waiting_oversized: AtomicUsize,
    /// Serializes only oversized writers. Small records never take this lock.
    oversized_serial: Mutex<()>,
    wait_mx: Mutex<()>,
    cv: Condvar,
}

impl Default for RecordLane {
    fn default() -> Self {
        Self {
            state: AtomicUsize::new(0),
            waiting_oversized: AtomicUsize::new(0),
            oversized_serial: Mutex::new(()),
            wait_mx: Mutex::new(()),
            cv: Condvar::new(),
        }
    }
}

impl RecordLane {
    fn enter_small(&self) -> RecordLanePermit<'_> {
        loop {
            if self.waiting_oversized.load(Ordering::Acquire) == 0 {
                let state = self.state.load(Ordering::Relaxed);
                if state & RECORD_LANE_WRITE_BIT == 0
                    && state & RECORD_LANE_COUNT_MASK != RECORD_LANE_COUNT_MASK
                    && self
                        .state
                        .compare_exchange_weak(
                            state,
                            state + 1,
                            Ordering::Acquire,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    return RecordLanePermit {
                        lane: self,
                        kind: RecordLaneKind::Small,
                        oversized_serial: None,
                    };
                }
            }

            let mut guard = self.wait_mx.lock().unwrap();
            while self.state.load(Ordering::Acquire) & RECORD_LANE_WRITE_BIT != 0
                || self.waiting_oversized.load(Ordering::Acquire) != 0
            {
                guard = self.cv.wait(guard).unwrap();
            }
        }
    }

    fn enter_oversized(&self) -> RecordLanePermit<'_> {
        self.waiting_oversized.fetch_add(1, Ordering::AcqRel);
        let oversized_serial = self.oversized_serial.lock().unwrap();

        // Only the oversized-serial owner changes the write bit. Preserve the
        // existing small count and block every later small CAS before waiting
        // for that finite predecessor set to drain.
        loop {
            let state = self.state.load(Ordering::Relaxed);
            debug_assert_eq!(state & RECORD_LANE_WRITE_BIT, 0);
            if self
                .state
                .compare_exchange_weak(
                    state,
                    state | RECORD_LANE_WRITE_BIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
        self.waiting_oversized.fetch_sub(1, Ordering::AcqRel);

        let mut guard = self.wait_mx.lock().unwrap();
        while self.state.load(Ordering::Acquire) & RECORD_LANE_COUNT_MASK != 0 {
            guard = self.cv.wait(guard).unwrap();
        }
        drop(guard);
        RecordLanePermit {
            lane: self,
            kind: RecordLaneKind::Oversized,
            oversized_serial: Some(oversized_serial),
        }
    }

    #[cfg(test)]
    fn oversized_waiting(&self) -> bool {
        self.waiting_oversized.load(Ordering::Acquire) != 0
            || self.state.load(Ordering::Acquire) & RECORD_LANE_WRITE_BIT != 0
    }

    fn notify_waiters(&self) {
        let _guard = self.wait_mx.lock().unwrap();
        self.cv.notify_all();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordLaneKind {
    Small,
    Oversized,
}

#[derive(Debug)]
struct RecordLanePermit<'a> {
    lane: &'a RecordLane,
    kind: RecordLaneKind,
    oversized_serial: Option<std::sync::MutexGuard<'a, ()>>,
}

impl Drop for RecordLanePermit<'_> {
    fn drop(&mut self) {
        match self.kind {
            RecordLaneKind::Small => {
                let previous = self.lane.state.fetch_sub(1, Ordering::Release);
                debug_assert_ne!(previous & RECORD_LANE_COUNT_MASK, 0);
                if previous & RECORD_LANE_WRITE_BIT != 0 && previous & RECORD_LANE_COUNT_MASK == 1 {
                    self.lane.notify_waiters();
                }
            }
            RecordLaneKind::Oversized => {
                debug_assert_eq!(
                    self.lane.state.load(Ordering::Relaxed),
                    RECORD_LANE_WRITE_BIT
                );
                self.lane.state.store(0, Ordering::Release);
                drop(self.oversized_serial.take());
                self.lane.notify_waiters();
            }
        }
    }
}

struct Shared {
    ring: WalRing,
    record_lane: RecordLane,
    writer: Mutex<WalWriter>,
    /// Reopen offset: records already on disk before this process's ring.
    record_base: u64,

    queued: AtomicU64,
    written: AtomicU64,
    flushed: AtomicU64,
    checkpointed: AtomicU64,
    /// Highest record count some waiter needs fsync-durable.
    sync_target: AtomicU64,
    /// Oversized records appended outside the ring. Added to the ring's
    /// committed-record count so every watermark remains in one record domain.
    oversized_records: AtomicU64,
    /// Production is fixed at [`MAX_ATOMIC_WAL_RECORD_BYTES`]. Atomic only so
    /// unit tests can exercise admission without allocating hundreds of MiB.
    max_record_bytes: AtomicUsize,

    appends: AtomicU64,
    batches: AtomicU64,
    syncs: AtomicU64,

    /// Shared fail-stop state for descriptor-backed file stores. Path-backed
    /// Journal unit tests have no owning FileBlobStore and therefore no gate.
    file_store_health: Option<Arc<FileStoreHealth>>,

    /// Sticky flusher phase/error; fanned out to commit waiters and future
    /// checkpoint barriers. Static context keeps the failure copyable without
    /// erasing whether append or sync became uncertain.
    err: Mutex<Option<JournalFailure>>,
    /// Condvar handshake for `flushed`/`err` waiters.
    flushed_mx: Mutex<()>,
    flushed_cv: Condvar,
    /// Condvar handshake for writers parked on ring backpressure.
    space_mx: Mutex<()>,
    space_cv: Condvar,
    /// Condvar handshake for an out-of-order publisher waiting until its byte
    /// ticket joins the contiguous physical prefix.
    prefix_mx: Mutex<()>,
    prefix_cv: Condvar,
    prefix_waiters: AtomicUsize,

    control_tx: Sender<Control>,
    wake_tx: Sender<()>,
    record_pool: Mutex<Vec<Vec<u8>>>,
    #[cfg(test)]
    shutdown_barrier: Mutex<Option<Arc<JournalShutdownBarrier>>>,
    #[cfg(test)]
    submit_pauses: Mutex<Vec<SubmitPause>>,
    #[cfg(test)]
    before_drain_barrier: Mutex<Option<Arc<JournalTestBarrier>>>,
    #[cfg(test)]
    after_snapshot_barrier: Mutex<Option<Arc<JournalTestBarrier>>>,
    #[cfg(test)]
    before_sync_barrier: Mutex<Option<Arc<JournalTestBarrier>>>,
}

/// Buffer admitted before a tree mutation begins.
///
/// `admitted_upper_bound` may be conservative for a DB batch, but submit
/// rejects any encoder drift where the actual bytes exceed it. The lane permit
/// also prevents a large record from being overtaken by later ring records.
pub(crate) struct PreparedRecord<'a> {
    shared: &'a Shared,
    bytes: Option<Vec<u8>>,
    admitted_upper_bound: usize,
    permit: RecordLanePermit<'a>,
    /// Descriptor-backed journals keep the shared health side from admission
    /// through small-record publication. Oversized submit transfers the health
    /// boundary to the flusher. This field follows the lane permit so drop
    /// releases ordering before allowing poison to proceed.
    file_store_health: Option<FileStoreHealthPermit<'a>>,
}

impl PreparedRecord<'_> {
    fn take_bytes(&mut self) -> Vec<u8> {
        self.bytes.take().expect("prepared record bytes taken once")
    }

    fn lane_kind(&self) -> RecordLaneKind {
        self.permit.kind
    }
}

impl Deref for PreparedRecord<'_> {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        self.bytes
            .as_ref()
            .expect("prepared record bytes available")
    }
}

impl DerefMut for PreparedRecord<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.bytes
            .as_mut()
            .expect("prepared record bytes available")
    }
}

impl Drop for PreparedRecord<'_> {
    fn drop(&mut self) {
        if self.permit.kind == RecordLaneKind::Small {
            if let Some(bytes) = self.bytes.take() {
                self.shared.recycle(bytes);
            }
        }
    }
}

/// Deterministic worker-shutdown probe for lifecycle regression tests.
///
/// The first gate parks the worker immediately before its next drain. The
/// second parks it after the final shutdown drain but before the worker-held
/// resource guard is released. Channels provide bounded waits to the test
/// thread, unlike a process-wide sleep or timing-only assertion.
#[cfg(test)]
pub(crate) struct JournalShutdownBarrier {
    before_drain_entered_tx: Sender<()>,
    before_drain_entered_rx: Receiver<()>,
    before_drain_release_tx: Sender<()>,
    before_drain_release_rx: Receiver<()>,
    before_exit_entered_tx: Sender<()>,
    before_exit_entered_rx: Receiver<()>,
    before_exit_release_tx: Sender<()>,
    before_exit_release_rx: Receiver<()>,
}

#[cfg(test)]
impl JournalShutdownBarrier {
    pub(crate) fn new() -> Self {
        let (before_drain_entered_tx, before_drain_entered_rx) = crossbeam_channel::bounded(1);
        let (before_drain_release_tx, before_drain_release_rx) = crossbeam_channel::bounded(1);
        let (before_exit_entered_tx, before_exit_entered_rx) = crossbeam_channel::bounded(1);
        let (before_exit_release_tx, before_exit_release_rx) = crossbeam_channel::bounded(1);
        Self {
            before_drain_entered_tx,
            before_drain_entered_rx,
            before_drain_release_tx,
            before_drain_release_rx,
            before_exit_entered_tx,
            before_exit_entered_rx,
            before_exit_release_tx,
            before_exit_release_rx,
        }
    }

    fn pause_before_drain(&self) {
        let _ = self.before_drain_entered_tx.send(());
        let _ = self.before_drain_release_rx.recv();
    }

    fn pause_before_exit(&self) {
        let _ = self.before_exit_entered_tx.send(());
        let _ = self.before_exit_release_rx.recv();
    }

    pub(crate) fn wait_before_drain(&self, timeout: Duration) -> bool {
        self.before_drain_entered_rx.recv_timeout(timeout).is_ok()
    }

    pub(crate) fn release_before_drain(&self) {
        let _ = self.before_drain_release_tx.send(());
    }

    pub(crate) fn wait_before_exit(&self, timeout: Duration) -> bool {
        self.before_exit_entered_rx.recv_timeout(timeout).is_ok()
    }

    pub(crate) fn release_before_exit(&self) {
        let _ = self.before_exit_release_tx.send(());
    }
}

/// One-shot deterministic barrier used by WAL interleaving tests.
#[cfg(test)]
#[derive(Debug)]
struct JournalTestBarrier {
    entered_tx: Sender<()>,
    entered_rx: Receiver<()>,
    release_tx: Sender<()>,
    release_rx: Receiver<()>,
}

#[cfg(test)]
impl JournalTestBarrier {
    fn new() -> Arc<Self> {
        let (entered_tx, entered_rx) = bounded(1);
        let (release_tx, release_rx) = bounded(1);
        Arc::new(Self {
            entered_tx,
            entered_rx,
            release_tx,
            release_rx,
        })
    }

    fn pause(&self) {
        let _ = self.entered_tx.send(());
        let _ = self.release_rx.recv();
    }

    fn wait(&self) {
        self.entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("journal test barrier was not reached");
    }

    fn release(&self) {
        self.release_tx
            .send(())
            .expect("journal test barrier waiter disappeared");
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubmitPauseStage {
    AfterReserve,
    AfterPublish,
}

#[cfg(test)]
#[derive(Debug)]
struct SubmitPause {
    seq: u64,
    stage: SubmitPauseStage,
    barrier: Arc<JournalTestBarrier>,
}

impl Shared {
    fn enter_file_store_health(&self) -> Result<Option<FileStoreHealthPermit<'_>>> {
        self.file_store_health
            .as_ref()
            .map(|health| health.enter())
            .transpose()
    }

    fn sticky_err(&self) -> Option<JournalFailure> {
        *self.err.lock().unwrap()
    }

    fn set_err(&self, failure: JournalFailure) {
        let mut slot = self.err.lock().unwrap();
        if slot.is_none() {
            *slot = Some(failure);
        }
        drop(slot);
        // Wake any sync waiters so they observe the error.
        let flushed_guard = self.flushed_mx.lock().unwrap();
        self.flushed_cv.notify_all();
        drop(flushed_guard);
        let _prefix_guard = self.prefix_mx.lock().unwrap();
        self.prefix_cv.notify_all();
    }

    #[cfg(test)]
    fn pause_submit(&self, record: &[u8], stage: SubmitPauseStage) {
        if record.len() < 16 {
            return;
        }
        let seq = u64::from_le_bytes(record[8..16].try_into().unwrap());
        let barrier = {
            let mut pauses = self.submit_pauses.lock().unwrap();
            pauses
                .iter()
                .position(|pause| pause.seq == seq && pause.stage == stage)
                .map(|position| pauses.remove(position).barrier)
        };
        if let Some(barrier) = barrier {
            barrier.pause();
        }
    }

    #[cfg(test)]
    fn pause_once(slot: &Mutex<Option<Arc<JournalTestBarrier>>>) {
        let barrier = slot.lock().unwrap().take();
        if let Some(barrier) = barrier {
            barrier.pause();
        }
    }

    fn physical_target(&self, prefix: CommittedPrefix) -> u64 {
        self.record_base
            .saturating_add(prefix.records)
            .saturating_add(self.oversized_records.load(Ordering::Acquire))
    }

    fn notify_prefix_advanced(&self) {
        if self.prefix_waiters.load(Ordering::Acquire) != 0 {
            let _guard = self.prefix_mx.lock().unwrap();
            self.prefix_cv.notify_all();
        }
    }

    fn wait_for_ticket_prefix(
        &self,
        ticket: &ReserveTicket,
        published: CommittedPrefix,
    ) -> std::result::Result<CommittedPrefix, JournalFailure> {
        if published.addr >= ticket.end {
            return Ok(published);
        }

        self.prefix_waiters.fetch_add(1, Ordering::AcqRel);
        let result = (|| {
            let mut guard = self.prefix_mx.lock().unwrap();
            loop {
                let prefix = self.ring.committed_prefix_snapshot();
                if prefix.addr >= ticket.end {
                    return Ok(prefix);
                }
                if let Some(failure) = self.sticky_err() {
                    return Err(failure);
                }
                guard = self.prefix_cv.wait(guard).unwrap();
            }
        })();
        self.prefix_waiters.fetch_sub(1, Ordering::AcqRel);
        result
    }

    fn request_wake(
        &self,
        disconnected: JournalFailure,
    ) -> std::result::Result<(), JournalFailure> {
        match self.wake_tx.try_send(()) {
            Ok(()) | Err(TrySendError::Full(())) => Ok(()),
            Err(TrySendError::Disconnected(())) => {
                self.set_err(disconnected);
                Err(disconnected)
            }
        }
    }

    /// Drain the committed prefix into the writer; if a sync target is
    /// outstanding, fsync and advance `flushed`. Flusher thread only.
    fn drain_and_maybe_sync(&self) {
        if self.sticky_err().is_some() {
            return;
        }
        let mut sink_err: Option<&'static str> = None;
        {
            let mut w = self.writer.lock().unwrap();
            let prefix = self.ring.committed_prefix_snapshot();
            #[cfg(test)]
            if prefix.addr > self.ring.flush_cursor() {
                Self::pause_once(&self.after_snapshot_barrier);
            }
            let copied = self.ring.copy_prefix(prefix, &mut |bytes| {
                if sink_err.is_none() && w.append_encoded(bytes).is_err() {
                    sink_err = Some("journal flusher append failed");
                }
            });
            if let Some(msg) = sink_err {
                drop(w);
                self.set_err(JournalFailure {
                    phase: CommitPhase::WalAppend,
                    context: msg,
                });
                return;
            }
            if copied.bytes > 0 {
                // wal_sync=false still promises prompt process-crash
                // durability: every poll batch reaches the OS page cache,
                // without implying sync_data/power-loss durability.
                if w.drain_to_os().is_err() {
                    drop(w);
                    self.set_err(JournalFailure {
                        phase: CommitPhase::WalAppend,
                        context: "journal flusher page-cache drain failed",
                    });
                    return;
                }
                let written_to = self.physical_target(copied.prefix);
                self.written.fetch_max(written_to, Ordering::AcqRel);
                self.batches.fetch_add(1, Ordering::Relaxed);
                // Publish the freed ring space before a durability gate can
                // wait behind manifest poison. A pre-poison submit holding a
                // shared health permit may itself be parked on this cursor.
                let _g = self.space_mx.lock().unwrap();
                self.space_cv.notify_all();
            }
            let sync_target = self.sync_target.load(Ordering::Acquire);
            let flushed = self.flushed.load(Ordering::Acquire);
            let written_to = self.written.load(Ordering::Acquire);
            let want_sync = sync_target > flushed && written_to >= sync_target;
            if want_sync {
                // Appending already-accepted records may be needed to free
                // ring space for a submit that linearized before poison. The
                // durability/ACK boundary itself is fenced here.
                let Ok(_health) = self.enter_file_store_health() else {
                    drop(w);
                    self.set_err(JournalFailure {
                        phase: CommitPhase::WalSync,
                        context: "file-store health gate rejected journal fsync",
                    });
                    return;
                };
                #[cfg(test)]
                Self::pause_once(&self.before_sync_barrier);
                if w.sync_data().is_err() {
                    drop(w);
                    self.set_err(JournalFailure {
                        phase: CommitPhase::WalSync,
                        context: "journal flusher sync_data failed",
                    });
                    return;
                }
                self.syncs.fetch_add(1, Ordering::Relaxed);
                self.flushed.fetch_max(written_to, Ordering::AcqRel);
            }
        }
        if self.flushed.load(Ordering::Acquire) >= self.sync_target.load(Ordering::Acquire) {
            let _g = self.flushed_mx.lock().unwrap();
            self.flushed_cv.notify_all();
        }
    }

    /// Block until the reserved range fits below `flush_cursor + capacity`
    /// (built-in backpressure). Parks on `space_cv` instead of spinning;
    /// rare in practice (the ring is sized to absorb bursts between
    /// checkpoints), but bounds RAM and CPU under sustained overload.
    fn wait_for_ring_space(
        &self,
        ticket: &ReserveTicket,
    ) -> std::result::Result<(), JournalFailure> {
        self.request_wake(JournalFailure {
            phase: CommitPhase::WalAppend,
            context: "journal flusher stopped during WAL ring backpressure",
        })?;
        let mut guard = self.space_mx.lock().unwrap();
        while !self.ring.reserve_space_ready(ticket) {
            if let Some(failure) = self.sticky_err() {
                return Err(failure);
            }
            self.request_wake(JournalFailure {
                phase: CommitPhase::WalAppend,
                context: "journal flusher stopped during WAL ring backpressure",
            })?;
            let (next, _timeout) = self
                .space_cv
                .wait_timeout(guard, FLUSH_POLL.saturating_mul(4))
                .unwrap();
            guard = next;
        }
        Ok(())
    }

    /// Block until `flushed >= target` (or a flusher error). Used by sync
    /// appends (`JournalAck::wait`) and `flush_up_to`.
    fn flush_to(&self, target: u64) -> std::result::Result<(), JournalFailure> {
        // Reject even an already-satisfied ACK after the owning store has
        // entered fail-stop. The final recheck below closes the wait race.
        drop(self.enter_file_store_health().map_err(|_| JournalFailure {
            phase: CommitPhase::WalSync,
            context: "file-store health gate rejected journal acknowledgement",
        })?);
        if target <= self.flushed.load(Ordering::Acquire) {
            return match self.sticky_err() {
                Some(failure) => Err(failure),
                None => Ok(()),
            };
        }
        self.sync_target.fetch_max(target, Ordering::AcqRel);
        // Wake the flusher. A disconnected receiver is an acknowledgement
        // failure, not an advisory miss: no worker remains to advance the
        // requested append/fsync boundary.
        self.request_wake(self.stopped_before_target(target))?;
        let mut guard = self.flushed_mx.lock().unwrap();
        loop {
            if let Some(failure) = self.sticky_err() {
                return Err(failure);
            }
            if self.flushed.load(Ordering::Acquire) >= target {
                let _health = self.enter_file_store_health().map_err(|_| JournalFailure {
                    phase: CommitPhase::WalSync,
                    context: "file-store health gate rejected journal acknowledgement",
                })?;
                if let Some(failure) = self.sticky_err() {
                    return Err(failure);
                }
                if self.flushed.load(Ordering::Acquire) >= target {
                    return Ok(());
                }
            }
            let (next, timeout) = self
                .flushed_cv
                .wait_timeout(guard, FLUSH_POLL.saturating_mul(4))
                .unwrap();
            guard = next;
            if timeout.timed_out() {
                drop(guard);
                self.request_wake(self.stopped_before_target(target))?;
                guard = self.flushed_mx.lock().unwrap();
            }
        }
    }

    fn stopped_before_target(&self, target: u64) -> JournalFailure {
        if target <= self.written.load(Ordering::Acquire) {
            JournalFailure {
                phase: CommitPhase::WalSync,
                context: "journal flusher stopped before WAL sync acknowledgement",
            }
        } else {
            JournalFailure {
                phase: CommitPhase::WalAppend,
                context: "journal flusher stopped before WAL append acknowledgement",
            }
        }
    }

    fn record_buffer(&self, min_capacity: usize) -> Result<Vec<u8>> {
        if min_capacity <= RECORD_BUFFER_RETAIN_MAX {
            if let Ok(mut pool) = self.record_pool.try_lock() {
                while let Some(mut buf) = pool.pop() {
                    if buf.capacity() >= min_capacity {
                        buf.clear();
                        return Ok(buf);
                    }
                }
            }
        }
        let mut buf = Vec::new();
        buf.try_reserve_exact(min_capacity)
            .map_err(|_| Error::Internal("journal record buffer allocation failed"))?;
        Ok(buf)
    }

    fn recycle(&self, mut buf: Vec<u8>) {
        if buf.capacity() == 0 || buf.capacity() > RECORD_BUFFER_RETAIN_MAX {
            return;
        }
        if let Ok(mut pool) = self.record_pool.try_lock() {
            if pool.len() < RECORD_BUFFER_POOL_LIMIT {
                buf.clear();
                pool.push(buf);
            }
        }
    }

    /// Finish a graceful async shutdown by handing the writer's sub-threshold
    /// buffer to the OS. This intentionally does not call `sync_data`: a
    /// `wal_sync = false` shutdown must preserve normal reopen semantics
    /// without silently changing its power-loss durability contract.
    fn drain_writer_to_os(&self) {
        if self.sticky_err().is_some() {
            return;
        }
        let Ok(_health) = self.enter_file_store_health() else {
            self.set_err(JournalFailure {
                phase: CommitPhase::WalAppend,
                context: "file-store health gate rejected journal final drain",
            });
            return;
        };
        if self.writer.lock().unwrap().drain_to_os().is_err() {
            self.set_err(JournalFailure {
                phase: CommitPhase::WalAppend,
                context: "journal flusher final drain failed",
            });
        }
    }

    /// Append one record outside the fixed ring. The caller owns the
    /// oversized lane exclusively, so every preceding small publisher has
    /// completed and no later small record can enter until this returns.
    fn append_oversized(
        &self,
        record: &[u8],
        target: u64,
    ) -> std::result::Result<(), JournalFailure> {
        if let Some(failure) = self.sticky_err() {
            return Err(failure);
        }

        // Oversized submit releases its preparatory health permit before
        // handing work to this thread. Re-enter here so the direct append is
        // itself ordered against poison without recursively acquiring a read
        // lock while a poison writer may be queued.
        let Ok(_health) = self.enter_file_store_health() else {
            let failure = JournalFailure {
                phase: CommitPhase::WalAppend,
                context: "file-store health gate rejected journal oversized append",
            };
            self.set_err(failure);
            return Err(failure);
        };

        // A prior ring drain can leave a sub-threshold prefix in WalWriter's
        // private buffer. Hand it to the OS before attempting the large
        // record, otherwise a failure on the latter could strand an already
        // acknowledged earlier prefix behind it.
        let result = self.writer.lock().unwrap().append_encoded_direct(record);
        if result.is_err() {
            let failure = JournalFailure {
                phase: CommitPhase::WalAppend,
                context: "journal oversized append failed",
            };
            self.set_err(failure);
            return Err(failure);
        }

        self.oversized_records.fetch_add(1, Ordering::AcqRel);
        self.written.fetch_max(target, Ordering::AcqRel);
        self.batches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    #[cfg(test)]
    fn take_shutdown_barrier(&self) -> Option<Arc<JournalShutdownBarrier>> {
        self.shutdown_barrier.lock().unwrap().take()
    }
}

/// Completion handle for one acknowledged journal append. Async appends
/// return `None`; sync appends return a handle whose `wait` blocks until the
/// record is fsync-durable.
pub(crate) struct JournalAck {
    shared: Arc<Shared>,
    target: u64,
}

impl JournalAck {
    pub(crate) fn wait(self) -> Result<()> {
        self.shared.flush_to(self.target).map_err(|failure| {
            // A sync failure proves only the prefix reported by `written`
            // reached the writer. A concurrently published successor beyond
            // that prefix failed before its own append boundary.
            if failure.phase == CommitPhase::WalSync
                && self.target > self.shared.written.load(Ordering::Acquire)
            {
                Error::CommitOutcomeUnknown {
                    phase: CommitPhase::WalAppend,
                    context: "journal stopped before this WAL record was appended",
                }
            } else {
                failure.unknown()
            }
        })
    }
}

pub(crate) struct Journal {
    shared: Arc<Shared>,
    handle: Mutex<Option<JoinHandle<()>>>,
    /// Owner of the descriptor set and health gate whose exclusion lifetime
    /// must cover the worker's final drain and join. `Drop` releases this
    /// explicitly after join; the worker receives its own clone so neither
    /// side depends on struct-field destruction order.
    resource_guard: Option<Arc<FileBlobStore>>,
}

impl Journal {
    #[cfg(test)]
    pub(crate) fn open_or_create(path: &std::path::Path, tree_id: u64) -> Result<Self> {
        let writer = WalWriter::open_or_create(path, tree_id)?;
        Self::from_writer(writer, None)
    }

    #[cfg(test)]
    fn open_or_create_with_limits(
        path: &std::path::Path,
        tree_id: u64,
        ring_capacity: usize,
        max_record_bytes: usize,
    ) -> Result<Self> {
        let writer = WalWriter::open_or_create(path, tree_id)?;
        Self::from_writer_with_limits(writer, None, ring_capacity, max_record_bytes)
    }

    /// Open a descriptor-backed WAL whose worker keeps `resource_guard` and
    /// its dynamic fail-stop state live.
    pub(crate) fn open_or_create_file(
        file: File,
        tree_id: u64,
        resource_guard: Arc<FileBlobStore>,
    ) -> Result<Self> {
        let writer = WalWriter::open_or_create_file(file, tree_id)?;
        Self::from_writer(writer, Some(resource_guard))
    }

    fn from_writer(writer: WalWriter, resource_guard: Option<Arc<FileBlobStore>>) -> Result<Self> {
        Self::from_writer_with_limits(
            writer,
            resource_guard,
            RING_CAPACITY_BYTES,
            MAX_ATOMIC_WAL_RECORD_BYTES,
        )
    }

    fn from_writer_with_limits(
        writer: WalWriter,
        resource_guard: Option<Arc<FileBlobStore>>,
        ring_capacity: usize,
        max_record_bytes: usize,
    ) -> Result<Self> {
        if max_record_bytes < ring_capacity {
            return Err(Error::Internal(
                "journal record ceiling is smaller than ring capacity",
            ));
        }
        let record_base = u64::from(writer.has_records());
        // Mirror legacy reopen seeding: a reopened non-empty WAL is queued
        // and unflushed, so the first checkpoint flushes before making
        // replayed effects durable.
        let initial_flushed = record_base.saturating_sub(1);

        let (control_tx, control_rx) = bounded::<Control>(ORDERED_CONTROL_CAPACITY);
        let (wake_tx, wake_rx) = bounded::<()>(WAKE_CAPACITY);
        let file_store_health = resource_guard
            .as_ref()
            .map(|file_store| file_store.journal_health());
        let shared = Arc::new(Shared {
            ring: WalRing::with_capacity(ring_capacity),
            record_lane: RecordLane::default(),
            writer: Mutex::new(writer),
            record_base,
            queued: AtomicU64::new(record_base),
            written: AtomicU64::new(record_base),
            flushed: AtomicU64::new(initial_flushed),
            checkpointed: AtomicU64::new(0),
            sync_target: AtomicU64::new(0),
            oversized_records: AtomicU64::new(0),
            max_record_bytes: AtomicUsize::new(max_record_bytes),
            appends: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            syncs: AtomicU64::new(0),
            file_store_health,
            err: Mutex::new(None),
            flushed_mx: Mutex::new(()),
            flushed_cv: Condvar::new(),
            space_mx: Mutex::new(()),
            space_cv: Condvar::new(),
            prefix_mx: Mutex::new(()),
            prefix_cv: Condvar::new(),
            prefix_waiters: AtomicUsize::new(0),
            control_tx,
            wake_tx,
            record_pool: Mutex::new(Vec::new()),
            #[cfg(test)]
            shutdown_barrier: Mutex::new(None),
            #[cfg(test)]
            submit_pauses: Mutex::new(Vec::new()),
            #[cfg(test)]
            before_drain_barrier: Mutex::new(None),
            #[cfg(test)]
            after_snapshot_barrier: Mutex::new(None),
            #[cfg(test)]
            before_sync_barrier: Mutex::new(None),
        });

        let worker_shared = Arc::clone(&shared);
        let worker_resource_guard = resource_guard.clone();
        let handle = thread::Builder::new()
            .name("holt-journal-ring".to_owned())
            .spawn(move || {
                run_flusher(worker_shared, control_rx, wake_rx, worker_resource_guard);
            })
            .map_err(|_| Error::Internal("OS rejected thread spawn for holt-journal-ring"))?;

        Ok(Self {
            shared,
            handle: Mutex::new(Some(handle)),
            resource_guard,
        })
    }

    /// Admit one record before any tree mutation and allocate its encoding
    /// buffer only after lane backpressure succeeds.
    pub(crate) fn prepare_record(&self, admitted_upper_bound: usize) -> Result<PreparedRecord<'_>> {
        if admitted_upper_bound == 0 {
            return Err(Error::Internal(
                "journal record admission requires a non-zero upper bound",
            ));
        }
        let max_bytes = self.shared.max_record_bytes.load(Ordering::Acquire);
        if admitted_upper_bound > max_bytes {
            return Err(Error::AtomicRecordTooLarge {
                encoded_bytes: admitted_upper_bound,
                max_bytes,
            });
        }
        if let Some(failure) = self.shared.sticky_err() {
            return Err(Error::Internal(failure.context));
        }
        // Lane admission precedes health admission. An oversized owner hands
        // direct append to the flusher, which must acquire health itself;
        // holding health while waiting behind that owner could deadlock with a
        // queued poison writer. No mutation begins until both admissions pass.
        let permit = if admitted_upper_bound as u64 <= self.shared.ring.capacity() {
            self.shared.record_lane.enter_small()
        } else {
            self.shared.record_lane.enter_oversized()
        };
        let file_store_health = match self.shared.enter_file_store_health() {
            Ok(health) => health,
            Err(error) => {
                drop(permit);
                return Err(error);
            }
        };
        // A failure may race admission while this caller waits behind an
        // oversized writer. It is still a definite rejection: no caller is
        // allowed to mutate until this method returns a PreparedRecord.
        if let Some(failure) = self.shared.sticky_err() {
            drop(permit);
            drop(file_store_health);
            return Err(Error::Internal(failure.context));
        }

        let bytes = match self.shared.record_buffer(admitted_upper_bound) {
            Ok(bytes) => bytes,
            Err(error) => {
                drop(permit);
                drop(file_store_health);
                return Err(error);
            }
        };
        Ok(PreparedRecord {
            shared: &self.shared,
            bytes: Some(bytes),
            admitted_upper_bound,
            permit,
            file_store_health,
        })
    }

    /// Submit one record whose size and lane were admitted before mutation.
    /// Sync appends return an ack whose wait is the fsync boundary.
    pub(crate) fn submit(
        &self,
        mut record: PreparedRecord<'_>,
        sync: bool,
    ) -> Result<Option<JournalAck>> {
        let append_unknown = |context| Error::CommitOutcomeUnknown {
            phase: CommitPhase::WalAppend,
            context,
        };
        if !std::ptr::eq(record.shared, self.shared.as_ref()) {
            return Err(append_unknown(
                "prepared WAL record belongs to another journal",
            ));
        }
        if let Some(failure) = self.shared.sticky_err() {
            return Err(append_unknown(failure.context));
        }
        let actual_len = record.len();
        if actual_len == 0 {
            return Err(append_unknown("journal record must not be empty"));
        }
        if actual_len > record.admitted_upper_bound {
            return Err(append_unknown(
                "encoded WAL record exceeded its admitted upper bound",
            ));
        }

        let n = match record.lane_kind() {
            RecordLaneKind::Small => {
                debug_assert!(actual_len as u64 <= self.shared.ring.capacity());
                // Reserve → backpressure wait → memcpy → publish.
                // The flusher's cursor provides bounded ring backpressure.
                let ticket = self.shared.ring.reserve(actual_len as u64);
                #[cfg(test)]
                self.shared
                    .pause_submit(&record, SubmitPauseStage::AfterReserve);
                if !self.shared.ring.reserve_space_ready(&ticket) {
                    self.shared
                        .wait_for_ring_space(&ticket)
                        .map_err(|failure| append_unknown(failure.context))?;
                }
                self.shared.ring.fill(&ticket, &record);
                let published = self.shared.ring.publish(&ticket);
                if published.advanced {
                    self.shared.notify_prefix_advanced();
                }
                #[cfg(test)]
                self.shared
                    .pause_submit(&record, SubmitPauseStage::AfterPublish);
                let bytes = record.take_bytes();
                self.shared.recycle(bytes);
                let prefix = self
                    .shared
                    .wait_for_ticket_prefix(&ticket, published.prefix)
                    .map_err(|failure| append_unknown(failure.context))?;
                let target = self.shared.physical_target(prefix);
                self.shared.queued.fetch_max(target, Ordering::AcqRel);
                target
            }
            RecordLaneKind::Oversized => {
                // Transfer the health boundary to the flusher before waiting
                // for its direct-append acknowledgement. Keeping this read
                // permit while the worker re-entered health could deadlock
                // behind a queued poison writer; the lane permit remains held
                // until every success/error/channel-close path returns.
                drop(record.file_store_health.take());
                let bytes = record.take_bytes();
                let prefix = self.shared.ring.committed_prefix_snapshot();
                let target = self.shared.physical_target(prefix).saturating_add(1);
                self.shared.queued.fetch_max(target, Ordering::AcqRel);
                let (ack, rx) = crossbeam_channel::bounded(1);
                if self
                    .shared
                    .control_tx
                    .send(Control::AppendOversized {
                        record: bytes,
                        target,
                        ack,
                    })
                    .is_err()
                {
                    let failure = JournalFailure {
                        phase: CommitPhase::WalAppend,
                        context: "journal flusher stopped before oversized append",
                    };
                    self.shared.set_err(failure);
                    return Err(failure.unknown());
                }
                match rx.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(failure)) => return Err(append_unknown(failure.context)),
                    Err(_) => {
                        let failure = JournalFailure {
                            phase: CommitPhase::WalAppend,
                            context: "journal flusher dropped oversized append acknowledgement",
                        };
                        self.shared.set_err(failure);
                        return Err(failure.unknown());
                    }
                }
                target
            }
        };

        self.shared.appends.fetch_add(1, Ordering::Relaxed);
        // Small async records retain the existing fast acknowledgement: the
        // flusher polls promptly, but submit does not wait for page-cache I/O.
        // Oversized records wait above because their bounded Vec and exclusive
        // ordering permit cannot be released before the direct append.
        if sync {
            Ok(Some(JournalAck {
                shared: Arc::clone(&self.shared),
                target: n,
            }))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn queued_work(&self) -> u64 {
        self.shared.queued.load(Ordering::Acquire)
    }

    pub(crate) fn max_record_bytes(&self) -> usize {
        self.shared.max_record_bytes.load(Ordering::Acquire)
    }

    pub(crate) fn flush_up_to(&self, observed: u64) -> Result<()> {
        self.shared
            .flush_to(observed)
            .map_err(|failure| Error::Internal(failure.context))
    }

    #[cfg(test)]
    pub(crate) fn set_max_record_bytes_for_test(&self, max_record_bytes: usize) {
        self.shared
            .max_record_bytes
            .store(max_record_bytes, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn fail_next_append_for_test(&self) {
        self.shared.writer.lock().unwrap().fail_next_append();
    }

    #[cfg(test)]
    pub(crate) fn fail_next_drain_for_test(&self) {
        self.shared.writer.lock().unwrap().fail_next_drain();
    }

    #[cfg(test)]
    pub(crate) fn fail_next_sync_for_test(&self) {
        self.shared.writer.lock().unwrap().fail_next_sync();
    }

    #[cfg(test)]
    pub(crate) fn stop_worker_for_test(&self) {
        self.shared.control_tx.send(Control::Stop).unwrap();
        if let Some(handle) = self.handle.lock().unwrap().take() {
            handle.join().unwrap();
        }
    }

    #[cfg(test)]
    fn oversized_waiting_for_test(&self) -> bool {
        self.shared.record_lane.oversized_waiting()
    }

    #[cfg(test)]
    fn install_submit_pause(
        &self,
        seq: u64,
        stage: SubmitPauseStage,
        barrier: Arc<JournalTestBarrier>,
    ) {
        self.shared.submit_pauses.lock().unwrap().push(SubmitPause {
            seq,
            stage,
            barrier,
        });
    }

    #[cfg(test)]
    fn install_before_drain_barrier(&self, barrier: Arc<JournalTestBarrier>) {
        assert!(self
            .shared
            .before_drain_barrier
            .lock()
            .unwrap()
            .replace(barrier)
            .is_none());
        let _ = self.shared.wake_tx.try_send(());
    }

    #[cfg(test)]
    fn install_after_snapshot_barrier(&self, barrier: Arc<JournalTestBarrier>) {
        assert!(self
            .shared
            .after_snapshot_barrier
            .lock()
            .unwrap()
            .replace(barrier)
            .is_none());
    }

    #[cfg(test)]
    fn install_before_sync_barrier(&self, barrier: Arc<JournalTestBarrier>) {
        assert!(self
            .shared
            .before_sync_barrier
            .lock()
            .unwrap()
            .replace(barrier)
            .is_none());
    }

    #[cfg(test)]
    fn wake_depth_for_test(&self) -> usize {
        self.shared.wake_tx.len()
    }

    #[cfg(test)]
    fn ordered_control_depth_for_test(&self) -> usize {
        self.shared.control_tx.len()
    }

    pub(crate) fn truncate(&self) -> Result<()> {
        let observed = self.shared.queued.load(Ordering::Acquire);
        if observed == self.shared.checkpointed.load(Ordering::Acquire) {
            return Ok(());
        }
        let (ack, rx) = crossbeam_channel::bounded(1);
        self.shared
            .control_tx
            .send(Control::Truncate(ack))
            .map_err(|_| Error::Internal("journal flusher stopped before truncate"))?;
        rx.recv()
            .map_err(|_| Error::Internal("journal flusher dropped truncate acknowledgement"))??;
        self.shared
            .checkpointed
            .fetch_max(observed, Ordering::AcqRel);
        Ok(())
    }

    pub(crate) fn needs_checkpoint(&self) -> bool {
        self.shared.queued.load(Ordering::Acquire)
            != self.shared.checkpointed.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn needs_flush(&self) -> bool {
        self.shared.queued.load(Ordering::Acquire) > self.shared.flushed.load(Ordering::Acquire)
    }

    pub(crate) fn stats(&self) -> JournalStats {
        let queued_work = self.shared.queued.load(Ordering::Acquire);
        let written_work = self.shared.written.load(Ordering::Acquire);
        let flushed_work = self.shared.flushed.load(Ordering::Acquire);
        let checkpointed_work = self.shared.checkpointed.load(Ordering::Acquire);
        JournalStats {
            appends: self.shared.appends.load(Ordering::Relaxed),
            batches: self.shared.batches.load(Ordering::Relaxed),
            syncs: self.shared.syncs.load(Ordering::Relaxed),
            queued_work,
            written_work,
            flushed_work,
            checkpointed_work,
            pending_work: queued_work.saturating_sub(flushed_work),
            checkpoint_debt: queued_work.saturating_sub(checkpointed_work),
        }
    }

    #[cfg(test)]
    pub(crate) fn install_shutdown_barrier(&self, barrier: Arc<JournalShutdownBarrier>) {
        let mut slot = self.shared.shutdown_barrier.lock().unwrap();
        assert!(slot.replace(barrier).is_none());
        drop(slot);
        // Wake the worker so it observes the newly-installed barrier without
        // relying on the idle poll deadline.
        let _ = self.shared.wake_tx.try_send(());
    }

    #[cfg(test)]
    pub(crate) fn sync_target_for_test(&self) -> u64 {
        self.shared.sync_target.load(Ordering::Acquire)
    }
}

impl Drop for Journal {
    fn drop(&mut self) {
        let _ = self.shared.control_tx.send(Control::Stop);
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
        // Release file-store exclusion only after the worker has completed
        // its final WAL drain and the join has observed thread exit.
        drop(self.resource_guard.take());
    }
}

fn run_flusher(
    shared: Arc<Shared>,
    control_rx: Receiver<Control>,
    wake_rx: Receiver<()>,
    _resource_guard: Option<Arc<FileBlobStore>>,
) {
    #[cfg(test)]
    let mut shutdown_barrier = None;
    loop {
        #[cfg(test)]
        if shutdown_barrier.is_none() {
            shutdown_barrier = shared.take_shutdown_barrier();
            if let Some(barrier) = &shutdown_barrier {
                barrier.pause_before_drain();
            }
        }
        #[cfg(test)]
        Shared::pause_once(&shared.before_drain_barrier);
        shared.drain_and_maybe_sync();
        crossbeam_channel::select_biased! {
            recv(control_rx) -> control => match control {
            Ok(Control::Truncate(ack)) => {
                // Drain anything outstanding, then reset. The caller holds
                // the commit gate exclusively at the checkpoint truncate
                // boundary, so no writer is mid-reserve here.
                shared.drain_and_maybe_sync();
                let result = do_truncate(&shared);
                let _ = ack.send(result);
            }
            Ok(Control::AppendOversized {
                record,
                target,
                ack,
            }) => {
                // The exclusive record-lane permit proves every preceding
                // small publisher has finished and blocks every successor.
                // Re-drain because some of that prefix may have published
                // after the loop's first drain but before this control recv.
                shared.drain_and_maybe_sync();
                let result = shared.append_oversized(&record, target);
                let _ = ack.send(result);
            }
            Ok(Control::Stop) => {
                shared.drain_and_maybe_sync();
                shared.drain_writer_to_os();
                break;
            }
            Err(_) => break,
            },
            // Wake tokens are coalesced and advisory. The ordered channel
            // above is biased so critical commands never depend on a wake.
            recv(wake_rx) -> _ => {},
            default(FLUSH_POLL) => {},
        }
    }
    #[cfg(test)]
    if let Some(barrier) = shutdown_barrier {
        barrier.pause_before_exit();
    }
}

fn do_truncate(shared: &Shared) -> Result<()> {
    if let Some(failure) = shared.sticky_err() {
        return Err(Error::Internal(failure.context));
    }
    let _health = shared.enter_file_store_health()?;
    let mut w = shared.writer.lock().unwrap();
    w.truncate()?;
    drop(w);
    // The ring is fully drained (the drain above caught up; no concurrent
    // writer under the checkpoint gate). Reset byte cursors; record count is
    // preserved as the stable cross-truncation order.
    shared.ring.reset_after_drain();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::codec::{encode_insert_record, FILE_HEADER_SIZE};
    use crate::journal::reader;
    use std::sync::mpsc;
    use std::time::Instant;

    fn prepared<'a>(journal: &'a Journal, bytes: &[u8]) -> PreparedRecord<'a> {
        let mut record = journal.prepare_record(bytes.len()).unwrap();
        record.extend_from_slice(bytes);
        record
    }

    // The 6 legacy contract tests, retargeted at the ring-backed Journal.

    #[test]
    fn fresh_journal_flush_and_truncate_are_noops() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        assert!(!journal.needs_checkpoint());
        journal.flush_up_to(journal.queued_work()).unwrap();
        journal.truncate().unwrap();

        let stats = journal.stats();
        assert_eq!(stats.appends, 0);
        assert_eq!(stats.syncs, 0);
        assert!(!journal.needs_checkpoint());
    }

    #[test]
    fn append_requires_one_checkpoint_truncate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Journal::open_or_create(&path, 0).unwrap();

        journal
            .submit(prepared(&journal, &[1, 2, 3, 4]), false)
            .unwrap();
        assert!(journal.needs_checkpoint());
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);

        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            FILE_HEADER_SIZE as u64
        );

        let syncs_after_truncate = journal.stats().syncs;
        journal.flush_up_to(journal.queued_work()).unwrap();
        journal.truncate().unwrap();
        assert_eq!(journal.stats().syncs, syncs_after_truncate);
    }

    #[test]
    fn durable_append_satisfies_later_flush_barrier() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        let ack = journal
            .submit(prepared(&journal, &[5, 6, 7, 8]), true)
            .unwrap()
            .expect("durable append returns an ack");
        ack.wait().unwrap();

        assert!(journal.needs_checkpoint());
        assert!(!journal.needs_flush());
        let syncs_after_append = journal.stats().syncs;
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert_eq!(journal.stats().syncs, syncs_after_append);

        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
    }

    #[test]
    fn enqueue_append_is_flushed_by_later_barrier() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Journal::open_or_create(&path, 0).unwrap();

        let ack = journal
            .submit(prepared(&journal, &[1, 3, 5, 7]), false)
            .unwrap();
        assert!(ack.is_none());

        journal.flush_up_to(journal.queued_work()).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);
        assert!(!journal.needs_flush());
        assert_eq!(journal.stats().syncs, 1);
        assert_eq!(journal.stats().appends, 1);
        assert!(journal.stats().batches >= 1);
    }

    #[test]
    fn invalid_record_size_is_rejected_without_poisoning_journal() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        let empty = journal.prepare_record(1).unwrap();
        assert!(matches!(
            journal.submit(empty, false),
            Err(Error::CommitOutcomeUnknown {
                phase: CommitPhase::WalAppend,
                ..
            })
        ));
        assert!(matches!(
            journal.prepare_record(MAX_ATOMIC_WAL_RECORD_BYTES + 1),
            Err(Error::AtomicRecordTooLarge { .. })
        ));
        let mut encoder_drift = journal.prepare_record(4).unwrap();
        encoder_drift.extend_from_slice(&[0; 5]);
        assert!(matches!(
            journal.submit(encoder_drift, false),
            Err(Error::CommitOutcomeUnknown {
                phase: CommitPhase::WalAppend,
                ..
            })
        ));

        journal
            .submit(prepared(&journal, &[1, 2, 3, 4]), false)
            .unwrap();
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert_eq!(journal.stats().appends, 1);
    }

    #[test]
    fn encoded_record_buffers_are_recycled_after_flusher_append() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        let mut record = journal.prepare_record(64).unwrap();
        let capacity = record.capacity();
        assert!(capacity >= 64);
        record.extend_from_slice(&[1; 32]);

        journal.submit(record, false).unwrap();
        journal.flush_up_to(journal.queued_work()).unwrap();

        let reused = journal.prepare_record(16).unwrap();
        assert!(reused.capacity() >= capacity);
    }

    #[test]
    fn reopened_nonempty_wal_still_needs_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        {
            let journal = Journal::open_or_create(&path, 0).unwrap();
            journal
                .submit(prepared(&journal, &[9, 8, 7, 6]), false)
                .unwrap();
            journal.flush_up_to(journal.queued_work()).unwrap();
            assert!(journal.needs_checkpoint());
        }

        let journal = Journal::open_or_create(&path, 0).unwrap();
        assert!(journal.needs_checkpoint());
        assert!(journal.needs_flush());
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert!(!journal.needs_flush());
        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
    }

    #[test]
    fn oversized_direct_append_failure_is_unknown_and_replays_no_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        {
            let journal = Journal::open_or_create_with_limits(&path, 0, 256, 4 * 1024).unwrap();
            let mut bytes = Vec::new();
            encode_insert_record(&mut bytes, 1, 0, b"large", &[0xA5; 512]);
            assert!(bytes.len() > journal.shared.ring.capacity() as usize);

            journal.fail_next_append_for_test();
            let Err(error) = journal.submit(prepared(&journal, &bytes), false) else {
                panic!("injected oversized append unexpectedly succeeded");
            };
            assert!(matches!(
                error,
                Error::CommitOutcomeUnknown {
                    phase: CommitPhase::WalAppend,
                    ..
                }
            ));
            let stats = journal.stats();
            assert_eq!(stats.queued_work, 1);
            assert_eq!(stats.written_work, 0);
            assert_eq!(stats.appends, 0);
        }

        let mut seqs = Vec::new();
        reader::replay(&path, |_, seq, _| {
            seqs.push(seq);
            Ok(())
        })
        .unwrap();
        assert!(seqs.is_empty());
    }

    #[test]
    fn dropped_sync_ack_after_append_reports_wal_sync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Journal::open_or_create(&path, 0).unwrap();
        let mut bytes = Vec::new();
        encode_insert_record(&mut bytes, 1, 0, b"key", b"value");
        let ack = journal
            .submit(prepared(&journal, &bytes), true)
            .unwrap()
            .expect("sync append returns an acknowledgement");

        let deadline = Instant::now() + Duration::from_secs(1);
        while journal.stats().written_work == 0 {
            assert!(Instant::now() < deadline, "record never reached WAL writer");
            thread::yield_now();
        }
        assert_eq!(journal.stats().flushed_work, 0);
        journal.stop_worker_for_test();

        assert!(matches!(
            ack.wait(),
            Err(Error::CommitOutcomeUnknown {
                phase: CommitPhase::WalSync,
                ..
            })
        ));
        let mut seqs = Vec::new();
        reader::replay(&path, |_, seq, _| {
            seqs.push(seq);
            Ok(())
        })
        .unwrap();
        assert_eq!(seqs, vec![1]);
    }

    #[test]
    fn oversized_waiter_has_priority_over_later_small_admissions() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Arc::new(
            Journal::open_or_create_with_limits(&dir.path().join("journal.wal"), 0, 256, 4 * 1024)
                .unwrap(),
        );

        let first_small = journal.prepare_record(32).unwrap();
        let oversized_journal = Arc::clone(&journal);
        let (oversized_acquired_tx, oversized_acquired_rx) = mpsc::channel();
        let (release_oversized_tx, release_oversized_rx) = mpsc::channel();
        let oversized = thread::spawn(move || {
            let record = oversized_journal.prepare_record(512).unwrap();
            oversized_acquired_tx.send(()).unwrap();
            release_oversized_rx.recv().unwrap();
            drop(record);
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        while !journal.oversized_waiting_for_test() {
            assert!(
                Instant::now() < deadline,
                "oversized admission never waited"
            );
            thread::yield_now();
        }

        let later_small_journal = Arc::clone(&journal);
        let (small_started_tx, small_started_rx) = mpsc::channel();
        let (small_acquired_tx, small_acquired_rx) = mpsc::channel();
        let later_small = thread::spawn(move || {
            small_started_tx.send(()).unwrap();
            let record = later_small_journal.prepare_record(32).unwrap();
            small_acquired_tx.send(()).unwrap();
            drop(record);
        });
        small_started_rx.recv().unwrap();
        assert!(oversized_acquired_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err());
        assert!(small_acquired_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err());

        drop(first_small);
        oversized_acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert!(small_acquired_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err());

        release_oversized_tx.send(()).unwrap();
        oversized.join().unwrap();
        small_acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        later_small.join().unwrap();
    }

    #[test]
    fn oversized_lane_preserves_small_large_small_order_and_watermarks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal =
            Arc::new(Journal::open_or_create_with_limits(&path, 0, 256, 4 * 1024).unwrap());

        let mut first_bytes = Vec::new();
        encode_insert_record(&mut first_bytes, 1, 0, b"first", b"small");
        journal
            .submit(prepared(&journal, &first_bytes), false)
            .unwrap();

        let mut large_bytes = Vec::new();
        encode_insert_record(&mut large_bytes, 2, 0, b"large", &[0xA5; 512]);
        assert!(large_bytes.len() > journal.shared.ring.capacity() as usize);
        let mut large = journal.prepare_record(large_bytes.len()).unwrap();
        large.extend_from_slice(&large_bytes);

        let later_journal = Arc::clone(&journal);
        let (later_started_tx, later_started_rx) = mpsc::channel();
        let (later_done_tx, later_done_rx) = mpsc::channel();
        let later = thread::spawn(move || {
            later_started_tx.send(()).unwrap();
            let mut bytes = Vec::new();
            encode_insert_record(&mut bytes, 3, 0, b"later", b"small");
            let result = later_journal.submit(prepared(&later_journal, &bytes), false);
            later_done_tx.send(result.map(|_| ())).unwrap();
        });
        later_started_rx.recv().unwrap();
        assert!(later_done_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err());

        journal.submit(large, false).unwrap();
        let after_async_large = journal.stats();
        assert!(after_async_large.written_work >= 2);
        assert_eq!(after_async_large.flushed_work, 0);
        assert_eq!(after_async_large.syncs, 0);
        later_done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        later.join().unwrap();

        journal.flush_up_to(journal.queued_work()).unwrap();
        let stats = journal.stats();
        assert_eq!(stats.queued_work, 3);
        assert_eq!(stats.written_work, 3);
        assert_eq!(stats.flushed_work, 3);
        assert_eq!(stats.checkpointed_work, 0);

        let mut seqs = Vec::new();
        reader::replay(&path, |_, seq, _| {
            seqs.push(seq);
            Ok(())
        })
        .unwrap();
        assert_eq!(seqs, vec![1, 2, 3]);

        journal.truncate().unwrap();
        let truncated = journal.stats();
        assert_eq!(truncated.queued_work, 3);
        assert_eq!(truncated.written_work, 3);
        assert_eq!(truncated.flushed_work, 3);
        assert_eq!(truncated.checkpointed_work, 3);
        assert!(!journal.needs_checkpoint());

        let mut after_bytes = Vec::new();
        encode_insert_record(&mut after_bytes, 4, 0, b"after", b"truncate");
        journal
            .submit(prepared(&journal, &after_bytes), false)
            .unwrap();
        journal.flush_up_to(journal.queued_work()).unwrap();
        let after = journal.stats();
        assert_eq!(after.queued_work, 4);
        assert_eq!(after.written_work, 4);
        assert_eq!(after.flushed_work, 4);
        assert_eq!(after.checkpointed_work, 3);
        assert!(journal.needs_checkpoint());
    }

    fn encoded_insert(seq: u64, key: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_insert_record(&mut bytes, seq, 0, key, b"value");
        bytes
    }

    fn replay_seqs(path: &std::path::Path) -> Vec<u64> {
        let mut seqs = Vec::new();
        reader::replay(path, |_, seq, _| {
            seqs.push(seq);
            Ok(())
        })
        .unwrap();
        seqs
    }

    fn assert_unknown_phase(result: Result<()>, phase: CommitPhase) {
        assert!(matches!(
            result,
            Err(Error::CommitOutcomeUnknown { phase: actual, .. }) if actual == phase
        ));
    }

    #[test]
    fn sync_ack_uses_contiguous_physical_record_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let crash_image = dir.path().join("crash-image.wal");
        let journal = Arc::new(Journal::open_or_create(&path, 0).unwrap());

        let a_after_reserve = JournalTestBarrier::new();
        let a_after_publish = JournalTestBarrier::new();
        let b_after_reserve = JournalTestBarrier::new();
        let c_after_publish = JournalTestBarrier::new();
        journal.install_submit_pause(
            1,
            SubmitPauseStage::AfterReserve,
            Arc::clone(&a_after_reserve),
        );
        journal.install_submit_pause(
            1,
            SubmitPauseStage::AfterPublish,
            Arc::clone(&a_after_publish),
        );
        journal.install_submit_pause(
            2,
            SubmitPauseStage::AfterReserve,
            Arc::clone(&b_after_reserve),
        );
        journal.install_submit_pause(
            3,
            SubmitPauseStage::AfterPublish,
            Arc::clone(&c_after_publish),
        );

        let (a_done_tx, a_done_rx) = mpsc::channel();
        let a_journal = Arc::clone(&journal);
        let a = thread::spawn(move || {
            let bytes = encoded_insert(1, b"A");
            a_done_tx
                .send(a_journal.submit(prepared(&a_journal, &bytes), false))
                .unwrap();
        });
        a_after_reserve.wait();

        let (b_done_tx, b_done_rx) = mpsc::channel();
        let b_journal = Arc::clone(&journal);
        let b = thread::spawn(move || {
            let bytes = encoded_insert(2, b"B");
            b_done_tx
                .send(b_journal.submit(prepared(&b_journal, &bytes), false))
                .unwrap();
        });
        b_after_reserve.wait();

        let (c_done_tx, c_done_rx) = mpsc::channel();
        let c_journal = Arc::clone(&journal);
        let c = thread::spawn(move || {
            let bytes = encoded_insert(3, b"C");
            let result = c_journal
                .submit(prepared(&c_journal, &bytes), true)
                .and_then(|ack| ack.expect("sync append acknowledgement").wait());
            c_done_tx.send(result).unwrap();
        });
        c_after_publish.wait();
        c_after_publish.release();

        // C is published first but cannot obtain an ACK target while the
        // earlier A/B byte reservations leave a physical gap.
        assert!(c_done_rx.recv_timeout(Duration::from_millis(50)).is_err());

        a_after_reserve.release();
        a_after_publish.wait();
        let deadline = Instant::now() + Duration::from_secs(2);
        while journal.stats().written_work < 1 {
            assert!(Instant::now() < deadline, "A never reached the page cache");
            thread::yield_now();
        }
        assert!(c_done_rx.recv_timeout(Duration::from_millis(50)).is_err());

        // A crash image taken while B is still unpublished contains A only;
        // C has neither ACKed nor become replay-visible.
        std::fs::copy(&path, &crash_image).unwrap();
        assert_eq!(replay_seqs(&crash_image), vec![1]);

        b_after_reserve.release();
        b_done_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        c_done_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        a_after_publish.release();
        a_done_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();

        a.join().unwrap();
        b.join().unwrap();
        c.join().unwrap();
        assert_eq!(replay_seqs(&path), vec![1, 2, 3]);
        let stats = journal.stats();
        assert_eq!(stats.queued_work, 3);
        assert_eq!(stats.written_work, 3);
        assert_eq!(stats.flushed_work, 3);
    }

    #[test]
    fn async_small_append_is_promptly_drained_without_fsync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Journal::open_or_create(&path, 0).unwrap();
        let bytes = encoded_insert(1, b"async");
        journal.submit(prepared(&journal, &bytes), false).unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while journal.stats().written_work != 1 {
            assert!(
                Instant::now() < deadline,
                "async record was not promptly drained"
            );
            thread::yield_now();
        }
        assert_eq!(journal.stats().syncs, 0);
        assert_eq!(journal.stats().flushed_work, 0);
        assert_eq!(replay_seqs(&path), vec![1]);
    }

    #[test]
    fn sync_waiter_wakes_are_coalesced_while_fsync_is_stalled() {
        let dir = tempfile::tempdir().unwrap();
        let journal =
            Arc::new(Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap());
        let before_sync = JournalTestBarrier::new();
        journal.install_before_sync_barrier(Arc::clone(&before_sync));
        let bytes = encoded_insert(1, b"coalesced");
        let ack = journal
            .submit(prepared(&journal, &bytes), true)
            .unwrap()
            .unwrap();

        let ack_waiter = thread::spawn(move || ack.wait());
        before_sync.wait();

        let target = journal.queued_work();
        let mut waiters = Vec::new();
        for _ in 0..32 {
            let journal = Arc::clone(&journal);
            waiters.push(thread::spawn(move || journal.flush_up_to(target)));
        }
        thread::sleep(Duration::from_millis(20));
        assert!(journal.wake_depth_for_test() <= WAKE_CAPACITY);
        assert_eq!(journal.ordered_control_depth_for_test(), 0);

        before_sync.release();
        ack_waiter.join().unwrap().unwrap();
        for waiter in waiters {
            waiter.join().unwrap().unwrap();
        }
        assert_eq!(journal.wake_depth_for_test(), 0);
    }

    #[test]
    fn pending_page_cache_drain_failure_is_wal_append_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Journal::open_or_create(&path, 0).unwrap();
        journal.fail_next_drain_for_test();
        let bytes = encoded_insert(1, b"drain-fail");
        let ack = journal
            .submit(prepared(&journal, &bytes), true)
            .unwrap()
            .unwrap();
        assert_unknown_phase(ack.wait(), CommitPhase::WalAppend);
        assert_eq!(journal.stats().written_work, 0);
        assert!(replay_seqs(&path).is_empty());
    }

    #[test]
    fn copied_records_report_sync_failure_but_later_prefix_reports_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Arc::new(Journal::open_or_create(&path, 0).unwrap());
        let after_snapshot = JournalTestBarrier::new();
        journal.install_after_snapshot_barrier(Arc::clone(&after_snapshot));
        journal.fail_next_sync_for_test();

        let first = encoded_insert(1, b"first");
        let first_ack = journal
            .submit(prepared(&journal, &first), true)
            .unwrap()
            .unwrap();
        let first_waiter = thread::spawn(move || first_ack.wait());
        after_snapshot.wait();

        let second = encoded_insert(2, b"second");
        let second_ack = journal
            .submit(prepared(&journal, &second), true)
            .unwrap()
            .unwrap();
        after_snapshot.release();

        assert_unknown_phase(first_waiter.join().unwrap(), CommitPhase::WalSync);
        assert_unknown_phase(second_ack.wait(), CommitPhase::WalAppend);
        assert_eq!(journal.stats().written_work, 1);
        assert_eq!(replay_seqs(&path), vec![1]);
    }

    #[test]
    fn exact_copied_snapshot_marks_every_included_record_wal_sync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Arc::new(Journal::open_or_create(&path, 0).unwrap());
        let before_drain = JournalTestBarrier::new();
        journal.install_before_drain_barrier(Arc::clone(&before_drain));
        before_drain.wait();
        journal.fail_next_sync_for_test();

        let first = encoded_insert(1, b"first");
        let second = encoded_insert(2, b"second");
        let first_ack = journal
            .submit(prepared(&journal, &first), true)
            .unwrap()
            .unwrap();
        let second_ack = journal
            .submit(prepared(&journal, &second), true)
            .unwrap()
            .unwrap();
        let first_waiter = thread::spawn(move || first_ack.wait());
        let deadline = Instant::now() + Duration::from_secs(2);
        while journal.shared.sync_target.load(Ordering::Acquire) < 1 {
            assert!(Instant::now() < deadline, "sync target was not published");
            thread::yield_now();
        }
        before_drain.release();

        assert_unknown_phase(first_waiter.join().unwrap(), CommitPhase::WalSync);
        assert_unknown_phase(second_ack.wait(), CommitPhase::WalSync);
        assert_eq!(journal.stats().written_work, 2);
        assert_eq!(replay_seqs(&path), vec![1, 2]);
    }
}
