//! Background checkpointer — three threads coordinating
//! through a bounded I/O queue + per-thread stop signals.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │ checkpoint_thread (planner / orchestrator)               │
//! │   park_timeout(idle_interval)                            │
//! │     ├─ run_merge_pass                                    │
//! │     ├─ snapshot_dirty + journal.flush                    │
//! │     ├─ submit one CheckpointEpoch                        │
//! │     ├─ reap completed epochs in FIFO order               │
//! │     └─ journal.truncate iff pipeline empty + clean BM     │
//! └────────┬─────────────────────────────────────────────────┘
//!          │ IoTask (bounded crossbeam channel)
//!          ▼
//! ┌──────────────────────────────────────────────────────────┐
//! │ io_thread (I/O executor)                                 │
//! │   recv IoTask -> write batch / sync / pending deletes     │
//! │                                                          │
//! │   ── Unix:  pread/pwritev through FileBlobStore           │
//! │   ── Linux: io_uring fixed-file/fixed-buffer fast path   │
//! └──────────────────────────────────────────────────────────┘
//!
//! ┌──────────────────────────────────────────────────────────┐
//! │ eviction_thread (independent cadence)                    │
//! │   park_timeout(eviction_interval)                        │
//! │     scan cache, drop cold non-dirty entries              │
//! │     using BufferManager::clock_tick + last_touched       │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Industrial references
//!
//! Mirrors the FractalBit ancestor's three-thread structure
//! (`checkpoint_thread` / `bss_io_thread` / `eviction_thread`).
//! The split-by-responsibility pattern also lines up with:
//!
//! - **sled** (`Flusher` thread) — `Arc<AtomicBool>` shutdown
//!   flag, parked between rounds.
//! - **fjall** (`FlushManager` + queue) — bounded queue between
//!   planner and I/O executor; never trim the journal until the
//!   corresponding flush succeeds.
//! - **LeanStore** — round-driven dirty-set drain on the planner
//!   thread.
//!
//! Three threads (rather than a single one) buy:
//!
//! 1. **Writers don't sit behind checkpoint I/O** — the planner
//!    captures dirty intent under the commit gate, flushes the WAL
//!    watermark, then clones version-matched bytes after releasing
//!    the gate while epoch I/O continues in the worker.
//! 2. **io_uring fit** — the I/O thread is the natural home for
//!    the SQE submit + CQE poll loop on the Linux fast path.
//! 3. **Eviction is decoupled** — runs on its own cadence
//!    against its own clock; doesn't compete with the planner
//!    for BM mutex time.
//!
//! ## Shutdown
//!
//! `Checkpointer::Drop`:
//!
//! 1. Set `checkpoint_stop`; unpark + join the planner thread so
//!    no new rounds start.
//! 2. Run one final synchronous round on the calling thread
//!    (still uses the I/O queue). Closes the window between the
//!    planner's last round and the Tree handle's drop — writes
//!    that landed in that window are otherwise lost when the
//!    BM/journal `Arc`s drop.
//! 3. Send `IoTask::Stop`; join the I/O thread.
//! 4. Set `eviction_stop`; unpark + join the eviction thread.

mod eviction;
mod io;
mod round;

use crossbeam_channel::{bounded, Sender};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::concurrency::{CommitGate, Gate};
use crate::journal::Journal;
use crate::store::BufferManager;

use self::io::IoTask;

// ---------- public config ----------

/// Background checkpointer policy + cadence.
///
/// The defaults enable the background thread group so file-backed
/// trees bound WAL growth and drain dirty blobs without requiring
/// callers to schedule [`crate::Tree::checkpoint`] manually.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Master switch. `true` (the default) spawns the planner, I/O,
    /// and eviction threads on tree open and stops them on tree drop.
    /// Set to `false` when callers want fully manual checkpointing
    /// through [`crate::Tree::checkpoint`].
    pub enabled: bool,
    /// Maximum interval between planner rounds. Smaller values
    /// = lower checkpoint latency, more wake-ups per second.
    ///
    /// Default 500 ms. The background checkpointer stays active
    /// by default, but avoids chasing every short write burst.
    pub idle_interval: Duration,
    /// Trigger an early round when the BufferManager's dirty
    /// blob count reaches this. Heuristic for "the dirty set is
    /// large enough that the next round is worth running before
    /// `idle_interval` elapses".
    ///
    /// Default 512. This bounds dirty growth without forcing an
    /// early checkpoint for a handful of hot blobs.
    pub dirty_blob_threshold: usize,
    /// Drain queued parent-merge candidates at the start of each
    /// round. Parent merge rewrites structural BlobNode edges and
    /// queues old child blobs for deferred manifest deletion, so it
    /// remains opt-in until reachability-safe delete GC is available
    /// for fully asynchronous checkpoint pipelines.
    ///
    /// Default `false`; foreground split/shape control remains
    /// enabled, and callers can still run explicit maintenance.
    pub auto_merge: bool,
    /// How often the eviction thread scans the cache.
    /// Default 1 s.
    pub eviction_interval: Duration,
    /// "Idle tick" threshold for the eviction thread. An entry
    /// whose `last_touched` lags the current BM clock by more
    /// than this is considered cold and evicted (unless dirty).
    ///
    /// Default 1024 — roughly "untouched for the last ~1024
    /// `pin`/`get_cached` operations".
    pub eviction_idle_ticks: u64,
    /// Bounded I/O queue capacity (depth of checkpoint I/O tasks
    /// in flight between the planner and the I/O thread).
    /// Bigger = more parallelism head-room for io_uring; smaller
    /// = tighter back-pressure on the planner.
    ///
    /// Default 16.
    pub io_queue_capacity: usize,
    /// Run store-level vacuum opportunistically after successful
    /// checkpoint epochs once tail-reclaimable space crosses
    /// [`Self::auto_vacuum_min_free_bytes`].
    ///
    /// Vacuum is a physical-space maintenance action only: it relocates
    /// live high-water slots into lower reusable holes, trims packed-file
    /// tails, and, on Linux, punches remaining reusable middle-slot holes.
    /// A failed auto-vacuum logs a warning but does not make the completed
    /// checkpoint fail.
    pub auto_vacuum: bool,
    /// Tail-reclaimable store space required before automatic
    /// vacuum is attempted. Default 256 MiB. Auto-vacuum keys off tail
    /// bytes so it does not repeatedly revisit already-sparse middle holes.
    pub auto_vacuum_min_free_bytes: u64,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_interval: Duration::from_millis(500),
            dirty_blob_threshold: 512,
            auto_merge: false,
            eviction_interval: Duration::from_secs(1),
            eviction_idle_ticks: 1024,
            io_queue_capacity: 16,
            auto_vacuum: true,
            auto_vacuum_min_free_bytes: 256 * 1024 * 1024,
        }
    }
}

impl CheckpointConfig {
    /// Convenience constructor: enabled with default cadence.
    #[must_use]
    pub fn enabled() -> Self {
        Self::default()
    }
}

// ---------- internal shared state ----------

pub(super) struct Shared {
    pub(super) bm: Arc<BufferManager>,
    pub(super) journal: Option<Arc<Journal>>,
    /// Same writer-shared / checkpoint-exclusive publish barrier
    /// used by foreground persistent writers. Checkpoint rounds hold
    /// its exclusive side only while draining dirty intent and
    /// capturing blob content versions. Byte cloning and store I/O
    /// happen after the gate is released.
    pub(super) commit_gate: Arc<CommitGate>,
    /// Shared structural gate with `Tree`: the merge pass enters
    /// the exclusive side so it cannot fold/delete a child blob
    /// while a foreground writer is lock-coupling through that
    /// edge.
    pub(super) maintenance_gate: Arc<Gate>,
    pub(super) cfg: CheckpointConfig,

    /// Submit side of the bounded I/O queue. Cloned by the planner
    /// thread; the receiver lives inside `io::run`.
    pub(super) io_tx: Sender<IoTask>,

    // Per-thread stop signals (independent so Drop can stop them
    // in the right order without racing the queue).
    pub(super) checkpoint_stop: AtomicBool,
    pub(super) eviction_stop: AtomicBool,

    // Telemetry — written by the threads, read by `Checkpointer`
    // accessors.
    pub(super) rounds_attempted: AtomicU64,
    pub(super) rounds_succeeded: AtomicU64,
    pub(super) rounds_failed: AtomicU64,
    pub(super) blobs_flushed: AtomicU64,
    pub(super) merges_total: AtomicU64,
    pub(super) truncates: AtomicU64,
    pub(super) evictions: AtomicU64,
    pub(super) last_dirty_count: AtomicUsize,
    pub(super) last_pending_delete_count: AtomicUsize,
    pub(super) last_round_micros: AtomicU64,
}

// ---------- handle ----------

/// Three-thread checkpointer handle. Dropping the handle signals
/// shutdown and joins all three threads in the documented order.
pub(crate) struct Checkpointer {
    shared: Arc<Shared>,
    checkpoint_handle: Option<JoinHandle<()>>,
    io_handle: Option<JoinHandle<()>>,
    eviction_handle: Option<JoinHandle<()>>,
}

impl Checkpointer {
    /// Spawn the three checkpointer threads bound to `bm` +
    /// optional journal. Returns `None` if `cfg.enabled == false` —
    /// the caller (typically `Tree::open_inner`) falls back to
    /// synchronous checkpointing in that case.
    #[must_use]
    pub(crate) fn spawn(
        bm: Arc<BufferManager>,
        journal: Option<Arc<Journal>>,
        maintenance_gate: Arc<Gate>,
        commit_gate: Arc<CommitGate>,
        cfg: CheckpointConfig,
    ) -> Option<Self> {
        if !cfg.enabled {
            return None;
        }
        let (io_tx, io_rx) = bounded::<IoTask>(cfg.io_queue_capacity.max(1));
        let shared = Arc::new(Shared {
            bm,
            journal,
            commit_gate,
            maintenance_gate,
            cfg,
            io_tx,
            checkpoint_stop: AtomicBool::new(false),
            eviction_stop: AtomicBool::new(false),
            rounds_attempted: AtomicU64::new(0),
            rounds_succeeded: AtomicU64::new(0),
            rounds_failed: AtomicU64::new(0),
            blobs_flushed: AtomicU64::new(0),
            merges_total: AtomicU64::new(0),
            truncates: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            last_dirty_count: AtomicUsize::new(0),
            last_pending_delete_count: AtomicUsize::new(0),
            last_round_micros: AtomicU64::new(0),
        });

        let io_handle = {
            let s = Arc::clone(&shared);
            thread::Builder::new()
                .name("holt-ckpt-io".to_owned())
                .spawn(move || io::run(&s, io_rx))
                .expect("OS rejected thread spawn for holt-ckpt-io")
        };
        let checkpoint_handle = {
            let s = Arc::clone(&shared);
            thread::Builder::new()
                .name("holt-ckpt-planner".to_owned())
                .spawn(move || checkpoint_main(&s))
                .expect("OS rejected thread spawn for holt-ckpt-planner")
        };
        let eviction_handle = {
            let s = Arc::clone(&shared);
            thread::Builder::new()
                .name("holt-ckpt-eviction".to_owned())
                .spawn(move || eviction::run(&s))
                .expect("OS rejected thread spawn for holt-ckpt-eviction")
        };

        Some(Self {
            shared,
            checkpoint_handle: Some(checkpoint_handle),
            io_handle: Some(io_handle),
            eviction_handle: Some(eviction_handle),
        })
    }

    /// Unpark the planner so it runs a round at the next park
    /// boundary (without waiting out the remainder of
    /// `idle_interval`). Safe to call from any thread; no-op if
    /// the planner is already running.
    ///
    /// Test hook for waking the planner without waiting out the
    /// remainder of `idle_interval`.
    #[cfg(test)]
    pub(crate) fn wake(&self) {
        if let Some(h) = &self.checkpoint_handle {
            h.thread().unpark();
        }
    }

    /// Number of rounds the planner has attempted.
    #[must_use]
    pub(crate) fn rounds_attempted(&self) -> u64 {
        self.shared.rounds_attempted.load(Ordering::Relaxed)
    }

    /// Number of rounds that completed without error.
    #[must_use]
    pub(crate) fn rounds_succeeded(&self) -> u64 {
        self.shared.rounds_succeeded.load(Ordering::Relaxed)
    }

    /// Number of failed rounds or failed submitted epochs.
    #[must_use]
    pub(crate) fn rounds_failed(&self) -> u64 {
        self.shared.rounds_failed.load(Ordering::Relaxed)
    }

    /// Total blobs flushed across all rounds.
    #[must_use]
    pub(crate) fn blobs_flushed(&self) -> u64 {
        self.shared.blobs_flushed.load(Ordering::Relaxed)
    }

    /// WAL truncates performed across all rounds.
    #[must_use]
    pub(crate) fn truncates(&self) -> u64 {
        self.shared.truncates.load(Ordering::Relaxed)
    }

    /// `BlobNode` crossings folded back into parents.
    #[must_use]
    pub(crate) fn merges_total(&self) -> u64 {
        self.shared.merges_total.load(Ordering::Relaxed)
    }

    /// Cache entries evicted by the eviction thread.
    #[must_use]
    pub(crate) fn evictions(&self) -> u64 {
        self.shared.evictions.load(Ordering::Relaxed)
    }

    /// Dirty blobs observed by the most recent planner round.
    #[must_use]
    pub(crate) fn last_dirty_count(&self) -> usize {
        self.shared.last_dirty_count.load(Ordering::Relaxed)
    }

    /// Pending deletes observed by the most recent planner round.
    #[must_use]
    pub(crate) fn last_pending_delete_count(&self) -> usize {
        self.shared
            .last_pending_delete_count
            .load(Ordering::Relaxed)
    }

    /// Wall-clock time spent in the most recent planner round.
    #[must_use]
    pub(crate) fn last_round_micros(&self) -> u64 {
        self.shared.last_round_micros.load(Ordering::Relaxed)
    }
}

impl Drop for Checkpointer {
    fn drop(&mut self) {
        // 1. Stop the planner so no new rounds start.
        self.shared.checkpoint_stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.checkpoint_handle.take() {
            h.thread().unpark();
            let _ = h.join();
        }

        // 2. Run one final synchronous round on this thread.
        //    The planner is joined, writers are gone (last Tree
        //    clone is dropping). I/O thread is still alive
        //    serving the round's submissions.
        if let Err(e) = round::run_round_sync(&self.shared) {
            eprintln!("holt: final checkpoint round during shutdown failed: {e}");
        }

        // 3. Stop the I/O thread.
        let _ = self.shared.io_tx.send(IoTask::Stop);
        if let Some(h) = self.io_handle.take() {
            let _ = h.join();
        }

        // 4. Stop the eviction thread.
        self.shared.eviction_stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.eviction_handle.take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

// ---------- checkpoint_thread main loop ----------

fn checkpoint_main(shared: &Arc<Shared>) {
    let mut pipeline = round::Pipeline::new(shared.cfg.io_queue_capacity);
    loop {
        if shared.checkpoint_stop.load(Ordering::Acquire) {
            break;
        }
        if let Err(e) = pipeline.reap_ready(shared) {
            eprintln!("holt: checkpoint epoch failed: {e}");
        }
        let has_pressure = shared.bm.dirty_count() >= shared.cfg.dirty_blob_threshold.max(1)
            || shared.bm.pending_delete_count() != 0;
        if !has_pressure {
            if !pipeline.is_empty() {
                thread::park_timeout(shared.cfg.idle_interval.min(Duration::from_millis(1)));
                continue;
            }
            thread::park_timeout(shared.cfg.idle_interval);
        }
        if shared.checkpoint_stop.load(Ordering::Acquire) {
            break;
        }
        loop {
            if let Err(e) = round::run_round(shared, &mut pipeline) {
                // Round failed; restored dirty entries (where
                // applicable) are still in the map for the next try.
                eprintln!("holt: checkpoint round failed: {e}");
                break;
            }
            if !pipeline.has_room()
                || shared.bm.dirty_count() < shared.cfg.dirty_blob_threshold.max(1)
                || shared.checkpoint_stop.load(Ordering::Acquire)
            {
                break;
            }
        }
    }
    if let Err(e) = pipeline.drain(shared) {
        eprintln!("holt: checkpoint pipeline drain during shutdown failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::errors::Result;
    use crate::layout::BlobGuid;
    use crate::store::blob_store::{AlignedBlobBuf, BlobStore, MemoryBlobStore};
    use crate::store::BlobFrame;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Instant;

    fn make_bm() -> Arc<BufferManager> {
        Arc::new(BufferManager::new(Arc::new(MemoryBlobStore::new()), 8))
    }

    fn test_blob(guid: BlobGuid) -> AlignedBlobBuf {
        let mut buf = AlignedBlobBuf::zeroed();
        let _frame = BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
        buf
    }

    fn maintenance_gate() -> Arc<Gate> {
        Arc::new(Gate::new())
    }

    fn commit_gate() -> Arc<CommitGate> {
        Arc::new(CommitGate::new())
    }

    struct BlockingBatchStore {
        inner: MemoryBlobStore,
        release: AtomicBool,
        started: AtomicUsize,
    }

    impl BlockingBatchStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                release: AtomicBool::new(false),
                started: AtomicUsize::new(0),
            }
        }

        fn release(&self) {
            self.release.store(true, AtomicOrdering::Release);
        }

        fn started(&self) -> usize {
            self.started.load(AtomicOrdering::Acquire)
        }
    }

    impl BlobStore for BlockingBatchStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            if !writes.is_empty() {
                self.started.fetch_add(1, AtomicOrdering::AcqRel);
                while !self.release.load(AtomicOrdering::Acquire) {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            self.inner.write_blobs_with_data_sync(writes)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.inner.delete_blob(guid)
        }

        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            self.inner.list_blobs()
        }

        fn flush(&self) -> Result<()> {
            self.inner.flush()
        }

        fn needs_flush(&self) -> bool {
            self.inner.needs_flush()
        }
    }

    /// A store whose `needs_flush()` is gated by an explicit flag, so a test
    /// can hold "store durability still pending" true while the BufferManager
    /// dirty/flushing/pending counters all read 0 — the exact window in which
    /// the I/O worker has retired a written-through blob (after `pwrite`) but
    /// not yet run the data fsync + manifest persist.
    struct FlushGateStore {
        inner: MemoryBlobStore,
        flush_pending: AtomicBool,
    }

    impl FlushGateStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                flush_pending: AtomicBool::new(true),
            }
        }

        fn release_flush(&self) {
            self.flush_pending.store(false, AtomicOrdering::Release);
        }
    }

    impl BlobStore for FlushGateStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }
        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }
        fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            self.inner.write_blobs_with_data_sync(writes)
        }
        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.inner.delete_blob(guid)
        }
        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            self.inner.list_blobs()
        }
        fn flush(&self) -> Result<()> {
            // Deliberately does NOT clear `flush_pending`; the test controls
            // when the store reports durable so the truncate gate is exercised.
            self.inner.flush()
        }
        fn needs_flush(&self) -> bool {
            self.flush_pending.load(AtomicOrdering::Acquire)
        }
    }

    /// Regression for the nightly crash-soak durability bug: `maybe_truncate`
    /// must NOT truncate the WAL while the store still owes a flush (data
    /// fsync / manifest-delta persist), even though dirty/flushing/pending all
    /// read 0. Truncating there drops the only recoverable copy of an
    /// acknowledged write across a crash. (On the pre-fix gate this test
    /// truncates immediately once the dirty blob is written through.)
    #[test]
    fn truncate_waits_for_store_flush() {
        let store = Arc::new(FlushGateStore::new()); // needs_flush() == true
        let bm = Arc::new(BufferManager::new(store.clone(), 8));

        let dir = tempfile::tempdir().unwrap();
        let journal = Arc::new(Journal::open_or_create(&dir.path().join("wal.log"), 0).unwrap());
        journal.submit(vec![1, 2, 3, 4], false).unwrap(); // -> needs_checkpoint()
        assert!(journal.needs_checkpoint());

        bm.write_blob([0x55; 16], &test_blob([0x55; 16])).unwrap();
        let _pin = bm.pin([0x55; 16]).unwrap();
        bm.mark_dirty([0x55; 16], 1);

        let ck = Checkpointer::spawn(
            Arc::clone(&bm),
            Some(Arc::clone(&journal)),
            maintenance_gate(),
            commit_gate(),
            no_merge_cfg(),
        )
        .expect("spawn");

        // Wait until the dirty blob is written through (dirty retired).
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if bm.dirty_count() == 0 && ck.blobs_flushed() >= 1 {
                break;
            }
            assert!(Instant::now() < deadline, "checkpoint never drained dirty");
            thread::sleep(Duration::from_millis(5));
        }

        // dirty/flushing/pending are all 0 now, but the store still needs a
        // flush — the WAL must stay intact across several truncate attempts.
        thread::sleep(Duration::from_millis(60));
        assert_eq!(
            ck.truncates(),
            0,
            "WAL truncated while store flush was still pending"
        );

        // Once the store reports durable, truncate is allowed to proceed
        // (no deadlock / unbounded WAL growth).
        store.release_flush();
        ck.wake();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if ck.truncates() >= 1 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "WAL never truncated after the store flush completed"
            );
            thread::sleep(Duration::from_millis(5));
        }
        drop(ck);
    }

    /// Tests that don't construct a real Tree skip the merge pass —
    /// `collect_blob_guids` would otherwise try to pin a
    /// non-existent root.
    fn no_merge_cfg() -> CheckpointConfig {
        CheckpointConfig {
            auto_merge: false,
            ..CheckpointConfig::enabled()
        }
    }

    #[test]
    fn disabled_config_spawns_nothing() {
        let bm = make_bm();
        let cfg = CheckpointConfig {
            enabled: false,
            ..CheckpointConfig::default()
        };
        assert!(!cfg.enabled);
        let ck = Checkpointer::spawn(bm, None, maintenance_gate(), commit_gate(), cfg);
        assert!(ck.is_none());
    }

    #[test]
    fn spawn_and_drop_is_leak_free() {
        let bm = make_bm();
        let ck = Checkpointer::spawn(bm, None, maintenance_gate(), commit_gate(), no_merge_cfg())
            .expect("spawn");
        // Give threads a tick to actually park.
        thread::sleep(Duration::from_millis(50));
        drop(ck);
        // If shutdown deadlocked, this test would hang on the test
        // harness's per-test timeout.
    }

    #[test]
    fn round_drains_dirty_set_via_io_queue() {
        let bm = make_bm();
        // Prime a cached entry so snapshot_bytes returns Some.
        let mut scratch = test_blob([0x42; 16]);
        scratch.as_mut_slice()[100] = 0xAB;
        bm.write_blob([0x42; 16], &scratch).unwrap();
        let _pin = bm.pin([0x42; 16]).unwrap();
        bm.mark_dirty([0x42; 16], 10);
        assert_eq!(bm.dirty_count(), 1);

        let ck = Checkpointer::spawn(
            Arc::clone(&bm),
            None,
            maintenance_gate(),
            commit_gate(),
            no_merge_cfg(),
        )
        .expect("spawn");
        // Wait for at least one round to drain the dirty set.
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if bm.dirty_count() == 0 && ck.blobs_flushed() >= 1 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "checkpoint round didn't drain dirty in time"
            );
            thread::sleep(Duration::from_millis(10));
        }
        drop(ck);
    }

    #[test]
    fn wake_short_circuits_idle_wait() {
        let bm = make_bm();
        let mut cfg = no_merge_cfg();
        cfg.idle_interval = Duration::from_secs(10);
        let ck = Checkpointer::spawn(
            Arc::clone(&bm),
            None,
            maintenance_gate(),
            commit_gate(),
            cfg,
        )
        .expect("spawn");

        // Need a cached blob so snapshot_bytes finds it.
        let scratch = test_blob([0x01; 16]);
        bm.write_blob([0x01; 16], &scratch).unwrap();
        let _pin = bm.pin([0x01; 16]).unwrap();
        bm.mark_dirty([0x01; 16], 1);
        ck.wake();

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if ck.rounds_succeeded() >= 1 && bm.dirty_count() == 0 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "checkpointer never drained dirty set after wake"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn planner_queues_second_epoch_while_first_io_is_blocked() {
        let store = Arc::new(BlockingBatchStore::new());
        let bm = Arc::new(BufferManager::new(store.clone(), 8));
        bm.write_blob([0x21; 16], &test_blob([0x21; 16])).unwrap();
        bm.write_blob([0x22; 16], &test_blob([0x22; 16])).unwrap();
        let _pin1 = bm.pin([0x21; 16]).unwrap();
        let _pin2 = bm.pin([0x22; 16]).unwrap();
        bm.mark_dirty([0x21; 16], 1);

        let mut cfg = no_merge_cfg();
        cfg.idle_interval = Duration::from_millis(10);
        cfg.dirty_blob_threshold = 1;
        cfg.io_queue_capacity = 2;
        let ck = Checkpointer::spawn(
            Arc::clone(&bm),
            None,
            maintenance_gate(),
            commit_gate(),
            cfg,
        )
        .expect("spawn");

        let deadline = Instant::now() + Duration::from_secs(2);
        while store.started() == 0 {
            assert!(
                Instant::now() < deadline,
                "first checkpoint epoch never entered store write"
            );
            thread::sleep(Duration::from_millis(5));
        }

        bm.mark_dirty([0x22; 16], 2);
        ck.wake();

        let mut queued_second = false;
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if ck.rounds_attempted() >= 2 {
                queued_second = true;
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        store.release();
        assert!(
            queued_second,
            "planner did not submit a second epoch while first I/O was blocked"
        );

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if bm.dirty_count() == 0 && ck.blobs_flushed() >= 2 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "queued checkpoint epochs did not drain"
            );
            thread::sleep(Duration::from_millis(5));
        }
        drop(ck);
    }

    #[test]
    fn eviction_thread_drops_cold_entries_only_under_capacity_pressure() {
        let bm = Arc::new(BufferManager::new(Arc::new(MemoryBlobStore::new()), 1));
        // Prime two cached entries and let the first go cold. With
        // capacity=1, exactly one cold entry may be evicted.
        let scratch = crate::store::blob_store::AlignedBlobBuf::zeroed();
        bm.write_blob([0xEE; 16], &scratch).unwrap();
        let cold = bm.pin([0xEE; 16]).unwrap();
        bm.write_blob([0xEF; 16], &scratch).unwrap();
        let _hot = bm.pin([0xEF; 16]).unwrap();
        // Drop the pin so try_evict_cold sees strong_count == 1.
        drop(cold);
        assert_eq!(bm.cached_count(), 2);

        // Bump the clock past the eviction threshold by hitting
        // get_cached for some other GUID a bunch of times.
        for _ in 0..5 {
            let _ = bm.pin([0xFF; 16]); // doesn't exist; just a tick advance
            let _ = bm.cached_count();
        }

        let cfg = CheckpointConfig {
            // Aggressive eviction for this test.
            eviction_interval: Duration::from_millis(20),
            eviction_idle_ticks: 1, // immediately stale after one tick advance
            ..no_merge_cfg()
        };
        let ck = Checkpointer::spawn(
            Arc::clone(&bm),
            None,
            maintenance_gate(),
            commit_gate(),
            cfg,
        )
        .expect("spawn");

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if ck.evictions() >= 1 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "eviction thread didn't drop a cold entry in time"
            );
            thread::sleep(Duration::from_millis(20));
        }
        drop(ck);
    }

    #[test]
    fn eviction_thread_keeps_cold_entries_when_cache_fits() {
        let bm = make_bm();
        let scratch = crate::store::blob_store::AlignedBlobBuf::zeroed();
        bm.write_blob([0xEE; 16], &scratch).unwrap();
        let _ = bm.pin([0xEE; 16]).unwrap();
        assert_eq!(bm.cached_count(), 1);

        for _ in 0..5 {
            let _ = bm.pin([0xFF; 16]);
            let _ = bm.cached_count();
        }

        let cfg = CheckpointConfig {
            eviction_interval: Duration::from_millis(20),
            eviction_idle_ticks: 1,
            ..no_merge_cfg()
        };
        let ck = Checkpointer::spawn(
            Arc::clone(&bm),
            None,
            maintenance_gate(),
            commit_gate(),
            cfg,
        )
        .expect("spawn");

        thread::sleep(Duration::from_millis(120));
        assert_eq!(ck.evictions(), 0);
        assert_eq!(bm.cached_count(), 1);
        drop(ck);
    }
}
