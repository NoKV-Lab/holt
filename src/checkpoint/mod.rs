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
//! │     ├─ submit one IoTask::FlushBatch for dirty blobs     │
//! │     ├─ wait completions                                  │
//! │     ├─ submit IoTask::Sync, wait                         │
//! │     └─ journal.truncate iff dirty_count == 0             │
//! └────────┬─────────────────────────────────────────────────┘
//!          │ IoTask (bounded crossbeam channel)
//!          ▼
//! ┌──────────────────────────────────────────────────────────┐
//! │ io_thread (I/O executor)                                 │
//! │   recv IoTask -> backend.write_blob / backend.flush      │
//! │                                                          │
//! │   ── Current: sync pread/pwrite via PersistentBackend    │
//! │   ── Next:    io_uring submit + CQE poll (feature flag)  │
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
//! 1. **Planner doesn't block on I/O** — once the queue has
//!    capacity, the planner can prepare the next merge / snapshot
//!    while previous tasks are still on the wire.
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

use crate::concurrency::{CommitGate, MaintenanceGate};
use crate::journal::group_commit::Journal;
use crate::store::BufferManager;

use self::io::IoTask;

// ---------- public config ----------

/// Background checkpointer policy + cadence.
///
/// The defaults leave the background thread group **disabled**
/// (`enabled = false`); flip it on with [`CheckpointConfig::enabled`]
/// or via [`crate::TreeBuilder::checkpoint`].
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Master switch. `false` (the default) leaves checkpointing
    /// fully synchronous — callers drive it via
    /// [`crate::Tree::checkpoint`]. `true` spawns the three
    /// background threads on tree open and stops them on tree
    /// drop.
    pub enabled: bool,
    /// Maximum interval between planner rounds. Smaller values
    /// = lower checkpoint latency, more wake-ups per second.
    ///
    /// **Default 100 ms** — tuned via the `bench_checkpoint_sweep`
    /// integration bench. The sweep showed:
    ///
    /// | interval | peak WAL (paced, vs disabled) | writer overhead |
    /// |---:|---:|---:|
    /// |   50 ms | 0%   | unchanged |
    /// |  100 ms | 23%  | unchanged |
    /// |  200 ms | 47%  | unchanged |
    /// |  500 ms | 54%  | unchanged |
    /// | 1000 ms | 100% | unchanged |
    ///
    /// 100 ms keeps the WAL bounded (4× tighter than 200 ms) at
    /// the cost of 10 wake-ups/sec; tighter intervals see
    /// diminishing returns on peak WAL but double the wake-up
    /// rate. Tune up for low-write workloads where the extra
    /// wake-ups cost more than the WAL bytes they save.
    pub idle_interval: Duration,
    /// Trigger an early round when the BufferManager's dirty
    /// blob count reaches this. Heuristic for "the dirty set is
    /// large enough that the next round is worth running before
    /// `idle_interval` elapses".
    ///
    /// Default 16 — chosen for "many child-blob trees but not
    /// pathological". For a single-root workload the dirty
    /// count is usually ≤ 1, so the timer dominates.
    pub dirty_blob_threshold: usize,
    /// Drain queued parent-merge candidates at the start of each
    /// round. The queue is populated by foreground spillovers and
    /// manual maintenance seeding, so idle rounds avoid walking
    /// every reachable blob just to rediscover that nothing can
    /// merge.
    ///
    /// Default `true` — keeping the blob tree in equilibrium
    /// against split/merge pressure is the whole point.
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
    /// Bounded I/O queue capacity (depth of `FlushBatch` / `Sync`
    /// tasks in flight between the planner and the I/O thread).
    /// Bigger = more parallelism head-room for io_uring; smaller
    /// = tighter back-pressure on the planner.
    ///
    /// Default 16.
    pub io_queue_capacity: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_interval: Duration::from_millis(100),
            dirty_blob_threshold: 16,
            auto_merge: true,
            eviction_interval: Duration::from_secs(1),
            eviction_idle_ticks: 1024,
            io_queue_capacity: 16,
        }
    }
}

impl CheckpointConfig {
    /// Convenience constructor: enabled with default cadence.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
}

// ---------- internal shared state ----------

pub(super) struct Shared {
    pub(super) bm: Arc<BufferManager>,
    pub(super) journal: Option<Arc<Journal>>,
    /// Same writer-shared / checkpoint-exclusive publish barrier
    /// used by foreground persistent writers. Checkpoint rounds
    /// hold its exclusive side while draining dirty entries and
    /// cloning bytes so backend writes never include unjournaled
    /// mutations.
    pub(super) commit_gate: Arc<CommitGate>,
    /// Shared structural gate with `Tree`: the merge pass enters
    /// the exclusive side so it cannot fold/delete a child blob
    /// while a foreground writer is lock-coupling through that
    /// edge.
    pub(super) maintenance_gate: Arc<MaintenanceGate>,
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
    pub(super) blobs_flushed: AtomicU64,
    pub(super) merges_total: AtomicU64,
    pub(super) truncates: AtomicU64,
    pub(super) evictions: AtomicU64,
    pub(super) last_dirty_count: AtomicUsize,
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
        maintenance_gate: Arc<MaintenanceGate>,
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
            blobs_flushed: AtomicU64::new(0),
            merges_total: AtomicU64::new(0),
            truncates: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            last_dirty_count: AtomicUsize::new(0),
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
        if let Err(e) = round::run_round(&self.shared) {
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
    loop {
        if shared.checkpoint_stop.load(Ordering::Acquire) {
            break;
        }
        thread::park_timeout(shared.cfg.idle_interval);
        if shared.checkpoint_stop.load(Ordering::Acquire) {
            break;
        }
        if let Err(e) = round::run_round(shared) {
            // Round failed; restored dirty entries (where
            // applicable) are still in the map for the next try.
            eprintln!("holt: checkpoint round failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::backend::{Backend, MemoryBackend};
    use std::time::Instant;

    fn make_bm() -> Arc<BufferManager> {
        Arc::new(BufferManager::new(Arc::new(MemoryBackend::new()), 8))
    }

    fn maintenance_gate() -> Arc<MaintenanceGate> {
        Arc::new(MaintenanceGate::new())
    }

    fn commit_gate() -> Arc<CommitGate> {
        Arc::new(CommitGate::new())
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
        let cfg = CheckpointConfig::default();
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
        let mut scratch = crate::store::backend::AlignedBlobBuf::zeroed();
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
        let scratch = crate::store::backend::AlignedBlobBuf::zeroed();
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
    fn eviction_thread_drops_cold_entries() {
        let bm = make_bm();
        // Prime a cached entry and let it go cold.
        let scratch = crate::store::backend::AlignedBlobBuf::zeroed();
        bm.write_blob([0xEE; 16], &scratch).unwrap();
        let _ = bm.pin([0xEE; 16]).unwrap();
        // Drop the pin so try_evict_cold sees strong_count == 1.
        // (The `_` binding drops at end of statement.)
        assert_eq!(bm.cached_count(), 1);

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
}
