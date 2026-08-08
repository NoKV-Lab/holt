//! `BufferManager` — frequency-aware blob cache.
//!
//! Sits between a [`Tree`](crate::Tree) and its underlying
//! [`BlobStore`]. Itself implements `BlobStore`, so it's a transparent
//! drop-in: callers see the same `read_blob` / `write_blob` /
//! `flush` API, but reads of recently-touched blobs hit the cache
//! and skip the inner store's I/O.
//!
//! ## Write protocol — staged through `dirty` + `pending_deletes`
//!
//! The walker mutates blobs via [`CachedBlob::write`] guards;
//! those edits stay in cache until a flush pushes them through.
//! Two flush paths exist:
//!
//! - **Synchronous checkpoint** — [`crate::Tree::checkpoint`]
//!   drains the dirty map, clones cached bytes, then calls
//!   [`BufferManager::write_through_batch`] with the snapshotted seqs.
//! - **Background checkpointer** — drives the same protocol from
//!   its planner/I/O threads; see [`BufferManager::snapshot_dirty`]
//!   / [`BufferManager::restore_dirty`].
//!
//! The `write_blob` trait method is still write-through (cache +
//! store in one call). Internal call sites that produce a new
//! blob (spillover) or unlink one (erase's `SubtreeGone`) goes through
//! [`BufferManager::install_new_blob`] / [`BufferManager::mark_for_delete`].
//! Structural merge instead stages the detached child against its exact
//! rewritten parent until a clean durable frontier can reclaim it. Store
//! writes and manifest mutations therefore stay behind invariant **W2D**.
//!
//! ## Dirty tracking + deferred deletes
//!
//! Every walker write tags its target blob via
//! [`BufferManager::mark_dirty`] with the WAL seq that authored
//! the change. The internal dirty state keeps the **lowest**
//! unflushed seq per blob — that value is the WAL trim watermark
//! for that blob (records below it are already in store, so the
//! WAL doesn't need them). A checkpoint round moves drained
//! entries into an in-flight `flushing` set until their cached
//! bytes have reached the store; eviction treats both maps as
//! protected.
//!
//! Erase ops that empty a child blob queue a deferred deletion
//! via [`BufferManager::mark_for_delete`] — the `store.delete_blob`
//! syscall runs only after the corresponding WAL record is on
//! disk. A checkpoint round moves queued deletes into an in-flight
//! delete-fence state while the I/O worker owns them; the fence
//! still hides the blob from stale pins until the manifest delete
//! has completed or the round restores the work.
//!
//! Invariants:
//!
//! - **I1**: a `(guid, _)` entry exists in `dirty` iff the cached
//!   image of `guid` is newer than the store image.
//! - **I2**: WAL `trim_id <= min(dirty.values()) - 1` (or
//!   `next_seq - 1` if `dirty` is empty).
//! - **I3**: [`BufferManager::snapshot_dirty`] drains the map
//!   atomically, so `mark_dirty` calls that race with a checkpoint
//!   round land in the new (empty) map and are tracked for the
//!   next round. [`BufferManager::snapshot_pending_deletes`]
//!   drains queued work into an in-flight delete fence rather than
//!   making the blob visible again.
//! - **W2D**: any byte written to `store.data_file` or any
//!   manifest mutation persisted to disk must have its
//!   corresponding WAL record durably on disk first.
//!
//! ## Per-blob locking — 3-mode `HybridLatch`
//!
//! Each cached blob lives behind a `HybridLatch` (LeanStore-style
//! 3-mode latch) wrapping an `UnsafeCell<AlignedBlobBuf>`:
//!
//! - **Optimistic** — wait-free. Snapshot the latch version, read
//!   the buffer without a real lock, then `validate()` afterwards.
//!   If a writer lapped the snapshot, the read is discarded and
//!   the caller restarts. Used by `Tree::get`'s walker.
//! - **Shared** — N readers run concurrently, mutually exclusive
//!   with writers. Checkpoint byte snapshots take this mode long
//!   enough to clone the cached image.
//! - **Exclusive** — single writer, mutually exclusive with all
//!   readers. Used by every walker mutation hop (`insert_multi`
//!   / `erase_multi` / spillover).
//!
//! ## Pin-and-operate
//!
//! Callers that want to operate on a blob without an intervening
//! 512 KB memcpy use [`BufferManager::pin`] — it returns an
//! `Arc<CachedBlob>` holding the buffer alive in cache. The
//! `Arc`'s strong count keeps eviction at bay. From there:
//!
//! - [`CachedBlob::read_optimistic`] → wait-free [`OptimisticGuard`]
//!   with `as_slice()` + `validate()`. Wrap with
//!   `BlobFrameRef::wrap(guard.as_slice())` for zero-copy traversal.
//! - [`CachedBlob::read`] → [`BlobReadGuard`] (shared). Same
//!   `BlobFrameRef::wrap` shape, but blocks behind any active writer.
//! - [`CachedBlob::write`] → [`BlobWriteGuard`] (exclusive). Use
//!   `guard.frame()` for in-place mutation. The owning tree later
//!   publishes dirty state and checkpoint writes it through via
//!   [`BufferManager::write_through_batch`].
//!
//! ## Eviction
//!
//! Two paths drop cold cache entries:
//!
//! - **Inline overflow** ([`Self::try_evict_for_point_insert`]) — fires inside
//!   [`Self::insert_into_cache`] when the new entry pushes the
//!   cache past `capacity`. Point inserts use a TinyLFU-style
//!   sketch to prefer evicting one-hit leaf blobs over frequently
//!   reused metadata blobs, with `last_touched` as the tie-breaker.
//! - **Background sweep** ([`crate::checkpoint`] eviction
//!   thread) — periodic overflow trim for entries that were still
//!   pinned during inline eviction. It uses the same
//!   `last_touched` threshold but only runs while cache size is
//!   above `capacity`.
//!
//! The cache may temporarily exceed `capacity` while every entry
//! is pinned; it shrinks back as readers drop their handles or
//! the background sweep catches up.
//!
//! ## Concurrent sharding
//!
//! The cache is a [`DashMap`] (sharded concurrent `HashMap`) so
//! `pin` / `get_cached` calls on different blobs hit different
//! shards — no single global mutex on the hot read path. The
//! sharded cache + tick-based eviction together replace what
//! would otherwise be a per-blob bottleneck on multi-threaded
//! workloads.

mod admission;
mod cached_blob;
mod guid_hash;
mod mutation;
mod residency;
mod telemetry;
mod write_delta;

use std::collections::{hash_map::Entry, BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use dashmap::DashMap;

use crate::api::stats::{StoreStats, VacuumStats};
use guid_hash::GuidBuildHasher;

use crate::api::errors::{Error, Result};
use crate::engine;
use crate::layout::{BlobGuid, HEADER_SIZE, PAGE_SIZE};
use crate::store::{BlobFrameRef, PAGE_4K};

use super::blob_store::{AlignedBlobBuf, BlobStore, FileStoreObjectIdentity};
use super::read_index::{ReadIndex, ReadIndexCache, ReadIndexStamp, ReadPageCache};

use admission::TinyLFU;
pub use cached_blob::{BlobWriteGuard, CachedBlob};
use mutation::{
    bookkeeping_shard_idx, pop_candidate_batch, CandidateKind, MutationState, BOOKKEEPING_SHARDS,
};
use residency::RouteResidency;
use telemetry::Telemetry;
use write_delta::{DeltaEntry, DeltaOp, WriteDelta};

pub(crate) use write_delta::DeltaEntry as WriteDeltaEntry;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WriteDeltaKeyState {
    Put { seq: u64 },
    Delete,
}

/// Sentinel seq for dirty / pending-delete entries that originate
/// from purely structural mutations (compact, merge pass) — they
/// have no corresponding WAL record and so must not pin the WAL
/// trim watermark. `min(dirty.values())` is what gates the
/// watermark; using `u64::MAX` ensures a structural entry only
/// matters for trim decisions if no real WAL-seqed entry is
/// present alongside it (in which case dirty is non-empty and
/// the truncate gate already refuses to fire).
pub const STRUCTURAL_SEQ: u64 = u64::MAX;

/// Live copy-on-write snapshot bookkeeping, behind one mutex so epoch
/// registration, retirement, and orphan recording stay consistent.
#[derive(Default)]
struct SnapshotState {
    /// Epoch → snapshot root for every live snapshot. The weak pin can
    /// be upgraded under the registry lock so GC holds a stable root while
    /// a last derived view/cursor concurrently releases its epoch lease.
    live: BTreeMap<u64, SnapshotRoot>,
    /// COW detachments grouped by the parent whose edge was repointed.
    /// A child stays non-reclaimable here until that exact parent publishes
    /// dirty/flushing debt; this closes the last-lease-drop window between
    /// edge mutation and `mark_dirty(parent)`.
    cow_pending: HashMap<BlobGuid, Vec<(BlobGuid, u64)>>,
    /// Frames forked away from the live tree, tagged with the
    /// `created_epoch` of the forked-away version. Once the fork barrier
    /// drops below the tag no live snapshot can reference it. Retirement only
    /// moves the exact GUID into the reclaim FIFO; physical deletion waits for
    /// a clean durable checkpoint frontier or a full reachability sweep.
    orphans: Vec<(BlobGuid, u64)>,
    /// Structural detachments grouped by their rewritten parent. These do
    /// not install a visibility fence because a stable snapshot may still
    /// lazily pin the old child. Parent dirty publication promotes them to
    /// `structural_orphans` or the reclaim FIFO.
    structural_pending: HashMap<BlobGuid, HashSet<BlobGuid>>,
    /// Children detached by structural merge while at least one snapshot
    /// lease is live. Unlike a COW fork, a merge has no per-snapshot epoch
    /// tag: any copied root may still point at the detached child. Hold the
    /// exact GUID until the last lease retires, then move it to the same
    /// post-checkpoint FIFO as ordinary retired COW frames.
    structural_orphans: HashSet<BlobGuid>,
    /// Copy-on-write and structural frames eligible for exact reclaim after
    /// their final snapshot epoch has retired. A clean checkpoint drains this
    /// FIFO in bounded batches without an all-store reachability walk; pinned
    /// candidates are restored for a later pass.
    retired_orphans: VecDeque<BlobGuid>,
    retired_orphan_set: HashSet<BlobGuid>,
}

struct SnapshotRoot {
    guid: BlobGuid,
    pin: Weak<CachedBlob>,
}

/// Shared lifetime token for one copy-on-write snapshot epoch.
///
/// Every `View`, range builder, and owned cursor derived from a snapshot
/// carries this token. The epoch retires only after the final derived read
/// handle drops, preventing GC from collecting descendants that an escaped
/// handle can still reach.
pub(crate) struct SnapshotLease {
    store: Arc<BufferManager>,
    epoch: u64,
    root_guid: BlobGuid,
    // Keeps the ephemeral root resident for the full lease lifetime. GC
    // upgrades the registry's weak pin while holding the registry lock.
    root_pin: Option<Arc<CachedBlob>>,
}

impl SnapshotLease {
    pub(crate) fn new(
        store: Arc<BufferManager>,
        epoch: u64,
        root_guid: BlobGuid,
        root_pin: Arc<CachedBlob>,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            epoch,
            root_guid,
            root_pin: Some(root_pin),
        })
    }
}

impl Drop for SnapshotLease {
    fn drop(&mut self) {
        self.store.retire_snapshot(self.epoch);
        drop(self.root_pin.take());
        self.store.discard_snapshot_root(self.root_guid);
    }
}

/// Strong snapshot-root pin owned by one GC reachability pass.
pub(crate) struct PinnedSnapshotRoot {
    store: Arc<BufferManager>,
    guid: BlobGuid,
    pin: Option<Arc<CachedBlob>>,
}

impl PinnedSnapshotRoot {
    #[must_use]
    pub(crate) fn guid(&self) -> BlobGuid {
        self.guid
    }
}

impl Drop for PinnedSnapshotRoot {
    fn drop(&mut self) {
        drop(self.pin.take());
        self.store.discard_snapshot_root(self.guid);
    }
}

/// One pre-snapshotted blob image ready for checkpoint write-through.
///
/// The bytes are owned by the checkpoint round / I/O task so the
/// store write never holds a cache read guard. `expected_seq` is
/// the dirty-map value that was drained into `flushing`; successful
/// batch writes retire that exact flushing entry without stomping a
/// racing writer's newer dirty entry.
pub(crate) struct WriteThroughEntry {
    pub(crate) guid: BlobGuid,
    pub(crate) bytes: AlignedBlobBuf,
    pub(crate) expected_seq: u64,
    pub(crate) content_version: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WriteThroughStatus {
    Written,
    Stale,
}

pub(crate) struct WriteThroughBatchReport {
    pub(crate) statuses: Vec<WriteThroughStatus>,
}

/// Dirty blob claimed by a checkpoint round before byte cloning.
///
/// `content_version` is captured under `CommitGate`; later
/// checkpoint cloning accepts the cached bytes only if the blob
/// still carries the same latch version under a shared blob guard.
/// If a foreground writer updates the blob first, the round restores
/// this dirty entry and retries it later instead of writing bytes
/// whose WAL record was outside the captured watermark.
#[derive(Clone, Copy)]
pub(crate) struct DirtySnapshotEntry {
    pub(crate) guid: BlobGuid,
    pub(crate) expected_seq: u64,
    pub(crate) content_version: u64,
}

#[derive(Clone, Copy)]
enum PinAccess {
    Point,
    Scan,
    Silent,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ReadIndexPolicy {
    PointRead,
    Liveness,
}

const READ_AUX_CACHE_MIN_BYTES: usize = 2 * 1024 * 1024;
const READ_AUX_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024;
const READ_AUX_CACHE_ENABLE_AT_BYTES: usize = 16 * 1024 * 1024;
const READ_INDEX_DIRECTORY_PROBE_BYTES: usize = PAGE_4K as usize;

#[derive(Clone, Copy, Debug)]
struct CacheBudget {
    blob_slots: usize,
    read_page_bytes: usize,
    read_index_bytes: usize,
}

impl CacheBudget {
    fn memory(total_blob_slots: usize) -> Self {
        Self {
            blob_slots: total_blob_slots.max(1),
            read_page_bytes: 0,
            read_index_bytes: 0,
        }
    }

    fn file(total_blob_slots: usize) -> Self {
        let total_blob_slots = total_blob_slots.max(1);
        let total_bytes = total_blob_slots.saturating_mul(PAGE_SIZE as usize);
        let read_aux_bytes = if total_bytes < READ_AUX_CACHE_ENABLE_AT_BYTES {
            0
        } else {
            (total_bytes / 2).clamp(READ_AUX_CACHE_MIN_BYTES, READ_AUX_CACHE_MAX_BYTES)
        };
        let read_index_bytes = read_aux_bytes * 7 / 8;
        let read_page_bytes = read_aux_bytes - read_index_bytes;
        let blob_bytes = total_bytes.saturating_sub(read_aux_bytes);
        Self {
            blob_slots: (blob_bytes / PAGE_SIZE as usize).max(1),
            read_page_bytes,
            read_index_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BufferStats {
    pub(crate) dirty_count: usize,
    pub(crate) pending_delete_count: usize,
    pub(crate) gc_orphan_backlog_count: usize,
    pub(crate) gc_reclaimed_count: u64,
    pub(crate) gc_last_full_sweep_deferred_count: usize,
    pub(crate) read_index_token_count: usize,
    pub(crate) read_index_cache_entries: usize,
    pub(crate) read_index_cache_bytes: usize,
    pub(crate) read_index_cache_budget_bytes: usize,
    pub(crate) read_page_cache_entries: usize,
    pub(crate) read_page_cache_bytes: usize,
    pub(crate) read_page_cache_ghost_entries: usize,
    pub(crate) read_page_cache_budget_bytes: usize,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) full_blob_reads: u64,
    pub(crate) full_blob_read_bytes: u64,
    pub(crate) point_full_blob_reads: u64,
    pub(crate) scan_full_blob_reads: u64,
    pub(crate) silent_full_blob_reads: u64,
    pub(crate) read_page_hits: u64,
    pub(crate) read_page_misses: u64,
    pub(crate) read_index_cache_hits: u64,
    pub(crate) read_index_cache_misses: u64,
    pub(crate) read_index_loads: u64,
    pub(crate) read_index_dir_read_bytes: u64,
    pub(crate) read_index_bucket_reads: u64,
    pub(crate) read_index_bucket_read_bytes: u64,
    pub(crate) read_index_inline_hits: u64,
    pub(crate) read_index_value_hits: u64,
    pub(crate) read_index_value_read_bytes: u64,
    pub(crate) read_index_offset_hits: u64,
    pub(crate) read_index_negative_hits: u64,
    pub(crate) read_index_crossing_hits: u64,
    pub(crate) read_index_unknowns: u64,
    pub(crate) optimistic_restarts: u64,
    pub(crate) range_restarts: u64,
    pub(crate) walker_ops: u64,
    pub(crate) walker_blob_hops: u64,
    pub(crate) max_blob_hops: u64,
    pub(crate) max_cross_blob_depth: u64,
    pub(crate) spillovers: u64,
    pub(crate) merges: u64,
    pub(crate) route_resident_count: usize,
    pub(crate) route_resident_demotions: u64,
    pub(crate) cache_evictions: u64,
    pub(crate) eviction_skips_protected: u64,
    pub(crate) eviction_skips_route_resident: u64,
    pub(crate) admission_protects: u64,
    pub(crate) write_delta_count: usize,
    pub(crate) store: StoreStats,
}

pub(crate) struct GcSweepOutcome {
    pub(crate) freed: usize,
    pub(crate) complete: bool,
}

/// Frequency-aware blob cache; see the module docs.
pub struct BufferManager {
    store: Arc<dyn BlobStore>,
    alloc_uninit: Arc<dyn Fn() -> AlignedBlobBuf + Send + Sync>,
    capacity: usize,
    /// Bounded 4 KiB cache for indexed-read header/routing pages.
    /// Counted inside the user-visible cache budget for file-backed
    /// trees; disabled for memory/custom stores.
    read_pages: ReadPageCache,
    read_indexes: ReadIndexCache,
    /// Per-blob invalidation token for read-index and indexed page reads.
    /// Read indexes are validated against the on-disk header
    /// when loaded; the token catches in-process writers/checkpoints
    /// after that without forcing every cold hit to reread the header.
    read_index_tokens: DashMap<BlobGuid, AtomicU64, GuidBuildHasher>,
    /// Monotonic source for cold invalidation tokens. A removed GUID
    /// that is later reintroduced receives a fresh token instead of
    /// reusing `0`, so stale read-index observations cannot become valid
    /// through token-map GC.
    read_index_token_clock: AtomicU64,
    /// Sharded blob cache. `DashMap` shards by `BlobGuid` so
    /// concurrent `pin` / `get_cached` on different blobs hit
    /// different shards — no single global mutex on the hot read
    /// path. The background eviction thread + each entry's
    /// `last_touched` tick give recency, while `admission` keeps
    /// one-shot point misses from displacing frequently reused
    /// metadata blobs. Keyed with [`GuidBuildHasher`] — a cheap
    /// avalanche over the already-high-entropy GUID, ~2.5x faster per
    /// hash than the default SipHash13 on this hot `pin` path.
    cache: DashMap<BlobGuid, Arc<CachedBlob>, GuidBuildHasher>,
    /// Approximate point-access frequency sketch. Scan and silent
    /// accesses deliberately do not update this so long list walks
    /// cannot pollute the point-read admission policy.
    admission: TinyLFU,
    /// Small protected tier for route-anchor blobs learned from
    /// the route cache.
    route_resident: RouteResidency,
    /// Per-blob mutation bookkeeping, sharded by `BlobGuid`.
    ///
    /// Each shard owns the dirty, flushing, and pending-delete
    /// entries for the same set of blobs. Keeping those three maps
    /// under one shard lock gives `mark_dirty` / `mark_for_delete`
    /// one short critical section with no global dirty mutex on the
    /// persistent write hot path.
    mutation: [Mutex<MutationState>; BOOKKEEPING_SHARDS],
    /// Deferred point writes. These are logical WAL-backed
    /// mutations that have been acknowledged to the caller but not
    /// merged into ART blob frames yet. Reads consult both pending and
    /// in-flight flush entries before the base tree, so checkpoint can
    /// publish a flush snapshot quickly and merge blob bytes outside
    /// the foreground writer gate.
    write_delta: WriteDelta,
    write_delta_flush: Mutex<()>,
    /// Serializes one complete checkpoint I/O phase: dependency-ordered
    /// dirty waves, their durability syncs, and the following pending-delete
    /// phase. Callers acquire this only after releasing CommitGate/capture
    /// locks; low-level write/flush helpers never acquire it recursively.
    checkpoint_io: Mutex<()>,
    delete_fence_total: AtomicUsize,
    /// Rotating shard cursors for advisory maintenance queues.
    /// Without this, a fixed shard-0-first drain can starve later
    /// shards when online maintenance has a small per-call budget.
    compact_candidate_cursor: AtomicUsize,
    merge_candidate_cursor: AtomicUsize,
    compact_candidate_total: AtomicUsize,
    merge_candidate_total: AtomicUsize,
    /// Monotonic logical clock used by the eviction thread to
    /// classify cache entries as cold. Every `pin` / `get_cached`
    /// stamps the touched entry's `last_touched` with
    /// `clock.fetch_add(1)`; the eviction thread compares the
    /// current clock to each entry's stamp to find candidates that
    /// haven't been used in the last N ticks. The same field also
    /// feeds the recency side of inline overflow eviction.
    ///
    /// Uses `Relaxed` ordering throughout — strict happens-before
    /// isn't required, only "more recent stamps look more recent".
    clock: AtomicU64,
    /// Hot-path observability counters. These are approximate
    /// metrics, not synchronization aids.
    telemetry: Telemetry,
    /// Monotonic global epoch driving copy-on-write snapshots. Bumped
    /// when a snapshot is taken; stamped into every newly-installed
    /// frame's `created_epoch` so a later mutation under a live
    /// snapshot knows whether it must fork the frame instead of
    /// overwriting it in place.
    current_epoch: AtomicU64,
    /// Highest epoch held by any LIVE snapshot — the copy-on-write
    /// fork barrier. A frame whose `created_epoch <= fork_barrier`
    /// may be visible to a snapshot and so must be forked before an
    /// in-place overwrite; `0` (no live snapshot) disables forking.
    fork_barrier: AtomicU64,
    /// Live CoW snapshot registry plus detached-frame staging/FIFO state.
    /// Retirement lowers the fork barrier and moves eligible COW/structural
    /// GUIDs into the exact-reclaim FIFO; it never deletes persisted data
    /// inline. Physical deletion requires a clean durable checkpoint frontier
    /// or a full reachability sweep.
    snapshots: Mutex<SnapshotState>,
    /// Even while idle and odd while physical GC deletion is active.
    /// Optimistic readers use this sequence to distinguish a concurrent
    /// reachability sweep from a stable missing-child corruption.
    gc_epoch: AtomicU64,
    physical_gc: Mutex<()>,
    gc_reclaimed_count: AtomicU64,
    /// Unreachable candidates deferred by the most recently completed full
    /// reachability sweep. Exact FIFO reclaim deliberately does not overwrite
    /// this value: an empty exact batch cannot prove that full-sweep debt is
    /// gone.
    gc_last_full_sweep_deferred_count: AtomicUsize,
    checkpoint_waker: Mutex<Option<std::thread::Thread>>,
}

struct GcEpochGuard<'a> {
    _serial: MutexGuard<'a, ()>,
    epoch: &'a AtomicU64,
}

#[cfg(test)]
struct ResidentPinBarrier {
    entered: std::sync::Barrier,
    release: std::sync::Barrier,
}

#[cfg(test)]
impl ResidentPinBarrier {
    fn new() -> Self {
        Self {
            entered: std::sync::Barrier::new(2),
            release: std::sync::Barrier::new(2),
        }
    }
}

#[cfg(test)]
thread_local! {
    static RESIDENT_PIN_BARRIER: std::cell::RefCell<Option<Arc<ResidentPinBarrier>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn set_resident_pin_barrier_for_current_thread(barrier: Arc<ResidentPinBarrier>) {
    RESIDENT_PIN_BARRIER.with(|slot| *slot.borrow_mut() = Some(barrier));
}

#[cfg(test)]
fn pause_resident_pin_after_fence_check() {
    let barrier = RESIDENT_PIN_BARRIER.with(|slot| slot.borrow_mut().take());
    if let Some(barrier) = barrier {
        barrier.entered.wait();
        barrier.release.wait();
    }
}

impl Drop for GcEpochGuard<'_> {
    fn drop(&mut self) {
        let previous = self.epoch.fetch_add(1, Ordering::AcqRel);
        debug_assert_eq!(previous & 1, 1, "GC epoch must be active on guard drop");
    }
}

impl BufferManager {
    // ---------- copy-on-write snapshots ----------

    /// Current global CoW epoch — the value stamped into every frame
    /// installed via [`Self::install_new_blob`] (forks included).
    #[must_use]
    pub(crate) fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::Acquire)
    }

    /// Restore the global CoW epoch on reopen, above every persisted
    /// frame's `created_epoch`, so snapshots taken after a reopen
    /// correctly fork pre-existing frames. Clamped to the floor of 1.
    pub(crate) fn set_current_epoch(&self, epoch: u64) {
        self.current_epoch.store(epoch.max(1), Ordering::Release);
    }

    /// The copy-on-write fork barrier: the highest epoch any live
    /// snapshot holds. A frame with `created_epoch <= fork_barrier`
    /// may be referenced by a snapshot and must be forked before an
    /// in-place overwrite. `0` means no live snapshot — the walker's
    /// hot path compares against it and never forks.
    #[must_use]
    pub(crate) fn fork_barrier(&self) -> u64 {
        self.fork_barrier.load(Ordering::Acquire)
    }

    /// Capture the physical-GC sequence for one optimistic reader attempt.
    #[must_use]
    pub(crate) fn gc_read_epoch(&self) -> u64 {
        self.gc_epoch.load(Ordering::Acquire)
    }

    /// Wait for a physical sweep to leave its odd epoch and return the next
    /// stable even sequence. Used only when a range cursor crosses a GC
    /// barrier; the uncontended path is one atomic load.
    #[must_use]
    pub(crate) fn gc_stable_read_epoch(&self) -> u64 {
        loop {
            let epoch = self.gc_read_epoch();
            if epoch & 1 == 0 {
                return epoch;
            }
            std::thread::yield_now();
        }
    }

    /// Return `true` when physical deletion was active at capture time or
    /// crossed the reader attempt. A stable even value means a missing blob
    /// is not explained by GC and must remain a hard error.
    #[must_use]
    pub(crate) fn gc_raced_since(&self, captured: u64) -> bool {
        captured & 1 != 0 || self.gc_epoch.load(Ordering::Acquire) != captured
    }

    /// Run a short publication closure while physical GC is excluded, but
    /// only if the caller's optimistic generation is still current.
    pub(crate) fn with_stable_gc_epoch<R>(
        &self,
        captured: u64,
        publish: impl FnOnce() -> R,
    ) -> Option<R> {
        let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
        (!self.gc_raced_since(captured)).then(publish)
    }

    #[must_use]
    pub(crate) fn gc_epoch_still_stable(&self, captured: u64) -> bool {
        self.with_stable_gc_epoch(captured, || ()).is_some()
    }

    fn begin_gc_epoch(&self) -> GcEpochGuard<'_> {
        let serial = self.physical_gc.lock().expect("physical GC lock poisoned");
        let previous = self.gc_epoch.fetch_add(1, Ordering::AcqRel);
        debug_assert_eq!(previous & 1, 0, "physical GC passes must be serialized");
        GcEpochGuard {
            _serial: serial,
            epoch: &self.gc_epoch,
        }
    }

    /// Fork a frame for copy-on-write: copy `src_bytes` to a fresh GUID
    /// `new_guid`, repatch the self-GUID, install it, and pin it. The
    /// install stamps the current epoch — strictly greater than any
    /// live fork barrier once a snapshot exists — so the fork is
    /// private to the live tree and is never itself re-forked for the
    /// snapshot that triggered it.
    pub(crate) fn fork_frame(
        &self,
        src_bytes: &[u8],
        new_guid: BlobGuid,
        seq: u64,
    ) -> Result<Arc<CachedBlob>> {
        let mut buf = self.alloc_blob_buf_zeroed();
        buf.as_mut_slice().copy_from_slice(src_bytes);
        crate::layout::set_frame_blob_guid(buf.as_mut_slice(), new_guid);
        self.install_new_blob(new_guid, buf, seq);
        self.pin(new_guid)
    }

    /// Copy `src`'s current frame image to a fresh in-memory GUID
    /// `new_guid` and pin it — the frozen root of a new CoW snapshot.
    ///
    /// The caller must hold the owning tree's mutation gate exclusively
    /// so the source frame is byte-stable for the copy. The copy keeps
    /// `src`'s entire structure (children are referenced by GUID, so
    /// they stay shared rather than deep-copied); only the self-GUID is
    /// repatched. Snapshot roots are ephemeral: they never enter the
    /// dirty map and never allocate a persistent store slot. Live writes
    /// under the snapshot still fork shared child frames through
    /// [`Self::fork_frame`], so crash recovery only has to reclaim those
    /// forked-away persistent frames.
    pub(crate) fn install_snapshot_root(
        &self,
        new_guid: BlobGuid,
        src: &CachedBlob,
    ) -> Arc<CachedBlob> {
        let guard = src.read();
        let mut buf = self.alloc_blob_buf_zeroed();
        buf.as_mut_slice().copy_from_slice(guard.as_slice());
        crate::layout::set_frame_blob_guid(buf.as_mut_slice(), new_guid);
        self.insert_owned_into_cache(new_guid, buf, PinAccess::Silent)
    }

    /// Register a live snapshot rooted at `root_guid`. Bumps the global
    /// epoch (so frames created afterwards are private to the live
    /// tree), raises the fork barrier to the snapshot's epoch, and
    /// returns that epoch.
    pub(crate) fn register_snapshot(
        &self,
        root_guid: BlobGuid,
        root_pin: &Arc<CachedBlob>,
    ) -> Result<u64> {
        let mut snaps = self.snapshots.lock().expect("snapshot registry poisoned");
        // The returned pre-bump value is this snapshot's barrier; the
        // post-bump value is persisted as the root high-water and stamps new
        // frames. MAX-1 is a valid but exhausted durable high-water, while
        // MAX can never be emitted by a successful registration. A checked
        // atomic update prevents MAX from wrapping to zero and clearing the
        // fork barrier while an exhausted snapshot remains live.
        let epoch = self
            .current_epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < u64::MAX - 1).then_some(current + 1)
            })
            .map_err(|_| Error::SnapshotEpochExhausted)?;
        snaps.live.insert(
            epoch,
            SnapshotRoot {
                guid: root_guid,
                pin: Arc::downgrade(root_pin),
            },
        );
        self.fork_barrier.store(epoch, Ordering::Release);
        Ok(epoch)
    }

    /// Stage a frame forked away from the live tree until its rewritten parent
    /// publishes dirty/flushing debt. `created_epoch` is the epoch of the
    /// forked-away version (≤ the barrier at fork time). Lease retirement only
    /// makes it eligible; a clean exact frontier or full GC deletes it.
    pub(crate) fn stage_cow_reclaim(
        &self,
        parent_guid: BlobGuid,
        guid: BlobGuid,
        created_epoch: u64,
    ) {
        let mut snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
        let pending = snapshots.cow_pending.entry(parent_guid).or_default();
        if !pending.iter().any(|(candidate, _)| *candidate == guid) {
            pending.push((guid, created_epoch));
        }
    }

    /// Stage a child detached by structural merge under the parent whose
    /// cached edge was rewritten. Staging is deliberately not reclaimable;
    /// `mark_dirty(parent)` promotes it only after the new parent image is
    /// protected by dirty/flushing checkpoint debt.
    pub(crate) fn stage_structural_reclaim(&self, parent_guid: BlobGuid, child_guid: BlobGuid) {
        let mut snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
        snapshots
            .structural_pending
            .entry(parent_guid)
            .or_default()
            .insert(child_guid);
    }

    /// Promote every COW/structural detachment owned by `parent_guid` after
    /// that parent's dirty image has been published. A concurrent final
    /// snapshot retirement can only make a staged child eligible here,
    /// after dirty/flushing state already closes the clean frontier.
    fn publish_parent_orphans(&self, parent_guid: BlobGuid) {
        let wake = {
            let mut snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
            let cow = snapshots
                .cow_pending
                .remove(&parent_guid)
                .unwrap_or_default();
            let structural = snapshots
                .structural_pending
                .remove(&parent_guid)
                .unwrap_or_default();
            let barrier = snapshots.live.keys().next_back().copied().unwrap_or(0);
            let mut wake = false;
            for (guid, created_epoch) in cow {
                if barrier != 0 && created_epoch <= barrier {
                    snapshots.orphans.push((guid, created_epoch));
                } else if snapshots.retired_orphan_set.insert(guid) {
                    snapshots.retired_orphans.push_back(guid);
                    wake = true;
                }
            }
            for guid in structural {
                if snapshots.live.is_empty() {
                    if snapshots.retired_orphan_set.insert(guid) {
                        snapshots.retired_orphans.push_back(guid);
                        wake = true;
                    }
                } else {
                    snapshots.structural_orphans.insert(guid);
                }
            }
            wake
        };
        if wake {
            self.wake_checkpointer();
        }
    }

    /// Retire the snapshot at `epoch`: lower the fork barrier to the
    /// highest remaining live snapshot epoch (or `0`), evict the
    /// snapshot's root frame, and stop tracking every orphan whose
    /// forked-away version is now newer than the barrier.
    ///
    /// Persisted blobs are deliberately retained. The live parent that
    /// stopped referencing a copy-on-write child may still have its older
    /// image on stable storage. Retirement only moves eligible COW and
    /// structural GUIDs into the exact-reclaim FIFO; physical deletion
    /// requires a clean durable checkpoint frontier or a full reachability
    /// sweep. Idempotent for an unknown epoch.
    pub(crate) fn retire_snapshot(&self, epoch: u64) {
        let (root, wake) = {
            let mut snaps = self.snapshots.lock().expect("snapshot registry poisoned");
            let root = snaps.live.remove(&epoch);
            let barrier = snaps.live.keys().next_back().copied().unwrap_or(0);
            self.fork_barrier.store(barrier, Ordering::Release);
            let mut active = Vec::with_capacity(snaps.orphans.len());
            for (guid, created_epoch) in std::mem::take(&mut snaps.orphans) {
                if barrier != 0 && created_epoch <= barrier {
                    active.push((guid, created_epoch));
                } else if snaps.retired_orphan_set.insert(guid) {
                    snaps.retired_orphans.push_back(guid);
                }
            }
            snaps.orphans = active;
            if barrier == 0 {
                let structural = std::mem::take(&mut snaps.structural_orphans);
                for guid in structural {
                    if snaps.retired_orphan_set.insert(guid) {
                        snaps.retired_orphans.push_back(guid);
                    }
                }
            }
            (root, !snaps.retired_orphans.is_empty())
        };
        if let Some(root) = root {
            self.discard_snapshot_root(root.guid);
        }
        if wake {
            self.wake_checkpointer();
        }
        // Persistent detachments deliberately keep their cache and dirty
        // bookkeeping. A clean checkpoint may reclaim their exact FIFO GUIDs;
        // a full GC additionally handles crash leftovers by reachability.
    }

    /// Discard an ephemeral snapshot root from process-local state.
    ///
    /// This must never call `store.delete_blob`: snapshot reachability is
    /// based on the current in-memory parent, which can be newer than the last
    /// durable parent. Persisted space is reclaimed only after a clean exact
    /// frontier or by [`Self::gc_sweep_unreachable_with_canonical`] under a
    /// durable reachability barrier.
    pub(crate) fn discard_snapshot_root(&self, guid: BlobGuid) {
        let _ = self.evict_reclaimable_blob(guid);
    }

    /// Evict all process-local state for `guid` when no pin, checkpoint,
    /// or pending delete still owns it. Returns `true` when a subsequent
    /// reachability-proven physical delete may proceed.
    fn evict_reclaimable_blob(&self, guid: BlobGuid) -> bool {
        {
            let state = self.mutation_shard(guid).lock().unwrap();
            if state.is_protected_or_pending(&guid) {
                return false;
            }
        }
        if let Some((_, entry)) = self.cache.remove_if(&guid, |_, entry| {
            if Arc::strong_count(entry) > 1 {
                return false;
            }
            let state = self.mutation_shard(guid).lock().unwrap();
            !state.is_protected_or_pending(&guid)
        }) {
            entry.clear_dirty_hint();
        } else if self.cache.contains_key(&guid) {
            return false;
        }
        self.route_resident.remove(guid);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state.remove_unclaimed_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
        true
    }

    /// Whether an API handle or in-flight reader still owns `guid` beyond
    /// the cache's own reference. Reachability GC uses this on a Dropping
    /// tree root to retain the whole closure until the old handle releases.
    pub(crate) fn blob_is_pinned(&self, guid: BlobGuid) -> bool {
        self.cache
            .get(&guid)
            .is_some_and(|entry| Arc::strong_count(entry.value()) > 1)
    }

    /// Physically delete one blob that a durability-barrier-protected
    /// reachability walk proved unreachable.
    fn reclaim_unreachable_blob(&self, guid: BlobGuid) -> Result<bool> {
        {
            let mut state = self.mutation_shard(guid).lock().unwrap();
            if state.is_protected_or_pending(&guid) {
                return Ok(false);
            }
            state.gc_deleting.insert(guid);
            self.delete_fence_total.fetch_add(1, Ordering::AcqRel);
        }

        let result = (|| {
            // Publish invalidation before eviction/delete. A cold indexed
            // reader either observes this token change or the delete fence;
            // a delayed reader that outlives both is covered by `gc_epoch`.
            self.invalidate_indexed_reads(guid);
            if let Some((_, entry)) = self.cache.remove_if(&guid, |_, entry| {
                if Arc::strong_count(entry) > 1 {
                    return false;
                }
                let state = self.mutation_shard(guid).lock().unwrap();
                !state.is_protected(&guid)
                    && !state.pending_deletes.contains_key(&guid)
                    && !state.deleting.contains_key(&guid)
            }) {
                entry.clear_dirty_hint();
            } else if self.cache.contains_key(&guid) {
                return Ok(false);
            }
            self.route_resident.remove(guid);
            let mut state = self.mutation_shard(guid).lock().unwrap();
            state.remove_unclaimed_dirty(&guid);
            let removed = state.remove_maintenance_candidates(&guid);
            drop(state);
            self.decrement_candidate_totals(removed);
            self.store.delete_blob(guid)?;
            Ok(true)
        })();

        let mut state = self.mutation_shard(guid).lock().unwrap();
        if state.gc_deleting.remove(&guid) {
            self.delete_fence_total.fetch_sub(1, Ordering::AcqRel);
        }
        drop(state);
        // The helper treats every delete fence as reachability protection.
        // Run token cleanup after retiring our GC-owned fence for every
        // result. A custom store may physically remove the GUID and then
        // report an I/O error; the helper rechecks cache/protection/store
        // reachability fail-closed, so it removes only truly dead tokens.
        self.remove_read_index_token_if_unreachable(guid);
        result
    }

    /// Pinned roots of every live snapshot epoch.
    ///
    /// Upgrading each weak pin while holding the registry lock closes the
    /// race with the last derived `View`/cursor dropping its epoch lease:
    /// GC either omits an already-retired root or owns a strong pin for the
    /// full reachability walk.
    pub(crate) fn snapshot_roots_pinned(self: &Arc<Self>) -> Result<Vec<PinnedSnapshotRoot>> {
        self.snapshots
            .lock()
            .expect("snapshot registry poisoned")
            .live
            .values()
            .map(|root| {
                root.pin
                    .upgrade()
                    .map(|pin| PinnedSnapshotRoot {
                        store: Arc::clone(self),
                        guid: root.guid,
                        pin: Some(pin),
                    })
                    .ok_or(Error::Internal("live snapshot registry lost its root pin"))
            })
            .collect()
    }

    /// Free every persisted frame not in `reachable`, returning the count
    /// reclaimed. The recovery-time sweep for copy-on-write frames
    /// orphaned by a crash that lost the in-memory orphan list. The
    /// caller must hold the tree quiescent, establish a durable checkpoint
    /// of the same root image, and pass the full reachable set (live root
    /// ∪ every live snapshot root).
    #[cfg(test)]
    pub(crate) fn gc_sweep_unreachable(&self, reachable: &HashSet<BlobGuid>) -> Result<usize> {
        self.gc_sweep_unreachable_bounded(reachable, usize::MAX)
            .map(|outcome| outcome.freed)
    }

    /// Sweep unreachable blobs while distinguishing the durable canonical
    /// topology from frames kept alive only by captured snapshot roots. Both
    /// closures protect physical deletion during this pass, but only
    /// canonical-live GUIDs invalidate retired-orphan FIFO entries.
    pub(crate) fn gc_sweep_unreachable_with_canonical(
        &self,
        reachable: &HashSet<BlobGuid>,
        canonical_reachable: &HashSet<BlobGuid>,
    ) -> Result<usize> {
        self.gc_sweep_unreachable_with_canonical_bounded(reachable, canonical_reachable, usize::MAX)
            .map(|outcome| outcome.freed)
    }

    /// Delete at most `limit` reachability-proven blobs after the caller has
    /// frozen and durably checkpointed the topology used for `reachable`.
    /// Pins are fail-closed: skipped blobs remain available to the next pass.
    #[cfg(test)]
    pub(crate) fn gc_sweep_unreachable_bounded(
        &self,
        reachable: &HashSet<BlobGuid>,
        limit: usize,
    ) -> Result<GcSweepOutcome> {
        self.gc_sweep_unreachable_with_canonical_bounded(reachable, reachable, limit)
    }

    /// Bounded sweep variant with a separate canonical-live closure.
    /// `reachable` is the union used to protect deletion; snapshot-only
    /// members remain in the retired FIFO so the ordinary exact-reclaim path
    /// can free them immediately after the GC-owned snapshot pins retire.
    pub(crate) fn gc_sweep_unreachable_with_canonical_bounded(
        &self,
        reachable: &HashSet<BlobGuid>,
        canonical_reachable: &HashSet<BlobGuid>,
        limit: usize,
    ) -> Result<GcSweepOutcome> {
        debug_assert!(canonical_reachable.is_subset(reachable));
        let _epoch = self.begin_gc_epoch();
        let mut freed = 0usize;
        let mut deferred = 0usize;
        let mut reclaimed = HashSet::new();
        // The GC epoch already owns `physical_gc`; bypass this type's public
        // BlobStore wrapper here so that wrapper can serialize external
        // enumerations without recursively taking the same mutex.
        let present = self.store.list_blobs()?;
        for guid in present.iter().copied() {
            if reachable.contains(&guid) {
                continue;
            }
            if freed >= limit {
                deferred = deferred.saturating_add(1);
                continue;
            }
            if self.reclaim_unreachable_blob(guid)? {
                freed += 1;
                reclaimed.insert(guid);
            } else {
                deferred = deferred.saturating_add(1);
            }
        }
        if freed != 0 {
            self.store.flush()?;
            self.gc_reclaimed_count
                .fetch_add(freed as u64, Ordering::Relaxed);
        }
        self.gc_last_full_sweep_deferred_count
            .store(deferred, Ordering::Relaxed);

        // A missing candidate was already reclaimed by an earlier pass or
        // recovery. Remove it without requiring another full sweep.
        let present: HashSet<_> = present.into_iter().collect();
        let mut snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
        snapshots.retired_orphans.retain(|guid| {
            present.contains(guid)
                && !reclaimed.contains(guid)
                && !canonical_reachable.contains(guid)
        });
        snapshots.retired_orphan_set = snapshots.retired_orphans.iter().copied().collect();
        Ok(GcSweepOutcome {
            freed,
            complete: deferred == 0,
        })
    }

    /// Reclaim a FIFO batch of COW or structural frames that are no longer
    /// protected by a snapshot lease. The caller must have completed a clean
    /// durable checkpoint; that frontier proves current parents no longer
    /// reference these exact GUIDs, so the normal path needs no all-store scan.
    pub(crate) fn reclaim_retired_orphans_bounded(&self, limit: usize) -> Result<usize> {
        let candidates = self.take_retired_orphans_bounded(limit);
        self.reclaim_retired_orphan_batch(candidates)
    }

    /// Move one FIFO batch out of the backlog while the caller owns the
    /// clean-frontier commit/maintenance barrier. Separating capture from
    /// I/O closes the race where a post-checkpoint writer could enqueue a
    /// not-yet-durable orphan between the clean test and queue drain.
    pub(crate) fn take_retired_orphans_bounded(&self, limit: usize) -> Vec<BlobGuid> {
        let mut snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
        let take = limit.min(snapshots.retired_orphans.len());
        let mut candidates = Vec::with_capacity(take);
        for _ in 0..take {
            let guid = snapshots
                .retired_orphans
                .pop_front()
                .expect("bounded retired orphan count");
            snapshots.retired_orphan_set.remove(&guid);
            candidates.push(guid);
        }
        candidates
    }

    pub(crate) fn reclaim_retired_orphan_batch(&self, candidates: Vec<BlobGuid>) -> Result<usize> {
        if candidates.is_empty() {
            return Ok(0);
        }

        let _epoch = self.begin_gc_epoch();
        let mut freed = 0usize;
        let mut retry = Vec::new();
        for (index, guid) in candidates.iter().copied().enumerate() {
            match self.store.has_blob(guid) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(error) => {
                    retry.extend(candidates[index..].iter().copied());
                    self.restore_retired_orphans(retry);
                    return Err(error);
                }
            }
            match self.reclaim_unreachable_blob(guid) {
                Ok(true) => freed += 1,
                Ok(false) => retry.push(guid),
                Err(error) => {
                    retry.extend(candidates[index..].iter().copied());
                    self.restore_retired_orphans(retry);
                    return Err(error);
                }
            }
        }
        if freed != 0 {
            if let Err(error) = self.store.flush() {
                self.restore_retired_orphans(candidates);
                return Err(error);
            }
            self.gc_reclaimed_count
                .fetch_add(freed as u64, Ordering::Relaxed);
        }
        self.restore_retired_orphans(retry);
        Ok(freed)
    }

    fn restore_retired_orphans(&self, guids: impl IntoIterator<Item = BlobGuid>) {
        let mut snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
        for guid in guids {
            if snapshots.retired_orphan_set.insert(guid) {
                snapshots.retired_orphans.push_back(guid);
            }
        }
    }

    #[must_use]
    pub(crate) fn gc_orphan_backlog_count(&self) -> usize {
        let snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
        snapshots.cow_pending.values().map(Vec::len).sum::<usize>()
            + snapshots.orphans.len()
            + snapshots
                .structural_pending
                .values()
                .map(HashSet::len)
                .sum::<usize>()
            + snapshots.structural_orphans.len()
            + snapshots.retired_orphans.len()
    }

    pub(crate) fn orphan_staging_count(&self) -> usize {
        let snapshots = self.snapshots.lock().expect("snapshot registry poisoned");
        snapshots.cow_pending.values().map(Vec::len).sum::<usize>()
            + snapshots
                .structural_pending
                .values()
                .map(HashSet::len)
                .sum::<usize>()
    }

    pub(crate) fn register_checkpoint_waker(&self, thread: std::thread::Thread) {
        *self
            .checkpoint_waker
            .lock()
            .expect("checkpoint waker lock poisoned") = Some(thread);
    }

    pub(crate) fn clear_checkpoint_waker(&self) {
        self.checkpoint_waker
            .lock()
            .expect("checkpoint waker lock poisoned")
            .take();
    }

    fn wake_checkpointer(&self) {
        if let Some(thread) = self
            .checkpoint_waker
            .lock()
            .expect("checkpoint waker lock poisoned")
            .as_ref()
        {
            thread.unpark();
        }
    }

    pub(crate) fn vacuum_storage(&self) -> Result<VacuumStats> {
        self.store.vacuum()
    }

    pub(crate) fn file_store_object_identity(&self) -> Option<FileStoreObjectIdentity> {
        self.store.file_store_object_identity()
    }
}

impl BufferManager {
    /// Wrap `store` with a cache of at most `capacity` blobs
    /// (each blob is 512 KB on the heap). A `capacity` of 0 is
    /// clamped to 1. Generic/custom stores do not receive a read-page
    /// side cache; file-backed trees call [`Self::new_file`] with their
    /// store-specific allocator.
    #[must_use]
    pub fn new(store: Arc<dyn BlobStore>, capacity: usize) -> Self {
        Self::new_with_budget_and_uninit_allocator(store, CacheBudget::memory(capacity), || {
            // SAFETY: BufferManager's uninitialized allocations are
            // filled by read_blob or a full-frame copy before read.
            unsafe { AlignedBlobBuf::uninit() }
        })
    }

    /// Wrap a file-backed store with one unified cache budget and a
    /// store-specific uninitialized-frame allocator.
    ///
    /// The public `capacity` is expressed in 512 KB blob-frame units.
    /// Internally, a bounded slice is reserved for rebuildable indexed-read
    /// indexes plus 4 KB read pages; the remainder becomes resident blob
    /// slots.
    #[must_use]
    pub(crate) fn new_file<F>(store: Arc<dyn BlobStore>, capacity: usize, alloc_uninit: F) -> Self
    where
        F: Fn() -> AlignedBlobBuf + Send + Sync + 'static,
    {
        Self::new_with_budget_and_uninit_allocator(store, CacheBudget::file(capacity), alloc_uninit)
    }

    fn new_with_budget_and_uninit_allocator<F>(
        store: Arc<dyn BlobStore>,
        budget: CacheBudget,
        alloc_uninit: F,
    ) -> Self
    where
        F: Fn() -> AlignedBlobBuf + Send + Sync + 'static,
    {
        let capacity = budget.blob_slots.max(1);
        Self {
            store,
            alloc_uninit: Arc::new(alloc_uninit),
            capacity,
            read_pages: ReadPageCache::new(budget.read_page_bytes),
            read_indexes: ReadIndexCache::new(budget.read_index_bytes),
            read_index_tokens: DashMap::with_hasher(GuidBuildHasher),
            read_index_token_clock: AtomicU64::new(1),
            cache: DashMap::with_hasher(GuidBuildHasher),
            admission: TinyLFU::new(),
            route_resident: RouteResidency::new(capacity),
            mutation: std::array::from_fn(|_| Mutex::new(MutationState::default())),
            write_delta: WriteDelta::default(),
            write_delta_flush: Mutex::new(()),
            checkpoint_io: Mutex::new(()),
            delete_fence_total: AtomicUsize::new(0),
            compact_candidate_cursor: AtomicUsize::new(0),
            merge_candidate_cursor: AtomicUsize::new(0),
            compact_candidate_total: AtomicUsize::new(0),
            merge_candidate_total: AtomicUsize::new(0),
            clock: AtomicU64::new(1),
            telemetry: Telemetry::default(),
            current_epoch: AtomicU64::new(1),
            fork_barrier: AtomicU64::new(0),
            snapshots: Mutex::new(SnapshotState::default()),
            gc_epoch: AtomicU64::new(0),
            physical_gc: Mutex::new(()),
            gc_reclaimed_count: AtomicU64::new(0),
            gc_last_full_sweep_deferred_count: AtomicUsize::new(0),
            checkpoint_waker: Mutex::new(None),
        }
    }

    fn alloc_blob_buf_uninit(&self) -> AlignedBlobBuf {
        (self.alloc_uninit)()
    }

    /// Enter the global checkpoint I/O phase.
    ///
    /// Lock order: capture/CommitGate first, release it, then acquire this
    /// guard before any child-first writes or pending deletes. The guard must
    /// cover the complete data-write → sync → delete → sync phase.
    pub(crate) fn enter_checkpoint_io(&self) -> MutexGuard<'_, ()> {
        self.checkpoint_io
            .lock()
            .expect("checkpoint I/O lock poisoned")
    }

    /// Current logical clock value. Read by the eviction
    /// thread to compare against each entry's `last_touched`. The
    /// returned tick is `Relaxed` — fine for "how cold is this
    /// entry" decisions, not for cross-thread synchronisation.
    pub(crate) fn clock_tick(&self) -> u64 {
        self.clock.load(Ordering::Relaxed)
    }

    /// Number of cache entries above the configured resident
    /// capacity. Background eviction uses this as a pressure gate:
    /// cold-but-resident entries are kept when the working set fits.
    pub(crate) fn cache_excess(&self) -> usize {
        self.cache.len().saturating_sub(self.capacity)
    }

    pub(crate) fn route_resident_count(&self) -> usize {
        self.route_resident.len()
    }

    pub(crate) fn mark_route_resident(&self, guid: BlobGuid) {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        for _ in 0..self.route_resident.mark(guid, tick) {
            self.telemetry.note_route_resident_demotion();
        }
    }

    fn is_route_resident(&self, guid: BlobGuid) -> bool {
        self.route_resident.contains(guid)
    }

    /// Iterate cached `(guid, entry)` pairs under a brief BM-state
    /// lock — the eviction thread snapshots this list, releases the
    /// lock, then makes its keep/drop decisions. The clone of the
    /// `Arc<CachedBlob>` bumps its strong count so `try_evict`
    /// won't fire on it mid-decision.
    pub(crate) fn snapshot_entries(&self) -> Vec<(BlobGuid, Arc<CachedBlob>)> {
        self.cache
            .iter()
            .map(|kv| (*kv.key(), Arc::clone(kv.value())))
            .collect()
    }

    fn decrement_candidate_totals(&self, removed: (bool, bool)) {
        if removed.0 {
            self.compact_candidate_total.fetch_sub(1, Ordering::Relaxed);
        }
        if removed.1 {
            self.merge_candidate_total.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Drop the cache entry for `guid` if (a) it's still cached,
    /// (b) we hold the only outside reference (caller's `Arc` was
    /// dropped before calling), and (c) dirty / pending-delete
    /// bookkeeping does not protect it.
    ///
    /// Returns `true` if an entry was actually evicted.
    pub(crate) fn try_evict_cold(&self, guid: BlobGuid) -> bool {
        if self.is_route_resident(guid) {
            self.telemetry.note_eviction_skip_route_resident();
            return false;
        }
        {
            let state = self.mutation_shard(guid).lock().unwrap();
            if state.is_protected_or_pending(&guid) {
                self.telemetry.note_eviction_skip_protected();
                return false;
            }
        }
        // `DashMap::remove_if` checks the predicate under the
        // shard lock. `strong_count == 1` means only the shard's
        // slot holds the `Arc` (the snapshot's clone was dropped
        // by the caller; see `eviction::run_scan`).
        let removed = self
            .cache
            .remove_if(&guid, |_, entry| {
                if self.is_route_resident(guid) {
                    self.telemetry.note_eviction_skip_route_resident();
                    return false;
                }
                if Arc::strong_count(entry) > 1 {
                    return false;
                }
                let state = self.mutation_shard(guid).lock().unwrap();
                let removable = !state.is_protected_or_pending(&guid);
                if !removable {
                    self.telemetry.note_eviction_skip_protected();
                }
                removable
            })
            .is_some();
        if removed {
            self.telemetry.note_cache_eviction();
        }
        removed
    }

    /// Current number of cached blobs.
    #[must_use]
    pub(crate) fn cached_count(&self) -> usize {
        self.cache.len()
    }

    pub(crate) fn stats(&self) -> BufferStats {
        let read_index_cache = self.read_indexes.snapshot();
        let read_page_cache = self.read_pages.snapshot();
        BufferStats {
            dirty_count: self.dirty_count(),
            pending_delete_count: self.pending_delete_count(),
            gc_orphan_backlog_count: self.gc_orphan_backlog_count(),
            gc_reclaimed_count: self.gc_reclaimed_count.load(Ordering::Relaxed),
            gc_last_full_sweep_deferred_count: self
                .gc_last_full_sweep_deferred_count
                .load(Ordering::Relaxed),
            read_index_token_count: self.read_index_tokens.len(),
            read_index_cache_entries: read_index_cache.entries,
            read_index_cache_bytes: read_index_cache.bytes,
            read_index_cache_budget_bytes: read_index_cache.budget_bytes,
            read_page_cache_entries: read_page_cache.entries,
            read_page_cache_bytes: read_page_cache.bytes,
            read_page_cache_ghost_entries: read_page_cache.ghost_entries,
            read_page_cache_budget_bytes: read_page_cache.budget_bytes,
            cache_hits: self.telemetry.cache_hits(),
            cache_misses: self.telemetry.cache_misses(),
            full_blob_reads: self.telemetry.full_blob_reads(),
            full_blob_read_bytes: self.telemetry.full_blob_reads() * PAGE_SIZE as u64,
            point_full_blob_reads: self.telemetry.point_full_blob_reads(),
            scan_full_blob_reads: self.telemetry.scan_full_blob_reads(),
            silent_full_blob_reads: self.telemetry.silent_full_blob_reads(),
            read_page_hits: self.telemetry.read_page_hits(),
            read_page_misses: self.telemetry.read_page_misses(),
            read_index_cache_hits: self.telemetry.read_index_cache_hits(),
            read_index_cache_misses: self.telemetry.read_index_cache_misses(),
            read_index_loads: self.telemetry.read_index_loads(),
            read_index_dir_read_bytes: self.telemetry.read_index_dir_read_bytes(),
            read_index_bucket_reads: self.telemetry.read_index_bucket_reads(),
            read_index_bucket_read_bytes: self.telemetry.read_index_bucket_read_bytes(),
            read_index_inline_hits: self.telemetry.read_index_inline_hits(),
            read_index_value_hits: self.telemetry.read_index_value_hits(),
            read_index_value_read_bytes: self.telemetry.read_index_value_read_bytes(),
            read_index_offset_hits: self.telemetry.read_index_offset_hits(),
            read_index_negative_hits: self.telemetry.read_index_negative_hits(),
            read_index_crossing_hits: self.telemetry.read_index_crossing_hits(),
            read_index_unknowns: self.telemetry.read_index_unknowns(),
            optimistic_restarts: self.telemetry.optimistic_restarts(),
            range_restarts: self.telemetry.range_restarts(),
            walker_ops: self.telemetry.walker_ops(),
            walker_blob_hops: self.telemetry.walker_blob_hops(),
            max_blob_hops: self.telemetry.max_blob_hops(),
            max_cross_blob_depth: self.telemetry.max_cross_blob_depth(),
            spillovers: self.telemetry.spillover_count(),
            merges: self.telemetry.merge_count(),
            route_resident_count: self.route_resident_count(),
            route_resident_demotions: self.telemetry.route_resident_demotions(),
            cache_evictions: self.telemetry.cache_evictions(),
            eviction_skips_protected: self.telemetry.eviction_skips_protected(),
            eviction_skips_route_resident: self.telemetry.eviction_skips_route_resident(),
            admission_protects: self.telemetry.admission_protects(),
            write_delta_count: self.write_delta.len(),
            store: self.store.store_stats(),
        }
    }

    /// Bump the optimistic-restart counter. Called from the
    /// lookup walker on `validate()` failure.
    pub(crate) fn note_optimistic_restart(&self) {
        self.telemetry.note_optimistic_restart();
    }

    pub(crate) fn note_range_restart(&self) {
        self.telemetry.note_range_restart();
    }

    /// Record one completed mutation walker traversal.
    pub(crate) fn note_walker_blob_hops(&self, hops: u64, max_cross_blob_depth: usize) {
        self.telemetry
            .note_walker_blob_hops(hops, max_cross_blob_depth);
    }

    /// Record one successful spillover.
    pub(crate) fn note_spillover(&self) {
        self.telemetry.note_spillover();
    }

    /// Record child-blob merge events.
    pub(crate) fn note_merges(&self, merged: u64) {
        self.telemetry.note_merges(merged);
    }

    /// Internal: look up `guid` in the cache under a declared
    /// access pattern.
    ///
    /// `Point` is the hot get/put path and refreshes recency.
    /// `Scan` counts cache telemetry but deliberately does not
    /// promote the entry, so a large range/list walk cannot rescue
    /// blobs that point lookups would otherwise evict. `Silent` is
    /// for observability and does not count or promote.
    fn get_cached_with_access(&self, guid: BlobGuid, access: PinAccess) -> Option<Arc<CachedBlob>> {
        let Some(entry) = self.cache.get(&guid) else {
            if !matches!(access, PinAccess::Silent) {
                self.telemetry.note_cache_miss();
            }
            if matches!(access, PinAccess::Point) {
                self.admission.record(guid);
            }
            return None;
        };
        let arc = Arc::clone(entry.value());
        drop(entry);
        match access {
            PinAccess::Point => {
                self.admission.record(guid);
                let tick = self.clock.fetch_add(1, Ordering::Relaxed);
                arc.last_touched.store(tick, Ordering::Relaxed);
                self.telemetry.note_cache_hit();
            }
            PinAccess::Scan => {
                self.telemetry.note_cache_hit();
            }
            PinAccess::Silent => {}
        }
        Some(arc)
    }

    fn mutation_shard(&self, guid: BlobGuid) -> &Mutex<MutationState> {
        &self.mutation[bookkeeping_shard_idx(&guid)]
    }

    fn is_pending_delete(&self, guid: BlobGuid) -> bool {
        if self.delete_fence_total.load(Ordering::Acquire) == 0 {
            return false;
        }
        self.mutation_shard(guid)
            .lock()
            .unwrap()
            .has_delete_fence(&guid)
    }

    /// True while `guid` is logically unlinked from the live tree but
    /// still fenced by the deferred-delete protocol.
    pub(crate) fn has_delete_fence(&self, guid: BlobGuid) -> bool {
        self.is_pending_delete(guid)
    }

    fn pending_delete_not_found(guid: BlobGuid) -> Error {
        Error::BlobStoreIo(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("blob {:02x?} is pending delete", &guid[..4]),
        ))
    }

    /// Internal: insert a freshly-loaded owned blob into the cache
    /// without cloning its 512 KB payload. Used on store read
    /// misses so an allocator-provided registered buffer can become
    /// the cached image directly.
    fn insert_owned_into_cache(
        &self,
        guid: BlobGuid,
        contents: AlignedBlobBuf,
        access: PinAccess,
    ) -> Arc<CachedBlob> {
        let hot_tick =
            matches!(access, PinAccess::Point).then(|| self.clock.fetch_add(1, Ordering::Relaxed));
        let inserted = self.cache.entry(guid).or_insert_with(|| {
            let entry = Arc::new(CachedBlob::new(contents));
            entry
                .last_touched
                .store(hot_tick.unwrap_or(0), Ordering::Relaxed);
            entry
        });
        let entry = Arc::clone(inserted.value());
        if let Some(tick) = hot_tick {
            // Re-stamp only hot point inserts. A scan miss racing
            // with an already-cached entry must not demote or
            // promote that entry; it should behave like a scan hit.
            entry.last_touched.store(tick, Ordering::Relaxed);
        }
        drop(inserted);

        // Inline overflow eviction. With the background eviction
        // thread running, capacity overflow is a rare burst
        // event — the bg sweep keeps it well below capacity in
        // steady state.
        //
        // The retry-with-yield loop tolerates the transient case
        // where every cache entry is currently pinned (every
        // `Arc::strong_count > 1`). Yielding gives concurrent
        // readers / writers a chance to drop their pins so the
        // next eviction attempt finds a victim. If after the
        // retry budget the cache still can't shrink, we let it
        // exceed capacity rather than failing the load — the
        // background sweep will catch up. `RETRY_BUDGET` is a
        // small constant (8) so we don't spin for long under
        // pathological pin pressure.
        const RETRY_BUDGET: u32 = 8;
        let mut retries_left = RETRY_BUDGET;
        let mut entry_spins = self.cache.len();
        while self.cache.len() > self.capacity {
            let evicted = match access {
                PinAccess::Point => self.try_evict_for_point_insert(guid),
                PinAccess::Scan | PinAccess::Silent => self.try_evict_scan_cold(),
            };
            if evicted {
                // Made progress — refresh the per-entry budget
                // (we only want to bound the total work, not
                // give up after one stuck victim).
                entry_spins = self.cache.len();
                continue;
            }
            if retries_left == 0 || entry_spins == 0 {
                break;
            }
            std::thread::yield_now();
            retries_left -= 1;
            entry_spins = entry_spins.saturating_sub(1);
        }
        entry
    }

    /// Internal: walk the cache for an unpinned clean victim and
    /// evict it. Point inserts prefer the lowest TinyLFU frequency
    /// and use `last_touched` as a tie-breaker; scan/silent
    /// overflow keeps the stricter "never evict point-touched
    /// blobs" path by requiring `last_touched == 0`.
    ///
    /// O(n) in the cache size, but called only on insert overflow
    /// — the background eviction thread handles steady-state
    /// reclaim with its own tick-driven cadence.
    ///
    /// **Dirty / pending-delete check is load-bearing** for the
    /// `dirty ⟺ cache image newer than store` (invariant I1)
    /// and `pending-delete ⟺ cache image must outlive the
    /// manifest unlink` properties. Without this check, an inline
    /// overflow can drop a cache image while its dirty entry stays
    /// in the dirty map — the next checkpoint's `snapshot_bytes`
    /// returns `None` for that guid and (pre-fix) silently skipped
    /// it; in memory mode the cache mutation was lost outright,
    /// in persistent mode the WAL truncate gate stuck closed
    /// forever. Matches `try_evict_cold`'s guard for the bg sweep.
    fn try_evict_for_point_insert(&self, candidate: BlobGuid) -> bool {
        self.try_evict_until(
            u64::MAX,
            Some((candidate, self.admission.estimate(candidate))),
        )
    }

    fn try_evict_scan_cold(&self) -> bool {
        self.try_evict_until(0, None)
    }

    fn try_evict_until(&self, max_last_touched: u64, candidate: Option<(BlobGuid, u8)>) -> bool {
        // Snapshot the dirty + pending-delete key sets under one
        // lock acquisition each, then scan the cache against the
        // snapshots. Holding the locks across the whole cache walk
        // would serialise reads against any concurrent writer.
        // Snapshotting and then re-validating under the per-shard
        // remove_if guard keeps the hot path lock-free.
        let protected_snap: std::collections::HashSet<BlobGuid> = {
            let mut out = std::collections::HashSet::new();
            for shard in &self.mutation {
                let state = shard.lock().unwrap();
                out.extend(state.dirty.keys().copied());
                out.extend(state.flushing.keys().copied());
                out.extend(state.pending_deletes.keys().copied());
            }
            out
        };

        let mut victim: Option<(BlobGuid, u8, u64)> = None;
        for kv in &self.cache {
            if Arc::strong_count(kv.value()) > 1 {
                continue;
            }
            let guid = *kv.key();
            if protected_snap.contains(&guid) {
                self.telemetry.note_eviction_skip_protected();
                continue;
            }
            if self.is_route_resident(guid) {
                self.telemetry.note_eviction_skip_route_resident();
                continue;
            }
            let tick = kv.value().last_touched.load(Ordering::Relaxed);
            if tick > max_last_touched {
                continue;
            }
            let freq = if candidate.is_some() {
                self.admission.estimate(guid)
            } else {
                0
            };
            match victim {
                None => victim = Some((guid, freq, tick)),
                Some((_, vfreq, vtick)) if (freq, tick) < (vfreq, vtick) => {
                    victim = Some((guid, freq, tick));
                }
                _ => {}
            }
        }
        if let (Some((candidate_guid, candidate_freq)), Some((victim_guid, victim_freq, _))) =
            (candidate, victim)
        {
            if victim_guid != candidate_guid && victim_freq > candidate_freq {
                self.telemetry.note_admission_protect();
                return false;
            }
        }
        if let Some((guid, _, _)) = victim {
            // `remove_if` re-checks strong_count + dirty + pending
            // under the shard lock — guards against a pin acquired
            // (or a fresh dirty / pending-delete mark) between our
            // scan and the remove.
            let removed = self
                .cache
                .remove_if(&guid, |_, e| {
                    if self.is_route_resident(guid) {
                        self.telemetry.note_eviction_skip_route_resident();
                        return false;
                    }
                    if Arc::strong_count(e) > 1 {
                        return false;
                    }
                    let state = self.mutation_shard(guid).lock().unwrap();
                    let removable = !state.is_protected_or_pending(&guid);
                    if !removable {
                        self.telemetry.note_eviction_skip_protected();
                    }
                    removable
                })
                .is_some();
            if removed {
                self.telemetry.note_cache_eviction();
            }
            return removed;
        }
        false
    }

    /// Internal: drop `guid` from cache (no-op if not cached) and
    /// clear any dirty bookkeeping for it. Called from
    /// `delete_blob`, where the blob is going away entirely and
    /// any pending dirty write would race with the delete in the
    /// store.
    #[cfg(test)]
    fn evict_from_cache(&self, guid: BlobGuid) -> bool {
        {
            let state = self.mutation_shard(guid).lock().unwrap();
            if state.checkpoint_owned_or_pending(&guid) {
                return false;
            }
        }
        if let Some((_, entry)) = self.cache.remove_if(&guid, |_, entry| {
            if Arc::strong_count(entry) > 1 {
                return false;
            }
            let state = self.mutation_shard(guid).lock().unwrap();
            !state.checkpoint_owned_or_pending(&guid)
        }) {
            entry.clear_dirty_hint();
        } else if self.cache.contains_key(&guid) {
            return false;
        }
        self.route_resident.remove(guid);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state.remove_unclaimed_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
        true
    }

    #[cfg(test)]
    pub(crate) fn evict_from_cache_for_test(&self, guid: BlobGuid) -> bool {
        self.evict_from_cache(guid)
    }

    /// Pin a blob in cache and return an `Arc<CachedBlob>` over it.
    ///
    /// On a cache miss, the blob is loaded from the inner store
    /// into a fresh cache entry first. The returned `Arc` keeps the
    /// entry alive (and unevictable) until it is dropped — callers
    /// should hold pins only as long as they're actively traversing
    /// or mutating, so eviction can make progress under pressure.
    ///
    /// From the returned handle, use:
    /// - [`CachedBlob::read_optimistic`] for wait-free reads
    ///   (snapshot + validate; restart on failure).
    /// - [`CachedBlob::read`] for blocking shared access.
    /// - [`CachedBlob::write`] for exclusive write access.
    pub fn pin(&self, guid: BlobGuid) -> Result<Arc<CachedBlob>> {
        self.pin_with_access(guid, PinAccess::Point)
    }

    /// Pin only if the blob is already resident in the hot blob cache.
    ///
    /// Point lookup uses this before trying the read index:
    /// resident blobs are authoritative and should not pay read-index
    /// eligibility/index checks. A miss returns `Ok(None)` without
    /// reading the backing store, so callers can decide whether to use
    /// read-index/page reads or fall back to a full [`Self::pin`].
    pub(crate) fn pin_cached(&self, guid: BlobGuid) -> Result<Option<Arc<CachedBlob>>> {
        self.pin_resident_stable(guid, PinAccess::Point)
    }

    /// Pin for range/list scans. Hits and misses remain visible in
    /// cache telemetry, but scan access does not refresh
    /// recency. This keeps large directory/object-list walks from
    /// evicting hot point-read blobs.
    pub(crate) fn pin_scan(&self, guid: BlobGuid) -> Result<Arc<CachedBlob>> {
        self.pin_with_access(guid, PinAccess::Scan)
    }

    /// Pin a batch of blobs for scanning, reading the cold ones in a
    /// single batched store read (device queue depth = batch size)
    /// instead of one serial round-trip each. The range scanner uses
    /// this to read-ahead upcoming child blobs so their indexed reads
    /// pipeline. The parallelism lives in [`BlobStore::read_blobs`]:
    /// the `pread` store fans the reads across worker threads, the
    /// `io_uring` store submits them as one ring batch.
    ///
    /// Returns one entry per `guids[i]`, in order: `Some(pin)` or `None`
    /// if pinning that guid failed (not-found / transient read error).
    /// Prefetch is best-effort — a `None` just means the caller pins that
    /// blob normally when it reaches it (surfacing any real error there),
    /// so dropping the error here is safe.
    ///
    /// Cache probe and insert run on the calling thread; only the cold
    /// frame reads are batched. This keeps the scan-access semantics
    /// (no recency bump, pending-delete re-check before insert)
    /// identical to a serial run of [`Self::pin_scan`].
    pub(crate) fn pin_scan_many(&self, guids: &[BlobGuid]) -> Vec<Option<Arc<CachedBlob>>> {
        let mut out: Vec<Option<Arc<CachedBlob>>> = Vec::with_capacity(guids.len());
        // Phase 1 — probe the cache on this thread. Hits and
        // pending-delete guids are finalised now; misses leave a
        // `None` placeholder and queue a indexed read.
        let mut miss_guids: Vec<BlobGuid> = Vec::new();
        let mut miss_slots: Vec<usize> = Vec::new();
        for &guid in guids {
            match self.pin_resident_stable(guid, PinAccess::Scan) {
                Ok(Some(entry)) => {
                    out.push(Some(entry));
                    continue;
                }
                Err(_) => {
                    out.push(None);
                    continue;
                }
                Ok(None) => {}
            }
            out.push(None);
            miss_slots.push(out.len() - 1);
            miss_guids.push(guid);
        }
        if miss_guids.is_empty() {
            return out;
        }
        let gc_epoch = self.gc_stable_read_epoch();

        // Phase 2 — read the cold frames in one batched store call.
        // SAFETY: `read_blobs` fills every PAGE_SIZE frame whose slot
        // it reports `Ok`; we only read a buffer back on `Ok` below.
        let mut bufs: Vec<AlignedBlobBuf> = (0..miss_guids.len())
            .map(|_| self.alloc_blob_buf_uninit())
            .collect();
        let results = self.store.read_blobs(&miss_guids, &mut bufs);

        // Phase 3 — insert each successful read, mirroring
        // `pin_with_access`: count the read, re-check pending-delete,
        // then insert with scan access (idempotent under a racing
        // insert).
        for (i, (buf, res)) in bufs.into_iter().zip(results).enumerate() {
            if res.is_err() {
                continue;
            }
            self.note_full_blob_read(PinAccess::Scan);
            let guid = miss_guids[i];
            out[miss_slots[i]] = self
                .insert_loaded_after_gc(guid, buf, PinAccess::Scan, gc_epoch)
                .ok()
                .flatten();
        }
        out
    }

    /// Like [`Self::pin`] but does not bump `cache_hits` /
    /// `cache_misses` and does not refresh the `last_touched`
    /// tick on a hit — used by introspection paths
    /// (`Tree::stats`, metrics scrapes, internal asserts) that
    /// must not perturb the very telemetry they're about to
    /// report or rescue cold entries from the eviction sweep
    /// just by looking at them.
    ///
    /// A miss still loads the blob because the pin contract must
    /// return a usable cache entry. The inserted entry is cold, so
    /// stats/maintenance walks do not promote blobs just by
    /// inspecting them.
    pub fn pin_silent(&self, guid: BlobGuid) -> Result<Arc<CachedBlob>> {
        self.pin_with_access(guid, PinAccess::Silent)
    }

    fn pin_with_access(&self, guid: BlobGuid, access: PinAccess) -> Result<Arc<CachedBlob>> {
        loop {
            if let Some(entry) = self.pin_resident_stable(guid, access)? {
                return Ok(entry);
            }
            let gc_epoch = self.gc_stable_read_epoch();
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }
            // SAFETY: a successful read_blob fills the full PAGE_SIZE frame
            // before `scratch` is inserted into the cache or read.
            let mut scratch = self.alloc_blob_buf_uninit();
            let read = self.store.read_blob(guid, &mut scratch);
            let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
            if self.gc_raced_since(gc_epoch) {
                continue;
            }
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }
            read?;
            self.note_full_blob_read(access);
            return Ok(self.insert_owned_into_cache(guid, scratch, access));
        }
    }

    /// Return a resident Arc only after revalidating the physical-GC epoch
    /// captured before the delete-fence check. Once cloned, the Arc itself
    /// makes a later delete fail closed; the terminal epoch check covers a
    /// delete that detached the map entry in the check-to-clone window.
    fn pin_resident_stable(
        &self,
        guid: BlobGuid,
        access: PinAccess,
    ) -> Result<Option<Arc<CachedBlob>>> {
        loop {
            let gc_epoch = self.gc_stable_read_epoch();
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }
            #[cfg(test)]
            pause_resident_pin_after_fence_check();
            let entry = self.get_cached_with_access(guid, access);
            if self.gc_raced_since(gc_epoch) {
                continue;
            }
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }
            return Ok(entry);
        }
    }

    fn insert_loaded_after_gc(
        &self,
        guid: BlobGuid,
        contents: AlignedBlobBuf,
        access: PinAccess,
        captured_gc_epoch: u64,
    ) -> Result<Option<Arc<CachedBlob>>> {
        let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
        if self.gc_raced_since(captured_gc_epoch) {
            return Ok(None);
        }
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        Ok(Some(self.insert_owned_into_cache(guid, contents, access)))
    }

    /// Whether `guid` may be served by a cold, page-granular read
    /// straight from the backing store.
    ///
    /// Returns `false` — meaning the caller must fall back to [`pin`]
    /// (which reads the authoritative resident/full-frame image) —
    /// when the blob is pending-delete, already resident in cache (a
    /// dirty cache image may be newer than the on-disk frame), the
    /// store still has unflushed data/manifest state, or the blob is
    /// protected/pending a structural op.
    ///
    /// [`pin`]: Self::pin
    pub(crate) fn indexed_read_eligible(&self, guid: BlobGuid) -> bool {
        self.read_index_eligible(guid, ReadIndexPolicy::PointRead)
    }

    /// Positional, page-granular read from the backing store, bypassing
    /// the 512 KiB blob cache. Linux builds route this through the
    /// file store's `io_uring` backend; other Unix builds use a
    /// positional `pread`. The caller owns 4 KiB alignment of
    /// `byte_offset` and `dst` (length + base).
    ///
    /// [`BlobStore::read_blob_range`]: crate::store::blob_store::BlobStore::read_blob_range
    pub(crate) fn read_blob_range(
        &self,
        guid: BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        loop {
            let gc_epoch = self.gc_stable_read_epoch();
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }
            let read = self.store.read_blob_range(guid, byte_offset, dst);
            let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
            if self.gc_raced_since(gc_epoch) {
                continue;
            }
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }
            return read;
        }
    }

    /// Fill `dst` with a cached 4 KiB indexed-read page. The caller must
    /// have checked [`Self::indexed_read_eligible`] before trusting it.
    pub(crate) fn read_page_cached(&self, guid: BlobGuid, page: u16, dst: &mut [u8]) -> bool {
        if self.read_pages.fill(guid, page, dst) {
            self.telemetry.note_read_page_hit();
            true
        } else {
            self.telemetry.note_read_page_miss();
            false
        }
    }

    /// Store a clean 4 KiB navigation page for later indexed reads.
    pub(crate) fn read_page_store(&self, guid: BlobGuid, page: u16, src: &[u8]) {
        self.read_pages.put(guid, page, src);
    }

    /// Store a clean 4 KiB leaf page only after repeated indexed reads.
    pub(crate) fn read_leaf_page_store(&self, guid: BlobGuid, page: u16, src: &[u8]) {
        self.read_pages.put_after_second_touch(guid, page, src);
    }

    pub(crate) fn read_index(&self, guid: BlobGuid) -> Option<(Arc<ReadIndex>, u64)> {
        self.read_index_load(guid, ReadIndexPolicy::PointRead)
    }

    pub(crate) fn read_index_for_liveness(&self, guid: BlobGuid) -> Option<(Arc<ReadIndex>, u64)> {
        self.read_index_load(guid, ReadIndexPolicy::Liveness)
    }

    fn read_index_load(
        &self,
        guid: BlobGuid,
        policy: ReadIndexPolicy,
    ) -> Option<(Arc<ReadIndex>, u64)> {
        if !self.read_index_eligible(guid, policy) {
            return None;
        }
        if let Some(index) = self.read_indexes.get(guid) {
            self.telemetry.note_read_index_cache_hit();
            return self.read_index_token(guid).map(|token| (index, token));
        }
        self.telemetry.note_read_index_cache_miss();
        let token = self.ensure_read_index_token(guid);
        let mut bytes = vec![0; READ_INDEX_DIRECTORY_PROBE_BYTES];
        if !self.store.read_index_range(guid, 0, &mut bytes).ok()? {
            return None;
        }
        let directory_len = ReadIndex::directory_len(&bytes).ok()?;
        let read_bytes = if directory_len > bytes.len() {
            let mut rest = vec![0; directory_len - bytes.len()];
            if !self
                .store
                .read_index_range(guid, bytes.len() as u64, &mut rest)
                .ok()?
            {
                return None;
            }
            bytes.extend_from_slice(&rest);
            bytes.len() as u64
        } else {
            bytes.truncate(directory_len);
            READ_INDEX_DIRECTORY_PROBE_BYTES as u64
        };
        let index = ReadIndex::decode_directory(bytes).ok()?;
        if !self.read_index_stamp_matches(guid, &index).ok()? {
            self.read_indexes.invalidate(guid);
            return None;
        }
        if self.read_index_token(guid) != Some(token) || !self.read_index_eligible(guid, policy) {
            return None;
        }
        self.telemetry.note_read_index_load(read_bytes);
        Some((self.read_indexes.insert(guid, index), token))
    }

    pub(crate) fn read_index_bucket(
        &self,
        guid: BlobGuid,
        index: &ReadIndex,
        user_key: &[u8],
        dst: &mut Vec<u8>,
    ) -> Option<()> {
        let (off, len) = index.bucket_range(user_key)?;
        dst.resize(len as usize, 0);
        self.telemetry.note_read_index_bucket_read(u64::from(len));
        if len != 0
            && !self
                .store
                .read_index_range(guid, u64::from(off), dst.as_mut_slice())
                .ok()?
        {
            return None;
        }
        Some(())
    }

    pub(crate) fn read_value_segment_range(
        &self,
        guid: BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> Option<()> {
        if !self
            .store
            .read_value_segment_range(guid, byte_offset, dst)
            .ok()?
        {
            return None;
        }
        Some(())
    }

    pub(crate) fn note_read_index_inline_hit(&self) {
        self.telemetry.note_read_index_inline_hit();
    }

    pub(crate) fn note_read_index_value_hit(&self, bytes: u64) {
        self.telemetry.note_read_index_value_hit(bytes);
    }

    pub(crate) fn note_read_index_offset_hit(&self) {
        self.telemetry.note_read_index_offset_hit();
    }

    pub(crate) fn note_read_index_negative_hit(&self) {
        self.telemetry.note_read_index_negative_hit();
    }

    pub(crate) fn note_read_index_crossing_hit(&self) {
        self.telemetry.note_read_index_crossing_hit();
    }

    pub(crate) fn note_read_index_unknown(&self) {
        self.telemetry.note_read_index_unknown();
    }

    fn invalidate_indexed_reads(&self, guid: BlobGuid) {
        self.bump_read_index_token(guid);
        self.read_pages.invalidate(guid);
        self.read_indexes.invalidate(guid);
    }

    pub(crate) fn read_index_token_valid(&self, guid: BlobGuid, token: u64) -> bool {
        self.indexed_read_eligible(guid) && self.read_index_token(guid) == Some(token)
    }

    pub(crate) fn read_index_liveness_token_valid(&self, guid: BlobGuid, token: u64) -> bool {
        self.read_index_eligible(guid, ReadIndexPolicy::Liveness)
            && self.read_index_token(guid) == Some(token)
    }

    fn read_index_eligible(&self, guid: BlobGuid, policy: ReadIndexPolicy) -> bool {
        if self.store.needs_flush() || self.is_pending_delete(guid) {
            return false;
        }
        if !self.store.has_blob(guid).unwrap_or(false) {
            return false;
        }
        if policy == ReadIndexPolicy::PointRead && self.cache.contains_key(&guid) {
            return false;
        }
        let state = self.mutation_shard(guid).lock().unwrap();
        !state.is_protected_or_pending(&guid)
    }

    fn read_index_token(&self, guid: BlobGuid) -> Option<u64> {
        self.read_index_tokens
            .get(&guid)
            .map(|entry| entry.load(Ordering::Acquire))
    }

    fn ensure_read_index_token(&self, guid: BlobGuid) -> u64 {
        if let Some(entry) = self.read_index_tokens.get(&guid) {
            return entry.load(Ordering::Acquire);
        }
        let token = self.next_read_index_token();
        let entry = self
            .read_index_tokens
            .entry(guid)
            .or_insert_with(|| AtomicU64::new(token));
        entry.load(Ordering::Acquire)
    }

    fn bump_read_index_token(&self, guid: BlobGuid) {
        let token = self.next_read_index_token();
        self.read_index_tokens
            .entry(guid)
            .or_insert_with(|| AtomicU64::new(token))
            .store(token, Ordering::Release);
    }

    fn next_read_index_token(&self) -> u64 {
        self.read_index_token_clock.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn remove_read_index_token_if_unreachable(&self, guid: BlobGuid) {
        if self.cache.contains_key(&guid) {
            return;
        }
        {
            let state = self.mutation_shard(guid).lock().unwrap();
            if state.is_protected_or_pending(&guid) {
                return;
            }
        }
        if self.store.has_blob(guid).unwrap_or(true) {
            return;
        }
        self.read_pages.invalidate(guid);
        self.read_indexes.invalidate(guid);
        self.read_index_tokens.remove(&guid);
    }

    fn read_index_stamp_matches(&self, guid: BlobGuid, index: &ReadIndex) -> Result<bool> {
        let mut scratch = AlignedBlobBuf::zeroed();
        self.store
            .read_blob_range(guid, 0, &mut scratch.as_mut_slice()[..HEADER_SIZE as usize])?;
        let frame = BlobFrameRef::wrap(scratch.as_slice());
        Ok(ReadIndexStamp::new(frame.header()) == index.stamp())
    }

    fn note_full_blob_read(&self, access: PinAccess) {
        match access {
            PinAccess::Point => self.telemetry.note_point_full_blob_read(),
            PinAccess::Scan => self.telemetry.note_scan_full_blob_read(),
            PinAccess::Silent => self.telemetry.note_silent_full_blob_read(),
        }
    }

    // ---------- dirty tracking ----------

    /// Stage a WAL-backed put without immediately mutating its target
    /// blob. `creates_key` records whether key-only scans must flush
    /// before answering prefix predicates. The latest staged op for
    /// the same `(tree_id, key)` wins; checkpoint later merges the
    /// logical op into ART frames under `CommitGate` before it may
    /// truncate WAL.
    pub(crate) fn stage_write_delta_put(
        &self,
        tree_id: u64,
        root_guid: BlobGuid,
        key: &[u8],
        value: &[u8],
        seq: u64,
        creates_key: bool,
    ) {
        self.write_delta
            .stage_put(tree_id, root_guid, key, value, seq, creates_key);
    }

    /// Stage a WAL-backed delete. See
    /// [`Self::stage_write_delta_put`].
    pub(crate) fn stage_write_delta_delete(
        &self,
        tree_id: u64,
        root_guid: BlobGuid,
        key: &[u8],
        seq: u64,
    ) {
        self.write_delta.stage_delete(tree_id, root_guid, key, seq);
    }

    pub(crate) fn lookup_write_delta(&self, tree_id: u64, key: &[u8]) -> Option<DeltaEntry> {
        self.write_delta.get(tree_id, key)
    }

    pub(crate) fn write_delta_count(&self) -> usize {
        self.write_delta.len()
    }

    pub(crate) fn write_delta_count_for_tree(&self, tree_id: u64) -> usize {
        self.write_delta.tree_len(tree_id)
    }

    pub(crate) fn write_delta_key_set_count_for_tree(&self, tree_id: u64) -> usize {
        self.write_delta.tree_key_set_len(tree_id)
    }

    pub(crate) fn write_delta_key_state(
        &self,
        tree_id: u64,
        key: &[u8],
    ) -> Option<WriteDeltaKeyState> {
        self.write_delta.get(tree_id, key).map(|entry| match entry {
            DeltaEntry::Put { seq, .. } => WriteDeltaKeyState::Put { seq },
            DeltaEntry::Delete { .. } => WriteDeltaKeyState::Delete,
        })
    }

    pub(crate) fn flush_write_deltas(&self) -> Result<()> {
        let _flush = self.write_delta_flush.lock().unwrap();
        let ops = self.write_delta.begin_flush_all();
        self.apply_write_delta_ops(ops)
    }

    pub(crate) fn flush_write_deltas_for_tree(&self, tree_id: u64) -> Result<()> {
        let _flush = self.write_delta_flush.lock().unwrap();
        let ops = self.write_delta.begin_flush_tree(tree_id);
        self.apply_write_delta_ops(ops)
    }

    fn apply_write_delta_ops(&self, ops: Vec<DeltaOp>) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        if let Err(e) = self.apply_write_delta_ops_inner(&ops) {
            self.write_delta.abort_flush(ops);
            return Err(e);
        }
        self.write_delta.finish_flush(&ops);
        Ok(())
    }

    fn apply_write_delta_ops_inner(&self, ops: &[DeltaOp]) -> Result<()> {
        let mut root_pins: HashMap<BlobGuid, Arc<CachedBlob>> = HashMap::new();
        let mut i = 0usize;
        while i < ops.len() {
            let op = &ops[i];
            let root_pin = match root_pins.entry(op.root_guid) {
                Entry::Occupied(entry) => Arc::clone(entry.get()),
                Entry::Vacant(entry) => Arc::clone(entry.insert(self.pin(op.root_guid)?)),
            };
            match &op.entry {
                DeltaEntry::Put { .. } => {
                    let mut end = i + 1;
                    while end < ops.len()
                        && ops[end].tree_id == op.tree_id
                        && ops[end].root_guid == op.root_guid
                        && matches!(ops[end].entry, DeltaEntry::Put { .. })
                    {
                        end += 1;
                    }
                    self.apply_write_delta_put_run(op.root_guid, &root_pin, &ops[i..end])?;
                    i = end;
                }
                DeltaEntry::Delete { seq } => {
                    let outcome = engine::erase_multi(
                        self,
                        &root_pin,
                        None,
                        engine::SearchKey::user(&op.key),
                        *seq,
                    )?;
                    if outcome.mutated && outcome.root_dirty {
                        self.mark_dirty_cached(op.root_guid, *seq, root_pin.as_ref());
                    }
                    i += 1;
                }
            }
        }
        Ok(())
    }

    fn apply_write_delta_put_run(
        &self,
        root_guid: BlobGuid,
        root_pin: &Arc<CachedBlob>,
        ops: &[DeltaOp],
    ) -> Result<()> {
        let mut items = Vec::with_capacity(ops.len());
        for op in ops {
            let DeltaEntry::Put { value, seq, .. } = &op.entry else {
                return Err(Error::Internal("write-delta put run contained non-put op"));
            };
            items.push(engine::InsertBatchItem::new(
                engine::SearchKey::user(&op.key),
                value,
                *seq,
                engine::InsertCondition::Always,
            ));
        }

        let mut applied = 0usize;
        while applied < items.len() {
            let outcome =
                engine::insert_multi_batch_conditional(self, root_pin, None, &items[applied..])?;
            if outcome.applied == 0 {
                return Err(Error::Internal("write-delta batch flush made no progress"));
            }
            if outcome.root_dirty {
                let dirty_seq = items[applied..applied + outcome.applied]
                    .iter()
                    .map(|item| item.seq)
                    .min()
                    .unwrap_or(u64::MAX);
                self.mark_dirty_cached(root_guid, dirty_seq, root_pin.as_ref());
            }
            applied += outcome.applied;
        }
        Ok(())
    }

    /// Tag `guid` as dirty at WAL seq `seq`.
    ///
    /// Called by every mutation path after a successful in-cache
    /// write to a blob. The internal dirty map keeps the **lowest**
    /// unflushed seq per blob — even though WAL seqs are
    /// monotonically allocated, two concurrent writers can run
    /// their `mark_dirty` calls in arrival order rather than seq
    /// order (writer B grabs seq 101 but its `mark_dirty(blob, 101)`
    /// can land before writer A's `mark_dirty(blob, 100)`). The
    /// `min`-merge keeps the dirty entry honest as a WAL trim
    /// watermark.
    ///
    /// This is the writer-side of the dirty-tracking contract; the
    /// checkpointer-side drains the map via
    /// [`Self::snapshot_dirty`].
    pub fn mark_dirty(&self, guid: BlobGuid, seq: u64) {
        let cached = self.get_cached_with_access(guid, PinAccess::Silent);
        if self.mark_dirty_with_hint(guid, seq, cached.as_deref()) {
            self.publish_parent_orphans(guid);
        }
    }

    /// Same contract as [`Self::mark_dirty`], but the caller
    /// already holds the cached blob pin from the walker descent.
    /// This avoids a second DashMap lookup on the mutation hot path.
    pub(crate) fn mark_dirty_cached(&self, guid: BlobGuid, seq: u64, entry: &CachedBlob) {
        if self.mark_dirty_with_hint(guid, seq, Some(entry)) {
            self.publish_parent_orphans(guid);
        }
    }

    fn mark_dirty_with_hint(&self, guid: BlobGuid, seq: u64, cached: Option<&CachedBlob>) -> bool {
        let Some(cached) = cached else {
            // No dirty entry without the newer cache image: that
            // would violate I1 and make checkpoint unable to
            // snapshot the bytes it is asked to flush.
            return false;
        };
        self.invalidate_indexed_reads(guid);
        let hint_covers_seq = !cached.dirty_hint_needs_map_publish(seq);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        // Logical unlink owns visibility and discards late writes. A
        // transient physical-GC fence is different: GC may defer because
        // this writer still pins the frame, so its acknowledged bytes must
        // remain dirty for the next checkpoint.
        if state.has_logical_delete_fence(&guid) {
            cached.clear_dirty_hint();
            return false;
        }
        if hint_covers_seq && matches!(state.dirty.get(&guid), Some(cur) if *cur <= seq) {
            return true;
        }
        if hint_covers_seq {
            cached.clear_dirty_hint();
            let _ = cached.dirty_hint_needs_map_publish(seq);
        }
        state
            .dirty
            .entry(guid)
            .and_modify(|cur| *cur = (*cur).min(seq))
            .or_insert(seq);
        true
    }

    /// Drain the current dirty entries from every bookkeeping shard,
    /// leaving empty per-shard dirty maps behind for concurrent
    /// writers.
    ///
    /// Returned map maps `guid -> lowest unflushed seq`. The
    /// caller (background checkpointer) is responsible for flushing
    /// each blob and either accepting the drain (on success) or
    /// restoring failed entries via [`Self::restore_dirty`].
    /// Persistent checkpoint rounds call this while holding the
    /// exclusive side of `CommitGate`, so the multi-shard drain is
    /// tree-wide stable for WAL trimming.
    #[must_use]
    pub fn snapshot_dirty(&self) -> HashMap<BlobGuid, u64> {
        let mut out = HashMap::new();
        for shard in &self.mutation {
            let (claimed, logically_deleted) = {
                let mut state = shard.lock().unwrap();
                let snap = std::mem::take(&mut state.dirty);
                let mut claimed = Vec::new();
                let mut logically_deleted = Vec::new();
                for (guid, seq) in snap {
                    if state.has_logical_delete_fence(&guid) {
                        logically_deleted.push(guid);
                    } else if state.gc_deleting.contains(&guid) {
                        // Transient physical GC may still defer on a pin.
                        // Keep the row in `dirty` and out of flushing.
                        state.dirty.insert(guid, seq);
                    } else {
                        state.add_flushing(guid);
                        claimed.push((guid, seq));
                    }
                }
                (claimed, logically_deleted)
            };

            // Never hold a mutation shard while touching DashMap: eviction
            // and GC remove cache→mutation, so the inverse order deadlocks.
            for guid in logically_deleted {
                if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
                    entry.clear_dirty_hint();
                }
            }
            for (guid, mut seq) in claimed {
                if let Some(hinted_seq) = self
                    .get_cached_with_access(guid, PinAccess::Silent)
                    .and_then(|entry| entry.take_dirty_hint())
                {
                    seq = seq.min(hinted_seq);
                }
                out.insert(guid, seq);
            }
        }
        out
    }

    /// Capture per-blob content versions for a just-drained dirty
    /// snapshot. Call while the caller still holds `CommitGate`,
    /// before foreground writers can publish newer dirty state.
    pub(crate) fn snapshot_dirty_versions(
        &self,
        snap: &HashMap<BlobGuid, u64>,
    ) -> Result<Vec<DirtySnapshotEntry>> {
        let mut out = Vec::with_capacity(snap.len());
        for (&guid, &seq) in snap {
            let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) else {
                return Err(Error::Internal(
                    "snapshot_dirty_versions: dirty entry lost cache image",
                ));
            };
            out.push(DirtySnapshotEntry {
                guid,
                expected_seq: seq,
                content_version: entry.content_version(),
            });
        }
        Ok(out)
    }

    /// Merge `entries` back into the dirty map, preserving the
    /// per-blob `min` between any existing entry (from a concurrent
    /// writer that ran after a snapshot drained the map) and the
    /// caller's value.
    ///
    /// Used by the checkpointer when a flush attempt fails — the
    /// snapshotted entries that didn't make it to store must stay
    /// tracked for the next round.
    pub fn restore_dirty(&self, entries: HashMap<BlobGuid, u64>) {
        if entries.is_empty() {
            return;
        }
        for (guid, t) in entries {
            let cached = self.get_cached_with_access(guid, PinAccess::Silent);
            if let Some(entry) = &cached {
                let _ = entry.dirty_hint_needs_map_publish(t);
            }
            let mut state = self.mutation_shard(guid).lock().unwrap();
            if state.has_logical_delete_fence(&guid) {
                if let Some(entry) = cached {
                    entry.clear_dirty_hint();
                }
                state.remove_one_flushing(&guid);
                continue;
            }
            state.remove_one_flushing(&guid);
            state
                .dirty
                .entry(guid)
                .and_modify(|cur| *cur = (*cur).min(t))
                .or_insert(t);
        }
    }

    /// Number of distinct dirty blobs currently tracked. Useful for
    /// metrics + checkpoint-policy thresholds.
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.mutation
            .iter()
            .map(|shard| shard.lock().unwrap().dirty.len())
            .sum()
    }

    /// Number of blobs currently owned by in-flight checkpoint
    /// epochs. WAL truncation must wait for this to reach zero:
    /// a drained dirty entry is no longer in `dirty`, but its
    /// bytes are not guaranteed durable until the epoch retires
    /// the corresponding flushing reference.
    #[must_use]
    pub(crate) fn flushing_count(&self) -> usize {
        self.mutation
            .iter()
            .map(|shard| shard.lock().unwrap().flushing.values().sum::<usize>())
            .sum()
    }

    // ---------- deferred delete (W2D for erase) ----------

    /// Tag `guid` for **deferred** store deletion at WAL seq
    /// `seq`. Removes the blob from cache + dirty (the cache
    /// image is dead; a lingering dirty entry would chase a
    /// soon-deleted slot) and queues the delete fence
    /// call for the next checkpoint round.
    ///
    /// Used by the erase walker's `SubtreeGone` branch. The naive
    /// alternative — calling `bm.delete_blob` inline — modifies
    /// the in-memory manifest before the WAL record covering the
    /// unlink is durable; a racing `store.flush` (from any other
    /// op's checkpoint) would persist the manifest's "child gone"
    /// view to disk while the WAL still lacks the erase record,
    /// and on reopen the root's `BlobNode` points at a slot the
    /// manifest no longer recognises (corruption). Deferring via
    /// this queue closes the window.
    ///
    /// The checkpoint round drains this set after Sync. User WAL
    /// deletes remove the blob from the store manifest and then
    /// re-Sync. `STRUCTURAL_SEQ` never enters this visibility fence:
    /// a copied snapshot root may still lazily load the detached child.
    /// Structural children instead wait in snapshot-protected orphan
    /// state and become exact reclaim candidates at a clean frontier.
    pub fn mark_for_delete(&self, guid: BlobGuid, seq: u64) {
        assert_ne!(
            seq, STRUCTURAL_SEQ,
            "STRUCTURAL_SEQ requires parent-scoped stage_structural_reclaim"
        );
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if let Some(seq_ref) = state.deleting.get_mut(&guid) {
            *seq_ref = (*seq_ref).min(seq);
            state.remove_unclaimed_dirty(&guid);
            let removed = state.remove_maintenance_candidates(&guid);
            drop(state);
            self.route_resident.remove(guid);
            self.decrement_candidate_totals(removed);
            return;
        }
        match state.pending_deletes.entry(guid) {
            Entry::Occupied(mut entry) => {
                let cur = entry.get_mut();
                *cur = (*cur).min(seq);
            }
            Entry::Vacant(entry) => {
                entry.insert(seq);
                self.delete_fence_total.fetch_add(1, Ordering::AcqRel);
            }
        }
        let keep_cached_for_flushing = state.flushing.contains_key(&guid);
        state.remove_unclaimed_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.route_resident.remove(guid);
        self.decrement_candidate_totals(removed);
        if keep_cached_for_flushing {
            if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
                entry.clear_dirty_hint();
            }
        } else if let Some((_, entry)) = self
            .cache
            .remove_if(&guid, |_, entry| Arc::strong_count(entry) == 1)
        {
            entry.clear_dirty_hint();
        } else if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
            entry.clear_dirty_hint();
        }
    }

    /// Drain the current pending-delete entries from every
    /// bookkeeping shard, leaving empty per-shard maps behind.
    /// Caller (checkpoint round / manual `Tree::checkpoint`) is
    /// responsible for executing each `store.delete_blob` or
    /// restoring on failure.
    #[must_use]
    pub fn snapshot_pending_deletes(&self) -> HashMap<BlobGuid, u64> {
        let mut out = HashMap::new();
        for shard in &self.mutation {
            let mut state = shard.lock().unwrap();
            let pending = std::mem::take(&mut state.pending_deletes);
            for (guid, seq) in &pending {
                state
                    .deleting
                    .entry(*guid)
                    .and_modify(|cur| *cur = (*cur).min(*seq))
                    .or_insert(*seq);
            }
            out.extend(pending);
        }
        out
    }

    /// Merge `entries` back into the pending-delete map, keeping
    /// the per-blob min seq.
    pub fn restore_pending_deletes(&self, entries: HashMap<BlobGuid, u64>) {
        if entries.is_empty() {
            return;
        }
        for (g, t) in entries {
            let mut state = self.mutation_shard(g).lock().unwrap();
            let mut seq = t;
            // `delete_fence_total` counts the transient GC fence and the
            // logical pending/deleting fence independently. Restoring a
            // logical row while GC owns the same GUID must therefore add one;
            // only an existing logical row suppresses the increment.
            let had_logical_fence = state.has_logical_delete_fence(&g);
            if let Some(claimed) = state.deleting.remove(&g) {
                seq = seq.min(claimed);
            }
            match state.pending_deletes.entry(g) {
                Entry::Occupied(mut entry) => {
                    let cur = entry.get_mut();
                    *cur = (*cur).min(seq);
                }
                Entry::Vacant(entry) => {
                    entry.insert(seq);
                    if !had_logical_fence {
                        self.delete_fence_total.fetch_add(1, Ordering::AcqRel);
                    }
                }
            }
        }
    }

    /// Number of blobs fenced for deferred store deletion. Counts
    /// queued deletes plus checkpoint-claimed deletes still in
    /// flight.
    /// Reads as zero under the WAL-truncate gate are part of the
    /// "WAL records are all redundant" invariant.
    #[must_use]
    pub fn pending_delete_count(&self) -> usize {
        self.delete_fence_total.load(Ordering::Acquire)
    }

    // ---------- online-maintenance candidates ----------

    /// Mark `guid` as a blob-local compaction candidate.
    ///
    /// Candidate state is an advisory in-memory scheduler hint. It
    /// is intentionally not persisted and not part of the WAL
    /// protocol: dirty / flushing / pending-delete bookkeeping owns
    /// correctness and eviction safety.
    pub(crate) fn note_compaction_candidate(&self, guid: BlobGuid) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if !state.has_delete_fence(&guid) && state.compact_candidates.insert(guid) {
            self.compact_candidate_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Mark `guid` as a parent-merge candidate.
    pub(crate) fn note_merge_candidate(&self, guid: BlobGuid) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if !state.has_delete_fence(&guid) && state.merge_candidates.insert(guid) {
            self.merge_candidate_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Pop up to `limit` blob-local compaction candidates.
    ///
    /// Popped candidates are removed from the queue. Callers that
    /// discover remaining debt should call `note_*_candidate`
    /// again, which pushes the guid to the back and prevents one
    /// stubborn candidate from starving later ones.
    #[must_use]
    pub(crate) fn pop_compaction_candidates(&self, limit: usize) -> Vec<BlobGuid> {
        pop_candidate_batch(
            &self.mutation,
            &self.compact_candidate_cursor,
            &self.compact_candidate_total,
            CandidateKind::Compact,
            limit,
        )
    }

    /// Pop up to `limit` parent-merge candidates.
    #[must_use]
    pub(crate) fn pop_merge_candidates(&self, limit: usize) -> Vec<BlobGuid> {
        pop_candidate_batch(
            &self.mutation,
            &self.merge_candidate_cursor,
            &self.merge_candidate_total,
            CandidateKind::Merge,
            limit,
        )
    }

    /// Number of blob-local compaction hints currently queued.
    #[must_use]
    pub(crate) fn compaction_candidate_count(&self) -> usize {
        self.compact_candidate_total.load(Ordering::Relaxed)
    }

    /// Number of parent-merge hints currently queued.
    #[must_use]
    pub(crate) fn merge_candidate_count(&self) -> usize {
        self.merge_candidate_total.load(Ordering::Relaxed)
    }

    /// Execute a queued user-delete fence. Structural orphans must never enter
    /// this queue: they use parent-scoped staging so a child cannot become
    /// reclaimable before its rewritten parent is published dirty. Returns
    /// `Ok(false)` when the blob still has dirty/flushing state and the caller
    /// should requeue it.
    pub(crate) fn execute_pending_delete(&self, guid: BlobGuid, seq: u64) -> Result<bool> {
        if seq == STRUCTURAL_SEQ {
            return Err(Error::Internal(
                "STRUCTURAL_SEQ bypassed parent-scoped staging",
            ));
        }
        {
            let state = self.mutation_shard(guid).lock().unwrap();
            if state.is_protected(&guid) {
                return Ok(false);
            }
        }
        if let Some((_, entry)) = self.cache.remove_if(&guid, |_, entry| {
            if Arc::strong_count(entry) > 1 {
                return false;
            }
            let state = self.mutation_shard(guid).lock().unwrap();
            !state.is_protected(&guid)
        }) {
            entry.clear_dirty_hint();
        } else if self.cache.contains_key(&guid) {
            return Ok(false);
        }
        self.store.delete_blob(guid)?;
        self.route_resident.remove(guid);
        self.finish_pending_delete(guid);
        self.remove_read_index_token_if_unreachable(guid);
        Ok(true)
    }

    fn finish_pending_delete(&self, guid: BlobGuid) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
        let had_claim = state.deleting.remove(&guid).is_some();
        if had_claim && !state.pending_deletes.contains_key(&guid) {
            self.delete_fence_total.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// `true` iff the inner store currently knows `guid`.
    ///
    /// This deliberately bypasses the cache: checkpoint dependency
    /// ordering needs to know whether a child blob has reached the
    /// store manifest, not whether the child is merely staged in
    /// memory.
    pub(crate) fn store_has_blob(&self, guid: BlobGuid) -> Result<bool> {
        self.store.has_blob(guid)
    }

    /// Snapshot the GUIDs known by the inner store, bypassing cache-only WAL
    /// replay state. DB open uses this only to distinguish a truly fresh
    /// store from a corrupted existing store whose catalog root is missing.
    pub(crate) fn store_blob_guids(&self) -> Result<Vec<BlobGuid>> {
        self.store.list_blobs()
    }

    /// `true` iff `guid` is visible in the store and the store does
    /// not owe a data/manifest flush.
    ///
    /// Checkpoint dependency ordering uses this before writing a
    /// parent blob that references an already-stored child. A
    /// `FileBlobStore` manifest update becomes visible to
    /// `has_blob` before it is necessarily durable in
    /// `manifest.log`; treating that as ready can persist a parent
    /// `BlobNode` that points to a child missing after crash
    /// recovery. This method is intentionally conservative: any
    /// pending store flush makes external children ineligible for a
    /// parent write until the I/O worker has crossed the durability
    /// boundary.
    pub(crate) fn store_has_durable_blob(&self, guid: BlobGuid) -> Result<bool> {
        if !self.store.has_blob(guid)? {
            return Ok(false);
        }
        Ok(!self.store.needs_flush())
    }

    /// `true` iff `guid` still has dirty or in-flight checkpoint
    /// state owned by the buffer manager.
    pub(crate) fn has_unflushed_blob(&self, guid: BlobGuid) -> bool {
        let state = self.mutation_shard(guid).lock().unwrap();
        state.dirty.contains_key(&guid) || state.flushing.contains_key(&guid)
    }

    /// Clone cached bytes only when the blob still has the
    /// checkpoint-captured content version.
    ///
    /// `Ok(None)` means a newer foreground writer reached the blob
    /// before this round could clone it; the caller should restore
    /// the dirty entry and retry later. `Err` means the dirty entry
    /// lost its protected cache image, which violates the flushing
    /// protection invariant.
    pub(crate) fn snapshot_bytes_if_version(
        &self,
        guid: BlobGuid,
        content_version: u64,
    ) -> Result<Option<AlignedBlobBuf>> {
        let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) else {
            return Err(Error::Internal(
                "snapshot_bytes_if_version: dirty entry lost cache image",
            ));
        };
        let buf = entry.read();
        if entry.content_version() != content_version {
            return Ok(None);
        }
        // SAFETY: copy_from_slice below writes the full PAGE_SIZE
        // frame before `out` is returned.
        let mut out = self.alloc_blob_buf_uninit();
        out.as_mut_slice().copy_from_slice(buf.as_slice());
        Ok(Some(out))
    }

    /// Allocate a zero-filled blob buffer from the inner store's
    /// preferred allocator.
    #[must_use]
    pub(crate) fn alloc_blob_buf_zeroed(&self) -> AlignedBlobBuf {
        self.store.alloc_blob_buf_zeroed()
    }

    /// Push a whole checkpoint snapshot to the inner store using
    /// its native batch path, then retire written flushing entries.
    /// Stale entries are reported to the caller and must be restored
    /// through [`Self::restore_dirty`], which retires that epoch's
    /// flushing reference exactly once.
    pub(crate) fn write_through_batch(
        &self,
        entries: &[WriteThroughEntry],
    ) -> Result<WriteThroughBatchReport> {
        if entries.is_empty() {
            return Ok(WriteThroughBatchReport {
                statuses: Vec::new(),
            });
        }
        let mut statuses = vec![WriteThroughStatus::Stale; entries.len()];
        let write_indices: Vec<_> = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| match self.write_snapshot_is_current(entry) {
                Ok(true) => Some(Ok(idx)),
                Ok(false) => None,
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<Vec<_>>>()?;
        let writes: Vec<_> = write_indices
            .iter()
            .map(|idx| (entries[*idx].guid, &entries[*idx].bytes))
            .collect();
        if !writes.is_empty() {
            self.store.write_blobs_with_data_sync(&writes)?;
        }
        for idx in &write_indices {
            let entry = &entries[*idx];
            self.invalidate_indexed_reads(entry.guid);
            self.try_publish_read_index(entry.guid, &entry.bytes);
        }
        for idx in write_indices {
            let entry = &entries[idx];
            if self.retire_write_through(entry.guid, entry.expected_seq, entry.content_version)? {
                statuses[idx] = WriteThroughStatus::Written;
            }
        }
        Ok(WriteThroughBatchReport { statuses })
    }

    fn write_snapshot_is_current(&self, entry: &WriteThroughEntry) -> Result<bool> {
        let Some(version) = entry.content_version else {
            return Ok(true);
        };
        let Some(cached) = self.get_cached_with_access(entry.guid, PinAccess::Silent) else {
            return Err(Error::Internal(
                "write_through_batch: flushing entry lost cache image",
            ));
        };
        Ok(cached.validate_content_version(version))
    }

    fn retire_write_through(
        &self,
        guid: BlobGuid,
        expected_seq: u64,
        content_version: Option<u64>,
    ) -> Result<bool> {
        let cached =
            if content_version.is_some() {
                Some(self.get_cached_with_access(guid, PinAccess::Silent).ok_or(
                    Error::Internal("retire_write_through: flushing entry lost cache image"),
                )?)
            } else {
                None
            };
        let mut state = self.mutation_shard(guid).lock().unwrap();
        // Revalidate while serializing with dirty publication. The first
        // check happens before store I/O; a writer can still update the
        // cached frame and publish a lower WAL seq in that I/O window. Such
        // bytes were not written by this batch and must remain dirty.
        if let (Some(expected), Some(cached)) = (content_version, cached.as_ref()) {
            if !cached.validate_content_version(expected) {
                return Ok(false);
            }
        }
        if expected_seq != STRUCTURAL_SEQ {
            if let std::collections::hash_map::Entry::Occupied(e) = state.dirty.entry(guid) {
                // Only retire the entry when no racing writer has
                // bumped it past this snapshot. `mark_dirty` keeps
                // the **minimum** unflushed seq; a lower/equal seq is
                // covered by this durable full-blob image, while a
                // higher seq belongs to a newer writer and must stay.
                if *e.get() <= expected_seq {
                    e.remove();
                }
            }
        }
        state.remove_one_flushing(&guid);
        let still_dirty = state.dirty.contains_key(&guid) || state.flushing.contains_key(&guid);
        drop(state);
        if !still_dirty {
            if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
                entry.clear_dirty_hint();
            }
        }
        Ok(true)
    }

    fn try_publish_read_index(&self, guid: BlobGuid, bytes: &AlignedBlobBuf) {
        let frame = BlobFrameRef::wrap(bytes.as_slice());
        let Ok(build) = ReadIndex::build(frame) else {
            return;
        };
        if self
            .store
            .publish_read_index(guid, &build.index, &build.values)
            .is_err()
        {
            self.read_indexes.invalidate(guid);
            return;
        }
        let Ok(directory_len) = ReadIndex::directory_len(&build.index[..ReadIndex::HEADER_LEN])
        else {
            self.read_indexes.invalidate(guid);
            return;
        };
        let Ok(index) = ReadIndex::decode_directory(build.index[..directory_len].to_vec()) else {
            self.read_indexes.invalidate(guid);
            return;
        };
        let _ = self.read_indexes.insert(guid, index);
    }

    /// Forward `flush` to the inner store without touching the
    /// cache. Used by the checkpoint I/O worker between epoch
    /// phases.
    pub(crate) fn flush_inner(&self) -> Result<()> {
        self.store.flush()
    }

    /// Write one public `BlobStore` image through to the inner store while
    /// preserving the cache's authoritative image on failure.
    ///
    /// The caller owns `checkpoint_io`. A resident frame is pinned and write
    /// latched before entering the physical-GC epoch: a walker may hold that
    /// same frame latch while pinning a child, so taking the locks in the
    /// opposite order would deadlock it behind the odd GC generation.
    fn write_blob_through_locked(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
        loop {
            let _stable = self.gc_stable_read_epoch();
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }

            if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
                // The Arc prevents reachability GC from reclaiming this frame
                // while we wait for its writer latch.
                let mut cached = entry.write();
                let captured_version = entry.content_version();
                let _epoch = self.begin_gc_epoch();
                if self.is_pending_delete(guid) {
                    return Err(Self::pending_delete_not_found(guid));
                }
                self.invalidate_indexed_reads(guid);
                self.store.write_blob(guid, src)?;

                // No other cache writer can change the frame while the latch
                // is held. Update it only after store success so a failed
                // write never exposes bytes that did not reach the store.
                debug_assert_eq!(entry.content_version(), captured_version);
                cached.as_mut_slice().copy_from_slice(src.as_slice());
                entry.clear_dirty_hint();
                let mut state = self.mutation_shard(guid).lock().unwrap();
                state.remove_unclaimed_dirty(&guid);
                let removed = state.remove_maintenance_candidates(&guid);
                drop(state);
                self.decrement_candidate_totals(removed);
                return Ok(());
            }

            // A cold reader can install the blob between the miss above and
            // this epoch. Recheck under `physical_gc`; if it won that race,
            // release the epoch and take its frame latch in the next attempt.
            let _epoch = self.begin_gc_epoch();
            if self.cache.contains_key(&guid) {
                continue;
            }
            if self.is_pending_delete(guid) {
                return Err(Self::pending_delete_not_found(guid));
            }
            self.invalidate_indexed_reads(guid);
            self.store.write_blob(guid, src)?;
            let mut state = self.mutation_shard(guid).lock().unwrap();
            state.remove_unclaimed_dirty(&guid);
            let removed = state.remove_maintenance_candidates(&guid);
            drop(state);
            self.decrement_candidate_totals(removed);
            return Ok(());
        }
    }

    /// Delete one public `BlobStore` GUID through an odd physical-GC epoch.
    /// The cached Arc is detached before inner I/O, closing the window where
    /// a reader that passed its first fence check could clone a soon-deleted
    /// resident entry. Failure reinstalls the same Arc before retiring the
    /// epoch and restores every dirty seq published behind the transient
    /// fence.
    fn delete_blob_through_locked(&self, guid: BlobGuid) -> Result<()> {
        let _epoch = self.begin_gc_epoch();
        let claimed_dirty = {
            let mut state = self.mutation_shard(guid).lock().unwrap();
            if state.checkpoint_owned_or_pending(&guid) {
                return Err(Error::Internal(
                    "delete_blob: checkpoint or delete fence owns blob",
                ));
            }
            // Claim unflushed debt atomically with the delete fence. A
            // concurrent checkpoint planner now sees neither a dirty row it
            // could discard because of the fence nor an unfenced row it could
            // write after this delete. Failure restores this exact seq.
            let claimed_dirty = state.dirty.remove(&guid);
            state.gc_deleting.insert(guid);
            self.delete_fence_total.fetch_add(1, Ordering::AcqRel);
            claimed_dirty
        };

        let detached_entry = self
            .cache
            .remove_if(&guid, |_, entry| Arc::strong_count(entry) == 1);
        let cache_claim_failed = detached_entry.is_none() && self.cache.contains_key(&guid);
        let result = (|| {
            if cache_claim_failed {
                return Err(Error::Internal(
                    "delete_blob: protected cache image cannot be evicted",
                ));
            }
            self.invalidate_indexed_reads(guid);
            self.store.delete_blob(guid)?;

            if let Some((_, entry)) = &detached_entry {
                entry.clear_dirty_hint();
            }
            self.route_resident.remove(guid);
            let mut state = self.mutation_shard(guid).lock().unwrap();
            state.remove_unclaimed_dirty(&guid);
            let removed = state.remove_maintenance_candidates(&guid);
            drop(state);
            self.decrement_candidate_totals(removed);
            Ok(())
        })();

        // Do not take the DashMap shard while holding the mutation shard.
        // A logical delete may win this gap; the final mutation-shard check
        // below detects that handoff and suppresses dirty resurrection.
        let logical_delete_before_reinsert = if result.is_err() {
            self.mutation_shard(guid)
                .lock()
                .unwrap()
                .has_logical_delete_fence(&guid)
        } else {
            false
        };
        if result.is_err() && !logical_delete_before_reinsert {
            if let Some((_, entry)) = &detached_entry {
                self.cache.entry(guid).or_insert_with(|| Arc::clone(entry));
            }
        }
        let claimed_entry = if result.is_err() {
            detached_entry
                .as_ref()
                .map(|(_, entry)| Arc::clone(entry))
                .or_else(|| self.get_cached_with_access(guid, PinAccess::Silent))
        } else {
            None
        };
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if state.gc_deleting.remove(&guid) {
            self.delete_fence_total.fetch_sub(1, Ordering::AcqRel);
        }
        let logical_delete_owned = state.has_logical_delete_fence(&guid);
        if result.is_err() {
            if let (Some(seq), false) = (claimed_dirty, logical_delete_owned) {
                if let Some(entry) = &claimed_entry {
                    let _ = entry.dirty_hint_needs_map_publish(seq);
                }
                state
                    .dirty
                    .entry(guid)
                    .and_modify(|current| *current = (*current).min(seq))
                    .or_insert(seq);
            } else if logical_delete_owned {
                if let Some(entry) = &claimed_entry {
                    entry.clear_dirty_hint();
                }
            }
        }
        drop(state);
        if result.is_ok() {
            self.remove_read_index_token_if_unreachable(guid);
        }
        result
    }

    /// Stage a freshly-created blob in cache and tag it dirty at
    /// `seq` — the unified `mark_dirty → checkpoint round → store
    /// write` protocol takes ownership from there.
    ///
    /// Used by spillover when it produces a new child blob: the
    /// bytes must NOT reach store before the WAL record covering
    /// the op that triggered spillover (invariant W2D). Deferring
    /// the store write via the dirty map preserves that ordering;
    /// the previous code's inline `write_blob → flush` here let the
    /// new child's bytes land on disk before the user's WAL record
    /// was durable, so a crash between the two left an orphan blob
    /// **and** could leave a parent `BlobNode` pointing at it (the
    /// parent's mutation was cached, but on recovery a subsequent
    /// op might flush the parent before the WAL record for the
    /// spillover-trigger op was durable).
    ///
    /// Overflow eviction can't fire on this fresh entry — its
    /// `dirty` entry would survive but the cache image wouldn't,
    /// breaking invariant **I1** (dirty ⟺ cache newer than
    /// store). Inline overflow eviction is therefore skipped
    /// here; the background eviction thread or the next round's
    /// flush will catch up.
    pub(crate) fn install_new_blob(&self, guid: BlobGuid, mut bytes: AlignedBlobBuf, seq: u64) {
        self.invalidate_indexed_reads(guid);
        // Stamp the creation epoch so copy-on-write snapshots can tell
        // whether a later mutation must fork this frame rather than
        // overwrite it in place.
        crate::layout::set_frame_created_epoch(
            bytes.as_mut_slice(),
            self.current_epoch.load(Ordering::Acquire),
        );
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        let entry = Arc::new(CachedBlob::new(bytes));
        entry.last_touched.store(tick, Ordering::Relaxed);
        // Defensive overwrite: a fresh GUID shouldn't collide, but
        // if it does we want the newest bytes to win (the dirty
        // entry below will also keep the lowest seq across both).
        //
        // Keep a local Arc clone until after dirty publication.
        // Eviction's remove_if requires `strong_count == 1`, so a
        // background sweep cannot drop this fresh cache entry in
        // the small window before the dirty bit is visible.
        self.cache.insert(guid, Arc::clone(&entry));
        let _ = entry.dirty_hint_needs_map_publish(seq);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state
            .dirty
            .entry(guid)
            .and_modify(|cur| *cur = (*cur).min(seq))
            .or_insert(seq);
        drop(entry);
    }
}

impl BlobStore for BufferManager {
    fn file_store_object_identity(&self) -> Option<FileStoreObjectIdentity> {
        self.store.file_store_object_identity()
    }

    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
        // `pin` is the authoritative full-frame read path: a cold load is
        // published only under a stable physical-GC generation, and the
        // returned strong reference keeps GC from reclaiming the frame while
        // its bytes are copied to the caller.
        let pin = self.pin(guid)?;
        let bytes = pin.read();
        dst.as_mut_slice().copy_from_slice(bytes.as_slice());
        Ok(())
    }

    fn read_blob_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<()> {
        // Public BlobStore reads must observe a newer dirty cache image. The
        // inherent range helper is intentionally a cold-store fast path used
        // only after indexed-read eligibility has ruled that case out.
        let pin = self.pin(guid)?;
        let bytes = pin.read();
        let start = usize::try_from(byte_offset).map_err(|_| {
            Error::BlobStoreIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "blob range offset exceeds addressable memory",
            ))
        })?;
        let end = start.checked_add(dst.len()).ok_or_else(|| {
            Error::BlobStoreIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "blob range end overflow",
            ))
        })?;
        let src = bytes.as_slice().get(start..end).ok_or_else(|| {
            Error::BlobStoreIo(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "blob range exceeds frame",
            ))
        })?;
        dst.copy_from_slice(src);
        Ok(())
    }

    fn read_index_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<bool> {
        loop {
            let gc_epoch = self.gc_stable_read_epoch();
            if !self.read_index_eligible(guid, ReadIndexPolicy::PointRead) {
                return Ok(false);
            }
            let read = self.store.read_index_range(guid, byte_offset, dst);
            let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
            if self.gc_raced_since(gc_epoch) {
                continue;
            }
            if !self.read_index_eligible(guid, ReadIndexPolicy::PointRead) {
                return Ok(false);
            }
            return read;
        }
    }

    fn read_value_segment_range(
        &self,
        guid: BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> Result<bool> {
        loop {
            let gc_epoch = self.gc_stable_read_epoch();
            if !self.read_index_eligible(guid, ReadIndexPolicy::PointRead) {
                return Ok(false);
            }
            let read = self.store.read_value_segment_range(guid, byte_offset, dst);
            let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
            if self.gc_raced_since(gc_epoch) {
                continue;
            }
            if !self.read_index_eligible(guid, ReadIndexPolicy::PointRead) {
                return Ok(false);
            }
            return read;
        }
    }

    fn publish_read_index(&self, guid: BlobGuid, bytes: &[u8], values: &[u8]) -> Result<()> {
        let _checkpoint_io = self.enter_checkpoint_io();
        let _epoch = self.begin_gc_epoch();
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        self.invalidate_indexed_reads(guid);
        self.store.publish_read_index(guid, bytes, values)
    }

    fn delete_read_index(&self, guid: BlobGuid) -> Result<()> {
        let _checkpoint_io = self.enter_checkpoint_io();
        let _epoch = self.begin_gc_epoch();
        self.invalidate_indexed_reads(guid);
        self.store.delete_read_index(guid)
    }

    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
        let _checkpoint_io = self.enter_checkpoint_io();
        self.write_blob_through_locked(guid, src)
    }

    fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        let _checkpoint_io = self.enter_checkpoint_io();
        // Preserve the trait's arbitrary-prefix failure contract while
        // keeping each successful prefix entry's cache identical to store.
        for (guid, src) in writes {
            self.write_blob_through_locked(*guid, src)?;
        }
        Ok(())
    }

    fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        let _checkpoint_io = self.enter_checkpoint_io();
        self.delete_blob_through_locked(guid)
    }

    fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
        let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
        self.store.list_blobs()
    }

    fn flush(&self) -> Result<()> {
        let _checkpoint_io = self.enter_checkpoint_io();
        let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
        self.store.flush()
    }

    fn needs_flush(&self) -> bool {
        let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
        self.store.needs_flush()
    }

    fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
        let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
        if self.is_pending_delete(guid) {
            return Ok(false);
        }
        // Fast path: shard-local check without consulting the
        // inner store.
        if self.cache.contains_key(&guid) {
            return Ok(true);
        }
        self.store.has_blob(guid)
    }

    fn store_stats(&self) -> StoreStats {
        let _gc = self.physical_gc.lock().expect("physical GC lock poisoned");
        self.store.store_stats()
    }

    fn vacuum(&self) -> Result<VacuumStats> {
        let _checkpoint_io = self.enter_checkpoint_io();
        let _epoch = self.begin_gc_epoch();
        self.store.vacuum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::blob_store::{FileBlobStore, MemoryBlobStore};
    use crate::store::{BlobFrame, PAGE_4K};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Barrier;

    fn make_buf(byte_at_100: u8) -> AlignedBlobBuf {
        let mut b = AlignedBlobBuf::zeroed();
        b.as_mut_slice()[100] = byte_at_100;
        b
    }

    fn snapshot_current_bytes(bm: &BufferManager, guid: BlobGuid) -> AlignedBlobBuf {
        let pin = bm.pin(guid).expect("test blob must be pinnable");
        bm.snapshot_bytes_if_version(guid, pin.content_version())
            .expect("test snapshot must keep its cache image")
            .expect("test snapshot version must stay current")
    }

    fn persist_all_dirty_for_test(bm: &BufferManager) {
        let dirty = bm.snapshot_dirty();
        let versioned = bm.snapshot_dirty_versions(&dirty).unwrap();
        let entries: Vec<_> = versioned
            .into_iter()
            .map(|snapshot| WriteThroughEntry {
                guid: snapshot.guid,
                bytes: bm
                    .snapshot_bytes_if_version(snapshot.guid, snapshot.content_version)
                    .unwrap()
                    .unwrap(),
                expected_seq: snapshot.expected_seq,
                content_version: Some(snapshot.content_version),
            })
            .collect();
        let _checkpoint_io = bm.enter_checkpoint_io();
        let report = bm.write_through_batch(&entries).unwrap();
        assert!(report
            .statuses
            .iter()
            .all(|status| *status == WriteThroughStatus::Written),);
        bm.flush_inner().unwrap();
    }

    struct FlushPendingStore {
        inner: MemoryBlobStore,
        pending: AtomicBool,
    }

    impl FlushPendingStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                pending: AtomicBool::new(false),
            }
        }
    }

    impl BlobStore for FlushPendingStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn read_blobs(&self, guids: &[BlobGuid], dsts: &mut [AlignedBlobBuf]) -> Vec<Result<()>> {
            self.inner.read_blobs(guids, dsts)
        }

        fn read_blob_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<()> {
            self.inner.read_blob_range(guid, byte_offset, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.pending.store(true, Ordering::Release);
            self.inner.write_blob(guid, src)
        }

        fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            self.pending.store(true, Ordering::Release);
            self.inner.write_blobs(writes)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.pending.store(true, Ordering::Release);
            self.inner.delete_blob(guid)
        }

        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            self.inner.list_blobs()
        }

        fn flush(&self) -> Result<()> {
            self.inner.flush()?;
            self.pending.store(false, Ordering::Release);
            Ok(())
        }

        fn needs_flush(&self) -> bool {
            self.pending.load(Ordering::Acquire) || self.inner.needs_flush()
        }
    }

    struct BlockingReadStore {
        inner: MemoryBlobStore,
        block_once: AtomicBool,
        entered: Barrier,
        release: Barrier,
    }

    impl BlockingReadStore {
        fn new(inner: MemoryBlobStore) -> Self {
            Self {
                inner,
                block_once: AtomicBool::new(true),
                entered: Barrier::new(2),
                release: Barrier::new(2),
            }
        }
    }

    struct BlockingWriteStore {
        inner: MemoryBlobStore,
        block_once: AtomicBool,
        entered: Barrier,
        release: Barrier,
    }

    impl BlockingWriteStore {
        fn new(inner: MemoryBlobStore) -> Self {
            Self {
                inner,
                block_once: AtomicBool::new(true),
                entered: Barrier::new(2),
                release: Barrier::new(2),
            }
        }
    }

    impl BlobStore for BlockingWriteStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            if self.block_once.swap(false, Ordering::AcqRel) {
                self.entered.wait();
                self.release.wait();
            }
            self.inner.write_blobs(writes)?;
            self.inner.flush()
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

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    impl BlobStore for BlockingReadStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)?;
            if self.block_once.swap(false, Ordering::AcqRel) {
                self.entered.wait();
                self.release.wait();
            }
            Ok(())
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
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

    struct BlockingTraitWriteStore {
        inner: MemoryBlobStore,
        block_once: AtomicBool,
        entered: Barrier,
        release: Barrier,
    }

    impl BlockingTraitWriteStore {
        fn new(inner: MemoryBlobStore) -> Self {
            Self {
                inner,
                block_once: AtomicBool::new(true),
                entered: Barrier::new(2),
                release: Barrier::new(2),
            }
        }
    }

    impl BlobStore for BlockingTraitWriteStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            if self.block_once.swap(false, Ordering::AcqRel) {
                self.entered.wait();
                self.release.wait();
            }
            self.inner.write_blob(guid, src)
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

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    struct FailingMutationStore {
        inner: MemoryBlobStore,
        fail_writes: AtomicBool,
        fail_deletes: AtomicBool,
        block_delete_once: AtomicBool,
        delete_entered: Barrier,
        delete_release: Barrier,
    }

    impl FailingMutationStore {
        fn new(inner: MemoryBlobStore) -> Self {
            Self {
                inner,
                fail_writes: AtomicBool::new(false),
                fail_deletes: AtomicBool::new(false),
                block_delete_once: AtomicBool::new(false),
                delete_entered: Barrier::new(2),
                delete_release: Barrier::new(2),
            }
        }
    }

    impl BlobStore for FailingMutationStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            if self.fail_writes.load(Ordering::Acquire) {
                return Err(Error::BlobStoreIo(std::io::Error::other(
                    "injected write failure",
                )));
            }
            self.inner.write_blob(guid, src)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            if self.block_delete_once.swap(false, Ordering::AcqRel) {
                self.delete_entered.wait();
                self.delete_release.wait();
            }
            if self.fail_deletes.load(Ordering::Acquire) {
                return Err(Error::BlobStoreIo(std::io::Error::other(
                    "injected delete failure",
                )));
            }
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

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    struct FailNthWriteStore {
        inner: MemoryBlobStore,
        fail_on: usize,
        writes: AtomicUsize,
    }

    struct FailAfterDeleteStore {
        inner: MemoryBlobStore,
        fail_once: AtomicBool,
    }

    impl FailAfterDeleteStore {
        fn new(inner: MemoryBlobStore) -> Self {
            Self {
                inner,
                fail_once: AtomicBool::new(true),
            }
        }
    }

    impl BlobStore for FailAfterDeleteStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.inner.delete_blob(guid)?;
            if self.fail_once.swap(false, Ordering::AcqRel) {
                return Err(Error::BlobStoreIo(std::io::Error::other(
                    "injected post-delete failure",
                )));
            }
            Ok(())
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

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    impl FailNthWriteStore {
        fn new(inner: MemoryBlobStore, fail_on: usize) -> Self {
            Self {
                inner,
                fail_on,
                writes: AtomicUsize::new(0),
            }
        }
    }

    impl BlobStore for FailNthWriteStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            let ordinal = self.writes.fetch_add(1, Ordering::AcqRel) + 1;
            if ordinal == self.fail_on {
                return Err(Error::BlobStoreIo(std::io::Error::other(
                    "injected batch-prefix failure",
                )));
            }
            self.inner.write_blob(guid, src)
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

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    #[test]
    fn cold_pin_does_not_reinsert_blob_after_gc_fence_retires() {
        let guid = [0xA1; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let store = Arc::new(BlockingReadStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));

        let reader_bm = Arc::clone(&bm);
        let reader = std::thread::spawn(move || reader_bm.pin(guid));
        store.entered.wait();

        let outcome = bm
            .gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
            .unwrap();
        assert_eq!(outcome.freed, 1);
        assert!(outcome.complete);
        store.release.wait();

        assert!(reader.join().unwrap().is_err());
        assert_eq!(bm.cached_count(), 0, "stale read must not create a zombie");
        assert!(!store.has_blob(guid).unwrap());
    }

    #[test]
    fn cold_pin_retries_after_unrelated_gc_epoch_change() {
        let target = [0xA6; 16];
        let unreachable = [0xA7; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(target, &make_buf(7)).unwrap();
        inner.write_blob(unreachable, &make_buf(8)).unwrap();
        let store = Arc::new(BlockingReadStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));

        let reader_bm = Arc::clone(&bm);
        let reader = std::thread::spawn(move || reader_bm.pin(target));
        store.entered.wait();

        let reachable = HashSet::from([target]);
        let outcome = bm
            .gc_sweep_unreachable_bounded(&reachable, usize::MAX)
            .unwrap();
        assert_eq!(outcome.freed, 1);
        assert!(outcome.complete);
        store.release.wait();

        let pin = reader
            .join()
            .unwrap()
            .expect("unrelated GC must be retried inside the cold pin");
        assert_eq!(pin.read().as_slice()[100], 7);
        assert!(store.has_blob(target).unwrap());
        assert!(!store.has_blob(unreachable).unwrap());
    }

    #[test]
    fn trait_full_read_does_not_publish_gc_zombie() {
        let guid = [0xB1; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let store = Arc::new(BlockingReadStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));

        let reader_bm = Arc::clone(&bm);
        let reader = std::thread::spawn(move || {
            let mut dst = AlignedBlobBuf::zeroed();
            BlobStore::read_blob(reader_bm.as_ref(), guid, &mut dst)
        });
        store.entered.wait();

        let outcome = bm
            .gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
            .unwrap();
        assert_eq!(outcome.freed, 1);
        store.release.wait();

        assert!(reader.join().unwrap().is_err());
        assert_eq!(bm.cached_count(), 0);
        assert!(!store.has_blob(guid).unwrap());
    }

    #[test]
    fn trait_range_read_does_not_publish_gc_zombie() {
        let guid = [0xB2; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let store = Arc::new(BlockingReadStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));

        let reader_bm = Arc::clone(&bm);
        let reader = std::thread::spawn(move || {
            let mut dst = [0u8; 8];
            BlobStore::read_blob_range(reader_bm.as_ref(), guid, 96, &mut dst)
        });
        store.entered.wait();

        let outcome = bm
            .gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
            .unwrap();
        assert_eq!(outcome.freed, 1);
        store.release.wait();

        assert!(reader.join().unwrap().is_err());
        assert_eq!(bm.cached_count(), 0);
        assert!(!store.has_blob(guid).unwrap());
    }

    #[test]
    fn trait_range_read_observes_dirty_cache_image() {
        let guid = [0xB3; 16];
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let pin = bm.pin(guid).unwrap();
        {
            let mut bytes = pin.write();
            bytes.as_mut_slice()[100] = 9;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());

        let mut dst = [0u8; 1];
        BlobStore::read_blob_range(&bm, guid, 100, &mut dst).unwrap();
        assert_eq!(dst, [9]);
    }

    #[test]
    fn trait_write_serializes_with_gc_epoch() {
        let guid = [0xB4; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store = Arc::new(BlockingTraitWriteStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));
        drop(bm.pin(guid).unwrap());

        let writer_bm = Arc::clone(&bm);
        let writer = std::thread::spawn(move || {
            BlobStore::write_blob(writer_bm.as_ref(), guid, &make_buf(9))
        });
        store.entered.wait();

        let gc_bm = Arc::clone(&bm);
        let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
        let gc = std::thread::spawn(move || {
            let result = gc_bm.gc_sweep_unreachable_bounded(&HashSet::from([guid]), usize::MAX);
            done_tx.send(result).unwrap();
        });
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "GC must wait while the public write owns its odd physical epoch",
        );

        store.release.wait();
        writer.join().unwrap().unwrap();
        let outcome = done_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap()
            .unwrap();
        gc.join().unwrap();
        assert_eq!(outcome.freed, 0);

        let mut stored = AlignedBlobBuf::zeroed();
        store.read_blob(guid, &mut stored).unwrap();
        assert_eq!(stored.as_slice()[100], 9);
        assert_eq!(bm.pin(guid).unwrap().read().as_slice()[100], 9);
    }

    #[test]
    fn trait_write_failure_preserves_cached_bytes_and_dirty_debt() {
        let guid = [0xB5; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store = Arc::new(FailingMutationStore::new(inner));
        let bm = BufferManager::new(store.clone(), 4);
        let pin = bm.pin(guid).unwrap();
        {
            let mut bytes = pin.write();
            bytes.as_mut_slice()[100] = 7;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        drop(pin);
        store.fail_writes.store(true, Ordering::Release);

        assert!(BlobStore::write_blob(&bm, guid, &make_buf(9)).is_err());
        assert_eq!(bm.pin(guid).unwrap().read().as_slice()[100], 7);
        assert_eq!(bm.dirty_count(), 1);
        let mut stored = AlignedBlobBuf::zeroed();
        store.inner.read_blob(guid, &mut stored).unwrap();
        assert_eq!(stored.as_slice()[100], 1);
    }

    #[test]
    fn trait_delete_failure_restores_planner_hidden_dirty_and_reopens_latest_bytes() {
        let guid = [0xB6; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store = Arc::new(FailingMutationStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));
        let pin = bm.pin(guid).unwrap();
        {
            let mut bytes = pin.write();
            bytes.as_mut_slice()[100] = 7;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        drop(pin);
        store.fail_deletes.store(true, Ordering::Release);
        store.block_delete_once.store(true, Ordering::Release);

        let delete_bm = Arc::clone(&bm);
        let delete = std::thread::spawn(move || BlobStore::delete_blob(delete_bm.as_ref(), guid));
        store.delete_entered.wait();

        assert!(
            bm.snapshot_dirty().is_empty(),
            "the public delete must hide its claimed dirty row from a planner",
        );
        store.delete_release.wait();
        assert!(delete.join().unwrap().is_err());

        assert_eq!(bm.pin(guid).unwrap().read().as_slice()[100], 7);
        assert_eq!(bm.dirty_count(), 1);
        assert_eq!(bm.pending_delete_count(), 0);
        assert!(store.inner.has_blob(guid).unwrap());

        persist_all_dirty_for_test(&bm);
        drop(bm);

        let store_dyn: Arc<dyn BlobStore> = store;
        let reopened = BufferManager::new(store_dyn, 4);
        assert_eq!(reopened.pin(guid).unwrap().read().as_slice()[100], 7);
    }

    #[test]
    fn trait_delete_failure_defers_to_concurrent_logical_delete() {
        let guid = [0xB9; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store = Arc::new(FailingMutationStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));
        let pin = bm.pin(guid).unwrap();
        {
            let mut bytes = pin.write();
            bytes.as_mut_slice()[100] = 7;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        drop(pin);
        store.fail_deletes.store(true, Ordering::Release);
        store.block_delete_once.store(true, Ordering::Release);

        let delete_bm = Arc::clone(&bm);
        let delete = std::thread::spawn(move || BlobStore::delete_blob(delete_bm.as_ref(), guid));
        store.delete_entered.wait();
        bm.mark_for_delete(guid, 20);
        assert_eq!(
            bm.pending_delete_count(),
            2,
            "generic and logical delete fences must be counted independently",
        );
        store.delete_release.wait();
        assert!(delete.join().unwrap().is_err());

        assert_eq!(bm.dirty_count(), 0, "logical deletion owns the old bytes");
        assert_eq!(bm.pending_delete_count(), 1);
        let pending = bm.snapshot_pending_deletes();
        assert_eq!(pending.get(&guid), Some(&20));
        bm.restore_pending_deletes(pending);
        assert_eq!(bm.pending_delete_count(), 1);
        assert!(store.inner.has_blob(guid).unwrap());
    }

    #[test]
    fn restored_logical_delete_counts_independently_from_transient_gc() {
        let guid = [0xBE; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store = Arc::new(FailingMutationStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));
        drop(bm.pin(guid).unwrap());
        store.fail_deletes.store(true, Ordering::Release);
        store.block_delete_once.store(true, Ordering::Release);

        let delete_bm = Arc::clone(&bm);
        let transient_delete =
            std::thread::spawn(move || BlobStore::delete_blob(delete_bm.as_ref(), guid));
        store.delete_entered.wait();
        assert_eq!(bm.pending_delete_count(), 1);

        // Model a failed trailing checkpoint sync restoring a logical row
        // while an unrelated transient delete fence still owns this GUID.
        bm.restore_pending_deletes(HashMap::from([(guid, 17)]));
        assert_eq!(
            bm.pending_delete_count(),
            2,
            "logical and transient fences require separate references",
        );

        store.delete_release.wait();
        assert!(transient_delete.join().unwrap().is_err());
        assert_eq!(bm.pending_delete_count(), 1);

        let claimed = bm.snapshot_pending_deletes();
        assert_eq!(claimed.get(&guid), Some(&17));
        store.fail_deletes.store(false, Ordering::Release);
        assert!(bm.execute_pending_delete(guid, 17).unwrap());
        assert_eq!(bm.pending_delete_count(), 0);
    }

    #[test]
    fn trait_delete_success_never_returns_detached_resident_pin() {
        let guid = [0xBA; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store = Arc::new(FailingMutationStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));
        drop(bm.pin(guid).unwrap());

        let pin_barrier = Arc::new(ResidentPinBarrier::new());
        let reader_bm = Arc::clone(&bm);
        let reader_barrier = Arc::clone(&pin_barrier);
        let reader = std::thread::spawn(move || {
            set_resident_pin_barrier_for_current_thread(reader_barrier);
            reader_bm.pin(guid)
        });
        pin_barrier.entered.wait();

        store.block_delete_once.store(true, Ordering::Release);
        let delete_bm = Arc::clone(&bm);
        let delete = std::thread::spawn(move || BlobStore::delete_blob(delete_bm.as_ref(), guid));
        store.delete_entered.wait();
        pin_barrier.release.wait();
        store.delete_release.wait();

        delete.join().unwrap().unwrap();
        assert!(
            reader.join().unwrap().is_err(),
            "a reader that passed the first fence check must not return the detached Arc",
        );
        assert!(!store.inner.has_blob(guid).unwrap());
        assert_eq!(bm.cached_count(), 0);
    }

    #[test]
    fn trait_delete_failure_reinstalls_resident_before_reader_retry() {
        let guid = [0xBB; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(3)).unwrap();
        let store = Arc::new(FailingMutationStore::new(inner));
        let bm = Arc::new(BufferManager::new(store.clone(), 4));
        drop(bm.pin(guid).unwrap());

        let pin_barrier = Arc::new(ResidentPinBarrier::new());
        let reader_bm = Arc::clone(&bm);
        let reader_barrier = Arc::clone(&pin_barrier);
        let reader = std::thread::spawn(move || {
            set_resident_pin_barrier_for_current_thread(reader_barrier);
            reader_bm.pin(guid)
        });
        pin_barrier.entered.wait();

        store.fail_deletes.store(true, Ordering::Release);
        store.block_delete_once.store(true, Ordering::Release);
        let delete_bm = Arc::clone(&bm);
        let delete = std::thread::spawn(move || BlobStore::delete_blob(delete_bm.as_ref(), guid));
        store.delete_entered.wait();
        pin_barrier.release.wait();
        store.delete_release.wait();

        assert!(delete.join().unwrap().is_err());
        let pin = reader
            .join()
            .unwrap()
            .expect("failed delete must make the reinstated resident visible on retry");
        assert_eq!(pin.read().as_slice()[100], 3);
        assert!(store.inner.has_blob(guid).unwrap());
    }

    #[test]
    fn trait_delete_protected_writer_keeps_gc_fence_dirty_debt_through_reopen() {
        let guid = [0xBC; 16];
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = Arc::new(BufferManager::new(inner.clone(), 4));
        let pin = bm.pin(guid).unwrap();
        {
            let mut bytes = pin.write();
            bytes.as_mut_slice()[100] = 7;
        }

        // Hold the cache shard so generic delete installs gc_deleting but
        // cannot complete its atomic remove_if before this writer publishes.
        let cache_ref = bm.cache.get(&guid).unwrap();
        let delete_bm = Arc::clone(&bm);
        let delete = std::thread::spawn(move || BlobStore::delete_blob(delete_bm.as_ref(), guid));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while bm.pending_delete_count() == 0 {
            assert!(std::time::Instant::now() < deadline, "delete fence timeout");
            std::thread::yield_now();
        }

        bm.mark_dirty_cached(guid, 30, pin.as_ref());
        assert!(
            bm.snapshot_dirty().is_empty(),
            "planner must leave transient-GC dirty rows in place",
        );
        assert_eq!(bm.dirty_count(), 1);
        drop(cache_ref);
        assert!(delete.join().unwrap().is_err());
        drop(pin);

        assert_eq!(bm.pending_delete_count(), 0);
        assert_eq!(bm.dirty_count(), 1);
        persist_all_dirty_for_test(&bm);
        drop(bm);

        let reopened = BufferManager::new(inner, 4);
        assert_eq!(reopened.pin(guid).unwrap().read().as_slice()[100], 7);
    }

    #[test]
    fn reachability_gc_protected_writer_keeps_dirty_debt_through_reopen() {
        let guid = [0xBD; 16];
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = Arc::new(BufferManager::new(inner.clone(), 4));
        let pin = bm.pin(guid).unwrap();

        // Hold the cache shard while reachability GC publishes its transient
        // fence. This gives the protected writer a deterministic window to
        // update and publish dirty debt before remove_if observes its pin.
        let cache_ref = bm.cache.get(&guid).unwrap();
        let gc_bm = Arc::clone(&bm);
        let gc = std::thread::spawn(move || {
            gc_bm.gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while bm.pending_delete_count() == 0 {
            assert!(std::time::Instant::now() < deadline, "GC fence timeout");
            std::thread::yield_now();
        }

        {
            let mut bytes = pin.write();
            bytes.as_mut_slice()[100] = 9;
        }
        bm.mark_dirty_cached(guid, 31, pin.as_ref());
        assert!(
            bm.snapshot_dirty().is_empty(),
            "planner must not claim a writer protected by transient GC",
        );
        assert_eq!(bm.dirty_count(), 1);

        drop(cache_ref);
        let outcome = gc.join().unwrap().unwrap();
        assert_eq!(outcome.freed, 0);
        assert!(!outcome.complete);
        assert_eq!(bm.pending_delete_count(), 0);
        assert_eq!(bm.cached_count(), 1);
        assert!(inner.has_blob(guid).unwrap());
        assert_eq!(bm.dirty_count(), 1);

        drop(pin);
        persist_all_dirty_for_test(&bm);
        drop(bm);

        let reopened = BufferManager::new(inner, 4);
        assert_eq!(reopened.pin(guid).unwrap().read().as_slice()[100], 9);
    }

    #[test]
    fn trait_batch_write_keeps_successful_prefix_cache_store_consistent() {
        let first = [0xB7; 16];
        let second = [0xB8; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(first, &make_buf(1)).unwrap();
        inner.write_blob(second, &make_buf(2)).unwrap();
        let store = Arc::new(FailNthWriteStore::new(inner, 2));
        let bm = BufferManager::new(store.clone(), 4);
        drop(bm.pin(first).unwrap());
        drop(bm.pin(second).unwrap());
        let first_new = make_buf(7);
        let second_new = make_buf(8);

        assert!(
            BlobStore::write_blobs(&bm, &[(first, &first_new), (second, &second_new)]).is_err()
        );
        assert_eq!(bm.pin(first).unwrap().read().as_slice()[100], 7);
        assert_eq!(bm.pin(second).unwrap().read().as_slice()[100], 2);
        let mut stored = AlignedBlobBuf::zeroed();
        store.inner.read_blob(first, &mut stored).unwrap();
        assert_eq!(stored.as_slice()[100], 7);
        store.inner.read_blob(second, &mut stored).unwrap();
        assert_eq!(stored.as_slice()[100], 2);
    }

    #[test]
    fn concurrent_gc_passes_keep_generation_serial_and_even() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0xA2; 16], &make_buf(1)).unwrap();
        inner.write_blob([0xA3; 16], &make_buf(2)).unwrap();
        let bm = Arc::new(BufferManager::new(inner.clone(), 4));
        let start = Arc::new(Barrier::new(3));

        let mut workers = Vec::new();
        for _ in 0..2 {
            let bm = Arc::clone(&bm);
            let start = Arc::clone(&start);
            workers.push(std::thread::spawn(move || {
                start.wait();
                bm.gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
                    .unwrap()
                    .freed
            }));
        }
        start.wait();
        let total: usize = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .sum();

        assert_eq!(total, 2);
        assert_eq!(bm.gc_read_epoch(), 4);
        assert_eq!(bm.gc_read_epoch() & 1, 0);
        assert!(inner.list_blobs().unwrap().is_empty());
    }

    #[test]
    fn bounded_gc_reports_incomplete_until_all_candidates_are_visited() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0xA8; 16], &make_buf(1)).unwrap();
        inner.write_blob([0xA9; 16], &make_buf(2)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        let first = bm.gc_sweep_unreachable_bounded(&HashSet::new(), 1).unwrap();
        assert_eq!(first.freed, 1);
        assert!(!first.complete);
        assert_eq!(inner.list_blobs().unwrap().len(), 1);
        assert_eq!(bm.stats().gc_last_full_sweep_deferred_count, 1);

        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 0);
        assert_eq!(
            bm.stats().gc_last_full_sweep_deferred_count,
            1,
            "an empty exact-reclaim batch must not clear full-sweep debt",
        );

        let second = bm.gc_sweep_unreachable_bounded(&HashSet::new(), 1).unwrap();
        assert_eq!(second.freed, 1);
        assert!(second.complete);
        assert!(inner.list_blobs().unwrap().is_empty());
        assert_eq!(bm.stats().gc_last_full_sweep_deferred_count, 0);
    }

    #[test]
    fn gc_releases_read_index_tokens_after_delete_fence_retires() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for suffix in 0..64u8 {
            let mut guid = [0xAB; 16];
            guid[15] = suffix;
            inner.write_blob(guid, &make_buf(suffix)).unwrap();
        }
        let bm = BufferManager::new(inner.clone(), 8);

        let outcome = bm
            .gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
            .unwrap();
        assert_eq!(outcome.freed, 64);
        assert!(outcome.complete);
        assert!(inner.list_blobs().unwrap().is_empty());
        assert_eq!(
            bm.read_index_tokens.len(),
            0,
            "successful GC must not retain per-GUID indexed-read tokens",
        );
    }

    #[test]
    fn gc_post_delete_error_still_releases_unreachable_index_token() {
        let guid = [0xAC; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let store = Arc::new(FailAfterDeleteStore::new(inner));
        let bm = BufferManager::new(store.clone(), 4);
        drop(bm.pin(guid).unwrap());
        let token = bm.ensure_read_index_token(guid);
        assert_eq!(bm.read_index_tokens.len(), 1);

        assert!(bm
            .gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
            .is_err());
        assert!(!store.inner.has_blob(guid).unwrap());
        assert_eq!(bm.cached_count(), 0);
        assert_eq!(bm.pending_delete_count(), 0);
        assert_eq!(bm.read_index_tokens.len(), 0);
        assert!(!bm.read_index_token_valid(guid, token));

        let retry = bm
            .gc_sweep_unreachable_bounded(&HashSet::new(), usize::MAX)
            .unwrap();
        assert_eq!(retry.freed, 0);
        assert!(retry.complete);
    }

    #[test]
    fn deferred_fifo_skips_pinned_head_then_reclaims_later_candidates() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let pinned_guid = [0xA4; 16];
        let free_guid = [0xA5; 16];
        inner.write_blob(pinned_guid, &make_buf(1)).unwrap();
        inner.write_blob(free_guid, &make_buf(2)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);
        let pin = bm.pin(pinned_guid).unwrap();
        bm.restore_retired_orphans([pinned_guid, free_guid]);

        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 0);
        assert_eq!(bm.gc_orphan_backlog_count(), 2);
        assert_eq!(bm.stats().gc_last_full_sweep_deferred_count, 0);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 1);
        assert!(!inner.has_blob(free_guid).unwrap());
        assert!(inner.has_blob(pinned_guid).unwrap());
        assert_eq!(bm.gc_orphan_backlog_count(), 1);
        assert_eq!(
            bm.stats().gc_last_full_sweep_deferred_count,
            0,
            "exact reclaim must not overwrite the full-sweep gauge",
        );

        drop(pin);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 1);
        assert_eq!(bm.gc_orphan_backlog_count(), 0);
        assert_eq!(bm.stats().gc_last_full_sweep_deferred_count, 0);
        assert!(!inner.has_blob(pinned_guid).unwrap());
    }

    struct CountingReadIndexStore {
        inner: FileBlobStore,
        index_reads: AtomicUsize,
    }

    impl CountingReadIndexStore {
        fn open(path: &std::path::Path) -> Result<Self> {
            Ok(Self {
                inner: FileBlobStore::open(path)?,
                index_reads: AtomicUsize::new(0),
            })
        }

        fn index_reads(&self) -> usize {
            self.index_reads.load(Ordering::Acquire)
        }

        fn reset_index_reads(&self) {
            self.index_reads.store(0, Ordering::Release);
        }
    }

    impl BlobStore for CountingReadIndexStore {
        fn alloc_blob_buf_zeroed(&self) -> AlignedBlobBuf {
            self.inner.alloc_blob_buf_zeroed()
        }

        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn read_blobs(&self, guids: &[BlobGuid], dsts: &mut [AlignedBlobBuf]) -> Vec<Result<()>> {
            self.inner.read_blobs(guids, dsts)
        }

        fn read_blob_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<()> {
            self.inner.read_blob_range(guid, byte_offset, dst)
        }

        fn read_index_range(
            &self,
            guid: BlobGuid,
            byte_offset: u64,
            dst: &mut [u8],
        ) -> Result<bool> {
            self.index_reads.fetch_add(1, Ordering::AcqRel);
            self.inner.read_index_range(guid, byte_offset, dst)
        }

        fn read_value_segment_range(
            &self,
            guid: BlobGuid,
            byte_offset: u64,
            dst: &mut [u8],
        ) -> Result<bool> {
            self.inner.read_value_segment_range(guid, byte_offset, dst)
        }

        fn publish_read_index(&self, guid: BlobGuid, bytes: &[u8], values: &[u8]) -> Result<()> {
            self.inner.publish_read_index(guid, bytes, values)
        }

        fn delete_read_index(&self, guid: BlobGuid) -> Result<()> {
            self.inner.delete_read_index(guid)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            self.inner.write_blobs(writes)
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
            self.inner.flush()
        }

        fn needs_flush(&self) -> bool {
            self.inner.needs_flush()
        }

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }

        fn store_stats(&self) -> StoreStats {
            self.inner.store_stats()
        }
    }

    #[test]
    fn file_cache_budget_splits_cold_aux_inside_total() {
        let budget = CacheBudget::file(256);

        assert_eq!(budget.read_page_bytes, 8 * 1024 * 1024);
        assert_eq!(budget.read_index_bytes, 56 * 1024 * 1024);
        assert_eq!(budget.blob_slots, 128);
    }

    #[test]
    fn small_file_cache_budget_preserves_blob_slots() {
        let budget = CacheBudget::file(16);

        assert_eq!(budget.read_page_bytes, 0);
        assert_eq!(budget.read_index_bytes, 0);
        assert_eq!(budget.blob_slots, 16);
    }

    #[test]
    fn memory_cache_budget_disables_cold_aux() {
        let budget = CacheBudget::memory(256);

        assert_eq!(budget.read_page_bytes, 0);
        assert_eq!(budget.read_index_bytes, 0);
        assert_eq!(budget.blob_slots, 256);
    }

    #[test]
    fn indexed_read_eligible_waits_for_store_flush() {
        let guid = [0xCE; 16];
        let inner = Arc::new(FlushPendingStore::new());
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let store: Arc<dyn BlobStore> = inner.clone();
        let bm = BufferManager::new_file(store, 128, AlignedBlobBuf::zeroed);

        assert!(
            !bm.indexed_read_eligible(guid),
            "partial indexed reads must not bypass a store with unflushed data or manifest state",
        );

        inner.flush().unwrap();
        assert!(
            bm.indexed_read_eligible(guid),
            "once the store is durable and the blob is not cached/protected, indexed reads may proceed",
        );
    }

    #[test]
    fn read_index_tokens_are_removed_after_final_delete() {
        let guid = [0xC1; 16];
        let inner = Arc::new(MemoryBlobStore::new());
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let store: Arc<dyn BlobStore> = inner.clone();
        let bm = BufferManager::new(store, 4);

        let token = bm.ensure_read_index_token(guid);
        assert!(bm.read_index_token_valid(guid, token));
        assert_eq!(bm.stats().read_index_token_count, 1);

        bm.delete_blob(guid).unwrap();
        assert_eq!(bm.stats().read_index_token_count, 0);
        assert!(!bm.read_index_token_valid(guid, token));
    }

    #[test]
    fn reintroduced_guid_gets_fresh_read_index_token() {
        let guid = [0xC2; 16];
        let inner = Arc::new(MemoryBlobStore::new());
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store: Arc<dyn BlobStore> = inner.clone();
        let bm = BufferManager::new(store, 4);

        let first = bm.ensure_read_index_token(guid);
        bm.delete_blob(guid).unwrap();
        inner.write_blob(guid, &make_buf(2)).unwrap();
        let second = bm.ensure_read_index_token(guid);

        assert_ne!(first, second);
        assert!(!bm.read_index_token_valid(guid, first));
        assert!(bm.read_index_token_valid(guid, second));
    }

    #[test]
    fn read_caches_after_first_load() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0xAB; 16], &make_buf(7)).unwrap();

        let bm = BufferManager::new(inner.clone(), 4);
        assert_eq!(bm.cached_count(), 0);

        // First read: miss + populate.
        let mut dst = AlignedBlobBuf::zeroed();
        bm.read_blob([0xAB; 16], &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 7);
        assert_eq!(bm.cached_count(), 1);

        // Second read: hit, no growth in cache size.
        bm.read_blob([0xAB; 16], &mut dst).unwrap();
        assert_eq!(bm.cached_count(), 1);
    }

    #[test]
    fn pin_scan_many_returns_each_blob_in_order_and_none_for_missing() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..10u8 {
            inner.write_blob([i; 16], &make_buf(i)).unwrap();
        }
        let bm = BufferManager::new(inner, 64);

        // A batch of >1 guids (exercises the concurrent fan-out) with a
        // missing guid in the middle.
        let mut guids: Vec<BlobGuid> = (0..10u8).map(|i| [i; 16]).collect();
        guids.insert(5, [0xFF; 16]);

        let pins = bm.pin_scan_many(&guids);
        assert_eq!(pins.len(), guids.len());
        for (g, pin) in guids.iter().zip(&pins) {
            if *g == [0xFF; 16] {
                assert!(pin.is_none(), "missing guid must map to None");
            } else {
                let pin = pin.as_ref().expect("present guid must be pinned");
                // make_buf(i) stamped byte 100 = i = g[0]: confirms each
                // entry is the blob for exactly its guid, in order.
                assert_eq!(pin.read().as_slice()[100], g[0]);
            }
        }
    }

    #[test]
    fn pin_miss_is_not_counted_as_a_hit() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0xCD; 16];
        inner.write_blob(guid, &make_buf(9)).unwrap();

        let bm = BufferManager::new(inner, 4);
        let first = bm.pin(guid).unwrap();
        assert_eq!(first.read().as_slice()[100], 9);
        drop(first);
        let stats = bm.stats();
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);

        let second = bm.pin(guid).unwrap();
        assert_eq!(second.read().as_slice()[100], 9);
        let stats = bm.stats();
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.full_blob_reads, 1);
        assert_eq!(stats.full_blob_read_bytes, PAGE_SIZE as u64);
    }

    #[test]
    fn full_blob_reads_are_classified_by_access_path() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..3u8 {
            let mut guid = [0u8; 16];
            guid[0] = i;
            inner.write_blob(guid, &make_buf(i)).unwrap();
        }

        let bm = BufferManager::new(inner, 4);
        drop(bm.pin([0; 16]).unwrap());

        let mut scan = [0u8; 16];
        scan[0] = 1;
        drop(bm.pin_scan(scan).unwrap());

        let mut silent = [0u8; 16];
        silent[0] = 2;
        drop(bm.pin_silent(silent).unwrap());

        let stats = bm.stats();
        assert_eq!(stats.full_blob_reads, 3);
        assert_eq!(stats.full_blob_read_bytes, 3 * PAGE_SIZE as u64);
        assert_eq!(stats.point_full_blob_reads, 1);
        assert_eq!(stats.scan_full_blob_reads, 1);
        assert_eq!(stats.silent_full_blob_reads, 1);
        assert_eq!(
            stats.cache_misses, 2,
            "silent miss does not count as a public cache miss"
        );
        assert_eq!(stats.cache_hits, 0);

        drop(bm.pin([0; 16]).unwrap());
        let stats = bm.stats();
        assert_eq!(
            stats.full_blob_reads, 3,
            "cache hits must not count as store reads"
        );
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn scan_misses_do_not_evict_hot_point_blob() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..5u8 {
            let mut guid = [0u8; 16];
            guid[0] = i;
            inner.write_blob(guid, &make_buf(i)).unwrap();
        }

        let bm = BufferManager::new(inner, 2);
        let hot = [0u8; 16];
        drop(bm.pin(hot).unwrap());

        for i in 1..5u8 {
            let mut guid = [0u8; 16];
            guid[0] = i;
            drop(bm.pin_scan(guid).unwrap());
        }

        assert_eq!(bm.cached_count(), 2);
        assert!(
            bm.cache.contains_key(&hot),
            "scan-loaded blobs must stay colder than point-read blobs",
        );
    }

    #[test]
    fn scan_miss_may_overshoot_instead_of_evicting_only_hot_blob() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..2u8 {
            let mut guid = [0u8; 16];
            guid[0] = i;
            inner.write_blob(guid, &make_buf(i)).unwrap();
        }

        let bm = BufferManager::new(inner, 1);
        let hot = [0u8; 16];
        let mut scan = [0u8; 16];
        scan[0] = 1;

        drop(bm.pin(hot).unwrap());
        drop(bm.pin_scan(scan).unwrap());

        assert!(
            bm.cache.contains_key(&hot),
            "scan miss must not evict the only point-hot blob",
        );
        assert_eq!(
            bm.cached_count(),
            2,
            "scan access may briefly exceed capacity to avoid hot-cache pollution",
        );
    }

    #[test]
    fn scan_hits_do_not_refresh_recency() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..3u8 {
            let mut guid = [0u8; 16];
            guid[0] = i;
            inner.write_blob(guid, &make_buf(i)).unwrap();
        }

        let bm = BufferManager::new(inner, 2);
        let first = [0u8; 16];
        let mut second = [0u8; 16];
        second[0] = 1;
        let mut third = [0u8; 16];
        third[0] = 2;

        drop(bm.pin(first).unwrap());
        drop(bm.pin(second).unwrap());
        drop(bm.pin_scan(first).unwrap());
        drop(bm.pin(third).unwrap());

        assert!(
            !bm.cache.contains_key(&first),
            "a scan hit must not make the oldest point blob look hot",
        );
        assert!(bm.cache.contains_key(&second));
        assert!(bm.cache.contains_key(&third));
    }

    #[test]
    fn frequency_aware_eviction_stays_at_capacity() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..10u8 {
            let mut g = [0u8; 16];
            g[0] = i;
            inner.write_blob(g, &make_buf(i)).unwrap();
        }

        let bm = BufferManager::new(inner, 4);
        for i in 0..10u8 {
            let mut g = [0u8; 16];
            g[0] = i;
            let mut dst = AlignedBlobBuf::zeroed();
            bm.read_blob(g, &mut dst).unwrap();
        }
        assert_eq!(
            bm.cached_count(),
            4,
            "cache must shrink to capacity after over-fill",
        );

        // The most-recently-loaded GUIDs should be the survivors.
        let mut g_last = [0u8; 16];
        g_last[0] = 9;
        let mut g_first = [0u8; 16];
        g_first[0] = 0;
        assert!(bm.cache.contains_key(&g_last));
        assert!(!bm.cache.contains_key(&g_first));
    }

    #[test]
    fn tinylfu_keeps_frequent_point_blob_against_one_hit_stream() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..12u8 {
            let mut g = [0u8; 16];
            g[0] = i;
            inner.write_blob(g, &make_buf(i)).unwrap();
        }

        let bm = BufferManager::new(inner, 2);
        let hot = [0u8; 16];
        for _ in 0..8 {
            drop(bm.pin(hot).unwrap());
        }

        for i in 1..12u8 {
            let mut cold = [0u8; 16];
            cold[0] = i;
            drop(bm.pin(cold).unwrap());
            assert!(
                bm.cache.contains_key(&hot),
                "frequent point blob should survive one-hit stream pressure",
            );
            assert!(
                bm.cached_count() <= 2,
                "unprotected one-hit blobs should be reclaimed immediately",
            );
        }
    }

    #[test]
    fn route_resident_anchor_survives_inline_eviction() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..9u8 {
            let mut g = [0u8; 16];
            g[0] = i;
            inner.write_blob(g, &make_buf(i)).unwrap();
        }

        let bm = BufferManager::new(inner, 8);
        let anchor = [0u8; 16];
        let mut dst = AlignedBlobBuf::zeroed();
        bm.read_blob(anchor, &mut dst).unwrap();
        bm.mark_route_resident(anchor);

        for i in 1..9u8 {
            let mut g = [0u8; 16];
            g[0] = i;
            bm.read_blob(g, &mut dst).unwrap();
        }

        assert_eq!(bm.cached_count(), 8);
        assert!(bm.cache.contains_key(&anchor));
        assert!(bm.is_route_resident(anchor));
        let mut first_non_route = [0u8; 16];
        first_non_route[0] = 1;
        assert!(
            !bm.cache.contains_key(&first_non_route),
            "oldest non-route blob should be evicted first",
        );
    }

    #[test]
    fn route_resident_tier_demotes_old_anchors_at_budget() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(inner, 4);

        bm.mark_route_resident([1; 16]);
        bm.mark_route_resident([2; 16]);

        assert_eq!(bm.route_resident_count(), 1);
        assert_eq!(bm.stats().route_resident_demotions, 1);
        assert!(!bm.is_route_resident([1; 16]));
        assert!(bm.is_route_resident([2; 16]));
    }

    /// Regression: prior to the v0.2.1 fix, inline eviction only
    /// checked `Arc::strong_count == 1` — it would happily evict
    /// a dirty cache image, leaving the dirty entry orphaned in
    /// the dirty map. That broke invariant I1 (dirty ⟺ cache
    /// newer than store) and silently lost the cache mutation
    /// (memory mode) / stuck the WAL truncate gate forever
    /// (persistent mode).
    #[test]
    fn inline_eviction_skips_dirty_entries() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        // Pre-populate the inner store with three blobs whose
        // bytes we'll be able to distinguish.
        for i in 0..3u8 {
            let mut g = [0u8; 16];
            g[0] = i;
            inner.write_blob(g, &make_buf(i)).unwrap();
        }

        // Capacity 2 — any third load must trigger overflow.
        let bm = BufferManager::new(inner, 2);

        let g_a = {
            let mut g = [0u8; 16];
            g[0] = 0;
            g
        };
        let g_b = {
            let mut g = [0u8; 16];
            g[0] = 1;
            g
        };
        let g_c = {
            let mut g = [0u8; 16];
            g[0] = 2;
            g
        };

        // Pin + dirty A. The pin is released right away; only
        // the dirty entry should keep A from being evicted.
        {
            let _pin = bm.pin(g_a).unwrap();
        }
        bm.mark_dirty(g_a, 10);
        assert_eq!(bm.dirty_count(), 1);
        assert!(bm.cache.contains_key(&g_a));

        // Load B (cache now at capacity = 2).
        {
            let _pin = bm.pin(g_b).unwrap();
        }
        assert!(bm.cache.contains_key(&g_a));
        assert!(bm.cache.contains_key(&g_b));

        // Load C — this must trigger overflow eviction. Pre-fix
        // it would pick A (oldest by tick); post-fix it must
        // skip A and pick B.
        {
            let _pin = bm.pin(g_c).unwrap();
        }

        assert!(
            bm.cache.contains_key(&g_a),
            "dirty entry A's cache image must survive inline eviction",
        );
        assert!(
            bm.cache.contains_key(&g_c),
            "newly-pinned C must be in cache",
        );
        // B (clean, oldest after A is protected) is the victim.
        assert!(
            !bm.cache.contains_key(&g_b),
            "B (clean, no pin) should have been evicted in A's stead",
        );
        // The dirty entry for A is still tracked.
        assert_eq!(
            bm.dirty_count(),
            1,
            "dirty bookkeeping must not be touched by eviction",
        );

        // And a versioned snapshot of A must still succeed — the invariant
        // downstream checkpoint code relies on.
        let _ = snapshot_current_bytes(&bm, g_a);
    }

    #[test]
    fn maintenance_candidates_are_unique_and_fifo_budgeted() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(inner, 4);
        let mut buckets = vec![Vec::<BlobGuid>::new(); BOOKKEEPING_SHARDS];
        for i in 0..=u8::MAX {
            let mut g = [0u8; 16];
            g[0] = i;
            buckets[bookkeeping_shard_idx(&g)].push(g);
        }
        let same_shard = buckets.into_iter().find(|b| b.len() >= 3).unwrap();
        let a = same_shard[0];
        let b = same_shard[1];
        let c = same_shard[2];

        bm.note_compaction_candidate(a);
        bm.note_compaction_candidate(b);
        bm.note_compaction_candidate(a);
        bm.note_compaction_candidate(c);

        assert_eq!(bm.compaction_candidate_count(), 3);
        assert_eq!(bm.pop_compaction_candidates(2), vec![a, b]);
        assert_eq!(bm.compaction_candidate_count(), 1);

        // Re-queued candidates go to the back rather than
        // starving entries that were already waiting.
        bm.note_compaction_candidate(a);
        assert_eq!(bm.pop_compaction_candidates(8), vec![c, a]);
        assert_eq!(bm.compaction_candidate_count(), 0);
    }

    #[test]
    fn maintenance_candidate_drain_rotates_across_shards() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(inner, 4);
        let mut by_shard = [None::<BlobGuid>; BOOKKEEPING_SHARDS];
        let mut counter = 0u32;
        while by_shard.iter().any(Option::is_none) {
            assert!(
                counter < 100_000,
                "test helper could not cover every bookkeeping shard"
            );
            let mut guid = [0u8; 16];
            guid[0..4].copy_from_slice(&counter.to_le_bytes());
            let shard = bookkeeping_shard_idx(&guid);
            by_shard[shard].get_or_insert(guid);
            counter += 1;
        }

        for guid in by_shard.iter().flatten() {
            bm.note_compaction_candidate(*guid);
        }

        for expected_shard in 0..4 {
            let batch = bm.pop_compaction_candidates(1);
            assert_eq!(batch.len(), 1);
            assert_eq!(bookkeeping_shard_idx(&batch[0]), expected_shard);
        }
    }

    // Note on pending-delete + cache: `mark_for_delete` removes
    // the cache image (`self.cache.remove(&guid)`) in the same
    // call as it queues the pending-delete. `pin` / `has_blob`
    // must still treat pending-delete as a visibility barrier
    // because the inner store manifest intentionally keeps the
    // blob until checkpoint applies the deferred delete.

    #[test]
    fn write_through_propagates_to_inner_store() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(inner.clone(), 4);

        bm.write_blob([0xCD; 16], &make_buf(0x42)).unwrap();

        // Inner sees the blob immediately (write-through).
        assert!(inner.has_blob([0xCD; 16]).unwrap());
        let mut dst = AlignedBlobBuf::zeroed();
        inner.read_blob([0xCD; 16], &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 0x42);
    }

    #[test]
    fn write_through_updates_cache_if_present() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0xEF; 16], &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        // Prime the cache.
        let mut dst = AlignedBlobBuf::zeroed();
        bm.read_blob([0xEF; 16], &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);

        // Overwrite via the BM.
        bm.write_blob([0xEF; 16], &make_buf(99)).unwrap();

        // Subsequent read through the BM sees the updated value
        // (came from the refreshed cache, not the inner store).
        bm.read_blob([0xEF; 16], &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 99);
    }

    #[test]
    fn delete_evicts_from_cache_and_inner() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0x33; 16], &make_buf(5)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        // Prime cache.
        let mut dst = AlignedBlobBuf::zeroed();
        bm.read_blob([0x33; 16], &mut dst).unwrap();
        assert_eq!(bm.cached_count(), 1);

        bm.delete_blob([0x33; 16]).unwrap();
        assert_eq!(bm.cached_count(), 0);
        assert!(!inner.has_blob([0x33; 16]).unwrap());
        assert!(!bm.has_blob([0x33; 16]).unwrap());
    }

    #[test]
    fn pending_delete_hides_blob_until_checkpoint_delete_applies() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0x44; 16], &make_buf(7)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        let _pin = bm.pin([0x44; 16]).unwrap();
        assert!(bm.has_blob([0x44; 16]).unwrap());
        bm.mark_dirty([0x44; 16], 10);
        bm.mark_for_delete([0x44; 16], 11);

        assert!(inner.has_blob([0x44; 16]).unwrap());
        assert!(!bm.has_blob([0x44; 16]).unwrap());
        assert!(
            bm.pin([0x44; 16]).is_err(),
            "pending-delete child must not be reloaded from store"
        );
        bm.mark_dirty([0x44; 16], 12);
        let mut restore = HashMap::new();
        restore.insert([0x44; 16], 13);
        bm.restore_dirty(restore);
        assert_eq!(bm.dirty_count(), 0);
        assert_eq!(bm.pending_delete_count(), 1);
    }

    #[test]
    fn pending_delete_count_tracks_snapshot_and_restore() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(inner, 4);
        let guid = [0x55; 16];

        bm.mark_for_delete(guid, 20);
        bm.mark_for_delete(guid, 10);
        assert_eq!(bm.pending_delete_count(), 1);

        let pending = bm.snapshot_pending_deletes();
        assert_eq!(pending.get(&guid), Some(&10));
        assert_eq!(
            bm.pending_delete_count(),
            1,
            "claimed deletes remain fenced while the I/O worker owns them",
        );

        bm.restore_pending_deletes(pending);
        assert_eq!(bm.pending_delete_count(), 1);
        let pending = bm.snapshot_pending_deletes();
        assert_eq!(pending.get(&guid), Some(&10));
        assert_eq!(bm.pending_delete_count(), 1);
    }

    #[test]
    fn claimed_pending_delete_still_hides_blob_from_stale_pins() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x5A; 16];
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        bm.mark_for_delete(guid, 10);
        let pending = bm.snapshot_pending_deletes();
        assert_eq!(pending.get(&guid), Some(&10));
        assert!(inner.has_blob(guid).unwrap());
        assert!(!bm.has_blob(guid).unwrap());
        assert!(
            bm.pin(guid).is_err(),
            "a claimed delete must keep stale walkers from reloading the blob",
        );
        let mut dst = AlignedBlobBuf::zeroed();
        assert!(
            bm.read_blob(guid, &mut dst).is_err(),
            "BlobStore reads must obey the same delete fence as pin()",
        );
        bm.mark_dirty(guid, 11);
        assert_eq!(bm.dirty_count(), 0);
        assert!(bm.write_blob(guid, &make_buf(9)).is_err());
        assert!(bm.delete_blob(guid).is_err());

        assert!(bm.execute_pending_delete(guid, 10).unwrap());
        assert_eq!(bm.pending_delete_count(), 0);
        assert!(!inner.has_blob(guid).unwrap());
    }

    #[test]
    fn structural_detach_waits_for_parent_dirty_before_fifo() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x5B; 16];
        let parent_guid = [0x5A; 16];
        inner.write_blob(guid, &make_buf(7)).unwrap();
        inner.write_blob(parent_guid, &make_buf(8)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);
        let parent = bm.pin(parent_guid).unwrap();

        bm.stage_structural_reclaim(parent_guid, guid);
        assert_eq!(bm.pending_delete_count(), 0);
        assert_eq!(bm.gc_orphan_backlog_count(), 1);
        assert_eq!(bm.orphan_staging_count(), 1);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 0);
        bm.mark_dirty_cached(parent_guid, STRUCTURAL_SEQ, parent.as_ref());
        assert_eq!(bm.orphan_staging_count(), 0);
        assert!(
            inner.has_blob(guid).unwrap(),
            "structural detach must not unlink before parent dirty publication"
        );
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 1);
        assert_eq!(bm.gc_orphan_backlog_count(), 0);
        assert!(!inner.has_blob(guid).unwrap());
    }

    #[test]
    fn structural_orphan_stays_visible_until_last_snapshot_lease_retires() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x5D; 16];
        let root_guid = [0x5E; 16];
        inner.write_blob(guid, &make_buf(7)).unwrap();
        inner.write_blob(root_guid, &make_buf(8)).unwrap();
        let bm = Arc::new(BufferManager::new(inner.clone(), 4));
        let root_pin = bm.pin(root_guid).unwrap();
        let epoch = bm.register_snapshot(root_guid, &root_pin).unwrap();

        bm.stage_structural_reclaim(root_guid, guid);
        bm.mark_dirty_cached(root_guid, STRUCTURAL_SEQ, root_pin.as_ref());
        assert_eq!(bm.pending_delete_count(), 0);
        assert_eq!(bm.gc_orphan_backlog_count(), 1);
        assert_eq!(bm.orphan_staging_count(), 0);
        assert_eq!(
            bm.pin(guid).unwrap().read().as_slice()[100],
            7,
            "structural detach must not fence a snapshot's first lazy child pin"
        );
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 0);
        assert!(inner.has_blob(guid).unwrap());

        bm.retire_snapshot(epoch);
        assert_eq!(bm.gc_orphan_backlog_count(), 1);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 1);
        assert!(!inner.has_blob(guid).unwrap());
    }

    #[test]
    fn snapshot_epoch_exhaustion_never_wraps_or_mutates_live_registry() {
        let root_guid = [0x63; 16];
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob(root_guid, &make_buf(1)).unwrap();
        let bm = Arc::new(BufferManager::new(inner, 4));
        let root = bm.pin(root_guid).unwrap();
        bm.set_current_epoch(u64::MAX - 2);

        let epoch = bm.register_snapshot(root_guid, &root).unwrap();
        assert_eq!(epoch, u64::MAX - 2);
        assert_eq!(bm.current_epoch(), u64::MAX - 1);
        assert_eq!(bm.fork_barrier(), u64::MAX - 2);
        assert_eq!(bm.snapshots.lock().unwrap().live.len(), 1);
        let cached_before = bm.cached_count();
        let dirty_before = bm.dirty_count();

        let error = bm.register_snapshot(root_guid, &root).unwrap_err();
        assert!(matches!(error, Error::SnapshotEpochExhausted));
        assert_eq!(bm.current_epoch(), u64::MAX - 1);
        assert_eq!(bm.fork_barrier(), u64::MAX - 2);
        assert_eq!(bm.snapshots.lock().unwrap().live.len(), 1);
        assert_eq!(bm.cached_count(), cached_before);
        assert_eq!(bm.dirty_count(), dirty_before);

        bm.retire_snapshot(epoch);
        assert_eq!(bm.fork_barrier(), 0);
        assert_eq!(bm.current_epoch(), u64::MAX - 1);
    }

    #[test]
    fn structural_last_lease_drop_before_parent_dirty_stays_staged() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let child_guid = [0x5F; 16];
        let parent_guid = [0x60; 16];
        inner.write_blob(child_guid, &make_buf(7)).unwrap();
        inner.write_blob(parent_guid, &make_buf(8)).unwrap();
        let bm = Arc::new(BufferManager::new(inner.clone(), 4));
        let parent = bm.pin(parent_guid).unwrap();
        let epoch = bm.register_snapshot(parent_guid, &parent).unwrap();

        bm.stage_structural_reclaim(parent_guid, child_guid);
        bm.retire_snapshot(epoch);
        assert_eq!(bm.orphan_staging_count(), 1);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 0);
        assert!(inner.has_blob(child_guid).unwrap());

        bm.mark_dirty_cached(parent_guid, STRUCTURAL_SEQ, parent.as_ref());
        assert_eq!(bm.orphan_staging_count(), 0);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 1);
        assert!(!inner.has_blob(child_guid).unwrap());
    }

    #[test]
    fn cow_epoch_zero_last_lease_drop_before_parent_dirty_stays_staged() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let child_guid = [0x61; 16];
        let parent_guid = [0x62; 16];
        inner.write_blob(child_guid, &make_buf(7)).unwrap();
        inner.write_blob(parent_guid, &make_buf(8)).unwrap();
        let bm = Arc::new(BufferManager::new(inner.clone(), 4));
        let parent = bm.pin(parent_guid).unwrap();
        let epoch = bm.register_snapshot(parent_guid, &parent).unwrap();

        // Epoch zero is valid for frames written by older Holt versions.
        bm.stage_cow_reclaim(parent_guid, child_guid, 0);
        bm.retire_snapshot(epoch);
        assert_eq!(bm.orphan_staging_count(), 1);
        assert_eq!(bm.gc_orphan_backlog_count(), 1);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 0);
        assert!(inner.has_blob(child_guid).unwrap());

        bm.mark_dirty_cached(parent_guid, 9, parent.as_ref());
        assert_eq!(bm.orphan_staging_count(), 0);
        assert_eq!(bm.reclaim_retired_orphans_bounded(1).unwrap(), 1);
        assert!(!inner.has_blob(child_guid).unwrap());
    }

    #[test]
    fn pending_delete_defers_until_existing_pin_drops() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x5C; 16];
        inner.write_blob(guid, &make_buf(7)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);
        let pin = bm.pin(guid).unwrap();

        bm.mark_for_delete(guid, 10);
        let pending = bm.snapshot_pending_deletes();
        assert!(
            !bm.execute_pending_delete(guid, 10).unwrap(),
            "delete must wait while an old walker still holds a cached blob pin",
        );
        bm.restore_pending_deletes(pending);

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x66;
        }
        bm.mark_dirty_cached(guid, 11, pin.as_ref());
        assert_eq!(
            bm.dirty_count(),
            0,
            "existing pins must not publish orphan dirty state while delete-fenced",
        );
        drop(pin);

        let pending = bm.snapshot_pending_deletes();
        assert_eq!(pending.get(&guid), Some(&10));
        assert!(bm.execute_pending_delete(guid, 10).unwrap());
        assert_eq!(bm.pending_delete_count(), 0);
        assert!(!inner.has_blob(guid).unwrap());
    }

    #[test]
    fn has_blob_fast_path_avoids_inner_when_cached() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0x77; 16], &make_buf(11)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        // Prime cache.
        let mut dst = AlignedBlobBuf::zeroed();
        bm.read_blob([0x77; 16], &mut dst).unwrap();

        assert!(bm.has_blob([0x77; 16]).unwrap());
        // Sanity: uncached GUID still works (inner check).
        assert!(!bm.has_blob([0x88; 16]).unwrap());
    }

    // ---------- dirty-tracking tests ----------

    #[test]
    fn mark_dirty_keeps_lowest_seq() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x01; 16];
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin(guid).unwrap();

        bm.mark_dirty(guid, 50);
        bm.mark_dirty(guid, 30);
        bm.mark_dirty(guid, 99);
        assert_eq!(bm.dirty_count(), 1);
        let snap = bm.snapshot_dirty();
        assert_eq!(snap[&guid], 30);
    }

    #[test]
    fn mark_dirty_without_cache_image_does_not_publish_orphan() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0xAB; 16];
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);

        bm.mark_dirty(guid, 10);

        assert!(
            bm.snapshot_dirty().is_empty(),
            "dirty map must not contain an entry without a cache image",
        );
    }

    #[test]
    fn cached_dirty_hint_resets_after_snapshot() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0xD1; 16];
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin(guid).unwrap();

        bm.mark_dirty(guid, 10);
        bm.mark_dirty(guid, 20);
        let snap = bm.snapshot_dirty();
        assert_eq!(snap[&guid], 10);
        assert_eq!(bm.dirty_count(), 0);

        bm.mark_dirty(guid, 30);
        let next = bm.snapshot_dirty();
        assert_eq!(
            next[&guid], 30,
            "mark_dirty after snapshot must publish a fresh dirty entry",
        );
    }

    #[test]
    fn stale_dirty_hint_cannot_skip_dirty_map_publish() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0xD3; 16];
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let pin = bm.pin(guid).unwrap();

        assert!(pin.dirty_hint_needs_map_publish(10));
        bm.mark_dirty(guid, 20);

        let snap = bm.snapshot_dirty();
        assert_eq!(
            snap[&guid], 20,
            "a stale hint without a dirty-map entry must not hide a fresh write",
        );
    }

    #[test]
    fn cached_dirty_hint_preserves_lower_restored_seq() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0xD2; 16];
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin(guid).unwrap();

        let mut restored = HashMap::new();
        restored.insert(guid, 40);
        bm.restore_dirty(restored);
        bm.mark_dirty(guid, 90);
        let snap = bm.snapshot_dirty();
        assert_eq!(
            snap[&guid], 40,
            "duplicate higher seq must be covered by restored low-watermark",
        );

        bm.restore_dirty(snap);
        bm.mark_dirty(guid, 20);
        let lowered = bm.snapshot_dirty();
        assert_eq!(
            lowered[&guid], 20,
            "lower seq must still update the dirty low-watermark",
        );
    }

    #[test]
    fn snapshot_dirty_drains_atomically() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for guid in [[0x01; 16], [0x02; 16], [0x03; 16]] {
            inner.write_blob(guid, &make_buf(1)).unwrap();
        }
        let bm = BufferManager::new(inner, 4);
        let _p1 = bm.pin([0x01; 16]).unwrap();
        let _p2 = bm.pin([0x02; 16]).unwrap();
        bm.mark_dirty([0x01; 16], 10);
        bm.mark_dirty([0x02; 16], 20);

        let snap = bm.snapshot_dirty();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[&[0x01; 16]], 10);
        assert_eq!(snap[&[0x02; 16]], 20);

        // After snapshot the live map is empty.
        assert_eq!(bm.dirty_count(), 0);

        // Concurrent mark_dirty lands in the fresh empty map.
        let _p3 = bm.pin([0x03; 16]).unwrap();
        bm.mark_dirty([0x03; 16], 99);
        assert_eq!(bm.dirty_count(), 1);
        let next = bm.snapshot_dirty();
        assert_eq!(next[&[0x03; 16]], 99);
    }

    #[test]
    fn snapshot_dirty_drains_every_bookkeeping_shard() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(Arc::clone(&inner), BOOKKEEPING_SHARDS);
        let mut guids: [Option<BlobGuid>; BOOKKEEPING_SHARDS] = [None; BOOKKEEPING_SHARDS];

        for i in 0..20_000u64 {
            let mut guid = [0u8; 16];
            guid[0..8].copy_from_slice(&i.to_le_bytes());
            guid[8..16].copy_from_slice(&i.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());
            let shard = bookkeeping_shard_idx(&guid);
            guids[shard].get_or_insert(guid);
            if guids.iter().all(Option::is_some) {
                break;
            }
        }

        assert!(
            guids.iter().all(Option::is_some),
            "test generator should hit every bookkeeping shard"
        );
        for (shard, guid) in guids.iter().enumerate() {
            let guid = guid.expect("filled");
            inner.write_blob(guid, &make_buf(1)).unwrap();
            let _pin = bm.pin(guid).unwrap();
            bm.mark_dirty(guid, shard as u64 + 1);
        }

        let snap = bm.snapshot_dirty();
        assert_eq!(snap.len(), BOOKKEEPING_SHARDS);
        assert_eq!(bm.dirty_count(), 0);
        for (shard, guid) in guids.iter().enumerate() {
            assert_eq!(snap[&guid.expect("filled")], shard as u64 + 1);
        }
    }

    #[test]
    fn snapshot_dirty_protects_flushing_entry_from_eviction() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x55; 16];
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 1);

        {
            let pin = bm.pin(guid).unwrap();
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0xAB;
        }
        bm.mark_dirty(guid, 42);

        let snap = bm.snapshot_dirty();
        assert_eq!(snap[&guid], 42);
        assert_eq!(
            bm.dirty_count(),
            0,
            "snapshot drains the live dirty map for racing writers",
        );

        assert!(
            !bm.try_evict_cold(guid),
            "checkpoint-owned flushing entries must stay cached until write-through",
        );
        let bytes = snapshot_current_bytes(&bm, guid);
        assert_eq!(bytes.as_slice()[123], 0xAB);

        bm.write_through_batch(&[WriteThroughEntry {
            guid,
            bytes,
            expected_seq: 42,
            content_version: None,
        }])
        .unwrap();
        assert!(
            bm.try_evict_cold(guid),
            "successful write-through releases flushing protection",
        );
    }

    #[test]
    fn cow_reclaim_does_not_drop_flushing_cache_image() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x56; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 1);
        let pin = bm.pin(guid).unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0xAA;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        assert_eq!(snap[&guid], 10);
        drop(pin);

        bm.discard_snapshot_root(guid);

        let version = bm.snapshot_dirty_versions(&snap).unwrap()[0].content_version;
        let bytes = bm
            .snapshot_bytes_if_version(guid, version)
            .unwrap()
            .expect("COW reclaim must not drop checkpoint-owned bytes");
        assert_eq!(bytes.as_slice()[123], 0xAA);
        assert!(inner.has_blob(guid).unwrap());
    }

    #[test]
    fn cow_reclaim_does_not_drop_pinned_cache_image() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x56; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 1);
        let pin = bm.pin(guid).unwrap();

        bm.discard_snapshot_root(guid);

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0xBB;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());

        let snap = bm.snapshot_dirty();
        let version = bm.snapshot_dirty_versions(&snap).unwrap()[0].content_version;
        let bytes = bm
            .snapshot_bytes_if_version(guid, version)
            .unwrap()
            .expect("pinned dirty image must stay reachable through cache");
        assert_eq!(bytes.as_slice()[123], 0xBB);
    }

    #[test]
    fn snapshot_bytes_if_version_rejects_stale_blob_image() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x56; 16];
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 1);
        let pin = bm.pin(guid).unwrap();

        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        let versioned = bm.snapshot_dirty_versions(&snap).unwrap();
        assert_eq!(versioned.len(), 1);

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0xEE;
        }

        assert!(
            bm.snapshot_bytes_if_version(guid, versioned[0].content_version)
                .unwrap()
                .is_none(),
            "checkpoint clone must reject bytes after a newer blob mutation"
        );
        let bytes = bm
            .snapshot_bytes_if_version(guid, pin.content_version())
            .unwrap()
            .expect("current version should clone");
        assert_eq!(bytes.as_slice()[123], 0xEE);
    }

    #[test]
    fn write_through_rejects_stale_snapshot_bytes() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x57; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 1);
        let pin = bm.pin(guid).unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x11;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        let versioned = bm.snapshot_dirty_versions(&snap).unwrap();
        let stale_version = versioned[0].content_version;
        let stale_bytes = bm
            .snapshot_bytes_if_version(guid, stale_version)
            .unwrap()
            .expect("snapshot bytes should clone while version still matches");

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0xEE;
        }
        bm.mark_dirty_cached(guid, 20, pin.as_ref());

        let report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes: stale_bytes,
                expected_seq: 10,
                content_version: Some(stale_version),
            }])
            .unwrap();
        assert_eq!(report.statuses, vec![WriteThroughStatus::Stale]);
        assert_eq!(
            bm.snapshot_dirty()[&guid],
            20,
            "newer writer entry must survive stale write-through retirement",
        );

        let mut stored = AlignedBlobBuf::zeroed();
        inner.read_blob(guid, &mut stored).unwrap();
        assert_eq!(
            stored.as_slice()[123],
            0,
            "stale checkpoint bytes must not overwrite the store"
        );
    }

    #[test]
    fn overlapping_checkpoint_epochs_keep_cache_image_protected() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x58; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner, 1);
        let pin = bm.pin(guid).unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x11;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let first = bm.snapshot_dirty();
        assert_eq!(bm.flushing_count(), 1);
        let first_version = bm.snapshot_dirty_versions(&first).unwrap()[0].content_version;
        let first_bytes = bm
            .snapshot_bytes_if_version(guid, first_version)
            .unwrap()
            .unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x22;
        }
        bm.mark_dirty_cached(guid, 20, pin.as_ref());
        let second = bm.snapshot_dirty();
        assert_eq!(bm.flushing_count(), 2);
        let second_version = bm.snapshot_dirty_versions(&second).unwrap()[0].content_version;
        let second_bytes = bm
            .snapshot_bytes_if_version(guid, second_version)
            .unwrap()
            .unwrap();
        drop(pin);

        let first_report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes: first_bytes,
                expected_seq: 10,
                content_version: Some(first_version),
            }])
            .unwrap();
        assert_eq!(first_report.statuses, vec![WriteThroughStatus::Stale]);
        bm.restore_dirty(first);
        assert!(
            !bm.try_evict_cold(guid),
            "second in-flight epoch must keep the blob cached after first retire",
        );
        assert_eq!(bm.flushing_count(), 1);

        let second_report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes: second_bytes,
                expected_seq: 20,
                content_version: Some(second_version),
            }])
            .unwrap();
        assert_eq!(second_report.statuses, vec![WriteThroughStatus::Written]);
        assert!(
            bm.try_evict_cold(guid),
            "last in-flight epoch can release eviction protection",
        );
        assert_eq!(bm.flushing_count(), 0);
    }

    #[test]
    fn pending_delete_preserves_in_flight_checkpoint_image() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x59; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 1);
        let pin = bm.pin(guid).unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x33;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        let version = bm.snapshot_dirty_versions(&snap).unwrap()[0].content_version;
        let bytes = bm
            .snapshot_bytes_if_version(guid, version)
            .unwrap()
            .unwrap();
        drop(pin);

        bm.mark_for_delete(guid, 20);

        assert_eq!(
            bm.flushing_count(),
            1,
            "a pending delete must not retire an in-flight checkpoint epoch",
        );
        assert!(
            bm.cache.contains_key(&guid),
            "a pending delete must keep the cache image needed by write-through validation",
        );

        let report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes,
                expected_seq: 10,
                content_version: Some(version),
            }])
            .unwrap();
        assert_eq!(report.statuses, vec![WriteThroughStatus::Written]);
        assert_eq!(bm.flushing_count(), 0);
        assert_eq!(bm.pending_delete_count(), 1);
        assert!(
            bm.pin(guid).is_err(),
            "pending delete must still hide the blob"
        );

        let mut stored = AlignedBlobBuf::zeroed();
        inner.read_blob(guid, &mut stored).unwrap();
        assert_eq!(
            stored.as_slice()[123],
            0x33,
            "checkpoint write-through must preserve the durable image until delete applies",
        );
    }

    #[test]
    fn execute_pending_delete_defers_while_blob_is_flushing() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x5B; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 1);
        let pin = bm.pin(guid).unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x44;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        let version = bm.snapshot_dirty_versions(&snap).unwrap()[0].content_version;
        let bytes = bm
            .snapshot_bytes_if_version(guid, version)
            .unwrap()
            .unwrap();
        drop(pin);

        bm.mark_for_delete(guid, 20);
        let pending = bm.snapshot_pending_deletes();
        assert_eq!(pending.get(&guid), Some(&20));
        assert!(
            !bm.execute_pending_delete(guid, 20).unwrap(),
            "delete must wait for the in-flight checkpoint image to retire",
        );
        assert!(inner.has_blob(guid).unwrap());
        bm.restore_pending_deletes(pending);
        assert_eq!(bm.pending_delete_count(), 1);

        let report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes,
                expected_seq: 10,
                content_version: Some(version),
            }])
            .unwrap();
        assert_eq!(report.statuses, vec![WriteThroughStatus::Written]);
        let pending = bm.snapshot_pending_deletes();
        assert!(bm.execute_pending_delete(guid, 20).unwrap());
        assert!(!inner.has_blob(guid).unwrap());
        assert_eq!(bm.pending_delete_count(), 0);
        assert_eq!(pending.get(&guid), Some(&20));
    }

    #[test]
    fn write_through_does_not_clear_in_flight_checkpoint_owner() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x5C; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 1);
        let pin = bm.pin(guid).unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x55;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        let version = bm.snapshot_dirty_versions(&snap).unwrap()[0].content_version;
        let bytes = bm
            .snapshot_bytes_if_version(guid, version)
            .unwrap()
            .unwrap();
        drop(pin);

        bm.write_blob(guid, &make_buf(0x66)).unwrap();

        assert_eq!(
            bm.flushing_count(),
            1,
            "direct write-through must not retire another checkpoint epoch",
        );
        assert!(
            bm.cache.contains_key(&guid),
            "direct write-through must keep the image required by version validation",
        );

        let report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes,
                expected_seq: 10,
                content_version: Some(version),
            }])
            .unwrap();
        assert_eq!(report.statuses, vec![WriteThroughStatus::Stale]);
    }

    #[test]
    fn delete_blob_rejects_in_flight_checkpoint_owner() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0x5D; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 1);
        let pin = bm.pin(guid).unwrap();

        {
            let mut guard = pin.write();
            guard.as_mut_slice()[123] = 0x77;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        let version = bm.snapshot_dirty_versions(&snap).unwrap()[0].content_version;
        let bytes = bm
            .snapshot_bytes_if_version(guid, version)
            .unwrap()
            .unwrap();
        drop(pin);

        assert!(bm.delete_blob(guid).is_err());
        assert_eq!(bm.flushing_count(), 1);
        assert!(bm.cache.contains_key(&guid));
        assert!(inner.has_blob(guid).unwrap());

        let report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes,
                expected_seq: 10,
                content_version: Some(version),
            }])
            .unwrap();
        assert_eq!(report.statuses, vec![WriteThroughStatus::Written]);
    }

    #[test]
    fn restore_dirty_merges_keeping_min() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for guid in [[0x01; 16], [0x02; 16], [0x03; 16]] {
            inner.write_blob(guid, &make_buf(1)).unwrap();
        }
        let bm = BufferManager::new(inner, 4);
        let _p1 = bm.pin([0x01; 16]).unwrap();
        let _p2 = bm.pin([0x02; 16]).unwrap();
        let _p3 = bm.pin([0x03; 16]).unwrap();
        // Pretend a flush snapshot drained these:
        let mut snap = HashMap::new();
        snap.insert([0x01; 16], 10);
        snap.insert([0x02; 16], 20);
        // Meanwhile a racing writer added a newer-seq entry for 0x01:
        bm.mark_dirty([0x01; 16], 50);
        // ...and a fresh blob 0x03:
        bm.mark_dirty([0x03; 16], 5);

        bm.restore_dirty(snap);

        // 0x01: min(50, 10) = 10. 0x02: 20. 0x03: 5 (untouched).
        assert_eq!(bm.dirty_count(), 3);
        let live = bm.snapshot_dirty();
        assert_eq!(live[&[0x01; 16]], 10);
        assert_eq!(live[&[0x02; 16]], 20);
        assert_eq!(live[&[0x03; 16]], 5);
    }

    #[test]
    fn write_through_keeps_racing_writer_dirty_entry() {
        // Reproduces the dirty-race fix: a checkpointer drains the
        // dirty map at snapshot time (snap_seq=50), then before
        // checkpoint write-through runs an in-process writer marks the
        // same blob dirty with a newer seq (200). The writer's
        // mutation is NOT in our snapshot bytes, so the entry
        // must survive the retire path.
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0xAA; 16], &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin([0xAA; 16]).unwrap();

        // Simulate the planner's drain by manually setting up the
        // "post-drain" state: dirty contains a NEW writer's entry.
        bm.mark_dirty([0xAA; 16], 200);
        let snap_bytes = snapshot_current_bytes(&bm, [0xAA; 16]);

        // The planner's snap had captured snap_seq=50 (a stale
        // pre-drain value). Pass that through.
        bm.write_through_batch(&[WriteThroughEntry {
            guid: [0xAA; 16],
            bytes: snap_bytes,
            expected_seq: 50,
            content_version: None,
        }])
        .unwrap();
        assert_eq!(
            bm.dirty_count(),
            1,
            "write-through must not stomp a racing newer-seq entry",
        );
        let live = bm.snapshot_dirty();
        assert_eq!(live[&[0xAA; 16]], 200, "racing writer's seq survives");
    }

    #[test]
    fn write_through_keeps_racing_structural_dirty_entry() {
        // `STRUCTURAL_SEQ` is a shared sentinel, not a unique WAL
        // sequence. A fresh structural mutation can therefore have
        // the same dirty value as a checkpoint's older snapshot;
        // equality alone must not retire it.
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0xA5; 16], &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin([0xA5; 16]).unwrap();

        bm.mark_dirty([0xA5; 16], STRUCTURAL_SEQ);
        let snap_bytes = snapshot_current_bytes(&bm, [0xA5; 16]);

        bm.write_through_batch(&[WriteThroughEntry {
            guid: [0xA5; 16],
            bytes: snap_bytes,
            expected_seq: STRUCTURAL_SEQ,
            content_version: None,
        }])
        .unwrap();
        assert_eq!(
            bm.dirty_count(),
            1,
            "structural sentinel equality is not enough to retire a racing entry",
        );
        let live = bm.snapshot_dirty();
        assert_eq!(live[&[0xA5; 16]], STRUCTURAL_SEQ);
    }

    #[test]
    fn write_through_retires_clean_snapshot() {
        // Counterpart to the race test: when the dirty entry
        // still matches the snapshot's seq (no racing writer),
        // checkpoint write-through does retire it.
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0xBB; 16], &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin([0xBB; 16]).unwrap();

        bm.mark_dirty([0xBB; 16], 42);
        let snap_bytes = snapshot_current_bytes(&bm, [0xBB; 16]);

        // expected_seq matches the current entry → safe to retire.
        bm.write_through_batch(&[WriteThroughEntry {
            guid: [0xBB; 16],
            bytes: snap_bytes,
            expected_seq: 42,
            content_version: None,
        }])
        .unwrap();
        assert_eq!(bm.dirty_count(), 0);
    }

    #[test]
    fn write_through_batch_retires_clean_snapshots() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let g1 = [0xB1; 16];
        let g2 = [0xB2; 16];
        inner.write_blob(g1, &make_buf(0)).unwrap();
        inner.write_blob(g2, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        for (guid, byte) in [(g1, 11), (g2, 22)] {
            let pin = bm.pin(guid).unwrap();
            let mut guard = pin.write();
            guard.as_mut_slice()[100] = byte;
            bm.mark_dirty(guid, u64::from(byte));
        }

        let snap = bm.snapshot_dirty();
        let entries: Vec<_> = snap
            .iter()
            .map(|(guid, expected_seq)| WriteThroughEntry {
                guid: *guid,
                bytes: snapshot_current_bytes(&bm, *guid),
                expected_seq: *expected_seq,
                content_version: None,
            })
            .collect();
        bm.write_through_batch(&entries).unwrap();

        assert_eq!(bm.dirty_count(), 0);
        let mut dst = AlignedBlobBuf::zeroed();
        inner.read_blob(g1, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 11);
        inner.read_blob(g2, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 22);
    }

    #[test]
    fn write_through_batch_invalidates_indexed_read_cache_before_retire() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0xBC; 16];
        inner.write_blob(guid, &make_buf(0)).unwrap();
        let bm = BufferManager::new_file(inner.clone(), 128, AlignedBlobBuf::zeroed);

        let old_page = [0xA5; PAGE_4K as usize];
        let mut dst = [0u8; PAGE_4K as usize];
        bm.read_page_store(guid, 0, &old_page);
        assert!(bm.read_page_cached(guid, 0, &mut dst));
        assert_eq!(dst, old_page);

        let pin = bm.pin(guid).unwrap();
        {
            let mut guard = pin.write();
            guard.as_mut_slice()[100] = 0x77;
        }
        bm.mark_dirty_cached(guid, 10, pin.as_ref());
        let snap = bm.snapshot_dirty();
        let bytes = snapshot_current_bytes(&bm, guid);

        bm.write_through_batch(&[WriteThroughEntry {
            guid,
            bytes,
            expected_seq: snap[&guid],
            content_version: None,
        }])
        .unwrap();

        assert!(
            !bm.read_page_cached(guid, 0, &mut dst),
            "checkpointed bytes must retire stale indexed navigation pages"
        );
    }

    #[test]
    fn write_through_batch_publishes_read_index() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(crate::store::blob_store::FileBlobStore::open(dir.path()).unwrap());
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let bm = BufferManager::new_file(store_dyn, 128, AlignedBlobBuf::zeroed);
        let guid = [0xBD; 16];
        let mut bytes = AlignedBlobBuf::zeroed();
        BlobFrame::init(bytes.as_mut_slice(), guid).unwrap();

        bm.write_through_batch(&[WriteThroughEntry {
            guid,
            bytes,
            expected_seq: 1,
            content_version: None,
        }])
        .unwrap();

        let mut index_bytes = vec![0; ReadIndex::HEADER_LEN];
        assert!(
            store.read_index_range(guid, 0, &mut index_bytes).unwrap(),
            "checkpoint write-through should publish read index"
        );
        let directory_len = ReadIndex::directory_len(&index_bytes)
            .expect("published read index header should parse");
        if directory_len > ReadIndex::HEADER_LEN {
            let mut rest = vec![0; directory_len - ReadIndex::HEADER_LEN];
            assert!(
                store
                    .read_index_range(guid, ReadIndex::HEADER_LEN as u64, &mut rest)
                    .unwrap(),
                "published read index directory should be readable"
            );
            index_bytes.extend_from_slice(&rest);
        }
        ReadIndex::decode_directory(index_bytes).expect("published read index should parse");
    }

    #[test]
    fn read_index_load_reads_small_directory_in_one_probe() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(CountingReadIndexStore::open(dir.path()).unwrap());
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let bm = BufferManager::new_file(store_dyn, 128, AlignedBlobBuf::zeroed);
        let guid = [0xBE; 16];
        let mut bytes = AlignedBlobBuf::zeroed();
        BlobFrame::init(bytes.as_mut_slice(), guid).unwrap();

        bm.write_through_batch(&[WriteThroughEntry {
            guid,
            bytes,
            expected_seq: 1,
            content_version: None,
        }])
        .unwrap();
        store.flush().unwrap();

        let mut index_bytes = vec![0; ReadIndex::HEADER_LEN];
        assert!(store.read_index_range(guid, 0, &mut index_bytes).unwrap());
        ReadIndex::directory_len(&index_bytes).expect("published read-index header parses");

        store.reset_index_reads();
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let bm = BufferManager::new_file(store_dyn, 128, AlignedBlobBuf::zeroed);

        assert!(bm.indexed_read_eligible(guid));
        let mut probe = vec![0; READ_INDEX_DIRECTORY_PROBE_BYTES];
        assert!(store.read_index_range(guid, 0, &mut probe).unwrap());
        let directory_len =
            ReadIndex::directory_len(&probe).expect("probe should parse read-index header");
        probe.truncate(directory_len);
        let index = ReadIndex::decode_directory(probe).expect("probe should decode directory");
        assert!(
            bm.read_index_stamp_matches(guid, &index).unwrap(),
            "probe-decoded read index should match the blob header"
        );
        store.reset_index_reads();

        assert!(bm.read_index(guid).is_some());
        assert_eq!(
            store.index_reads(),
            1,
            "small read-index directories fit in the first 4 KiB probe"
        );
    }

    #[test]
    fn write_through_batch_keeps_racing_writer_dirty_entry() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let g1 = [0xC1; 16];
        let g2 = [0xC2; 16];
        inner.write_blob(g1, &make_buf(0)).unwrap();
        inner.write_blob(g2, &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _ = bm.pin(g1).unwrap();
        let _ = bm.pin(g2).unwrap();

        bm.mark_dirty(g1, 50);
        bm.mark_dirty(g2, 60);
        let snap = bm.snapshot_dirty();
        bm.mark_dirty(g1, 200);

        let entries: Vec<_> = snap
            .iter()
            .map(|(guid, expected_seq)| WriteThroughEntry {
                guid: *guid,
                bytes: snapshot_current_bytes(&bm, *guid),
                expected_seq: *expected_seq,
                content_version: None,
            })
            .collect();
        bm.write_through_batch(&entries).unwrap();

        let live = bm.snapshot_dirty();
        assert_eq!(live.len(), 1);
        assert_eq!(live[&g1], 200);
    }

    #[test]
    fn write_through_revalidates_version_after_store_io() {
        let guid = [0xC3; 16];
        let inner = MemoryBlobStore::new();
        inner.write_blob(guid, &make_buf(1)).unwrap();
        let store = Arc::new(BlockingWriteStore::new(inner));
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let bm = Arc::new(BufferManager::new(store_dyn, 4));
        let pin = bm.pin(guid).unwrap();

        bm.mark_dirty(guid, 100);
        let first_dirty = bm.snapshot_dirty();
        let first_version = bm.snapshot_dirty_versions(&first_dirty).unwrap()[0].content_version;
        let first_bytes = bm
            .snapshot_bytes_if_version(guid, first_version)
            .unwrap()
            .unwrap();
        let write_bm = Arc::clone(&bm);
        let writer = std::thread::spawn(move || {
            write_bm.write_through_batch(&[WriteThroughEntry {
                guid,
                bytes: first_bytes,
                expected_seq: 100,
                content_version: Some(first_version),
            }])
        });

        // The first content-version check has passed and the stale bytes are
        // waiting at store I/O. Publish a lower-seq writer in that exact
        // window; seq comparison alone would incorrectly retire it.
        store.entered.wait();
        {
            let mut frame = pin.write();
            frame.as_mut_slice()[100] = 2;
        }
        bm.mark_dirty_cached(guid, 50, pin.as_ref());
        store.release.wait();

        let report = writer.join().unwrap().unwrap();
        assert_eq!(report.statuses, vec![WriteThroughStatus::Stale]);
        bm.restore_dirty(first_dirty);
        let retained = bm.snapshot_dirty();
        assert_eq!(retained.get(&guid), Some(&50));
        bm.restore_dirty(retained);

        let second_dirty = bm.snapshot_dirty();
        let second_version = bm.snapshot_dirty_versions(&second_dirty).unwrap()[0].content_version;
        let second_bytes = bm
            .snapshot_bytes_if_version(guid, second_version)
            .unwrap()
            .unwrap();
        let report = bm
            .write_through_batch(&[WriteThroughEntry {
                guid,
                bytes: second_bytes,
                expected_seq: second_dirty[&guid],
                content_version: Some(second_version),
            }])
            .unwrap();
        assert_eq!(report.statuses, vec![WriteThroughStatus::Written]);
        assert_eq!(bm.dirty_count(), 0);
        assert_eq!(bm.flushing_count(), 0);

        drop(pin);
        drop(bm);
        let store_dyn: Arc<dyn BlobStore> = store;
        let reopened = BufferManager::new(store_dyn, 4);
        assert_eq!(
            reopened.pin(guid).unwrap().read().as_slice()[100],
            2,
            "the racing writer must survive the retry and reopen",
        );
    }

    #[test]
    fn write_blob_through_trait_clears_dirty() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0x88; 16], &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin([0x88; 16]).unwrap();

        bm.mark_dirty([0x88; 16], 100);
        assert_eq!(bm.dirty_count(), 1);

        // The BlobStore-trait write_blob is write-through and so
        // satisfies the dirty entry by construction.
        BlobStore::write_blob(&bm, [0x88; 16], &make_buf(9)).unwrap();
        assert_eq!(bm.dirty_count(), 0);
    }

    #[test]
    fn delete_blob_drops_dirty_entry() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        inner.write_blob([0x99; 16], &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);

        let _ = bm.pin([0x99; 16]).unwrap();
        bm.mark_dirty([0x99; 16], 7);
        assert_eq!(bm.dirty_count(), 1);

        BlobStore::delete_blob(&bm, [0x99; 16]).unwrap();
        assert_eq!(
            bm.dirty_count(),
            0,
            "deleted blobs must not linger as flush candidates"
        );
    }

    #[test]
    fn install_new_blob_caches_and_marks_dirty_without_store_write() {
        // The unified-protocol fix: spillover's new child blob
        // must land in cache + dirty, NOT in the inner store,
        // so the checkpoint round can enforce the W2D ordering
        // (WAL flush THEN store write).
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(Arc::clone(&inner), 4);

        let new_guid = [0xCC; 16];
        let mut bytes = AlignedBlobBuf::zeroed();
        bytes.as_mut_slice()[200] = 0x77;

        bm.install_new_blob(new_guid, bytes, /*seq=*/ 42);

        // BM cached + dirty.
        assert_eq!(bm.cached_count(), 1);
        assert_eq!(bm.dirty_count(), 1);
        let snap = bm.snapshot_dirty();
        assert_eq!(snap[&new_guid], 42);
        bm.restore_dirty(snap);

        // Inner store has nothing yet.
        assert!(
            !inner.has_blob(new_guid).unwrap(),
            "install_new_blob must defer the store write to the checkpoint round",
        );

        // Pinning the blob returns the cached image.
        let pin = bm.pin(new_guid).unwrap();
        let guard = pin.read();
        assert_eq!(guard.as_slice()[200], 0x77);
        drop(guard);
        drop(pin);

        // After the production checkpoint primitive runs, the inner
        // store has the bytes and the dirty entry is cleared.
        let snap = bm.snapshot_dirty();
        let bytes = snapshot_current_bytes(&bm, new_guid);
        bm.write_through_batch(&[WriteThroughEntry {
            guid: new_guid,
            bytes,
            expected_seq: snap[&new_guid],
            content_version: None,
        }])
        .unwrap();
        bm.flush_inner().unwrap();
        assert_eq!(bm.dirty_count(), 0);
        assert!(inner.has_blob(new_guid).unwrap());
        let mut dst = AlignedBlobBuf::zeroed();
        inner.read_blob(new_guid, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[200], 0x77);
    }

    #[test]
    fn concurrent_reads_on_different_blobs_progress() {
        use std::thread;

        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        for i in 0..16u8 {
            let mut g = [0u8; 16];
            g[0] = i;
            inner.write_blob(g, &make_buf(i)).unwrap();
        }

        let bm = Arc::new(BufferManager::new(inner, 16));
        let handles: Vec<_> = (0..8u8)
            .map(|t| {
                let bm = bm.clone();
                thread::spawn(move || {
                    for _ in 0..50 {
                        let mut g = [0u8; 16];
                        g[0] = t * 2; // each thread targets its own blob
                        let mut dst = AlignedBlobBuf::zeroed();
                        bm.read_blob(g, &mut dst).unwrap();
                        assert_eq!(dst.as_slice()[100], t * 2);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // All 8 thread targets cached.
        assert_eq!(bm.cached_count(), 8);
    }
}
