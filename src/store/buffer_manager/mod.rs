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
//! blob (spillover) or unlink one (erase's `SubtreeGone` /
//! merge) go through [`BufferManager::install_new_blob`] /
//! [`BufferManager::mark_for_delete`] instead, so the store
//! write or manifest mutation is deferred until the next flush —
//! invariant **W2D** below.
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

use std::collections::{hash_map::Entry, BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use guid_hash::GuidBuildHasher;

use crate::api::errors::{Error, Result};
use crate::layout::{BlobGuid, PAGE_SIZE};

use super::blob_store::{AlignedBlobBuf, BlobStore};

use admission::TinyLFU;
pub use cached_blob::{BlobWriteGuard, CachedBlob};
use mutation::{
    bookkeeping_shard_idx, pop_candidate_batch, CandidateKind, MutationState, BOOKKEEPING_SHARDS,
};
use residency::RouteResidency;
use telemetry::Telemetry;

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
    /// Epoch → snapshot root GUID for every live snapshot.
    live: BTreeMap<u64, BlobGuid>,
    /// Frames forked away from the live tree, tagged with the
    /// `created_epoch` of the forked-away version. Safe to free once the
    /// fork barrier drops below the tag (no live snapshot can reference
    /// it). One entry per fork.
    orphans: Vec<(BlobGuid, u64)>,
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

/// Frequency-aware blob cache; see the module docs.
pub struct BufferManager {
    store: Arc<dyn BlobStore>,
    alloc_uninit: Arc<dyn Fn() -> AlignedBlobBuf + Send + Sync>,
    capacity: usize,
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
    /// Live CoW snapshot registry + forked-away orphan list. Drives
    /// fork-barrier recomputation and orphan reclaim on retire.
    snapshots: Mutex<SnapshotState>,
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

    /// Copy `src`'s current frame image to a fresh GUID `new_guid`,
    /// install it, and pin it — the frozen root of a new CoW snapshot.
    ///
    /// The caller must hold the owning tree's mutation gate exclusively
    /// so the source frame is byte-stable for the copy. The copy keeps
    /// `src`'s entire structure (children are referenced by GUID, so
    /// they stay shared rather than deep-copied); only the self-GUID is
    /// repatched. The copy's `created_epoch` is irrelevant because a
    /// snapshot root is read-only and never forked.
    pub(crate) fn install_snapshot_root(
        &self,
        new_guid: BlobGuid,
        src: &CachedBlob,
        seq: u64,
    ) -> Result<Arc<CachedBlob>> {
        let guard = src.read();
        self.fork_frame(guard.as_slice(), new_guid, seq)
    }

    /// Register a live snapshot rooted at `root_guid`. Bumps the global
    /// epoch (so frames created afterwards are private to the live
    /// tree), raises the fork barrier to the snapshot's epoch, and
    /// returns that epoch.
    pub(crate) fn register_snapshot(&self, root_guid: BlobGuid) -> u64 {
        let mut snaps = self.snapshots.lock().expect("snapshot registry poisoned");
        // `fetch_add` returns the pre-bump value: that is this snapshot's
        // epoch and, because `current_epoch` only ever increases, the
        // largest key in the registry — hence the new barrier.
        let epoch = self.current_epoch.fetch_add(1, Ordering::AcqRel);
        snaps.live.insert(epoch, root_guid);
        self.fork_barrier.store(epoch, Ordering::Release);
        epoch
    }

    /// Record a frame forked away from the live tree so it can be freed
    /// once no live snapshot can reference it. `created_epoch` is the
    /// epoch of the forked-away version (≤ the barrier at fork time).
    pub(crate) fn record_orphan(&self, guid: BlobGuid, created_epoch: u64) {
        self.snapshots
            .lock()
            .expect("snapshot registry poisoned")
            .orphans
            .push((guid, created_epoch));
    }

    /// Retire the snapshot at `epoch`: lower the fork barrier to the
    /// highest remaining live snapshot epoch (or `0`), free the
    /// snapshot's root frame, and reclaim every orphan whose forked-away
    /// version is now newer than the barrier — no live snapshot can
    /// reference it. Idempotent for an unknown epoch.
    pub(crate) fn retire_snapshot(&self, epoch: u64) {
        let (root, to_free) = {
            let mut snaps = self.snapshots.lock().expect("snapshot registry poisoned");
            let root = snaps.live.remove(&epoch);
            let barrier = snaps.live.keys().next_back().copied().unwrap_or(0);
            self.fork_barrier.store(barrier, Ordering::Release);
            let mut to_free = Vec::new();
            snaps.orphans.retain(|&(guid, created_epoch)| {
                let still_referenced = created_epoch <= barrier;
                if !still_referenced {
                    to_free.push(guid);
                }
                still_referenced
            });
            (root, to_free)
        };
        if let Some(root) = root {
            self.reclaim_blob(root);
        }
        for guid in to_free {
            self.reclaim_blob(guid);
        }
    }

    /// Free a copy-on-write frame no longer referenced by the live tree
    /// or any live snapshot. Checkpoint-owned or pending-delete frames
    /// are still load-bearing for checkpoint correctness, so reclamation
    /// is best-effort: those frames are left for a later DB-wide GC after
    /// their checkpoint owner has retired.
    fn reclaim_blob(&self, guid: BlobGuid) {
        {
            let state = self.mutation_shard(guid).lock().unwrap();
            if state.is_protected_or_pending(&guid) {
                return;
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
            return;
        }
        self.route_resident.remove(guid);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state.remove_unclaimed_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
        let _ = self.store.delete_blob(guid);
    }

    /// GUIDs of every live snapshot's frozen root frame.
    pub(crate) fn snapshot_roots(&self) -> Vec<BlobGuid> {
        self.snapshots
            .lock()
            .expect("snapshot registry poisoned")
            .live
            .values()
            .copied()
            .collect()
    }

    /// Free every persisted frame not in `reachable`, returning the count
    /// reclaimed. The recovery-time sweep for copy-on-write frames
    /// orphaned by a crash that lost the in-memory orphan list. The
    /// caller must hold the tree quiescent and pass the full reachable
    /// set (live root ∪ every live snapshot root).
    pub(crate) fn gc_sweep_unreachable(&self, reachable: &HashSet<BlobGuid>) -> Result<usize> {
        let mut freed = 0;
        for guid in self.list_blobs()? {
            if !reachable.contains(&guid) {
                self.reclaim_blob(guid);
                freed += 1;
            }
        }
        Ok(freed)
    }
}

impl BufferManager {
    /// Wrap `store` with a cache of at most `capacity` blobs
    /// (each blob is 512 KB on the heap). A `capacity` of 0 is
    /// clamped to 1.
    #[must_use]
    pub fn new(store: Arc<dyn BlobStore>, capacity: usize) -> Self {
        Self::new_with_uninit_allocator(store, capacity, || {
            // SAFETY: BufferManager's uninitialized allocations are
            // filled by read_blob or a full-frame copy before read.
            unsafe { AlignedBlobBuf::uninit() }
        })
    }

    /// Construct with a crate-private uninitialized-frame allocator.
    ///
    /// File-backed trees use this to preserve Linux fixed-buffer
    /// allocation without exposing an uninitialized constructor in
    /// the public BlobStore trait.
    #[must_use]
    pub(crate) fn new_with_uninit_allocator<F>(
        store: Arc<dyn BlobStore>,
        capacity: usize,
        alloc_uninit: F,
    ) -> Self
    where
        F: Fn() -> AlignedBlobBuf + Send + Sync + 'static,
    {
        let capacity = capacity.max(1);
        Self {
            store,
            alloc_uninit: Arc::new(alloc_uninit),
            capacity,
            cache: DashMap::with_hasher(GuidBuildHasher),
            admission: TinyLFU::new(),
            route_resident: RouteResidency::new(capacity),
            mutation: std::array::from_fn(|_| Mutex::new(MutationState::default())),
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
        }
    }

    fn alloc_blob_buf_uninit(&self) -> AlignedBlobBuf {
        (self.alloc_uninit)()
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

    pub(crate) fn route_resident_demotions(&self) -> u64 {
        self.telemetry.route_resident_demotions()
    }

    pub(crate) fn cache_evictions(&self) -> u64 {
        self.telemetry.cache_evictions()
    }

    pub(crate) fn eviction_skips_protected(&self) -> u64 {
        self.telemetry.eviction_skips_protected()
    }

    pub(crate) fn eviction_skips_route_resident(&self) -> u64 {
        self.telemetry.eviction_skips_route_resident()
    }

    pub(crate) fn admission_protects(&self) -> u64 {
        self.telemetry.admission_protects()
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
    #[cfg(test)]
    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Cumulative cache lookup hits (`get_cached` found the entry
    /// without consulting the inner store). Relaxed-ordered;
    /// reads are observability-only.
    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.telemetry.cache_hits()
    }

    /// Cumulative cache lookup misses — every miss is followed by
    /// an `inner_store.read_blob` and an `insert_into_cache`.
    #[must_use]
    pub fn cache_misses(&self) -> u64 {
        self.telemetry.cache_misses()
    }

    /// Successful full-frame reads from the inner store after a
    /// BufferManager miss. Each read pulls one `PAGE_SIZE` blob.
    #[must_use]
    pub fn full_blob_reads(&self) -> u64 {
        self.telemetry.full_blob_reads()
    }

    /// Bytes read by successful full-frame inner-store reads.
    #[must_use]
    pub fn full_blob_read_bytes(&self) -> u64 {
        self.full_blob_reads() * PAGE_SIZE as u64
    }

    /// Full-frame reads caused by point get/put paths.
    #[must_use]
    pub fn point_full_blob_reads(&self) -> u64 {
        self.telemetry.point_full_blob_reads()
    }

    /// Full-frame reads caused by range/list scan paths.
    #[must_use]
    pub fn scan_full_blob_reads(&self) -> u64 {
        self.telemetry.scan_full_blob_reads()
    }

    /// Full-frame reads caused by silent introspection paths.
    #[must_use]
    pub fn silent_full_blob_reads(&self) -> u64 {
        self.telemetry.silent_full_blob_reads()
    }

    /// Cumulative optimistic-read restarts. Bumped by the lookup
    /// walker every time a `validate()` after a wait-free read
    /// returns `false` — a concurrent writer lapped the snapshot
    /// and the walk has to restart from the root.
    #[must_use]
    pub fn optimistic_restarts(&self) -> u64 {
        self.telemetry.optimistic_restarts()
    }

    /// Bump the optimistic-restart counter. Called from the
    /// lookup walker on `validate()` failure.
    pub(crate) fn note_optimistic_restart(&self) {
        self.telemetry.note_optimistic_restart();
    }

    /// Cumulative range-iterator cursor restarts. Bumped when a
    /// versioned range cursor detects that a writer rewrote a blob
    /// on its descent path and must rebuild from its monotonic
    /// lower bound.
    #[must_use]
    pub fn range_restarts(&self) -> u64 {
        self.telemetry.range_restarts()
    }

    pub(crate) fn note_range_restart(&self) {
        self.telemetry.note_range_restart();
    }

    /// Cumulative mutation walker calls (`insert_multi` /
    /// `erase_multi`). A `rename` or `atomic` contributes one count per
    /// inner walker invocation, not one count per public API call.
    #[must_use]
    pub fn walker_ops(&self) -> u64 {
        self.telemetry.walker_ops()
    }

    /// Total blob hops across mutation walkers. Divide by
    /// [`Self::walker_ops`] to derive average blob-hop count.
    #[must_use]
    pub fn walker_blob_hops(&self) -> u64 {
        self.telemetry.walker_blob_hops()
    }

    /// Maximum blob hops observed for a single mutation walker call.
    #[must_use]
    pub fn max_blob_hops(&self) -> u64 {
        self.telemetry.max_blob_hops()
    }

    /// Largest key-depth at which a mutation walker entered a blob.
    /// This is a cross-blob boundary-depth signal rather than a full
    /// per-node ART-depth trace.
    #[must_use]
    pub fn max_cross_blob_depth(&self) -> u64 {
        self.telemetry.max_cross_blob_depth()
    }

    /// Number of successful foreground spillover events.
    #[must_use]
    pub fn spillover_count(&self) -> u64 {
        self.telemetry.spillover_count()
    }

    /// Number of `BlobNode` children folded back into parents by
    /// manual compact or background merge passes.
    #[must_use]
    pub fn merge_count(&self) -> u64 {
        self.telemetry.merge_count()
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

    fn pending_delete_not_found(guid: BlobGuid) -> Error {
        Error::BlobStoreIo(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("blob {:02x?} is pending delete", &guid[..4]),
        ))
    }

    /// Internal: insert a freshly-loaded blob into the cache.
    /// Idempotent under concurrent inserts. Stamps the new entry's
    /// `last_touched` so it doesn't look cold to the eviction
    /// thread on its very next sweep.
    fn insert_into_cache(&self, guid: BlobGuid, contents: &AlignedBlobBuf) {
        self.insert_owned_into_cache(guid, contents.clone(), PinAccess::Point);
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

    /// Pin for range/list scans. Hits and misses remain visible in
    /// cache telemetry, but scan access does not refresh
    /// recency. This keeps large directory/object-list walks from
    /// evicting hot point-read blobs.
    pub(crate) fn pin_scan(&self, guid: BlobGuid) -> Result<Arc<CachedBlob>> {
        self.pin_with_access(guid, PinAccess::Scan)
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
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        if let Some(entry) = self.get_cached_with_access(guid, access) {
            return Ok(entry);
        }
        // SAFETY: read_blob fills the full PAGE_SIZE frame before
        // `scratch` is inserted into the cache or read.
        let mut scratch = self.alloc_blob_buf_uninit();
        self.store.read_blob(guid, &mut scratch)?;
        self.note_full_blob_read(access);
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        Ok(self.insert_owned_into_cache(guid, scratch, access))
    }

    /// Whether `guid` may be served by a cold, page-granular read
    /// straight from the backing store (the stage-3 routed read).
    ///
    /// Returns `false` — meaning the caller must fall back to [`pin`]
    /// (which reads the authoritative resident/full-frame image) —
    /// when the blob is pending-delete, already resident in cache (a
    /// dirty cache image may be newer than the on-disk frame), or
    /// protected/pending a structural op.
    ///
    /// [`pin`]: Self::pin
    pub(crate) fn cold_read_eligible(&self, guid: BlobGuid) -> bool {
        if self.is_pending_delete(guid) || self.cache.contains_key(&guid) {
            return false;
        }
        let state = self.mutation_shard(guid).lock().unwrap();
        !state.is_protected_or_pending(&guid)
    }

    /// Positional, page-granular read from the backing store, bypassing
    /// the cache and the 512 KB io_uring ring (see
    /// [`BlobStore::read_blob_range`]). The caller owns 4 KB alignment
    /// of `byte_offset` and `dst` (length + base). Used by the stage-3
    /// cold routed read to fetch the header page + routing region + one
    /// leaf page instead of pinning the whole frame.
    ///
    /// [`BlobStore::read_blob_range`]: crate::store::blob_store::BlobStore::read_blob_range
    pub(crate) fn read_blob_range(
        &self,
        guid: BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> Result<()> {
        self.store.read_blob_range(guid, byte_offset, dst)
    }

    fn note_full_blob_read(&self, access: PinAccess) {
        match access {
            PinAccess::Point => self.telemetry.note_point_full_blob_read(),
            PinAccess::Scan => self.telemetry.note_scan_full_blob_read(),
            PinAccess::Silent => self.telemetry.note_silent_full_blob_read(),
        }
    }

    // ---------- dirty tracking ----------

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
        self.mark_dirty_with_hint(guid, seq, cached.as_deref());
    }

    /// Same contract as [`Self::mark_dirty`], but the caller
    /// already holds the cached blob pin from the walker descent.
    /// This avoids a second DashMap lookup on the mutation hot path.
    pub(crate) fn mark_dirty_cached(&self, guid: BlobGuid, seq: u64, entry: &CachedBlob) {
        self.mark_dirty_with_hint(guid, seq, Some(entry));
    }

    fn mark_dirty_with_hint(&self, guid: BlobGuid, seq: u64, cached: Option<&CachedBlob>) {
        let Some(cached) = cached else {
            // No dirty entry without the newer cache image: that
            // would violate I1 and make checkpoint unable to
            // snapshot the bytes it is asked to flush.
            return;
        };
        let hint_covers_seq = !cached.dirty_hint_needs_map_publish(seq);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if state.has_delete_fence(&guid) {
            cached.clear_dirty_hint();
            return;
        }
        if hint_covers_seq && matches!(state.dirty.get(&guid), Some(cur) if *cur <= seq) {
            return;
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
            let mut state = shard.lock().unwrap();
            for (guid, seq) in &mut state.dirty {
                if let Some(hinted_seq) = self
                    .get_cached_with_access(*guid, PinAccess::Silent)
                    .and_then(|entry| entry.take_dirty_hint())
                {
                    *seq = (*seq).min(hinted_seq);
                }
            }
            let snap = std::mem::take(&mut state.dirty);
            for (guid, seq) in snap {
                if state.has_delete_fence(&guid) {
                    if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
                        entry.clear_dirty_hint();
                    }
                    continue;
                }
                state.add_flushing(guid);
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
            if state.has_delete_fence(&guid) {
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
    /// soon-deleted slot) and queues the `store.delete_blob`
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
    /// The checkpoint round drains this set after Sync (data file
    /// plus initial manifest snapshot durable) and re-Syncs once
    /// the deletions have been applied — only then can the WAL
    /// be truncated.
    pub fn mark_for_delete(&self, guid: BlobGuid, seq: u64) {
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
            let had_fence = state.has_delete_fence(&g);
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
                    if !had_fence {
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

    /// Execute a queued deletion against the inner store.
    /// Manifest mutation is in-memory; subsequent `store.flush`
    /// makes it durable. Returns `Ok(false)` when the blob still
    /// has dirty/flushing state and the caller should requeue the
    /// delete for a later round.
    pub(crate) fn execute_pending_delete(&self, guid: BlobGuid) -> Result<bool> {
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

    /// `true` iff `guid` still has dirty or in-flight checkpoint
    /// state owned by the buffer manager.
    pub(crate) fn has_unflushed_blob(&self, guid: BlobGuid) -> bool {
        let state = self.mutation_shard(guid).lock().unwrap();
        state.dirty.contains_key(&guid) || state.flushing.contains_key(&guid)
    }

    /// Snapshot the cached bytes for `guid` into a freshly allocated
    /// `AlignedBlobBuf`. Returns `None` if the blob isn't cached.
    ///
    /// Used by the background checkpointer to hand off bytes to
    /// the I/O worker thread without keeping the shared read guard
    /// open across the actual `store.write_blob` call. The read
    /// guard is held only for the duration of the 512 KB memcpy, so
    /// writers don't block on long-running (especially io_uring)
    /// I/O.
    pub(crate) fn snapshot_bytes(&self, guid: BlobGuid) -> Option<AlignedBlobBuf> {
        let entry = self.get_cached_with_access(guid, PinAccess::Silent)?;
        let buf = entry.read();
        // SAFETY: copy_from_slice below writes the full PAGE_SIZE
        // frame before `out` is returned.
        let mut out = self.alloc_blob_buf_uninit();
        out.as_mut_slice().copy_from_slice(buf.as_slice());
        Some(out)
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
        for idx in write_indices {
            let entry = &entries[idx];
            self.retire_write_through(entry.guid, entry.expected_seq);
            statuses[idx] = WriteThroughStatus::Written;
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

    fn retire_write_through(&self, guid: BlobGuid, expected_seq: u64) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
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
    }

    /// Forward `flush` to the inner store without touching the
    /// cache. Used by the checkpoint I/O worker between epoch
    /// phases.
    pub(crate) fn flush_inner(&self) -> Result<()> {
        self.store.flush()
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
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        // Cache hit?
        if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Point) {
            let buf = entry.read();
            dst.as_mut_slice().copy_from_slice(buf.as_slice());
            return Ok(());
        }
        // Cache miss — load from inner store and cache.
        self.store.read_blob(guid, dst)?;
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        self.insert_into_cache(guid, dst);
        Ok(())
    }

    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        // Transparent write-through: if cached, refresh the
        // cached image; either way, always write to the inner
        // store in the same call so durability is unchanged.
        if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
            let mut buf = entry.write();
            buf.as_mut_slice().copy_from_slice(src.as_slice());
            entry.clear_dirty_hint();
        }
        self.store.write_blob(guid, src)?;
        // BlobStore now holds these exact bytes; any pending dirty
        // entry for this blob is satisfied. Subsequent writes via
        // the pin/write-guard path will re-mark it.
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state.remove_unclaimed_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
        Ok(())
    }

    fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        for (guid, _) in writes {
            if self.is_pending_delete(*guid) {
                return Err(Self::pending_delete_not_found(*guid));
            }
        }
        for (guid, src) in writes {
            if let Some(entry) = self.get_cached_with_access(*guid, PinAccess::Silent) {
                let mut buf = entry.write();
                buf.as_mut_slice().copy_from_slice(src.as_slice());
                entry.clear_dirty_hint();
            }
        }
        self.store.write_blobs(writes)?;
        for (guid, _) in writes {
            let mut state = self.mutation_shard(*guid).lock().unwrap();
            state.remove_unclaimed_dirty(guid);
            let removed = state.remove_maintenance_candidates(guid);
            drop(state);
            self.decrement_candidate_totals(removed);
        }
        Ok(())
    }

    fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        if !self.evict_from_cache(guid) {
            return Err(Error::Internal(
                "delete_blob: protected cache image cannot be evicted",
            ));
        }
        self.store.delete_blob(guid)
    }

    fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
        self.store.list_blobs()
    }

    fn flush(&self) -> Result<()> {
        // Write-through mode: nothing pending in cache.
        self.store.flush()
    }

    fn needs_flush(&self) -> bool {
        self.store.needs_flush()
    }

    fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::blob_store::MemoryBlobStore;

    fn make_buf(byte_at_100: u8) -> AlignedBlobBuf {
        let mut b = AlignedBlobBuf::zeroed();
        b.as_mut_slice()[100] = byte_at_100;
        b
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
    fn pin_miss_is_not_counted_as_a_hit() {
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let guid = [0xCD; 16];
        inner.write_blob(guid, &make_buf(9)).unwrap();

        let bm = BufferManager::new(inner, 4);
        let first = bm.pin(guid).unwrap();
        assert_eq!(first.read().as_slice()[100], 9);
        drop(first);
        assert_eq!(bm.cache_misses(), 1);
        assert_eq!(bm.cache_hits(), 0);

        let second = bm.pin(guid).unwrap();
        assert_eq!(second.read().as_slice()[100], 9);
        assert_eq!(bm.cache_misses(), 1);
        assert_eq!(bm.cache_hits(), 1);
        assert_eq!(bm.full_blob_reads(), 1);
        assert_eq!(bm.full_blob_read_bytes(), PAGE_SIZE as u64);
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

        assert_eq!(bm.full_blob_reads(), 3);
        assert_eq!(bm.full_blob_read_bytes(), 3 * PAGE_SIZE as u64);
        assert_eq!(bm.point_full_blob_reads(), 1);
        assert_eq!(bm.scan_full_blob_reads(), 1);
        assert_eq!(bm.silent_full_blob_reads(), 1);
        assert_eq!(
            bm.cache_misses(),
            2,
            "silent miss does not count as a public cache miss"
        );
        assert_eq!(bm.cache_hits(), 0);

        drop(bm.pin([0; 16]).unwrap());
        assert_eq!(
            bm.full_blob_reads(),
            3,
            "cache hits must not count as store reads"
        );
        assert_eq!(bm.cache_hits(), 1);
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
        assert_eq!(bm.route_resident_demotions(), 1);
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

        // And snapshot_bytes(A) must still return Some — the
        // invariant downstream checkpoint code relies on.
        assert!(
            bm.snapshot_bytes(g_a).is_some(),
            "dirty entry's cache image must be snapshottable",
        );
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

        assert!(bm.execute_pending_delete(guid).unwrap());
        assert_eq!(bm.pending_delete_count(), 0);
        assert!(!inner.has_blob(guid).unwrap());
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
            !bm.execute_pending_delete(guid).unwrap(),
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
        assert!(bm.execute_pending_delete(guid).unwrap());
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
        let bytes = bm
            .snapshot_bytes(guid)
            .expect("flushing protection must preserve cached bytes");
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

        bm.reclaim_blob(guid);

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

        bm.reclaim_blob(guid);

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
            !bm.execute_pending_delete(guid).unwrap(),
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
        assert!(bm.execute_pending_delete(guid).unwrap());
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
        let snap_bytes = bm.snapshot_bytes([0xAA; 16]).unwrap();

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
        let snap_bytes = bm.snapshot_bytes([0xA5; 16]).unwrap();

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
        let snap_bytes = bm.snapshot_bytes([0xBB; 16]).unwrap();

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
                bytes: bm.snapshot_bytes(*guid).unwrap(),
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
                bytes: bm.snapshot_bytes(*guid).unwrap(),
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
        let bytes = bm.snapshot_bytes(new_guid).unwrap();
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
