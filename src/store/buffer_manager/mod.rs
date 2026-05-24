//! `BufferManager` — LRU-bounded blob cache.
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
//! disk.
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
//!   next round. [`BufferManager::snapshot_pending_deletes`] has
//!   the same drain semantics.
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
//! - **Inline overflow** ([`Self::try_evict_lru`]) — fires inside
//!   [`Self::insert_into_cache`] when the new entry pushes the
//!   cache past `capacity`. Picks the entry with the oldest
//!   `last_touched` tick whose `Arc::strong_count == 1` (no
//!   outside pin).
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

mod cached_blob;
mod mutation;

use std::collections::{hash_map::Entry, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use crate::api::errors::{Error, Result};
use crate::layout::BlobGuid;

use super::blob_store::{AlignedBlobBuf, BlobStore};

pub use cached_blob::{BlobWriteGuard, CachedBlob};
use mutation::{
    bookkeeping_shard_idx, pop_candidate_batch, CandidateKind, MutationState, BOOKKEEPING_SHARDS,
};

/// Sentinel seq for dirty / pending-delete entries that originate
/// from purely structural mutations (compact, merge pass) — they
/// have no corresponding WAL record and so must not pin the WAL
/// trim watermark. `min(dirty.values())` is what gates the
/// watermark; using `u64::MAX` ensures a structural entry only
/// matters for trim decisions if no real WAL-seqed entry is
/// present alongside it (in which case dirty is non-empty and
/// the truncate gate already refuses to fire).
pub const STRUCTURAL_SEQ: u64 = u64::MAX;

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
}

#[derive(Clone, Copy)]
enum PinAccess {
    Point,
    Scan,
    Silent,
}

/// LRU-bounded blob cache; see the module docs.
pub struct BufferManager {
    store: Arc<dyn BlobStore>,
    capacity: usize,
    route_resident_budget: usize,
    /// Sharded blob cache. `DashMap` shards by `BlobGuid` so
    /// concurrent `pin` / `get_cached` on different blobs hit
    /// different shards — no single global mutex on the hot read
    /// path. The background eviction thread + each entry's
    /// `last_touched` tick give "approximate LRU" without needing
    /// an O(n) front-of-deque touch on every hit.
    cache: DashMap<BlobGuid, Arc<CachedBlob>>,
    /// Small protected tier for route-anchor blobs learned from
    /// the route cache. Dirty and pending-delete state still lives
    /// in `mutation`; this tier only prevents ordinary clean-cache
    /// pressure from evicting the top of a large path-shaped tree.
    route_resident: DashMap<BlobGuid, u64>,
    /// Per-blob mutation bookkeeping, sharded by `BlobGuid`.
    ///
    /// Each shard owns the dirty, flushing, and pending-delete
    /// entries for the same set of blobs. Keeping those three maps
    /// under one shard lock gives `mark_dirty` / `mark_for_delete`
    /// one short critical section with no global dirty mutex on the
    /// persistent write hot path.
    mutation: [Mutex<MutationState>; BOOKKEEPING_SHARDS],
    pending_delete_total: AtomicUsize,
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
    /// drives inline overflow eviction (`try_evict_lru`).
    ///
    /// Uses `Relaxed` ordering throughout — strict happens-before
    /// isn't required, only "more recent stamps look more recent".
    clock: AtomicU64,
    /// Telemetry counters — incremented on the hot path, read by
    /// [`crate::Tree::stats`] for observability. All `Relaxed`;
    /// they're approximate metrics, not synchronisation aids.
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    optimistic_restarts: AtomicU64,
    range_restarts: AtomicU64,
    walker_ops: AtomicU64,
    walker_blob_hops: AtomicU64,
    max_blob_hops: AtomicU64,
    max_cross_blob_depth: AtomicU64,
    spillover_count: AtomicU64,
    merge_count: AtomicU64,
    route_resident_demotions: AtomicU64,
}

fn route_resident_budget(capacity: usize) -> usize {
    if capacity < 4 {
        0
    } else {
        (capacity / 4).min(4096)
    }
}

#[inline]
fn fetch_max_relaxed(atom: &AtomicU64, value: u64) {
    let mut cur = atom.load(Ordering::Relaxed);
    while value > cur {
        match atom.compare_exchange_weak(cur, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

impl BufferManager {
    /// Wrap `store` with a cache of at most `capacity` blobs
    /// (each blob is 512 KB on the heap). A `capacity` of 0 is
    /// clamped to 1.
    #[must_use]
    pub fn new(store: Arc<dyn BlobStore>, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            store,
            capacity,
            route_resident_budget: route_resident_budget(capacity),
            cache: DashMap::new(),
            route_resident: DashMap::new(),
            mutation: std::array::from_fn(|_| Mutex::new(MutationState::default())),
            pending_delete_total: AtomicUsize::new(0),
            compact_candidate_cursor: AtomicUsize::new(0),
            merge_candidate_cursor: AtomicUsize::new(0),
            compact_candidate_total: AtomicUsize::new(0),
            merge_candidate_total: AtomicUsize::new(0),
            clock: AtomicU64::new(1),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            optimistic_restarts: AtomicU64::new(0),
            range_restarts: AtomicU64::new(0),
            walker_ops: AtomicU64::new(0),
            walker_blob_hops: AtomicU64::new(0),
            max_blob_hops: AtomicU64::new(0),
            max_cross_blob_depth: AtomicU64::new(0),
            spillover_count: AtomicU64::new(0),
            merge_count: AtomicU64::new(0),
            route_resident_demotions: AtomicU64::new(0),
        }
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
        self.route_resident_demotions.load(Ordering::Relaxed)
    }

    pub(crate) fn mark_route_resident(&self, guid: BlobGuid) {
        if self.route_resident_budget == 0 {
            return;
        }
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        if let Some(mut entry) = self.route_resident.get_mut(&guid) {
            *entry = tick;
            return;
        }
        self.route_resident.insert(guid, tick);
        while self.route_resident.len() > self.route_resident_budget {
            if !self.demote_oldest_route_resident() {
                break;
            }
        }
    }

    fn demote_oldest_route_resident(&self) -> bool {
        let mut victim: Option<(BlobGuid, u64)> = None;
        for kv in &self.route_resident {
            let guid = *kv.key();
            let tick = *kv.value();
            match victim {
                None => victim = Some((guid, tick)),
                Some((_, vmin)) if tick < vmin => victim = Some((guid, tick)),
                _ => {}
            }
        }
        if let Some((guid, _)) = victim {
            self.route_resident.remove(&guid);
            self.route_resident_demotions
                .fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    fn is_route_resident(&self, guid: BlobGuid) -> bool {
        self.route_resident.contains_key(&guid)
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
            return false;
        }
        {
            let state = self.mutation_shard(guid).lock().unwrap();
            if state.is_protected_or_pending(&guid) {
                return false;
            }
        }
        // `DashMap::remove_if` checks the predicate under the
        // shard lock. `strong_count == 1` means only the shard's
        // slot holds the `Arc` (the snapshot's clone was dropped
        // by the caller; see `eviction::run_scan`).
        self.cache
            .remove_if(&guid, |_, entry| {
                if Arc::strong_count(entry) > 1 {
                    return false;
                }
                let state = self.mutation_shard(guid).lock().unwrap();
                !state.is_protected_or_pending(&guid)
            })
            .is_some()
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
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Cumulative cache lookup misses — every miss is followed by
    /// an `inner_store.read_blob` and an `insert_into_cache`.
    #[must_use]
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Cumulative optimistic-read restarts. Bumped by the lookup
    /// walker every time a `validate()` after a wait-free read
    /// returns `false` — a concurrent writer lapped the snapshot
    /// and the walk has to restart from the root.
    #[must_use]
    pub fn optimistic_restarts(&self) -> u64 {
        self.optimistic_restarts.load(Ordering::Relaxed)
    }

    /// Bump the optimistic-restart counter. Called from the
    /// lookup walker on `validate()` failure.
    pub(crate) fn note_optimistic_restart(&self) {
        self.optimistic_restarts.fetch_add(1, Ordering::Relaxed);
    }

    /// Cumulative range-iterator cursor restarts. Bumped when a
    /// versioned range cursor detects that a writer rewrote a blob
    /// on its descent path and must rebuild from its monotonic
    /// lower bound.
    #[must_use]
    pub fn range_restarts(&self) -> u64 {
        self.range_restarts.load(Ordering::Relaxed)
    }

    pub(crate) fn note_range_restart(&self) {
        self.range_restarts.fetch_add(1, Ordering::Relaxed);
    }

    /// Cumulative mutation walker calls (`insert_multi` /
    /// `erase_multi`). A `rename` or `atomic` contributes one count per
    /// inner walker invocation, not one count per public API call.
    #[must_use]
    pub fn walker_ops(&self) -> u64 {
        self.walker_ops.load(Ordering::Relaxed)
    }

    /// Total blob hops across mutation walkers. Divide by
    /// [`Self::walker_ops`] to derive average blob-hop count.
    #[must_use]
    pub fn walker_blob_hops(&self) -> u64 {
        self.walker_blob_hops.load(Ordering::Relaxed)
    }

    /// Maximum blob hops observed for a single mutation walker call.
    #[must_use]
    pub fn max_blob_hops(&self) -> u64 {
        self.max_blob_hops.load(Ordering::Relaxed)
    }

    /// Largest key-depth at which a mutation walker entered a blob.
    /// This is a cross-blob boundary-depth signal rather than a full
    /// per-node ART-depth trace.
    #[must_use]
    pub fn max_cross_blob_depth(&self) -> u64 {
        self.max_cross_blob_depth.load(Ordering::Relaxed)
    }

    /// Number of successful foreground spillover events.
    #[must_use]
    pub fn spillover_count(&self) -> u64 {
        self.spillover_count.load(Ordering::Relaxed)
    }

    /// Number of `BlobNode` children folded back into parents by
    /// manual compact or background merge passes.
    #[must_use]
    pub fn merge_count(&self) -> u64 {
        self.merge_count.load(Ordering::Relaxed)
    }

    /// Record one completed mutation walker traversal.
    pub(crate) fn note_walker_blob_hops(&self, hops: u64, max_cross_blob_depth: usize) {
        self.walker_ops.fetch_add(1, Ordering::Relaxed);
        self.walker_blob_hops.fetch_add(hops, Ordering::Relaxed);
        fetch_max_relaxed(&self.max_blob_hops, hops);
        fetch_max_relaxed(&self.max_cross_blob_depth, max_cross_blob_depth as u64);
    }

    /// Record one successful spillover.
    pub(crate) fn note_spillover(&self) {
        self.spillover_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record child-blob merge events.
    pub(crate) fn note_merges(&self, merged: u64) {
        if merged != 0 {
            self.merge_count.fetch_add(merged, Ordering::Relaxed);
        }
    }

    /// Internal: look up `guid` in the cache under a declared
    /// access pattern.
    ///
    /// `Point` is the hot get/put path and refreshes LRU recency.
    /// `Scan` counts cache telemetry but deliberately does not
    /// promote the entry, so a large range/list walk cannot rescue
    /// blobs that point lookups would otherwise evict. `Silent` is
    /// for observability and does not count or promote.
    fn get_cached_with_access(&self, guid: BlobGuid, access: PinAccess) -> Option<Arc<CachedBlob>> {
        let Some(entry) = self.cache.get(&guid) else {
            if !matches!(access, PinAccess::Silent) {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        };
        let arc = Arc::clone(entry.value());
        drop(entry);
        match access {
            PinAccess::Point => {
                let tick = self.clock.fetch_add(1, Ordering::Relaxed);
                arc.last_touched.store(tick, Ordering::Relaxed);
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            PinAccess::Scan => {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            PinAccess::Silent => {}
        }
        Some(arc)
    }

    fn mutation_shard(&self, guid: BlobGuid) -> &Mutex<MutationState> {
        &self.mutation[bookkeeping_shard_idx(&guid)]
    }

    fn is_pending_delete(&self, guid: BlobGuid) -> bool {
        if self.pending_delete_total.load(Ordering::Acquire) == 0 {
            return false;
        }
        self.mutation_shard(guid)
            .lock()
            .unwrap()
            .pending_deletes
            .contains_key(&guid)
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
        // next `try_evict_lru` finds a victim. If after the
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
                PinAccess::Point => self.try_evict_lru(),
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

    /// Internal: walk the cache for the entry with the oldest
    /// `last_touched` tick whose `Arc::strong_count == 1` (i.e.
    /// no outside pin) and whose dirty / pending-delete bookkeeping
    /// is empty, and evict it. Returns `true` if an entry was
    /// dropped.
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
    fn try_evict_lru(&self) -> bool {
        self.try_evict_lru_until(u64::MAX)
    }

    fn try_evict_scan_cold(&self) -> bool {
        self.try_evict_lru_until(0)
    }

    fn try_evict_lru_until(&self, max_last_touched: u64) -> bool {
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

        let mut victim: Option<(BlobGuid, u64)> = None;
        for kv in &self.cache {
            if Arc::strong_count(kv.value()) > 1 {
                continue;
            }
            let guid = *kv.key();
            if protected_snap.contains(&guid) || self.is_route_resident(guid) {
                continue;
            }
            let tick = kv.value().last_touched.load(Ordering::Relaxed);
            if tick > max_last_touched {
                continue;
            }
            match victim {
                None => victim = Some((guid, tick)),
                Some((_, vmin)) if tick < vmin => {
                    victim = Some((guid, tick));
                }
                _ => {}
            }
        }
        if let Some((guid, _)) = victim {
            // `remove_if` re-checks strong_count + dirty + pending
            // under the shard lock — guards against a pin acquired
            // (or a fresh dirty / pending-delete mark) between our
            // scan and the remove.
            return self
                .cache
                .remove_if(&guid, |_, e| {
                    if Arc::strong_count(e) > 1 {
                        return false;
                    }
                    let state = self.mutation_shard(guid).lock().unwrap();
                    !state.is_protected_or_pending(&guid)
                })
                .is_some();
        }
        false
    }

    /// Internal: drop `guid` from cache (no-op if not cached) and
    /// clear any dirty bookkeeping for it. Called from
    /// `delete_blob`, where the blob is going away entirely and
    /// any pending dirty write would race with the delete in the
    /// store.
    fn evict_from_cache(&self, guid: BlobGuid) {
        if let Some((_, entry)) = self.cache.remove(&guid) {
            entry.clear_dirty_hint();
        }
        self.route_resident.remove(&guid);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state.remove_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
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
    /// cache telemetry, but scan access does not refresh LRU
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
        let mut scratch = self.store.alloc_blob_buf_uninit();
        self.store.read_blob(guid, &mut scratch)?;
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        Ok(self.insert_owned_into_cache(guid, scratch, access))
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
        if state.pending_deletes.contains_key(&guid) {
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
                if state.pending_deletes.contains_key(&guid) {
                    if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
                        entry.clear_dirty_hint();
                    }
                    continue;
                }
                state
                    .flushing
                    .entry(guid)
                    .and_modify(|cur| *cur = (*cur).min(seq))
                    .or_insert(seq);
                out.insert(guid, seq);
            }
        }
        out
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
            if state.pending_deletes.contains_key(&guid) {
                if let Some(entry) = cached {
                    entry.clear_dirty_hint();
                }
                state.flushing.remove(&guid);
                continue;
            }
            if matches!(state.flushing.get(&guid), Some(cur) if *cur == t) {
                state.flushing.remove(&guid);
            }
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
        match state.pending_deletes.entry(guid) {
            Entry::Occupied(mut entry) => {
                let cur = entry.get_mut();
                *cur = (*cur).min(seq);
            }
            Entry::Vacant(entry) => {
                entry.insert(seq);
                self.pending_delete_total.fetch_add(1, Ordering::AcqRel);
            }
        }
        state.remove_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
        if let Some((_, entry)) = self.cache.remove(&guid) {
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
            let count = pending.len();
            if count != 0 {
                self.pending_delete_total.fetch_sub(count, Ordering::AcqRel);
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
            match state.pending_deletes.entry(g) {
                Entry::Occupied(mut entry) => {
                    let cur = entry.get_mut();
                    *cur = (*cur).min(t);
                }
                Entry::Vacant(entry) => {
                    entry.insert(t);
                    self.pending_delete_total.fetch_add(1, Ordering::AcqRel);
                }
            }
        }
    }

    /// Number of blobs waiting for deferred store deletion.
    /// Reads as zero under the WAL-truncate gate are part of the
    /// "WAL records are all redundant" invariant.
    #[must_use]
    pub fn pending_delete_count(&self) -> usize {
        self.pending_delete_total.load(Ordering::Acquire)
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
        if !state.pending_deletes.contains_key(&guid) && state.compact_candidates.insert(guid) {
            self.compact_candidate_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Mark `guid` as a parent-merge candidate.
    pub(crate) fn note_merge_candidate(&self, guid: BlobGuid) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if !state.pending_deletes.contains_key(&guid) && state.merge_candidates.insert(guid) {
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
    /// makes it durable. Failure is the caller's restoration
    /// concern.
    pub(crate) fn execute_pending_delete(&self, guid: BlobGuid) -> Result<()> {
        self.store.delete_blob(guid)
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
        let mut out = self.store.alloc_blob_buf_uninit();
        out.as_mut_slice().copy_from_slice(buf.as_slice());
        Some(out)
    }

    /// Snapshot the latest image of `guid` whether or not it is
    /// currently cached.
    ///
    /// Cached entries win because they may contain dirty bytes that
    /// have not reached the inner store yet. On a cache miss, the
    /// blob is known to be clean from the buffer manager's point of
    /// view and can be copied directly from the store.
    pub(crate) fn snapshot_blob_image(&self, guid: BlobGuid) -> Result<AlignedBlobBuf> {
        if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Silent) {
            let buf = entry.read();
            let mut out = self.store.alloc_blob_buf_uninit();
            out.as_mut_slice().copy_from_slice(buf.as_slice());
            return Ok(out);
        }
        let mut out = self.store.alloc_blob_buf_uninit();
        self.store.read_blob(guid, &mut out)?;
        Ok(out)
    }

    /// Allocate a zero-filled blob buffer from the inner store's
    /// preferred allocator.
    #[must_use]
    pub(crate) fn alloc_blob_buf_zeroed(&self) -> AlignedBlobBuf {
        self.store.alloc_blob_buf_zeroed()
    }

    /// Push a whole checkpoint snapshot to the inner store using
    /// its native batch path, then retire each matching flushing
    /// entry. On store error the caller must restore the whole
    /// dirty snapshot; we intentionally retire nothing because the
    /// store contract permits an arbitrary written prefix.
    pub(crate) fn write_through_batch(&self, entries: &[WriteThroughEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let writes: Vec<_> = entries
            .iter()
            .map(|entry| (entry.guid, &entry.bytes))
            .collect();
        self.store.write_blobs_with_data_sync(&writes)?;
        for entry in entries {
            self.retire_write_through(entry.guid, entry.expected_seq);
        }
        Ok(())
    }

    fn retire_write_through(&self, guid: BlobGuid, expected_seq: u64) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if expected_seq != STRUCTURAL_SEQ {
            if let std::collections::hash_map::Entry::Occupied(e) = state.dirty.entry(guid) {
                // Only retire the entry when no racing writer has
                // bumped it. `mark_dirty` keeps the **minimum**
                // unflushed seq, so a survivor here has a seq newer
                // than ours iff a racer landed after we drained.
                if *e.get() == expected_seq {
                    e.remove();
                }
            }
        }
        if matches!(state.flushing.get(&guid), Some(seq) if *seq == expected_seq) {
            state.flushing.remove(&guid);
        }
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
    pub(crate) fn install_new_blob(&self, guid: BlobGuid, bytes: AlignedBlobBuf, seq: u64) {
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
        // Cache hit?
        if let Some(entry) = self.get_cached_with_access(guid, PinAccess::Point) {
            let buf = entry.read();
            dst.as_mut_slice().copy_from_slice(buf.as_slice());
            return Ok(());
        }
        // Cache miss — load from inner store and cache.
        self.store.read_blob(guid, dst)?;
        self.insert_into_cache(guid, dst);
        Ok(())
    }

    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
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
        state.remove_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
        Ok(())
    }

    fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
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
            state.remove_dirty(guid);
            let removed = state.remove_maintenance_candidates(guid);
            drop(state);
            self.decrement_candidate_totals(removed);
        }
        Ok(())
    }

    fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        self.evict_from_cache(guid);
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
    fn scan_hits_do_not_refresh_lru_recency() {
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
    fn lru_eviction_at_capacity() {
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
    fn route_resident_anchor_survives_inline_lru() {
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

    /// Regression: prior to the v0.2.1 fix, `try_evict_lru` only
    /// checked `Arc::strong_count == 1` — it would happily evict
    /// a dirty cache image, leaving the dirty entry orphaned in
    /// the dirty map. That broke invariant I1 (dirty ⟺ cache
    /// newer than store) and silently lost the cache mutation
    /// (memory mode) / stuck the WAL truncate gate forever
    /// (persistent mode).
    #[test]
    fn lru_eviction_skips_dirty_entries() {
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
            "dirty entry A's cache image must survive inline LRU eviction",
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
        assert_eq!(bm.pending_delete_count(), 0);

        bm.restore_pending_deletes(pending);
        assert_eq!(bm.pending_delete_count(), 1);
        let pending = bm.snapshot_pending_deletes();
        assert_eq!(pending.get(&guid), Some(&10));
        assert_eq!(bm.pending_delete_count(), 0);
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
        }])
        .unwrap();
        assert!(
            bm.try_evict_cold(guid),
            "successful write-through releases flushing protection",
        );
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
