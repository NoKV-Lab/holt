//! `BufferManager` — LRU-bounded blob cache.
//!
//! Sits between a [`Tree`](crate::Tree) and its underlying
//! [`Backend`]. Itself implements `Backend`, so it's a transparent
//! drop-in: callers see the same `read_blob` / `write_blob` /
//! `flush` API, but reads of recently-touched blobs hit the cache
//! and skip the inner backend's I/O.
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
//! backend in one call). Internal call sites that produce a new
//! blob (spillover) or unlink one (erase's `SubtreeGone` /
//! merge) go through [`BufferManager::install_new_blob`] /
//! [`BufferManager::mark_for_delete`] instead, so the backend
//! write or manifest mutation is deferred until the next flush —
//! invariant **W2D** below.
//!
//! ## Dirty tracking + deferred deletes
//!
//! Every walker write tags its target blob via
//! [`BufferManager::mark_dirty`] with the WAL seq that authored
//! the change. The internal dirty state keeps the **lowest**
//! unflushed seq per blob — that value is the WAL trim watermark
//! for that blob (records below it are already in backend, so the
//! WAL doesn't need them). A checkpoint round moves drained
//! entries into an in-flight `flushing` set until their cached
//! bytes have reached the backend; eviction treats both maps as
//! protected.
//!
//! Erase ops that empty a child blob queue a deferred deletion
//! via [`BufferManager::mark_for_delete`] — the `backend.delete_blob`
//! syscall runs only after the corresponding WAL record is on
//! disk.
//!
//! Invariants:
//!
//! - **I1**: a `(guid, _)` entry exists in `dirty` iff the cached
//!   image of `guid` is newer than the backend image.
//! - **I2**: WAL `trim_id <= min(dirty.values()) - 1` (or
//!   `next_seq - 1` if `dirty` is empty).
//! - **I3**: [`BufferManager::snapshot_dirty`] drains the map
//!   atomically, so `mark_dirty` calls that race with a checkpoint
//!   round land in the new (empty) map and are tracked for the
//!   next round. [`BufferManager::snapshot_pending_deletes`] has
//!   the same drain semantics.
//! - **W2D**: any byte written to `backend.data_file` or any
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
//!   outside pin). O(n) walk over the cache, called only on the
//!   overflow path; the background eviction thread handles
//!   steady-state reclaim cheaply.
//! - **Background sweep** ([`crate::checkpoint`] eviction
//!   thread) — periodic walk based on the same `last_touched`
//!   tick + `eviction_idle_ticks` threshold. Snapshots the cache
//!   under shard locks, then drops the snapshot's Arc clones
//!   before calling `try_evict_cold` so the BM's `strong_count`
//!   check sees only the shard's own reference.
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

use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use crate::api::errors::{Error, Result};
use crate::concurrency::{Guard as LatchGuard, HybridLatch};
use crate::layout::BlobGuid;

use super::backend::{AlignedBlobBuf, Backend};

/// Sentinel seq for dirty / pending-delete entries that originate
/// from purely structural mutations (compact, merge pass) — they
/// have no corresponding WAL record and so must not pin the WAL
/// trim watermark. `min(dirty.values())` is what gates the
/// watermark; using `u64::MAX` ensures a structural entry only
/// matters for trim decisions if no real WAL-seqed entry is
/// present alongside it (in which case dirty is non-empty and
/// the truncate gate already refuses to fire).
pub const STRUCTURAL_SEQ: u64 = u64::MAX;

const BOOKKEEPING_SHARDS: usize = 64;

/// One pre-snapshotted blob image ready for checkpoint write-through.
///
/// The bytes are owned by the checkpoint round / I/O task so the
/// backend write never holds a cache read guard. `expected_seq` is
/// the dirty-map value that was drained into `flushing`; successful
/// batch writes retire that exact flushing entry without stomping a
/// racing writer's newer dirty entry.
pub(crate) struct WriteThroughEntry {
    pub(crate) guid: BlobGuid,
    pub(crate) bytes: AlignedBlobBuf,
    pub(crate) expected_seq: u64,
}

#[derive(Default)]
struct MutationState {
    /// New dirty entries not yet claimed by a checkpoint round.
    dirty: HashMap<BlobGuid, u64>,
    /// Dirty entries drained by a checkpoint round whose cached
    /// image still has to survive until checkpoint write-through
    /// completes.
    flushing: HashMap<BlobGuid, u64>,
    /// Blobs unlinked from the tree but not yet deleted from the
    /// backend manifest because WAL/checkpoint ordering still owns
    /// them.
    pending_deletes: HashMap<BlobGuid, u64>,
    /// In-memory maintenance hints for blobs whose local garbage
    /// is worth checking before the next online compact pass.
    ///
    /// This is advisory only. Dirty / flushing / pending-delete own
    /// correctness; candidate loss can only delay maintenance until
    /// a later seed scan or explicit compact pass rediscovers it.
    compact_candidates: MaintenanceQueue,
    /// In-memory maintenance hints for parent blobs that own at
    /// least one `BlobNode` crossing and may be worth a merge pass.
    merge_candidates: MaintenanceQueue,
}

impl MutationState {
    fn is_protected(&self, guid: &BlobGuid) -> bool {
        self.dirty.contains_key(guid) || self.flushing.contains_key(guid)
    }

    fn is_protected_or_pending(&self, guid: &BlobGuid) -> bool {
        self.is_protected(guid) || self.pending_deletes.contains_key(guid)
    }

    fn remove_dirty(&mut self, guid: &BlobGuid) {
        self.dirty.remove(guid);
        self.flushing.remove(guid);
    }

    fn remove_maintenance_candidates(&mut self, guid: &BlobGuid) -> (bool, bool) {
        (
            self.compact_candidates.remove(guid),
            self.merge_candidates.remove(guid),
        )
    }
}

#[derive(Default)]
struct MaintenanceQueue {
    set: HashSet<BlobGuid>,
    queue: VecDeque<BlobGuid>,
}

impl MaintenanceQueue {
    fn insert(&mut self, guid: BlobGuid) -> bool {
        if self.set.insert(guid) {
            self.queue.push_back(guid);
            true
        } else {
            false
        }
    }

    fn remove(&mut self, guid: &BlobGuid) -> bool {
        self.set.remove(guid)
    }

    fn pop_batch(&mut self, limit: usize) -> Vec<BlobGuid> {
        let mut out = Vec::new();
        while out.len() < limit {
            let Some(guid) = self.queue.pop_front() else {
                break;
            };
            if self.set.remove(&guid) {
                out.push(guid);
            }
        }
        out
    }
}

fn bookkeeping_shard_idx(guid: &BlobGuid) -> usize {
    debug_assert!(BOOKKEEPING_SHARDS.is_power_of_two());

    let mut lo_bytes = [0u8; 8];
    let mut hi_bytes = [0u8; 8];
    lo_bytes.copy_from_slice(&guid[0..8]);
    hi_bytes.copy_from_slice(&guid[8..16]);
    let lo = u64::from_le_bytes(lo_bytes);
    let hi = u64::from_le_bytes(hi_bytes);
    let mut h = lo ^ hi.rotate_left(27);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as usize) & (BOOKKEEPING_SHARDS - 1)
}

#[derive(Debug, Clone, Copy)]
enum CandidateKind {
    Compact,
    Merge,
}

fn pop_candidate_batch(
    shards: &[Mutex<MutationState>; BOOKKEEPING_SHARDS],
    cursor: &AtomicUsize,
    total: &AtomicUsize,
    kind: CandidateKind,
    limit: usize,
) -> Vec<BlobGuid> {
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let start = cursor.fetch_add(1, Ordering::Relaxed) & (BOOKKEEPING_SHARDS - 1);
    for offset in 0..BOOKKEEPING_SHARDS {
        let idx = (start + offset) & (BOOKKEEPING_SHARDS - 1);
        let shard = &shards[idx];
        let mut state = shard.lock().unwrap();
        let queue = match kind {
            CandidateKind::Compact => &mut state.compact_candidates,
            CandidateKind::Merge => &mut state.merge_candidates,
        };
        let remaining = limit - out.len();
        let popped = queue.pop_batch(remaining);
        total.fetch_sub(popped.len(), Ordering::Relaxed);
        out.extend(popped);
        if out.len() == limit {
            return out;
        }
    }
    out
}

/// LRU-bounded blob cache; see the module docs.
pub struct BufferManager {
    backend: Arc<dyn Backend>,
    capacity: usize,
    /// Sharded blob cache. `DashMap` shards by `BlobGuid` so
    /// concurrent `pin` / `get_cached` on different blobs hit
    /// different shards — no single global mutex on the hot read
    /// path. The background eviction thread + each entry's
    /// `last_touched` tick give "approximate LRU" without needing
    /// an O(n) front-of-deque touch on every hit.
    cache: DashMap<BlobGuid, Arc<CachedBlob>>,
    /// Per-blob mutation bookkeeping, sharded by `BlobGuid`.
    ///
    /// Each shard owns the dirty, flushing, and pending-delete
    /// entries for the same set of blobs. Keeping those three maps
    /// under one shard lock gives `mark_dirty` / `mark_for_delete`
    /// one short critical section with no global dirty mutex on the
    /// persistent write hot path.
    mutation: [Mutex<MutationState>; BOOKKEEPING_SHARDS],
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
    walker_ops: AtomicU64,
    walker_blob_hops: AtomicU64,
    max_blob_hops: AtomicU64,
    max_cross_blob_depth: AtomicU64,
    spillover_count: AtomicU64,
    merge_count: AtomicU64,
}

/// A single cached blob. Callers obtain one via
/// [`BufferManager::pin`] and then take an optimistic / shared /
/// exclusive guard on it to access the underlying 512 KB buffer
/// with zero copies.
///
/// Holding the `Arc<CachedBlob>` prevents the entry from being
/// evicted, so traversals that pin a blob can borrow into it for
/// as long as the pin is alive.
pub struct CachedBlob {
    latch: HybridLatch,
    buf: UnsafeCell<AlignedBlobBuf>,
    /// Stamp set by `BufferManager` on every `pin` / `get_cached`.
    /// Read by the eviction thread to decide if this entry is
    /// cold enough to drop. Relaxed reads/writes — see
    /// [`BufferManager::clock`].
    last_touched: AtomicU64,
}

// SAFETY: every access to `buf` is gated by `latch`, which provides
// the standard reader-writer exclusion (plus an optimistic mode
// whose reads are revalidated by the caller before being trusted).
// The `UnsafeCell` only marks the interior-mutability; the actual
// concurrency contract is enforced by `HybridLatch`.
unsafe impl Sync for CachedBlob {}

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

impl CachedBlob {
    fn new(buf: AlignedBlobBuf) -> Self {
        Self {
            latch: HybridLatch::new(),
            buf: UnsafeCell::new(buf),
            last_touched: AtomicU64::new(0),
        }
    }

    /// Logical tick at which this entry was last looked up. Used
    /// by the eviction thread to classify the entry as cold.
    #[must_use]
    pub(crate) fn last_touched(&self) -> u64 {
        self.last_touched.load(Ordering::Relaxed)
    }

    /// Wait-free read snapshot. No real lock taken — the caller
    /// reads bytes through [`OptimisticGuard::as_slice`] and then
    /// calls [`OptimisticGuard::validate`] to confirm no writer
    /// lapped the snapshot. If validation fails the caller must
    /// discard everything read and restart.
    pub fn read_optimistic(&self) -> OptimisticGuard<'_> {
        OptimisticGuard {
            latch: LatchGuard::optimistic(&self.latch),
            buf: &self.buf,
        }
    }

    /// Shared read access — blocks while a writer holds the latch
    /// exclusively, but N shared readers run concurrently.
    pub fn read(&self) -> BlobReadGuard<'_> {
        BlobReadGuard {
            _latch: LatchGuard::shared(&self.latch),
            buf: &self.buf,
        }
    }

    /// Exclusive write access — blocks until idle, then runs
    /// alone. Bumps the version on release so concurrent
    /// optimistic readers detect the change and restart.
    pub fn write(&self) -> BlobWriteGuard<'_> {
        BlobWriteGuard {
            _latch: LatchGuard::exclusive(&self.latch),
            buf: &self.buf,
        }
    }
}

/// Wait-free guard returned by [`CachedBlob::read_optimistic`].
///
/// Reads from `as_slice()` may be **torn** (a concurrent writer
/// could be mid-mutation). The caller must finish reading and
/// call [`OptimisticGuard::validate`]; if `validate` returns
/// `false`, every byte read through this guard is potentially
/// stale and must be discarded.
pub struct OptimisticGuard<'a> {
    latch: LatchGuard<'a>,
    buf: &'a UnsafeCell<AlignedBlobBuf>,
}

impl<'a> OptimisticGuard<'a> {
    /// Pointer-style view of the 512 KB buffer. Bytes may be torn
    /// — see the type-level docs.
    #[must_use]
    pub fn as_slice(&self) -> &'a [u8] {
        // SAFETY: the optimistic guard holds the latch in
        // `Optimistic` mode (no real lock); reads through this
        // borrow may race with a writer. The walker treats any
        // result derived from such a borrow as untrusted until
        // `validate()` confirms it; corrupt bodies surface as
        // `Error::NodeCorrupt` rather than panics because the
        // layout decoders bounds-check every field.
        unsafe { (&*self.buf.get()).as_slice() }
    }

    /// Returns `true` if no exclusive writer modified the buffer
    /// between the snapshot and now.
    #[must_use]
    pub fn validate(&self) -> bool {
        self.latch.validate()
    }
}

/// Shared-mode read guard returned by [`CachedBlob::read`].
///
/// Derefs to `&AlignedBlobBuf`; call `.as_slice()` for byte-level
/// access.
pub struct BlobReadGuard<'a> {
    _latch: LatchGuard<'a>,
    buf: &'a UnsafeCell<AlignedBlobBuf>,
}

impl Deref for BlobReadGuard<'_> {
    type Target = AlignedBlobBuf;
    fn deref(&self) -> &AlignedBlobBuf {
        // SAFETY: shared-mode latch excludes writers.
        unsafe { &*self.buf.get() }
    }
}

/// Exclusive-mode write guard returned by [`CachedBlob::write`].
///
/// Derefs to `&mut AlignedBlobBuf`; call `.as_mut_slice()` for
/// byte-level access. For walker paths that mutate the typed
/// [`crate::store::BlobFrame`] view, prefer [`Self::frame`].
pub struct BlobWriteGuard<'a> {
    _latch: LatchGuard<'a>,
    buf: &'a UnsafeCell<AlignedBlobBuf>,
}

impl BlobWriteGuard<'_> {
    /// Construct a [`crate::store::BlobFrame`] view over this guard's buffer.
    pub fn frame(&mut self) -> super::BlobFrame<'_> {
        super::BlobFrame::wrap(self.as_mut_slice())
    }
}

impl Deref for BlobWriteGuard<'_> {
    type Target = AlignedBlobBuf;
    fn deref(&self) -> &AlignedBlobBuf {
        // SAFETY: exclusive-mode latch excludes all other access.
        unsafe { &*self.buf.get() }
    }
}

impl DerefMut for BlobWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut AlignedBlobBuf {
        // SAFETY: exclusive-mode latch excludes all other access,
        // and `&mut self` ensures no other borrow of this guard
        // exists.
        unsafe { &mut *self.buf.get() }
    }
}

impl BufferManager {
    /// Wrap `backend` with a cache of at most `capacity` blobs
    /// (each blob is 512 KB on the heap). A `capacity` of 0 is
    /// clamped to 1.
    #[must_use]
    pub fn new(backend: Arc<dyn Backend>, capacity: usize) -> Self {
        Self {
            backend,
            capacity: capacity.max(1),
            cache: DashMap::new(),
            mutation: std::array::from_fn(|_| Mutex::new(MutationState::default())),
            compact_candidate_cursor: AtomicUsize::new(0),
            merge_candidate_cursor: AtomicUsize::new(0),
            compact_candidate_total: AtomicUsize::new(0),
            merge_candidate_total: AtomicUsize::new(0),
            clock: AtomicU64::new(1),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            optimistic_restarts: AtomicU64::new(0),
            walker_ops: AtomicU64::new(0),
            walker_blob_hops: AtomicU64::new(0),
            max_blob_hops: AtomicU64::new(0),
            max_cross_blob_depth: AtomicU64::new(0),
            spillover_count: AtomicU64::new(0),
            merge_count: AtomicU64::new(0),
        }
    }

    /// Current logical clock value. Read by the eviction
    /// thread to compare against each entry's `last_touched`. The
    /// returned tick is `Relaxed` — fine for "how cold is this
    /// entry" decisions, not for cross-thread synchronisation.
    pub(crate) fn clock_tick(&self) -> u64 {
        self.clock.load(Ordering::Relaxed)
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
    /// without consulting the inner backend). Relaxed-ordered;
    /// reads are observability-only.
    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Cumulative cache lookup misses — every miss is followed by
    /// an `inner_backend.read_blob` and an `insert_into_cache`.
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

    /// Cumulative mutation walker calls (`insert_multi` /
    /// `erase_multi`). A `rename` or `txn` contributes one count per
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

    /// Internal: look up `guid` in the cache. On a hit, stamps
    /// the entry's `last_touched` with the current clock tick so
    /// the eviction thread treats this hit as fresh. Bumps the
    /// `cache_hits` / `cache_misses` telemetry counter accordingly.
    fn get_cached(&self, guid: BlobGuid) -> Option<Arc<CachedBlob>> {
        let Some(entry) = self.cache.get(&guid) else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let arc = Arc::clone(entry.value());
        // Drop the shard read guard before touching the atomic —
        // not strictly required (the atomic is independent) but
        // keeps shard occupancy short.
        drop(entry);
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        arc.last_touched.store(tick, Ordering::Relaxed);
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        Some(arc)
    }

    /// Internal: same as [`Self::get_cached`] but **does not** bump
    /// `cache_hits` / `cache_misses` and **does not** refresh the
    /// entry's `last_touched` tick. Used by introspection paths
    /// (`Tree::stats`, metrics scrapes) that need to read blob
    /// state without polluting the very counters they're about
    /// to report or skewing the LRU sweep's view of which entries
    /// are cold.
    fn get_cached_silent(&self, guid: BlobGuid) -> Option<Arc<CachedBlob>> {
        let entry = self.cache.get(&guid)?;
        let arc = Arc::clone(entry.value());
        drop(entry);
        Some(arc)
    }

    fn mutation_shard(&self, guid: BlobGuid) -> &Mutex<MutationState> {
        &self.mutation[bookkeeping_shard_idx(&guid)]
    }

    fn is_pending_delete(&self, guid: BlobGuid) -> bool {
        self.mutation_shard(guid)
            .lock()
            .unwrap()
            .pending_deletes
            .contains_key(&guid)
    }

    fn pending_delete_not_found(guid: BlobGuid) -> Error {
        Error::BackendIo(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("blob {:02x?} is pending delete", &guid[..4]),
        ))
    }

    /// Internal: insert a freshly-loaded blob into the cache.
    /// Idempotent under concurrent inserts. Stamps the new entry's
    /// `last_touched` so it doesn't look cold to the eviction
    /// thread on its very next sweep.
    fn insert_into_cache(&self, guid: BlobGuid, contents: &AlignedBlobBuf) {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        let inserted = self.cache.entry(guid).or_insert_with(|| {
            let entry = Arc::new(CachedBlob::new(contents.clone()));
            entry.last_touched.store(tick, Ordering::Relaxed);
            entry
        });
        // Re-stamp even on existing entries — a concurrent thread
        // may have populated the slot while we read from backend;
        // either way "just observed" is "freshly touched".
        inserted.value().last_touched.store(tick, Ordering::Relaxed);
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
            if self.try_evict_lru() {
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
    /// `dirty ⟺ cache image newer than backend` (invariant I1)
    /// and `pending-delete ⟺ cache image must outlive the
    /// manifest unlink` properties. Without this check, an inline
    /// overflow can drop a cache image while its dirty entry stays
    /// in the dirty map — the next checkpoint's `snapshot_bytes`
    /// returns `None` for that guid and (pre-fix) silently skipped
    /// it; in memory mode the cache mutation was lost outright,
    /// in persistent mode the WAL truncate gate stuck closed
    /// forever. Matches `try_evict_cold`'s guard for the bg sweep.
    fn try_evict_lru(&self) -> bool {
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
            if protected_snap.contains(&guid) {
                continue;
            }
            let tick = kv.value().last_touched.load(Ordering::Relaxed);
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
    /// backend.
    fn evict_from_cache(&self, guid: BlobGuid) {
        self.cache.remove(&guid);
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state.remove_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
    }

    /// Pin a blob in cache and return an `Arc<CachedBlob>` over it.
    ///
    /// On a cache miss, the blob is loaded from the inner backend
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
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        if let Some(entry) = self.get_cached(guid) {
            return Ok(entry);
        }
        // Cache miss — load from inner backend, then take a second
        // lookup so the cache, not our scratch buffer, owns the
        // canonical entry.
        let mut scratch = AlignedBlobBuf::zeroed();
        self.backend.read_blob(guid, &mut scratch)?;
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        self.insert_into_cache(guid, &scratch);
        // Almost always cached now; if another thread evicted it
        // in the gap, fall back to a fresh insert with our scratch.
        if let Some(entry) = self.get_cached(guid) {
            return Ok(entry);
        }
        // Pathological: insert raced with eviction. Build an
        // entry directly from scratch and force-insert it.
        let entry = Arc::new(CachedBlob::new(scratch));
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        entry.last_touched.store(tick, Ordering::Relaxed);
        self.cache.insert(guid, Arc::clone(&entry));
        Ok(entry)
    }

    /// Like [`Self::pin`] but does not bump `cache_hits` /
    /// `cache_misses` and does not refresh the `last_touched`
    /// tick on a hit — used by introspection paths
    /// (`Tree::stats`, metrics scrapes, internal asserts) that
    /// must not perturb the very telemetry they're about to
    /// report or rescue cold entries from the eviction sweep
    /// just by looking at them.
    ///
    /// **Miss-path behaviour**: a `pin_silent` miss still loads
    /// the blob from the inner backend and inserts it into the
    /// cache (via `insert_into_cache`, which stamps
    /// `last_touched` like any other insert) — the alternative
    /// (return `Err`) would surprise callers and the load is
    /// the only sane way to fulfil the pin contract. The miss
    /// itself is just not reflected in `cache_misses`. Hot
    /// scrape paths should expect most calls to be hits.
    pub fn pin_silent(&self, guid: BlobGuid) -> Result<Arc<CachedBlob>> {
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        if let Some(entry) = self.get_cached_silent(guid) {
            return Ok(entry);
        }
        let mut scratch = AlignedBlobBuf::zeroed();
        self.backend.read_blob(guid, &mut scratch)?;
        if self.is_pending_delete(guid) {
            return Err(Self::pending_delete_not_found(guid));
        }
        self.insert_into_cache(guid, &scratch);
        if let Some(entry) = self.get_cached_silent(guid) {
            return Ok(entry);
        }
        let entry = Arc::new(CachedBlob::new(scratch));
        // We still stamp last_touched on the truly-pathological
        // race-with-eviction fallback path — the entry is being
        // freshly inserted, the tick reflects that creation, not
        // a "touch" by the scrape.
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        entry.last_touched.store(tick, Ordering::Relaxed);
        self.cache.insert(guid, Arc::clone(&entry));
        Ok(entry)
    }

    // ---------- dirty tracking ----------

    /// Tag `guid` as dirty at WAL seq `txn_id`.
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
    pub fn mark_dirty(&self, guid: BlobGuid, txn_id: u64) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
        if state.pending_deletes.contains_key(&guid) {
            return;
        }
        state
            .dirty
            .entry(guid)
            .and_modify(|cur| *cur = (*cur).min(txn_id))
            .or_insert(txn_id);
    }

    /// Drain the current dirty entries from every bookkeeping shard,
    /// leaving empty per-shard dirty maps behind for concurrent
    /// writers.
    ///
    /// Returned map maps `guid -> lowest unflushed txn_id`. The
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
            let mut snap = std::mem::take(&mut state.dirty);
            snap.retain(|guid, _| !state.pending_deletes.contains_key(guid));
            for (guid, txn_id) in snap {
                state
                    .flushing
                    .entry(guid)
                    .and_modify(|cur| *cur = (*cur).min(txn_id))
                    .or_insert(txn_id);
                out.insert(guid, txn_id);
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
    /// snapshotted entries that didn't make it to backend must stay
    /// tracked for the next round.
    pub fn restore_dirty(&self, entries: HashMap<BlobGuid, u64>) {
        if entries.is_empty() {
            return;
        }
        for (guid, t) in entries {
            let mut state = self.mutation_shard(guid).lock().unwrap();
            if state.pending_deletes.contains_key(&guid) {
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

    /// Tag `guid` for **deferred** backend deletion at WAL seq
    /// `txn_id`. Removes the blob from cache + dirty (the cache
    /// image is dead; a lingering dirty entry would chase a
    /// soon-deleted slot) and queues the `backend.delete_blob`
    /// call for the next checkpoint round.
    ///
    /// Used by the erase walker's `SubtreeGone` branch. The naive
    /// alternative — calling `bm.delete_blob` inline — modifies
    /// the in-memory manifest before the WAL record covering the
    /// unlink is durable; a racing `backend.flush` (from any other
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
    pub fn mark_for_delete(&self, guid: BlobGuid, txn_id: u64) {
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state
            .pending_deletes
            .entry(guid)
            .and_modify(|cur| *cur = (*cur).min(txn_id))
            .or_insert(txn_id);
        state.remove_dirty(&guid);
        let removed = state.remove_maintenance_candidates(&guid);
        drop(state);
        self.decrement_candidate_totals(removed);
        self.cache.remove(&guid);
    }

    /// Drain the current pending-delete entries from every
    /// bookkeeping shard, leaving empty per-shard maps behind.
    /// Caller (checkpoint round / manual `Tree::checkpoint`) is
    /// responsible for executing each `backend.delete_blob` or
    /// restoring on failure.
    #[must_use]
    pub fn snapshot_pending_deletes(&self) -> HashMap<BlobGuid, u64> {
        let mut out = HashMap::new();
        for shard in &self.mutation {
            let mut state = shard.lock().unwrap();
            out.extend(std::mem::take(&mut state.pending_deletes));
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
            state
                .pending_deletes
                .entry(g)
                .and_modify(|cur| *cur = (*cur).min(t))
                .or_insert(t);
        }
    }

    /// Number of blobs waiting for deferred backend deletion.
    /// Reads as zero under the WAL-truncate gate are part of the
    /// "WAL records are all redundant" invariant.
    #[must_use]
    pub fn pending_delete_count(&self) -> usize {
        self.mutation
            .iter()
            .map(|shard| shard.lock().unwrap().pending_deletes.len())
            .sum()
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

    /// Execute a queued deletion against the inner backend.
    /// Manifest mutation is in-memory; subsequent `backend.flush`
    /// makes it durable. Failure is the caller's restoration
    /// concern.
    pub(crate) fn execute_pending_delete(&self, guid: BlobGuid) -> Result<()> {
        self.backend.delete_blob(guid)
    }

    /// Snapshot the cached bytes for `guid` into a freshly allocated
    /// `AlignedBlobBuf`. Returns `None` if the blob isn't cached.
    ///
    /// Used by the background checkpointer to hand off bytes to
    /// the I/O worker thread without keeping the shared read guard
    /// open across the actual `backend.write_blob` call. The read
    /// guard is held only for the duration of the 512 KB memcpy, so
    /// writers don't block on long-running (especially io_uring)
    /// I/O.
    pub(crate) fn snapshot_bytes(&self, guid: BlobGuid) -> Option<AlignedBlobBuf> {
        let entry = self.get_cached(guid)?;
        let buf = entry.read();
        Some(buf.clone())
    }

    /// Push a whole checkpoint snapshot to the inner backend using
    /// its native batch path, then retire each matching flushing
    /// entry. On backend error the caller must restore the whole
    /// dirty snapshot; we intentionally retire nothing because the
    /// backend contract permits an arbitrary written prefix.
    pub(crate) fn write_through_batch(&self, entries: &[WriteThroughEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let writes: Vec<_> = entries
            .iter()
            .map(|entry| (entry.guid, &entry.bytes))
            .collect();
        self.backend.write_blobs(&writes)?;
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
    }

    /// Forward `flush` to the inner backend without touching the
    /// cache. Used by the I/O worker for `IoTask::Sync`.
    pub(crate) fn backend_flush(&self) -> Result<()> {
        self.backend.flush()
    }

    /// Stage a freshly-created blob in cache and tag it dirty at
    /// `seq` — the unified `mark_dirty → checkpoint round → backend
    /// write` protocol takes ownership from there.
    ///
    /// Used by spillover when it produces a new child blob: the
    /// bytes must NOT reach backend before the WAL record covering
    /// the op that triggered spillover (invariant W2D). Deferring
    /// the backend write via the dirty map preserves that ordering;
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
    /// backend). Inline overflow eviction is therefore skipped
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
        let mut state = self.mutation_shard(guid).lock().unwrap();
        state
            .dirty
            .entry(guid)
            .and_modify(|cur| *cur = (*cur).min(seq))
            .or_insert(seq);
        drop(entry);
    }
}

impl Backend for BufferManager {
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
        // Cache hit?
        if let Some(entry) = self.get_cached(guid) {
            let buf = entry.read();
            dst.as_mut_slice().copy_from_slice(buf.as_slice());
            return Ok(());
        }
        // Cache miss — load from inner backend and cache.
        self.backend.read_blob(guid, dst)?;
        self.insert_into_cache(guid, dst);
        Ok(())
    }

    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
        // Transparent write-through: if cached, refresh the
        // cached image; either way, always write to the inner
        // backend in the same call so durability is unchanged.
        if let Some(entry) = self.get_cached(guid) {
            let mut buf = entry.write();
            buf.as_mut_slice().copy_from_slice(src.as_slice());
        }
        self.backend.write_blob(guid, src)?;
        // Backend now holds these exact bytes; any pending dirty
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
            if let Some(entry) = self.get_cached(*guid) {
                let mut buf = entry.write();
                buf.as_mut_slice().copy_from_slice(src.as_slice());
            }
        }
        self.backend.write_blobs(writes)?;
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
        self.backend.delete_blob(guid)
    }

    fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
        self.backend.list_blobs()
    }

    fn flush(&self) -> Result<()> {
        // Write-through mode: nothing pending in cache.
        self.backend.flush()
    }

    fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
        if self.is_pending_delete(guid) {
            return Ok(false);
        }
        // Fast path: shard-local check without consulting the
        // inner backend.
        if self.cache.contains_key(&guid) {
            return Ok(true);
        }
        self.backend.has_blob(guid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::backend::MemoryBackend;

    fn make_buf(byte_at_100: u8) -> AlignedBlobBuf {
        let mut b = AlignedBlobBuf::zeroed();
        b.as_mut_slice()[100] = byte_at_100;
        b
    }

    #[test]
    fn read_caches_after_first_load() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
    fn lru_eviction_at_capacity() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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

    /// Regression: prior to the v0.2.1 fix, `try_evict_lru` only
    /// checked `Arc::strong_count == 1` — it would happily evict
    /// a dirty cache image, leaving the dirty entry orphaned in
    /// the dirty map. That broke invariant I1 (dirty ⟺ cache
    /// newer than backend) and silently lost the cache mutation
    /// (memory mode) / stuck the WAL truncate gate forever
    /// (persistent mode).
    #[test]
    fn lru_eviction_skips_dirty_entries() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        // Pre-populate the inner backend with three blobs whose
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
    // because the inner backend manifest intentionally keeps the
    // blob until checkpoint applies the deferred delete.

    #[test]
    fn write_through_propagates_to_inner_backend() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        inner.write_blob([0xEF; 16], &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner.clone(), 4);

        // Prime the cache.
        let mut dst = AlignedBlobBuf::zeroed();
        bm.read_blob([0xEF; 16], &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);

        // Overwrite via the BM.
        bm.write_blob([0xEF; 16], &make_buf(99)).unwrap();

        // Subsequent read through the BM sees the updated value
        // (came from the refreshed cache, not the inner backend).
        bm.read_blob([0xEF; 16], &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 99);
    }

    #[test]
    fn delete_evicts_from_cache_and_inner() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
            "pending-delete child must not be reloaded from backend"
        );
        bm.mark_dirty([0x44; 16], 12);
        let mut restore = HashMap::new();
        restore.insert([0x44; 16], 13);
        bm.restore_dirty(restore);
        assert_eq!(bm.dirty_count(), 0);
        assert_eq!(bm.pending_delete_count(), 1);
    }

    #[test]
    fn has_blob_fast_path_avoids_inner_when_cached() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
    fn mark_dirty_keeps_lowest_txn_id() {
        let bm = BufferManager::new(Arc::new(MemoryBackend::new()), 4);
        bm.mark_dirty([0x01; 16], 50);
        bm.mark_dirty([0x01; 16], 30);
        bm.mark_dirty([0x01; 16], 99);
        assert_eq!(bm.dirty_count(), 1);
        let snap = bm.snapshot_dirty();
        assert_eq!(snap[&[0x01; 16]], 30);
    }

    #[test]
    fn snapshot_dirty_drains_atomically() {
        let bm = BufferManager::new(Arc::new(MemoryBackend::new()), 4);
        bm.mark_dirty([0x01; 16], 10);
        bm.mark_dirty([0x02; 16], 20);

        let snap = bm.snapshot_dirty();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[&[0x01; 16]], 10);
        assert_eq!(snap[&[0x02; 16]], 20);

        // After snapshot the live map is empty.
        assert_eq!(bm.dirty_count(), 0);

        // Concurrent mark_dirty lands in the fresh empty map.
        bm.mark_dirty([0x03; 16], 99);
        assert_eq!(bm.dirty_count(), 1);
        let next = bm.snapshot_dirty();
        assert_eq!(next[&[0x03; 16]], 99);
    }

    #[test]
    fn snapshot_dirty_drains_every_bookkeeping_shard() {
        let bm = BufferManager::new(Arc::new(MemoryBackend::new()), 4);
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
            bm.mark_dirty(guid.expect("filled"), shard as u64 + 1);
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let bm = BufferManager::new(Arc::new(MemoryBackend::new()), 4);
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        let bm = BufferManager::new(inner, 4);

        bm.mark_dirty([0x88; 16], 100);
        assert_eq!(bm.dirty_count(), 1);

        // The Backend-trait write_blob is write-through and so
        // satisfies the dirty entry by construction.
        Backend::write_blob(&bm, [0x88; 16], &make_buf(9)).unwrap();
        assert_eq!(bm.dirty_count(), 0);
    }

    #[test]
    fn delete_blob_drops_dirty_entry() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        inner.write_blob([0x99; 16], &make_buf(1)).unwrap();
        let bm = BufferManager::new(inner, 4);

        let _ = bm.pin([0x99; 16]).unwrap();
        bm.mark_dirty([0x99; 16], 7);
        assert_eq!(bm.dirty_count(), 1);

        Backend::delete_blob(&bm, [0x99; 16]).unwrap();
        assert_eq!(
            bm.dirty_count(),
            0,
            "deleted blobs must not linger as flush candidates"
        );
    }

    #[test]
    fn install_new_blob_caches_and_marks_dirty_without_backend_write() {
        // The unified-protocol fix: spillover's new child blob
        // must land in cache + dirty, NOT in the inner backend,
        // so the checkpoint round can enforce the W2D ordering
        // (WAL flush THEN backend write).
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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

        // Inner backend has nothing yet.
        assert!(
            !inner.has_blob(new_guid).unwrap(),
            "install_new_blob must defer the backend write to the checkpoint round",
        );

        // Pinning the blob returns the cached image.
        let pin = bm.pin(new_guid).unwrap();
        let guard = pin.read();
        assert_eq!(guard.as_slice()[200], 0x77);
        drop(guard);
        drop(pin);

        // After the production checkpoint primitive runs, the inner
        // backend has the bytes and the dirty entry is cleared.
        let snap = bm.snapshot_dirty();
        let bytes = bm.snapshot_bytes(new_guid).unwrap();
        bm.write_through_batch(&[WriteThroughEntry {
            guid: new_guid,
            bytes,
            expected_seq: snap[&new_guid],
        }])
        .unwrap();
        bm.backend_flush().unwrap();
        assert_eq!(bm.dirty_count(), 0);
        assert!(inner.has_blob(new_guid).unwrap());
        let mut dst = AlignedBlobBuf::zeroed();
        inner.read_blob(new_guid, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[200], 0x77);
    }

    #[test]
    fn concurrent_reads_on_different_blobs_progress() {
        use std::thread;

        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
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
