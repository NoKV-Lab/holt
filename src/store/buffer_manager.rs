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
//! - **Synchronous** [`BufferManager::commit`] — drives one blob
//!   per call from [`crate::Tree::checkpoint`] or per-op
//!   `memory_flush_on_write` mode. Writes the cache image to backend
//!   and atomically clears the blob's dirty entry on success.
//! - **Background checkpointer** — drives a round-based flush of
//!   the entire dirty set; see [`BufferManager::snapshot_dirty`]
//!   / [`BufferManager::restore_dirty`] /
//!   [`BufferManager::min_unflushed_txn`].
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
//! the change. The internal `dirty: Mutex<HashMap<BlobGuid, u64>>`
//! keeps the **lowest** unflushed seq per blob — that value is
//! the WAL trim watermark for that blob (records below it are
//! already in backend, so the WAL doesn't need them).
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
//!   with writers. Used by `BufferManager::commit` (durable write-
//!   through reads the cached image under shared).
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
//!   `BlobFrameRef::wrap` shape, but blocks behind any active
//!   writer.
//! - [`CachedBlob::write`] → [`BlobWriteGuard`] (exclusive). Wrap
//!   with `BlobFrame::wrap(guard.as_mut_slice())` for in-place
//!   mutation. Drop the guard, then call
//!   [`BufferManager::commit`] to flush the change to disk.
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
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use crate::api::errors::Result;
use crate::concurrency::{Guard as LatchGuard, HybridLatch};
use crate::layout::{BlobGuid, MAX_SLOTS};

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
    /// Per-blob lowest unflushed WAL seq. An entry exists ⟺ the
    /// cached image of that blob is newer than the backend image
    /// (invariant **I1**; see module docs). Drained atomically by
    /// [`BufferManager::snapshot_dirty`] so checkpoint rounds and
    /// concurrent writers don't step on each other.
    dirty: Mutex<HashMap<BlobGuid, u64>>,
    /// Blobs the walker has unlinked from their parent in cache
    /// but whose backend slot can't be released yet — invariant
    /// W2D forbids removing them from the manifest before the WAL
    /// record covering the unlink op is durable. The checkpoint
    /// round drains this set **after** Sync (data + manifest
    /// stable) and **before** WAL truncate, so the truncate gate
    /// can promise "everything in this WAL is either redundant
    /// with backend or already pending in dirty/pending-delete".
    ///
    /// `guid -> seq` mirrors `dirty`'s shape: `seq` is the WAL
    /// seq of the op that unlinked this blob.
    pending_deletes: Mutex<HashMap<BlobGuid, u64>>,
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
    /// Per-slot version counters, indexed `[slot - 1]` (1-based
    /// slot indices in the layout, 0-based in this array). Bumped
    /// on every slot mutation routed through [`crate::store::BlobFrame`]
    /// — `alloc_node`, `free_node`, and body-rewrite via
    /// `write_struct_to_slot`. Sized to `MAX_SLOTS` (= 10240) so
    /// every possible slot index has its own counter; ~80 KB per
    /// blob (10240 × `AtomicU64`).
    ///
    /// **Phase 1 status (current):** bumps are emitted on the
    /// writer side but not yet consulted on the reader side. The
    /// walker still uses blob-level [`HybridLatch`] for all
    /// concurrency. Phases 2-3 will introduce per-slot optimistic
    /// reads + cross-blob lock-coupling using these counters as
    /// the re-acquire tokens (walker releases parent latch before
    /// descending into a child blob; re-validates the parent's
    /// slot counter on the way back up).
    ///
    /// Default 64-blob pool → ~5.2 MB total for these counters,
    /// 128-blob pool → ~10.4 MB. Acceptable for the concurrency
    /// gains they unlock.
    slot_versions: Box<[AtomicU64]>,
}

// SAFETY: every access to `buf` is gated by `latch`, which provides
// the standard reader-writer exclusion (plus an optimistic mode
// whose reads are revalidated by the caller before being trusted).
// The `UnsafeCell` only marks the interior-mutability; the actual
// concurrency contract is enforced by `HybridLatch`.
unsafe impl Sync for CachedBlob {}

impl CachedBlob {
    fn new(buf: AlignedBlobBuf) -> Self {
        // 10240 AtomicU64s = 80 KB. Allocated once per CachedBlob;
        // amortised across the blob's lifetime in the cache.
        let slot_versions = (0..MAX_SLOTS as usize)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            latch: HybridLatch::new(),
            buf: UnsafeCell::new(buf),
            last_touched: AtomicU64::new(0),
            slot_versions,
        }
    }

    /// Logical tick at which this entry was last looked up. Used
    /// by the eviction thread to classify the entry as cold.
    #[must_use]
    pub(crate) fn last_touched(&self) -> u64 {
        self.last_touched.load(Ordering::Relaxed)
    }

    /// Acquire-load the current per-slot version counter for the
    /// given 1-based slot index. Returns 0 for `slot == 0` or any
    /// out-of-range index.
    ///
    /// Pairs with the Release-store performed by
    /// [`crate::store::BlobFrame::bump_slot_version`] inside the walker's
    /// mutation paths. Consumers capture a version before a
    /// latch-free read of the slot, then re-load after the read
    /// completes — a mismatch means a writer landed in between
    /// and the caller must restart.
    ///
    /// Phase 1 (current): only consumers are tests. Phases 2-3
    /// of the per-node-latch milestone wire this into the walker's
    /// cross-blob descent paths.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn slot_version(&self, slot: u16) -> u64 {
        if slot == 0 {
            return 0;
        }
        self.slot_versions
            .get((slot as usize) - 1)
            .map_or(0, |av| av.load(Ordering::Acquire))
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
            slot_versions: &self.slot_versions,
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
/// [`crate::store::BlobFrame`] view, prefer [`Self::frame`] over
/// `BlobFrame::wrap(guard.as_mut_slice())` — the frame
/// constructed via `frame()` is wired to the owning blob's
/// per-slot version counters, so mutations through it bump those
/// counters (the foundation for cross-blob lock-coupling in
/// Phases 2-3 of the per-node-latch milestone).
pub struct BlobWriteGuard<'a> {
    _latch: LatchGuard<'a>,
    buf: &'a UnsafeCell<AlignedBlobBuf>,
    /// Reference to the owning blob's per-slot version counters,
    /// threaded through from [`CachedBlob::write`]. Used by
    /// [`Self::frame`] to construct a version-tracking
    /// `BlobFrame`.
    slot_versions: &'a [AtomicU64],
}

impl BlobWriteGuard<'_> {
    /// Construct a version-tracking [`crate::store::BlobFrame`] view over this
    /// guard's buffer. Equivalent to
    /// `BlobFrame::wrap_versioned(self.as_mut_slice(), <slot_versions>)`
    /// — every slot mutation routed through the returned frame
    /// (alloc / free / `write_struct_to_slot`) bumps the
    /// corresponding per-slot version counter on the owning
    /// `CachedBlob`.
    pub fn frame(&mut self) -> super::BlobFrame<'_> {
        let svs = self.slot_versions;
        super::BlobFrame::wrap_versioned(self.as_mut_slice(), svs)
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
            dirty: Mutex::new(HashMap::new()),
            pending_deletes: Mutex::new(HashMap::new()),
            clock: AtomicU64::new(1),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            optimistic_restarts: AtomicU64::new(0),
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

    /// Drop the cache entry for `guid` if (a) it's still cached,
    /// (b) we hold the only outside reference (caller's `Arc` was
    /// dropped before calling), and (c) nothing in the dirty map
    /// references it.
    ///
    /// Returns `true` if an entry was actually evicted.
    pub(crate) fn try_evict_cold(&self, guid: BlobGuid) -> bool {
        {
            let dirty_guard = self.dirty.lock().unwrap();
            if dirty_guard.contains_key(&guid) {
                return false;
            }
        }
        // `DashMap::remove_if` checks the predicate under the
        // shard lock. `strong_count == 1` means only the shard's
        // slot holds the `Arc` (the snapshot's clone was dropped
        // by the caller; see `eviction::run_scan`).
        self.cache
            .remove_if(&guid, |_, entry| Arc::strong_count(entry) == 1)
            .is_some()
    }

    /// Current number of cached blobs. Exercised by the
    /// checkpointer unit tests; not consumed by production code
    /// directly.
    #[allow(dead_code)]
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
        let dirty_snap: std::collections::HashSet<BlobGuid> = {
            let d = self.dirty.lock().unwrap();
            d.keys().copied().collect()
        };
        let pending_snap: std::collections::HashSet<BlobGuid> = {
            let p = self.pending_deletes.lock().unwrap();
            p.keys().copied().collect()
        };

        let mut victim: Option<(BlobGuid, u64)> = None;
        for kv in &self.cache {
            if Arc::strong_count(kv.value()) > 1 {
                continue;
            }
            let guid = *kv.key();
            if dirty_snap.contains(&guid) || pending_snap.contains(&guid) {
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
                    let d = self.dirty.lock().unwrap();
                    if d.contains_key(&guid) {
                        return false;
                    }
                    drop(d);
                    let p = self.pending_deletes.lock().unwrap();
                    if p.contains_key(&guid) {
                        return false;
                    }
                    true
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
        self.dirty.lock().unwrap().remove(&guid);
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
        if let Some(entry) = self.get_cached(guid) {
            return Ok(entry);
        }
        // Cache miss — load from inner backend, then take a second
        // lookup so the cache, not our scratch buffer, owns the
        // canonical entry.
        let mut scratch = AlignedBlobBuf::zeroed();
        self.backend.read_blob(guid, &mut scratch)?;
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
        if let Some(entry) = self.get_cached_silent(guid) {
            return Ok(entry);
        }
        let mut scratch = AlignedBlobBuf::zeroed();
        self.backend.read_blob(guid, &mut scratch)?;
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

    /// Durably write the cached image of `guid` to the inner backend.
    ///
    /// Used by mutation paths after they've finished editing a
    /// pinned buffer: pin → write-guard → mutate → drop guard →
    /// `commit`. Acquires a shared read-guard on the cache entry,
    /// so multiple commits on different blobs run concurrently and
    /// in-flight readers on the same blob are not blocked.
    ///
    /// If `guid` is **not** in cache the call is a no-op — there
    /// is nothing dirty to commit (the inner backend already has
    /// the canonical bytes). This matches the natural use case of
    /// `Tree::checkpoint` running on a freshly-opened tree before
    /// any mutation has loaded the root into cache.
    ///
    /// **Dirty bookkeeping** (invariants I1/I3 in the module docs):
    /// the dirty entry for `guid`, if any, is *drained* before the
    /// backend write so a concurrent `mark_dirty` lands a fresh
    /// (newer-seq) entry rather than getting merged into the one
    /// we're about to clear. On write failure the drained entry is
    /// restored (taking `min` with anything the racing writer
    /// added in the meantime); on success it stays removed.
    ///
    /// Exercised by the buffer-manager unit tests; production
    /// checkpoint paths use `write_through` (CAS-on-seq) instead.
    #[allow(dead_code)]
    pub fn commit(&self, guid: BlobGuid) -> Result<()> {
        let drained = {
            let mut d = self.dirty.lock().unwrap();
            d.remove(&guid)
        };
        if let Some(entry) = self.get_cached(guid) {
            let buf = entry.read();
            if let Err(e) = self.backend.write_blob(guid, &buf) {
                // Backend write failed; put the dirty entry back so
                // a future round retries. Merge with min in case a
                // racing writer already re-added an entry.
                if let Some(t) = drained {
                    let mut d = self.dirty.lock().unwrap();
                    d.entry(guid)
                        .and_modify(|cur| *cur = (*cur).min(t))
                        .or_insert(t);
                }
                return Err(e);
            }
        }
        Ok(())
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
        let mut d = self.dirty.lock().unwrap();
        d.entry(guid)
            .and_modify(|cur| *cur = (*cur).min(txn_id))
            .or_insert(txn_id);
    }

    /// Atomically take the current dirty map, leaving an empty one
    /// behind for concurrent writers.
    ///
    /// Returned map maps `guid -> lowest unflushed txn_id`. The
    /// caller (background checkpointer) is responsible for flushing
    /// each blob and either accepting the drain (on success) or
    /// restoring failed entries via [`Self::restore_dirty`].
    #[must_use]
    pub fn snapshot_dirty(&self) -> HashMap<BlobGuid, u64> {
        let mut d = self.dirty.lock().unwrap();
        std::mem::take(&mut *d)
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
        let mut d = self.dirty.lock().unwrap();
        for (guid, t) in entries {
            d.entry(guid)
                .and_modify(|cur| *cur = (*cur).min(t))
                .or_insert(t);
        }
    }

    /// Lowest unflushed WAL seq across all dirty blobs, or `None`
    /// if every cached image is durable.
    ///
    /// This is the WAL trim watermark: records below this seq can
    /// be discarded because their effects are already in the
    /// backend. If the dirty map is empty, every seq up to
    /// `next_seq - 1` is durable.
    ///
    /// Exercised by the buffer-manager unit tests; the conditional
    /// truncate gate in `Tree::checkpoint` / the bg round uses
    /// `dirty_count()` + `pending_delete_count()` instead, so this
    /// accessor is currently test-only.
    #[allow(dead_code)]
    #[must_use]
    pub fn min_unflushed_txn(&self) -> Option<u64> {
        let d = self.dirty.lock().unwrap();
        d.values().copied().min()
    }

    /// Number of distinct dirty blobs currently tracked. Useful for
    /// metrics + checkpoint-policy thresholds.
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.dirty.lock().unwrap().len()
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
        self.cache.remove(&guid);
        self.dirty.lock().unwrap().remove(&guid);
        let mut p = self.pending_deletes.lock().unwrap();
        p.entry(guid)
            .and_modify(|cur| *cur = (*cur).min(txn_id))
            .or_insert(txn_id);
    }

    /// Atomically take the current pending-delete map, leaving an
    /// empty one behind. Caller (checkpoint round / manual
    /// `Tree::checkpoint`) is responsible for executing each
    /// `backend.delete_blob` or restoring on failure.
    #[must_use]
    pub fn snapshot_pending_deletes(&self) -> HashMap<BlobGuid, u64> {
        let mut p = self.pending_deletes.lock().unwrap();
        std::mem::take(&mut *p)
    }

    /// Merge `entries` back into the pending-delete map, keeping
    /// the per-blob min seq.
    pub fn restore_pending_deletes(&self, entries: HashMap<BlobGuid, u64>) {
        if entries.is_empty() {
            return;
        }
        let mut p = self.pending_deletes.lock().unwrap();
        for (g, t) in entries {
            p.entry(g)
                .and_modify(|cur| *cur = (*cur).min(t))
                .or_insert(t);
        }
    }

    /// Number of blobs waiting for deferred backend deletion.
    /// Reads as zero under the WAL-truncate gate are part of the
    /// "WAL records are all redundant" invariant.
    #[must_use]
    pub fn pending_delete_count(&self) -> usize {
        self.pending_deletes.lock().unwrap().len()
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

    /// Push pre-snapshotted bytes for `guid` directly to the inner
    /// backend, bypassing the cache. Used by the I/O worker
    /// thread, which receives bytes that were snapshotted by the
    /// orchestrator under a shared read guard.
    ///
    /// `expected_seq` is the dirty-map value the checkpointer
    /// observed when it snapshotted this blob. On a successful
    /// backend write the dirty entry is removed **only if it
    /// still equals `expected_seq`** — if a writer raced in
    /// between snapshot and write and bumped the entry to a newer
    /// seq, the new entry survives so the next round picks it up
    /// (the snapshot's bytes don't include that writer's mutation,
    /// so we mustn't claim the blob is clean).
    ///
    /// On failure both the unflushed-snapshot fact and any racing
    /// writer's entry survive in the dirty map — restoration is
    /// the caller's responsibility (see `round::run_round`).
    pub(crate) fn write_through(
        &self,
        guid: BlobGuid,
        bytes: &AlignedBlobBuf,
        expected_seq: u64,
    ) -> Result<()> {
        self.backend.write_blob(guid, bytes)?;
        let mut d = self.dirty.lock().unwrap();
        if let std::collections::hash_map::Entry::Occupied(e) = d.entry(guid) {
            // Only retire the entry when no racing writer has
            // bumped it. `mark_dirty` keeps the **minimum**
            // unflushed seq, so a survivor here has a seq newer
            // than ours iff a racer landed after we drained.
            if *e.get() == expected_seq {
                e.remove();
            }
        }
        Ok(())
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
        self.cache.insert(guid, entry);
        let mut d = self.dirty.lock().unwrap();
        d.entry(guid)
            .and_modify(|cur| *cur = (*cur).min(seq))
            .or_insert(seq);
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
        self.dirty.lock().unwrap().remove(&guid);
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

    // Note on pending-delete + cache: `mark_for_delete` already
    // removes the cache image (`self.cache.remove(&guid)`) in the
    // same call as it queues the pending-delete, so under the
    // engine's current invariant set a blob is never both cached
    // and in `pending_deletes` simultaneously. `try_evict_lru`'s
    // pending-delete check is kept as defense in depth — cheap
    // (one lock + contains_key per scan) and documents the
    // invariant for future readers — but isn't exercised by a
    // test today.

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
        assert_eq!(bm.min_unflushed_txn(), Some(30));
        assert_eq!(bm.dirty_count(), 1);
    }

    #[test]
    fn min_unflushed_txn_returns_none_when_clean() {
        let bm = BufferManager::new(Arc::new(MemoryBackend::new()), 4);
        assert_eq!(bm.min_unflushed_txn(), None);
        assert_eq!(bm.dirty_count(), 0);
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
        assert_eq!(bm.min_unflushed_txn(), None);

        // Concurrent mark_dirty lands in the fresh empty map.
        bm.mark_dirty([0x03; 16], 99);
        assert_eq!(bm.dirty_count(), 1);
        assert_eq!(bm.min_unflushed_txn(), Some(99));
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
    fn commit_clears_dirty_on_success() {
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        inner.write_blob([0x77; 16], &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner, 4);

        // Pin + write-guard to populate cache + mark dirty.
        let pin = bm.pin([0x77; 16]).unwrap();
        {
            let mut g = pin.write();
            g.as_mut_slice()[200] = 0xCD;
        }
        bm.mark_dirty([0x77; 16], 42);
        assert_eq!(bm.dirty_count(), 1);

        bm.commit([0x77; 16]).unwrap();
        assert_eq!(bm.dirty_count(), 0, "successful commit must clear dirty");
    }

    #[test]
    fn write_through_keeps_racing_writer_dirty_entry() {
        // Reproduces the dirty-race fix: a checkpointer drains the
        // dirty map at snapshot time (snap_seq=50), then before
        // `write_through` runs an in-process writer marks the
        // same blob dirty with a newer seq (200). The writer's
        // mutation is NOT in our snapshot bytes, so the entry
        // must survive `write_through`'s clear.
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
        bm.write_through([0xAA; 16], &snap_bytes, 50).unwrap();
        assert_eq!(
            bm.dirty_count(),
            1,
            "write_through must not stomp a racing newer-seq entry",
        );
        let live = bm.snapshot_dirty();
        assert_eq!(live[&[0xAA; 16]], 200, "racing writer's seq survives");
    }

    #[test]
    fn write_through_retires_clean_snapshot() {
        // Counterpart to the race test: when the dirty entry
        // still matches the snapshot's seq (no racing writer),
        // `write_through` does retire it.
        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        inner.write_blob([0xBB; 16], &make_buf(0)).unwrap();
        let bm = BufferManager::new(inner, 4);
        let _pin = bm.pin([0xBB; 16]).unwrap();

        bm.mark_dirty([0xBB; 16], 42);
        let snap_bytes = bm.snapshot_bytes([0xBB; 16]).unwrap();

        // expected_seq matches the current entry → safe to retire.
        bm.write_through([0xBB; 16], &snap_bytes, 42).unwrap();
        assert_eq!(bm.dirty_count(), 0);
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
        assert_eq!(bm.min_unflushed_txn(), Some(42));

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

        // After commit, the inner backend has the bytes and the
        // dirty entry is cleared.
        bm.commit(new_guid).unwrap();
        assert_eq!(bm.dirty_count(), 0);
        assert!(inner.has_blob(new_guid).unwrap());
        let mut dst = AlignedBlobBuf::zeroed();
        inner.read_blob(new_guid, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[200], 0x77);
    }

    /// Phase 1 of the per-node-latch milestone: writes routed
    /// through a versioned `BlobFrame` (via
    /// `BlobWriteGuard::frame()`) must bump the owning
    /// `CachedBlob`'s per-slot version counter so observers can
    /// detect the mutation across latch releases.
    ///
    /// This test isolates the contract end-to-end without going
    /// through the walker: install a fresh blob, take a write
    /// guard, do one `alloc_node`, release the guard, then check
    /// that `CachedBlob::slot_version(newly_allocated_slot)`
    /// changed from 0 to something > 0.
    #[test]
    fn slot_version_bumps_on_versioned_frame_mutation() {
        use crate::layout::{BlobGuid, NodeType, PAGE_SIZE};
        use crate::store::BlobFrame;

        let inner: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        let bm = BufferManager::new(Arc::clone(&inner), 4);

        // Format a fresh blob buffer (init seeds the EmptyRoot
        // sentinel at slot 1).
        let guid: BlobGuid = [0xCC; 16];
        let mut bytes = AlignedBlobBuf::zeroed();
        {
            let _ = BlobFrame::init(bytes.as_mut_slice(), guid).unwrap();
        }
        bm.install_new_blob(guid, bytes, /*seq=*/ 1);

        // Pin + write guard.
        let pin = bm.pin(guid).unwrap();

        // Before the alloc, the to-be-allocated slot's version is
        // 0 (no mutation yet). After `init` ran above on a raw
        // `&mut [u8]` (not a versioned frame), slot 1's counter
        // is also 0 — the EmptyRoot sentinel's bump went to the
        // None branch.
        assert_eq!(
            pin.slot_version(2),
            0,
            "slot 2 should be unallocated → version 0"
        );

        // Run one alloc through a versioned frame (= what the
        // walker does once `BlobWriteGuard::frame()` is on the
        // hot path).
        let allocated_slot = {
            let mut guard = pin.write();
            let mut frame = guard.frame();
            frame.alloc_node(NodeType::Node4).unwrap().slot
        };

        assert_eq!(
            allocated_slot, 2,
            "first alloc after init should land at slot 2"
        );

        // The bump landed on slot 2's counter.
        assert!(
            pin.slot_version(2) > 0,
            "alloc_node through versioned frame should bump slot version"
        );

        // Untouched slots stay at 0.
        assert_eq!(
            pin.slot_version(3),
            0,
            "neighbour slots must not have spurious bumps"
        );

        // Sanity: out-of-range / zero slot indices stay at 0.
        assert_eq!(pin.slot_version(0), 0);
        assert_eq!(pin.slot_version(u16::MAX), 0);

        // Ensure the install_new_blob seeding stays clean for the
        // rest of the test suite.
        drop(pin);
        bm.commit(guid).unwrap();
        assert!(inner.has_blob(guid).unwrap());
        // The freshly-formatted blob is one PAGE_SIZE of bytes
        // — just confirm we didn't accidentally truncate.
        assert_eq!(PAGE_SIZE as usize, 524_288);
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
