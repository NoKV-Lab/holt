//! DB, tree, and per-blob counter snapshots returned by public
//! stats APIs.
//!
//! These mirror the `BlobHeader`
//! counter fields and are read in a single shared-guard pass
//! per blob, so a snapshot of any one blob is internally
//! consistent but the cross-blob aggregate is not linearised
//! against concurrent writers.

use crate::layout::BlobGuid;
use crate::layout::{DATA_AREA_START, PAGE_SIZE};

/// Per-blob counters captured by [`Tree::stats`](crate::Tree::stats).
///
/// Each field mirrors a `BlobHeader`
/// counter and is read in a single shared-guard pass over the blob.
#[derive(Debug, Clone, Copy)]
pub struct BlobStats {
    /// GUID identifying this blob within the tree.
    pub guid: BlobGuid,
    /// Bytes currently consumed in the blob's data area (bump
    /// cursor, monotonically advancing).
    pub space_used: u32,
    /// Bytes lost to bump-allocator waste (extents freed without
    /// recycling). Compaction reclaims this to zero.
    pub gap_space: u32,
    /// High-water slot count — slot indices `[1, num_slots)` have
    /// ever held a node body in this blob.
    pub num_slots: u16,
    /// Count of cross-blob crossings (`BlobNode` slots) currently
    /// installed in this blob.
    pub num_ext_blobs: u16,
    /// Number of times this blob has been rebuilt by the in-place
    /// compactor. Bumped at the end of every successful compaction.
    pub compact_times: u32,
    /// Count of leaves in this blob currently in tombstone state
    /// (soft-deleted, awaiting reclaim by compaction).
    pub tombstone_leaf_cnt: u32,
}

/// Prefix-route-cache counters captured by [`Tree::stats`](crate::Tree::stats).
///
/// The route cache is a small parent-validated BlobNode crossing
/// accelerator for path-shaped large trees. These counters diagnose
/// whether point reads and writes are using stable cached anchors or
/// churning through stale routes.
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteCacheStats {
    /// Number of cached route entries currently resident.
    pub entries: usize,
    /// Cumulative successful route-cache lookups.
    pub hits: u64,
    /// Cumulative route-cache lookup misses for keys not covered by
    /// any cached prefix.
    pub misses: u64,
    /// Cumulative learned routes. Updating an existing cached prefix
    /// counts as a learn because the route was refreshed from a live
    /// parent descent.
    pub learns: u64,
    /// Cumulative capacity replacements after the route cache filled.
    pub evictions: u64,
    /// Cumulative stale route probes caused by parent-version
    /// changes. Stale entries are removed and can be relearned by a
    /// later descent.
    pub invalidations: u64,
}

/// Tree-wide aggregate counters from [`Tree::stats`](crate::Tree::stats).
///
/// `blobs` carries the per-blob breakdown in BFS order from the
/// root; the totals are pre-summed for the common "how big is the
/// tree?" question. All bytes / counts are sums over `blobs`.
#[derive(Debug, Clone)]
pub struct TreeStats {
    /// Number of distinct blobs reachable from the tree root.
    pub blob_count: u32,
    /// Sum of `space_used` over every blob.
    pub total_space_used: u64,
    /// Sum of `gap_space` over every blob.
    pub total_gap_space: u64,
    /// Sum of `num_slots` over every blob.
    pub total_slots: u64,
    /// Sum of `compact_times` over every blob (lifetime
    /// compactions across the whole tree).
    pub total_compactions: u64,
    /// Sum of `tombstone_leaf_cnt` over every blob.
    pub total_tombstones: u64,
    /// Sum of `num_ext_blobs` over every blob. In a tree-shaped
    /// blob graph this is the number of cross-blob edges.
    pub total_blob_edges: u64,
    /// Number of reachable blobs with no installed `BlobNode`
    /// children.
    pub leaf_blob_count: u32,
    /// Maximum cross-blob depth from the root blob. Root-only
    /// trees report `0`.
    pub max_blob_depth: u32,
    /// Sum of per-blob depths, used to derive
    /// [`Self::avg_blob_depth`].
    pub total_blob_depth: u64,
    /// Highest data-area occupancy among reachable blobs, in
    /// per-mille units (`1000` = full data area).
    pub max_blob_fill_per_mille: u32,
    /// Non-root blobs below the shape-control fill band.
    pub underfilled_child_blobs: u32,
    /// Non-root blobs above the shape-control fill band.
    pub overfull_child_blobs: u32,
    /// Per-blob breakdown in BFS order from the root.
    pub blobs: Vec<BlobStats>,
    /// Number of blobs currently dirty in the BufferManager —
    /// modified in cache but not yet flushed to store. With the
    /// background checkpointer enabled this stays bounded by the
    /// checkpoint cadence; without it, it tracks the user's
    /// explicit `Tree::checkpoint` schedule.
    pub bm_dirty_count: usize,
    /// Number of blobs fenced for deferred store deletion. Includes
    /// queued deletes and deletes already claimed by a checkpoint
    /// epoch but not yet completed.
    pub bm_pending_delete_count: usize,
    /// Deferred blind writes waiting to be merged into ART blob
    /// frames. WAL truncation waits for this to reach zero.
    pub bm_write_delta_count: usize,
    /// Cumulative cache lookups served from BM cache without
    /// going to the inner store. Read by external observers to
    /// derive a hit rate (`bm_cache_hits / (bm_cache_hits +
    /// bm_cache_misses)`); higher is better.
    pub bm_cache_hits: u64,
    /// Cumulative cache lookups that fell through to
    /// `inner_store.read_blob` because the entry was absent or
    /// evicted. Tracks cold-start + eviction churn.
    pub bm_cache_misses: u64,
    /// Successful full-frame reads from the blob store after a BM
    /// miss. A read here means Holt pulled one complete blob frame
    /// from the backing store.
    pub bm_full_blob_reads: u64,
    /// Bytes read by successful full-frame blob-store reads.
    pub bm_full_blob_read_bytes: u64,
    /// Full-frame reads caused by point get/put paths.
    pub bm_point_full_blob_reads: u64,
    /// Full-frame reads caused by range/list scan paths.
    pub bm_scan_full_blob_reads: u64,
    /// Full-frame reads caused by silent stats/maintenance paths.
    pub bm_silent_full_blob_reads: u64,
    /// 4 KiB cold-page cache hits served without a blob-frame read.
    pub bm_cold_page_hits: u64,
    /// 4 KiB cold-page cache misses that fell through to positional
    /// reads or full-frame fallback.
    pub bm_cold_page_misses: u64,
    /// Cold-index directory lookups served from the in-memory index
    /// cache.
    pub bm_cold_index_cache_hits: u64,
    /// Cold-index directory lookups that had to read the sidecar.
    pub bm_cold_index_cache_misses: u64,
    /// Cold-index directories loaded from the sidecar.
    pub bm_cold_index_loads: u64,
    /// Bytes read while loading cold-index directories.
    pub bm_cold_index_dir_read_bytes: u64,
    /// Cold-index bucket reads issued after a positive filter probe.
    pub bm_cold_index_bucket_reads: u64,
    /// Bytes read by cold-index bucket reads.
    pub bm_cold_index_bucket_read_bytes: u64,
    /// Positive cold reads served directly from inline sidecar values.
    pub bm_cold_index_inline_hits: u64,
    /// Positive cold reads that used a sidecar offset plus a leaf-page
    /// validation read.
    pub bm_cold_index_offset_hits: u64,
    /// Negative cold reads proven by a cold-index filter/summary.
    pub bm_cold_index_negative_hits: u64,
    /// Cold reads routed to a child blob by a cold-index crossing.
    pub bm_cold_index_crossing_hits: u64,
    /// Cold-index probes that could not publish an answer and fell
    /// back to the authoritative blob path.
    pub bm_cold_index_unknowns: u64,
    /// Cumulative wait-free read restarts in `Tree::get` — each
    /// one means a concurrent writer lapped an optimistic
    /// snapshot and the lookup walked the tree from scratch.
    /// Spikes here indicate writer/reader contention.
    pub bm_optimistic_restarts: u64,
    /// Cumulative range-iterator cursor restarts. Each one means a
    /// range/list cursor detected a rewritten blob on its descent
    /// path and rebuilt from its monotonic lower bound.
    pub bm_range_restarts: u64,
    /// Cumulative mutation walker invocations (`insert_multi` /
    /// `erase_multi`). `rename` and `atomic` count their inner walker
    /// calls separately.
    pub bm_walker_ops: u64,
    /// Total blob hops across mutation walker invocations. Divide
    /// by [`Self::bm_walker_ops`] for the average.
    pub bm_walker_blob_hops: u64,
    /// Maximum blob hops observed for one mutation walker call.
    pub bm_max_blob_hops: u64,
    /// Largest key-depth at which a mutation walker entered a blob.
    /// This is a cross-blob boundary-depth signal, not a full
    /// per-node ART-depth trace.
    pub bm_max_cross_blob_depth: u64,
    /// Successful foreground spillover events.
    pub bm_spillovers: u64,
    /// `BlobNode` children folded back into parents by manual
    /// compact or background merge passes.
    pub bm_merges: u64,
    /// Number of route-anchor blobs protected from ordinary clean
    /// cache eviction. These are upper-tree blobs learned by the
    /// route cache so cold leaf misses do not evict the routing
    /// layer of large path-shaped trees.
    pub bm_route_resident_count: usize,
    /// Route-anchor entries demoted from the protected tier after
    /// its memory budget filled. This is a tier demotion counter,
    /// not a cache eviction counter.
    pub bm_route_resident_demotions: u64,
    /// Clean cache entries evicted by inline overflow or the
    /// background eviction sweep.
    pub bm_cache_evictions: u64,
    /// Eviction candidates skipped because dirty / flushing /
    /// delete-fence bookkeeping protected them.
    pub bm_eviction_skips_protected: u64,
    /// Eviction candidates skipped because they are route-resident
    /// anchor blobs.
    pub bm_eviction_skips_route_resident: u64,
    /// Cache overflows where TinyLFU kept a hotter resident victim
    /// instead of evicting it for a one-hit point-miss candidate.
    pub bm_admission_protects: u64,
    /// Root route-cache counters for large path-shaped trees.
    pub route_cache: RouteCacheStats,
    /// WAL replay telemetry captured during `Tree::open`.
    pub open: OpenStats,
    /// WAL/journal worker counters, or `None` for memory trees and
    /// caller-supplied stores opened without holt's WAL.
    pub journal: Option<JournalStats>,
    /// Background checkpointer telemetry, or `None` if the caller
    /// explicitly disabled the thread group.
    pub checkpointer: Option<CheckpointerStats>,
}

impl TreeStats {
    /// Average blob hops per mutation walker invocation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // observability gauge; exact integer totals are exposed too
    pub fn bm_avg_blob_hops(&self) -> f64 {
        if self.bm_walker_ops == 0 {
            0.0
        } else {
            self.bm_walker_blob_hops as f64 / self.bm_walker_ops as f64
        }
    }

    /// Average cross-blob graph depth across reachable blobs.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_blob_depth(&self) -> f64 {
        if self.blob_count == 0 {
            0.0
        } else {
            self.total_blob_depth as f64 / f64::from(self.blob_count)
        }
    }

    /// Fraction of reachable blobs that are leaves in the blob graph.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn leaf_blob_ratio(&self) -> f64 {
        if self.blob_count == 0 {
            0.0
        } else {
            f64::from(self.leaf_blob_count) / f64::from(self.blob_count)
        }
    }

    /// Average data-area occupancy across reachable blobs.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_blob_fill_ratio(&self) -> f64 {
        if self.blob_count == 0 {
            return 0.0;
        }
        let header_bytes = u64::from(self.blob_count) * DATA_AREA_START as u64;
        let data_used = self.total_space_used.saturating_sub(header_bytes);
        let data_capacity = u64::from(self.blob_count) * (PAGE_SIZE - DATA_AREA_START) as u64;
        if data_capacity == 0 {
            0.0
        } else {
            data_used as f64 / data_capacity as f64
        }
    }

    /// Maximum data-area occupancy among reachable blobs.
    #[must_use]
    pub fn max_blob_fill_ratio(&self) -> f64 {
        f64::from(self.max_blob_fill_per_mille) / 1000.0
    }
}

/// DB-wide resource counters from [`DB::stats`](crate::DB::stats).
///
/// These counters describe shared resources owned by a multi-tree
/// `DB`, not any one ART root. Per-tree shape counters still live in
/// [`TreeStats`] because shape is root-specific.
#[derive(Debug, Clone)]
pub struct DBStats {
    /// Number of named trees opened by this process.
    pub open_tree_count: usize,
    /// Number of dirty blobs across every tree in the shared
    /// BufferManager.
    pub bm_dirty_count: usize,
    /// Number of deferred deletes across every tree.
    pub bm_pending_delete_count: usize,
    /// Deferred blind writes waiting to be merged into ART blob
    /// frames. WAL truncation waits for this to reach zero.
    pub bm_write_delta_count: usize,
    /// Shared BufferManager cache hits.
    pub bm_cache_hits: u64,
    /// Shared BufferManager cache misses.
    pub bm_cache_misses: u64,
    /// Shared successful full-frame reads from the blob store.
    pub bm_full_blob_reads: u64,
    /// Shared bytes read by successful full-frame blob-store reads.
    pub bm_full_blob_read_bytes: u64,
    /// Shared full-frame reads caused by point get/put paths.
    pub bm_point_full_blob_reads: u64,
    /// Shared full-frame reads caused by range/list scan paths.
    pub bm_scan_full_blob_reads: u64,
    /// Shared full-frame reads caused by silent stats/maintenance paths.
    pub bm_silent_full_blob_reads: u64,
    /// Shared 4 KiB cold-page cache hits.
    pub bm_cold_page_hits: u64,
    /// Shared 4 KiB cold-page cache misses.
    pub bm_cold_page_misses: u64,
    /// Shared cold-index directory cache hits.
    pub bm_cold_index_cache_hits: u64,
    /// Shared cold-index directory cache misses.
    pub bm_cold_index_cache_misses: u64,
    /// Shared cold-index directory loads.
    pub bm_cold_index_loads: u64,
    /// Shared bytes read while loading cold-index directories.
    pub bm_cold_index_dir_read_bytes: u64,
    /// Shared cold-index bucket reads.
    pub bm_cold_index_bucket_reads: u64,
    /// Shared bytes read by cold-index bucket reads.
    pub bm_cold_index_bucket_read_bytes: u64,
    /// Shared inline cold-index hits.
    pub bm_cold_index_inline_hits: u64,
    /// Shared offset cold-index hits.
    pub bm_cold_index_offset_hits: u64,
    /// Shared negative cold-index hits.
    pub bm_cold_index_negative_hits: u64,
    /// Shared cold-index crossing hits.
    pub bm_cold_index_crossing_hits: u64,
    /// Shared cold-index fallbacks.
    pub bm_cold_index_unknowns: u64,
    /// Shared optimistic point-read restarts.
    pub bm_optimistic_restarts: u64,
    /// Shared range cursor restarts.
    pub bm_range_restarts: u64,
    /// Shared mutation walker invocations.
    pub bm_walker_ops: u64,
    /// Shared mutation walker blob hops.
    pub bm_walker_blob_hops: u64,
    /// Maximum blob hops observed by a shared mutation walker.
    pub bm_max_blob_hops: u64,
    /// Maximum cross-blob key depth observed by a shared walker.
    pub bm_max_cross_blob_depth: u64,
    /// Shared foreground spillover count.
    pub bm_spillovers: u64,
    /// Shared merge count.
    pub bm_merges: u64,
    /// Route-resident blobs protected in the shared cache.
    pub bm_route_resident_count: usize,
    /// Route-resident demotions in the shared cache.
    pub bm_route_resident_demotions: u64,
    /// Shared clean cache evictions.
    pub bm_cache_evictions: u64,
    /// Eviction candidates skipped because they were protected.
    pub bm_eviction_skips_protected: u64,
    /// Eviction candidates skipped because they were route anchors.
    pub bm_eviction_skips_route_resident: u64,
    /// TinyLFU admission rejections in the shared cache.
    pub bm_admission_protects: u64,
    /// WAL replay telemetry captured while opening the DB.
    pub open: OpenStats,
    /// Shared WAL/journal counters, or `None` for memory DBs.
    pub journal: Option<JournalStats>,
    /// Shared background checkpointer counters, or `None` when
    /// checkpointing is disabled.
    pub checkpointer: Option<CheckpointerStats>,
}

impl DBStats {
    /// Average blob hops per mutation walker invocation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bm_avg_blob_hops(&self) -> f64 {
        if self.bm_walker_ops == 0 {
            0.0
        } else {
            self.bm_walker_blob_hops as f64 / self.bm_walker_ops as f64
        }
    }
}

/// Snapshot of WAL append and sync counters.
#[derive(Debug, Clone, Copy)]
pub struct JournalStats {
    /// Number of WAL append requests submitted by foreground
    /// mutation paths.
    pub appends: u64,
    /// Number of WAL write groups emitted by the journal worker.
    pub batches: u64,
    /// Number of `sync_data` calls issued by WAL flush paths,
    /// including explicit checkpoint barriers.
    pub syncs: u64,
    /// Highest WAL work id accepted by foreground append paths.
    pub queued_work: u64,
    /// Highest WAL work id written by the journal worker.
    pub written_work: u64,
    /// Highest WAL work id known to have passed the WAL durability
    /// boundary.
    pub flushed_work: u64,
    /// Highest WAL work id made redundant by a completed checkpoint
    /// truncate.
    pub checkpointed_work: u64,
    /// WAL work accepted by foreground writers but not yet known
    /// durable.
    pub pending_work: u64,
    /// WAL work not yet made redundant by checkpoint.
    pub checkpoint_debt: u64,
}

/// Reopen-time recovery telemetry captured while opening a tree.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenStats {
    /// WAL records scanned during reopen.
    pub wal_replay_records: u64,
    /// Bytes scanned from the WAL during reopen.
    pub wal_replay_bytes: u64,
    /// Wall-clock time spent replaying the WAL, in microseconds.
    pub wal_replay_micros: u64,
    /// Non-zero iff reopen stopped at a torn WAL tail.
    pub wal_torn_tail: bool,
}

/// Snapshot of the background checkpointer's accumulated
/// counters. Returned inside [`TreeStats::checkpointer`] when the
/// thread group is enabled. All counters are cumulative since
/// the threads were spawned.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointerStats {
    /// Rounds the planner has started — succeeded + failed
    /// combined. Empty rounds (no dirty entries, nothing to
    /// merge) still increment.
    pub rounds_attempted: u64,
    /// Rounds that completed without an error path. A successful
    /// round can still have done no flush work — see
    /// [`Self::blobs_flushed`] for actual durability progress.
    pub rounds_succeeded: u64,
    /// Rounds or submitted epochs that failed and restored their
    /// dirty / pending-delete state for retry.
    pub rounds_failed: u64,
    /// Cumulative number of blob commits the I/O thread has
    /// processed across all rounds.
    pub blobs_flushed: u64,
    /// Total `BlobNode` crossings folded back into their parents
    /// across every merge pass.
    pub merges_total: u64,
    /// WAL `truncate` calls — once per round where the planner
    /// observed `dirty_count == 0` under the commit gate after
    /// flushing.
    pub truncates: u64,
    /// Cache entries the eviction thread has dropped because
    /// they were cold + non-dirty + held only by the cache.
    pub evictions: u64,
    /// Dirty blobs observed by the most recent planner round.
    pub last_dirty_count: usize,
    /// Pending deletes observed by the most recent planner round.
    pub last_pending_delete_count: usize,
    /// Wall-clock time spent in the most recent planner round, in
    /// microseconds. I/O completion may happen later in the I/O
    /// thread.
    pub last_round_micros: u64,
}
