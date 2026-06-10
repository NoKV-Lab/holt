//! Prometheus text-format renderer for [`TreeStats`](crate::TreeStats).
//!
//! Enabled via the `metrics` feature flag. Pure Rust — no
//! `prometheus` / `metrics` crate dependency. The caller hosts
//! its own HTTP `/metrics` endpoint (axum, hyper, warp, …) and
//! invokes [`render_prometheus`](crate::metrics::render_prometheus) inside the handler.
//!
//! ```ignore
//! use holt::{Tree, TreeConfig};
//! use holt::metrics::render_prometheus;
//!
//! let tree = Tree::open(TreeConfig::memory()).unwrap();
//! let stats = tree.stats().unwrap();
//! let body = render_prometheus(&stats);
//! println!("{body}");
//! ```
//!
//! ## Metrics emitted
//!
//! All metric names are prefixed with `holt_` and follow the
//! [Prometheus naming convention][1]:
//!
//! - **`_total` suffix**: monotonically-non-decreasing counters,
//!   never reset across a process's lifetime.
//! - **`_bytes` suffix**: gauges measured in bytes.
//! - **No suffix**: arbitrary gauges (instantaneous values that
//!   can go up or down).
//!
//! [1]: https://prometheus.io/docs/practices/naming/
//!
//! | Metric                                  | Type    | Source field |
//! | --------------------------------------- | ------- | ------------ |
//! | `holt_blob_count`                       | gauge   | `TreeStats::blob_count`                |
//! | `holt_space_used_bytes`                 | gauge   | `TreeStats::total_space_used`          |
//! | `holt_gap_space_bytes`                  | gauge   | `TreeStats::total_gap_space`           |
//! | `holt_slots`                            | gauge   | `TreeStats::total_slots`               |
//! | `holt_compactions`                      | gauge   | `TreeStats::total_compactions` (sum of `compact_times` across **currently reachable** blobs — can go DOWN if a blob is merged/deleted; it's a tree-shape metric, not a lifetime counter) |
//! | `holt_tombstones`                       | gauge   | `TreeStats::total_tombstones`          |
//! | `holt_blob_edges`                       | gauge   | `TreeStats::total_blob_edges`          |
//! | `holt_leaf_blob_count`                  | gauge   | `TreeStats::leaf_blob_count`           |
//! | `holt_blob_leaf_ratio`                  | gauge   | `TreeStats::leaf_blob_ratio()`         |
//! | `holt_blob_max_depth`                   | gauge   | `TreeStats::max_blob_depth`            |
//! | `holt_blob_avg_depth`                   | gauge   | `TreeStats::avg_blob_depth()`          |
//! | `holt_blob_avg_fill_ratio`              | gauge   | `TreeStats::avg_blob_fill_ratio()`     |
//! | `holt_blob_max_fill_ratio`              | gauge   | `TreeStats::max_blob_fill_ratio()`     |
//! | `holt_blob_underfilled_children`         | gauge   | `TreeStats::underfilled_child_blobs`    |
//! | `holt_blob_overfull_children`            | gauge   | `TreeStats::overfull_child_blobs`       |
//! | `holt_bm_dirty_count`                   | gauge   | `TreeStats::bm_dirty_count`            |
//! | `holt_bm_pending_delete_count`          | gauge   | `TreeStats::bm_pending_delete_count`   |
//! | `holt_bm_cache_hits_total`              | counter | `TreeStats::bm_cache_hits`             |
//! | `holt_bm_cache_misses_total`            | counter | `TreeStats::bm_cache_misses`           |
//! | `holt_bm_full_blob_reads_total`         | counter | `TreeStats::bm_full_blob_reads`        |
//! | `holt_bm_full_blob_read_bytes_total`    | counter | `TreeStats::bm_full_blob_read_bytes`   |
//! | `holt_bm_point_full_blob_reads_total`   | counter | `TreeStats::bm_point_full_blob_reads`  |
//! | `holt_bm_scan_full_blob_reads_total`    | counter | `TreeStats::bm_scan_full_blob_reads`   |
//! | `holt_bm_silent_full_blob_reads_total`  | counter | `TreeStats::bm_silent_full_blob_reads` |
//! | `holt_bm_optimistic_restarts_total`     | counter | `TreeStats::bm_optimistic_restarts`    |
//! | `holt_bm_range_restarts_total`          | counter | `TreeStats::bm_range_restarts`         |
//! | `holt_bm_walker_ops_total`              | counter | `TreeStats::bm_walker_ops`             |
//! | `holt_bm_walker_blob_hops_total`        | counter | `TreeStats::bm_walker_blob_hops`       |
//! | `holt_bm_avg_blob_hops`                 | gauge   | `TreeStats::bm_avg_blob_hops()`        |
//! | `holt_bm_max_blob_hops`                 | gauge   | `TreeStats::bm_max_blob_hops`          |
//! | `holt_bm_max_cross_blob_depth`          | gauge   | `TreeStats::bm_max_cross_blob_depth`   |
//! | `holt_bm_spillovers_total`              | counter | `TreeStats::bm_spillovers`             |
//! | `holt_bm_merges_total`                  | counter | `TreeStats::bm_merges`                 |
//! | `holt_route_cache_entries`              | gauge   | `TreeStats::route_cache.entries`       |
//! | `holt_route_cache_hits_total`           | counter | `TreeStats::route_cache.hits`          |
//! | `holt_route_cache_misses_total`         | counter | `TreeStats::route_cache.misses`        |
//! | `holt_route_cache_learns_total`         | counter | `TreeStats::route_cache.learns`        |
//! | `holt_route_cache_evictions_total`      | counter | `TreeStats::route_cache.evictions`     |
//! | `holt_route_cache_invalidations_total`  | counter | `TreeStats::route_cache.invalidations` |
//! | `holt_bm_route_resident_count`          | gauge   | `TreeStats::bm_route_resident_count`   |
//! | `holt_bm_route_resident_demotions_total`| counter | `TreeStats::bm_route_resident_demotions` |
//! | `holt_bm_cache_evictions_total`         | counter | `TreeStats::bm_cache_evictions`        |
//! | `holt_bm_eviction_skips_protected_total`| counter | `TreeStats::bm_eviction_skips_protected` |
//! | `holt_bm_eviction_skips_route_resident_total` | counter | `TreeStats::bm_eviction_skips_route_resident` |
//! | `holt_bm_admission_protects_total`      | counter | `TreeStats::bm_admission_protects`     |
//! | `holt_open_wal_replay_records_total`    | counter | `OpenStats::wal_replay_records`        |
//! | `holt_open_wal_replay_bytes`            | gauge   | `OpenStats::wal_replay_bytes`          |
//! | `holt_open_wal_replay_duration_seconds` | gauge   | `OpenStats::wal_replay_micros`         |
//! | `holt_open_wal_torn_tail`               | gauge   | `OpenStats::wal_torn_tail`             |
//! | `holt_journal_appends_total`             | counter | `JournalStats::appends`                |
//! | `holt_journal_batches_total`             | counter | `JournalStats::batches`                |
//! | `holt_journal_syncs_total`               | counter | `JournalStats::syncs`                  |
//! | `holt_journal_queued_work`               | gauge   | `JournalStats::queued_work`            |
//! | `holt_journal_written_work`              | gauge   | `JournalStats::written_work`           |
//! | `holt_journal_flushed_work`              | gauge   | `JournalStats::flushed_work`           |
//! | `holt_journal_checkpointed_work`         | gauge   | `JournalStats::checkpointed_work`      |
//! | `holt_journal_pending_work`              | gauge   | `JournalStats::pending_work`           |
//! | `holt_journal_checkpoint_debt`           | gauge   | `JournalStats::checkpoint_debt`        |
//! | `holt_checkpoint_rounds_attempted_total`| counter | `CheckpointerStats::rounds_attempted`  |
//! | `holt_checkpoint_rounds_succeeded_total`| counter | `CheckpointerStats::rounds_succeeded`  |
//! | `holt_checkpoint_rounds_failed_total`   | counter | `CheckpointerStats::rounds_failed`     |
//! | `holt_checkpoint_blobs_flushed_total`   | counter | `CheckpointerStats::blobs_flushed`     |
//! | `holt_checkpoint_merges_total`          | counter | `CheckpointerStats::merges_total`      |
//! | `holt_checkpoint_truncates_total`       | counter | `CheckpointerStats::truncates`         |
//! | `holt_checkpoint_evictions_total`       | counter | `CheckpointerStats::evictions`         |
//! | `holt_checkpoint_last_dirty_count`      | gauge   | `CheckpointerStats::last_dirty_count`  |
//! | `holt_checkpoint_last_pending_delete_count` | gauge | `CheckpointerStats::last_pending_delete_count` |
//! | `holt_checkpoint_last_round_duration_seconds` | gauge | `CheckpointerStats::last_round_micros` |
//!
//! `JournalStats` and `CheckpointerStats` lines are emitted only
//! when the corresponding worker exists. The journal worker exists
//! for persistent trees opened through `Tree::open`; the background
//! checkpointer exists when `TreeStats::checkpointer` is `Some`.

use std::fmt::Write;

use crate::TreeStats;

/// Render `stats` as a Prometheus text-format payload.
///
/// The output is one HELP + TYPE + sample line per metric,
/// terminated by a `\n`. Suitable as the body of an HTTP 200
/// response with `Content-Type: text/plain; version=0.0.4`.
#[allow(clippy::too_many_lines)] // one `metric(...)` call per emit — splitting hides the export shape
#[must_use]
pub fn render_prometheus(stats: &TreeStats) -> String {
    // Pre-size for the typical payload (~3 KB) to avoid the
    // first few `String::push_str` reallocations.
    let mut out = String::with_capacity(3072);

    metric(
        &mut out,
        "holt_blob_count",
        "Number of distinct blobs reachable from the tree root.",
        "gauge",
        u64::from(stats.blob_count),
    );
    metric(
        &mut out,
        "holt_space_used_bytes",
        "Sum of `space_used` across every blob (live extent bytes).",
        "gauge",
        stats.total_space_used,
    );
    metric(
        &mut out,
        "holt_gap_space_bytes",
        "Sum of `gap_space` across every blob (reclaimable on compact).",
        "gauge",
        stats.total_gap_space,
    );
    metric(
        &mut out,
        "holt_slots",
        "Sum of `num_slots` across every reachable blob.",
        "gauge",
        stats.total_slots,
    );
    metric(
        &mut out,
        "holt_compactions",
        "Sum of `compact_times` across currently-reachable blobs. \
         A gauge, not a counter — a blob that merges into its \
         parent (or is deleted) takes its `compact_times` with \
         it, so this can go down.",
        "gauge",
        stats.total_compactions,
    );
    metric(
        &mut out,
        "holt_tombstones",
        "Sum of `tombstone_leaf_cnt` across every reachable blob.",
        "gauge",
        stats.total_tombstones,
    );
    metric(
        &mut out,
        "holt_blob_edges",
        "Number of cross-blob `BlobNode` edges in the reachable blob graph.",
        "gauge",
        stats.total_blob_edges,
    );
    metric(
        &mut out,
        "holt_leaf_blob_count",
        "Number of reachable blobs with no `BlobNode` children.",
        "gauge",
        u64::from(stats.leaf_blob_count),
    );
    metric_f64(
        &mut out,
        "holt_blob_leaf_ratio",
        "Fraction of reachable blobs that are leaves in the blob graph.",
        "gauge",
        stats.leaf_blob_ratio(),
    );
    metric(
        &mut out,
        "holt_blob_max_depth",
        "Maximum cross-blob depth from the root blob.",
        "gauge",
        u64::from(stats.max_blob_depth),
    );
    metric_f64(
        &mut out,
        "holt_blob_avg_depth",
        "Average cross-blob graph depth across reachable blobs.",
        "gauge",
        stats.avg_blob_depth(),
    );
    metric_f64(
        &mut out,
        "holt_blob_avg_fill_ratio",
        "Average data-area occupancy across reachable blobs.",
        "gauge",
        stats.avg_blob_fill_ratio(),
    );
    metric_f64(
        &mut out,
        "holt_blob_max_fill_ratio",
        "Maximum data-area occupancy among reachable blobs.",
        "gauge",
        stats.max_blob_fill_ratio(),
    );
    metric(
        &mut out,
        "holt_blob_underfilled_children",
        "Non-root blobs below the shape-control fill band.",
        "gauge",
        u64::from(stats.underfilled_child_blobs),
    );
    metric(
        &mut out,
        "holt_blob_overfull_children",
        "Non-root blobs above the shape-control fill band.",
        "gauge",
        u64::from(stats.overfull_child_blobs),
    );
    metric(
        &mut out,
        "holt_bm_dirty_count",
        "Number of blobs in the buffer manager dirty set.",
        "gauge",
        stats.bm_dirty_count as u64,
    );
    metric(
        &mut out,
        "holt_bm_pending_delete_count",
        "Number of blobs queued for deferred store deletion.",
        "gauge",
        stats.bm_pending_delete_count as u64,
    );
    metric(
        &mut out,
        "holt_bm_cache_hits_total",
        "Cumulative buffer-manager cache hits.",
        "counter",
        stats.bm_cache_hits,
    );
    metric(
        &mut out,
        "holt_bm_cache_misses_total",
        "Cumulative buffer-manager cache misses (fell through to store).",
        "counter",
        stats.bm_cache_misses,
    );
    metric(
        &mut out,
        "holt_bm_full_blob_reads_total",
        "Successful full-frame blob reads after buffer-manager misses.",
        "counter",
        stats.bm_full_blob_reads,
    );
    metric(
        &mut out,
        "holt_bm_full_blob_read_bytes_total",
        "Bytes read by successful full-frame blob reads.",
        "counter",
        stats.bm_full_blob_read_bytes,
    );
    metric(
        &mut out,
        "holt_bm_point_full_blob_reads_total",
        "Full-frame blob reads caused by point get/put paths.",
        "counter",
        stats.bm_point_full_blob_reads,
    );
    metric(
        &mut out,
        "holt_bm_scan_full_blob_reads_total",
        "Full-frame blob reads caused by range/list scan paths.",
        "counter",
        stats.bm_scan_full_blob_reads,
    );
    metric(
        &mut out,
        "holt_bm_silent_full_blob_reads_total",
        "Full-frame blob reads caused by silent stats/maintenance paths.",
        "counter",
        stats.bm_silent_full_blob_reads,
    );
    metric(
        &mut out,
        "holt_bm_optimistic_restarts_total",
        "Cumulative wait-free read restarts (concurrent writer lapped snapshot).",
        "counter",
        stats.bm_optimistic_restarts,
    );
    metric(
        &mut out,
        "holt_bm_range_restarts_total",
        "Cumulative range cursor restarts after versioned-path invalidation.",
        "counter",
        stats.bm_range_restarts,
    );
    metric(
        &mut out,
        "holt_bm_walker_ops_total",
        "Cumulative mutation walker invocations.",
        "counter",
        stats.bm_walker_ops,
    );
    metric(
        &mut out,
        "holt_bm_walker_blob_hops_total",
        "Total blob hops across mutation walkers.",
        "counter",
        stats.bm_walker_blob_hops,
    );
    metric_f64(
        &mut out,
        "holt_bm_avg_blob_hops",
        "Average blob hops per mutation walker invocation.",
        "gauge",
        stats.bm_avg_blob_hops(),
    );
    metric(
        &mut out,
        "holt_bm_max_blob_hops",
        "Maximum blob hops observed for one mutation walker call.",
        "gauge",
        stats.bm_max_blob_hops,
    );
    metric(
        &mut out,
        "holt_bm_max_cross_blob_depth",
        "Largest key-depth at which a mutation walker entered a blob.",
        "gauge",
        stats.bm_max_cross_blob_depth,
    );
    metric(
        &mut out,
        "holt_bm_spillovers_total",
        "Successful foreground spillover events.",
        "counter",
        stats.bm_spillovers,
    );
    metric(
        &mut out,
        "holt_bm_merges_total",
        "BlobNode children folded back into parents by compact or merge passes.",
        "counter",
        stats.bm_merges,
    );
    metric(
        &mut out,
        "holt_bm_route_resident_count",
        "Route-anchor blobs protected from ordinary LRU eviction.",
        "gauge",
        stats.bm_route_resident_count as u64,
    );
    metric(
        &mut out,
        "holt_bm_route_resident_demotions_total",
        "Route-anchor entries demoted after the protected tier filled.",
        "counter",
        stats.bm_route_resident_demotions,
    );
    metric(
        &mut out,
        "holt_bm_cache_evictions_total",
        "Clean cache entries evicted by inline overflow or background sweep.",
        "counter",
        stats.bm_cache_evictions,
    );
    metric(
        &mut out,
        "holt_bm_eviction_skips_protected_total",
        "Eviction candidates skipped because dirty or pending state protected them.",
        "counter",
        stats.bm_eviction_skips_protected,
    );
    metric(
        &mut out,
        "holt_bm_eviction_skips_route_resident_total",
        "Eviction candidates skipped because they are route-resident anchors.",
        "counter",
        stats.bm_eviction_skips_route_resident,
    );
    metric(
        &mut out,
        "holt_bm_admission_protects_total",
        "Cache overflows where TinyLFU protected a hotter resident victim.",
        "counter",
        stats.bm_admission_protects,
    );
    metric(
        &mut out,
        "holt_route_cache_entries",
        "Number of root route-cache entries currently resident.",
        "gauge",
        stats.route_cache.entries as u64,
    );
    metric(
        &mut out,
        "holt_route_cache_hits_total",
        "Cumulative successful root route-cache lookups.",
        "counter",
        stats.route_cache.hits,
    );
    metric(
        &mut out,
        "holt_route_cache_misses_total",
        "Cumulative root route-cache misses.",
        "counter",
        stats.route_cache.misses,
    );
    metric(
        &mut out,
        "holt_route_cache_learns_total",
        "Cumulative root route-cache learned routes.",
        "counter",
        stats.route_cache.learns,
    );
    metric(
        &mut out,
        "holt_route_cache_evictions_total",
        "Cumulative root route-cache capacity replacements.",
        "counter",
        stats.route_cache.evictions,
    );
    metric(
        &mut out,
        "holt_route_cache_invalidations_total",
        "Cumulative root route-cache stale probes after root-version changes.",
        "counter",
        stats.route_cache.invalidations,
    );

    metric(
        &mut out,
        "holt_open_wal_replay_records_total",
        "WAL records scanned during tree open.",
        "counter",
        stats.open.wal_replay_records,
    );
    metric(
        &mut out,
        "holt_open_wal_replay_bytes",
        "Bytes scanned from the WAL during tree open.",
        "gauge",
        stats.open.wal_replay_bytes,
    );
    metric_f64(
        &mut out,
        "holt_open_wal_replay_duration_seconds",
        "Wall-clock time spent replaying WAL during tree open.",
        "gauge",
        micros_to_seconds(stats.open.wal_replay_micros),
    );
    metric(
        &mut out,
        "holt_open_wal_torn_tail",
        "1 if tree open stopped at a torn WAL tail; otherwise 0.",
        "gauge",
        u64::from(stats.open.wal_torn_tail),
    );

    if let Some(journal) = &stats.journal {
        metric(
            &mut out,
            "holt_journal_appends_total",
            "WAL append requests submitted to the journal worker.",
            "counter",
            journal.appends,
        );
        metric(
            &mut out,
            "holt_journal_batches_total",
            "Append batches processed by the journal worker.",
            "counter",
            journal.batches,
        );
        metric(
            &mut out,
            "holt_journal_syncs_total",
            "WAL sync_data calls issued by the journal worker.",
            "counter",
            journal.syncs,
        );
        metric(
            &mut out,
            "holt_journal_queued_work",
            "Highest WAL work id accepted by foreground append paths.",
            "gauge",
            journal.queued_work,
        );
        metric(
            &mut out,
            "holt_journal_written_work",
            "Highest WAL work id written by the journal worker.",
            "gauge",
            journal.written_work,
        );
        metric(
            &mut out,
            "holt_journal_flushed_work",
            "Highest WAL work id known durable.",
            "gauge",
            journal.flushed_work,
        );
        metric(
            &mut out,
            "holt_journal_checkpointed_work",
            "Highest WAL work id made redundant by checkpoint.",
            "gauge",
            journal.checkpointed_work,
        );
        metric(
            &mut out,
            "holt_journal_pending_work",
            "WAL work accepted but not yet known durable.",
            "gauge",
            journal.pending_work,
        );
        metric(
            &mut out,
            "holt_journal_checkpoint_debt",
            "WAL work not yet made redundant by checkpoint.",
            "gauge",
            journal.checkpoint_debt,
        );
    }

    if let Some(ck) = &stats.checkpointer {
        metric(
            &mut out,
            "holt_checkpoint_rounds_attempted_total",
            "Background checkpoint rounds the planner started.",
            "counter",
            ck.rounds_attempted,
        );
        metric(
            &mut out,
            "holt_checkpoint_rounds_succeeded_total",
            "Background checkpoint rounds that completed without error.",
            "counter",
            ck.rounds_succeeded,
        );
        metric(
            &mut out,
            "holt_checkpoint_rounds_failed_total",
            "Checkpoint rounds or submitted epochs that failed and were restored.",
            "counter",
            ck.rounds_failed,
        );
        metric(
            &mut out,
            "holt_checkpoint_blobs_flushed_total",
            "Blobs the checkpointer's I/O worker wrote through to store.",
            "counter",
            ck.blobs_flushed,
        );
        metric(
            &mut out,
            "holt_checkpoint_merges_total",
            "Cross-blob `BlobNode` crossings folded back into a parent.",
            "counter",
            ck.merges_total,
        );
        metric(
            &mut out,
            "holt_checkpoint_truncates_total",
            "WAL truncations performed at the round's truncate gate.",
            "counter",
            ck.truncates,
        );
        metric(
            &mut out,
            "holt_checkpoint_evictions_total",
            "Cache entries the eviction thread dropped.",
            "counter",
            ck.evictions,
        );
        metric(
            &mut out,
            "holt_checkpoint_last_dirty_count",
            "Dirty blobs observed by the most recent planner round.",
            "gauge",
            ck.last_dirty_count as u64,
        );
        metric(
            &mut out,
            "holt_checkpoint_last_pending_delete_count",
            "Pending deletes observed by the most recent planner round.",
            "gauge",
            ck.last_pending_delete_count as u64,
        );
        metric_f64(
            &mut out,
            "holt_checkpoint_last_round_duration_seconds",
            "Wall-clock time spent in the most recent planner round.",
            "gauge",
            micros_to_seconds(ck.last_round_micros),
        );
    }
    out
}

#[allow(clippy::cast_precision_loss)]
fn micros_to_seconds(micros: u64) -> f64 {
    micros as f64 / 1_000_000.0
}

#[inline]
fn metric(out: &mut String, name: &str, help: &str, ty: &str, value: u64) {
    // `# HELP <name> <help>\n# TYPE <name> <ty>\n<name> <value>\n`
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {ty}");
    let _ = writeln!(out, "{name} {value}");
}

#[inline]
fn metric_f64(out: &mut String, name: &str, help: &str, ty: &str, value: f64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {ty}");
    let _ = writeln!(out, "{name} {value:.6}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CheckpointerStats, JournalStats, OpenStats, RouteCacheStats, TreeStats};

    fn stats_fixture(with_journal: bool, with_checkpointer: bool) -> TreeStats {
        TreeStats {
            blob_count: 3,
            total_space_used: 1024,
            total_gap_space: 256,
            total_slots: 42,
            total_compactions: 7,
            total_tombstones: 5,
            total_blob_edges: 2,
            leaf_blob_count: 2,
            max_blob_depth: 2,
            total_blob_depth: 3,
            max_blob_fill_per_mille: 750,
            underfilled_child_blobs: 1,
            overfull_child_blobs: 2,
            blobs: Vec::new(),
            bm_dirty_count: 2,
            bm_pending_delete_count: 1,
            bm_cache_hits: 1_000,
            bm_cache_misses: 25,
            bm_full_blob_reads: 20,
            bm_full_blob_read_bytes: 10_485_760,
            bm_point_full_blob_reads: 12,
            bm_scan_full_blob_reads: 7,
            bm_silent_full_blob_reads: 1,
            bm_optimistic_restarts: 3,
            bm_range_restarts: 2,
            bm_walker_ops: 4,
            bm_walker_blob_hops: 10,
            bm_max_blob_hops: 3,
            bm_max_cross_blob_depth: 17,
            bm_spillovers: 2,
            bm_merges: 1,
            bm_route_resident_count: 3,
            bm_route_resident_demotions: 4,
            bm_cache_evictions: 12,
            bm_eviction_skips_protected: 13,
            bm_eviction_skips_route_resident: 14,
            bm_admission_protects: 15,
            route_cache: RouteCacheStats {
                entries: 6,
                hits: 70,
                misses: 8,
                learns: 9,
                evictions: 2,
                invalidations: 1,
            },
            open: OpenStats {
                wal_replay_records: 21,
                wal_replay_bytes: 4096,
                wal_replay_micros: 12_500,
                wal_torn_tail: true,
            },
            journal: with_journal.then_some(JournalStats {
                appends: 20,
                batches: 5,
                syncs: 4,
                queued_work: 30,
                written_work: 29,
                flushed_work: 28,
                checkpointed_work: 24,
                pending_work: 2,
                checkpoint_debt: 6,
            }),
            checkpointer: with_checkpointer.then_some(CheckpointerStats {
                rounds_attempted: 11,
                rounds_succeeded: 10,
                rounds_failed: 1,
                blobs_flushed: 30,
                merges_total: 4,
                truncates: 8,
                evictions: 17,
                last_dirty_count: 18,
                last_pending_delete_count: 19,
                last_round_micros: 20_000,
            }),
        }
    }

    #[test]
    fn renders_core_metrics_for_stats_without_checkpointer() {
        let out = render_prometheus(&stats_fixture(false, false));
        assert!(out.contains("# HELP holt_blob_count "));
        assert!(out.contains("# TYPE holt_blob_count gauge\n"));
        assert!(out.contains("holt_blob_count 3\n"));
        // Monotonic counters keep the `_total` suffix...
        assert!(out.contains("holt_bm_cache_hits_total 1000\n"));
        assert!(out.contains("holt_bm_full_blob_reads_total 20\n"));
        assert!(out.contains("holt_bm_full_blob_read_bytes_total 10485760\n"));
        assert!(out.contains("holt_bm_point_full_blob_reads_total 12\n"));
        assert!(out.contains("holt_bm_scan_full_blob_reads_total 7\n"));
        assert!(out.contains("holt_bm_silent_full_blob_reads_total 1\n"));
        assert!(out.contains("holt_bm_optimistic_restarts_total 3\n"));
        assert!(out.contains("holt_bm_range_restarts_total 2\n"));
        assert!(out.contains("holt_bm_walker_ops_total 4\n"));
        assert!(out.contains("holt_bm_avg_blob_hops 2.500000\n"));
        assert!(out.contains("holt_bm_spillovers_total 2\n"));
        assert!(out.contains("holt_route_cache_entries 6\n"));
        assert!(out.contains("holt_route_cache_hits_total 70\n"));
        assert!(out.contains("holt_route_cache_misses_total 8\n"));
        assert!(out.contains("holt_route_cache_learns_total 9\n"));
        assert!(out.contains("holt_route_cache_evictions_total 2\n"));
        assert!(out.contains("holt_route_cache_invalidations_total 1\n"));
        assert!(out.contains("holt_bm_cache_evictions_total 12\n"));
        assert!(out.contains("holt_bm_eviction_skips_protected_total 13\n"));
        assert!(out.contains("holt_bm_eviction_skips_route_resident_total 14\n"));
        assert!(out.contains("holt_bm_admission_protects_total 15\n"));
        assert!(out.contains("holt_open_wal_replay_records_total 21\n"));
        assert!(out.contains("holt_open_wal_replay_bytes 4096\n"));
        assert!(out.contains("holt_open_wal_replay_duration_seconds 0.012500\n"));
        assert!(out.contains("holt_open_wal_torn_tail 1\n"));
        // ...non-monotonic gauges (sum-over-reachable-blobs) drop it.
        assert!(out.contains("# TYPE holt_slots gauge\n"));
        assert!(out.contains("holt_slots 42\n"));
        assert!(out.contains("# TYPE holt_compactions gauge\n"));
        assert!(out.contains("holt_compactions 7\n"));
        assert!(out.contains("# TYPE holt_tombstones gauge\n"));
        assert!(out.contains("holt_tombstones 5\n"));
        assert!(out.contains("holt_blob_edges 2\n"));
        assert!(out.contains("holt_leaf_blob_count 2\n"));
        assert!(out.contains("holt_blob_leaf_ratio 0.666667\n"));
        assert!(out.contains("holt_blob_max_depth 2\n"));
        assert!(out.contains("holt_blob_avg_depth 1.000000\n"));
        assert!(out.contains("holt_blob_max_fill_ratio 0.750000\n"));
        assert!(out.contains("holt_blob_underfilled_children 1\n"));
        assert!(out.contains("holt_blob_overfull_children 2\n"));
        // None of the gauges leak `_total`.
        assert!(!out.contains("holt_slots_total"));
        assert!(!out.contains("holt_compactions_total"));
        assert!(!out.contains("holt_tombstones_total"));
        // No checkpointer block.
        assert!(!out.contains("holt_checkpoint_"));
        assert!(!out.contains("holt_journal_"));
    }

    #[test]
    fn renders_journal_block_when_present() {
        let out = render_prometheus(&stats_fixture(true, false));
        assert!(out.contains("holt_journal_appends_total 20\n"));
        assert!(out.contains("holt_journal_batches_total 5\n"));
        assert!(out.contains("holt_journal_syncs_total 4\n"));
        assert!(out.contains("holt_journal_queued_work 30\n"));
        assert!(out.contains("holt_journal_written_work 29\n"));
        assert!(out.contains("holt_journal_flushed_work 28\n"));
        assert!(out.contains("holt_journal_checkpointed_work 24\n"));
        assert!(out.contains("holt_journal_pending_work 2\n"));
        assert!(out.contains("holt_journal_checkpoint_debt 6\n"));
    }

    #[test]
    fn renders_checkpoint_block_when_present() {
        let out = render_prometheus(&stats_fixture(false, true));
        assert!(out.contains("holt_checkpoint_rounds_attempted_total 11\n"));
        assert!(out.contains("holt_checkpoint_rounds_failed_total 1\n"));
        assert!(out.contains("holt_checkpoint_blobs_flushed_total 30\n"));
        assert!(out.contains("holt_checkpoint_evictions_total 17\n"));
        assert!(out.contains("holt_checkpoint_last_dirty_count 18\n"));
        assert!(out.contains("holt_checkpoint_last_pending_delete_count 19\n"));
        assert!(out.contains("holt_checkpoint_last_round_duration_seconds 0.020000\n"));
    }

    #[test]
    fn output_ends_with_newline() {
        let out = render_prometheus(&stats_fixture(false, false));
        assert!(out.ends_with('\n'), "Prometheus expects a trailing newline");
    }
}
