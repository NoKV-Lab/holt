use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub(super) struct Telemetry {
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    full_blob_reads: AtomicU64,
    point_full_blob_reads: AtomicU64,
    scan_full_blob_reads: AtomicU64,
    silent_full_blob_reads: AtomicU64,
    optimistic_restarts: AtomicU64,
    range_restarts: AtomicU64,
    walker_ops: AtomicU64,
    walker_blob_hops: AtomicU64,
    max_blob_hops: AtomicU64,
    max_cross_blob_depth: AtomicU64,
    spillover_count: AtomicU64,
    merge_count: AtomicU64,
    route_resident_demotions: AtomicU64,
    cache_evictions: AtomicU64,
    eviction_skips_protected: AtomicU64,
    eviction_skips_route_resident: AtomicU64,
    admission_protects: AtomicU64,
}

impl Telemetry {
    pub(super) fn note_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_point_full_blob_read(&self) {
        self.full_blob_reads.fetch_add(1, Ordering::Relaxed);
        self.point_full_blob_reads.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_scan_full_blob_read(&self) {
        self.full_blob_reads.fetch_add(1, Ordering::Relaxed);
        self.scan_full_blob_reads.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_silent_full_blob_read(&self) {
        self.full_blob_reads.fetch_add(1, Ordering::Relaxed);
        self.silent_full_blob_reads.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_optimistic_restart(&self) {
        self.optimistic_restarts.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_range_restart(&self) {
        self.range_restarts.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_walker_blob_hops(&self, hops: u64, max_cross_blob_depth: usize) {
        self.walker_ops.fetch_add(1, Ordering::Relaxed);
        self.walker_blob_hops.fetch_add(hops, Ordering::Relaxed);
        fetch_max_relaxed(&self.max_blob_hops, hops);
        fetch_max_relaxed(&self.max_cross_blob_depth, max_cross_blob_depth as u64);
    }

    pub(super) fn note_spillover(&self) {
        self.spillover_count.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_merges(&self, merged: u64) {
        if merged != 0 {
            self.merge_count.fetch_add(merged, Ordering::Relaxed);
        }
    }

    pub(super) fn note_route_resident_demotion(&self) {
        self.route_resident_demotions
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_cache_eviction(&self) {
        self.cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_eviction_skip_protected(&self) {
        self.eviction_skips_protected
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_eviction_skip_route_resident(&self) {
        self.eviction_skips_route_resident
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn note_admission_protect(&self) {
        self.admission_protects.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    pub(super) fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    pub(super) fn full_blob_reads(&self) -> u64 {
        self.full_blob_reads.load(Ordering::Relaxed)
    }

    pub(super) fn point_full_blob_reads(&self) -> u64 {
        self.point_full_blob_reads.load(Ordering::Relaxed)
    }

    pub(super) fn scan_full_blob_reads(&self) -> u64 {
        self.scan_full_blob_reads.load(Ordering::Relaxed)
    }

    pub(super) fn silent_full_blob_reads(&self) -> u64 {
        self.silent_full_blob_reads.load(Ordering::Relaxed)
    }

    pub(super) fn optimistic_restarts(&self) -> u64 {
        self.optimistic_restarts.load(Ordering::Relaxed)
    }

    pub(super) fn range_restarts(&self) -> u64 {
        self.range_restarts.load(Ordering::Relaxed)
    }

    pub(super) fn walker_ops(&self) -> u64 {
        self.walker_ops.load(Ordering::Relaxed)
    }

    pub(super) fn walker_blob_hops(&self) -> u64 {
        self.walker_blob_hops.load(Ordering::Relaxed)
    }

    pub(super) fn max_blob_hops(&self) -> u64 {
        self.max_blob_hops.load(Ordering::Relaxed)
    }

    pub(super) fn max_cross_blob_depth(&self) -> u64 {
        self.max_cross_blob_depth.load(Ordering::Relaxed)
    }

    pub(super) fn spillover_count(&self) -> u64 {
        self.spillover_count.load(Ordering::Relaxed)
    }

    pub(super) fn merge_count(&self) -> u64 {
        self.merge_count.load(Ordering::Relaxed)
    }

    pub(super) fn route_resident_demotions(&self) -> u64 {
        self.route_resident_demotions.load(Ordering::Relaxed)
    }

    pub(super) fn cache_evictions(&self) -> u64 {
        self.cache_evictions.load(Ordering::Relaxed)
    }

    pub(super) fn eviction_skips_protected(&self) -> u64 {
        self.eviction_skips_protected.load(Ordering::Relaxed)
    }

    pub(super) fn eviction_skips_route_resident(&self) -> u64 {
        self.eviction_skips_route_resident.load(Ordering::Relaxed)
    }

    pub(super) fn admission_protects(&self) -> u64 {
        self.admission_protects.load(Ordering::Relaxed)
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
