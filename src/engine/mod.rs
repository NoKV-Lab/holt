//! ART engine — descent, mutation, range-scan, and the
//! per-blob hot-path primitives the walker is built from.
//!
//! Submodules:
//!
//! - [`walker`] — the recursive walker, split into focused
//!   files: `lookup` / `insert` / `erase` / `range` / `merge`
//!   / `scan` (read-side walkers + stats/cold-seed scans),
//!   `spillover` / `migrate` (write-side restructuring), and
//!   the internal `readers` / `writers` / `types` primitives
//!   they share.
//! - [`simd`] — SIMD hot paths the walker calls into:
//!   `Node16` byte search, longest-common-prefix, Node48 /
//!   Node256 sparse-child scans, and delimiter byte search
//!   (SSE2 / NEON / scalar fallback).
//!
//! Read paths take [`crate::store::BlobFrameRef`] and run
//! zero-copy against `BufferManager`-pinned buffers; writes
//! take an exclusive `HybridLatch` for the duration of the
//! mutation. See `concurrency` for the latch contract.

mod route_cache;
mod simd;
mod walker;

// Re-export only the items consumed outside the `walker` subtree.
// Walker-internal types stay hidden behind `mod walker;`.
pub(crate) use route_cache::{RouteCache, RouteHit};
pub(crate) use simd::prefetch_read_data;
pub use walker::{
    blob_needs_compaction, collect_blob_guids, collect_blob_topology_silent, compact_blob,
    erase_multi, erase_multi_conditional, insert_multi, insert_multi_conditional,
    lookup_multi_with, try_merge_children, EraseCondition, EraseOutcome, InsertCondition,
    InsertOutcome, KeyRangeBuilder, KeyRangeEntry, KeyRangeEntryRef, KeyRangeIter, KeyScanOutcome,
    PrefixCount, RangeBuilder, RangeEntry, RangeIter, ScanStats,
};
pub(crate) use walker::{
    collect_blob_children_from_frame, fresh_blob_guid, insert_multi_batch_conditional,
    InsertBatchItem, PrefixListCache, SearchKey,
};
