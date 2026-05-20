//! ART engine — descent, mutation, range-scan, and the
//! per-blob hot-path primitives the walker is built from.
//!
//! Submodules:
//!
//! - [`walker`] — the recursive walker, split into focused
//!   files: `lookup` / `insert` / `erase` / `range` / `merge`
//!   / `scan` (read-side walkers + tree-wide passes),
//!   `spillover` / `migrate` (write-side restructuring), and
//!   the internal `readers` / `writers` / `types` primitives
//!   they share.
//! - [`simd`] — SIMD hot paths the walker calls into:
//!   `Node16` byte search and longest-common-prefix
//!   (SSE2 / NEON / scalar fallback).
//!
//! Read paths take [`crate::store::BlobFrameRef`] and run
//! zero-copy against `BufferManager`-pinned buffers; writes
//! take an exclusive `HybridLatch` for the duration of the
//! mutation. See `concurrency` for the latch contract.

pub mod simd;
pub mod walker;

// Re-export only the items consumed outside the `walker` subtree
// (api::tree, api::range, api::stats). Walker-internal types stay
// hidden behind `mod walker;`.
pub use walker::{
    collect_blob_guids, collect_blob_guids_silent, compact_blob, erase_multi, insert_multi,
    lookup_multi_with, refresh_blob_node_pointers, try_merge_children, EraseOutcome, RangeBuilder,
    RangeEntry, RangeIter,
};
