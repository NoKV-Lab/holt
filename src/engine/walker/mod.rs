//! Recursive ART walker — descent / insert / erase / spillover /
//! migrate. Split into focused submodules (all `pub(super)` —
//! internal to the walker, re-exported as the public surface from
//! this `mod.rs`):
//!
//! - `types` — public outcomes ([`LookupResult`], [`InsertOutcome`],
//!   …) + internal signals (`EraseSignal`, `Victim`, …).
//! - `readers` — decode slot bodies + leaf extents off a
//!   `BlobFrameRef`.
//! - `writers` — allocate fresh slots / extents and populate them.
//! - `lookup` — [`lookup`] / [`lookup_at`] / [`lookup_multi_with`]
//!   and single-blob descent arms. Zero-copy: walks against
//!   `BlobFrameRef` so it's safe to run under a `BufferManager`
//!   shared read-guard.
//! - `insert` — [`insert`] / [`insert_multi`] + `insert_at`
//!   dispatch + `insert_at_blob_node` cross-blob arm.
//! - `erase` — [`erase`] / [`erase_multi`] + `erase_at` dispatch +
//!   `erase_at_blob_node` cross-blob arm + lone-child collapse
//!   rewiring.
//! - `spillover` — when a blob fills, pick a victim subtree,
//!   migrate it via [`make_blob_from_node`], free the source slots,
//!   install a `BlobNode` placeholder.
//! - `migrate` — deep-clone primitives: [`make_blob_from_node`]
//!   (spillover) + [`compact_blob`] (in-place repack). Share the
//!   internal `clone_subtree` machinery.
//! - `scan` — tree-wide BFS over reachable blobs ([`collect_blob_guids`]).
//!   Used by [`crate::Tree::stats`] and [`crate::Tree::compact`] to
//!   fan out across the whole on-disk tree.
//! - `merge` — tree-wide single-pass walker ([`try_merge_children`])
//!   that folds every mergeable `BlobNode` child back into its
//!   parent via [`merge_blob`]. Wired into [`crate::Tree::compact`]
//!   after the per-blob compaction pass.

use std::mem::size_of;

mod erase;
mod insert;
mod lookup;
mod merge;
mod migrate;
mod range;
mod readers;
mod scan;
mod spillover;
#[cfg(test)]
mod tests;
mod types;
mod writers;

// ---------- public-to-engine surface ----------
//
// Only the multi-blob entry points + tree-wide passes are reachable
// from outside the walker. Single-blob primitives (`insert`, `erase`,
// `lookup`, `lookup_at`) and the Outcome types live behind their
// submodule paths and are only consumed by sibling submodules and
// the walker's own `tests`.

pub use erase::erase_multi;
pub use insert::insert_multi;
pub use lookup::lookup_multi_with;
pub use merge::try_merge_children;
pub use migrate::compact_blob;
pub use range::{RangeBuilder, RangeEntry, RangeIter};
pub use scan::{collect_blob_guids, collect_blob_guids_silent, refresh_blob_node_pointers};
pub use types::EraseOutcome;

// ---------- shared internals ----------

/// Cap on the spillover-retry loop inside `insert_multi` /
/// `insert_at_blob_node`. Each spillover migrates the largest non-
/// Blob subtree out of the current blob.
///
/// With the current heuristic (pick-largest, skip BlobNodes, cross-
/// type Prefix↔Blob free-list fallback) one spillover frees roughly
/// `(largest-child-subtree-size)` worth of slot entries.
///
/// 64 covers a 2-3× workload-vs-blob-capacity ratio for the
/// uniform-key regimes the benchmark + integration tests exercise.
pub(super) const MAX_SPILLOVER_ATTEMPTS: u32 = 64;

/// Reinterpret a slot body as a `#[repr(C)]` layout struct.
///
/// SAFETY: layout types are POD; body length + alignment are
/// guaranteed by `BlobFrame`'s invariants.
pub(super) fn cast<T>(body: &[u8]) -> &T {
    debug_assert_eq!(body.len(), size_of::<T>());
    debug_assert_eq!(body.as_ptr() as usize % std::mem::align_of::<T>(), 0);
    unsafe { &*body.as_ptr().cast::<T>() }
}
