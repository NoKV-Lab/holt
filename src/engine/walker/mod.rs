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
//! - `key` — virtual-terminator search key used by point
//!   read/write paths.
//! - `lookup` — [`lookup`] / [`lookup_at`] / [`lookup_multi_with`]
//!   and single-blob descent arms. Zero-copy: walks against
//!   `BlobFrameRef` so it's safe to run under a `BufferManager`
//!   shared read-guard.
//! - `insert` — [`insert`] / [`insert_multi`] + single-descent
//!   lock-coupled cross-blob mutation.
//! - `erase` — [`erase`] / [`erase_multi`] + single-descent
//!   lock-coupled cross-blob mutation + lone-child collapse
//!   rewiring for single-blob callers.
//! - `spillover` — when a blob fills, pick a victim subtree,
//!   migrate it via [`make_blob_from_node`], free the source slots,
//!   install a `BlobNode` placeholder.
//! - `migrate` — deep-clone primitives: [`make_blob_from_node`]
//!   (spillover) + [`compact_blob`] (in-place repack). Share the
//!   internal `clone_subtree` machinery.
//! - `scan` — tree-wide BFS over reachable blobs
//!   ([`collect_blob_guids`] / `collect_blob_topology_silent`). Used
//!   by [`crate::Tree::stats`] and by `compact` only for cold
//!   maintenance seeding when no candidate hints exist.
//! - `merge` — parent-local single-pass walker ([`try_merge_children`])
//!   that folds every mergeable `BlobNode` child back into its
//!   parent via [`merge_blob`]. Maintenance calls it only for
//!   queued parent candidates.

use std::mem::size_of;

mod erase;
mod insert;
mod key;
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
// Only the multi-blob entry points + maintenance passes are reachable
// from outside the walker. Single-blob primitives (`insert`, `erase`,
// `lookup`, `lookup_at`) and the Outcome types live behind their
// submodule paths and are only consumed by sibling submodules and
// the walker's own `tests`.

pub use erase::{erase_multi, erase_multi_conditional};
pub use insert::{insert_multi, insert_multi_conditional};
pub(crate) use insert::{insert_multi_batch_conditional, InsertBatchItem};
pub(crate) use key::SearchKey;
pub use lookup::lookup_multi_with;
pub use merge::try_merge_children;
pub use migrate::{blob_needs_compaction, compact_blob};
pub use range::{RangeBuilder, RangeEntry, RangeIter};
pub use scan::{collect_blob_guids, collect_blob_topology_silent};
pub use types::{EraseCondition, EraseOutcome, InsertCondition, InsertOutcome};

// ---------- shared internals ----------

/// Cap on the spillover-retry loop inside `insert_multi`. Each
/// spillover migrates an occupancy-aware non-Blob subtree out of
/// the current blob.
///
/// The current heuristic skips BlobNodes, descends inside overfull
/// path branches, and aims for a child fill band instead of blindly
/// peeling off the largest branch. One spillover should free enough
/// slot/extent pressure for the triggering retry while avoiding an
/// immediately-full child blob.
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
