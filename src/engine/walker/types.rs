//! Walker types — public outcomes + internal signals.

use crate::api::errors::Error;
use crate::layout::{BlobGuid, NodeType};
use crate::store::blob_store::AlignedBlobBuf;

// ---------- public types ----------

/// Outcome of a [`super::lookup`] descent.
#[derive(Debug)]
pub enum LookupResult<'a> {
    /// Match found — borrowed view of the value bytes.
    Found(LookupHit<'a>),
    /// No leaf in the tree matches `key`.
    NotFound,
    /// Descent reached a [`NodeType::Blob`] crossing. The caller
    /// (typically `Tree::get`) must load the child blob by its GUID
    /// and continue from the child blob's own `header.root_slot`
    /// with `depth = child_depth`.
    Crossing(BlobNodeCrossing),
}

/// Borrowed lookup hit.
#[derive(Debug, Clone, Copy)]
pub struct LookupHit<'a> {
    /// Value bytes borrowed from the pinned blob.
    pub value: &'a [u8],
    /// Leaf sequence attached to the value.
    pub seq: u64,
}

/// Where a single-blob walker descent stopped at a BlobNode.
#[derive(Debug, Clone, Copy)]
pub struct BlobNodeCrossing {
    /// GUID of the blob to walk into next.
    pub child_guid: BlobGuid,
    /// `depth` to pass to the next [`super::lookup_at`] call (the
    /// parent blob's depth plus the BlobNode's inline prefix length).
    pub child_depth: usize,
}

/// Outcome of an [`super::insert::insert`] / [`super::insert::insert_multi`].
#[derive(Debug)]
pub struct InsertOutcome {
    /// `true` iff the root blob's cached bytes changed. Cross-
    /// blob updates usually mutate only a child blob; the walker
    /// marks those children dirty itself, and the `Tree` caller
    /// should only mark the root when this flag is set.
    pub root_dirty: bool,
    /// `true` iff the walker inserted or updated a leaf. Conditional
    /// insert paths use `false` for "guard did not pass".
    pub mutated: bool,
}

/// Outcome of an [`super::erase::erase`] / [`super::erase::erase_multi`].
#[derive(Debug)]
pub struct EraseOutcome {
    /// `true` iff the root blob's cached bytes changed. Cross-
    /// blob erases usually mutate only a child blob; the walker
    /// marks those children dirty itself, and the `Tree` caller
    /// should only mark the root when this flag is set.
    pub root_dirty: bool,
    /// `true` iff the walker actually tombstoned a live leaf —
    /// `Tree::delete` uses this to decide dirty-mark + WAL-append.
    /// `false` means "key was not in the tree" / "leaf was already
    /// tombstoned" — the call is then a no-op.
    pub mutated: bool,
}

/// Outcome of [`super::make_blob_from_node`] — a freshly-built blob
/// image holding a clone of the source subtree.
#[derive(Debug)]
pub struct MakeBlobOutcome {
    /// New blob's full 512 KB image — write this to the store
    /// under `new_guid`.
    pub buf: AlignedBlobBuf,
}

/// Guard applied to an insert/update attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertCondition {
    /// Always insert or replace.
    Always,
    /// Insert only when no live record currently exists at the key.
    IfAbsent,
    /// Replace only when a live record exists with this sequence.
    IfVersion(u64),
}

/// Guard applied to an erase attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EraseCondition {
    /// Delete any live record matching the key.
    Always,
    /// Delete only when the live leaf carries this sequence.
    IfVersion(u64),
}

// ---------- internal types (pub(super) for sibling submodules) ----------

/// Return-value carried up an `insert_at` recursion.
#[derive(Debug)]
pub(super) struct InsertReturn {
    /// Byte offset the parent should now point at — may be the same
    /// as the input node's offset or a freshly-allocated promotion.
    pub(super) off_after: u32,
    /// `true` iff bytes changed in the blob.
    pub(super) mutated: bool,
}

/// What an erase descent tells its parent to do.
#[derive(Debug)]
pub(super) enum EraseSignal {
    /// Node stays as-is — nothing to rewire above.
    Unchanged,
    /// The subtree at this node disappeared entirely. Parent should
    /// drop the corresponding child entry and (if it now has 0
    /// remaining children) collapse itself in turn.
    SubtreeGone,
    /// The subtree shrank to a single node. Parent should rewrite
    /// its child pointer to the carried byte offset.
    Replaced(u32),
}

#[derive(Debug)]
pub(super) struct EraseReturn {
    pub(super) signal: EraseSignal,
    /// `true` iff a live leaf was tombstoned during the descent.
    pub(super) mutated: bool,
}

pub(super) const STALE_BLOB_CROSSING: &str = "stale blob crossing";

pub(super) const fn stale_blob_crossing(where_: &'static str) -> Error {
    Error::Internal(where_)
}

pub(super) fn is_stale_blob_crossing(error: &Error) -> bool {
    matches!(error, Error::Internal(msg) if msg.starts_with(STALE_BLOB_CROSSING))
}

/// What kind of edge the parent of a victim subtree has.
#[derive(Debug, Clone, Copy)]
pub(super) enum VictimEdgeKind {
    Prefix,
    Inner(NodeType),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Victim {
    /// Byte offset of the parent node that points at the victim.
    pub(super) parent_off: u32,
    /// What kind of edge it is.
    pub(super) kind: VictimEdgeKind,
    /// The byte routing to the victim in the parent (irrelevant
    /// for `Prefix` edges).
    pub(super) byte: u8,
    /// Byte offset of the victim subtree's root.
    pub(super) victim_off: u32,
    /// `true` iff the victim is reached via `header.root_slot`
    /// rather than via a regular parent node — used to dispatch
    /// the parent rewrite path.
    pub(super) via_header_root: bool,
}
