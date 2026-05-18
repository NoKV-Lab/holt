//! Spillover infra — pick a subtree to migrate when a blob fills,
//! write it through to a fresh child blob, free the source's
//! slots, and install a `BlobNode` placeholder.
//!
//! Also hosts:
//! - `free_subtree` (recursive slot reclaim after migration)
//! - `fresh_blob_guid` (cheap process-local GUIDs)
//! - `compact_blob` (in-place repack, re-exported from
//!   [`super::migrate`])

use crate::api::errors::{Error, Result};
use crate::layout::{BlobGuid, BlobNode, Node16, Node256, Node4, Node48, NodeType, Prefix};
use crate::store::backend::Backend;
use crate::store::{BlobFrame, BufferManager};

use super::cast;
use super::migrate::make_blob_from_node;
use super::readers::{ntype_of, read_node16, read_node256, read_node4, read_node48, read_prefix};
use super::types::{Victim, VictimEdgeKind};
use super::writers::{inner_update_child, set_prefix_child, write_struct_to_slot};

// Re-export `compact_blob` so `insert_multi` / `insert_at_blob_node`
// can reach it via `super::spillover::compact_blob`.
pub(super) use super::migrate::compact_blob;

/// Trigger spillover on `frame`: migrate a subtree out to a fresh
/// child blob (via [`make_blob_from_node`]), free the migrated
/// slots, and install a [`BlobNode`] placeholder at the migrated
/// location.
///
/// Heuristic: pick the **largest non-Blob** subtree at the root's
/// first branching node (i.e. skip BlobNode children — those are
/// already migrated). This maximises space freed per spillover
/// iteration.
///
/// Returns the BlobNode slot installed in `frame` so callers /
/// tests can verify. The new blob is **already written to the
/// backend** at the time of return.
pub(super) fn spillover_blob(bm: &BufferManager, frame: &mut BlobFrame<'_>) -> Result<u16> {
    let root_slot = frame.header().root_slot;
    let victim = pick_victim_subtree(frame, root_slot)?;

    let new_guid = fresh_blob_guid();
    let outcome = make_blob_from_node(frame, victim.victim_slot, new_guid)?;

    // Persist the new blob BEFORE installing the BlobNode in the
    // source. If we crash between these two writes, the new blob
    // sits orphaned (recoverable via a future GC pass); we never
    // end up with a parent BlobNode pointing at a non-existent
    // child blob. `BufferManager::write_blob` caches the fresh
    // image and writes through to the inner backend in one call.
    bm.write_blob(new_guid, &outcome.buf)?;
    bm.flush()?;

    // Free the migrated subtree's slots in the source blob.
    free_subtree(frame, victim.victim_slot)?;

    // Allocate a BlobNode pointing at (new_guid, entry_slot).
    let bn_alloc = frame.alloc_node(NodeType::Blob)?;
    let bn = BlobNode::new(&[], new_guid, u32::from(outcome.entry_slot));
    write_struct_to_slot(frame, bn_alloc.slot, &bn)?;

    // Wire the parent of the migrated subtree to point at the new
    // BlobNode instead of the now-freed victim slot.
    if victim.parent_slot == root_slot && victim.via_header_root {
        frame.header_mut().root_slot = bn_alloc.slot;
    } else {
        match victim.kind {
            VictimEdgeKind::Prefix => {
                set_prefix_child(frame, victim.parent_slot, u32::from(bn_alloc.slot))?;
            }
            VictimEdgeKind::Inner(parent_ntype) => {
                inner_update_child(
                    frame,
                    victim.parent_slot,
                    parent_ntype,
                    victim.byte,
                    u32::from(bn_alloc.slot),
                )?;
            }
        }
    }

    Ok(bn_alloc.slot)
}

/// Count the total number of node slots reachable from `root`
/// in `frame`. Bounded by `MAX_SLOTS` (= 10240). Used by the
/// spillover heuristic to pick the largest migration candidate.
pub(super) fn count_subtree_nodes(frame: &BlobFrame<'_>, root: u16) -> Result<u32> {
    let ntype = ntype_of(frame.as_ref(), root)?;
    let body = frame.body_of_slot(root).ok_or(Error::NodeCorrupt {
        context: "count_subtree_nodes: body resolution failed",
    })?;
    let mut count: u32 = 1;
    match ntype {
        NodeType::Invalid => {
            return Err(Error::NodeCorrupt {
                context: "count_subtree_nodes: Invalid",
            });
        }
        NodeType::Leaf | NodeType::EmptyRoot | NodeType::Blob => {}
        NodeType::Prefix => {
            let p = cast::<Prefix>(body);
            count = count.saturating_add(count_subtree_nodes(frame, p.child as u16)?);
        }
        NodeType::Node4 => {
            let n = cast::<Node4>(body);
            for i in 0..(n.count as usize).min(4) {
                count = count.saturating_add(count_subtree_nodes(frame, n.children[i] as u16)?);
            }
        }
        NodeType::Node16 => {
            let n = cast::<Node16>(body);
            for i in 0..(n.count as usize).min(16) {
                count = count.saturating_add(count_subtree_nodes(frame, n.children[i] as u16)?);
            }
        }
        NodeType::Node48 => {
            let n = cast::<Node48>(body);
            for c in &n.children {
                if *c != 0 {
                    count = count.saturating_add(count_subtree_nodes(frame, *c as u16)?);
                }
            }
        }
        NodeType::Node256 => {
            let n = cast::<Node256>(body);
            for c in &n.children {
                if *c != 0 {
                    count = count.saturating_add(count_subtree_nodes(frame, *c as u16)?);
                }
            }
        }
    }
    Ok(count)
}

/// Pick the largest non-`BlobNode` subtree at the root's first
/// branching node. Walks through chained `Prefix` nodes to reach
/// the first `Node4/16/48/256`.
///
/// **Heuristic rationale:**
/// - Skipping `Blob` children avoids spillover-stutter (previously-
///   migrated children would otherwise get re-migrated into
///   wrapper blobs without freeing any actual data).
/// - Picking the *largest* child (by node count) maximises space
///   freed per spillover iteration.
#[allow(clippy::too_many_lines)] // intentional — one match over NodeType arms
fn pick_victim_subtree(frame: &BlobFrame<'_>, start_slot: u16) -> Result<Victim> {
    let mut current = start_slot;
    loop {
        let ntype = ntype_of(frame.as_ref(), current)?;
        match ntype {
            NodeType::Node4 => {
                let n = read_node4(frame.as_ref(), current)?;
                return pick_largest_non_blob(
                    frame,
                    current,
                    NodeType::Node4,
                    (n.count as usize).min(4),
                    &n.keys[..],
                    &n.children[..],
                    false,
                );
            }
            NodeType::Node16 => {
                let n = read_node16(frame.as_ref(), current)?;
                return pick_largest_non_blob(
                    frame,
                    current,
                    NodeType::Node16,
                    (n.count as usize).min(16),
                    &n.keys[..],
                    &n.children[..],
                    false,
                );
            }
            NodeType::Node48 => {
                let n = read_node48(frame.as_ref(), current)?;
                let mut best: Option<Victim> = None;
                let mut best_size: u32 = 0;
                for b in 0..256usize {
                    let idx = n.index[b];
                    if idx == 0 {
                        continue;
                    }
                    let child_slot = n.children[idx as usize - 1] as u16;
                    if ntype_of(frame.as_ref(), child_slot)? == NodeType::Blob {
                        continue;
                    }
                    let size = count_subtree_nodes(frame, child_slot)?;
                    if size > best_size {
                        best_size = size;
                        best = Some(Victim {
                            parent_slot: current,
                            kind: VictimEdgeKind::Inner(NodeType::Node48),
                            byte: b as u8,
                            victim_slot: child_slot,
                            via_header_root: false,
                        });
                    }
                }
                return best.ok_or(Error::NotYetImplemented(
                    "spillover: no non-Blob children to migrate (Node48)",
                ));
            }
            NodeType::Node256 => {
                let n = read_node256(frame.as_ref(), current)?;
                let mut best: Option<Victim> = None;
                let mut best_size: u32 = 0;
                for (i, c) in n.children.iter().enumerate() {
                    if *c == 0 {
                        continue;
                    }
                    let child_slot = *c as u16;
                    if ntype_of(frame.as_ref(), child_slot)? == NodeType::Blob {
                        continue;
                    }
                    let size = count_subtree_nodes(frame, child_slot)?;
                    if size > best_size {
                        best_size = size;
                        best = Some(Victim {
                            parent_slot: current,
                            kind: VictimEdgeKind::Inner(NodeType::Node256),
                            byte: i as u8,
                            victim_slot: child_slot,
                            via_header_root: false,
                        });
                    }
                }
                return best.ok_or(Error::NotYetImplemented(
                    "spillover: no non-Blob children to migrate (Node256)",
                ));
            }
            NodeType::Prefix => {
                let p = read_prefix(frame.as_ref(), current)?;
                let child_slot = p.child as u16;
                let child_ntype = ntype_of(frame.as_ref(), child_slot)?;
                match child_ntype {
                    NodeType::Node4
                    | NodeType::Node16
                    | NodeType::Node48
                    | NodeType::Node256
                    | NodeType::Prefix => {
                        current = child_slot;
                    }
                    NodeType::Leaf | NodeType::EmptyRoot | NodeType::Blob => {
                        return Ok(Victim {
                            parent_slot: current,
                            kind: VictimEdgeKind::Prefix,
                            byte: 0,
                            victim_slot: child_slot,
                            via_header_root: false,
                        });
                    }
                    NodeType::Invalid => {
                        return Err(Error::NodeCorrupt {
                            context: "pick_victim_subtree: Prefix child Invalid",
                        });
                    }
                }
            }
            NodeType::Leaf | NodeType::EmptyRoot | NodeType::Blob => {
                return Err(Error::NotYetImplemented(
                    "spillover: tree too degenerate to migrate (root is Leaf/Empty/Blob)",
                ));
            }
            NodeType::Invalid => {
                return Err(Error::NodeCorrupt {
                    context: "pick_victim_subtree: Invalid",
                });
            }
        }
    }
}

/// Scan a Node4/Node16's `keys[]`+`children[]` parallel arrays for
/// the largest non-`BlobNode` child.
fn pick_largest_non_blob(
    frame: &BlobFrame<'_>,
    parent_slot: u16,
    parent_ntype: NodeType,
    count: usize,
    keys: &[u8],
    children: &[u32],
    via_header_root: bool,
) -> Result<Victim> {
    let mut best: Option<Victim> = None;
    let mut best_size: u32 = 0;
    for i in 0..count {
        let child_slot = children[i] as u16;
        if ntype_of(frame.as_ref(), child_slot)? == NodeType::Blob {
            continue;
        }
        let size = count_subtree_nodes(frame, child_slot)?;
        if size > best_size {
            best_size = size;
            best = Some(Victim {
                parent_slot,
                kind: VictimEdgeKind::Inner(parent_ntype),
                byte: keys[i],
                victim_slot: child_slot,
                via_header_root,
            });
        }
    }
    best.ok_or(Error::NotYetImplemented(
        "spillover: no non-Blob children to migrate",
    ))
}

/// Recursively free every slot of the subtree rooted at `root` in
/// `frame`. Used by spillover to reclaim source-side slot entries
/// after `make_blob_from_node` has copied them out.
pub(super) fn free_subtree(frame: &mut BlobFrame<'_>, root: u16) -> Result<()> {
    let ntype = ntype_of(frame.as_ref(), root)?;
    // Snapshot the body bytes before mutating the slot table so the
    // following `frame.free_node` calls can't invalidate them.
    let body_copy = frame
        .body_of_slot(root)
        .ok_or(Error::NodeCorrupt {
            context: "free_subtree: body resolution failed",
        })?
        .to_vec();

    match ntype {
        NodeType::Invalid => {
            return Err(Error::NodeCorrupt {
                context: "free_subtree: Invalid in source",
            });
        }
        NodeType::Leaf | NodeType::EmptyRoot | NodeType::Blob => {}
        NodeType::Prefix => {
            let p = cast::<Prefix>(&body_copy);
            free_subtree(frame, p.child as u16)?;
        }
        NodeType::Node4 => {
            let n = cast::<Node4>(&body_copy);
            for i in 0..(n.count as usize).min(4) {
                free_subtree(frame, n.children[i] as u16)?;
            }
        }
        NodeType::Node16 => {
            let n = cast::<Node16>(&body_copy);
            for i in 0..(n.count as usize).min(16) {
                free_subtree(frame, n.children[i] as u16)?;
            }
        }
        NodeType::Node48 => {
            let n = cast::<Node48>(&body_copy);
            for c in &n.children {
                if *c != 0 {
                    free_subtree(frame, *c as u16)?;
                }
            }
        }
        NodeType::Node256 => {
            let n = cast::<Node256>(&body_copy);
            for c in &n.children {
                if *c != 0 {
                    free_subtree(frame, *c as u16)?;
                }
            }
        }
    }

    frame.free_node(root)?;
    Ok(())
}

/// Produce a fresh blob GUID. Process-local uniqueness for v0.1:
/// monotonic counter + process ID + magic suffix. Tag the high
/// bytes so a fresh GUID never collides with `ROOT_BLOB_GUID =
/// [0; 16]`. A full UUID v4 generator can replace this later.
pub(super) fn fresh_blob_guid() -> BlobGuid {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let c = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id() as u64;
    let mut g = [0u8; 16];
    g[0..8].copy_from_slice(&c.to_le_bytes());
    g[8..12].copy_from_slice(&(pid as u32).to_le_bytes());
    g[12] = 0xA1;
    g[13] = 0xB2;
    g[14] = 0xC3;
    g[15] = 0xD4;
    g
}
