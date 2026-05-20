//! Tree-wide traversal — enumerate every blob reachable from a
//! starting root, in BFS order, by scanning each blob's tree shape
//! for [`NodeType::Blob`] crossings.
//!
//! Used by [`crate::api::Tree::stats`] and [`crate::api::Tree::compact`]
//! to fan out across the whole on-disk tree without each caller
//! having to reimplement cross-blob descent.
//!
//! Also hosts [`refresh_blob_node_pointers`], the post-compact
//! invariant repair pass that brings every `BlobNode.child_entry_ptr`
//! back into sync with the corresponding child blob's
//! `header.root_slot` — insert / erase keep the pair in lock-step,
//! but `compact_blob` rewrites a blob's root_slot in isolation, so
//! parents need a separate sweep to catch up.

use crate::api::errors::{Error, Result};
use crate::layout::{
    BlobGuid, BlobNode, Node16, Node256, Node4, Node48, NodeType, Prefix, BLOB_MAX_INLINE,
};
use crate::store::{BlobFrameRef, BufferManager};

use super::cast;
use super::readers::resolve_typed;
use super::writers::write_struct_to_slot;

/// Return every blob GUID reachable from `root_guid` (including
/// `root_guid` itself), in BFS order.
///
/// Each blob is pinned + read under a shared guard exactly once; no
/// blob bytes are copied. The returned vector's first element is
/// always `root_guid`.
///
/// Uses `BufferManager::pin`, which bumps cache hit/miss
/// counters and refreshes `last_touched`. Callers on the
/// observability path (`Tree::stats`, metrics scrapes) should
/// use [`collect_blob_guids_silent`] instead to avoid the
/// scrape polluting the counters it's about to report.
pub fn collect_blob_guids(bm: &BufferManager, root_guid: BlobGuid) -> Result<Vec<BlobGuid>> {
    collect_blob_guids_inner(bm, root_guid, /*silent=*/ false)
}

/// Same shape as [`collect_blob_guids`] but uses
/// `BufferManager::pin_silent`, so the walk does not bump
/// `cache_hits` / `cache_misses` and does not refresh
/// `last_touched`. Used by `Tree::stats`.
pub fn collect_blob_guids_silent(bm: &BufferManager, root_guid: BlobGuid) -> Result<Vec<BlobGuid>> {
    collect_blob_guids_inner(bm, root_guid, /*silent=*/ true)
}

fn collect_blob_guids_inner(
    bm: &BufferManager,
    root_guid: BlobGuid,
    silent: bool,
) -> Result<Vec<BlobGuid>> {
    let mut all = vec![root_guid];
    let mut queue: Vec<BlobGuid> = vec![root_guid];
    while let Some(guid) = queue.pop() {
        let pin = if silent {
            bm.pin_silent(guid)?
        } else {
            bm.pin(guid)?
        };
        let mut found = Vec::new();
        {
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let root_slot = frame.header().root_slot;
            scan_subtree(frame, root_slot, &mut found)?;
        }
        for child_guid in found {
            all.push(child_guid);
            queue.push(child_guid);
        }
    }
    Ok(all)
}

fn scan_subtree(frame: BlobFrameRef<'_>, slot: u16, out: &mut Vec<BlobGuid>) -> Result<()> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::scan::scan_subtree: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot | NodeType::Leaf => Ok(()),
        NodeType::Prefix => {
            let p = cast::<Prefix>(body);
            scan_subtree(frame, p.child as u16, out)
        }
        NodeType::Node4 => {
            let n = cast::<Node4>(body);
            let count = (n.count as usize).min(4);
            for i in 0..count {
                scan_subtree(frame, n.children[i] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node16 => {
            let n = cast::<Node16>(body);
            let count = (n.count as usize).min(16);
            for i in 0..count {
                scan_subtree(frame, n.children[i] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node48 => {
            let n = cast::<Node48>(body);
            for i in 0..256usize {
                let idx = n.index[i];
                if idx == 0 {
                    continue;
                }
                let ci = idx as usize - 1;
                if ci >= 48 {
                    return Err(Error::node_corrupt(
                        "walker::scan::scan_subtree: Node48 index out of range",
                    ));
                }
                scan_subtree(frame, n.children[ci] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node256 => {
            let n = cast::<Node256>(body);
            for c in n.children {
                if c != 0 {
                    scan_subtree(frame, c as u16, out)?;
                }
            }
            Ok(())
        }
        NodeType::Blob => {
            let b = cast::<BlobNode>(body);
            let plen = b.prefix_len as usize;
            if plen > BLOB_MAX_INLINE {
                return Err(Error::node_corrupt(
                    "walker::scan::scan_subtree: BlobNode prefix_len exceeds inline",
                ));
            }
            out.push(b.child_blob_guid);
            Ok(())
        }
    }
}

/// Bring every `BlobNode.child_entry_ptr` reachable from `root_guid`
/// back into sync with the corresponding child blob's
/// `header.root_slot`.
///
/// `compact_blob` rewrites a blob's root in isolation — the child
/// blob's `header.root_slot` advances to whatever slot the rebuilt
/// subtree landed in, but parents have no way to learn that from
/// inside `compact_blob`. Insert / erase keep the pair in lock-step
/// inline, so this sweep is only needed after a structural rewrite
/// (today: post-compact). Returns the number of `BlobNode`
/// crossings whose pointer it had to update.
pub fn refresh_blob_node_pointers(bm: &BufferManager, root_guid: BlobGuid) -> Result<u32> {
    let guids = collect_blob_guids(bm, root_guid)?;
    let mut updated: u32 = 0;
    for parent_guid in guids {
        // Read pass — collect each BlobNode's (slot, child_guid).
        let mut crossings: Vec<(u16, BlobGuid)> = Vec::new();
        {
            let parent_pin = bm.pin(parent_guid)?;
            let guard = parent_pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            collect_blob_nodes_with_slot(frame, frame.header().root_slot, &mut crossings)?;
        }
        if crossings.is_empty() {
            continue;
        }
        // Resolve each child's current header.root_slot via a fresh
        // pin (different blob from `parent_guid` — no deadlock).
        let mut want: Vec<(u16, u32)> = Vec::with_capacity(crossings.len());
        for (bn_slot, child_guid) in &crossings {
            let child_pin = bm.pin(*child_guid)?;
            let child_guard = child_pin.read();
            let child_frame = BlobFrameRef::wrap(child_guard.as_slice());
            want.push((*bn_slot, u32::from(child_frame.header().root_slot)));
        }
        // Write pass — apply only the rewrites where `child_entry_ptr`
        // disagrees with the child's current `root_slot`.
        let mut touched = false;
        {
            let parent_pin = bm.pin(parent_guid)?;
            let mut guard = parent_pin.write();
            let mut frame = guard.frame();
            for (bn_slot, new_child_entry) in want {
                let body = frame.body_of_slot(bn_slot).ok_or(Error::node_corrupt(
                    "refresh_blob_node_pointers: body resolution failed",
                ))?;
                let mut bn = *cast::<BlobNode>(body);
                if bn.child_entry_ptr != new_child_entry {
                    bn.child_entry_ptr = new_child_entry;
                    write_struct_to_slot(&mut frame, bn_slot, &bn)?;
                    updated += 1;
                    touched = true;
                }
            }
        }
        if touched {
            // Stage the rewrite through the unified dirty-set
            // protocol; `Tree::compact` (the only caller) drives
            // the trailing `flush_dirty_inline` → `backend.flush`
            // → conditional WAL truncate sequence under W2D.
            //
            // An inline `bm.commit(parent_guid)` here would push
            // the cache image (which may still include unflushed
            // user-write WAL records' effects on `parent_guid`)
            // straight to backend, re-opening the W2D hole that
            // `Tree::compact`'s phase 1/2 already closed. The
            // `STRUCTURAL_SEQ` sentinel keeps this dirty entry
            // out of the WAL trim watermark — the refresh itself
            // doesn't correspond to any WAL record.
            bm.mark_dirty(parent_guid, crate::store::buffer_manager::STRUCTURAL_SEQ);
        }
    }
    Ok(updated)
}

/// Collect every `BlobNode`'s `(slot, child_blob_guid)` reachable
/// from `slot`. Inner-node / prefix recursion just descends; the
/// `Blob` arm records the pair.
fn collect_blob_nodes_with_slot(
    frame: BlobFrameRef<'_>,
    slot: u16,
    out: &mut Vec<(u16, BlobGuid)>,
) -> Result<()> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "collect_blob_nodes_with_slot: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot | NodeType::Leaf => Ok(()),
        NodeType::Prefix => {
            let p = cast::<Prefix>(body);
            collect_blob_nodes_with_slot(frame, p.child as u16, out)
        }
        NodeType::Node4 => {
            let n = cast::<Node4>(body);
            let count = (n.count as usize).min(4);
            for i in 0..count {
                collect_blob_nodes_with_slot(frame, n.children[i] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node16 => {
            let n = cast::<Node16>(body);
            let count = (n.count as usize).min(16);
            for i in 0..count {
                collect_blob_nodes_with_slot(frame, n.children[i] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node48 => {
            let n = cast::<Node48>(body);
            for b in 0..256usize {
                let idx = n.index[b];
                if idx == 0 {
                    continue;
                }
                let ci = idx as usize - 1;
                if ci >= 48 {
                    return Err(Error::node_corrupt(
                        "collect_blob_nodes_with_slot: Node48 index out of range",
                    ));
                }
                collect_blob_nodes_with_slot(frame, n.children[ci] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node256 => {
            let n = cast::<Node256>(body);
            for c in n.children {
                if c != 0 {
                    collect_blob_nodes_with_slot(frame, c as u16, out)?;
                }
            }
            Ok(())
        }
        NodeType::Blob => {
            let b = cast::<BlobNode>(body);
            let plen = b.prefix_len as usize;
            if plen > BLOB_MAX_INLINE {
                return Err(Error::node_corrupt(
                    "collect_blob_nodes_with_slot: BlobNode prefix_len exceeds inline",
                ));
            }
            out.push((slot, b.child_blob_guid));
            Ok(())
        }
    }
}
