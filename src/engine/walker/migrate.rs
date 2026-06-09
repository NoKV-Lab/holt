//! Deep-clone primitives — `make_blob_from_node` (spillover) and
//! `compact_blob` (in-place repack). Share the same recursive
//! `clone_subtree` machinery; both produce a fresh, packed image
//! containing a deep copy of a source subtree.
//!
//! `clone_subtree` runs in two modes:
//!
//! - **preserve** (`filter_tombstones = false`) — copies every byte
//!   verbatim, tombstones included. The result is always `Some`.
//!   Used by `make_blob_from_node` to migrate a subtree wholesale
//!   into a fresh blob without changing its observable shape.
//! - **filter** (`filter_tombstones = true`) — drops tombstoned
//!   leaves and collapses inner nodes whose live-child count falls
//!   below the natural threshold (lone-child → `Prefix([byte])`;
//!   smaller-tier `NodeType` if the count slips below its grow
//!   point). Returns `None` only when the whole subtree under
//!   `src_slot` has no live leaves. Used by `compact_blob` to
//!   reclaim tombstone leaves + bump-area waste in one rebuild.

use crate::api::errors::{Error, Result};
use crate::layout::{
    leaf_extent_size, BlobGuid, BlobNode, Leaf, LeafInline, Node16, Node256, Node4, Node48,
    NodeType, Prefix, BLOB_MAX_INLINE, DATA_AREA_START, MAX_SLOTS, PAGE_SIZE, PREFIX_MAX_INLINE,
};
use crate::store::blob_store::AlignedBlobBuf;
use crate::store::{BlobFrame, BlobFrameRef, BufferManager};

use super::cast;
use super::types::MakeBlobOutcome;
use super::writers::{write_prefix_chain, write_struct_to_slot};

/// Conservative bump-area headroom kept free during a merge.
///
/// Larger than `SPILLOVER_RESERVATION` (128 B) so the parent
/// retains room for slot-table growth + a future emergency
/// spillover after the merge completes. Tuning past 4 KB rarely
/// helps; smaller leaves merges flaky under realistic workloads.
const MERGE_RESERVE: u32 = 0x1000;

/// Deep-clone the subtree rooted at `src_slot` of `src_frame` into
/// a fresh 512 KB blob keyed by `new_guid`.
///
/// Used by spillover: when an insert into a blob overflows, the
/// caller migrates a subtree out via this primitive, installs a
/// [`BlobNode`] placeholder where the subtree used to live, and
/// writes both blobs back.
///
/// **Leaf extents are deep-copied as well** — they live in the new
/// blob's data area at fresh offsets pointed at by each cloned
/// Leaf's `key_offset`. The original blob is untouched; freeing
/// the migrated slots is the caller's responsibility (typical
/// pattern is one `BlobFrame::free_node` per migrated slot).
///
/// Migration is **preserve-mode** — tombstones in the source travel
/// to the destination verbatim. Compaction (in either blob) is the
/// place to drop them.
#[cfg(test)]
pub fn make_blob_from_node(
    src_frame: &BlobFrame<'_>,
    src_slot: u16,
    new_guid: BlobGuid,
) -> Result<MakeBlobOutcome> {
    make_blob_from_node_with_buf(AlignedBlobBuf::zeroed(), src_frame, src_slot, new_guid)
}

/// Same as [`make_blob_from_node`], but allocates the destination
/// from the buffer manager's store-preferred allocator. Spillover
/// uses this path so fresh child blobs enter the cache backed by
/// registered `io_uring` buffers when the persistent store has a
/// fixed-buffer pool.
pub fn make_blob_from_node_in(
    bm: &BufferManager,
    src_frame: &BlobFrame<'_>,
    src_slot: u16,
    new_guid: BlobGuid,
) -> Result<MakeBlobOutcome> {
    make_blob_from_node_with_buf(bm.alloc_blob_buf_zeroed(), src_frame, src_slot, new_guid)
}

fn make_blob_from_node_with_buf(
    mut buf: AlignedBlobBuf,
    src_frame: &BlobFrame<'_>,
    src_slot: u16,
    new_guid: BlobGuid,
) -> Result<MakeBlobOutcome> {
    let cloned_root_slot;
    {
        let mut new_frame = BlobFrame::init(buf.as_mut_slice(), new_guid)?;
        cloned_root_slot = clone_subtree(src_frame, &mut new_frame, src_slot, false)?
            .expect("preserve mode never returns None");

        // Release the EmptyRoot sentinel that `BlobFrame::init`
        // seeded at slot 1; it's unreachable now.
        if new_frame.header().root_slot == 1 && cloned_root_slot != 1 {
            new_frame.free_node(1)?;
        }
        new_frame.header_mut().root_slot = cloned_root_slot;
    }
    Ok(MakeBlobOutcome { buf })
}

/// Cheap header-level predicate for whether `compact_blob` can
/// reclaim anything.
///
/// Tombstoned leaves keep their key/value extents until compaction;
/// freed leaf slots also imply leaked leaf extents from alloc-fresh
/// same-key updates. Other free-list entries are normal ART shape
/// churn (EmptyRoot release, Node4→Node16 growth, prefix splits)
/// and are cheap to reuse in place, so they are not enough to make
/// online `Tree::compact` pay a 512 KB scratch allocation + memcpy.
#[must_use]
pub fn blob_needs_compaction(frame: BlobFrameRef<'_>) -> bool {
    let h = frame.header();
    // A freed leaf slot is the churn signal. Regular `Leaf` nodes
    // land on their own free list; `LeafInline` nodes are 56 B and
    // share `Node16`'s free list (see `free_list_class`), so a
    // small-record same-key update parks the old slot there instead.
    // We check both lists — the Node16 list can also hold a freed
    // inner node, which only makes the heuristic slightly more eager
    // (that slot is reclaimable space too, and compaction is a safe,
    // idempotent background pass).
    let leaf_free_list = h.free_list_head[NodeType::Leaf as usize - 1];
    let inline_free_list = h.free_list_head[NodeType::Node16 as usize - 1];
    h.tombstone_leaf_cnt != 0 || leaf_free_list != 0 || inline_free_list != 0
}

/// Repack `buf` in place, discarding all unreachable bytes plus
/// every tombstoned leaf.
///
/// Builds a fresh `BlobFrame` image in a scratch `AlignedBlobBuf`,
/// deep-clones the live subtree from `buf` into it under
/// **filter-mode** (tombstones dropped, inner-node collapse
/// applied wherever a live-child count falls below its
/// `NodeType`'s threshold), then memcpys the scratch image back
/// over `buf`.
///
/// Post-conditions on the rebuilt blob:
///
/// - Contiguous packed data area (every byte in
///   `DATA_AREA_START .. space_used` is live).
/// - Empty free lists (no leftover stale slot entries).
/// - `tombstone_leaf_cnt = 0` (every survivor is by definition live).
/// - `compact_times` bumped by one.
/// - `gap_space` reset to whatever fresh allocations report.
/// - Original `blob_guid` preserved.
/// - If every leaf in the source was tombstoned, the root becomes
///   the freshly-allocated `EmptyRoot` sentinel.
///
/// **What this reclaims:** the leaf key/value extents (allocated
/// via `alloc_extent`, which has no free list), dead node bodies
/// whose slots returned to a per-NodeType free list but whose
/// `NodeType` isn't being allocated any more, and every leaf body
/// + extent whose `tombstone` byte was set.
///
/// **What this costs:** one scratch `AlignedBlobBuf` (512 KB on
/// the heap, lives for the duration of the call) plus one full
/// blob memcpy at the end. Roughly tens of µs on a modern machine.
pub fn compact_blob(buf: &mut AlignedBlobBuf) -> Result<()> {
    let (blob_guid, old_root, old_compact_times) = {
        let old_frame = BlobFrame::wrap(buf.as_mut_slice());
        let h = old_frame.header();
        (h.blob_guid, h.root_slot, h.compact_times)
    };

    let mut new_buf = buf.zeroed_like();
    {
        let mut new_frame = BlobFrame::init(new_buf.as_mut_slice(), blob_guid)?;
        let old_frame = BlobFrame::wrap(buf.as_mut_slice());
        let cloned = clone_subtree(&old_frame, &mut new_frame, old_root, true)?;
        let entry = match cloned {
            Some(slot) => slot,
            None => {
                // Every leaf below the old root was tombstoned —
                // the new tree is empty. Re-seed the EmptyRoot
                // sentinel as the new root.
                new_frame.alloc_node(NodeType::EmptyRoot)?.slot
            }
        };
        if new_frame.header().root_slot == 1 && entry != 1 {
            new_frame.free_node(1)?;
        }
        let h = new_frame.header_mut();
        h.root_slot = entry;
        h.tombstone_leaf_cnt = 0;
        h.compact_times = old_compact_times.saturating_add(1);
    }

    buf.as_mut_slice().copy_from_slice(new_buf.as_slice());

    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "holt::engine::compact",
        blob_guid = ?&blob_guid[..4],
        compact_times = old_compact_times.saturating_add(1),
        "compact_blob: in-place rebuild complete",
    );

    Ok(())
}

// ---------- merge primitives ----------

/// Decide whether the child blob beneath `parent_bn_slot` is safe
/// to fold back into the parent in a single pass.
///
/// Returns `true` when **all** of:
///
/// 1. The combined data-area usage fits in `PAGE_SIZE` with
///    `MERGE_RESERVE` headroom (rephrased below as a
///    child-fits-into-parent-remaining test).
/// 2. The combined slot-table usage stays under `MAX_SLOTS`.
/// 3. The child has **no** own `BlobNode` crossings
///    (`child.num_ext_blobs == 0`) — the merge pass doesn't
///    unfold nested crossings; a child whose subtree itself
///    spans multiple blobs needs that handled by a separate
///    pass first.
/// 4. The child has no tombstoned leaves (`tombstone_leaf_cnt == 0`).
///    Compact the child first if the workload has just churned
///    deletes through it; merging tombstone weight is wasted work.
pub fn is_mergeable(
    bm: &BufferManager,
    parent_frame: &BlobFrame<'_>,
    parent_bn_slot: u16,
) -> Result<bool> {
    let bn = read_blob_node(parent_frame, parent_bn_slot)?;
    let child_pin = bm.pin(bn.child_blob_guid)?;
    let guard = child_pin.read();
    let child_frame = BlobFrameRef::wrap(guard.as_slice());

    let parent_h = parent_frame.header();
    let child_h = child_frame.header();

    let parent_remaining = PAGE_SIZE
        .saturating_sub(parent_h.space_used)
        .saturating_sub(MERGE_RESERVE);
    let child_data_bytes = child_h.space_used.saturating_sub(DATA_AREA_START);
    let space_ok = child_data_bytes <= parent_remaining;

    let combined_slots = u32::from(parent_h.num_slots) + u32::from(child_h.num_slots);
    let slots_ok = combined_slots <= MAX_SLOTS;

    let no_grandchild = child_h.num_ext_blobs == 0;
    let no_tombstones = child_h.tombstone_leaf_cnt == 0;

    Ok(space_ok && slots_ok && no_grandchild && no_tombstones)
}

/// Inline a child blob's subtree back into its parent, replacing
/// the cross-blob `BlobNode` crossing with the child's contents.
///
/// Reads the child via an exclusive guard, deep-clones the child's
/// entry-point subtree into `parent_frame` (preserve-mode — caller
/// should compact the child first if dropping tombstones matters),
/// optionally wraps the cloned root in the `BlobNode`'s inline
/// prefix, frees the parent's `BlobNode` slot, and drops the child
/// blob from the BM. Returns the slot in `parent_frame` where the
/// inlined subtree's root now lives.
///
/// **The caller's responsibility**: rewire the grandparent's
/// pointer to the returned slot. Typical pattern: a recursive
/// merge walker that returns the new slot up the chain so each
/// parent rewires its own child pointer; if `parent_bn_slot` was
/// the parent's `root_slot`, the caller writes the new slot back
/// into the parent's header.
///
/// `is_mergeable(bm, parent_frame, parent_bn_slot)` should return
/// `true` before this is called. Calling without that check risks
/// `OutOfSpace` mid-clone on a too-big merge or wasted work on a
/// merge that violates the no-nested-crossings precondition.
/// `seq` is stamped on the deferred-delete entry the merged
/// child generates. Callers from a user op path should pass the
/// op's WAL seq so the W2D protocol can pair the manifest delete
/// with a real WAL record. Internal callers (compact, the
/// checkpoint round's merge pass) pass
/// [`crate::store::STRUCTURAL_SEQ`] — the merge
/// has no WAL record and shouldn't pin the trim watermark.
pub fn merge_blob(
    bm: &BufferManager,
    parent_frame: &mut BlobFrame<'_>,
    parent_bn_slot: u16,
    seq: u64,
) -> Result<u16> {
    let bn = read_blob_node(parent_frame, parent_bn_slot)?;
    let child_guid = bn.child_blob_guid;
    let plen = (bn.prefix_len as usize).min(BLOB_MAX_INLINE);
    let prefix_bytes: Vec<u8> = bn.bytes[..plen].to_vec();

    let new_subtree_root = {
        let child_pin = bm.pin(child_guid)?;
        let mut child_guard = child_pin.write();
        let child_frame = child_guard.frame();
        let child_root = child_frame.header().root_slot;
        clone_subtree(&child_frame, parent_frame, child_root, false)?
            .expect("preserve mode never returns None")
    };

    let inlined_root = if prefix_bytes.is_empty() {
        new_subtree_root
    } else {
        write_prefix_chain(parent_frame, &prefix_bytes, new_subtree_root)?
    };

    parent_frame.free_node(parent_bn_slot)?;
    // Hand the now-orphaned child blob to the deferred-delete
    // protocol. An inline `bm.delete_blob` here would push the
    // manifest mutation to in-memory before the caller's WAL
    // flush (or, for internal callers like compact / merge_pass,
    // before the next checkpoint round's Sync); on crash that
    // would leave the parent in cache pointing at no child
    // while manifest persistence raced ahead through any
    // unrelated `store.flush`.
    bm.mark_for_delete(child_guid, seq);

    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "holt::engine::merge",
        child_guid = ?&child_guid[..4],
        parent_bn_slot = parent_bn_slot,
        inlined_root = inlined_root,
        "merge_blob: folded child into parent + queued delete",
    );

    Ok(inlined_root)
}

fn read_blob_node(frame: &BlobFrame<'_>, slot: u16) -> Result<BlobNode> {
    let body = frame.body_of_slot(slot).ok_or(Error::node_corrupt(
        "read_blob_node: body resolution failed",
    ))?;
    Ok(*cast::<BlobNode>(body))
}

// ---------- clone primitives ----------

/// Recursively clone the subtree at `src_slot` into `dst`.
///
/// When `filter_tombstones` is false the result is always `Some`
/// — the entire source subtree is copied byte-for-byte. When true,
/// tombstoned leaves are dropped, prefix wrappers over dead
/// children collapse upward, and inner-node arms whose live-child
/// count slips into a smaller `NodeType`'s range re-allocate as
/// the smaller variant. A `None` return means the subtree had no
/// live leaves — caller decides what to substitute (typically
/// `EmptyRoot` at the root, or just "drop this branch" further
/// down).
fn clone_subtree(
    src: &BlobFrame<'_>,
    dst: &mut BlobFrame<'_>,
    src_slot: u16,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let entry = src
        .slot_entry(src_slot)
        .ok_or(Error::node_corrupt("clone_subtree: invalid src slot"))?;
    let ntype = entry
        .node_type()
        .ok_or(Error::node_corrupt("clone_subtree: undecodable src ntype"))?;
    let body = src.body_of_slot(src_slot).ok_or(Error::node_corrupt(
        "clone_subtree: src body resolution failed",
    ))?;

    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "clone_subtree: NodeType::Invalid in source",
        )),
        NodeType::EmptyRoot => {
            let out = dst.alloc_node(NodeType::EmptyRoot)?;
            Ok(Some(out.slot))
        }
        NodeType::Leaf => clone_leaf(src, body, dst, filter_tombstones),
        NodeType::LeafInline => clone_leaf_inline(body, dst, filter_tombstones),
        NodeType::Prefix => clone_prefix(src, body, dst, filter_tombstones),
        NodeType::Node4 => clone_node4(src, body, dst, filter_tombstones),
        NodeType::Node16 => clone_node16(src, body, dst, filter_tombstones),
        NodeType::Node48 => clone_node48(src, body, dst, filter_tombstones),
        NodeType::Node256 => clone_node256(src, body, dst, filter_tombstones),
        NodeType::Blob => clone_blob_node(body, dst),
    }
}

fn clone_leaf(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let src_leaf = *cast::<Leaf>(src_body);
    if filter_tombstones && src_leaf.tombstone != 0 {
        return Ok(None);
    }
    let hdr = src
        .bytes_at(src_leaf.key_offset, 2)
        .ok_or(Error::node_corrupt(
            "clone_leaf: extent header out of range",
        ))?;
    let key_len = u32::from(u16::from_le_bytes([hdr[0], hdr[1]]));
    let ext_total = leaf_extent_size(key_len, u32::from(src_leaf.value_size));
    let src_ext = src
        .bytes_at(src_leaf.key_offset, ext_total)
        .ok_or(Error::node_corrupt("clone_leaf: extent body out of range"))?
        .to_vec();

    let dst_ext = dst.alloc_extent(ext_total)?;
    dst.bytes_at_mut(dst_ext.byte_offset, ext_total)
        .ok_or(Error::node_corrupt("clone_leaf: dst extent out of range"))?
        .copy_from_slice(&src_ext);

    let leaf_out = dst.alloc_node(NodeType::Leaf)?;
    // Preserve tombstone byte in preserve-mode; filter-mode bailed
    // out above so the survivor is always live.
    let new_leaf = if filter_tombstones {
        // Carry the source key fingerprint over — the key bytes are
        // copied verbatim, so the fingerprint is still valid.
        Leaf::live(
            dst_ext.byte_offset,
            src_leaf.value_size,
            src_leaf.seq,
            src_leaf.key_fp,
        )
    } else {
        let mut copy = src_leaf;
        copy.key_offset = dst_ext.byte_offset;
        copy
    };
    write_struct_to_slot(dst, leaf_out.slot, &new_leaf)?;
    Ok(Some(leaf_out.slot))
}

/// Clone a [`LeafInline`]. Unlike [`clone_leaf`] there is no
/// separate key/value extent to copy — the bytes live inline in the
/// 56-byte body — so a straight struct copy reproduces the node
/// verbatim. The tombstone byte is preserved in preserve-mode;
/// filter-mode drops tombstoned survivors.
fn clone_leaf_inline(
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let src_li = *cast::<LeafInline>(src_body);
    if filter_tombstones && src_li.tombstone != 0 {
        return Ok(None);
    }
    let out = dst.alloc_node(NodeType::LeafInline)?;
    write_struct_to_slot(dst, out.slot, &src_li)?;
    Ok(Some(out.slot))
}

fn clone_prefix(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let p = *cast::<Prefix>(src_body);
    let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
    let Some(new_child) = clone_subtree(src, dst, p.child as u16, filter_tombstones)? else {
        return Ok(None);
    };
    let out = dst.alloc_node(NodeType::Prefix)?;
    let new_p = Prefix::new(&p.bytes[..plen], u32::from(new_child));
    write_struct_to_slot(dst, out.slot, &new_p)?;
    Ok(Some(out.slot))
}

fn clone_node4(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let src_n = *cast::<Node4>(src_body);
    let count = (src_n.count as usize).min(4);
    if filter_tombstones {
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(count);
        for i in 0..count {
            if let Some(new_child) = clone_subtree(src, dst, src_n.children[i] as u16, true)? {
                survivors.push((src_n.keys[i], u32::from(new_child)));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 4];
        for (i, slot) in new_children.iter_mut().enumerate().take(count) {
            let cloned = clone_subtree(src, dst, src_n.children[i] as u16, false)?
                .expect("preserve mode never returns None");
            *slot = cloned;
        }
        let out = dst.alloc_node(NodeType::Node4)?;
        let mut new_n = Node4::empty();
        new_n.count = src_n.count;
        new_n.keys = src_n.keys;
        new_n.children = new_children;
        write_struct_to_slot(dst, out.slot, &new_n)?;
        Ok(Some(out.slot))
    }
}

fn clone_node16(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let src_n = *cast::<Node16>(src_body);
    let count = (src_n.count as usize).min(16);
    if filter_tombstones {
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(count);
        for i in 0..count {
            if let Some(new_child) = clone_subtree(src, dst, src_n.children[i] as u16, true)? {
                survivors.push((src_n.keys[i], u32::from(new_child)));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 16];
        for (i, slot) in new_children.iter_mut().enumerate().take(count) {
            let cloned = clone_subtree(src, dst, src_n.children[i] as u16, false)?
                .expect("preserve mode never returns None");
            *slot = cloned;
        }
        let out = dst.alloc_node(NodeType::Node16)?;
        let mut new_n = Node16::empty();
        new_n.count = src_n.count;
        new_n.keys = src_n.keys;
        new_n.children = new_children;
        write_struct_to_slot(dst, out.slot, &new_n)?;
        Ok(Some(out.slot))
    }
}

fn clone_node48(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let src_n = *cast::<Node48>(src_body);
    if filter_tombstones {
        // Iterate bytes 0..256 in order — naturally yields survivors
        // sorted by byte, which `pack_inner_node` requires.
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(48);
        for b in 0..256usize {
            let idx = src_n.index[b];
            if idx == 0 {
                continue;
            }
            let ci = idx as usize - 1;
            if ci >= 48 {
                return Err(Error::node_corrupt("clone_node48: index out of range"));
            }
            if let Some(new_child) = clone_subtree(src, dst, src_n.children[ci] as u16, true)? {
                survivors.push((b as u8, u32::from(new_child)));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 48];
        for (i, slot) in new_children.iter_mut().enumerate() {
            if src_n.children[i] != 0 {
                let cloned = clone_subtree(src, dst, src_n.children[i] as u16, false)?
                    .expect("preserve mode never returns None");
                *slot = cloned;
            }
        }
        let out = dst.alloc_node(NodeType::Node48)?;
        let mut new_n = Node48::empty();
        new_n.count = src_n.count;
        new_n.index = src_n.index;
        new_n.children = new_children;
        write_struct_to_slot(dst, out.slot, &new_n)?;
        Ok(Some(out.slot))
    }
}

fn clone_node256(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
) -> Result<Option<u16>> {
    let src_n = *cast::<Node256>(src_body);
    if filter_tombstones {
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(64);
        for (b, &child_slot) in src_n.children.iter().enumerate() {
            if child_slot == 0 {
                continue;
            }
            if let Some(new_child) = clone_subtree(src, dst, child_slot, true)? {
                survivors.push((b as u8, u32::from(new_child)));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 256];
        for (i, slot) in new_children.iter_mut().enumerate() {
            if src_n.children[i] != 0 {
                let cloned = clone_subtree(src, dst, src_n.children[i] as u16, false)?
                    .expect("preserve mode never returns None");
                *slot = cloned;
            }
        }
        let out = dst.alloc_node(NodeType::Node256)?;
        let mut new_n = Node256::empty();
        new_n.count = src_n.count;
        new_n.children = new_children;
        write_struct_to_slot(dst, out.slot, &new_n)?;
        Ok(Some(out.slot))
    }
}

fn clone_blob_node(src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<Option<u16>> {
    let src_b = *cast::<BlobNode>(src_body);
    let plen = (src_b.prefix_len as usize).min(BLOB_MAX_INLINE);
    let new_b = BlobNode::new(&src_b.bytes[..plen], src_b.child_blob_guid);
    let out = dst.alloc_node(NodeType::Blob)?;
    write_struct_to_slot(dst, out.slot, &new_b)?;
    Ok(Some(out.slot))
}

/// Pack `survivors` into the smallest inner-node variant that fits.
///
/// Used during filter-mode cloning to collapse inner nodes whose
/// live-child count has shrunk into a smaller `NodeType`'s range:
///
/// - 0 children → `None` (drop the branch).
/// - 1 child → `Prefix([byte])` wrapping the child slot; this
///   preserves the descent depth invariant (the parent expected
///   one byte of routing here, and `Prefix` consumes it).
/// - 2–4 → `Node4`; 5–16 → `Node16`; 17–48 → `Node48`; 49+ → `Node256`.
///
/// `survivors` must be byte-sorted ascending — `Node4` / `Node16`
/// store keys in sorted order and their lookup paths break out
/// early on `keys[i] > byte`, so out-of-order entries would corrupt
/// future descents.
fn pack_inner_node(dst: &mut BlobFrame<'_>, survivors: &[(u8, u32)]) -> Result<Option<u16>> {
    debug_assert!(
        survivors.windows(2).all(|w| w[0].0 < w[1].0),
        "pack_inner_node: survivors must be byte-sorted ascending"
    );
    match survivors.len() {
        0 => Ok(None),
        1 => {
            let (byte, child) = survivors[0];
            let slot = write_prefix_chain(dst, &[byte], child as u16)?;
            Ok(Some(slot))
        }
        2..=4 => {
            let out = dst.alloc_node(NodeType::Node4)?;
            let mut n = Node4::empty();
            n.count = survivors.len() as u8;
            for (i, &(b, c)) in survivors.iter().enumerate() {
                n.keys[i] = b;
                n.children[i] = c as u16;
            }
            write_struct_to_slot(dst, out.slot, &n)?;
            Ok(Some(out.slot))
        }
        5..=16 => {
            let out = dst.alloc_node(NodeType::Node16)?;
            let mut n = Node16::empty();
            n.count = survivors.len() as u8;
            for (i, &(b, c)) in survivors.iter().enumerate() {
                n.keys[i] = b;
                n.children[i] = c as u16;
            }
            write_struct_to_slot(dst, out.slot, &n)?;
            Ok(Some(out.slot))
        }
        17..=48 => {
            let out = dst.alloc_node(NodeType::Node48)?;
            let mut n = Node48::empty();
            n.count = survivors.len() as u8;
            for (ci, &(b, c)) in survivors.iter().enumerate() {
                n.children[ci] = c as u16;
                n.index[b as usize] = (ci as u8) + 1;
            }
            write_struct_to_slot(dst, out.slot, &n)?;
            Ok(Some(out.slot))
        }
        _ => {
            let out = dst.alloc_node(NodeType::Node256)?;
            let mut n = Node256::empty();
            // `count: u8` wraps to 0 at 256 children; tolerate that
            // — the lookup path only consults `children[byte]` so
            // the count field is informational.
            n.count = survivors.len() as u8;
            for &(b, c) in survivors {
                n.children[b as usize] = c as u16;
            }
            write_struct_to_slot(dst, out.slot, &n)?;
            Ok(Some(out.slot))
        }
    }
}
