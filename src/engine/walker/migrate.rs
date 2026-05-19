//! Deep-clone primitives — `make_blob_from_node` (spillover) and
//! `compact_blob` (in-place repack). Share the same recursive
//! `clone_subtree` machinery; both produce a fresh, packed image
//! containing a deep copy of a source subtree.

use crate::api::errors::{Error, Result};
use crate::layout::{
    leaf_extent_size, BlobGuid, BlobNode, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix,
    BLOB_MAX_INLINE, PREFIX_MAX_INLINE,
};
use crate::store::backend::AlignedBlobBuf;
use crate::store::BlobFrame;

use super::cast;
use super::types::{CompactStats, MakeBlobOutcome};
use super::writers::write_struct_to_slot;

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
pub fn make_blob_from_node(
    src_frame: &BlobFrame<'_>,
    src_slot: u16,
    new_guid: BlobGuid,
) -> Result<MakeBlobOutcome> {
    let mut buf = AlignedBlobBuf::zeroed();
    let entry_slot;
    {
        let mut new_frame = BlobFrame::init(buf.as_mut_slice(), new_guid)?;
        entry_slot = clone_subtree(src_frame, &mut new_frame, src_slot)?;

        // Release the EmptyRoot sentinel that `BlobFrame::init`
        // seeded at slot 1; it's unreachable now.
        if new_frame.header().root_slot == 1 && entry_slot != 1 {
            new_frame.free_node(1)?;
        }
        new_frame.header_mut().root_slot = entry_slot;
    }
    Ok(MakeBlobOutcome { buf, entry_slot })
}

/// Repack `buf` in place, discarding all unreachable bytes.
///
/// Builds a fresh `BlobFrame` image in a scratch `AlignedBlobBuf`,
/// deep-clones the live subtree from `buf` into it (via the
/// internal `clone_subtree` helper, shared with
/// [`make_blob_from_node`]), then memcpys the scratch image back
/// over `buf`. This guarantees the
/// resulting blob has:
///
/// - A contiguous packed data area (every byte in
///   `DATA_AREA_START .. space_used` is live)
/// - Empty free lists (no leftover stale slot entries)
/// - `num_slots` equal to the live-subtree node count + 1 (sentinel)
/// - `gap_space` reset to whatever fresh allocations report
/// - The original `blob_guid` preserved
///
/// **What this reclaims:** the leaf key/value extents (allocated
/// via `alloc_extent`, which has no free list) and dead node
/// bodies whose slots returned to a per-NodeType free list but
/// whose NodeType isn't being allocated any more.
///
/// **What this costs:** one scratch `AlignedBlobBuf` (512 KB on
/// the heap, lives for the duration of the call) plus one full
/// blob memcpy at the end. Roughly tens of µs on a modern machine.
pub fn compact_blob(buf: &mut AlignedBlobBuf) -> Result<CompactStats> {
    let (old_space_used, blob_guid, old_root, old_compact_times) = {
        let old_frame = BlobFrame::wrap(buf.as_mut_slice());
        let h = old_frame.header();
        (h.space_used, h.blob_guid, h.root_slot, h.compact_times)
    };

    let mut new_buf = AlignedBlobBuf::zeroed();
    let (new_root, new_space_used) = {
        let mut new_frame = BlobFrame::init(new_buf.as_mut_slice(), blob_guid)?;
        let old_frame = BlobFrame::wrap(buf.as_mut_slice());
        let entry = clone_subtree(&old_frame, &mut new_frame, old_root)?;
        if new_frame.header().root_slot == 1 && entry != 1 {
            new_frame.free_node(1)?;
        }
        new_frame.header_mut().root_slot = entry;
        // Compaction is a fresh-start rebuild — `compact_times`
        // carries the lifetime count of rebuilds for this blob,
        // bumped by one for the round we just performed.
        new_frame.header_mut().compact_times = old_compact_times.saturating_add(1);
        let used = new_frame.header().space_used;
        (entry, used)
    };

    buf.as_mut_slice().copy_from_slice(new_buf.as_slice());

    Ok(CompactStats {
        bytes_before: old_space_used,
        bytes_after: new_space_used,
        bytes_reclaimed: old_space_used.saturating_sub(new_space_used),
        old_root,
        new_root,
    })
}

// ---------- clone primitives ----------

/// Recursively clone the subtree at `src_slot` into `dst`, returning
/// the slot in `dst` corresponding to the migrated subtree root.
///
/// Every NodeType is handled. BlobNode bodies copy verbatim — their
/// `child_blob_guid` / `child_entry_ptr` still reference the same
/// external blob, which is not migrated by this primitive.
fn clone_subtree(src: &BlobFrame<'_>, dst: &mut BlobFrame<'_>, src_slot: u16) -> Result<u16> {
    let entry = src.slot_entry(src_slot).ok_or(Error::NodeCorrupt {
        context: "clone_subtree: invalid src slot",
    })?;
    let ntype = entry.node_type().ok_or(Error::NodeCorrupt {
        context: "clone_subtree: undecodable src ntype",
    })?;
    let body = src.body_of_slot(src_slot).ok_or(Error::NodeCorrupt {
        context: "clone_subtree: src body resolution failed",
    })?;

    match ntype {
        NodeType::Invalid => Err(Error::NodeCorrupt {
            context: "clone_subtree: NodeType::Invalid in source",
        }),
        NodeType::EmptyRoot => {
            let out = dst.alloc_node(NodeType::EmptyRoot)?;
            Ok(out.slot)
        }
        NodeType::Leaf => clone_leaf(src, body, dst),
        NodeType::Prefix => clone_prefix(src, body, dst),
        NodeType::Node4 => clone_node4(src, body, dst),
        NodeType::Node16 => clone_node16(src, body, dst),
        NodeType::Node48 => clone_node48(src, body, dst),
        NodeType::Node256 => clone_node256(src, body, dst),
        NodeType::Blob => clone_blob_node(body, dst),
    }
}

fn clone_leaf(src: &BlobFrame<'_>, src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<u16> {
    let src_leaf = *cast::<Leaf>(src_body);
    let hdr = src
        .bytes_at(src_leaf.key_offset, 2)
        .ok_or(Error::NodeCorrupt {
            context: "clone_leaf: extent header out of range",
        })?;
    let key_len = u32::from(u16::from_le_bytes([hdr[0], hdr[1]]));
    let ext_total = leaf_extent_size(key_len, u32::from(src_leaf.value_size));
    let src_ext = src
        .bytes_at(src_leaf.key_offset, ext_total)
        .ok_or(Error::NodeCorrupt {
            context: "clone_leaf: extent body out of range",
        })?
        .to_vec();

    let dst_ext = dst.alloc_extent(ext_total)?;
    dst.bytes_at_mut(dst_ext.byte_offset, ext_total)
        .ok_or(Error::NodeCorrupt {
            context: "clone_leaf: dst extent out of range",
        })?
        .copy_from_slice(&src_ext);

    let leaf_out = dst.alloc_node(NodeType::Leaf)?;
    let new_leaf = Leaf::live(dst_ext.byte_offset, src_leaf.value_size, src_leaf.seq);
    write_struct_to_slot(dst, leaf_out.slot, &new_leaf)?;
    Ok(leaf_out.slot)
}

fn clone_prefix(src: &BlobFrame<'_>, src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<u16> {
    let p = *cast::<Prefix>(src_body);
    let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
    let new_child = clone_subtree(src, dst, p.child as u16)?;
    let out = dst.alloc_node(NodeType::Prefix)?;
    let new_p = Prefix::new(&p.bytes[..plen], u32::from(new_child));
    write_struct_to_slot(dst, out.slot, &new_p)?;
    Ok(out.slot)
}

fn clone_node4(src: &BlobFrame<'_>, src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<u16> {
    let src_n = *cast::<Node4>(src_body);
    let count = (src_n.count as usize).min(4);
    let mut new_children = [0u32; 4];
    for (i, slot) in new_children.iter_mut().enumerate().take(count) {
        let cloned = clone_subtree(src, dst, src_n.children[i] as u16)?;
        *slot = u32::from(cloned);
    }
    let out = dst.alloc_node(NodeType::Node4)?;
    let mut new_n = Node4::empty();
    new_n.count = src_n.count;
    new_n.keys = src_n.keys;
    new_n.children = new_children;
    write_struct_to_slot(dst, out.slot, &new_n)?;
    Ok(out.slot)
}

fn clone_node16(src: &BlobFrame<'_>, src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<u16> {
    let src_n = *cast::<Node16>(src_body);
    let count = (src_n.count as usize).min(16);
    let mut new_children = [0u32; 16];
    for (i, slot) in new_children.iter_mut().enumerate().take(count) {
        let cloned = clone_subtree(src, dst, src_n.children[i] as u16)?;
        *slot = u32::from(cloned);
    }
    let out = dst.alloc_node(NodeType::Node16)?;
    let mut new_n = Node16::empty();
    new_n.count = src_n.count;
    new_n.keys = src_n.keys;
    new_n.children = new_children;
    write_struct_to_slot(dst, out.slot, &new_n)?;
    Ok(out.slot)
}

fn clone_node48(src: &BlobFrame<'_>, src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<u16> {
    let src_n = *cast::<Node48>(src_body);
    let mut new_children = [0u32; 48];
    for (i, slot) in new_children.iter_mut().enumerate() {
        if src_n.children[i] != 0 {
            let cloned = clone_subtree(src, dst, src_n.children[i] as u16)?;
            *slot = u32::from(cloned);
        }
    }
    let out = dst.alloc_node(NodeType::Node48)?;
    let mut new_n = Node48::empty();
    new_n.count = src_n.count;
    new_n.index = src_n.index;
    new_n.children = new_children;
    write_struct_to_slot(dst, out.slot, &new_n)?;
    Ok(out.slot)
}

fn clone_node256(src: &BlobFrame<'_>, src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<u16> {
    let src_n = *cast::<Node256>(src_body);
    let mut new_children = [0u32; 256];
    for (i, slot) in new_children.iter_mut().enumerate() {
        if src_n.children[i] != 0 {
            let cloned = clone_subtree(src, dst, src_n.children[i] as u16)?;
            *slot = u32::from(cloned);
        }
    }
    let out = dst.alloc_node(NodeType::Node256)?;
    let mut new_n = Node256::empty();
    new_n.count = src_n.count;
    new_n.children = new_children;
    write_struct_to_slot(dst, out.slot, &new_n)?;
    Ok(out.slot)
}

fn clone_blob_node(src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<u16> {
    let src_b = *cast::<BlobNode>(src_body);
    let plen = (src_b.prefix_len as usize).min(BLOB_MAX_INLINE);
    let new_b = BlobNode::new(
        &src_b.bytes[..plen],
        src_b.child_blob_guid,
        src_b.child_entry_ptr,
    );
    let out = dst.alloc_node(NodeType::Blob)?;
    write_struct_to_slot(dst, out.slot, &new_b)?;
    Ok(out.slot)
}
