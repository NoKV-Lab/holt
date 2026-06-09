//! Write-side helpers — allocate fresh slots / extents in a
//! [`BlobFrame`] and populate them with node bodies.
//!
//! Everything here is `pub(super)`, scoped to walker submodules.
//! Pure mutation; no recursion into the walker's descent logic.

use std::mem::{offset_of, size_of};

use crate::api::errors::{Error, Result};
use crate::engine::simd;
use crate::layout::{
    fits_leaf_inline, leaf_extent_size, BlobGuid, BlobNode, Leaf, LeafInline, Node16, Node256,
    Node4, Node48, NodeType, Prefix, LEAF_INLINE_CAP, PREFIX_MAX_INLINE,
};
use crate::store::BlobFrame;

use super::cast;
use super::readers::{
    read_node16, read_node256, read_node256_child, read_node4, read_node48, read_prefix,
};
use super::SearchKey;

pub(super) fn write_struct_to_slot<T>(frame: &mut BlobFrame<'_>, slot: u16, v: &T) -> Result<()> {
    {
        let body = frame
            .body_of_slot_mut(slot)
            .ok_or(Error::node_corrupt("write_struct_to_slot: body"))?;
        debug_assert_eq!(body.len(), size_of::<T>());
        // SAFETY: layout types are #[repr(C)] POD; body sized and
        // aligned per BlobFrame invariants.
        let bytes = unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref::<T>(v).cast::<u8>(), size_of::<T>())
        };
        body.copy_from_slice(bytes);
    }
    Ok(())
}

/// Repoint an existing `BlobNode`'s child GUID. Used by copy-on-write
/// forking to redirect a parent frame's crossing at `slot` from the
/// shared child frame to its freshly-installed private fork. The
/// parent frame must be exclusively latched by the caller.
pub(super) fn repoint_blob_node(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    new_child: BlobGuid,
) -> Result<()> {
    let mut bn = *cast::<BlobNode>(
        frame
            .body_of_slot(slot)
            .ok_or(Error::node_corrupt("repoint_blob_node: body"))?,
    );
    bn.child_blob_guid = new_child;
    write_struct_to_slot(frame, slot, &bn)
}

pub(super) fn write_leaf(
    frame: &mut BlobFrame<'_>,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
) -> Result<u16> {
    let key_len = key.len();
    // Small records inline key+value into a single 56-byte node,
    // skipping the extent allocation and the second cache miss.
    if fits_leaf_inline(key_len, value.len()) {
        let mut keybuf = [0u8; LEAF_INLINE_CAP];
        key.write_to_slice(&mut keybuf[..key_len]);
        let li = LeafInline::live(&keybuf[..key_len], value, seq);
        let out = frame.alloc_node(NodeType::LeafInline)?;
        write_struct_to_slot(frame, out.slot, &li)?;
        return Ok(out.slot);
    }
    let ext_size = leaf_extent_size(key_len as u32, value.len() as u32);
    let ext = frame.alloc_extent(ext_size)?;
    {
        let s = frame
            .bytes_at_mut(ext.byte_offset, ext_size)
            .ok_or(Error::node_corrupt("write_leaf: extent out of range"))?;
        s[..2].copy_from_slice(&(key_len as u16).to_le_bytes());
        key.write_to_slice(&mut s[2..2 + key_len]);
        s[2 + key_len..2 + key_len + value.len()].copy_from_slice(value);
    }
    let leaf_out = frame.alloc_node(NodeType::Leaf)?;
    let leaf = Leaf::live(ext.byte_offset, value.len() as u16, seq, key.fingerprint());
    write_struct_to_slot(frame, leaf_out.slot, &leaf)?;
    Ok(leaf_out.slot)
}

pub(super) fn write_leaf_seq(frame: &mut BlobFrame<'_>, slot: u16, seq: u64) -> Result<()> {
    let body = frame
        .body_of_slot_mut(slot)
        .ok_or(Error::node_corrupt("write_leaf_seq: body"))?;
    if body.len() != size_of::<Leaf>() {
        return Err(Error::node_corrupt("write_leaf_seq: non-leaf slot"));
    }
    let seq_off = offset_of!(Leaf, seq);
    body[seq_off..seq_off + size_of::<u64>()].copy_from_slice(&seq.to_le_bytes());
    Ok(())
}

/// Build a Prefix-node chain spanning `bytes`, ending at
/// `child_slot`. `bytes` may exceed `PREFIX_MAX_INLINE`; if so,
/// multiple chained Prefix nodes are allocated.
pub(super) fn write_prefix_chain(
    frame: &mut BlobFrame<'_>,
    bytes: &[u8],
    child_slot: u16,
) -> Result<u16> {
    debug_assert!(!bytes.is_empty(), "write_prefix_chain on empty bytes");
    let mut next_child = child_slot;
    let mut remaining = bytes;
    let mut head = 0u16;
    while !remaining.is_empty() {
        let chunk_len = remaining.len().min(PREFIX_MAX_INLINE);
        let chunk_start = remaining.len() - chunk_len;
        let chunk = &remaining[chunk_start..];
        let out = frame.alloc_node(NodeType::Prefix)?;
        let p = Prefix::new(chunk, u32::from(next_child));
        write_struct_to_slot(frame, out.slot, &p)?;
        next_child = out.slot;
        head = out.slot;
        remaining = &remaining[..chunk_start];
    }
    Ok(head)
}

/// Build a fresh Node4 with the given `(byte, child_slot)` pairs.
/// Keys are sorted ascending inside the Node4.
pub(super) fn write_node4_with(frame: &mut BlobFrame<'_>, children: &[(u8, u32)]) -> Result<u16> {
    debug_assert!(!children.is_empty() && children.len() <= 4);
    let out = frame.alloc_node(NodeType::Node4)?;
    let mut n = Node4::empty();
    let mut sorted = children.to_vec();
    sorted.sort_by_key(|(b, _)| *b);
    n.count = sorted.len() as u8;
    for (i, (b, c)) in sorted.iter().enumerate() {
        n.keys[i] = *b;
        n.children[i] = *c as u16;
    }
    write_struct_to_slot(frame, out.slot, &n)?;
    Ok(out.slot)
}

pub(super) fn set_prefix_child(
    frame: &mut BlobFrame<'_>,
    pfx_slot: u16,
    new_child: u32,
) -> Result<()> {
    let mut p = read_prefix(frame.as_ref(), pfx_slot)?;
    p.child = new_child;
    write_struct_to_slot(frame, pfx_slot, &p)
}

// ---------- inner-node ops (find / update / add+grow) ----------

pub(super) fn inner_find_child(
    frame: &BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
) -> Result<Option<u16>> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame.as_ref(), slot)?;
            let count = (n.count as usize).min(4);
            for i in 0..count {
                if n.keys[i] == byte {
                    return Ok(Some(n.children[i] as u16));
                }
                if n.keys[i] > byte {
                    break;
                }
            }
            Ok(None)
        }
        NodeType::Node16 => {
            let n = read_node16(frame.as_ref(), slot)?;
            if let Some(i) = simd::node16_find_byte(&n.keys, n.count, byte) {
                Ok(Some(n.children[i as usize] as u16))
            } else {
                Ok(None)
            }
        }
        NodeType::Node48 => {
            let n = read_node48(frame.as_ref(), slot)?;
            let idx = n.index[byte as usize];
            if idx == 0 {
                Ok(None)
            } else {
                Ok(Some(n.children[idx as usize - 1] as u16))
            }
        }
        NodeType::Node256 => {
            let s = read_node256_child(frame.as_ref(), slot, byte)?;
            if s == 0 {
                Ok(None)
            } else {
                Ok(Some(s as u16))
            }
        }
        _ => Err(Error::node_corrupt("inner_find_child: not an inner node")),
    }
}

pub(super) fn inner_update_child(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
    new_child: u32,
) -> Result<()> {
    match ntype {
        NodeType::Node4 => {
            let mut n = read_node4(frame.as_ref(), slot)?;
            let count = (n.count as usize).min(4);
            for i in 0..count {
                if n.keys[i] == byte {
                    n.children[i] = new_child as u16;
                    return write_struct_to_slot(frame, slot, &n);
                }
            }
            Err(Error::node_corrupt(
                "inner_update_child: byte not found in Node4",
            ))
        }
        NodeType::Node16 => {
            let mut n = read_node16(frame.as_ref(), slot)?;
            if let Some(i) = simd::node16_find_byte(&n.keys, n.count, byte) {
                n.children[i as usize] = new_child as u16;
                return write_struct_to_slot(frame, slot, &n);
            }
            Err(Error::node_corrupt(
                "inner_update_child: byte not found in Node16",
            ))
        }
        NodeType::Node48 => {
            let mut n = read_node48(frame.as_ref(), slot)?;
            let idx = n.index[byte as usize];
            if idx == 0 {
                return Err(Error::node_corrupt(
                    "inner_update_child: byte not found in Node48",
                ));
            }
            n.children[idx as usize - 1] = new_child as u16;
            write_struct_to_slot(frame, slot, &n)
        }
        NodeType::Node256 => {
            let mut n = read_node256(frame.as_ref(), slot)?;
            n.children[byte as usize] = new_child as u16;
            write_struct_to_slot(frame, slot, &n)
        }
        _ => Err(Error::node_corrupt("inner_update_child: not an inner node")),
    }
}

/// Add `(byte, child_slot)` to an inner node, growing to the next
/// NodeType variant if the current one is full. Returns the slot
/// to be used as parent's child pointer (changes on growth).
pub(super) fn inner_add_child(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
    new_child: u32,
) -> Result<u16> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame.as_ref(), slot)?;
            if n.count < 4 {
                let mut new = n;
                node4_insert_sorted(&mut new, byte, new_child);
                write_struct_to_slot(frame, slot, &new)?;
                Ok(slot)
            } else {
                let n16_slot = grow_node4_to_node16(frame, slot, n)?;
                inner_add_child(frame, n16_slot, NodeType::Node16, byte, new_child)
            }
        }
        NodeType::Node16 => {
            let n = read_node16(frame.as_ref(), slot)?;
            if n.count < 16 {
                let mut new = n;
                node16_insert_sorted(&mut new, byte, new_child);
                write_struct_to_slot(frame, slot, &new)?;
                Ok(slot)
            } else {
                let n48_slot = grow_node16_to_node48(frame, slot, n)?;
                inner_add_child(frame, n48_slot, NodeType::Node48, byte, new_child)
            }
        }
        NodeType::Node48 => {
            let n = read_node48(frame.as_ref(), slot)?;
            if n.count < 48 {
                let mut new = n;
                node48_insert(&mut new, byte, new_child)?;
                write_struct_to_slot(frame, slot, &new)?;
                Ok(slot)
            } else {
                let n256_slot = grow_node48_to_node256(frame, slot, n)?;
                inner_add_child(frame, n256_slot, NodeType::Node256, byte, new_child)
            }
        }
        NodeType::Node256 => {
            let mut n = read_node256(frame.as_ref(), slot)?;
            if n.children[byte as usize] != 0 {
                return Err(Error::node_corrupt(
                    "inner_add_child: byte already present on Node256",
                ));
            }
            n.children[byte as usize] = new_child as u16;
            // Node256 capacity is 256 but `count` is u8 (max 255).
            // Saturate at 255 — the bit-set in `children[]` is the
            // authoritative population check; `count` is a stat
            // used by spillover / shrink heuristics.
            n.count = n.count.saturating_add(1);
            write_struct_to_slot(frame, slot, &n)?;
            Ok(slot)
        }
        _ => Err(Error::node_corrupt("inner_add_child: not an inner node")),
    }
}

fn node4_insert_sorted(n: &mut Node4, byte: u8, child: u32) {
    let count = n.count as usize;
    debug_assert!(count < 4);
    let mut pos = count;
    for i in 0..count {
        if n.keys[i] > byte {
            pos = i;
            break;
        }
    }
    let mut i = count;
    while i > pos {
        n.keys[i] = n.keys[i - 1];
        n.children[i] = n.children[i - 1];
        i -= 1;
    }
    n.keys[pos] = byte;
    n.children[pos] = child as u16;
    n.count += 1;
}

fn node16_insert_sorted(n: &mut Node16, byte: u8, child: u32) {
    let count = n.count as usize;
    debug_assert!(count < 16);
    let mut pos = count;
    for i in 0..count {
        if n.keys[i] > byte {
            pos = i;
            break;
        }
    }
    let mut i = count;
    while i > pos {
        n.keys[i] = n.keys[i - 1];
        n.children[i] = n.children[i - 1];
        i -= 1;
    }
    n.keys[pos] = byte;
    n.children[pos] = child as u16;
    n.count += 1;
}

fn node48_insert(n: &mut Node48, byte: u8, child: u32) -> Result<()> {
    if n.index[byte as usize] != 0 {
        return Err(Error::node_corrupt("node48_insert: byte already present"));
    }
    for i in 0..48 {
        if n.children[i] == 0 {
            n.children[i] = child as u16;
            n.index[byte as usize] = (i + 1) as u8;
            n.count += 1;
            return Ok(());
        }
    }
    Err(Error::node_corrupt(
        "node48_insert: no free children[] slot despite count < 48",
    ))
}

// ---------- node growth ----------

fn grow_node4_to_node16(frame: &mut BlobFrame<'_>, old_slot: u16, old: Node4) -> Result<u16> {
    let out = frame.alloc_node(NodeType::Node16)?;
    let mut n = Node16::empty();
    n.count = old.count;
    for i in 0..old.count as usize {
        n.keys[i] = old.keys[i];
        n.children[i] = old.children[i];
    }
    write_struct_to_slot(frame, out.slot, &n)?;
    frame.free_node(old_slot)?;
    Ok(out.slot)
}

fn grow_node16_to_node48(frame: &mut BlobFrame<'_>, old_slot: u16, old: Node16) -> Result<u16> {
    let out = frame.alloc_node(NodeType::Node48)?;
    let mut n = Node48::empty();
    n.count = old.count;
    for i in 0..old.count as usize {
        n.children[i] = old.children[i];
        n.index[old.keys[i] as usize] = (i + 1) as u8;
    }
    write_struct_to_slot(frame, out.slot, &n)?;
    frame.free_node(old_slot)?;
    Ok(out.slot)
}

fn grow_node48_to_node256(frame: &mut BlobFrame<'_>, old_slot: u16, old: Node48) -> Result<u16> {
    let out = frame.alloc_node(NodeType::Node256)?;
    let mut n = Node256::empty();
    let mut count = 0u16;
    for byte in 0..256usize {
        let idx = old.index[byte];
        if idx != 0 {
            n.children[byte] = old.children[idx as usize - 1];
            count += 1;
        }
    }
    n.count = count.min(255) as u8;
    write_struct_to_slot(frame, out.slot, &n)?;
    frame.free_node(old_slot)?;
    Ok(out.slot)
}

// ---------- node shrink (erase-time, hysteresis-aware) ----------

/// Threshold below which a `Node16` shrinks to `Node4`. The
/// `Node4` capacity is 4; we shrink at `count ≤ 3` so the next
/// insert at this byte position doesn't immediately re-grow.
pub(super) const SHRINK_NODE16_TO_NODE4_AT: u8 = 3;

/// Threshold below which a `Node48` shrinks to `Node16`. The
/// `Node16` capacity is 16; we shrink at `count ≤ 12` for
/// hysteresis vs the grow threshold.
pub(super) const SHRINK_NODE48_TO_NODE16_AT: u8 = 12;

/// Threshold below which a `Node256` shrinks to `Node48`. The
/// `Node48` capacity is 48; we shrink at `count ≤ 37` for
/// hysteresis vs the grow threshold.
pub(super) const SHRINK_NODE256_TO_NODE48_AT: u8 = 37;

pub(super) fn shrink_node16_to_node4(
    frame: &mut BlobFrame<'_>,
    old_slot: u16,
    old: Node16,
) -> Result<u16> {
    debug_assert!(old.count <= 4, "shrink target must fit in Node4");
    let out = frame.alloc_node(NodeType::Node4)?;
    let mut n = Node4::empty();
    n.count = old.count;
    for i in 0..old.count as usize {
        n.keys[i] = old.keys[i];
        n.children[i] = old.children[i];
    }
    write_struct_to_slot(frame, out.slot, &n)?;
    frame.free_node(old_slot)?;
    Ok(out.slot)
}

pub(super) fn shrink_node48_to_node16(
    frame: &mut BlobFrame<'_>,
    old_slot: u16,
    old: Node48,
) -> Result<u16> {
    debug_assert!(old.count <= 16, "shrink target must fit in Node16");
    let out = frame.alloc_node(NodeType::Node16)?;
    let mut n = Node16::empty();
    n.count = old.count;
    let mut i = 0usize;
    let mut byte = 0usize;
    while let Some(next_byte) = simd::find_next_nonzero_byte(&old.index, byte) {
        byte = next_byte + 1;
        let idx = old.index[next_byte];
        n.keys[i] = next_byte as u8;
        n.children[i] = old.children[idx as usize - 1];
        i += 1;
    }
    debug_assert_eq!(i, old.count as usize);
    write_struct_to_slot(frame, out.slot, &n)?;
    frame.free_node(old_slot)?;
    Ok(out.slot)
}

pub(super) fn shrink_node256_to_node48(
    frame: &mut BlobFrame<'_>,
    old_slot: u16,
    old: Node256,
) -> Result<u16> {
    debug_assert!(old.count <= 48, "shrink target must fit in Node48");
    let out = frame.alloc_node(NodeType::Node48)?;
    let mut n = Node48::empty();
    n.count = old.count;
    // Pack the populated children into the first `count` slots
    // of the destination `children[]`; rewrite `index[byte]` to
    // point at the new packed position.
    let mut packed = 0usize;
    let mut byte = 0usize;
    while let Some(next_byte) = simd::find_next_nonzero_u16(&old.children, byte) {
        byte = next_byte + 1;
        let child = old.children[next_byte];
        n.children[packed] = child;
        n.index[next_byte] = (packed + 1) as u8;
        packed += 1;
    }
    debug_assert_eq!(packed, old.count as usize);
    write_struct_to_slot(frame, out.slot, &n)?;
    frame.free_node(old_slot)?;
    Ok(out.slot)
}

/// Shared collapse / writeback for the Node4 + Node16 arms whose
/// `keys[]` array is sorted in-place; `surviving_byte` and
/// `surviving_child` are `keys[0]` / `children[0]` (only consulted
/// when `new_count == 1`).
pub(super) fn finish_inner_with_sorted<T>(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    new_count: u8,
    body: &T,
    surviving_byte: u8,
    surviving_child: u16,
) -> Result<super::types::EraseSignal> {
    use super::types::EraseSignal;
    if new_count == 0 {
        frame.free_node(slot)?;
        return Ok(EraseSignal::SubtreeGone);
    }
    if new_count == 1 {
        frame.free_node(slot)?;
        let new_slot = write_prefix_chain(frame, &[surviving_byte], surviving_child)?;
        return Ok(EraseSignal::Replaced(new_slot));
    }
    write_struct_to_slot(frame, slot, body)?;
    Ok(EraseSignal::Unchanged)
}
