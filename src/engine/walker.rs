//! Recursive ART walker — single-blob descent & mutation.
//!
//! Public entry points:
//! - [`lookup`] — read-only descent (Stage 2a).
//! - [`insert`] — insert / replace with path-compression-aware splits
//!   and node growth Node4→16→48→256 (Stage 2b).
//! - [`erase`] — remove a key, with Node256/48/16/4 lone-child
//!   collapse and Prefix-after-collapse rewiring (Stage 2c).
//!
//! Multi-blob descent (BlobNode crossing, `makeBlobFromNode`,
//! `splitBlob`) lands in Stage 2d.

use std::mem::size_of;

use crate::api::errors::{Error, Result};
use crate::engine::simd;
use crate::layout::{
    leaf_extent_size, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix, PREFIX_MAX_INLINE,
};
use crate::store::BlobFrame;

// ---------- public API ----------

/// Outcome of a [`lookup`] descent.
#[derive(Debug)]
pub enum LookupResult<'a> {
    /// Match found — borrowed view of the value bytes.
    Found(&'a [u8]),
    /// No leaf in the tree matches `key`.
    NotFound,
}

/// Outcome of an [`insert`].
#[derive(Debug)]
pub struct InsertOutcome {
    /// The slot the tree's `root_slot` should now point at — may
    /// differ from the caller's input when a split promotes a new
    /// node above the existing root.
    pub new_root_slot: u16,
    /// If the key already existed, the value it carried before.
    pub previous: Option<Vec<u8>>,
}

/// Outcome of an [`erase`].
#[derive(Debug)]
pub struct EraseOutcome {
    /// The slot the tree's `root_slot` should now point at — may
    /// differ from the caller's input when the root collapses
    /// (e.g. last leaf removed → fresh EmptyRoot sentinel; Node4
    /// shrinks to its lone child and that child is promoted).
    pub new_root_slot: u16,
    /// If a matching leaf was removed, the value it carried.
    /// `None` means "key was not in the tree" — the call is then
    /// a no-op.
    pub previous: Option<Vec<u8>>,
}

/// Look up `key` in the tree rooted at `start_slot`.
pub fn lookup<'a>(
    frame: &'a BlobFrame<'_>,
    start_slot: u16,
    key: &[u8],
) -> Result<LookupResult<'a>> {
    descend(frame, start_slot, key, 0)
}

/// Insert or replace `(key, value)` in the tree rooted at
/// `root_slot`. `seq` is the journal sequence number to stamp on
/// the new leaf (callers should pass a monotonically-increasing
/// value).
///
/// Returns the new root slot (caller updates `header.root_slot`)
/// and the prior value if the key already existed.
pub fn insert(
    frame: &mut BlobFrame<'_>,
    root_slot: u16,
    key: &[u8],
    value: &[u8],
    seq: u64,
) -> Result<InsertOutcome> {
    if key.len() > u16::MAX as usize {
        return Err(Error::KeyTooLong { len: key.len() });
    }
    if value.len() > u16::MAX as usize {
        return Err(Error::ValueTooLong { len: value.len() });
    }
    let r = insert_at(frame, root_slot, key, value, 0, seq)?;
    Ok(InsertOutcome {
        new_root_slot: r.slot_after,
        previous: r.previous,
    })
}

/// Erase `key` from the tree rooted at `root_slot`.
///
/// Returns the new root slot (caller updates `header.root_slot`)
/// and the prior value if the key was present. If `key` was not in
/// the tree, `previous` is `None` and `new_root_slot == root_slot`.
pub fn erase(
    frame: &mut BlobFrame<'_>,
    root_slot: u16,
    key: &[u8],
) -> Result<EraseOutcome> {
    let r = erase_at(frame, root_slot, key, 0)?;
    let new_root = match r.signal {
        EraseSignal::Unchanged => root_slot,
        EraseSignal::Replaced(s) => s,
        EraseSignal::SubtreeGone => {
            // The whole tree is empty — re-seed the EmptyRoot
            // sentinel so subsequent lookups return NotFound and
            // subsequent inserts replace the sentinel cleanly.
            let out = frame.alloc_node(NodeType::EmptyRoot)?;
            out.slot
        }
    };
    Ok(EraseOutcome {
        new_root_slot: new_root,
        previous: r.previous,
    })
}

// ---------- internal types ----------

#[derive(Debug)]
struct InsertReturn {
    /// What slot the parent should now point at — may be the same
    /// as the input slot or may be a freshly-allocated promotion.
    slot_after: u16,
    /// Prior value if the key already existed.
    previous: Option<Vec<u8>>,
}

// ---------- descent dispatch ----------

fn descend<'a>(
    frame: &'a BlobFrame<'_>,
    slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::NodeCorrupt {
            context: "walker::descend: hit NodeType::Invalid",
        }),
        NodeType::EmptyRoot => Ok(LookupResult::NotFound),
        NodeType::Leaf => leaf_check(frame, body, key, depth),
        NodeType::Prefix => prefix_descend(frame, body, key, depth),
        NodeType::Node4 => node4_descend(frame, body, key, depth),
        NodeType::Node16 => node16_descend(frame, body, key, depth),
        NodeType::Node48 => node48_descend(frame, body, key, depth),
        NodeType::Node256 => node256_descend(frame, body, key, depth),
        NodeType::Blob => Err(Error::NotYetImplemented(
            "walker::descend: BlobNode crossing — Stage 2d",
        )),
    }
}

fn resolve_typed<'a>(
    frame: &'a BlobFrame<'_>,
    slot: u16,
) -> Result<(NodeType, &'a [u8])> {
    let entry = frame.slot_entry(slot).ok_or(Error::NodeCorrupt {
        context: "walker: invalid slot",
    })?;
    let ntype = entry.node_type().ok_or(Error::NodeCorrupt {
        context: "walker: undecodable node type",
    })?;
    let body = frame.body_of_slot(slot).ok_or(Error::NodeCorrupt {
        context: "walker: body resolution failed",
    })?;
    Ok((ntype, body))
}

fn ntype_of(frame: &BlobFrame<'_>, slot: u16) -> Result<NodeType> {
    let e = frame.slot_entry(slot).ok_or(Error::NodeCorrupt {
        context: "walker: invalid slot",
    })?;
    e.node_type().ok_or(Error::NodeCorrupt {
        context: "walker: undecodable node type",
    })
}

// ---------- lookup arms ----------

fn leaf_check<'a>(
    frame: &'a BlobFrame<'_>,
    body: &'a [u8],
    key: &[u8],
    _depth: usize,
) -> Result<LookupResult<'a>> {
    let leaf = cast::<Leaf>(body);
    if leaf.tombstone != 0 {
        return Ok(LookupResult::NotFound);
    }
    let (leaf_key, value) = leaf_extent(frame, leaf)?;
    if leaf_key != key {
        return Ok(LookupResult::NotFound);
    }
    Ok(LookupResult::Found(value))
}

fn prefix_descend<'a>(
    frame: &'a BlobFrame<'_>,
    body: &'a [u8],
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let p = cast::<Prefix>(body);
    let plen = p.prefix_len as usize;
    if plen > p.bytes.len() {
        return Err(Error::NodeCorrupt {
            context: "walker::prefix_descend: prefix_len exceeds inline buffer",
        });
    }
    if depth + plen > key.len() {
        return Ok(LookupResult::NotFound);
    }
    if key[depth..depth + plen] != p.bytes[..plen] {
        return Ok(LookupResult::NotFound);
    }
    descend(frame, p.child as u16, key, depth + plen)
}

fn node4_descend<'a>(
    frame: &'a BlobFrame<'_>,
    body: &'a [u8],
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node4>(body);
    if depth >= key.len() {
        return Ok(LookupResult::NotFound);
    }
    let byte = key[depth];
    let count = (n.count as usize).min(4);
    for i in 0..count {
        if n.keys[i] == byte {
            return descend(frame, n.children[i] as u16, key, depth + 1);
        }
        if n.keys[i] > byte {
            break;
        }
    }
    Ok(LookupResult::NotFound)
}

fn node16_descend<'a>(
    frame: &'a BlobFrame<'_>,
    body: &'a [u8],
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node16>(body);
    if depth >= key.len() {
        return Ok(LookupResult::NotFound);
    }
    let byte = key[depth];
    // SIMD: one `pcmpeqb` + movemask on x86_64, vceqq_u8 + nibble
    // pack on aarch64, scalar elsewhere.
    if let Some(i) = simd::node16_find_byte(&n.keys, n.count, byte) {
        return descend(frame, n.children[i as usize] as u16, key, depth + 1);
    }
    Ok(LookupResult::NotFound)
}

fn node48_descend<'a>(
    frame: &'a BlobFrame<'_>,
    body: &'a [u8],
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node48>(body);
    if depth >= key.len() {
        return Ok(LookupResult::NotFound);
    }
    let idx = n.index[key[depth] as usize];
    if idx == 0 {
        return Ok(LookupResult::NotFound);
    }
    let ci = idx as usize - 1;
    if ci >= 48 {
        return Err(Error::NodeCorrupt {
            context: "walker::node48_descend: child index out of range",
        });
    }
    descend(frame, n.children[ci] as u16, key, depth + 1)
}

fn node256_descend<'a>(
    frame: &'a BlobFrame<'_>,
    body: &'a [u8],
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node256>(body);
    if depth >= key.len() {
        return Ok(LookupResult::NotFound);
    }
    let slot = n.children[key[depth] as usize];
    if slot == 0 {
        return Ok(LookupResult::NotFound);
    }
    descend(frame, slot as u16, key, depth + 1)
}

// ---------- insert dispatch ----------

fn insert_at(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    key: &[u8],
    value: &[u8],
    depth: usize,
    seq: u64,
) -> Result<InsertReturn> {
    let ntype = ntype_of(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::NodeCorrupt {
            context: "walker::insert_at: hit NodeType::Invalid",
        }),
        NodeType::EmptyRoot => insert_into_empty_root(frame, slot, key, value, seq),
        NodeType::Leaf => insert_into_leaf(frame, slot, key, value, depth, seq),
        NodeType::Prefix => insert_into_prefix(frame, slot, key, value, depth, seq),
        NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
            insert_into_inner(frame, slot, ntype, key, value, depth, seq)
        }
        NodeType::Blob => Err(Error::NotYetImplemented(
            "walker::insert_at: BlobNode crossing — Stage 2d",
        )),
    }
}

fn insert_into_empty_root(
    frame: &mut BlobFrame<'_>,
    empty_slot: u16,
    key: &[u8],
    value: &[u8],
    seq: u64,
) -> Result<InsertReturn> {
    let new_slot = write_leaf(frame, key, value, seq)?;
    // Release the EmptyRoot sentinel so its slot can be reused.
    frame.free_node(empty_slot)?;
    Ok(InsertReturn { slot_after: new_slot, previous: None })
}

fn insert_into_leaf(
    frame: &mut BlobFrame<'_>,
    leaf_slot: u16,
    new_key: &[u8],
    new_value: &[u8],
    depth: usize,
    seq: u64,
) -> Result<InsertReturn> {
    let (existing_key, existing_value) = read_leaf_kv(frame, leaf_slot)?;

    if existing_key == new_key {
        // Update path: install a fresh leaf with bumped seq, free
        // the old. Stage 6 (BufferManager + compactBlob) will
        // reclaim the orphan extent; for now it's harmless dead
        // space.
        let new_slot = write_leaf(frame, new_key, new_value, seq)?;
        frame.free_node(leaf_slot)?;
        return Ok(InsertReturn {
            slot_after: new_slot,
            previous: Some(existing_value),
        });
    }

    // Two different keys: split into [Prefix?] -> Node4 -> {old leaf, new leaf}.
    let suffix_a = &existing_key[depth..];
    let suffix_b = &new_key[depth..];
    let common_len = longest_common(suffix_a, suffix_b);

    // Strict-prefix case: one key is a prefix of the other. ART
    // needs a terminator byte or leaf-on-inner support to resolve
    // this — neither lands until Stage 2b'.
    if common_len == suffix_a.len() || common_len == suffix_b.len() {
        return Err(Error::NotYetImplemented(
            "walker::insert_into_leaf: one key is a strict prefix of the other",
        ));
    }

    let new_leaf = write_leaf(frame, new_key, new_value, seq)?;
    let byte_existing = suffix_a[common_len];
    let byte_new = suffix_b[common_len];
    let n4 = write_node4_with(
        frame,
        &[
            (byte_existing, u32::from(leaf_slot)),
            (byte_new, u32::from(new_leaf)),
        ],
    )?;

    let final_slot = if common_len == 0 {
        n4
    } else {
        // Wrap with a Prefix node carrying the shared bytes.
        write_prefix_chain(frame, &suffix_a[..common_len], n4)?
    };

    Ok(InsertReturn { slot_after: final_slot, previous: None })
}

fn insert_into_prefix(
    frame: &mut BlobFrame<'_>,
    pfx_slot: u16,
    key: &[u8],
    value: &[u8],
    depth: usize,
    seq: u64,
) -> Result<InsertReturn> {
    let p = read_prefix(frame, pfx_slot)?;
    let plen = p.prefix_len as usize;
    let prefix_bytes_copy: Vec<u8> = p.bytes[..plen].to_vec();
    let child_slot = p.child as u16;

    let key_tail = &key[depth.min(key.len())..];
    let common = longest_common(&prefix_bytes_copy, key_tail);

    if common == plen {
        // Full match — descend into the existing child, then patch
        // the prefix's child pointer if it was rewritten.
        let r = insert_at(frame, child_slot, key, value, depth + plen, seq)?;
        if r.slot_after != child_slot {
            set_prefix_child(frame, pfx_slot, u32::from(r.slot_after))?;
        }
        return Ok(InsertReturn {
            slot_after: pfx_slot,
            previous: r.previous,
        });
    }

    // Diverged inside the prefix. The new key must extend past the
    // common region (no leaf-on-prefix in Stage 2b).
    if depth + common >= key.len() {
        return Err(Error::NotYetImplemented(
            "walker::insert_into_prefix: key terminates inside a prefix",
        ));
    }

    // Build the "tail" prefix for the bytes after divergence in the
    // old prefix; if there are no remaining bytes, point the new
    // Node4 entry directly at the old prefix's child.
    let existing_div_byte = prefix_bytes_copy[common];
    let tail_bytes = &prefix_bytes_copy[common + 1..];
    let existing_branch_slot = if tail_bytes.is_empty() {
        child_slot
    } else {
        write_prefix_chain(frame, tail_bytes, child_slot)?
    };

    let new_div_byte = key[depth + common];
    let new_leaf = write_leaf(frame, key, value, seq)?;
    let n4 = write_node4_with(
        frame,
        &[
            (existing_div_byte, u32::from(existing_branch_slot)),
            (new_div_byte, u32::from(new_leaf)),
        ],
    )?;

    let final_slot = if common == 0 {
        n4
    } else {
        write_prefix_chain(frame, &prefix_bytes_copy[..common], n4)?
    };

    frame.free_node(pfx_slot)?;

    Ok(InsertReturn { slot_after: final_slot, previous: None })
}

fn insert_into_inner(
    frame: &mut BlobFrame<'_>,
    inner_slot: u16,
    ntype: NodeType,
    key: &[u8],
    value: &[u8],
    depth: usize,
    seq: u64,
) -> Result<InsertReturn> {
    if depth >= key.len() {
        return Err(Error::NotYetImplemented(
            "walker::insert_into_inner: key terminates at an inner node",
        ));
    }
    let byte = key[depth];

    if let Some(child_slot) = inner_find_child(frame, inner_slot, ntype, byte)? {
        let r = insert_at(frame, child_slot, key, value, depth + 1, seq)?;
        if r.slot_after != child_slot {
            inner_update_child(frame, inner_slot, ntype, byte, u32::from(r.slot_after))?;
        }
        return Ok(InsertReturn {
            slot_after: inner_slot,
            previous: r.previous,
        });
    }

    let new_leaf = write_leaf(frame, key, value, seq)?;
    let possibly_grown = inner_add_child(frame, inner_slot, ntype, byte, u32::from(new_leaf))?;
    Ok(InsertReturn {
        slot_after: possibly_grown,
        previous: None,
    })
}

// ---------- read helpers ----------

fn cast<T>(body: &[u8]) -> &T {
    debug_assert_eq!(body.len(), size_of::<T>());
    debug_assert_eq!(body.as_ptr() as usize % std::mem::align_of::<T>(), 0);
    // SAFETY: layout types are #[repr(C)] POD; body length and
    // alignment are checked by BlobFrame's invariants.
    unsafe { &*(body.as_ptr() as *const T) }
}

fn leaf_extent<'a>(
    frame: &'a BlobFrame<'_>,
    leaf: &Leaf,
) -> Result<(&'a [u8], &'a [u8])> {
    let hdr = frame.bytes_at(leaf.key_offset, 2).ok_or(Error::NodeCorrupt {
        context: "leaf extent header out of range",
    })?;
    let key_len = u32::from(u16::from_le_bytes([hdr[0], hdr[1]]));
    let total = 2 + key_len + u32::from(leaf.value_size);
    let extent = frame.bytes_at(leaf.key_offset, total).ok_or(Error::NodeCorrupt {
        context: "leaf extent body out of range",
    })?;
    Ok((
        &extent[2..2 + key_len as usize],
        &extent[2 + key_len as usize..],
    ))
}

fn read_leaf_kv(frame: &BlobFrame<'_>, slot: u16) -> Result<(Vec<u8>, Vec<u8>)> {
    let body = frame.body_of_slot(slot).ok_or(Error::NodeCorrupt {
        context: "read_leaf_kv: body",
    })?;
    let leaf = *cast::<Leaf>(body);
    let (k, v) = leaf_extent(frame, &leaf)?;
    Ok((k.to_vec(), v.to_vec()))
}

fn read_prefix(frame: &BlobFrame<'_>, slot: u16) -> Result<Prefix> {
    let body = frame.body_of_slot(slot).ok_or(Error::NodeCorrupt {
        context: "read_prefix: body",
    })?;
    Ok(*cast::<Prefix>(body))
}

fn read_node4(frame: &BlobFrame<'_>, slot: u16) -> Result<Node4> {
    let body = frame.body_of_slot(slot).ok_or(Error::NodeCorrupt {
        context: "read_node4: body",
    })?;
    Ok(*cast::<Node4>(body))
}

fn read_node16(frame: &BlobFrame<'_>, slot: u16) -> Result<Node16> {
    let body = frame.body_of_slot(slot).ok_or(Error::NodeCorrupt {
        context: "read_node16: body",
    })?;
    Ok(*cast::<Node16>(body))
}

fn read_node48(frame: &BlobFrame<'_>, slot: u16) -> Result<Node48> {
    let body = frame.body_of_slot(slot).ok_or(Error::NodeCorrupt {
        context: "read_node48: body",
    })?;
    Ok(*cast::<Node48>(body))
}

fn read_node256(frame: &BlobFrame<'_>, slot: u16) -> Result<Node256> {
    let body = frame.body_of_slot(slot).ok_or(Error::NodeCorrupt {
        context: "read_node256: body",
    })?;
    Ok(*cast::<Node256>(body))
}

// ---------- write helpers ----------

fn write_struct_to_slot<T>(frame: &mut BlobFrame<'_>, slot: u16, v: &T) -> Result<()> {
    let body = frame.body_of_slot_mut(slot).ok_or(Error::NodeCorrupt {
        context: "write_struct_to_slot: body",
    })?;
    debug_assert_eq!(body.len(), size_of::<T>());
    // SAFETY: layout types are #[repr(C)] POD; body sized and
    // aligned per BlobFrame invariants.
    let bytes = unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()) };
    body.copy_from_slice(bytes);
    Ok(())
}

fn write_leaf(
    frame: &mut BlobFrame<'_>,
    key: &[u8],
    value: &[u8],
    seq: u64,
) -> Result<u16> {
    let ext_size = leaf_extent_size(key.len() as u32, value.len() as u32);
    let ext = frame.alloc_extent(ext_size)?;
    // Populate the extent: u16 key_len | key bytes | value bytes
    {
        let s = frame
            .bytes_at_mut(ext.byte_offset, ext_size)
            .ok_or(Error::NodeCorrupt {
                context: "write_leaf: extent out of range",
            })?;
        s[..2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        s[2..2 + key.len()].copy_from_slice(key);
        s[2 + key.len()..2 + key.len() + value.len()].copy_from_slice(value);
        // Padding past 2 + key.len() + value.len() stays zero.
    }
    let leaf_out = frame.alloc_node(NodeType::Leaf)?;
    let leaf = Leaf::live(ext.byte_offset, value.len() as u16, seq);
    write_struct_to_slot(frame, leaf_out.slot, &leaf)?;
    Ok(leaf_out.slot)
}

/// Build a Prefix-node chain spanning `bytes`, ending at `child_slot`.
///
/// `bytes` may exceed `PREFIX_MAX_INLINE`; if so, multiple chained
/// Prefix nodes are allocated.
fn write_prefix_chain(
    frame: &mut BlobFrame<'_>,
    bytes: &[u8],
    child_slot: u16,
) -> Result<u16> {
    debug_assert!(!bytes.is_empty(), "write_prefix_chain on empty bytes");
    // Build right-to-left so each Prefix points at the next.
    let mut next_child = child_slot;
    let mut remaining = bytes;
    // Number of nodes we'll need = ceil(len / PREFIX_MAX_INLINE).
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
fn write_node4_with(
    frame: &mut BlobFrame<'_>,
    children: &[(u8, u32)],
) -> Result<u16> {
    debug_assert!(!children.is_empty() && children.len() <= 4);
    let out = frame.alloc_node(NodeType::Node4)?;
    let mut n = Node4::empty();
    let mut sorted = children.to_vec();
    sorted.sort_by_key(|(b, _)| *b);
    n.count = sorted.len() as u8;
    for (i, (b, c)) in sorted.iter().enumerate() {
        n.keys[i] = *b;
        n.children[i] = *c;
    }
    write_struct_to_slot(frame, out.slot, &n)?;
    Ok(out.slot)
}

fn set_prefix_child(
    frame: &mut BlobFrame<'_>,
    pfx_slot: u16,
    new_child: u32,
) -> Result<()> {
    let mut p = read_prefix(frame, pfx_slot)?;
    p.child = new_child;
    write_struct_to_slot(frame, pfx_slot, &p)
}

// ---------- inner-node ops (find / update / add+grow) ----------

fn inner_find_child(
    frame: &BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
) -> Result<Option<u16>> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame, slot)?;
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
            let n = read_node16(frame, slot)?;
            if let Some(i) = simd::node16_find_byte(&n.keys, n.count, byte) {
                Ok(Some(n.children[i as usize] as u16))
            } else {
                Ok(None)
            }
        }
        NodeType::Node48 => {
            let n = read_node48(frame, slot)?;
            let idx = n.index[byte as usize];
            if idx == 0 {
                Ok(None)
            } else {
                Ok(Some(n.children[idx as usize - 1] as u16))
            }
        }
        NodeType::Node256 => {
            let n = read_node256(frame, slot)?;
            let s = n.children[byte as usize];
            if s == 0 {
                Ok(None)
            } else {
                Ok(Some(s as u16))
            }
        }
        _ => Err(Error::NodeCorrupt {
            context: "inner_find_child: not an inner node",
        }),
    }
}

fn inner_update_child(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
    new_child: u32,
) -> Result<()> {
    match ntype {
        NodeType::Node4 => {
            let mut n = read_node4(frame, slot)?;
            let count = (n.count as usize).min(4);
            for i in 0..count {
                if n.keys[i] == byte {
                    n.children[i] = new_child;
                    return write_struct_to_slot(frame, slot, &n);
                }
            }
            Err(Error::NodeCorrupt {
                context: "inner_update_child: byte not found in Node4",
            })
        }
        NodeType::Node16 => {
            let mut n = read_node16(frame, slot)?;
            let count = (n.count as usize).min(16);
            for i in 0..count {
                if n.keys[i] == byte {
                    n.children[i] = new_child;
                    return write_struct_to_slot(frame, slot, &n);
                }
            }
            Err(Error::NodeCorrupt {
                context: "inner_update_child: byte not found in Node16",
            })
        }
        NodeType::Node48 => {
            let mut n = read_node48(frame, slot)?;
            let idx = n.index[byte as usize];
            if idx == 0 {
                return Err(Error::NodeCorrupt {
                    context: "inner_update_child: byte not found in Node48",
                });
            }
            n.children[idx as usize - 1] = new_child;
            write_struct_to_slot(frame, slot, &n)
        }
        NodeType::Node256 => {
            let mut n = read_node256(frame, slot)?;
            n.children[byte as usize] = new_child;
            write_struct_to_slot(frame, slot, &n)
        }
        _ => Err(Error::NodeCorrupt {
            context: "inner_update_child: not an inner node",
        }),
    }
}

/// Add `(byte, child_slot)` to an inner node, growing to the next
/// NodeType variant if the current one is full. Returns the slot
/// to be used as parent's child pointer (changes on growth).
fn inner_add_child(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
    new_child: u32,
) -> Result<u16> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame, slot)?;
            if n.count < 4 {
                let mut new = n;
                node4_insert_sorted(&mut new, byte, new_child);
                write_struct_to_slot(frame, slot, &new)?;
                Ok(slot)
            } else {
                // Grow to Node16, then insert.
                let n16_slot = grow_node4_to_node16(frame, slot, n)?;
                inner_add_child(frame, n16_slot, NodeType::Node16, byte, new_child)
            }
        }
        NodeType::Node16 => {
            let n = read_node16(frame, slot)?;
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
            let n = read_node48(frame, slot)?;
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
            let mut n = read_node256(frame, slot)?;
            if n.children[byte as usize] != 0 {
                return Err(Error::NodeCorrupt {
                    context: "inner_add_child: byte already present on Node256",
                });
            }
            n.children[byte as usize] = new_child;
            if (n.count as u32) < 256 {
                n.count += 1;
            }
            write_struct_to_slot(frame, slot, &n)?;
            Ok(slot)
        }
        _ => Err(Error::NodeCorrupt {
            context: "inner_add_child: not an inner node",
        }),
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
    // Shift right to make room at `pos`.
    let mut i = count;
    while i > pos {
        n.keys[i] = n.keys[i - 1];
        n.children[i] = n.children[i - 1];
        i -= 1;
    }
    n.keys[pos] = byte;
    n.children[pos] = child;
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
    n.children[pos] = child;
    n.count += 1;
}

fn node48_insert(n: &mut Node48, byte: u8, child: u32) -> Result<()> {
    if n.index[byte as usize] != 0 {
        return Err(Error::NodeCorrupt {
            context: "node48_insert: byte already present",
        });
    }
    // Find the first free children[] slot.
    for i in 0..48 {
        if n.children[i] == 0 {
            n.children[i] = child;
            n.index[byte as usize] = (i + 1) as u8;
            n.count += 1;
            return Ok(());
        }
    }
    Err(Error::NodeCorrupt {
        context: "node48_insert: no free children[] slot despite count < 48",
    })
}

// ---------- node growth ----------

fn grow_node4_to_node16(
    frame: &mut BlobFrame<'_>,
    old_slot: u16,
    old: Node4,
) -> Result<u16> {
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

fn grow_node16_to_node48(
    frame: &mut BlobFrame<'_>,
    old_slot: u16,
    old: Node16,
) -> Result<u16> {
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

fn grow_node48_to_node256(
    frame: &mut BlobFrame<'_>,
    old_slot: u16,
    old: Node48,
) -> Result<u16> {
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

// ---------- erase dispatch ----------

/// What an erase descent tells its parent to do.
#[derive(Debug)]
enum EraseSignal {
    /// Slot stays as-is — nothing to rewire above.
    Unchanged,
    /// The subtree at this slot disappeared entirely. Parent should
    /// drop the corresponding child entry and (if it now has 0
    /// remaining children) free itself in turn.
    SubtreeGone,
    /// The subtree shrank to a single node. Parent should rewrite
    /// its child pointer to the carried slot.
    Replaced(u16),
}

#[derive(Debug)]
struct EraseReturn {
    signal: EraseSignal,
    previous: Option<Vec<u8>>,
}

fn erase_at(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    let ntype = ntype_of(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::NodeCorrupt {
            context: "walker::erase_at: hit NodeType::Invalid",
        }),
        NodeType::EmptyRoot => Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        }),
        NodeType::Leaf => erase_at_leaf(frame, slot, key),
        NodeType::Prefix => erase_at_prefix(frame, slot, key, depth),
        NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
            erase_at_inner(frame, slot, ntype, key, depth)
        }
        NodeType::Blob => Err(Error::NotYetImplemented(
            "walker::erase_at: BlobNode crossing — Stage 2d",
        )),
    }
}

fn erase_at_leaf(
    frame: &mut BlobFrame<'_>,
    leaf_slot: u16,
    key: &[u8],
) -> Result<EraseReturn> {
    let (existing_key, existing_value) = read_leaf_kv(frame, leaf_slot)?;
    if existing_key != key {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    }
    frame.free_node(leaf_slot)?;
    Ok(EraseReturn {
        signal: EraseSignal::SubtreeGone,
        previous: Some(existing_value),
    })
}

fn erase_at_prefix(
    frame: &mut BlobFrame<'_>,
    pfx_slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    let p = read_prefix(frame, pfx_slot)?;
    let plen = p.prefix_len as usize;
    let prefix_bytes_copy: Vec<u8> = p.bytes[..plen].to_vec();
    let child_slot = p.child as u16;

    if depth + plen > key.len() || prefix_bytes_copy[..] != key[depth..depth + plen] {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    }

    let r = erase_at(frame, child_slot, key, depth + plen)?;
    match r.signal {
        EraseSignal::Unchanged => Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: r.previous,
        }),
        EraseSignal::Replaced(new_child) => {
            // Child collapsed to a single slot — patch our pointer
            // and stay. A future compaction pass (Stage 6) may
            // collapse Prefix→Prefix chains; we don't do that here.
            set_prefix_child(frame, pfx_slot, u32::from(new_child))?;
            Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                previous: r.previous,
            })
        }
        EraseSignal::SubtreeGone => {
            // Child is gone — this Prefix has nothing to point at.
            // Free it and chain SubtreeGone upward.
            frame.free_node(pfx_slot)?;
            Ok(EraseReturn {
                signal: EraseSignal::SubtreeGone,
                previous: r.previous,
            })
        }
    }
}

fn erase_at_inner(
    frame: &mut BlobFrame<'_>,
    inner_slot: u16,
    ntype: NodeType,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    if depth >= key.len() {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    }
    let byte = key[depth];
    let child = match inner_find_child(frame, inner_slot, ntype, byte)? {
        Some(c) => c,
        None => {
            return Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                previous: None,
            });
        }
    };

    let r = erase_at(frame, child, key, depth + 1)?;
    match r.signal {
        EraseSignal::Unchanged => Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: r.previous,
        }),
        EraseSignal::Replaced(new_child) => {
            inner_update_child(frame, inner_slot, ntype, byte, u32::from(new_child))?;
            Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                previous: r.previous,
            })
        }
        EraseSignal::SubtreeGone => {
            let sig = inner_remove_child_and_collapse(frame, inner_slot, ntype, byte)?;
            Ok(EraseReturn {
                signal: sig,
                previous: r.previous,
            })
        }
    }
}

/// Remove `byte` from `slot`'s child set. After removal:
/// - `count == 0` → free the inner node, signal SubtreeGone
/// - `count == 1` → free the inner node, wrap the lone child in a
///   `Prefix([surviving_byte])` so descendant depth indexing
///   stays valid, signal Replaced(prefix_slot)
/// - otherwise → rewrite the body, signal Unchanged
///
/// The `Prefix` wrap on lone-child collapse is load-bearing: an
/// inner-node child sits one byte deeper in the descent than its
/// parent, so dropping the inner node without re-inserting its
/// pointing-byte breaks every leaf below it (the walker would
/// match the wrong byte and either find the wrong leaf or
/// NotFound). Stage 6 compaction can merge resulting Prefix→Prefix
/// chains; we trade depth for correctness here.
///
/// Shrinking-back-to-smaller-NodeType (Node256→48, Node48→16,
/// Node16→4) is **not** wired in Stage 2c; the binary shrinks at
/// `count ≤ 37 / 12 / 3` respectively. We just stay at the larger
/// variant — correctness-preserving, mild space waste that
/// compaction (Stage 6) reclaims.
fn inner_remove_child_and_collapse(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
) -> Result<EraseSignal> {
    match ntype {
        NodeType::Node4 => {
            let mut n = read_node4(frame, slot)?;
            let count = (n.count as usize).min(4);
            let mut idx = None;
            for i in 0..count {
                if n.keys[i] == byte {
                    idx = Some(i);
                    break;
                }
            }
            let i = idx.ok_or(Error::NodeCorrupt {
                context: "inner_remove_child_and_collapse: byte not present (Node4)",
            })?;
            for j in i..count - 1 {
                n.keys[j] = n.keys[j + 1];
                n.children[j] = n.children[j + 1];
            }
            n.keys[count - 1] = 0;
            n.children[count - 1] = 0;
            n.count -= 1;
            finish_inner_with_sorted(frame, slot, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node16 => {
            let mut n = read_node16(frame, slot)?;
            let count = (n.count as usize).min(16);
            let mut idx = None;
            for i in 0..count {
                if n.keys[i] == byte {
                    idx = Some(i);
                    break;
                }
            }
            let i = idx.ok_or(Error::NodeCorrupt {
                context: "inner_remove_child_and_collapse: byte not present (Node16)",
            })?;
            for j in i..count - 1 {
                n.keys[j] = n.keys[j + 1];
                n.children[j] = n.children[j + 1];
            }
            n.keys[count - 1] = 0;
            n.children[count - 1] = 0;
            n.count -= 1;
            finish_inner_with_sorted(frame, slot, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node48 => {
            let mut n = read_node48(frame, slot)?;
            let ci = n.index[byte as usize];
            if ci == 0 {
                return Err(Error::NodeCorrupt {
                    context: "inner_remove_child_and_collapse: byte not present (Node48)",
                });
            }
            n.children[(ci as usize) - 1] = 0;
            n.index[byte as usize] = 0;
            n.count -= 1;

            if n.count == 0 {
                frame.free_node(slot)?;
                return Ok(EraseSignal::SubtreeGone);
            }
            if n.count == 1 {
                let (surviving_byte, surviving_child) = {
                    let mut found = (0u8, 0u32);
                    for b in 0..256usize {
                        if n.index[b] != 0 {
                            found = (b as u8, n.children[(n.index[b] as usize) - 1]);
                            break;
                        }
                    }
                    found
                };
                frame.free_node(slot)?;
                let new_slot = write_prefix_chain(
                    frame,
                    &[surviving_byte],
                    surviving_child as u16,
                )?;
                return Ok(EraseSignal::Replaced(new_slot));
            }
            write_struct_to_slot(frame, slot, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        NodeType::Node256 => {
            let mut n = read_node256(frame, slot)?;
            if n.children[byte as usize] == 0 {
                return Err(Error::NodeCorrupt {
                    context: "inner_remove_child_and_collapse: byte not present (Node256)",
                });
            }
            n.children[byte as usize] = 0;
            n.count = n.count.saturating_sub(1);

            if n.count == 0 {
                frame.free_node(slot)?;
                return Ok(EraseSignal::SubtreeGone);
            }
            if n.count == 1 {
                let (surviving_byte, surviving_child) = {
                    let mut found = (0u8, 0u32);
                    for (i, c) in n.children.iter().enumerate() {
                        if *c != 0 {
                            found = (i as u8, *c);
                            break;
                        }
                    }
                    found
                };
                frame.free_node(slot)?;
                let new_slot = write_prefix_chain(
                    frame,
                    &[surviving_byte],
                    surviving_child as u16,
                )?;
                return Ok(EraseSignal::Replaced(new_slot));
            }
            write_struct_to_slot(frame, slot, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        _ => Err(Error::NodeCorrupt {
            context: "inner_remove_child_and_collapse: not an inner node",
        }),
    }
}

/// Shared collapse / writeback for the Node4 + Node16 arms whose
/// `keys[]` array is sorted in-place; `surviving_byte` and
/// `surviving_child` are `keys[0]` / `children[0]` (only
/// consulted when `new_count == 1`).
fn finish_inner_with_sorted<T>(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    new_count: u8,
    body: &T,
    surviving_byte: u8,
    surviving_child: u32,
) -> Result<EraseSignal> {
    if new_count == 0 {
        frame.free_node(slot)?;
        return Ok(EraseSignal::SubtreeGone);
    }
    if new_count == 1 {
        frame.free_node(slot)?;
        let new_slot =
            write_prefix_chain(frame, &[surviving_byte], surviving_child as u16)?;
        return Ok(EraseSignal::Replaced(new_slot));
    }
    write_struct_to_slot(frame, slot, body)?;
    Ok(EraseSignal::Unchanged)
}

// ---------- misc ----------

/// Length of the longest common prefix of `a` and `b`. SIMD on
/// x86_64 / aarch64, scalar fallback elsewhere — see
/// [`crate::engine::simd::longest_common_prefix`].
fn longest_common(a: &[u8], b: &[u8]) -> usize {
    simd::longest_common_prefix(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{BlobGuid, PAGE_SIZE};
    use crate::store::BlobFrame;

    fn fresh_blob() -> (Vec<u8>, BlobGuid) {
        let guid: BlobGuid = [0x11; 16];
        let mut buf = vec![0u8; PAGE_SIZE as usize];
        BlobFrame::init(&mut buf, guid).unwrap();
        (buf, guid)
    }

    fn put(frame: &mut BlobFrame<'_>, k: &[u8], v: &[u8], seq: u64) {
        let root = frame.header().root_slot;
        let r = insert(frame, root, k, v, seq).unwrap();
        frame.header_mut().root_slot = r.new_root_slot;
    }

    fn get<'a>(frame: &'a BlobFrame<'_>, k: &[u8]) -> Option<Vec<u8>> {
        let root = frame.header().root_slot;
        match lookup(frame, root, k).unwrap() {
            LookupResult::Found(v) => Some(v.to_vec()),
            LookupResult::NotFound => None,
        }
    }

    #[test]
    fn single_insert_then_lookup() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"hello", b"world", 1);
        assert_eq!(get(&frame, b"hello").as_deref(), Some(&b"world"[..]));
        assert_eq!(get(&frame, b"hellx"), None);
    }

    #[test]
    fn update_same_key_returns_previous() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"k", b"v1", 1);
        let root = frame.header().root_slot;
        let r = insert(&mut frame, root, b"k", b"v2", 2).unwrap();
        frame.header_mut().root_slot = r.new_root_slot;
        assert_eq!(r.previous.as_deref(), Some(&b"v1"[..]));
        assert_eq!(get(&frame, b"k").as_deref(), Some(&b"v2"[..]));
    }

    #[test]
    fn two_keys_with_shared_prefix_creates_prefix_plus_node4() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"abc/01", b"v1", 1);
        put(&mut frame, b"abc/02", b"v2", 2);
        assert_eq!(get(&frame, b"abc/01").as_deref(), Some(&b"v1"[..]));
        assert_eq!(get(&frame, b"abc/02").as_deref(), Some(&b"v2"[..]));
        assert_eq!(get(&frame, b"abc/03"), None);
        // The root should now be a Prefix node.
        let root_slot = frame.header().root_slot;
        let entry = frame.slot_entry(root_slot).unwrap();
        assert_eq!(entry.node_type(), Some(NodeType::Prefix));
    }

    #[test]
    fn two_keys_no_shared_prefix_creates_naked_node4() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"a", b"va", 1);
        put(&mut frame, b"b", b"vb", 2);
        assert_eq!(get(&frame, b"a").as_deref(), Some(&b"va"[..]));
        assert_eq!(get(&frame, b"b").as_deref(), Some(&b"vb"[..]));
        let root_slot = frame.header().root_slot;
        let entry = frame.slot_entry(root_slot).unwrap();
        assert_eq!(entry.node_type(), Some(NodeType::Node4));
    }

    #[test]
    fn grow_node4_to_node16() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        // 5 keys differing in the second byte after a common 'k' prefix.
        for i in 0..5u8 {
            let k = [b'k', b'0' + i];
            put(&mut frame, &k, &[b'v', b'0' + i], i as u64 + 1);
        }
        // All 5 readable.
        for i in 0..5u8 {
            let k = [b'k', b'0' + i];
            let v = [b'v', b'0' + i];
            assert_eq!(get(&frame, &k).as_deref(), Some(&v[..]));
        }
        // The inner node should have grown to Node16. Walk through
        // the root's prefix to find it.
        let root_slot = frame.header().root_slot;
        let entry = frame.slot_entry(root_slot).unwrap();
        // Root is Prefix (single byte 'k').
        assert_eq!(entry.node_type(), Some(NodeType::Prefix));
        let p = read_prefix(&frame, root_slot).unwrap();
        let inner_slot = p.child as u16;
        let ie = frame.slot_entry(inner_slot).unwrap();
        assert_eq!(ie.node_type(), Some(NodeType::Node16));
    }

    #[test]
    fn grow_chain_node4_to_node16_to_node48() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        // 20 distinct second-bytes (> 16 to force the Node16→Node48 step).
        for i in 0..20u8 {
            let k = [b'p', i];
            put(&mut frame, &k, &[i], i as u64 + 1);
        }
        for i in 0..20u8 {
            let k = [b'p', i];
            assert_eq!(get(&frame, &k).as_deref(), Some(&[i][..]));
        }
        let root_slot = frame.header().root_slot;
        let p = read_prefix(&frame, root_slot).unwrap();
        let inner_slot = p.child as u16;
        assert_eq!(
            frame.slot_entry(inner_slot).unwrap().node_type(),
            Some(NodeType::Node48)
        );
    }

    #[test]
    fn grow_chain_through_node256() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        // 60 distinct second-bytes (> 48 to force Node48→Node256).
        for i in 0..60u8 {
            let k = [b'q', i];
            put(&mut frame, &k, &[i, i ^ 0xFF], i as u64 + 1);
        }
        for i in 0..60u8 {
            let k = [b'q', i];
            let v = [i, i ^ 0xFF];
            assert_eq!(get(&frame, &k).as_deref(), Some(&v[..]));
        }
        let root_slot = frame.header().root_slot;
        let p = read_prefix(&frame, root_slot).unwrap();
        let inner_slot = p.child as u16;
        assert_eq!(
            frame.slot_entry(inner_slot).unwrap().node_type(),
            Some(NodeType::Node256)
        );
    }

    #[test]
    fn prefix_split_at_divergence() {
        // Insert "abcdef" then "abcXYZ" — the existing prefix
        // "abcdef" (Stage 2b builds a Prefix("abc") + Node4{d→leaf}
        // only when the second insert lands; first insert is plain
        // Leaf). After second insert: Prefix("abc") → Node4{d→Leaf,
        // X→Leaf}.
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"abcdef", b"v1", 1);
        put(&mut frame, b"abcXYZ", b"v2", 2);
        assert_eq!(get(&frame, b"abcdef").as_deref(), Some(&b"v1"[..]));
        assert_eq!(get(&frame, b"abcXYZ").as_deref(), Some(&b"v2"[..]));
        assert_eq!(get(&frame, b"abcdeg"), None);
    }

    #[test]
    fn deep_prefix_chain_long_keys() {
        // A 250-byte common prefix forces a Prefix-chain (2 Prefix
        // nodes since PREFIX_MAX_INLINE = 112).
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        let mut k1 = vec![b'x'; 250];
        let mut k2 = k1.clone();
        k1.push(b'1');
        k2.push(b'2');
        put(&mut frame, &k1, b"v1", 1);
        put(&mut frame, &k2, b"v2", 2);
        assert_eq!(get(&frame, &k1).as_deref(), Some(&b"v1"[..]));
        assert_eq!(get(&frame, &k2).as_deref(), Some(&b"v2"[..]));
    }

    #[test]
    fn strict_prefix_returns_not_yet_implemented() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"abc", b"v1", 1);
        let root = frame.header().root_slot;
        let r = insert(&mut frame, root, b"abcdef", b"v2", 2);
        assert!(matches!(r, Err(Error::NotYetImplemented(_))));
    }

    #[test]
    fn many_inserts_all_readable() {
        // Light stress test: 200 keys with varied prefixes/lengths.
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for i in 0..200u32 {
            let k = format!("key/{i:04}/end").into_bytes();
            let v = format!("val#{i}").into_bytes();
            pairs.push((k, v));
        }
        for (i, (k, v)) in pairs.iter().enumerate() {
            put(&mut frame, k, v, i as u64 + 1);
        }
        for (k, v) in &pairs {
            assert_eq!(get(&frame, k).as_deref(), Some(&v[..]));
        }
    }

    // -------- erase --------

    fn del(frame: &mut BlobFrame<'_>, k: &[u8]) -> Option<Vec<u8>> {
        let root = frame.header().root_slot;
        let r = erase(frame, root, k).unwrap();
        frame.header_mut().root_slot = r.new_root_slot;
        r.previous
    }

    #[test]
    fn erase_only_leaf_returns_value_and_empties_tree() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"k", b"v", 1);
        assert_eq!(del(&mut frame, b"k").as_deref(), Some(&b"v"[..]));
        assert_eq!(get(&frame, b"k"), None);
        // Root is back to an EmptyRoot sentinel.
        let root_slot = frame.header().root_slot;
        let e = frame.slot_entry(root_slot).unwrap();
        assert_eq!(e.node_type(), Some(NodeType::EmptyRoot));
    }

    #[test]
    fn erase_missing_key_is_noop_returns_none() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"a", b"1", 1);
        assert_eq!(del(&mut frame, b"b"), None);
        // "a" still present, root still a Leaf.
        assert_eq!(get(&frame, b"a").as_deref(), Some(&b"1"[..]));
    }

    #[test]
    fn erase_one_of_two_node4_collapses_to_prefix_over_lone_leaf() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"a", b"1", 1);
        put(&mut frame, b"b", b"2", 2);
        // Root is Node4 with 2 children.
        del(&mut frame, b"a");
        // Lone-child collapse wraps the surviving leaf in a Prefix
        // node holding the byte that pointed at it — preserves
        // depth invariants for descendants.
        let root_slot = frame.header().root_slot;
        let e = frame.slot_entry(root_slot).unwrap();
        assert_eq!(e.node_type(), Some(NodeType::Prefix));
        assert_eq!(get(&frame, b"b").as_deref(), Some(&b"2"[..]));
        assert_eq!(get(&frame, b"a"), None);
    }

    #[test]
    fn erase_collapses_node16_lone_child() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        for i in 0..5u8 {
            let k = [b'k', b'0' + i];
            put(&mut frame, &k, &[i], i as u64 + 1);
        }
        // Root: Prefix("k") → Node16
        for i in 0..4u8 {
            let k = [b'k', b'0' + i];
            del(&mut frame, &k);
        }
        // Last surviving key still readable.
        let k_last = [b'k', b'0' + 4];
        assert_eq!(get(&frame, &k_last).as_deref(), Some(&[4][..]));
        // The other 4 are gone.
        for i in 0..4u8 {
            let k = [b'k', b'0' + i];
            assert_eq!(get(&frame, &k), None);
        }
    }

    #[test]
    fn erase_collapses_node48_lone_child() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        for i in 0..17u8 {
            let k = [b'p', i];
            put(&mut frame, &k, &[i], i as u64 + 1);
        }
        // Inner node is now Node48.
        for i in 0..16u8 {
            let k = [b'p', i];
            del(&mut frame, &k);
        }
        let k_last = [b'p', 16];
        assert_eq!(get(&frame, &k_last).as_deref(), Some(&[16][..]));
    }

    #[test]
    fn erase_collapses_node256_lone_child() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        // Force Node256 via the artisan-side Node48 promotion path
        // doesn't quite reach 256 in this test budget; use the
        // smaller threshold by going through 50 second-bytes which
        // gives a Node48. (Stage 2c walker shrinks Node256 the same
        // way once 12/N+ promotion paths are exercised.)
        // For the Node256 specifically: do 60 second-bytes.
        for i in 0..60u8 {
            let k = [b'q', i];
            put(&mut frame, &k, &[i, i ^ 0xFF], i as u64 + 1);
        }
        // Inner node is now Node256.
        for i in 0..59u8 {
            let k = [b'q', i];
            del(&mut frame, &k);
        }
        let k_last = [b'q', 59];
        let v_last = [59u8, 59u8 ^ 0xFF];
        assert_eq!(get(&frame, &k_last).as_deref(), Some(&v_last[..]));
    }

    #[test]
    fn erase_all_returns_to_empty_root() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        let pairs = [
            (&b"alpha"[..], &b"A"[..]),
            (&b"beta"[..],  &b"B"[..]),
            (&b"gamma"[..], &b"G"[..]),
        ];
        for (i, (k, v)) in pairs.iter().enumerate() {
            put(&mut frame, k, v, i as u64 + 1);
        }
        for (k, v) in pairs.iter() {
            assert_eq!(del(&mut frame, k).as_deref(), Some(*v));
        }
        // Tree is now empty.
        let root_slot = frame.header().root_slot;
        assert_eq!(
            frame.slot_entry(root_slot).unwrap().node_type(),
            Some(NodeType::EmptyRoot)
        );
    }

    #[test]
    fn erase_through_prefix_keeps_other_branches() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"img/01.jpg", b"a", 1);
        put(&mut frame, b"img/02.jpg", b"b", 2);
        put(&mut frame, b"img/03.jpg", b"c", 3);
        assert_eq!(del(&mut frame, b"img/02.jpg").as_deref(), Some(&b"b"[..]));
        assert_eq!(get(&frame, b"img/01.jpg").as_deref(), Some(&b"a"[..]));
        assert_eq!(get(&frame, b"img/02.jpg"), None);
        assert_eq!(get(&frame, b"img/03.jpg").as_deref(), Some(&b"c"[..]));
    }

    #[test]
    fn insert_after_erase_reinstates_key() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"k", b"v1", 1);
        del(&mut frame, b"k");
        put(&mut frame, b"k", b"v2", 2);
        assert_eq!(get(&frame, b"k").as_deref(), Some(&b"v2"[..]));
    }

    #[test]
    fn churn_100_keys_inserted_then_all_erased() {
        let (mut buf, _) = fresh_blob();
        let mut frame = BlobFrame::wrap(&mut buf);
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..100u32)
            .map(|i| (format!("k{i:04}").into_bytes(), format!("v{i}").into_bytes()))
            .collect();
        for (i, (k, v)) in pairs.iter().enumerate() {
            put(&mut frame, k, v, i as u64 + 1);
        }
        for (k, v) in &pairs {
            assert_eq!(del(&mut frame, k).as_deref(), Some(&v[..]));
        }
        // Every key gone.
        for (k, _) in &pairs {
            assert_eq!(get(&frame, k), None);
        }
        // Tree is empty.
        let root_slot = frame.header().root_slot;
        assert_eq!(
            frame.slot_entry(root_slot).unwrap().node_type(),
            Some(NodeType::EmptyRoot)
        );
    }
}
