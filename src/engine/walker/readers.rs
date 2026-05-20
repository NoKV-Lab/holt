//! Read-side helpers — borrow into a [`BlobFrameRef`] and decode
//! slot bodies or extract leaf extents.
//!
//! Everything here is `pub(super)` so the other walker submodules
//! (lookup / insert / erase / spillover / migrate) can share these
//! decoders. They do **not** mutate the frame; mutation lives in
//! [`super::writers`].

use crate::api::errors::{Error, Result};
use crate::engine::simd;
use crate::layout::{Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix};
use crate::store::BlobFrameRef;

use super::cast;

pub(super) fn resolve_typed(frame: BlobFrameRef<'_>, slot: u16) -> Result<(NodeType, &[u8])> {
    let entry = frame
        .slot_entry(slot)
        .ok_or(Error::node_corrupt("walker: invalid slot"))?;
    let ntype = entry
        .node_type()
        .ok_or(Error::node_corrupt("walker: undecodable node type"))?;
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("walker: body resolution failed"))?;
    Ok((ntype, body))
}

pub(super) fn ntype_of(frame: BlobFrameRef<'_>, slot: u16) -> Result<NodeType> {
    let e = frame
        .slot_entry(slot)
        .ok_or(Error::node_corrupt("walker: invalid slot"))?;
    e.node_type()
        .ok_or(Error::node_corrupt("walker: undecodable node type"))
}

pub(super) fn leaf_extent<'a>(
    frame: BlobFrameRef<'a>,
    leaf: &Leaf,
) -> Result<(&'a [u8], &'a [u8])> {
    let hdr = frame
        .bytes_at(leaf.key_offset, 2)
        .ok_or(Error::node_corrupt("leaf extent header out of range"))?;
    let key_len = u32::from(u16::from_le_bytes([hdr[0], hdr[1]]));
    let total = 2 + key_len + u32::from(leaf.value_size);
    let extent = frame
        .bytes_at(leaf.key_offset, total)
        .ok_or(Error::node_corrupt("leaf extent body out of range"))?;
    Ok((
        &extent[2..2 + key_len as usize],
        &extent[2 + key_len as usize..],
    ))
}

pub(super) fn read_leaf_kv(frame: BlobFrameRef<'_>, slot: u16) -> Result<(Vec<u8>, Vec<u8>)> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_leaf_kv: body"))?;
    let leaf = *cast::<Leaf>(body);
    let (k, v) = leaf_extent(frame, &leaf)?;
    Ok((k.to_vec(), v.to_vec()))
}

/// Like [`read_leaf_kv`] but skips the value clone — the value
/// bytes still live in the blob, but we never copy them into a
/// fresh `Vec`. Used by the blind-put walker path
/// ([`super::insert::insert_multi`] with `wants_prev = false`),
/// which needs only the existing key to decide
/// `update / divergence / extent-grow` and never returns the prior
/// value to the caller.
pub(super) fn read_leaf_key_only(frame: BlobFrameRef<'_>, slot: u16) -> Result<(Vec<u8>, Leaf)> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_leaf_key_only: body"))?;
    let leaf = *cast::<Leaf>(body);
    let (k, _v) = leaf_extent(frame, &leaf)?;
    Ok((k.to_vec(), leaf))
}

pub(super) fn read_prefix(frame: BlobFrameRef<'_>, slot: u16) -> Result<Prefix> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_prefix: body"))?;
    Ok(*cast::<Prefix>(body))
}

pub(super) fn read_node4(frame: BlobFrameRef<'_>, slot: u16) -> Result<Node4> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_node4: body"))?;
    Ok(*cast::<Node4>(body))
}

pub(super) fn read_node16(frame: BlobFrameRef<'_>, slot: u16) -> Result<Node16> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_node16: body"))?;
    Ok(*cast::<Node16>(body))
}

pub(super) fn read_node48(frame: BlobFrameRef<'_>, slot: u16) -> Result<Node48> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_node48: body"))?;
    Ok(*cast::<Node48>(body))
}

pub(super) fn read_node256(frame: BlobFrameRef<'_>, slot: u16) -> Result<Node256> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_node256: body"))?;
    Ok(*cast::<Node256>(body))
}

/// Length of the longest common prefix of `a` and `b`. SIMD on
/// x86_64 / aarch64, scalar fallback elsewhere — see
/// [`crate::engine::simd::longest_common_prefix`].
pub(super) fn longest_common(a: &[u8], b: &[u8]) -> usize {
    simd::longest_common_prefix(a, b)
}
