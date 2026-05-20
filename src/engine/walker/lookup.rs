//! Read-path descent — `lookup` / `lookup_at` / `lookup_multi_with`.
//!
//! All entry points take a [`BlobFrameRef`] (or a
//! [`BufferManager`] for the multi-blob variant) so the walker
//! borrows into the cached buffer with **zero memcpy**.

use crate::api::errors::{Error, Result};
use crate::engine::simd;
use crate::layout::{
    BlobNode, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix, BLOB_MAX_INLINE,
};
use std::sync::Arc;

use crate::store::{BlobFrameRef, BufferManager, CachedBlob};

use super::cast;
use super::readers::{leaf_extent, resolve_typed};
use super::types::{BlobNodeCrossing, LookupResult};

/// Look up `key` in the tree rooted at `start_slot` (depth 0).
///
/// Takes a [`BlobFrameRef`] so the read path can run against a
/// shared buffer (e.g. a `BufferManager` read-guard) with no
/// copies. Returned borrows are tied to the lifetime of that
/// underlying buffer.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn lookup<'a>(
    frame: BlobFrameRef<'a>,
    start_slot: u16,
    key: &[u8],
) -> Result<LookupResult<'a>> {
    descend(frame, start_slot, key, 0)
}

/// Continue a lookup at `start_slot` with a non-zero `depth` — used
/// by callers driving cross-blob descent through
/// [`LookupResult::Crossing`].
pub(super) fn lookup_at<'a>(
    frame: BlobFrameRef<'a>,
    start_slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    descend(frame, start_slot, key, depth)
}

/// Multi-blob lookup — wait-free in the common case.
///
/// Walks every blob via [`crate::store::CachedBlob::read_optimistic`]: snapshot
/// the latch version, read raw bytes, then `validate()` after the
/// hop. If a writer lapped the snapshot mid-walk the hop is
/// discarded and the entire lookup restarts from the root.
///
/// Why restart from the root: a writer who modifies any blob may
/// also have moved the `BlobNode` crossing that pointed there, so
/// the parent-side path is stale too. Restarting catches the
/// new tree shape from the top.
///
/// On match `consume` is invoked on the live cache-pin slice and
/// its return value is wrapped into `Some(_)`; on `NotFound`
/// returns `Ok(None)`. The closure runs after the optimistic
/// `validate()` succeeds — same race contract as the v0.2 owned
/// variant (`|v| v.to_vec()` recreates it byte-for-byte). Keep
/// the closure short: it borrows directly into the cache buffer
/// and a slow closure widens the optimistic race window.
///
/// `F: FnMut` rather than `FnOnce` so the restart loop can refer
/// to the same closure across multiple iterations — the closure
/// is invoked at most once per successful lookup (no restart);
/// callers can treat the bound as effectively `FnOnce` for
/// reasoning purposes.
pub fn lookup_multi_with<R, F>(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    key: &[u8],
    mut consume: F,
) -> Result<Option<R>>
where
    F: FnMut(&[u8]) -> R,
{
    // Outer loop: each iteration is one full attempt; we restart
    // here when an optimistic snapshot is invalidated.
    'restart: loop {
        // Hop 0: the cached root blob — `Tree` keeps this pinned
        // for its lifetime so we skip BM's pin-Mutex on the
        // common case where every op starts at the root.
        let crossing = {
            let guard = root_pin.read_optimistic();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let root_slot = frame.header().root_slot;
            let result = lookup_at(frame, root_slot, key, 0);

            // Validate AFTER consuming any borrowed data from the
            // frame so a torn read can't escape past this point.
            if !guard.validate() {
                bm.note_optimistic_restart();
                continue 'restart;
            }
            match result {
                Err(e) => return Err(e),
                Ok(LookupResult::Found(v)) => return Ok(Some(consume(v))),
                Ok(LookupResult::NotFound) => return Ok(None),
                Ok(LookupResult::Crossing(c)) => c,
            }
        };
        // (No drop needed for `root_pin`: it's a borrow held by
        // the caller, not an owned `Arc` we'd be releasing here.)

        // Cross-blob hops. Same pattern; on a torn read we restart
        // the whole walk from the root (the parent BlobNode that
        // pointed us here may also have moved).
        let mut current_guid = crossing.child_guid;
        let mut start_slot = crossing.child_slot;
        let mut depth = crossing.child_depth;
        loop {
            let pin = bm.pin(current_guid)?;
            let guard = pin.read_optimistic();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let result = lookup_at(frame, start_slot, key, depth);
            if !guard.validate() {
                bm.note_optimistic_restart();
                continue 'restart;
            }
            match result {
                Err(e) => return Err(e),
                Ok(LookupResult::Found(v)) => return Ok(Some(consume(v))),
                Ok(LookupResult::NotFound) => return Ok(None),
                Ok(LookupResult::Crossing(c)) => {
                    current_guid = c.child_guid;
                    start_slot = c.child_slot;
                    depth = c.child_depth;
                }
            }
        }
    }
}

// ---------- descent dispatch ----------

fn descend<'a>(
    frame: BlobFrameRef<'a>,
    slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::descend: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot => Ok(LookupResult::NotFound),
        NodeType::Leaf => leaf_check(frame, body, key, depth),
        NodeType::Prefix => prefix_descend(frame, body, key, depth),
        NodeType::Node4 => node4_descend(frame, body, key, depth),
        NodeType::Node16 => node16_descend(frame, body, key, depth),
        NodeType::Node48 => node48_descend(frame, body, key, depth),
        NodeType::Node256 => node256_descend(frame, body, key, depth),
        NodeType::Blob => blob_descend(body, key, depth),
    }
}

fn blob_descend<'a>(body: &[u8], key: &[u8], depth: usize) -> Result<LookupResult<'a>> {
    let b = cast::<BlobNode>(body);
    let plen = b.prefix_len as usize;
    if plen > BLOB_MAX_INLINE {
        return Err(Error::node_corrupt(
            "walker::blob_descend: prefix_len exceeds inline buffer",
        ));
    }
    if depth + plen > key.len() {
        return Ok(LookupResult::NotFound);
    }
    if key[depth..depth + plen] != b.bytes[..plen] {
        return Ok(LookupResult::NotFound);
    }
    Ok(LookupResult::Crossing(BlobNodeCrossing {
        child_guid: b.child_blob_guid,
        child_slot: b.child_entry_ptr as u16,
        child_depth: depth + plen,
    }))
}

fn leaf_check<'a>(
    frame: BlobFrameRef<'a>,
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
    frame: BlobFrameRef<'a>,
    body: &'a [u8],
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let p = cast::<Prefix>(body);
    let plen = p.prefix_len as usize;
    if plen > p.bytes.len() {
        return Err(Error::node_corrupt(
            "walker::prefix_descend: prefix_len exceeds inline buffer",
        ));
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
    frame: BlobFrameRef<'a>,
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
    frame: BlobFrameRef<'a>,
    body: &'a [u8],
    key: &[u8],
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node16>(body);
    if depth >= key.len() {
        return Ok(LookupResult::NotFound);
    }
    let byte = key[depth];
    if let Some(i) = simd::node16_find_byte(&n.keys, n.count, byte) {
        return descend(frame, n.children[i as usize] as u16, key, depth + 1);
    }
    Ok(LookupResult::NotFound)
}

fn node48_descend<'a>(
    frame: BlobFrameRef<'a>,
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
        return Err(Error::node_corrupt(
            "walker::node48_descend: child index out of range",
        ));
    }
    descend(frame, n.children[ci] as u16, key, depth + 1)
}

fn node256_descend<'a>(
    frame: BlobFrameRef<'a>,
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
