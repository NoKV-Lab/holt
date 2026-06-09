//! Read-path descent — `lookup` / `lookup_at` / `lookup_multi_with`.
//!
//! All entry points take a [`BlobFrameRef`] (or a
//! [`BufferManager`] for the multi-blob variant) so the walker
//! borrows into the cached buffer with **zero memcpy**.

use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::engine::simd;
use crate::engine::{RouteCache, RouteHit};
use crate::layout::{
    BlobGuid, BlobNode, Leaf, LeafInline, Node16, Node256, Node4, Node48, NodeType, Prefix,
    BLOB_MAX_INLINE,
};
use std::sync::Arc;

use crate::store::{BlobFrameRef, BufferManager, CachedBlob};

use super::cast;
use super::readers::{leaf_extent, resolve_typed};
use super::route::{pin_route_parent, validate_route_edge};
use super::types::{BlobNodeCrossing, LookupHit, LookupResult};
use super::SearchKey;

/// Look up `key` in the tree rooted at `start_slot` (depth 0).
///
/// Takes a [`BlobFrameRef`] so the read path can run against a
/// shared buffer (e.g. a `BufferManager` read-guard) with no
/// copies. Returned borrows are tied to the lifetime of that
/// underlying buffer.
#[cfg(test)]
pub(super) fn lookup<'a>(
    frame: BlobFrameRef<'a>,
    start_slot: u16,
    key: &[u8],
) -> Result<LookupResult<'a>> {
    descend(frame, start_slot, SearchKey::exact(key), 0)
}

/// Continue a lookup at `start_slot` with a non-zero `depth` — used
/// by callers driving cross-blob descent through
/// [`LookupResult::Crossing`].
pub(super) fn lookup_at<'a>(
    frame: BlobFrameRef<'a>,
    start_slot: u16,
    key: SearchKey<'_>,
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
/// Cross-blob hops are pinned under a short shared guard on the
/// parent blob after revalidating the `BlobNode` edge, so point
/// reads do not need the tree-wide maintenance gate to keep a
/// child blob alive between "saw edge" and "pinned child".
///
/// Why restart from the root: a writer who modifies any blob may
/// also have moved the `BlobNode` crossing that pointed there, so
/// the parent-side path is stale too. Restarting catches the
/// new tree shape from the top.
///
/// On match `consume` is invoked on the live cache-pin hit and
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
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    mut consume: F,
) -> Result<Option<R>>
where
    F: FnMut(LookupHit<'_>) -> R,
{
    // Outer loop: each iteration is one full attempt; we restart
    // here when an optimistic snapshot is invalidated.
    'restart: loop {
        if let Some(cache) = route_cache {
            if let Some(route) = cache.lookup(key) {
                match lookup_from_cached_route(bm, root_pin, cache, key, route, &mut consume)? {
                    RouteLookup::Done(result) => return Ok(result),
                    RouteLookup::Stale => {}
                    RouteLookup::Restart => {
                        bm.note_optimistic_restart();
                        continue 'restart;
                    }
                }
            }
        }

        // Hop 0: the cached root blob — `Tree` keeps this pinned
        // for its lifetime so we skip BM's pin-Mutex on the
        // common case where every op starts at the root.
        let crossing = {
            let root_version = root_pin.content_version();
            let guard = root_pin.read_optimistic();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let root_slot = frame.header().root_slot;
            let result = lookup_at(frame, root_slot, key, 0);

            // Validate AFTER consuming any borrowed data from the
            // frame so a torn read can't escape past this point.
            if !guard.validate() || !root_pin.validate_content_version(root_version) {
                bm.note_optimistic_restart();
                continue 'restart;
            }
            match result {
                Err(e) => return Err(e),
                Ok(LookupResult::Found(hit)) => return Ok(Some(consume(hit))),
                Ok(LookupResult::NotFound) => return Ok(None),
                Ok(LookupResult::Crossing(crossing)) => crossing,
            }
        };
        // (No drop needed for `root_pin`: it's a borrow held by
        // the caller, not an owned `Arc` we'd be releasing here.)

        let Some((child_pin, child_depth)) =
            pin_validated_child(bm, route_cache, key, root_pin, 0, crossing)?
        else {
            bm.note_optimistic_restart();
            continue 'restart;
        };

        // Cross-blob hops. Same pattern; on a torn read we restart
        // the whole walk from the root (the parent BlobNode that
        // pointed us here may also have moved).
        match lookup_from_pinned_blob(bm, route_cache, key, child_pin, child_depth, &mut consume)? {
            CrossBlobLookup::Done(result) => return Ok(result),
            CrossBlobLookup::Restart => {
                bm.note_optimistic_restart();
            }
        }
    }
}

enum RouteLookup<R> {
    Done(Option<R>),
    Restart,
    Stale,
}

enum CrossBlobLookup<R> {
    Done(Option<R>),
    Restart,
}

fn lookup_from_cached_route<R, F>(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    cache: &RouteCache,
    key: SearchKey<'_>,
    route: RouteHit,
    consume: &mut F,
) -> Result<RouteLookup<R>>
where
    F: FnMut(LookupHit<'_>) -> R,
{
    let parent_pin = match pin_route_parent(bm, root_pin, route) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) => {
            cache.invalidate(key, route);
            return Ok(RouteLookup::Stale);
        }
        Err(e) => return Err(e),
    };
    let parent_guard = parent_pin.read();
    let parent_version = parent_pin.content_version();
    if parent_version != route.parent_version {
        let frame = BlobFrameRef::wrap(parent_guard.as_slice());
        if !validate_route_edge(frame, key, route)? {
            drop(parent_guard);
            cache.invalidate(key, route);
            return Ok(RouteLookup::Stale);
        }
        cache.refresh_parent_version(key, route, parent_version);
    }
    let child_pin = match bm.pin(route.child_guid) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) => {
            drop(parent_guard);
            cache.invalidate(key, route);
            return Ok(RouteLookup::Stale);
        }
        Err(e) => return Err(e),
    };
    child_pin.prefetch_header();
    drop(parent_guard);

    let crossing = {
        let child_version = child_pin.content_version();
        let guard = child_pin.read_optimistic();
        let frame = BlobFrameRef::wrap(guard.as_slice());
        let start_slot = frame.header().root_slot;
        let result = lookup_at(frame, start_slot, key, route.child_depth);
        if !guard.validate() || !child_pin.validate_content_version(child_version) {
            return Ok(RouteLookup::Restart);
        }
        match result {
            Err(e) => return Err(e),
            Ok(LookupResult::Found(hit)) => return Ok(RouteLookup::Done(Some(consume(hit)))),
            Ok(LookupResult::NotFound) => return Ok(RouteLookup::Done(None)),
            Ok(LookupResult::Crossing(crossing)) => crossing,
        }
    };
    {
        let Some((next_pin, next_depth)) = pin_validated_child(
            bm,
            Some(cache),
            key,
            &child_pin,
            route.child_depth,
            crossing,
        )?
        else {
            return Ok(RouteLookup::Restart);
        };
        match lookup_from_pinned_blob(bm, Some(cache), key, next_pin, next_depth, consume)? {
            CrossBlobLookup::Done(result) => Ok(RouteLookup::Done(result)),
            CrossBlobLookup::Restart => Ok(RouteLookup::Restart),
        }
    }
}

fn lookup_from_pinned_blob<R, F>(
    bm: &BufferManager,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    mut pin: Arc<CachedBlob>,
    mut depth: usize,
    consume: &mut F,
) -> Result<CrossBlobLookup<R>>
where
    F: FnMut(LookupHit<'_>) -> R,
{
    loop {
        pin.prefetch_header();
        let crossing = {
            let parent_version = pin.content_version();
            let guard = pin.read_optimistic();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let start_slot = frame.header().root_slot;
            let result = lookup_at(frame, start_slot, key, depth);
            if !guard.validate() || !pin.validate_content_version(parent_version) {
                return Ok(CrossBlobLookup::Restart);
            }
            match result {
                Err(e) => return Err(e),
                Ok(LookupResult::Found(hit)) => {
                    return Ok(CrossBlobLookup::Done(Some(consume(hit))));
                }
                Ok(LookupResult::NotFound) => return Ok(CrossBlobLookup::Done(None)),
                Ok(LookupResult::Crossing(crossing)) => crossing,
            }
        };

        let Some((child_pin, child_depth)) =
            pin_validated_child(bm, route_cache, key, &pin, depth, crossing)?
        else {
            return Ok(CrossBlobLookup::Restart);
        };
        pin = child_pin;
        depth = child_depth;
    }
}

fn pin_validated_child(
    bm: &BufferManager,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    parent_pin: &Arc<CachedBlob>,
    parent_depth: usize,
    expected: BlobNodeCrossing,
) -> Result<Option<(Arc<CachedBlob>, usize)>> {
    let parent_guard = parent_pin.read();
    let parent_version = parent_pin.content_version();
    let frame = BlobFrameRef::wrap(parent_guard.as_slice());
    let parent_guid: BlobGuid = frame.header().blob_guid;
    let start_slot = frame.header().root_slot;
    let actual = match lookup_at(frame, start_slot, key, parent_depth)? {
        LookupResult::Crossing(crossing)
            if crossing.child_guid == expected.child_guid
                && crossing.child_depth == expected.child_depth =>
        {
            crossing
        }
        LookupResult::Crossing(_) | LookupResult::Found(_) | LookupResult::NotFound => {
            return Ok(None);
        }
    };

    if let Some(cache) = route_cache {
        cache.learn(
            key,
            parent_guid,
            parent_depth,
            parent_version,
            actual.child_guid,
            actual.child_depth,
        );
        bm.mark_route_resident(actual.child_guid);
    }
    let child_pin = bm.pin(actual.child_guid)?;
    child_pin.prefetch_header();
    drop(parent_guard);
    Ok(Some((child_pin, actual.child_depth)))
}

// ---------- descent dispatch ----------

fn descend<'a>(
    frame: BlobFrameRef<'a>,
    slot: u16,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::descend: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot => Ok(LookupResult::NotFound),
        NodeType::Leaf => leaf_check(frame, body, key, depth),
        NodeType::LeafInline => Ok(inline_leaf_check(body, key)),
        NodeType::Prefix => prefix_descend(frame, body, key, depth),
        NodeType::Node4 => node4_descend(frame, body, key, depth),
        NodeType::Node16 => node16_descend(frame, body, key, depth),
        NodeType::Node48 => node48_descend(frame, body, key, depth),
        NodeType::Node256 => node256_descend(frame, body, key, depth),
        NodeType::Blob => blob_descend(body, key, depth),
    }
}

fn blob_descend<'a>(body: &[u8], key: SearchKey<'_>, depth: usize) -> Result<LookupResult<'a>> {
    let b = cast::<BlobNode>(body);
    let plen = b.prefix_len as usize;
    if plen > BLOB_MAX_INLINE {
        return Err(Error::node_corrupt(
            "walker::blob_descend: prefix_len exceeds inline buffer",
        ));
    }
    if !key.range_eq(depth, &b.bytes[..plen]) {
        return Ok(LookupResult::NotFound);
    }
    Ok(LookupResult::Crossing(BlobNodeCrossing {
        child_guid: b.child_blob_guid,
        child_depth: depth + plen,
    }))
}

fn leaf_check<'a>(
    frame: BlobFrameRef<'a>,
    body: &'a [u8],
    key: SearchKey<'_>,
    _depth: usize,
) -> Result<LookupResult<'a>> {
    let leaf = cast::<Leaf>(body);
    if leaf.tombstone != 0 {
        return Ok(LookupResult::NotFound);
    }
    // Fingerprint gate: a path-compressed ART reaches a leaf whose key
    // may still differ from the search key (lazy expansion). When the
    // leaf carries a fingerprint (`!= 0`) and it disagrees with the
    // search key's, the keys cannot be equal — reject without the
    // second cache miss of reading the key extent. A match (or an
    // un-fingerprinted older leaf) still does the full compare below,
    // so this is never a false negative.
    if leaf.key_fp != 0 && leaf.key_fp != key.fingerprint() {
        return Ok(LookupResult::NotFound);
    }
    let (leaf_key, value) = leaf_extent(frame, leaf)?;
    if !key.eq_slice(leaf_key) {
        return Ok(LookupResult::NotFound);
    }
    Ok(LookupResult::Found(LookupHit {
        value,
        seq: leaf.seq,
    }))
}

/// `leaf_check` for an inline leaf: key and value both live in the
/// node body, so no extent read is needed.
fn inline_leaf_check<'a>(body: &'a [u8], key: SearchKey<'_>) -> LookupResult<'a> {
    let li = cast::<LeafInline>(body);
    if li.tombstone != 0 {
        return LookupResult::NotFound;
    }
    if !key.eq_slice(li.key()) {
        return LookupResult::NotFound;
    }
    LookupResult::Found(LookupHit {
        value: li.value(),
        seq: li.seq,
    })
}

fn prefix_descend<'a>(
    frame: BlobFrameRef<'a>,
    body: &'a [u8],
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    let p = cast::<Prefix>(body);
    let plen = p.prefix_len as usize;
    if plen > p.bytes.len() {
        return Err(Error::node_corrupt(
            "walker::prefix_descend: prefix_len exceeds inline buffer",
        ));
    }
    if !key.range_eq(depth, &p.bytes[..plen]) {
        return Ok(LookupResult::NotFound);
    }
    descend(frame, p.child as u16, key, depth + plen)
}

fn node4_descend<'a>(
    frame: BlobFrameRef<'a>,
    body: &'a [u8],
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node4>(body);
    let Some(byte) = key.byte_at(depth) else {
        return Ok(LookupResult::NotFound);
    };
    let count = (n.count as usize).min(4);
    for i in 0..count {
        if n.keys[i] == byte {
            let child = n.children[i] as u16;
            frame.prefetch_node(child);
            return descend(frame, child, key, depth + 1);
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
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node16>(body);
    let Some(byte) = key.byte_at(depth) else {
        return Ok(LookupResult::NotFound);
    };
    if let Some(i) = simd::node16_find_byte(&n.keys, n.count, byte) {
        let child = n.children[i as usize] as u16;
        frame.prefetch_node(child);
        return descend(frame, child, key, depth + 1);
    }
    Ok(LookupResult::NotFound)
}

fn node48_descend<'a>(
    frame: BlobFrameRef<'a>,
    body: &'a [u8],
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node48>(body);
    let Some(byte) = key.byte_at(depth) else {
        return Ok(LookupResult::NotFound);
    };
    let idx = n.index[byte as usize];
    if idx == 0 {
        return Ok(LookupResult::NotFound);
    }
    let ci = idx as usize - 1;
    if ci >= 48 {
        return Err(Error::node_corrupt(
            "walker::node48_descend: child index out of range",
        ));
    }
    let child = n.children[ci] as u16;
    frame.prefetch_node(child);
    descend(frame, child, key, depth + 1)
}

fn node256_descend<'a>(
    frame: BlobFrameRef<'a>,
    body: &'a [u8],
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    let n = cast::<Node256>(body);
    let Some(byte) = key.byte_at(depth) else {
        return Ok(LookupResult::NotFound);
    };
    let slot = n.children[byte as usize];
    if slot == 0 {
        return Ok(LookupResult::NotFound);
    }
    let child = slot as u16;
    frame.prefetch_node(child);
    descend(frame, child, key, depth + 1)
}
