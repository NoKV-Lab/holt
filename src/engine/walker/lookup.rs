//! Read-path descent — `lookup` / `lookup_at` / `lookup_multi_with`.
//!
//! All entry points take a [`BlobFrameRef`] (or a
//! [`BufferManager`] for the multi-blob variant) so the walker
//! borrows into the cached buffer with **zero memcpy**.

use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::engine::simd;
use crate::engine::{RouteCache, RouteHit};
use crate::layout::{
    BlobGuid, BlobNode, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix, BLOB_MAX_INLINE,
};
use std::mem::size_of;
use std::sync::Arc;

use crate::store::{BlobFrameRef, BufferManager, CachedBlob, ColdBlobLookup};

use super::cast;
use super::readers::{child_offset, resolve_typed};
use super::route::{pin_route_parent, validate_route_edge};
use super::types::{BlobNodeCrossing, LookupHit, LookupResult};
use super::SearchKey;
use crate::store::decode_child_off;

/// Look up `key` in the tree whose root is the encoded offset
/// `start_root` (depth 0).
///
/// `start_root` is the *encoded* root offset as stored in
/// `header.root_slot` (see `encode_child_off`); it is decoded once
/// before descent. Takes a [`BlobFrameRef`] so the read path can run
/// against a shared buffer (e.g. a `BufferManager` read-guard) with
/// no copies. Returned borrows are tied to the lifetime of that
/// underlying buffer.
#[cfg(test)]
pub(super) fn lookup<'a>(
    frame: BlobFrameRef<'a>,
    start_root: u16,
    key: &[u8],
) -> Result<LookupResult<'a>> {
    descend(
        frame,
        decode_child_off(start_root),
        SearchKey::exact(key),
        0,
    )
}

/// Continue a lookup at the encoded root `start_root` with a non-zero
/// `depth` — used by callers driving cross-blob descent through
/// [`LookupResult::Crossing`].
pub(super) fn lookup_at<'a>(
    frame: BlobFrameRef<'a>,
    start_root: u16,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    descend(frame, decode_child_off(start_root), key, depth)
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

        let Some(crossing) = validate_child_crossing(bm, route_cache, key, root_pin, 0, crossing)?
        else {
            bm.note_optimistic_restart();
            continue 'restart;
        };
        let (child_pin, child_depth) = match cold_lookup_or_pin(bm, key, crossing, &mut consume)? {
            ColdLookupOrPin::Done(result) => return Ok(result),
            ColdLookupOrPin::Pin { pin, depth } => (pin, depth),
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
    let child_pin = match cold_lookup_or_pin(
        bm,
        key,
        BlobNodeCrossing {
            child_guid: route.child_guid,
            child_depth: route.child_depth,
        },
        consume,
    ) {
        Ok(ColdLookupOrPin::Done(result)) => return Ok(RouteLookup::Done(result)),
        Ok(ColdLookupOrPin::Pin { pin, .. }) => pin,
        Err(e) if is_blob_store_not_found(&e) => {
            drop(parent_guard);
            cache.invalidate(key, route);
            return Ok(RouteLookup::Stale);
        }
        Err(e) => return Err(e),
    };
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
        let Some(crossing) = validate_child_crossing(
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
        let (next_pin, next_depth) = match cold_lookup_or_pin(bm, key, crossing, consume)? {
            ColdLookupOrPin::Done(result) => return Ok(RouteLookup::Done(result)),
            ColdLookupOrPin::Pin { pin, depth } => (pin, depth),
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

        let Some(crossing) = validate_child_crossing(bm, route_cache, key, &pin, depth, crossing)?
        else {
            return Ok(CrossBlobLookup::Restart);
        };
        match cold_lookup_or_pin(bm, key, crossing, consume)? {
            ColdLookupOrPin::Done(result) => return Ok(CrossBlobLookup::Done(result)),
            ColdLookupOrPin::Pin {
                pin: child_pin,
                depth: child_depth,
            } => {
                pin = child_pin;
                depth = child_depth;
            }
        }
    }
}

fn validate_child_crossing(
    bm: &BufferManager,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    parent_pin: &Arc<CachedBlob>,
    parent_depth: usize,
    expected: BlobNodeCrossing,
) -> Result<Option<BlobNodeCrossing>> {
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
    Ok(Some(actual))
}

enum ColdLookupOrPin<R> {
    Done(Option<R>),
    Pin { pin: Arc<CachedBlob>, depth: usize },
}

fn cold_lookup_or_pin<R, F>(
    bm: &BufferManager,
    key: SearchKey<'_>,
    crossing: BlobNodeCrossing,
    consume: &mut F,
) -> Result<ColdLookupOrPin<R>>
where
    F: FnMut(LookupHit<'_>) -> R,
{
    let Some(user_key) = key.user_bytes() else {
        let pin = bm.pin(crossing.child_guid)?;
        pin.prefetch_header();
        return Ok(ColdLookupOrPin::Pin {
            pin,
            depth: crossing.child_depth,
        });
    };

    let mut child_guid = crossing.child_guid;
    let mut child_depth = crossing.child_depth;
    loop {
        match bm.cold_lookup_blob(child_guid, user_key, child_depth)? {
            ColdBlobLookup::Unknown => {
                let pin = bm.pin(child_guid)?;
                pin.prefetch_header();
                return Ok(ColdLookupOrPin::Pin {
                    pin,
                    depth: child_depth,
                });
            }
            ColdBlobLookup::Found { value, seq } => {
                let out = consume(LookupHit { value: &value, seq });
                return Ok(ColdLookupOrPin::Done(Some(out)));
            }
            ColdBlobLookup::Crossing {
                child_guid: next_guid,
                child_depth: next_depth,
            } => {
                child_guid = next_guid;
                child_depth = next_depth;
            }
            ColdBlobLookup::NotFound => return Ok(ColdLookupOrPin::Done(None)),
        }
    }
}

// ---------- descent dispatch ----------

fn descend<'a>(
    frame: BlobFrameRef<'a>,
    off: u32,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<LookupResult<'a>> {
    let (ntype, body) = resolve_typed(frame, off)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::descend: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot => Ok(LookupResult::NotFound),
        NodeType::Leaf => leaf_check(body, key, depth),
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

fn leaf_check<'a>(body: &'a [u8], key: SearchKey<'_>, _depth: usize) -> Result<LookupResult<'a>> {
    // The leaf is one contiguous, self-describing node:
    // `[16B header][key][value]`. Cast ONLY the 16-byte header.
    let leaf = *cast::<Leaf>(&body[..size_of::<Leaf>()]);
    if leaf.tombstone != 0 {
        return Ok(LookupResult::NotFound);
    }
    // Fingerprint gate: a path-compressed ART reaches a leaf whose key
    // may still differ from the search key (lazy expansion). When the
    // leaf carries a fingerprint (`!= 0`) and it disagrees with the
    // search key's, the keys cannot be equal — reject without the SIMD
    // key compare against the inline key bytes. A match (or an
    // un-fingerprinted older leaf) still does the full compare below,
    // so this is never a false negative.
    if leaf.key_fp != 0 && leaf.key_fp != key.fingerprint() {
        return Ok(LookupResult::NotFound);
    }
    let key_len = leaf.key_len as usize;
    let value_len = leaf.value_len as usize;
    let key_end = 16 + key_len;
    let value_end = key_end + value_len;
    if value_end > body.len() {
        return Err(Error::node_corrupt("leaf_check: key/value out of range"));
    }
    let leaf_key = &body[16..key_end];
    if !key.eq_slice(leaf_key) {
        return Ok(LookupResult::NotFound);
    }
    Ok(LookupResult::Found(LookupHit {
        value: &body[key_end..value_end],
        seq: leaf.seq,
    }))
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
    descend(frame, child_offset(p.child as u16), key, depth + plen)
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
            let child_off = child_offset(n.children[i]);
            frame.prefetch_at(child_off);
            return descend(frame, child_off, key, depth + 1);
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
        let child_off = child_offset(n.children[i as usize]);
        frame.prefetch_at(child_off);
        return descend(frame, child_off, key, depth + 1);
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
    let child_off = child_offset(n.children[ci]);
    frame.prefetch_at(child_off);
    descend(frame, child_off, key, depth + 1)
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
    let encoded = n.children[byte as usize];
    if encoded == 0 {
        return Ok(LookupResult::NotFound);
    }
    let child_off = child_offset(encoded);
    frame.prefetch_at(child_off);
    descend(frame, child_off, key, depth + 1)
}
