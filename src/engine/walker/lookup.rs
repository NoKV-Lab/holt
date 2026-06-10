//! Read-path descent — `lookup` / `lookup_at` / `lookup_multi_with`.
//!
//! All entry points take a [`BlobFrameRef`] (or a
//! [`BufferManager`] for the multi-blob variant) so the walker
//! borrows into the cached buffer with **zero memcpy**.

use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::engine::simd;
use crate::engine::{RouteCache, RouteHit};
use crate::layout::{
    leaf_body_size, BlobGuid, BlobNode, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix,
    BLOB_MAX_INLINE, HEADER_SIZE, PREFIX_MAX_INLINE,
};
use std::mem::size_of;
use std::sync::Arc;

use crate::store::blob_store::AlignedBlobBuf;
use crate::store::{page_align_up, BlobFrameRef, BufferManager, CachedBlob, ColdBlobLookup, PAGE_4K};

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
    // Only exact point lookups (a user-style key) take the cold path;
    // range/prefix/non-exact searches pin directly.
    if key.user_bytes().is_none() {
        let pin = bm.pin(crossing.child_guid)?;
        pin.prefetch_header();
        return Ok(ColdLookupOrPin::Pin {
            pin,
            depth: crossing.child_depth,
        });
    }

    let mut child_guid = crossing.child_guid;
    let mut child_depth = crossing.child_depth;
    loop {
        // Answer cold from the in-blob routing region (stage 3); any
        // uncertainty falls back to the authoritative full pin.
        match cold_read_routed(bm, child_guid, key, child_depth) {
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

// ---------- stage 3: cold routed read ----------

/// Answer a cold point lookup against a **routed** blob by reading only
/// the header page + routing region + one leaf page via
/// `read_blob_range`, instead of pinning the whole 512 KB frame.
///
/// A pure accelerator: on ANY uncertainty — the blob isn't
/// cold-eligible (cached/pending-delete/protected), it's in the legacy
/// whole-frame layout (`routing_len == 0`), a read fails, or descent
/// reaches an unexpected node — it returns [`ColdBlobLookup::Unknown`]
/// and the caller falls back to `bm.pin`, which reads the authoritative
/// image. So it can never change `get()` semantics; it only avoids I/O.
fn cold_read_routed(
    bm: &BufferManager,
    guid: BlobGuid,
    key: SearchKey<'_>,
    depth: usize,
) -> ColdBlobLookup {
    if !bm.cold_read_eligible(guid) {
        return ColdBlobLookup::Unknown;
    }
    // Descend with the SAME `key` the pin-fallback frame descent uses
    // (the caller only reaches here with a user-style key), so a routed
    // and a full-frame read are byte-for-byte equivalent.
    let mut scratch = AlignedBlobBuf::zeroed();
    let mut read_range = |off: u64, dst: &mut [u8]| bm.read_blob_range(guid, off, dst);
    cold_read_routed_into(scratch.as_mut_slice(), &mut read_range, key, depth)
        .unwrap_or(ColdBlobLookup::Unknown)
}

/// Routed-read core, decoupled from the buffer manager via a
/// `read_range(byte_offset, dst)` closure so it can be unit-tested
/// against an in-memory routed frame.
///
/// `scratch` must be a `PAGE_SIZE`, 4 KB-aligned, zeroed buffer; the
/// header page, routing region, and one leaf page are read into it at
/// their absolute offsets. Returns `Unknown` for a legacy
/// (`routing_len == 0`) blob.
pub(super) fn cold_read_routed_into(
    scratch: &mut [u8],
    read_range: &mut dyn FnMut(u64, &mut [u8]) -> Result<()>,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<ColdBlobLookup> {
    // Header page → routing geometry + root.
    read_range(0, &mut scratch[..HEADER_SIZE as usize])?;
    let (root_off, rr) = {
        let frame = BlobFrameRef::wrap(&scratch[..]);
        let h = frame.header();
        match h.routing_region() {
            Some(rr) => (decode_child_off(h.root_slot), rr),
            None => return Ok(ColdBlobLookup::Unknown), // legacy → full pin
        }
    };
    // Routing region (internal nodes): [routing_off, leaf_region_start)
    // — both page-aligned, so the read length is a 4 KB multiple.
    read_range(
        u64::from(rr.off),
        &mut scratch[rr.off as usize..rr.leaf_region_start as usize],
    )?;
    descend_routed(scratch, read_range, root_off, key, depth, rr.leaf_region_start)
}

/// One step of the routed descent: the next child offset to visit, or a
/// terminal answer.
enum RoutedStep {
    Visit(u32, usize),
    Done(ColdBlobLookup),
}

/// Resolve the (resident, internal) node at `off` and decide the next
/// routed step. Mirrors `descend`'s per-node dispatch; everything is
/// copied out so the frame borrow can end before the caller pages in a
/// leaf or recurses.
fn routed_step(
    frame: BlobFrameRef<'_>,
    off: u32,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<RoutedStep> {
    let (ntype, body) = resolve_typed(frame, off)?;
    let not_found = RoutedStep::Done(ColdBlobLookup::NotFound);
    Ok(match ntype {
        NodeType::Prefix => {
            let p = *cast::<Prefix>(body);
            let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
            if key.range_eq(depth, &p.bytes[..plen]) {
                RoutedStep::Visit(child_offset(p.child as u16), depth + plen)
            } else {
                not_found
            }
        }
        NodeType::Node4 => {
            let n = *cast::<Node4>(body);
            let Some(byte) = key.byte_at(depth) else {
                return Ok(not_found);
            };
            let mut child = None;
            for i in 0..(n.count as usize).min(4) {
                if n.keys[i] == byte {
                    child = Some(child_offset(n.children[i]));
                    break;
                }
                if n.keys[i] > byte {
                    break;
                }
            }
            child.map_or(not_found, |c| RoutedStep::Visit(c, depth + 1))
        }
        NodeType::Node16 => {
            let n = *cast::<Node16>(body);
            match key
                .byte_at(depth)
                .and_then(|byte| simd::node16_find_byte(&n.keys, n.count, byte))
            {
                Some(i) => RoutedStep::Visit(child_offset(n.children[i as usize]), depth + 1),
                None => not_found,
            }
        }
        NodeType::Node48 => {
            let n = *cast::<Node48>(body);
            let idx = key.byte_at(depth).map_or(0, |byte| n.index[byte as usize]);
            if idx == 0 {
                not_found
            } else {
                let ci = idx as usize - 1;
                if ci >= 48 {
                    return Err(Error::node_corrupt("cold_read_routed: node48 index out of range"));
                }
                RoutedStep::Visit(child_offset(n.children[ci]), depth + 1)
            }
        }
        NodeType::Node256 => {
            let n = *cast::<Node256>(body);
            match key.byte_at(depth) {
                Some(byte) if n.children[byte as usize] != 0 => {
                    RoutedStep::Visit(child_offset(n.children[byte as usize]), depth + 1)
                }
                _ => not_found,
            }
        }
        NodeType::Blob => {
            let b = *cast::<BlobNode>(body);
            let plen = (b.prefix_len as usize).min(BLOB_MAX_INLINE);
            if key.range_eq(depth, &b.bytes[..plen]) {
                RoutedStep::Done(ColdBlobLookup::Crossing {
                    child_guid: b.child_blob_guid,
                    child_depth: depth + plen,
                })
            } else {
                not_found
            }
        }
        // A Leaf/EmptyRoot/Invalid at an internal position
        // (off < leaf_region_start) is unexpected — bail to the
        // authoritative full pin.
        NodeType::Leaf | NodeType::EmptyRoot | NodeType::Invalid => {
            RoutedStep::Done(ColdBlobLookup::Unknown)
        }
    })
}

fn descend_routed(
    scratch: &mut [u8],
    read_range: &mut dyn FnMut(u64, &mut [u8]) -> Result<()>,
    off: u32,
    key: SearchKey<'_>,
    depth: usize,
    leaf_region_start: u32,
) -> Result<ColdBlobLookup> {
    // The decision is taken (and copied out) under a short frame borrow
    // so we can page in a leaf or recurse with `&mut scratch` after.
    let step = routed_step(BlobFrameRef::wrap(&scratch[..]), off, key, depth)?;
    match step {
        RoutedStep::Done(answer) => Ok(answer),
        RoutedStep::Visit(child_off, new_depth) => {
            if child_off >= leaf_region_start {
                // Leaf: page it in, then run the leaf compare.
                page_in_leaf(scratch, read_range, child_off)?;
                let frame = BlobFrameRef::wrap(&scratch[..]);
                let body = frame
                    .body_at_offset(child_off)
                    .ok_or(Error::node_corrupt("cold_read_routed: leaf body range"))?;
                leaf_check_owned(body, key)
            } else {
                descend_routed(scratch, read_range, child_off, key, new_depth, leaf_region_start)
            }
        }
    }
}

/// Page the leaf at `loff` (>= leaf_region_start) into `scratch` at its
/// absolute offset: read the page(s) covering its 16-byte header, then
/// extend to cover the full `[16B hdr][key][value]` body (a large
/// value can straddle pages).
fn page_in_leaf(
    scratch: &mut [u8],
    read_range: &mut dyn FnMut(u64, &mut [u8]) -> Result<()>,
    loff: u32,
) -> Result<()> {
    let page0 = loff & !(PAGE_4K - 1);
    let hdr_end = page_align_up(loff + size_of::<Leaf>() as u32);
    read_range(u64::from(page0), &mut scratch[page0 as usize..hdr_end as usize])?;
    let (key_len, value_len) = {
        let leaf = cast::<Leaf>(&scratch[loff as usize..loff as usize + size_of::<Leaf>()]);
        (u32::from(leaf.key_len), u32::from(leaf.value_len))
    };
    let body_end = page_align_up(loff + leaf_body_size(key_len, value_len));
    if body_end > hdr_end {
        read_range(
            u64::from(hdr_end),
            &mut scratch[hdr_end as usize..body_end as usize],
        )?;
    }
    Ok(())
}

/// Like `leaf_check` but returns an owned [`ColdBlobLookup`] — the value
/// is copied out of the paged-in buffer, which the caller drops.
fn leaf_check_owned(body: &[u8], key: SearchKey<'_>) -> Result<ColdBlobLookup> {
    let leaf = *cast::<Leaf>(&body[..size_of::<Leaf>()]);
    if leaf.tombstone != 0 {
        return Ok(ColdBlobLookup::NotFound);
    }
    if leaf.key_fp != 0 && leaf.key_fp != key.fingerprint() {
        return Ok(ColdBlobLookup::NotFound);
    }
    let key_len = leaf.key_len as usize;
    let value_len = leaf.value_len as usize;
    let key_end = 16 + key_len;
    let value_end = key_end + value_len;
    if value_end > body.len() {
        return Err(Error::node_corrupt("cold_read_routed: leaf key/value range"));
    }
    if !key.eq_slice(&body[16..key_end]) {
        return Ok(ColdBlobLookup::NotFound);
    }
    Ok(ColdBlobLookup::Found {
        value: body[key_end..value_end].to_vec(),
        seq: leaf.seq,
    })
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
