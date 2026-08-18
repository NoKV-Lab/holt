//! Read-path descent — `lookup` / `lookup_at` / `lookup_multi_with`.
//!
//! All entry points take a [`BlobFrameRef`] (or a
//! [`BufferManager`] for the multi-blob variant) so the walker
//! borrows into the cached buffer with **zero memcpy**.

use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::engine::simd;
use crate::engine::{RouteCache, RouteHit};
use crate::layout::{
    leaf_body_size, size_of_node, BlobGuid, BlobNode, Leaf, Node16, Node256, Node4, Node48,
    NodeType, Prefix, BLOB_MAX_INLINE, HEADER_SIZE, PREFIX_MAX_INLINE,
};
use std::cell::RefCell;
use std::mem::size_of;
use std::sync::Arc;

use crate::store::blob_store::AlignedBlobBuf;
use crate::store::{
    page_align_up, BlobFrameRef, BufferManager, CachedBlob, IndexedBlobLookup, ReadIndex,
    ReadIndexAnswer, ReadIndexHit, ReadIndexStamp, PAGE_4K,
};

use super::cast;
use super::readers::{child_offset, resolve_typed, root_child_offset};
use super::route::{pin_route_parent, validate_route_edge};
use super::types::{BlobNodeCrossing, LongestPrefixHit, LookupHit, LookupResult};
use super::SearchKey;

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
        root_child_offset(start_root, "lookup: root child")?,
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
    descend(
        frame,
        root_child_offset(start_root, "lookup_at: root child")?,
        key,
        depth,
    )
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
    lookup_multi_with_gc_policy(bm, root_pin, route_cache, key, &mut consume, true)
}

/// Snapshot/view lookup variant. Its root closure is pinned for the full
/// lease, so a missing descendant is stable corruption rather than a GC
/// race and must not be hidden behind an optimistic restart.
pub(crate) fn lookup_multi_with_snapshot<R, F>(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    mut consume: F,
) -> Result<Option<R>>
where
    F: FnMut(LookupHit<'_>) -> R,
{
    lookup_multi_with_gc_policy(bm, root_pin, route_cache, key, &mut consume, false)
}

/// Return the longest live user key that prefixes `query`.
///
/// Unlike exact lookup, this path intentionally pins every crossed blob
/// because a negative exact-read-index answer cannot prove that no shorter
/// terminator child matched earlier in the query path. Every visited blob
/// version is validated before publishing the result; concurrent mutation
/// restarts the walk from the root.
pub(crate) fn longest_prefix_multi(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    query: &[u8],
    restart_on_gc: bool,
) -> Result<Option<LongestPrefixHit>> {
    'restart: loop {
        let mut child_pin = None;
        let mut depth = 0;
        let mut state = LongestPrefixState {
            bm,
            key: SearchKey::user(query),
            best: None,
            root_version: None,
            visited: Vec::new(),
            unstable: false,
            gc_epoch: restart_on_gc.then(|| bm.gc_read_epoch()),
        };
        loop {
            let pin = child_pin.as_ref().unwrap_or(root_pin);
            let version = pin.content_version();
            let guard = pin.read_optimistic();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let root_slot = frame.header().root_slot;
            let step = root_child_offset(root_slot, "longest_prefix: root child")
                .and_then(|root| longest_prefix_at(frame, root, depth, &mut state));
            if state.unstable || !guard.validate() || !pin.validate_content_version(version) {
                bm.note_optimistic_restart();
                continue 'restart;
            }
            let step = step?;
            drop(guard);
            if let Some(pin) = child_pin.take() {
                state.visited.push((pin, version));
            } else {
                state.root_version = Some(version);
            }
            match step {
                LongestPrefixStep::Done => {
                    if state
                        .root_version
                        .is_some_and(|root_version| root_pin.validate_content_version(root_version))
                        && state.visited.iter().all(|(visited_pin, visited_version)| {
                            visited_pin.validate_content_version(*visited_version)
                        })
                    {
                        return Ok(state.best);
                    }
                    bm.note_optimistic_restart();
                    continue 'restart;
                }
                LongestPrefixStep::Crossing(crossing) => {
                    let Some(pin) = pin_longest_prefix_child(&mut state, crossing.child_guid)?
                    else {
                        bm.note_optimistic_restart();
                        continue 'restart;
                    };
                    child_pin = Some(pin);
                    depth = crossing.child_depth;
                }
            }
        }
    }
}

struct LongestPrefixState<'bm, 'key> {
    bm: &'bm BufferManager,
    key: SearchKey<'key>,
    best: Option<LongestPrefixHit>,
    root_version: Option<u64>,
    visited: Vec<(Arc<CachedBlob>, u64)>,
    unstable: bool,
    gc_epoch: Option<u64>,
}

enum LongestPrefixStep {
    Done,
    Crossing(BlobNodeCrossing),
}

fn longest_prefix_at(
    frame: BlobFrameRef<'_>,
    mut off: u32,
    mut depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<LongestPrefixStep> {
    loop {
        let (ntype, body) = resolve_typed(frame, off)?;
        match ntype {
            NodeType::Invalid => {
                return Err(Error::node_corrupt(
                    "walker::longest_prefix: hit NodeType::Invalid",
                ));
            }
            NodeType::EmptyRoot => return Ok(LongestPrefixStep::Done),
            NodeType::Leaf => {
                update_longest_prefix_leaf(body, state)?;
                return Ok(LongestPrefixStep::Done);
            }
            NodeType::Prefix => {
                let Some((next, next_depth)) =
                    longest_prefix_through_prefix(body, state.key, depth)?
                else {
                    return Ok(LongestPrefixStep::Done);
                };
                off = next;
                depth = next_depth;
            }
            NodeType::Node4 => {
                let Some(next) = longest_prefix_through_node4(frame, body, depth, state)? else {
                    return Ok(LongestPrefixStep::Done);
                };
                off = next;
                depth += 1;
            }
            NodeType::Node16 => {
                let Some(next) = longest_prefix_through_node16(frame, body, depth, state)? else {
                    return Ok(LongestPrefixStep::Done);
                };
                off = next;
                depth += 1;
            }
            NodeType::Node48 => {
                let Some(next) = longest_prefix_through_node48(frame, body, depth, state)? else {
                    return Ok(LongestPrefixStep::Done);
                };
                off = next;
                depth += 1;
            }
            NodeType::Node256 => {
                let Some(next) = longest_prefix_through_node256(frame, body, depth, state)? else {
                    return Ok(LongestPrefixStep::Done);
                };
                off = next;
                depth += 1;
            }
            NodeType::Blob => {
                let blob = cast::<BlobNode>(body);
                let prefix_len = usize::from(blob.prefix_len);
                if prefix_len > BLOB_MAX_INLINE
                    || !state.key.range_eq(depth, &blob.bytes[..prefix_len])
                {
                    return Ok(LongestPrefixStep::Done);
                }
                return Ok(LongestPrefixStep::Crossing(BlobNodeCrossing {
                    child_guid: blob.child_blob_guid,
                    child_depth: depth + prefix_len,
                }));
            }
        }
    }
}

fn longest_prefix_through_prefix(
    body: &[u8],
    key: SearchKey<'_>,
    depth: usize,
) -> Result<Option<(u32, usize)>> {
    let prefix = cast::<Prefix>(body);
    let prefix_len = usize::from(prefix.prefix_len);
    if prefix_len > prefix.bytes.len() {
        return Err(Error::node_corrupt(
            "longest_prefix: prefix_len exceeds inline buffer",
        ));
    }
    Ok(key
        .range_eq(depth, &prefix.bytes[..prefix_len])
        .then(|| (child_offset(prefix.child as u16), depth + prefix_len)))
}

fn longest_prefix_through_node4(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<Option<u32>> {
    let node = cast::<Node4>(body);
    let count = usize::from(node.count).min(4);
    update_inner_terminator_candidate(
        frame,
        node.keys[..count].iter().zip(&node.children),
        depth,
        state,
    )?;
    let Some(byte) = query_byte_at(state.key, depth) else {
        return Ok(None);
    };
    Ok(node.keys[..count]
        .iter()
        .position(|candidate| *candidate == byte)
        .map(|index| child_offset(node.children[index])))
}

fn longest_prefix_through_node16(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<Option<u32>> {
    let node = cast::<Node16>(body);
    let count = usize::from(node.count).min(16);
    update_inner_terminator_candidate(
        frame,
        node.keys[..count].iter().zip(&node.children),
        depth,
        state,
    )?;
    Ok(query_byte_at(state.key, depth)
        .and_then(|byte| simd::node16_find_byte(&node.keys, node.count, byte))
        .map(|index| child_offset(node.children[usize::from(index)])))
}

fn longest_prefix_through_node48(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<Option<u32>> {
    let node = cast::<Node48>(body);
    update_indexed_terminator_candidate(frame, node.index[0], &node.children, depth, state)?;
    let Some(index) = query_byte_at(state.key, depth).map(|byte| node.index[usize::from(byte)])
    else {
        return Ok(None);
    };
    if index == 0 {
        return Ok(None);
    }
    let child_index = usize::from(index - 1);
    if child_index >= node.children.len() {
        return Err(Error::node_corrupt(
            "longest_prefix: node48 index out of range",
        ));
    }
    Ok(Some(child_offset(node.children[child_index])))
}

fn longest_prefix_through_node256(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<Option<u32>> {
    let node = cast::<Node256>(body);
    if node.children[0] != 0 {
        update_terminator_candidate_at(frame, child_offset(node.children[0]), depth, state)?;
    }
    Ok(query_byte_at(state.key, depth).and_then(|byte| {
        let child = node.children[usize::from(byte)];
        (child != 0).then(|| child_offset(child))
    }))
}

fn query_byte_at(key: SearchKey<'_>, depth: usize) -> Option<u8> {
    key.user_bytes().and_then(|bytes| bytes.get(depth)).copied()
}

fn update_inner_terminator_candidate<'a>(
    frame: BlobFrameRef<'_>,
    children: impl Iterator<Item = (&'a u8, &'a u16)>,
    depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<()> {
    if let Some((_, child)) = children.into_iter().find(|(byte, _)| **byte == 0) {
        update_terminator_candidate_at(frame, child_offset(*child), depth, state)?;
    }
    Ok(())
}

fn update_indexed_terminator_candidate(
    frame: BlobFrameRef<'_>,
    index: u8,
    children: &[u16],
    depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<()> {
    if index == 0 {
        return Ok(());
    }
    let child_index = usize::from(index - 1);
    if child_index >= children.len() {
        return Err(Error::node_corrupt(
            "longest_prefix: node48 terminator index out of range",
        ));
    }
    update_terminator_candidate_at(frame, child_offset(children[child_index]), depth, state)
}

fn update_terminator_candidate_at(
    frame: BlobFrameRef<'_>,
    off: u32,
    depth: usize,
    state: &mut LongestPrefixState<'_, '_>,
) -> Result<()> {
    let key = state.key;
    let Some(query) = key.user_bytes() else {
        return Err(Error::Internal("longest_prefix requires a user search key"));
    };
    if depth > query.len() {
        return Ok(());
    }
    let candidate = SearchKey::user(&query[..depth]);
    let hit = match descend(frame, off, candidate, depth + 1)? {
        LookupResult::Found(hit) => Some(LongestPrefixHit {
            key: query[..depth].to_vec(),
            value: hit.value.to_vec(),
            seq: hit.seq,
        }),
        LookupResult::NotFound => None,
        LookupResult::Crossing(crossing) => {
            exact_lookup_from_crossing(state, candidate, crossing, &query[..depth])?
        }
    };
    if let Some(hit) = hit {
        if state
            .best
            .as_ref()
            .is_none_or(|current| current.key.len() < hit.key.len())
        {
            state.best = Some(hit);
        }
    }
    Ok(())
}

fn exact_lookup_from_crossing(
    state: &mut LongestPrefixState<'_, '_>,
    key: SearchKey<'_>,
    mut crossing: BlobNodeCrossing,
    user_key: &[u8],
) -> Result<Option<LongestPrefixHit>> {
    loop {
        let Some(pin) = pin_longest_prefix_child(state, crossing.child_guid)? else {
            return Ok(None);
        };
        let version = pin.content_version();
        let guard = pin.read_optimistic();
        let frame = BlobFrameRef::wrap(guard.as_slice());
        let result = lookup_at(frame, frame.header().root_slot, key, crossing.child_depth);
        if !guard.validate() || !pin.validate_content_version(version) {
            state.unstable = true;
            return Ok(None);
        }
        let result = result?;
        drop(guard);
        state.visited.push((Arc::clone(&pin), version));
        match result {
            LookupResult::Found(hit) => {
                return Ok(Some(LongestPrefixHit {
                    key: user_key.to_vec(),
                    value: hit.value.to_vec(),
                    seq: hit.seq,
                }));
            }
            LookupResult::NotFound => return Ok(None),
            LookupResult::Crossing(next) => crossing = next,
        }
    }
}

fn pin_longest_prefix_child(
    state: &mut LongestPrefixState<'_, '_>,
    child_guid: BlobGuid,
) -> Result<Option<Arc<CachedBlob>>> {
    match state.bm.pin(child_guid) {
        Ok(pin) => Ok(Some(pin)),
        Err(error)
            if is_blob_store_not_found(&error)
                && missing_child_is_retryable(state.bm, child_guid, state.gc_epoch) =>
        {
            state.unstable = true;
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn update_longest_prefix_leaf(body: &[u8], state: &mut LongestPrefixState<'_, '_>) -> Result<()> {
    let leaf = *cast::<Leaf>(&body[..size_of::<Leaf>()]);
    if leaf.tombstone != 0 {
        return Ok(());
    }
    let key_len = usize::from(leaf.key_len);
    let value_len = usize::from(leaf.value_len);
    let key_end = size_of::<Leaf>()
        .checked_add(key_len)
        .ok_or(Error::node_corrupt("longest_prefix: key length overflow"))?;
    let value_end = key_end
        .checked_add(value_len)
        .ok_or(Error::node_corrupt("longest_prefix: value length overflow"))?;
    if value_end > body.len() {
        return Err(Error::node_corrupt(
            "longest_prefix: leaf key/value out of range",
        ));
    }
    let stored_key = &body[size_of::<Leaf>()..key_end];
    let user_key = stored_key.strip_suffix(&[0]).unwrap_or(stored_key);
    let Some(query_bytes) = state.key.user_bytes() else {
        return Err(Error::Internal("longest_prefix requires a user search key"));
    };
    if !query_bytes.starts_with(user_key)
        || state
            .best
            .as_ref()
            .is_some_and(|candidate| candidate.key.len() >= user_key.len())
    {
        return Ok(());
    }
    state.best = Some(LongestPrefixHit {
        key: user_key.to_vec(),
        value: body[key_end..value_end].to_vec(),
        seq: leaf.seq,
    });
    Ok(())
}

fn lookup_multi_with_gc_policy<R, F>(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    consume: &mut F,
    restart_on_gc: bool,
) -> Result<Option<R>>
where
    F: FnMut(LookupHit<'_>) -> R,
{
    // Outer loop: each iteration is one full attempt; we restart
    // here when an optimistic snapshot is invalidated.
    'restart: loop {
        let gc_epoch = restart_on_gc.then(|| bm.gc_read_epoch());
        if let Some(cache) = route_cache {
            if let Some(route) = cache.lookup(key) {
                match lookup_from_cached_route(bm, root_pin, cache, key, route, consume, gc_epoch)?
                {
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
        let (child_pin, child_depth) =
            match indexed_lookup_or_pin(bm, key, crossing, consume, gc_epoch)? {
                IndexedLookupOrPin::Done(result) => return Ok(result),
                IndexedLookupOrPin::Pin { pin, depth } => (pin, depth),
                IndexedLookupOrPin::Restart => {
                    if let Some(cache) = route_cache {
                        cache.clear();
                    }
                    bm.note_optimistic_restart();
                    continue 'restart;
                }
            };

        // Cross-blob hops. Same pattern; on a torn read we restart
        // the whole walk from the root (the parent BlobNode that
        // pointed us here may also have moved).
        match lookup_from_pinned_blob(
            bm,
            route_cache,
            key,
            child_pin,
            child_depth,
            consume,
            gc_epoch,
        )? {
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
    gc_epoch: Option<u64>,
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
    let frame = BlobFrameRef::wrap(parent_guard.as_slice());
    if !validate_route_edge(frame, key, route)? {
        drop(parent_guard);
        cache.invalidate(key, route);
        return Ok(RouteLookup::Stale);
    }
    if parent_version != route.parent_version {
        cache.refresh_parent_version(key, route, parent_version);
    }
    let child_pin = match indexed_lookup_or_pin(
        bm,
        key,
        BlobNodeCrossing {
            child_guid: route.child_guid,
            child_depth: route.child_depth,
        },
        consume,
        gc_epoch,
    ) {
        Ok(IndexedLookupOrPin::Done(result)) => return Ok(RouteLookup::Done(result)),
        Ok(IndexedLookupOrPin::Pin { pin, .. }) => pin,
        Ok(IndexedLookupOrPin::Restart) => {
            drop(parent_guard);
            cache.clear();
            return Ok(RouteLookup::Restart);
        }
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
        let (next_pin, next_depth) =
            match indexed_lookup_or_pin(bm, key, crossing, consume, gc_epoch)? {
                IndexedLookupOrPin::Done(result) => return Ok(RouteLookup::Done(result)),
                IndexedLookupOrPin::Pin { pin, depth } => (pin, depth),
                IndexedLookupOrPin::Restart => {
                    cache.clear();
                    return Ok(RouteLookup::Restart);
                }
            };
        match lookup_from_pinned_blob(
            bm,
            Some(cache),
            key,
            next_pin,
            next_depth,
            consume,
            gc_epoch,
        )? {
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
    gc_epoch: Option<u64>,
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
        match indexed_lookup_or_pin(bm, key, crossing, consume, gc_epoch)? {
            IndexedLookupOrPin::Done(result) => return Ok(CrossBlobLookup::Done(result)),
            IndexedLookupOrPin::Restart => {
                if let Some(cache) = route_cache {
                    cache.clear();
                }
                return Ok(CrossBlobLookup::Restart);
            }
            IndexedLookupOrPin::Pin {
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

enum IndexedLookupOrPin<R> {
    Done(Option<R>),
    Pin { pin: Arc<CachedBlob>, depth: usize },
    Restart,
}

fn indexed_lookup_or_pin<R, F>(
    bm: &BufferManager,
    key: SearchKey<'_>,
    crossing: BlobNodeCrossing,
    consume: &mut F,
    gc_epoch: Option<u64>,
) -> Result<IndexedLookupOrPin<R>>
where
    F: FnMut(LookupHit<'_>) -> R,
{
    match bm.pin_cached(crossing.child_guid) {
        Ok(Some(pin)) => {
            pin.prefetch_header();
            return Ok(IndexedLookupOrPin::Pin {
                pin,
                depth: crossing.child_depth,
            });
        }
        Ok(None) => {}
        Err(e)
            if is_blob_store_not_found(&e)
                && missing_child_is_retryable(bm, crossing.child_guid, gc_epoch) =>
        {
            return Ok(IndexedLookupOrPin::Restart);
        }
        Err(e) => return Err(e),
    }

    // Only exact point lookups (a user-style key) take the indexed path;
    // range/prefix/non-exact searches pin directly.
    if key.user_bytes().is_none() {
        let pin = match bm.pin(crossing.child_guid) {
            Ok(pin) => pin,
            Err(e)
                if is_blob_store_not_found(&e)
                    && missing_child_is_retryable(bm, crossing.child_guid, gc_epoch) =>
            {
                return Ok(IndexedLookupOrPin::Restart);
            }
            Err(e) => return Err(e),
        };
        pin.prefetch_header();
        return Ok(IndexedLookupOrPin::Pin {
            pin,
            depth: crossing.child_depth,
        });
    }

    // Answer from the checkpoint-built read index or, if that is
    // unavailable, from the in-blob routing region. Any uncertainty falls
    // back to the authoritative full pin. Exact hits, negatives, and
    // crossings are published only while the blob's read-index token is stable.
    match indexed_read_chain(bm, crossing.child_guid, key, crossing.child_depth) {
        IndexedBlobLookup::Unknown => {
            let pin = match bm.pin(crossing.child_guid) {
                Ok(pin) => pin,
                Err(e)
                    if is_blob_store_not_found(&e)
                        && missing_child_is_retryable(bm, crossing.child_guid, gc_epoch) =>
                {
                    return Ok(IndexedLookupOrPin::Restart);
                }
                Err(e) => return Err(e),
            };
            pin.prefetch_header();
            Ok(IndexedLookupOrPin::Pin {
                pin,
                depth: crossing.child_depth,
            })
        }
        IndexedBlobLookup::NotFound => {
            if gc_epoch.is_some_and(|captured| !bm.gc_epoch_still_stable(captured)) {
                Ok(IndexedLookupOrPin::Restart)
            } else {
                Ok(IndexedLookupOrPin::Done(None))
            }
        }
        IndexedBlobLookup::Found { value, seq } => {
            let mut hit = || consume(LookupHit { value: &value, seq });
            if let Some(captured) = gc_epoch {
                match bm.with_stable_gc_epoch(captured, hit) {
                    Some(out) => Ok(IndexedLookupOrPin::Done(Some(out))),
                    None => Ok(IndexedLookupOrPin::Restart),
                }
            } else {
                Ok(IndexedLookupOrPin::Done(Some(hit())))
            }
        }
        IndexedBlobLookup::Crossing { .. } => Ok(IndexedLookupOrPin::Restart),
    }
}

fn missing_child_is_retryable(
    bm: &BufferManager,
    child_guid: BlobGuid,
    gc_epoch: Option<u64>,
) -> bool {
    gc_epoch.is_some_and(|captured| bm.has_delete_fence(child_guid) || bm.gc_raced_since(captured))
}

// ---------- indexed routed read ----------

const MAX_INDEXED_CHAIN_HOPS: usize = 8;

thread_local! {
    static READ_INDEX_BUCKET_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static READ_PAGE_SCRATCH: RefCell<Option<AlignedBlobBuf>> = const { RefCell::new(None) };
}

fn indexed_read_chain(
    bm: &BufferManager,
    mut guid: BlobGuid,
    key: SearchKey<'_>,
    mut depth: usize,
) -> IndexedBlobLookup {
    for _ in 0..MAX_INDEXED_CHAIN_HOPS {
        match indexed_read(bm, guid, key, depth) {
            IndexedBlobLookup::Crossing {
                child_guid,
                child_depth,
            } => {
                guid = child_guid;
                depth = child_depth;
            }
            answer => return answer,
        }
    }
    IndexedBlobLookup::Unknown
}

/// Answer an indexed point lookup against a **routed** blob by reading only
/// the header page + routing region + one leaf page via
/// `read_blob_range`, instead of pinning the whole 512 KB frame.
///
/// A pure accelerator: it first tries the checkpoint read index, then
/// falls back to the in-blob routed layout. Any remaining uncertainty
/// falls through to `bm.pin`, which reads the authoritative image.
/// Exact read-index hits and misses are returned only while the
/// BufferManager token observed at read-index load is still valid.
///
fn indexed_read(
    bm: &BufferManager,
    guid: BlobGuid,
    key: SearchKey<'_>,
    depth: usize,
) -> IndexedBlobLookup {
    if !bm.indexed_read_eligible(guid) {
        return IndexedBlobLookup::Unknown;
    }
    if let Some(user_key) = key.user_bytes() {
        match read_index_lookup(bm, guid, user_key, depth) {
            IndexedBlobLookup::Unknown => {}
            answer => return answer,
        }
    }

    // Descend with the SAME `key` the pin-fallback frame descent uses
    // (the caller only reaches here with a user-style key), so a routed
    // and a full-frame read are byte-for-byte equivalent.
    match routed_read_cached(bm, guid, key, depth) {
        Ok(IndexedBlobLookup::Unknown) | Err(_) => {}
        Ok(answer) => return answer,
    }

    IndexedBlobLookup::Unknown
}

fn read_index_lookup(
    bm: &BufferManager,
    guid: BlobGuid,
    user_key: &[u8],
    depth: usize,
) -> IndexedBlobLookup {
    let Some((index, token)) = bm.read_index(guid) else {
        bm.note_read_index_unknown();
        return IndexedBlobLookup::Unknown;
    };

    if let Some((child_guid, child_depth)) = index.crossing(user_key, depth) {
        return if read_index_token_valid(bm, guid, token) {
            bm.note_read_index_crossing_hit();
            IndexedBlobLookup::Crossing {
                child_guid,
                child_depth,
            }
        } else {
            IndexedBlobLookup::Unknown
        };
    }

    if !index.may_have_key(user_key) {
        return if read_index_token_valid(bm, guid, token) {
            bm.note_read_index_negative_hit();
            IndexedBlobLookup::NotFound
        } else {
            IndexedBlobLookup::Unknown
        };
    }

    if let Some(answer) = read_index_leaf_hit(bm, guid, &index, token, user_key) {
        return answer;
    }

    match index.route_or_absent(user_key, depth) {
        ReadIndexAnswer::NotFound => {
            if read_index_token_valid(bm, guid, token) {
                bm.note_read_index_negative_hit();
                IndexedBlobLookup::NotFound
            } else {
                IndexedBlobLookup::Unknown
            }
        }
        ReadIndexAnswer::Crossing {
            child_guid,
            child_depth,
            ..
        } => {
            if read_index_token_valid(bm, guid, token) {
                bm.note_read_index_crossing_hit();
                IndexedBlobLookup::Crossing {
                    child_guid,
                    child_depth,
                }
            } else {
                IndexedBlobLookup::Unknown
            }
        }
    }
}

fn read_index_leaf_hit(
    bm: &BufferManager,
    guid: BlobGuid,
    index: &ReadIndex,
    token: u64,
    user_key: &[u8],
) -> Option<IndexedBlobLookup> {
    let Some(bucket_lookup) = READ_INDEX_BUCKET_BUF.with(|cell| {
        let mut bucket = cell.borrow_mut();
        bm.read_index_bucket(guid, index, user_key, &mut bucket)
            .map(|()| index.lookup_leaf_in_bucket(user_key, &bucket))
    }) else {
        bm.note_read_index_unknown();
        return Some(IndexedBlobLookup::Unknown);
    };
    match bucket_lookup {
        Ok(Some(ReadIndexHit::Inline { value, seq })) => {
            return if read_index_token_valid(bm, guid, token) {
                bm.note_read_index_inline_hit();
                Some(IndexedBlobLookup::Found { value, seq })
            } else {
                Some(IndexedBlobLookup::Unknown)
            };
        }
        Ok(Some(ReadIndexHit::ValueSegment {
            value_off,
            value_len,
            value_crc32,
            seq,
        })) => {
            let mut value = vec![0; value_len as usize];
            if bm
                .read_value_segment_range(guid, u64::from(value_off), value.as_mut_slice())
                .is_none()
            {
                bm.note_read_index_unknown();
                return Some(IndexedBlobLookup::Unknown);
            }
            if crc32fast::hash(&value) != value_crc32 {
                bm.note_read_index_unknown();
                return Some(IndexedBlobLookup::Unknown);
            }
            return if read_index_token_valid(bm, guid, token) {
                bm.note_read_index_value_hit(u64::from(value_len));
                Some(IndexedBlobLookup::Found { value, seq })
            } else {
                bm.note_read_index_unknown();
                Some(IndexedBlobLookup::Unknown)
            };
        }
        Ok(Some(ReadIndexHit::BlobOffset {
            value_off,
            value_len,
            seq,
        })) => {
            return with_read_page_scratch(|scratch| {
                let mut pages = ReadPageReader::new(bm, guid, scratch.as_mut_slice());
                let Ok(value) = read_value_paged(&mut pages, value_off, value_len) else {
                    bm.note_read_index_unknown();
                    return Some(IndexedBlobLookup::Unknown);
                };
                if read_index_token_valid(bm, guid, token) {
                    bm.note_read_index_offset_hit();
                    Some(IndexedBlobLookup::Found { value, seq })
                } else {
                    bm.note_read_index_unknown();
                    Some(IndexedBlobLookup::Unknown)
                }
            });
        }
        Ok(None) => {}
        Err(_) => {
            bm.note_read_index_unknown();
            return Some(IndexedBlobLookup::Unknown);
        }
    }
    None
}

#[inline]
fn read_index_token_valid(bm: &BufferManager, guid: BlobGuid, token: u64) -> bool {
    let valid = bm.read_index_token_valid(guid, token);
    if !valid {
        bm.note_read_index_unknown();
    }
    valid
}

fn with_read_page_scratch<R>(f: impl FnOnce(&mut AlignedBlobBuf) -> R) -> R {
    READ_PAGE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.is_none() {
            *scratch = Some(AlignedBlobBuf::zeroed());
        }
        f(scratch.as_mut().expect("cold page scratch initialized"))
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RoutedLookup {
    Unknown,
    Found {
        value: Vec<u8>,
        seq: u64,
    },
    Crossing {
        child_guid: BlobGuid,
        child_depth: usize,
    },
    NegativeHint,
}

/// Routed indexed read with the read-page cache: cache reusable
/// header/routing pages immediately and admit leaf pages only after a
/// repeated cold touch. Exact hits and negatives are validated against
/// a stable cold blob stamp before they become user-visible. Local
/// child crossings remain advisory because the live walker owns
/// cross-blob route validation.
fn routed_read_cached(
    bm: &BufferManager,
    guid: BlobGuid,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<IndexedBlobLookup> {
    with_read_page_scratch(|scratch| {
        let buf = scratch.as_mut_slice();

        let mut pages = ReadPageReader::new(bm, guid, buf);
        pages.load_range(0, HEADER_SIZE, ReadPageAdmission::Navigation)?;
        let (root_off, rr, stamp) = {
            let frame = pages.frame();
            let h = frame.header();
            match h.routing_region() {
                Some(rr) => (
                    root_child_offset(h.root_slot, "routed_read_cached: root child")?,
                    rr,
                    ReadIndexStamp::new(h),
                ),
                None => return Ok(IndexedBlobLookup::Unknown), // legacy -> full pin
            }
        };

        match descend_routed_paged(&mut pages, root_off, key, depth, rr.leaf_region_start)? {
            RoutedLookup::Unknown => Ok(IndexedBlobLookup::Unknown),
            RoutedLookup::Crossing {
                child_guid,
                child_depth,
            } => {
                if validate_read_index_stamp(bm, guid, buf, stamp)? {
                    Ok(IndexedBlobLookup::Crossing {
                        child_guid,
                        child_depth,
                    })
                } else {
                    Ok(IndexedBlobLookup::Unknown)
                }
            }
            RoutedLookup::NegativeHint => {
                if validate_read_index_stamp(bm, guid, buf, stamp)? {
                    Ok(IndexedBlobLookup::NotFound)
                } else {
                    Ok(IndexedBlobLookup::Unknown)
                }
            }
            RoutedLookup::Found { value, seq } => {
                if validate_read_index_stamp(bm, guid, buf, stamp)? {
                    Ok(IndexedBlobLookup::Found { value, seq })
                } else {
                    Ok(IndexedBlobLookup::Unknown)
                }
            }
        }
    })
}

const READ_PAGES_PER_BLOB: usize = (crate::layout::PAGE_SIZE as usize) / (PAGE_4K as usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadPageAdmission {
    None,
    Navigation,
    Leaf,
}

struct ReadPageReader<'a> {
    bm: &'a BufferManager,
    guid: BlobGuid,
    scratch: &'a mut [u8],
    loaded: [bool; READ_PAGES_PER_BLOB],
}

impl<'a> ReadPageReader<'a> {
    fn new(bm: &'a BufferManager, guid: BlobGuid, scratch: &'a mut [u8]) -> Self {
        Self {
            bm,
            guid,
            scratch,
            loaded: [false; READ_PAGES_PER_BLOB],
        }
    }

    fn frame(&self) -> BlobFrameRef<'_> {
        BlobFrameRef::wrap(&self.scratch[..])
    }

    fn load_range(&mut self, start: u32, end: u32, admission: ReadPageAdmission) -> Result<()> {
        if end <= start {
            return Ok(());
        }
        if end > crate::layout::PAGE_SIZE {
            return Err(Error::node_corrupt("indexed_read: page range"));
        }
        let mut page = (start / PAGE_4K) as u16;
        let end_page = (page_align_up(end) / PAGE_4K) as u16;
        while page < end_page {
            if self.loaded[usize::from(page)] {
                page += 1;
                continue;
            }
            if admission != ReadPageAdmission::None && self.try_fill_cached_page(page) {
                page += 1;
                continue;
            }

            let run_start = page;
            let mut run_end = page + 1;
            while run_end < end_page {
                if self.loaded[usize::from(run_end)] {
                    break;
                }
                if admission != ReadPageAdmission::None && self.try_fill_cached_page(run_end) {
                    break;
                }
                run_end += 1;
            }

            self.load_page_run(run_start, run_end, admission)?;
            page = run_end;
        }
        Ok(())
    }

    fn try_fill_cached_page(&mut self, page: u16) -> bool {
        let page_usize = usize::from(page);
        debug_assert!(!self.loaded[page_usize]);
        let start = page_usize * PAGE_4K as usize;
        let end = start + PAGE_4K as usize;
        let dst = &mut self.scratch[start..end];
        if self.bm.read_page_cached(self.guid, page, dst) {
            self.loaded[page_usize] = true;
            true
        } else {
            false
        }
    }

    fn load_page_run(
        &mut self,
        start_page: u16,
        end_page: u16,
        admission: ReadPageAdmission,
    ) -> Result<()> {
        debug_assert!(start_page < end_page);
        let start = usize::from(start_page) * PAGE_4K as usize;
        let end = usize::from(end_page) * PAGE_4K as usize;
        self.bm
            .read_blob_range(self.guid, start as u64, &mut self.scratch[start..end])?;

        for page in start_page..end_page {
            self.loaded[usize::from(page)] = true;
        }

        if admission != ReadPageAdmission::None && self.bm.indexed_read_eligible(self.guid) {
            for page in start_page..end_page {
                let page_start = usize::from(page) * PAGE_4K as usize;
                let page_end = page_start + PAGE_4K as usize;
                let src = &self.scratch[page_start..page_end];
                match admission {
                    ReadPageAdmission::None => {}
                    ReadPageAdmission::Navigation => self.bm.read_page_store(self.guid, page, src),
                    ReadPageAdmission::Leaf => self.bm.read_leaf_page_store(self.guid, page, src),
                }
            }
        }
        Ok(())
    }
}

fn validate_read_index_stamp(
    bm: &BufferManager,
    guid: BlobGuid,
    scratch: &mut [u8],
    expected: ReadIndexStamp,
) -> Result<bool> {
    if !bm.indexed_read_eligible(guid) {
        return Ok(false);
    }
    bm.read_blob_range(guid, 0, &mut scratch[..HEADER_SIZE as usize])?;
    if !bm.indexed_read_eligible(guid) {
        return Ok(false);
    }
    let frame = BlobFrameRef::wrap(&scratch[..]);
    Ok(ReadIndexStamp::new(frame.header()) == expected)
}

fn descend_routed_paged(
    pages: &mut ReadPageReader<'_>,
    off: u32,
    key: SearchKey<'_>,
    depth: usize,
    leaf_region_start: u32,
) -> Result<RoutedLookup> {
    if off >= leaf_region_start {
        page_in_leaf_paged(pages, off)?;
        let frame = pages.frame();
        let body = frame
            .body_at_offset(off)
            .ok_or(Error::node_corrupt("indexed_read: leaf body range"))?;
        return leaf_check_owned(body, key);
    }

    page_in_fixed_node(pages, off)?;
    let step = routed_step(pages.frame(), off, key, depth)?;
    match step {
        RoutedStep::Done(answer) => Ok(answer),
        RoutedStep::Visit(child_off, new_depth) => {
            descend_routed_paged(pages, child_off, key, new_depth, leaf_region_start)
        }
    }
}

fn page_in_fixed_node(pages: &mut ReadPageReader<'_>, off: u32) -> Result<()> {
    pages.load_range(off, off + 2, ReadPageAdmission::Navigation)?;
    let ntype = pages
        .frame()
        .ntype_at(off)
        .ok_or(Error::node_corrupt("indexed_read: undecodable node type"))?;
    if ntype == NodeType::Leaf || ntype == NodeType::Invalid {
        return Ok(());
    }
    let end = off
        .checked_add(size_of_node(ntype))
        .ok_or(Error::node_corrupt("indexed_read: node size overflow"))?;
    pages.load_range(off, end, ReadPageAdmission::Navigation)
}

fn page_in_leaf_paged(pages: &mut ReadPageReader<'_>, loff: u32) -> Result<()> {
    let hdr_end = page_align_up(loff + size_of::<Leaf>() as u32);
    pages.load_range(loff, hdr_end, ReadPageAdmission::Leaf)?;
    let (key_len, value_len) = {
        let leaf = cast::<Leaf>(&pages.scratch[loff as usize..loff as usize + size_of::<Leaf>()]);
        (u32::from(leaf.key_len), u32::from(leaf.value_len))
    };
    let body_end = page_align_up(loff + leaf_body_size(key_len, value_len));
    pages.load_range(hdr_end, body_end, ReadPageAdmission::Leaf)
}

fn read_value_paged(
    pages: &mut ReadPageReader<'_>,
    value_off: u32,
    value_len: u32,
) -> Result<Vec<u8>> {
    let value_end = value_off
        .checked_add(value_len)
        .ok_or(Error::node_corrupt("indexed_read: value range"))?;
    if value_end > crate::layout::PAGE_SIZE {
        return Err(Error::node_corrupt("indexed_read: value range"));
    }
    pages.load_range(value_off, value_end, ReadPageAdmission::Leaf)?;
    let start = value_off as usize;
    let end = value_end as usize;
    Ok(pages.scratch[start..end].to_vec())
}

/// Routed-read core, decoupled from the buffer manager via a
/// `read_range(byte_offset, dst)` closure so it can be unit-tested
/// against an in-memory routed frame.
///
/// `scratch` must be a `PAGE_SIZE`, 4 KB-aligned, zeroed buffer; the
/// header page, routing region, and one leaf page are read into it at
/// their absolute offsets. Returns `Unknown` for a legacy
/// (`routing_len == 0`) blob.
#[cfg(test)]
pub(super) fn indexed_read_into(
    scratch: &mut [u8],
    read_range: &mut dyn FnMut(u64, &mut [u8]) -> Result<()>,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<IndexedBlobLookup> {
    // Header page → routing geometry + root.
    read_range(0, &mut scratch[..HEADER_SIZE as usize])?;
    let (root_off, rr) = {
        let frame = BlobFrameRef::wrap(&scratch[..]);
        let h = frame.header();
        match h.routing_region() {
            Some(rr) => (
                root_child_offset(h.root_slot, "indexed_read_into: root child")?,
                rr,
            ),
            None => return Ok(IndexedBlobLookup::Unknown), // legacy → full pin
        }
    };
    // Routing region (internal nodes): [routing_off, leaf_region_start)
    // — both page-aligned, so the read length is a 4 KB multiple.
    read_range(
        u64::from(rr.off),
        &mut scratch[rr.off as usize..rr.leaf_region_start as usize],
    )?;
    match descend_routed(
        scratch,
        read_range,
        root_off,
        key,
        depth,
        rr.leaf_region_start,
    )? {
        RoutedLookup::Unknown => Ok(IndexedBlobLookup::Unknown),
        RoutedLookup::Found { value, seq } => Ok(IndexedBlobLookup::Found { value, seq }),
        RoutedLookup::Crossing {
            child_guid,
            child_depth,
        } => Ok(IndexedBlobLookup::Crossing {
            child_guid,
            child_depth,
        }),
        RoutedLookup::NegativeHint => Ok(IndexedBlobLookup::NotFound),
    }
}

/// One step of the routed descent: the next child offset to visit, or a
/// terminal answer.
enum RoutedStep {
    Visit(u32, usize),
    Done(RoutedLookup),
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
    let not_found = RoutedStep::Done(RoutedLookup::NegativeHint);
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
                    return Err(Error::node_corrupt(
                        "indexed_read: node48 index out of range",
                    ));
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
                RoutedStep::Done(RoutedLookup::Crossing {
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
            RoutedStep::Done(RoutedLookup::Unknown)
        }
    })
}

#[cfg(test)]
fn descend_routed(
    scratch: &mut [u8],
    read_range: &mut dyn FnMut(u64, &mut [u8]) -> Result<()>,
    off: u32,
    key: SearchKey<'_>,
    depth: usize,
    leaf_region_start: u32,
) -> Result<RoutedLookup> {
    // The decision is taken (and copied out) under a short frame borrow
    // so we can page in a leaf or recurse with `&mut scratch` after.
    let step = routed_step(BlobFrameRef::wrap(&scratch[..]), off, key, depth)?;
    match step {
        RoutedStep::Done(answer) => Ok(answer),
        RoutedStep::Visit(child_off, new_depth) => {
            if child_off >= leaf_region_start {
                page_in_leaf(scratch, read_range, child_off)?;
                let frame = BlobFrameRef::wrap(&scratch[..]);
                let body = frame
                    .body_at_offset(child_off)
                    .ok_or(Error::node_corrupt("indexed_read: leaf body range"))?;
                leaf_check_owned(body, key)
            } else {
                descend_routed(
                    scratch,
                    read_range,
                    child_off,
                    key,
                    new_depth,
                    leaf_region_start,
                )
            }
        }
    }
}

/// Page the leaf at `loff` (>= leaf_region_start) into `scratch` at its
/// absolute offset: read the page(s) covering its 16-byte header, then
/// extend to cover the full `[16B hdr][key][value]` body (a large
/// value can straddle pages).
#[cfg(test)]
fn page_in_leaf(
    scratch: &mut [u8],
    read_range: &mut dyn FnMut(u64, &mut [u8]) -> Result<()>,
    loff: u32,
) -> Result<()> {
    let page0 = loff & !(PAGE_4K - 1);
    let hdr_end = page_align_up(loff + size_of::<Leaf>() as u32);
    read_range(
        u64::from(page0),
        &mut scratch[page0 as usize..hdr_end as usize],
    )?;
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

/// Like `leaf_check` but returns an owned [`IndexedBlobLookup`] — the value
/// is copied out of the paged-in buffer, which the caller drops.
fn leaf_check_owned(body: &[u8], key: SearchKey<'_>) -> Result<RoutedLookup> {
    let leaf = *cast::<Leaf>(&body[..size_of::<Leaf>()]);
    if leaf.tombstone != 0 {
        return Ok(RoutedLookup::NegativeHint);
    }
    if leaf.key_fp != 0 && leaf.key_fp != key.fingerprint() {
        return Ok(RoutedLookup::NegativeHint);
    }
    let key_len = leaf.key_len as usize;
    let value_len = leaf.value_len as usize;
    let key_end = 16 + key_len;
    let value_end = key_end + value_len;
    if value_end > body.len() {
        return Err(Error::node_corrupt("indexed_read: leaf key/value range"));
    }
    if !key.eq_slice(&body[16..key_end]) {
        return Ok(RoutedLookup::NegativeHint);
    }
    Ok(RoutedLookup::Found {
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

#[cfg(test)]
mod tests {
    use super::super::erase::erase;
    use super::super::insert::insert;
    use super::super::migrate::compact_blob;
    use super::*;
    use crate::store::blob_store::{BlobStore, FileBlobStore, MemoryBlobStore};
    use crate::store::{encode_child_off, BlobFrame, WriteThroughEntry};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use tempfile::tempdir;

    fn routed_blob(guid: BlobGuid) -> AlignedBlobBuf {
        let mut buf = AlignedBlobBuf::zeroed();
        BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
        {
            let mut frame = BlobFrame::wrap(buf.as_mut_slice());
            let mut seq = 1u64;
            for i in 0..60u8 {
                let value = if i % 7 == 0 {
                    vec![i; 5000]
                } else {
                    vec![i, i ^ 0xFF]
                };
                let root = frame.header().root_slot;
                insert(&mut frame, root, &[b'q', i], &value, seq).unwrap();
                seq += 1;
            }
            for i in 0..20u8 {
                let root = frame.header().root_slot;
                insert(&mut frame, root, &[b'p', i], &[i, 0xAB, i], seq).unwrap();
                seq += 1;
            }
            let root = frame.header().root_slot;
            for i in (0..60u8).step_by(2) {
                erase(&mut frame, root, &[b'q', i]).unwrap();
            }
        }
        compact_blob(&mut buf).unwrap();
        assert!(
            BlobFrame::wrap(buf.as_mut_slice())
                .header()
                .routing_region()
                .is_some(),
            "test needs a routed blob",
        );
        buf
    }

    fn install_exact_leaf(frame: &mut BlobFrame<'_>, key: &[u8], value: &[u8], seq: u64) {
        let total = leaf_body_size(key.len() as u32, value.len() as u32);
        let slot = frame.alloc_leaf(total).unwrap().slot;
        let off = frame.offset_of_slot(slot).unwrap();
        let body = frame.bytes_at_mut(off, total).unwrap();
        let leaf = Leaf::live(key.len() as u16, value.len() as u16, seq, 0);
        let hdr = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref::<Leaf>(&leaf).cast::<u8>(),
                size_of::<Leaf>(),
            )
        };
        body[..size_of::<Leaf>()].copy_from_slice(hdr);
        body[size_of::<Leaf>()..size_of::<Leaf>() + key.len()].copy_from_slice(key);
        let value_start = size_of::<Leaf>() + key.len();
        body[value_start..value_start + value.len()].copy_from_slice(value);
        frame.header_mut().root_slot = encode_child_off(off);
    }

    struct HeaderFlipStore {
        guid: BlobGuid,
        first: AlignedBlobBuf,
        second_header: AlignedBlobBuf,
        header_reads: AtomicUsize,
        full_reads: AtomicUsize,
    }

    impl HeaderFlipStore {
        fn new(guid: BlobGuid, first: AlignedBlobBuf, second_header: AlignedBlobBuf) -> Self {
            Self {
                guid,
                first,
                second_header,
                header_reads: AtomicUsize::new(0),
                full_reads: AtomicUsize::new(0),
            }
        }
    }

    struct RangeCountStore {
        guid: BlobGuid,
        blob: AlignedBlobBuf,
        ranges: Mutex<Vec<(u64, usize)>>,
    }

    impl RangeCountStore {
        fn new(guid: BlobGuid) -> Self {
            let mut blob = AlignedBlobBuf::zeroed();
            for (idx, byte) in blob.as_mut_slice().iter_mut().enumerate() {
                *byte = (idx / PAGE_4K as usize) as u8;
            }
            Self::from_blob(guid, blob)
        }

        fn from_blob(guid: BlobGuid, blob: AlignedBlobBuf) -> Self {
            Self {
                guid,
                blob,
                ranges: Mutex::new(Vec::new()),
            }
        }
    }

    impl BlobStore for RangeCountStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            assert_eq!(guid, self.guid);
            dst.as_mut_slice().copy_from_slice(self.blob.as_slice());
            Ok(())
        }

        fn read_blob_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<()> {
            assert_eq!(guid, self.guid);
            self.ranges.lock().unwrap().push((byte_offset, dst.len()));
            let off = byte_offset as usize;
            dst.copy_from_slice(&self.blob.as_slice()[off..off + dst.len()]);
            Ok(())
        }

        fn write_blob(&self, _guid: BlobGuid, _src: &AlignedBlobBuf) -> Result<()> {
            unreachable!("test store is read-only")
        }

        fn delete_blob(&self, _guid: BlobGuid) -> Result<()> {
            unreachable!("test store is read-only")
        }

        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            Ok(vec![self.guid])
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }

        fn needs_flush(&self) -> bool {
            false
        }
    }

    impl BlobStore for HeaderFlipStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            assert_eq!(guid, self.guid);
            self.full_reads.fetch_add(1, Ordering::Relaxed);
            dst.as_mut_slice().copy_from_slice(self.first.as_slice());
            Ok(())
        }

        fn read_blob_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<()> {
            assert_eq!(guid, self.guid);
            let source = if byte_offset == 0 {
                let read = self.header_reads.fetch_add(1, Ordering::Relaxed);
                if read == 0 {
                    &self.first
                } else {
                    &self.second_header
                }
            } else {
                &self.first
            };
            let off = byte_offset as usize;
            dst.copy_from_slice(&source.as_slice()[off..off + dst.len()]);
            Ok(())
        }

        fn write_blob(&self, _guid: BlobGuid, _src: &AlignedBlobBuf) -> Result<()> {
            unreachable!("test store is read-only")
        }

        fn delete_blob(&self, _guid: BlobGuid) -> Result<()> {
            unreachable!("test store is read-only")
        }

        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            Ok(vec![self.guid])
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }

        fn needs_flush(&self) -> bool {
            false
        }
    }

    #[test]
    fn read_page_reader_coalesces_adjacent_misses() {
        let guid = [0x41; 16];
        let store = Arc::new(RangeCountStore::new(guid));
        let store_dyn: Arc<dyn crate::store::blob_store::BlobStore> = store.clone();
        let bm = BufferManager::new_file(store_dyn, 128, || {
            // SAFETY: The read-page reader fills the requested ranges
            // before the test inspects them.
            unsafe { AlignedBlobBuf::uninit() }
        });

        let mut scratch = AlignedBlobBuf::zeroed();
        let mut reader = ReadPageReader::new(&bm, guid, scratch.as_mut_slice());
        reader
            .load_range(PAGE_4K, PAGE_4K * 4, ReadPageAdmission::Navigation)
            .expect("first cold range read");

        assert_eq!(
            *store.ranges.lock().unwrap(),
            vec![(u64::from(PAGE_4K), (PAGE_4K * 3) as usize)],
            "adjacent cache misses should be one backing read"
        );
        assert_eq!(scratch.as_slice()[PAGE_4K as usize], 1);
        assert_eq!(scratch.as_slice()[(PAGE_4K * 3) as usize], 3);

        let mut scratch2 = AlignedBlobBuf::zeroed();
        let mut reader2 = ReadPageReader::new(&bm, guid, scratch2.as_mut_slice());
        reader2
            .load_range(PAGE_4K, PAGE_4K * 4, ReadPageAdmission::Navigation)
            .expect("second cold range read");

        assert_eq!(
            store.ranges.lock().unwrap().len(),
            1,
            "page cache should satisfy the second reader"
        );
        assert_eq!(scratch2.as_slice()[PAGE_4K as usize], 1);
        assert_eq!(scratch2.as_slice()[(PAGE_4K * 3) as usize], 3);
    }

    #[test]
    fn repeated_indexed_hits_admit_leaf_pages_after_second_touch() {
        let guid = [0x46; 16];
        let blob = routed_blob(guid);
        let store = Arc::new(RangeCountStore::from_blob(guid, blob));
        let store_dyn: Arc<dyn crate::store::blob_store::BlobStore> = store.clone();
        let bm = BufferManager::new_file(store_dyn, 128, AlignedBlobBuf::zeroed);
        let key = SearchKey::exact(&[b'q', 7]);

        let read = || match indexed_read(&bm, guid, key, 0) {
            IndexedBlobLookup::Found { value, .. } => assert_eq!(value, vec![7; 5000]),
            other => panic!("expected cold routed hit: {other:?}"),
        };

        read();
        let first = store.ranges.lock().unwrap().len();
        read();
        let second = store.ranges.lock().unwrap().len();
        read();
        let third = store.ranges.lock().unwrap().len();

        let second_delta = second - first;
        let third_delta = third - second;
        assert!(
            third_delta < second_delta,
            "third read should reuse leaf pages admitted on the second touch; ranges={:?}",
            *store.ranges.lock().unwrap()
        );
    }

    #[test]
    fn read_index_hit_reads_large_value_from_value_segment() {
        let dir = tempdir().unwrap();
        let guid = [0x47; 16];
        let value = vec![0xD3; 4096];
        let mut bytes = AlignedBlobBuf::zeroed();
        {
            BlobFrame::init(bytes.as_mut_slice(), guid).unwrap();
            let mut frame = BlobFrame::wrap(bytes.as_mut_slice());
            install_exact_leaf(&mut frame, b"large-value-key\0", &value, 11);
        }

        let store = Arc::new(FileBlobStore::open(dir.path()).unwrap());
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        {
            let bm = BufferManager::new_file(store_dyn.clone(), 128, AlignedBlobBuf::zeroed);
            bm.write_through_batch(&[WriteThroughEntry {
                guid,
                bytes,
                expected_seq: 11,
                content_version: None,
            }])
            .unwrap();
            bm.flush_inner().unwrap();
        }

        let bm = BufferManager::new_file(store_dyn, 128, AlignedBlobBuf::zeroed);
        match indexed_read(&bm, guid, SearchKey::user(b"large-value-key"), 0) {
            IndexedBlobLookup::Found { value: got, seq } => {
                assert_eq!(got, value);
                assert_eq!(seq, 11);
            }
            other => panic!("large value should come from value.seg: {other:?}"),
        }
        let stats = bm.stats();
        assert_eq!(stats.read_index_value_hits, 1);
        assert_eq!(stats.read_index_value_read_bytes, value.len() as u64);
        assert_eq!(stats.read_index_offset_hits, 0);
    }

    #[test]
    fn stale_gc_epoch_does_not_consume_indexed_hit() {
        let dir = tempdir().unwrap();
        let guid = [0x49; 16];
        let value = vec![0xC4; 128];
        let mut bytes = AlignedBlobBuf::zeroed();
        {
            BlobFrame::init(bytes.as_mut_slice(), guid).unwrap();
            let mut frame = BlobFrame::wrap(bytes.as_mut_slice());
            install_exact_leaf(&mut frame, b"consumer-key\0", &value, 21);
        }

        let store = Arc::new(FileBlobStore::open(dir.path()).unwrap());
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        {
            let bm = BufferManager::new_file(store_dyn.clone(), 128, AlignedBlobBuf::zeroed);
            bm.write_through_batch(&[WriteThroughEntry {
                guid,
                bytes,
                expected_seq: 21,
                content_version: None,
            }])
            .unwrap();
            bm.flush_inner().unwrap();
        }

        let bm = BufferManager::new_file(store_dyn, 128, AlignedBlobBuf::zeroed);
        let captured = bm.gc_read_epoch();
        bm.gc_sweep_unreachable(&std::collections::HashSet::from([guid]))
            .unwrap();

        let calls = AtomicUsize::new(0);
        let mut consume = |hit: LookupHit<'_>| {
            calls.fetch_add(1, Ordering::Relaxed);
            hit.value.to_vec()
        };
        let crossing = BlobNodeCrossing {
            child_guid: guid,
            child_depth: 0,
        };
        assert!(matches!(
            indexed_lookup_or_pin(
                &bm,
                SearchKey::user(b"consumer-key"),
                crossing,
                &mut consume,
                Some(captured),
            )
            .unwrap(),
            IndexedLookupOrPin::Restart
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        let stable = bm.gc_read_epoch();
        let result = indexed_lookup_or_pin(
            &bm,
            SearchKey::user(b"consumer-key"),
            crossing,
            &mut consume,
            Some(stable),
        )
        .unwrap();
        match result {
            IndexedLookupOrPin::Done(Some(got)) => assert_eq!(got, value),
            _ => panic!("stable indexed hit was not published"),
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stable_point_lookup_never_retries_missing_child_from_delete_fence() {
        let child = [0x47; 16];
        let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
        let bm = BufferManager::new(inner, 4);
        bm.mark_for_delete(child, 1);

        assert!(
            !missing_child_is_retryable(&bm, child, None),
            "stable point lookup must surface a missing referenced child",
        );
        assert!(
            missing_child_is_retryable(&bm, child, Some(bm.gc_read_epoch())),
            "GC-fenced readers may restart while the delete fence is live",
        );
    }

    #[test]
    fn corrupt_value_segment_is_not_returned() {
        use std::fs::OpenOptions;
        use std::os::unix::fs::FileExt;

        let dir = tempdir().unwrap();
        let guid = [0x48; 16];
        let value = vec![0xB7; 4096];
        let mut bytes = AlignedBlobBuf::zeroed();
        {
            BlobFrame::init(bytes.as_mut_slice(), guid).unwrap();
            let mut frame = BlobFrame::wrap(bytes.as_mut_slice());
            install_exact_leaf(&mut frame, b"large-value-key\0", &value, 12);
        }

        let store = Arc::new(FileBlobStore::open(dir.path()).unwrap());
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        {
            let bm = BufferManager::new_file(store_dyn.clone(), 128, AlignedBlobBuf::zeroed);
            bm.write_through_batch(&[WriteThroughEntry {
                guid,
                bytes,
                expected_seq: 12,
                content_version: None,
            }])
            .unwrap();
            bm.flush_inner().unwrap();
        }

        let value_segments = OpenOptions::new()
            .write(true)
            .open(dir.path().join("value.seg"))
            .unwrap();
        value_segments.write_at(&[value[0] ^ 0xff], 0).unwrap();

        let bm = BufferManager::new_file(store_dyn, 128, AlignedBlobBuf::zeroed);
        assert_eq!(
            indexed_read(&bm, guid, SearchKey::user(b"large-value-key"), 0),
            IndexedBlobLookup::Unknown,
            "corrupt advisory value-segment payload must fall back to the authoritative blob"
        );
        let stats = bm.stats();
        assert_eq!(stats.read_index_value_hits, 0);
        assert_eq!(stats.read_index_unknowns, 1);
    }

    #[test]
    fn production_indexed_read_publishes_stamp_validated_negative() {
        let guid = [0x42; 16];
        let blob = routed_blob(guid);
        let src = blob.as_slice().to_vec();

        let mut scratch = AlignedBlobBuf::zeroed();
        let mut read = |off: u64, dst: &mut [u8]| -> Result<()> {
            let off = off as usize;
            dst.copy_from_slice(&src[off..off + dst.len()]);
            Ok(())
        };
        assert!(
            matches!(
                indexed_read_into(
                    scratch.as_mut_slice(),
                    &mut read,
                    SearchKey::exact(b"absent"),
                    0,
                )
                .unwrap(),
                IndexedBlobLookup::NotFound
            ),
            "the routed core may prove local absence",
        );

        let store = Arc::new(MemoryBlobStore::new());
        store.write_blob(guid, &blob).unwrap();
        let bm = BufferManager::new(store, 4);

        match indexed_read(&bm, guid, SearchKey::exact(&[b'q', 1]), 0) {
            IndexedBlobLookup::Found { value, seq } => {
                assert_eq!(value, vec![1, 1 ^ 0xFF]);
                assert_eq!(seq, 2);
            }
            other => panic!("present key should stay on the positive indexed path: {other:?}"),
        }

        assert!(
            matches!(
                indexed_read(&bm, guid, SearchKey::exact(b"absent"), 0),
                IndexedBlobLookup::NotFound
            ),
            "stable routed negatives should avoid the authoritative full pin",
        );
    }

    #[test]
    fn production_indexed_read_rejects_stale_negative_without_full_pin() {
        let guid = [0x43; 16];
        let blob = routed_blob(guid);
        let mut derouted = blob.clone();
        {
            let mut frame = BlobFrame::wrap(derouted.as_mut_slice());
            frame.header_mut().routing_len = 0;
        }
        let store = Arc::new(HeaderFlipStore::new(guid, blob, derouted));
        let bm = BufferManager::new(store.clone(), 4);

        assert!(
            matches!(
                indexed_read(&bm, guid, SearchKey::exact(b"absent"), 0),
                IndexedBlobLookup::Unknown
            ),
            "a changed header stamp must force the authoritative full-pin path",
        );
        assert_eq!(
            store.full_reads.load(Ordering::Relaxed),
            0,
            "private indexed_read reports uncertainty; the caller owns the full pin",
        );
    }

    #[test]
    fn production_indexed_read_rejects_stale_routed_found() {
        let guid = [0x45; 16];
        let blob = routed_blob(guid);
        let mut derouted = blob.clone();
        {
            let mut frame = BlobFrame::wrap(derouted.as_mut_slice());
            frame.header_mut().routing_len = 0;
        }
        let store = Arc::new(HeaderFlipStore::new(guid, blob, derouted));
        let bm = BufferManager::new(store.clone(), 4);

        assert!(
            matches!(
                indexed_read(&bm, guid, SearchKey::exact(&[b'q', 1]), 0),
                IndexedBlobLookup::Unknown
            ),
            "a changed header stamp must not publish a stale routed hit",
        );
        assert_eq!(
            store.full_reads.load(Ordering::Relaxed),
            0,
            "stale positive validation must fall back without bm.pin/read_blob here",
        );
    }
}
