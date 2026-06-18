//! Insert path — `insert` / `insert_multi` + recursive
//! `insert_at` dispatch + per-NodeType arms.

use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::layout::{leaf_body_size, BlobNode, Leaf, NodeType, BLOB_MAX_INLINE};
use std::sync::Arc;

use super::cast;
use super::cow::{child_is_snapshot_shared, fork_child_if_shared};
use super::lookup::lookup_at;
use super::migrate::blob_needs_compaction;
use super::readers::{child_offset, ntype_of, read_leaf_key_ref, read_prefix};
use super::route::{pin_route_parent, validate_route_edge};
use super::spillover::{compact_blob, spillover_blob};
use super::types::{is_stale_blob_crossing, stale_blob_crossing};
use super::types::{InsertCondition, InsertOutcome, InsertReturn, LookupResult};
use super::writers::{
    inner_add_child, inner_find_child, inner_update_child, set_prefix_child, write_leaf,
    write_node4_with, write_prefix_chain, write_struct_at,
};
use super::SearchKey;
use super::MAX_SPILLOVER_ATTEMPTS;
use crate::engine::RouteCache;
use crate::store::BlobWriteGuard;
use crate::store::{decode_child_off, encode_child_off};
use crate::store::{BlobFrame, BlobFrameRef, BufferManager, CachedBlob};

// ---------- public entry points ----------

/// Single-blob insert. Surfaces [`Error::NotYetImplemented`] if
/// the descent has to follow a matching [`NodeType::Blob`]
/// crossing — callers that need cross-blob support should use
/// [`insert_multi`]. Divergent BlobNode inline prefixes can still
/// be split locally in the current blob.
///
/// `seq` is the journal sequence number to stamp on the new leaf
/// (callers should pass a monotonically-increasing value). Updates
/// `header.root_slot` in place.
#[cfg(test)]
pub(super) fn insert(
    frame: &mut BlobFrame<'_>,
    root_enc: u16,
    key: &[u8],
    value: &[u8],
    seq: u64,
) -> Result<InsertOutcome> {
    let key = SearchKey::exact(key);
    if key.len() > u16::MAX as usize {
        return Err(Error::KeyTooLong { len: key.len() });
    }
    if value.len() > u16::MAX as usize {
        return Err(Error::ValueTooLong { len: value.len() });
    }
    let r = insert_at(frame, decode_child_off(root_enc), key, value, 0, seq)?;
    frame.header_mut().root_slot = encode_child_off(r.off_after);
    Ok(InsertOutcome {
        root_dirty: true,
        mutated: true,
    })
}

/// Multi-blob insert. Pins the root via the [`BufferManager`] and
/// walks across [`NodeType::Blob`] crossings, automatically
/// triggering `splitBlob` spillover when any blob hits
/// [`crate::store::AllocError::OutOfSpace`].
///
/// Child blobs encountered during descent are pinned in the same
/// BM cache and mutated in place. The walker tags every touched
/// child via `bm.mark_dirty(child_guid, seq)`; the actual
/// store write is the checkpoint round's job (and only happens
/// after the WAL record for `seq` is durable — invariant W2D).
///
pub fn insert_multi(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
) -> Result<InsertOutcome> {
    insert_multi_conditional(
        bm,
        root_pin,
        route_cache,
        key,
        value,
        seq,
        InsertCondition::Always,
    )
}

/// Conditional variant of [`insert_multi`]. Used by the public
/// compare-and-set APIs so the existence/version check and mutation
/// happen while the target blob is exclusively latched.
#[allow(clippy::too_many_arguments)]
pub fn insert_multi_conditional(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
) -> Result<InsertOutcome> {
    if key.len() > u16::MAX as usize {
        return Err(Error::KeyTooLong { len: key.len() });
    }
    if value.len() > u16::MAX as usize {
        return Err(Error::ValueTooLong { len: value.len() });
    }
    let mut restarts = 0u32;
    loop {
        match insert_multi_conditional_once(bm, root_pin, route_cache, key, value, seq, condition) {
            Err(e) if is_stale_blob_crossing(&e) => {
                if let Some(cache) = route_cache {
                    cache.clear();
                }
                bm.note_optimistic_restart();
                restarts = restarts.saturating_add(1);
                if restarts >= 128 {
                    return Err(e);
                }
                if restarts % 16 == 0 {
                    std::thread::yield_now();
                }
            }
            out => return out,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_multi_conditional_once(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
) -> Result<InsertOutcome> {
    let mut blob_hops = 0u64;
    let mut max_cross_blob_depth = 0usize;

    if let Some(outcome) = try_insert_from_optimistic_route(
        bm,
        root_pin,
        route_cache,
        key,
        value,
        seq,
        condition,
        &mut blob_hops,
        &mut max_cross_blob_depth,
    )? {
        return Ok(outcome);
    }

    if let Some(outcome) = try_insert_from_root_router(
        bm,
        root_pin,
        route_cache,
        key,
        value,
        seq,
        condition,
        &mut blob_hops,
        &mut max_cross_blob_depth,
    )? {
        return Ok(outcome);
    }

    insert_multi_from_root(
        bm,
        root_pin,
        key,
        value,
        seq,
        condition,
        &mut blob_hops,
        &mut max_cross_blob_depth,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_insert_from_root_router(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<Option<InsertOutcome>> {
    let root_read = root_pin.read();
    let root_version = root_pin.content_version();
    let root_crossing = {
        let frame = BlobFrameRef::wrap(root_read.as_slice());
        let root_guid = frame.header().blob_guid;
        let root_slot = frame.header().root_slot;
        match lookup_at(frame, root_slot, key, 0)? {
            LookupResult::Crossing(crossing) => Some((root_guid, crossing)),
            LookupResult::Found(_) | LookupResult::NotFound => None,
        }
    };
    let Some((root_guid, crossing)) = root_crossing else {
        return Ok(None);
    };
    let child_pin = match bm.pin(crossing.child_guid) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) && bm.has_delete_fence(crossing.child_guid) => {
            drop(root_read);
            return Err(stale_blob_crossing(
                "stale blob crossing: insert root router",
            ));
        }
        Err(e) if is_blob_store_not_found(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    if child_is_snapshot_shared(bm, child_pin.as_ref()) {
        return Ok(None);
    }
    if let Some(cache) = route_cache {
        cache.learn(
            key,
            root_guid,
            0,
            root_version,
            crossing.child_guid,
            crossing.child_depth,
        );
        bm.mark_route_resident(crossing.child_guid);
    }
    child_pin.prefetch_header();
    let child_guard = child_pin.write();
    drop(root_read);

    *blob_hops = 1;
    let outcome = lock_coupled_insert_in_blob(
        bm,
        child_guard,
        child_pin.as_ref(),
        crossing.child_guid,
        false,
        key,
        value,
        seq,
        condition,
        crossing.child_depth,
        blob_hops,
        max_cross_blob_depth,
    );
    drop(child_pin);
    if outcome.is_ok() {
        bm.note_walker_blob_hops(*blob_hops, *max_cross_blob_depth);
    }
    outcome.map(Some)
}

#[allow(clippy::too_many_arguments)]
fn insert_multi_from_root(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<InsertOutcome> {
    let mut guard = root_pin.write();
    let root_guid = {
        let frame = guard.frame();
        frame.header().blob_guid
    };
    let outcome = lock_coupled_insert_in_blob(
        bm,
        guard,
        root_pin.as_ref(),
        root_guid,
        true,
        key,
        value,
        seq,
        condition,
        0,
        blob_hops,
        max_cross_blob_depth,
    );
    if outcome.is_ok() {
        bm.note_walker_blob_hops(*blob_hops, *max_cross_blob_depth);
    }
    outcome
}

/// Try the large-tree steady-state route-cache path without a full
/// root descent. The cached parent edge is validated under the
/// parent's shared latch before the child is pinned + exclusively
/// latched.
#[allow(clippy::too_many_arguments)]
fn try_insert_from_optimistic_route(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<Option<InsertOutcome>> {
    let Some(cache) = route_cache else {
        return Ok(None);
    };
    let Some(route) = cache.lookup(key) else {
        return Ok(None);
    };

    let parent_pin = match pin_route_parent(bm, root_pin, route) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) => {
            cache.invalidate(key, route);
            return Ok(None);
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
            return Ok(None);
        }
        cache.refresh_parent_version(key, route, parent_version);
    }
    let child_pin = match bm.pin(route.child_guid) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) && bm.has_delete_fence(route.child_guid) => {
            drop(parent_guard);
            cache.invalidate(key, route);
            return Err(stale_blob_crossing("stale blob crossing: insert route"));
        }
        Err(e) if is_blob_store_not_found(&e) => {
            drop(parent_guard);
            cache.invalidate(key, route);
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    // Copy-on-write: a snapshot-shared child must be forked, which needs
    // the parent's exclusive latch — bail to the full root descent (it
    // forks at the crossing). Already-private children take this fast
    // path normally, so a held snapshot only slows the first write per
    // forked region, not every write.
    if child_is_snapshot_shared(bm, child_pin.as_ref()) {
        drop(parent_guard);
        return Ok(None);
    }
    child_pin.prefetch_header();
    let child_guard = child_pin.write();
    drop(parent_guard);

    *blob_hops = 1;
    let outcome = lock_coupled_insert_in_blob(
        bm,
        child_guard,
        child_pin.as_ref(),
        route.child_guid,
        false,
        key,
        value,
        seq,
        condition,
        route.child_depth,
        blob_hops,
        max_cross_blob_depth,
    )?;
    drop(child_pin);
    bm.note_walker_blob_hops(*blob_hops, *max_cross_blob_depth);
    Ok(Some(outcome))
}

#[derive(Clone, Copy)]
pub(crate) struct InsertBatchItem<'a> {
    pub(crate) key: SearchKey<'a>,
    pub(crate) value: &'a [u8],
    pub(crate) seq: u64,
    condition: InsertCondition,
}

impl<'a> InsertBatchItem<'a> {
    pub(crate) const fn new(
        key: SearchKey<'a>,
        value: &'a [u8],
        seq: u64,
        condition: InsertCondition,
    ) -> Self {
        Self {
            key,
            value,
            seq,
            condition,
        }
    }
}

pub(crate) struct InsertBatchOutcome {
    pub(crate) root_dirty: bool,
    pub(crate) applied: usize,
}

/// Apply a consecutive atomic-batch insert run while reusing the
/// first pinned blob when possible. This deliberately stops at the
/// first deeper BlobNode crossing or blob-space miss and lets the
/// caller retry the remaining suffix through the normal
/// single-operation walker. That keeps split/cross-blob correctness
/// on the mature path while removing latch/pin churn from the common
/// "many same-prefix metadata updates in one atomic batch" case.
pub(crate) fn insert_multi_batch_conditional(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    items: &[InsertBatchItem<'_>],
) -> Result<InsertBatchOutcome> {
    if items.is_empty() {
        return Ok(InsertBatchOutcome {
            root_dirty: false,
            applied: 0,
        });
    }
    for item in items {
        if item.key.len() > u16::MAX as usize {
            return Err(Error::KeyTooLong {
                len: item.key.len(),
            });
        }
        if item.value.len() > u16::MAX as usize {
            return Err(Error::ValueTooLong {
                len: item.value.len(),
            });
        }
    }
    let mut restarts = 0u32;
    loop {
        match insert_multi_batch_conditional_once(bm, root_pin, route_cache, items) {
            Err(e) if is_stale_blob_crossing(&e) => {
                if let Some(cache) = route_cache {
                    cache.clear();
                }
                bm.note_optimistic_restart();
                restarts = restarts.saturating_add(1);
                if restarts >= 128 {
                    return Err(e);
                }
                if restarts % 16 == 0 {
                    std::thread::yield_now();
                }
            }
            out => return out,
        }
    }
}

pub(crate) fn insert_multi_batch_conditional_once(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    items: &[InsertBatchItem<'_>],
) -> Result<InsertBatchOutcome> {
    let batched = try_insert_batch_from_first_blob(bm, root_pin, route_cache, items)?;
    if batched.applied != 0 {
        return Ok(batched);
    }

    let first = items[0];
    let outcome = insert_multi_conditional(
        bm,
        root_pin,
        route_cache,
        first.key,
        first.value,
        first.seq,
        first.condition,
    )?;
    if !outcome.mutated {
        return Err(Error::Internal(
            "insert batch condition unexpectedly failed",
        ));
    }
    Ok(InsertBatchOutcome {
        root_dirty: outcome.root_dirty,
        applied: 1,
    })
}

fn try_insert_batch_from_first_blob(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    items: &[InsertBatchItem<'_>],
) -> Result<InsertBatchOutcome> {
    let first_key = items[0].key;

    if let Some(cache) = route_cache {
        if let Some(outcome) = try_insert_batch_from_route(bm, root_pin, cache, first_key, items)? {
            return Ok(outcome);
        }
    }

    {
        let root_read = root_pin.read();
        let root_version = root_pin.content_version();
        let root_crossing = {
            let frame = BlobFrameRef::wrap(root_read.as_slice());
            let root_guid = frame.header().blob_guid;
            let root_slot = frame.header().root_slot;
            match lookup_at(frame, root_slot, first_key, 0)? {
                LookupResult::Crossing(crossing) => Some((root_guid, crossing)),
                LookupResult::Found(_) | LookupResult::NotFound => None,
            }
        };
        if let Some((root_guid, crossing)) = root_crossing {
            if let Some(cache) = route_cache {
                cache.learn(
                    first_key,
                    root_guid,
                    0,
                    root_version,
                    crossing.child_guid,
                    crossing.child_depth,
                );
                bm.mark_route_resident(crossing.child_guid);
            }
            let run_len = same_child_prefix_run_len(items, crossing.child_depth);
            let child_pin = match bm.pin(crossing.child_guid) {
                Ok(pin) => pin,
                Err(e)
                    if is_blob_store_not_found(&e) && bm.has_delete_fence(crossing.child_guid) =>
                {
                    drop(root_read);
                    return Err(stale_blob_crossing(
                        "stale blob crossing: insert batch root router",
                    ));
                }
                Err(e) if is_blob_store_not_found(&e) => {
                    drop(root_read);
                    return insert_batch_from_root(bm, root_pin, items);
                }
                Err(e) => return Err(e),
            };
            child_pin.prefetch_header();
            let child_guard = child_pin.write();
            drop(root_read);

            let outcome = insert_batch_in_pinned_blob(
                bm,
                child_guard,
                child_pin.as_ref(),
                crossing.child_guid,
                false,
                &items[..run_len],
                crossing.child_depth,
                2,
            );
            drop(child_pin);
            return outcome;
        }
        drop(root_read);
    }

    insert_batch_from_root(bm, root_pin, items)
}

fn insert_batch_from_root(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    items: &[InsertBatchItem<'_>],
) -> Result<InsertBatchOutcome> {
    let mut guard = root_pin.write();
    let root_guid = {
        let frame = guard.frame();
        frame.header().blob_guid
    };
    insert_batch_in_pinned_blob(bm, guard, root_pin.as_ref(), root_guid, true, items, 0, 1)
}

fn try_insert_batch_from_route(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    cache: &RouteCache,
    first_key: SearchKey<'_>,
    items: &[InsertBatchItem<'_>],
) -> Result<Option<InsertBatchOutcome>> {
    let Some(route) = cache.lookup(first_key) else {
        return Ok(None);
    };
    let run_len = same_child_prefix_run_len(items, route.child_depth);
    let parent_pin = match pin_route_parent(bm, root_pin, route) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) => {
            cache.invalidate(first_key, route);
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    let parent_guard = parent_pin.read();
    let parent_version = parent_pin.content_version();
    if parent_version != route.parent_version {
        let frame = BlobFrameRef::wrap(parent_guard.as_slice());
        if !validate_route_edge(frame, first_key, route)? {
            drop(parent_guard);
            cache.invalidate(first_key, route);
            return Ok(None);
        }
        cache.refresh_parent_version(first_key, route, parent_version);
    }
    let child_pin = match bm.pin(route.child_guid) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) && bm.has_delete_fence(route.child_guid) => {
            drop(parent_guard);
            cache.invalidate(first_key, route);
            return Err(stale_blob_crossing(
                "stale blob crossing: insert batch route",
            ));
        }
        Err(e) if is_blob_store_not_found(&e) => {
            drop(parent_guard);
            cache.invalidate(first_key, route);
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    child_pin.prefetch_header();
    let child_guard = child_pin.write();
    drop(parent_guard);
    let outcome = insert_batch_in_pinned_blob(
        bm,
        child_guard,
        child_pin.as_ref(),
        route.child_guid,
        false,
        &items[..run_len],
        route.child_depth,
        2,
    );
    drop(child_pin);
    outcome.map(Some)
}

fn same_child_prefix_run_len(items: &[InsertBatchItem<'_>], child_depth: usize) -> usize {
    let Some(prefix) = items[0].key.user_prefix(child_depth) else {
        return 1;
    };
    let mut len = 1usize;
    while len < items.len() {
        match items[len].key.user_prefix(child_depth) {
            Some(candidate) if candidate == prefix => len += 1,
            _ => break,
        }
    }
    len
}

#[allow(clippy::too_many_arguments)]
fn insert_batch_in_pinned_blob(
    bm: &BufferManager,
    mut guard: BlobWriteGuard<'_>,
    current_entry: &CachedBlob,
    current_guid: crate::layout::BlobGuid,
    is_top_blob: bool,
    items: &[InsertBatchItem<'_>],
    depth: usize,
    blob_hops_per_item: u64,
) -> Result<InsertBatchOutcome> {
    let mut applied = 0usize;
    let mut dirty = false;
    let mut needs_compaction = false;

    for item in items {
        let r = {
            let mut frame = guard.frame();
            let root_off = decode_child_off(frame.header().root_slot);
            insert_at_step(
                &mut frame,
                root_off,
                item.key,
                item.value,
                depth,
                item.seq,
                item.condition,
                true,
            )
        };
        match r {
            Ok(InsertStep::Done(out)) => {
                if !out.mutated {
                    return Err(Error::Internal(
                        "insert batch condition unexpectedly failed",
                    ));
                }
                {
                    let mut frame = guard.frame();
                    frame.header_mut().root_slot = encode_child_off(out.off_after);
                    needs_compaction |= blob_needs_compaction(frame.as_ref());
                }
                applied += 1;
                dirty = true;
                bm.note_walker_blob_hops(blob_hops_per_item, depth);
            }
            Ok(InsertStep::Crossing(_))
            | Err(Error::Alloc(
                crate::store::AllocError::OutOfSpace { .. } | crate::store::AllocError::OutOfSlots,
            )) => break,
            Err(e) => return Err(e.with_blob_guid(current_guid)),
        }
    }

    drop(guard);

    if needs_compaction {
        bm.note_compaction_candidate(current_guid);
    }
    if dirty && !is_top_blob {
        bm.mark_dirty_cached(current_guid, items[0].seq, current_entry);
    }

    Ok(InsertBatchOutcome {
        root_dirty: is_top_blob && dirty,
        applied,
    })
}

#[derive(Debug, Clone, Copy)]
struct InsertBlobCrossing {
    child_guid: crate::layout::BlobGuid,
    child_depth: usize,
    /// Byte offset of the `BlobNode` in the parent frame that points
    /// at this child — the edge a copy-on-write fork repoints at the
    /// child's private fork.
    parent_off: u32,
}

enum InsertStep {
    Done(InsertReturn),
    Crossing(InsertBlobCrossing),
}

#[allow(clippy::too_many_arguments)] // hot-path helper mirrors insert_at's call shape
fn lock_coupled_insert_in_blob(
    bm: &BufferManager,
    mut guard: BlobWriteGuard<'_>,
    current_entry: &CachedBlob,
    current_guid: crate::layout::BlobGuid,
    is_top_blob: bool,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
    depth: usize,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<InsertOutcome> {
    *blob_hops = blob_hops.saturating_add(1);
    *max_cross_blob_depth = (*max_cross_blob_depth).max(depth);
    let mut current_dirty = false;

    for _attempt in 0..MAX_SPILLOVER_ATTEMPTS {
        let r = {
            let mut frame = guard.frame();
            let root_off = decode_child_off(frame.header().root_slot);
            insert_at_step(
                &mut frame, root_off, key, value, depth, seq, condition, true,
            )
        };
        match r {
            Ok(InsertStep::Done(out)) => {
                let needs_compaction = {
                    let mut frame = guard.frame();
                    if out.mutated {
                        frame.header_mut().root_slot = encode_child_off(out.off_after);
                        blob_needs_compaction(frame.as_ref())
                    } else {
                        false
                    }
                };
                drop(guard);
                if needs_compaction {
                    bm.note_compaction_candidate(current_guid);
                }
                if out.mutated && !is_top_blob {
                    bm.mark_dirty_cached(current_guid, seq, current_entry);
                }

                return Ok(InsertOutcome {
                    root_dirty: is_top_blob && out.mutated,
                    mutated: out.mutated,
                });
            }
            Ok(InsertStep::Crossing(crossing)) => {
                return cross_and_insert(
                    bm,
                    guard,
                    crossing,
                    is_top_blob,
                    current_guid,
                    current_entry,
                    current_dirty,
                    key,
                    value,
                    seq,
                    condition,
                    blob_hops,
                    max_cross_blob_depth,
                );
            }
            Err(Error::Alloc(crate::store::AllocError::OutOfSpace { .. })) => {
                {
                    let mut frame = guard.frame();
                    spillover_blob(bm, &mut frame, seq)
                        .map_err(|e| e.with_blob_guid(current_guid))?;
                }
                bm.note_merge_candidate(current_guid);
                bm.note_spillover();
                compact_blob(&mut guard).map_err(|e| e.with_blob_guid(current_guid))?;
                current_dirty = true;
            }
            Err(Error::Alloc(crate::store::AllocError::OutOfSlots)) => {
                // The slot table filled. Structural ops abandon old nodes
                // (offset-addressing R1 = abandon-on-free); those dead
                // slots are reclaimed only by compaction, and under
                // concurrent write pressure the *background* compactor
                // can be starved of this blob's exclusive latch — so
                // reclaim inline here, where we already hold it. If
                // compaction frees slots the retry proceeds; only if the
                // blob is genuinely full of *live* nodes do we spill a
                // subtree out (now with post-compaction slot headroom for
                // the spillover BlobNode).
                let before = guard.frame().header().num_slots;
                compact_blob(&mut guard).map_err(|e| e.with_blob_guid(current_guid))?;
                current_dirty = true;
                if guard.frame().header().num_slots >= before {
                    {
                        let mut frame = guard.frame();
                        spillover_blob(bm, &mut frame, seq)
                            .map_err(|e| e.with_blob_guid(current_guid))?;
                    }
                    bm.note_merge_candidate(current_guid);
                    bm.note_spillover();
                    compact_blob(&mut guard).map_err(|e| e.with_blob_guid(current_guid))?;
                }
            }
            Err(e) => return Err(e.with_blob_guid(current_guid)),
        }
    }

    Err(Error::NotYetImplemented(
        "lock_coupled_insert_in_blob: spillover retry loop exhausted",
    ))
}

/// Handle an insert that crosses a `BlobNode` into a child frame.
///
/// If a live snapshot may reference the child, fork it first (repointing
/// this frame's edge at the fork — see [`fork_child_if_shared`]), then
/// lock-couple the insert into the child (or its fork) and propagate
/// this frame's dirtiness upward. `parent_dirty` carries any dirtiness
/// this frame already accrued (e.g. from a spillover in the caller's
/// retry loop).
#[allow(clippy::too_many_arguments)]
fn cross_and_insert(
    bm: &BufferManager,
    mut parent_guard: BlobWriteGuard<'_>,
    crossing: InsertBlobCrossing,
    is_top_blob: bool,
    current_guid: crate::layout::BlobGuid,
    current_entry: &CachedBlob,
    mut parent_dirty: bool,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<InsertOutcome> {
    let child_pin = match bm.pin(crossing.child_guid) {
        Ok(pin) => pin,
        Err(e) if is_blob_store_not_found(&e) && bm.has_delete_fence(crossing.child_guid) => {
            drop(parent_guard);
            if parent_dirty {
                bm.mark_dirty_cached(current_guid, seq, current_entry);
            }
            return Err(stale_blob_crossing(
                "stale blob crossing: insert deep child",
            ));
        }
        Err(e) => return Err(e.with_blob_guid(crossing.child_guid)),
    };
    child_pin.prefetch_header();
    let child_guard = child_pin.write();

    let mut outcome = if let Some((fork_guid, fork_pin)) = fork_child_if_shared(
        bm,
        &mut parent_guard,
        crossing.child_guid,
        child_guard.as_slice(),
        crossing.parent_off,
        seq,
    )? {
        parent_dirty = true;
        drop(child_guard);
        drop(child_pin);
        let fork_guard = fork_pin.write();
        drop(parent_guard);
        let out = lock_coupled_insert_in_blob(
            bm,
            fork_guard,
            fork_pin.as_ref(),
            fork_guid,
            false,
            key,
            value,
            seq,
            condition,
            crossing.child_depth,
            blob_hops,
            max_cross_blob_depth,
        );
        drop(fork_pin);
        out
    } else {
        drop(parent_guard);
        let out = lock_coupled_insert_in_blob(
            bm,
            child_guard,
            child_pin.as_ref(),
            crossing.child_guid,
            false,
            key,
            value,
            seq,
            condition,
            crossing.child_depth,
            blob_hops,
            max_cross_blob_depth,
        );
        drop(child_pin);
        out
    };

    if outcome.is_ok() && parent_dirty && !is_top_blob {
        bm.mark_dirty_cached(current_guid, seq, current_entry);
    }
    if let Ok(outcome) = &mut outcome {
        outcome.root_dirty |= is_top_blob && parent_dirty;
    }
    outcome
}

// ---------- recursive dispatch ----------

#[cfg(test)]
#[allow(clippy::too_many_arguments)] // test-only helper mirrors insert_at_step
pub(super) fn insert_at(
    frame: &mut BlobFrame<'_>,
    off: u32,
    key: SearchKey<'_>,
    value: &[u8],
    depth: usize,
    seq: u64,
) -> Result<InsertReturn> {
    match insert_at_step(
        frame,
        off,
        key,
        value,
        depth,
        seq,
        InsertCondition::Always,
        false,
    )? {
        InsertStep::Done(r) => Ok(r),
        InsertStep::Crossing(_) => Err(Error::NotYetImplemented(
            "walker::insert_at: BlobNode crossing requires BufferManager — use insert_multi",
        )),
    }
}

#[allow(clippy::too_many_arguments)] // condition/crossing flags mirror every node arm
fn insert_at_step(
    frame: &mut BlobFrame<'_>,
    off: u32,
    key: SearchKey<'_>,
    value: &[u8],
    depth: usize,
    seq: u64,
    condition: InsertCondition,
    allow_crossing: bool,
) -> Result<InsertStep> {
    let ntype = ntype_of(frame.as_ref(), off)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::insert_at: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot => {
            insert_into_empty_root(frame, off, key, value, seq, condition).map(InsertStep::Done)
        }
        NodeType::Leaf => {
            insert_into_leaf(frame, off, key, value, depth, seq, condition).map(InsertStep::Done)
        }
        NodeType::Prefix => insert_into_prefix_step(
            frame,
            off,
            key,
            value,
            depth,
            seq,
            condition,
            allow_crossing,
        ),
        NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
            insert_into_inner_step(
                frame,
                off,
                ntype,
                key,
                value,
                depth,
                seq,
                condition,
                allow_crossing,
            )
        }
        NodeType::Blob => blob_node_insert_step(
            frame,
            off,
            key,
            value,
            depth,
            seq,
            condition,
            allow_crossing,
        ),
    }
}

#[allow(clippy::too_many_arguments)] // condition threads through same walker shape
fn blob_node_insert_step(
    frame: &mut BlobFrame<'_>,
    off: u32,
    key: SearchKey<'_>,
    value: &[u8],
    depth: usize,
    seq: u64,
    condition: InsertCondition,
    allow_crossing: bool,
) -> Result<InsertStep> {
    let body = frame.body_at_offset(off).ok_or(Error::node_corrupt(
        "blob_node_insert_step: BlobNode body resolution failed",
    ))?;
    let bn = *cast::<BlobNode>(body);
    let plen = bn.prefix_len as usize;
    if plen > BLOB_MAX_INLINE {
        return Err(Error::node_corrupt(
            "blob_node_insert_step: BlobNode prefix_len exceeds inline buffer",
        ));
    }
    let prefix = &bn.bytes[..plen];
    let common = key.common_prefix_with_slice(depth, prefix);

    if common == plen {
        if !allow_crossing {
            return Err(Error::NotYetImplemented(
                "walker::insert_at: BlobNode crossing requires BufferManager — use insert_multi",
            ));
        }
        return Ok(InsertStep::Crossing(InsertBlobCrossing {
            child_guid: bn.child_blob_guid,
            child_depth: depth + plen,
            parent_off: off,
        }));
    }

    let Some(new_div_byte) = key.byte_at(depth + common) else {
        return Err(Error::NotYetImplemented(
            "blob_node_insert_step: key terminates inside BlobNode prefix",
        ));
    };
    let existing_div_byte = prefix[common];
    debug_assert_ne!(existing_div_byte, new_div_byte);

    if matches!(condition, InsertCondition::IfVersion(_)) {
        return Ok(InsertStep::Done(InsertReturn {
            off_after: off,
            mutated: false,
        }));
    }

    // Keep the old BlobNode in place so its offset stays stable for
    // the Node4 edge below. The branch byte is consumed by the new
    // Node4, so the BlobNode only keeps the remaining inline tail
    // before crossing to the unchanged child blob.
    let existing_tail = &prefix[common + 1..];
    let new_leaf = write_leaf(frame, key, value, seq)?;
    let n4 = write_node4_with(frame, &[(existing_div_byte, off), (new_div_byte, new_leaf)])?;
    let final_off = if common == 0 {
        n4
    } else {
        write_prefix_chain(frame, &prefix[..common], n4)?
    };

    let adjusted = BlobNode::new(existing_tail, bn.child_blob_guid);
    write_struct_at(frame, off, &adjusted)?;

    Ok(InsertStep::Done(InsertReturn {
        off_after: final_off,
        mutated: true,
    }))
}

fn insert_into_empty_root(
    frame: &mut BlobFrame<'_>,
    empty_off: u32,
    key: SearchKey<'_>,
    value: &[u8],
    seq: u64,
    condition: InsertCondition,
) -> Result<InsertReturn> {
    if matches!(condition, InsertCondition::IfVersion(_)) {
        return Ok(InsertReturn {
            off_after: empty_off,
            mutated: false,
        });
    }
    // Abandon-on-free: the EmptyRoot sentinel becomes unreachable once
    // the root points at the new leaf; reclaimed at next compaction.
    let new_off = write_leaf(frame, key, value, seq)?;
    frame.note_abandoned(empty_off);
    Ok(InsertReturn {
        off_after: new_off,
        mutated: true,
    })
}

struct LeafSplitPlan {
    common_prefix: Vec<u8>,
    byte_existing: u8,
    byte_new: u8,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn insert_into_leaf(
    frame: &mut BlobFrame<'_>,
    leaf_off: u32,
    new_key: SearchKey<'_>,
    new_value: &[u8],
    depth: usize,
    seq: u64,
    condition: InsertCondition,
) -> Result<InsertReturn> {
    enum LeafInsertPlan {
        SameKey(Leaf),
        Split(LeafSplitPlan),
    }

    // Always read the existing key (needed for both same-key
    // update and divergence-split paths), but keep it borrowed
    // from the blob. Only the split path materialises the shared
    // prefix bytes because subsequent writes mutate the frame.
    let plan = {
        let (existing_key, existing_leaf) = read_leaf_key_ref(frame.as_ref(), leaf_off)?;
        if new_key.eq_slice(existing_key) {
            LeafInsertPlan::SameKey(existing_leaf)
        } else {
            let suffix_a = &existing_key[depth..];
            let common_len = new_key.common_prefix_with_slice(depth, suffix_a);

            if common_len == suffix_a.len() || common_len == new_key.remaining_len(depth) {
                return Err(Error::NotYetImplemented(
                    "walker::insert_into_leaf: one key is a strict prefix of the other",
                ));
            }

            LeafInsertPlan::Split(LeafSplitPlan {
                common_prefix: suffix_a[..common_len].to_vec(),
                byte_existing: suffix_a[common_len],
                byte_new: new_key
                    .byte_at(depth + common_len)
                    .expect("new key has divergence byte"),
            })
        }
    };

    let split = match plan {
        LeafInsertPlan::SameKey(existing_leaf) => {
            if existing_leaf.tombstone == 0 {
                match condition {
                    InsertCondition::Always => {}
                    InsertCondition::IfVersion(expected) if existing_leaf.seq == expected => {}
                    InsertCondition::IfAbsent | InsertCondition::IfVersion(_) => {
                        return Ok(InsertReturn {
                            off_after: leaf_off,
                            mutated: false,
                        });
                    }
                }
            } else if matches!(condition, InsertCondition::IfVersion(_)) {
                return Ok(InsertReturn {
                    off_after: leaf_off,
                    mutated: false,
                });
            }
            // Same-key update path (covers two semantic cases via the
            // same alloc machinery):
            //
            // 1. **Resurrect**: the existing leaf is tombstoned — the
            //    user just put the key back after deleting it. From
            //    the user's view this is a fresh insert (`previous`
            //    is `None`) and the blob's `tombstone_leaf_cnt` drops
            //    by one because the slot leaves the tombstone state.
            // 2. **Update**: the existing leaf is live — overwrite in
            //    place when the new total fits the slot's current
            //    allocated body; otherwise alloc-fresh + free-old.
            //
            // `Leaf::live` always pins `tombstone = 0` so both write
            // paths naturally clear the bit in the new leaf body.
            let was_tombstoned = existing_leaf.tombstone != 0;
            let key_len = existing_leaf.key_len as usize;
            let new_value_len = new_value.len();
            let new_total = leaf_body_size(key_len as u32, new_value_len as u32) as usize;
            // The leaf is one contiguous self-describing node; resolve
            // its current allocated body size via `body_at_offset`.
            let cur_body_len = frame
                .body_at_offset(leaf_off)
                .ok_or(Error::node_corrupt("insert_into_leaf: leaf body"))?
                .len();

            if new_total <= cur_body_len {
                // In-place overwrite: the key stays put at
                // `body[16..16+key_len]`; rewrite the header (fresh
                // value_len, seq, fingerprint; tombstone cleared) and
                // the value at `body[16+key_len..]`, zeroing any
                // shrink tail.
                let new_leaf = Leaf::live(
                    key_len as u16,
                    new_value_len as u16,
                    seq,
                    new_key.fingerprint(),
                );
                {
                    let body = frame
                        .body_at_offset_mut(leaf_off)
                        .ok_or(Error::node_corrupt("insert_into_leaf: leaf body (mut)"))?;
                    let hdr = unsafe {
                        std::slice::from_raw_parts(
                            std::ptr::from_ref::<Leaf>(&new_leaf).cast::<u8>(),
                            std::mem::size_of::<Leaf>(),
                        )
                    };
                    body[..16].copy_from_slice(hdr);
                    let value_start = 16 + key_len;
                    body[value_start..value_start + new_value_len].copy_from_slice(new_value);
                    for b in &mut body[value_start + new_value_len..] {
                        *b = 0;
                    }
                }
                if was_tombstoned {
                    let h = frame.header_mut();
                    h.tombstone_leaf_cnt = h.tombstone_leaf_cnt.saturating_sub(1);
                }
                return Ok(InsertReturn {
                    off_after: leaf_off,
                    mutated: true,
                });
            }

            // Value grew past the current allocated body — alloc a
            // fresh leaf and abandon the old one (abandon-on-free):
            // the old body bytes become unreachable and are reclaimed
            // at the next compaction.
            let new_off = write_leaf(frame, new_key, new_value, seq)?;
            frame.note_abandoned(leaf_off);
            if was_tombstoned {
                let h = frame.header_mut();
                h.tombstone_leaf_cnt = h.tombstone_leaf_cnt.saturating_sub(1);
            }
            return Ok(InsertReturn {
                off_after: new_off,
                mutated: true,
            });
        }
        LeafInsertPlan::Split(split) => split,
    };

    if matches!(condition, InsertCondition::IfVersion(_)) {
        return Ok(InsertReturn {
            off_after: leaf_off,
            mutated: false,
        });
    }

    // Two different keys: split into [Prefix?] -> Node4 -> {old leaf, new leaf}.
    let final_off = write_leaf_split(frame, leaf_off, new_key, new_value, seq, &split)?;
    Ok(InsertReturn {
        off_after: final_off,
        mutated: true,
    })
}

fn write_leaf_split(
    frame: &mut BlobFrame<'_>,
    leaf_off: u32,
    new_key: SearchKey<'_>,
    new_value: &[u8],
    seq: u64,
    split: &LeafSplitPlan,
) -> Result<u32> {
    let new_leaf = write_leaf(frame, new_key, new_value, seq)?;
    let n4 = write_node4_with(
        frame,
        &[(split.byte_existing, leaf_off), (split.byte_new, new_leaf)],
    )?;

    let final_off = if split.common_prefix.is_empty() {
        n4
    } else {
        write_prefix_chain(frame, &split.common_prefix, n4)?
    };

    Ok(final_off)
}

#[allow(clippy::too_many_arguments)] // mirrors insert_at_step's call shape
fn insert_into_prefix_step(
    frame: &mut BlobFrame<'_>,
    pfx_off: u32,
    key: SearchKey<'_>,
    value: &[u8],
    depth: usize,
    seq: u64,
    condition: InsertCondition,
    allow_crossing: bool,
) -> Result<InsertStep> {
    // `Prefix` is `Copy` and `read_prefix` returns it by value, so
    // `p` is owned on the stack. The inline prefix bytes live in
    // `p.bytes` — no need to allocate a `Vec` to keep them alive
    // across the `frame.*` mutations below.
    let p = read_prefix(frame.as_ref(), pfx_off)?;
    let plen = p.prefix_len as usize;
    let prefix_bytes = &p.bytes[..plen];
    let child_off = child_offset(p.child as u16);

    let common = key.common_prefix_with_slice(depth, prefix_bytes);

    if common == plen {
        let r = insert_at_step(
            frame,
            child_off,
            key,
            value,
            depth + plen,
            seq,
            condition,
            allow_crossing,
        )?;
        let InsertStep::Done(r) = r else {
            return Ok(r);
        };
        if r.off_after != child_off {
            set_prefix_child(frame, pfx_off, r.off_after)?;
        }
        return Ok(InsertStep::Done(InsertReturn {
            off_after: pfx_off,
            mutated: r.mutated,
        }));
    }

    if depth + common >= key.len() {
        return Err(Error::NotYetImplemented(
            "walker::insert_into_prefix: key terminates inside a prefix",
        ));
    }

    let existing_div_byte = prefix_bytes[common];
    let new_div_byte = key
        .byte_at(depth + common)
        .expect("new key has prefix divergence byte");

    if matches!(condition, InsertCondition::IfVersion(_)) {
        return Ok(InsertStep::Done(InsertReturn {
            off_after: pfx_off,
            mutated: false,
        }));
    }

    let tail_bytes = &prefix_bytes[common + 1..];
    let existing_branch_off = if tail_bytes.is_empty() {
        child_off
    } else {
        write_prefix_chain(frame, tail_bytes, child_off)?
    };
    let new_leaf = write_leaf(frame, key, value, seq)?;
    let n4 = write_node4_with(
        frame,
        &[
            (existing_div_byte, existing_branch_off),
            (new_div_byte, new_leaf),
        ],
    )?;

    let final_off = if common == 0 {
        n4
    } else {
        write_prefix_chain(frame, &prefix_bytes[..common], n4)?
    };

    // Abandon-on-free: the old Prefix node is unreachable now that the
    // parent points at `final_off`; reclaimed at next compaction.
    frame.note_abandoned(pfx_off);

    Ok(InsertStep::Done(InsertReturn {
        off_after: final_off,
        mutated: true,
    }))
}

#[allow(clippy::too_many_arguments)] // mirrors insert_at's call shape
fn insert_into_inner_step(
    frame: &mut BlobFrame<'_>,
    inner_off: u32,
    ntype: NodeType,
    key: SearchKey<'_>,
    value: &[u8],
    depth: usize,
    seq: u64,
    condition: InsertCondition,
    allow_crossing: bool,
) -> Result<InsertStep> {
    let Some(byte) = key.byte_at(depth) else {
        return Err(Error::NotYetImplemented(
            "walker::insert_into_inner: key terminates at an inner node",
        ));
    };

    if let Some(child_off) = inner_find_child(frame, inner_off, ntype, byte)? {
        let r = insert_at_step(
            frame,
            child_off,
            key,
            value,
            depth + 1,
            seq,
            condition,
            allow_crossing,
        )?;
        let InsertStep::Done(r) = r else {
            return Ok(r);
        };
        if r.off_after != child_off {
            inner_update_child(frame, inner_off, ntype, byte, r.off_after)?;
        }
        return Ok(InsertStep::Done(InsertReturn {
            off_after: inner_off,
            mutated: r.mutated,
        }));
    }

    if matches!(condition, InsertCondition::IfVersion(_)) {
        return Ok(InsertStep::Done(InsertReturn {
            off_after: inner_off,
            mutated: false,
        }));
    }
    let new_leaf = write_leaf(frame, key, value, seq)?;
    // `inner_add_child` may grow (abandoning the old node) and return a
    // new offset — the parent rewires to it.
    let possibly_grown = inner_add_child(frame, inner_off, ntype, byte, new_leaf)?;
    Ok(InsertStep::Done(InsertReturn {
        off_after: possibly_grown,
        mutated: true,
    }))
}
