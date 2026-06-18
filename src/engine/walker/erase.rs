//! Erase path — `erase` / `erase_multi` + recursive `erase_at`
//! dispatch + per-NodeType arms + collapse-on-lone-child rewiring.

use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::layout::{BlobNode, Leaf, NodeType, BLOB_MAX_INLINE};
use std::mem::offset_of;
use std::sync::Arc;

use super::cast;
use super::cow::{child_is_snapshot_shared, fork_child_if_shared};
use super::lookup::lookup_at;
use super::readers::{
    child_offset, ntype_of, read_leaf_key_ref, read_node16, read_node256, read_node4, read_node48,
    read_prefix,
};
use super::route::{pin_route_parent, validate_route_edge};
use super::types::{is_stale_blob_crossing, stale_blob_crossing};
use super::types::{EraseCondition, EraseOutcome, EraseReturn, EraseSignal, LookupResult};
use super::writers::{
    finish_inner_with_sorted, inner_find_child, inner_update_child, set_prefix_child,
    shrink_node16_to_node4, shrink_node256_to_node48, shrink_node48_to_node16, write_prefix_chain,
    write_struct_at, SHRINK_NODE16_TO_NODE4_AT, SHRINK_NODE256_TO_NODE48_AT,
    SHRINK_NODE48_TO_NODE16_AT,
};
use super::SearchKey;
use crate::engine::{simd, RouteCache};
use crate::store::BlobWriteGuard;
use crate::store::{decode_child_off, encode_child_off};
use crate::store::{BlobFrame, BlobFrameRef, BufferManager, CachedBlob};

// ---------- public entry points ----------

/// Single-blob erase. Surfaces [`Error::NotYetImplemented`] if the
/// descent reaches a [`NodeType::Blob`] crossing — callers wanting
/// cross-blob erase should use [`erase_multi`].
///
/// Updates `header.root_slot` in place.
#[cfg(test)]
pub(super) fn erase(frame: &mut BlobFrame<'_>, root_enc: u16, key: &[u8]) -> Result<EraseOutcome> {
    let root_off = decode_child_off(root_enc);
    let r = erase_at(frame, root_off, key, 0)?;
    let root_dirty = r.mutated || !matches!(r.signal, EraseSignal::Unchanged);
    let new_root = resolve_new_root_after_erase(frame, root_off, &r.signal)?;
    frame.header_mut().root_slot = encode_child_off(new_root);
    Ok(EraseOutcome {
        root_dirty,
        mutated: r.mutated,
    })
}

/// Multi-blob erase. Pins the root via the [`BufferManager`] and
/// walks across [`NodeType::Blob`] crossings. The lock-coupled
/// child path keeps parent BlobNodes stable and records child root
/// changes in the child blob's own header.
///
pub fn erase_multi(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    seq: u64,
) -> Result<EraseOutcome> {
    erase_multi_conditional(bm, root_pin, route_cache, key, seq, EraseCondition::Always)
}

/// Conditional variant of [`erase_multi`]. Used by
/// `Tree::delete_if_version` so the version check and tombstone
/// write happen under the same exclusive blob latch.
pub fn erase_multi_conditional(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    seq: u64,
    condition: EraseCondition,
) -> Result<EraseOutcome> {
    let mut restarts = 0u32;
    loop {
        match erase_multi_conditional_once(bm, root_pin, route_cache, key, seq, condition) {
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

fn erase_multi_conditional_once(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    seq: u64,
    condition: EraseCondition,
) -> Result<EraseOutcome> {
    // The caller (typically `Tree`) keeps `root_pin` alive across
    // every op so we skip `BufferManager`'s pin-Mutex on the hot
    // root hop. The guard-aware walker performs a single descent:
    // it tombstones in the current blob directly, or if the path
    // reaches a BlobNode it lock-couples into the child and
    // releases the parent before descendant mutation.
    //
    // `seq` is the WAL seq the caller pre-allocated for this op;
    // every child blob the walker mutates gets a corresponding
    // `bm.mark_dirty(child_guid, seq)` so the checkpoint round
    // flushes WAL **before** the child bytes reach the store.
    let mut blob_hops = 0u64;
    let mut max_cross_blob_depth = 0usize;

    if let Some(outcome) = try_erase_from_route(
        bm,
        root_pin,
        route_cache,
        key,
        seq,
        condition,
        &mut blob_hops,
        &mut max_cross_blob_depth,
    )? {
        return Ok(outcome);
    }

    {
        let root_read = root_pin.read();
        let root_version = root_pin.content_version();
        let root_lookup = {
            let frame = BlobFrameRef::wrap(root_read.as_slice());
            let root_guid = frame.header().blob_guid;
            let root_slot = frame.header().root_slot;
            (root_guid, lookup_at(frame, root_slot, key, 0)?)
        };
        match root_lookup {
            (root_guid, LookupResult::Crossing(crossing)) => {
                let child_pin = match bm.pin(crossing.child_guid) {
                    Ok(pin) => pin,
                    Err(e) if is_blob_store_not_found(&e) => {
                        drop(root_read);
                        return erase_from_root(
                            bm,
                            root_pin,
                            key,
                            seq,
                            condition,
                            &mut blob_hops,
                            &mut max_cross_blob_depth,
                        );
                    }
                    Err(e) => return Err(e),
                };
                // Copy-on-write: a shared root child must be forked by
                // repointing the root's BlobNode, which needs the root's
                // exclusive latch — bail to the root-local path below.
                let child_shared = child_is_snapshot_shared(bm, child_pin.as_ref());
                if !child_shared {
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

                    blob_hops = 1;
                    let outcome = lock_coupled_erase_in_blob(
                        bm,
                        child_guard,
                        child_pin.as_ref(),
                        crossing.child_guid,
                        false,
                        key,
                        seq,
                        condition,
                        crossing.child_depth,
                        &mut blob_hops,
                        &mut max_cross_blob_depth,
                    );
                    drop(child_pin);
                    if outcome.is_ok() {
                        bm.note_walker_blob_hops(blob_hops, max_cross_blob_depth);
                    }
                    return outcome;
                }
                drop(child_pin);
            }
            (_, LookupResult::NotFound) => {
                bm.note_walker_blob_hops(1, 0);
                return Ok(EraseOutcome {
                    root_dirty: false,
                    mutated: false,
                });
            }
            (_, LookupResult::Found(_)) => {}
        }
    }

    erase_from_root(
        bm,
        root_pin,
        key,
        seq,
        condition,
        &mut blob_hops,
        &mut max_cross_blob_depth,
    )
}

fn erase_from_root(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    key: SearchKey<'_>,
    seq: u64,
    condition: EraseCondition,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<EraseOutcome> {
    let mut guard = root_pin.write();
    let root_guid = {
        let frame = guard.frame();
        frame.header().blob_guid
    };
    let outcome = lock_coupled_erase_in_blob(
        bm,
        guard,
        root_pin.as_ref(),
        root_guid,
        true,
        key,
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

#[allow(clippy::too_many_arguments)]
fn try_erase_from_route(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route_cache: Option<&RouteCache>,
    key: SearchKey<'_>,
    seq: u64,
    condition: EraseCondition,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<Option<EraseOutcome>> {
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
        Err(e) if is_blob_store_not_found(&e) => {
            drop(parent_guard);
            cache.invalidate(key, route);
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    // Copy-on-write: a snapshot-shared child must be forked under the
    // parent's exclusive latch — bail to the full root descent (it forks
    // at the crossing). Private children take this fast path normally.
    if child_is_snapshot_shared(bm, child_pin.as_ref()) {
        drop(parent_guard);
        return Ok(None);
    }
    child_pin.prefetch_header();
    let child_guard = child_pin.write();
    drop(parent_guard);

    *blob_hops = 1;
    let outcome = lock_coupled_erase_in_blob(
        bm,
        child_guard,
        child_pin.as_ref(),
        route.child_guid,
        false,
        key,
        seq,
        condition,
        route.child_depth,
        blob_hops,
        max_cross_blob_depth,
    );
    drop(child_pin);
    if outcome.is_ok() {
        bm.note_walker_blob_hops(*blob_hops, *max_cross_blob_depth);
    }
    outcome.map(Some)
}

#[derive(Debug, Clone, Copy)]
struct EraseBlobCrossing {
    child_guid: crate::layout::BlobGuid,
    child_depth: usize,
    /// Byte offset of the `BlobNode` in the parent frame that points
    /// at this child — the edge a copy-on-write fork repoints at the
    /// child's private fork.
    parent_off: u32,
}

enum EraseStep {
    Done(EraseReturn),
    Crossing(EraseBlobCrossing),
}

#[allow(clippy::too_many_arguments)] // mirrors erase_at's call shape
fn lock_coupled_erase_in_blob(
    bm: &BufferManager,
    mut guard: BlobWriteGuard<'_>,
    current_entry: &CachedBlob,
    current_guid: crate::layout::BlobGuid,
    is_top_blob: bool,
    key: SearchKey<'_>,
    seq: u64,
    condition: EraseCondition,
    depth: usize,
    blob_hops: &mut u64,
    max_cross_blob_depth: &mut usize,
) -> Result<EraseOutcome> {
    *blob_hops = blob_hops.saturating_add(1);
    *max_cross_blob_depth = (*max_cross_blob_depth).max(depth);
    let step = {
        let mut frame = guard.frame();
        let root_off = decode_child_off(frame.header().root_slot);
        erase_at_step(&mut frame, root_off, key, depth, condition, true)
            .map_err(|e| e.with_blob_guid(current_guid))?
    };

    let r = match step {
        EraseStep::Done(r) => r,
        EraseStep::Crossing(crossing) => {
            let child_pin = match bm.pin(crossing.child_guid) {
                Ok(pin) => pin,
                Err(e)
                    if is_blob_store_not_found(&e) && bm.has_delete_fence(crossing.child_guid) =>
                {
                    drop(guard);
                    return Err(stale_blob_crossing("stale blob crossing: erase deep child"));
                }
                Err(e) => return Err(e.with_blob_guid(crossing.child_guid)),
            };
            child_pin.prefetch_header();
            let child_guard = child_pin.write();

            if let Some((fork_guid, fork_pin)) = fork_child_if_shared(
                bm,
                &mut guard,
                crossing.child_guid,
                child_guard.as_slice(),
                crossing.parent_off,
                seq,
            )? {
                drop(child_guard);
                drop(child_pin);
                let fork_guard = fork_pin.write();
                drop(guard);
                let mut outcome = lock_coupled_erase_in_blob(
                    bm,
                    fork_guard,
                    fork_pin.as_ref(),
                    fork_guid,
                    false,
                    key,
                    seq,
                    condition,
                    crossing.child_depth,
                    blob_hops,
                    max_cross_blob_depth,
                )?;
                drop(fork_pin);
                // Repointing this frame's BlobNode at the fork changed it.
                if is_top_blob {
                    outcome.root_dirty = true;
                } else {
                    bm.mark_dirty_cached(current_guid, seq, current_entry);
                }
                return Ok(outcome);
            }

            drop(guard);
            let outcome = lock_coupled_erase_in_blob(
                bm,
                child_guard,
                child_pin.as_ref(),
                crossing.child_guid,
                false,
                key,
                seq,
                condition,
                crossing.child_depth,
                blob_hops,
                max_cross_blob_depth,
            )?;
            drop(child_pin);
            return Ok(outcome);
        }
    };

    let child_touched = {
        let mut frame = guard.frame();
        let root_off = decode_child_off(frame.header().root_slot);
        let child_touched = !matches!(r.signal, EraseSignal::Unchanged) || r.mutated;
        if child_touched {
            let new_root = resolve_new_root_after_erase(&mut frame, root_off, &r.signal)?;
            frame.header_mut().root_slot = encode_child_off(new_root);
        }
        child_touched
    };

    drop(guard);
    if child_touched {
        bm.note_compaction_candidate(current_guid);
        if !is_top_blob {
            bm.mark_dirty_cached(current_guid, seq, current_entry);
        }
    }

    Ok(EraseOutcome {
        root_dirty: is_top_blob && child_touched,
        mutated: r.mutated,
    })
}

fn resolve_new_root_after_erase(
    frame: &mut BlobFrame<'_>,
    root_off: u32,
    signal: &EraseSignal,
) -> Result<u32> {
    match signal {
        EraseSignal::Unchanged => Ok(root_off),
        EraseSignal::Replaced(off) => Ok(*off),
        EraseSignal::SubtreeGone => {
            // The whole tree is empty — re-seed the EmptyRoot
            // sentinel so subsequent lookups return NotFound and
            // subsequent inserts replace the sentinel cleanly. Stamp
            // its self-describing `node_type @ +1` byte so the
            // offset-addressed reader resolves it as EmptyRoot.
            let out = frame.alloc_node(NodeType::EmptyRoot)?;
            let off = frame
                .offset_of_slot(out.slot)
                .ok_or(Error::node_corrupt("resolve_new_root: sentinel offset"))?;
            if let Some(body) = frame.bytes_at_mut(off, 8) {
                body[1] = NodeType::EmptyRoot.as_u8();
            }
            Ok(off)
        }
    }
}

// ---------- recursive dispatch ----------

#[cfg(test)]
pub(super) fn erase_at(
    frame: &mut BlobFrame<'_>,
    off: u32,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    match erase_at_step(
        frame,
        off,
        SearchKey::exact(key),
        depth,
        EraseCondition::Always,
        false,
    )? {
        EraseStep::Done(r) => Ok(r),
        EraseStep::Crossing(_) => Err(Error::NotYetImplemented(
            "walker::erase_at: BlobNode crossing requires BufferManager — use erase_multi",
        )),
    }
}

#[allow(clippy::too_many_arguments)] // condition/crossing flags mirror every node arm
fn erase_at_step(
    frame: &mut BlobFrame<'_>,
    off: u32,
    key: SearchKey<'_>,
    depth: usize,
    condition: EraseCondition,
    allow_crossing: bool,
) -> Result<EraseStep> {
    let ntype = ntype_of(frame.as_ref(), off)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::erase_at: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot => Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        })),
        NodeType::Leaf => erase_at_leaf(frame, off, key, condition).map(EraseStep::Done),
        NodeType::Prefix => erase_at_prefix_step(frame, off, key, depth, condition, allow_crossing),
        NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
            erase_at_inner_step(frame, off, ntype, key, depth, condition, allow_crossing)
        }
        NodeType::Blob => {
            if allow_crossing {
                blob_node_erase_step(frame, off, key, depth)
            } else {
                Err(Error::NotYetImplemented(
                    "walker::erase_at: BlobNode crossing requires BufferManager — use erase_multi",
                ))
            }
        }
    }
}

fn blob_node_erase_step(
    frame: &BlobFrame<'_>,
    off: u32,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<EraseStep> {
    let body = frame.body_at_offset(off).ok_or(Error::node_corrupt(
        "blob_node_erase_step: BlobNode body resolution failed",
    ))?;
    let bn = *cast::<BlobNode>(body);
    let plen = bn.prefix_len as usize;
    if plen > BLOB_MAX_INLINE {
        return Err(Error::node_corrupt(
            "blob_node_erase_step: BlobNode prefix_len exceeds inline buffer",
        ));
    }
    if !key.range_eq(depth, &bn.bytes[..plen]) {
        return Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        }));
    }
    Ok(EraseStep::Crossing(EraseBlobCrossing {
        child_guid: bn.child_blob_guid,
        child_depth: depth + plen,
        parent_off: off,
    }))
}

/// Soft-delete a leaf in place: flip its `tombstone` byte and bump
/// the blob's `tombstone_leaf_cnt`. The leaf body stays in its slot
/// (so the parent never sees the deletion) and its bytes stay
/// allocated until [`super::compact_blob`] rebuilds the blob.
///
/// Only the single `tombstone` byte in the 16-byte header is written
/// (the rest of the variable-size body is untouched).
///
/// Returns `EraseSignal::Unchanged` so descending callers do not
/// rewire parents — structural collapse is now a compaction-time
/// responsibility.
///
/// Replaying an erase against an already-tombstoned leaf is a
/// no-op and the counter is not double-bumped.
fn erase_at_leaf(
    frame: &mut BlobFrame<'_>,
    leaf_off: u32,
    key: SearchKey<'_>,
    condition: EraseCondition,
) -> Result<EraseReturn> {
    // Always read the existing key; the value bytes are not needed
    // for delete.
    let leaf = {
        let (existing_key, leaf) = read_leaf_key_ref(frame.as_ref(), leaf_off)?;
        if !key.eq_slice(existing_key) {
            return Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                mutated: false,
            });
        }
        leaf
    };
    if leaf.tombstone != 0 {
        // Already soft-deleted — replay-idempotent.
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        });
    }
    if let EraseCondition::IfVersion(expected) = condition {
        if leaf.seq != expected {
            return Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                mutated: false,
            });
        }
    }
    // Flip only the tombstone byte in the contiguous body's header;
    // the key/value bytes that follow are left in place.
    {
        let body = frame
            .body_at_offset_mut(leaf_off)
            .ok_or(Error::node_corrupt("erase_at_leaf: leaf body"))?;
        body[offset_of!(Leaf, tombstone)] = 1;
    }
    let h = frame.header_mut();
    h.tombstone_leaf_cnt = h.tombstone_leaf_cnt.saturating_add(1);
    Ok(EraseReturn {
        signal: EraseSignal::Unchanged,
        mutated: true,
    })
}

fn erase_at_prefix_step(
    frame: &mut BlobFrame<'_>,
    pfx_off: u32,
    key: SearchKey<'_>,
    depth: usize,
    condition: EraseCondition,
    allow_crossing: bool,
) -> Result<EraseStep> {
    // `Prefix` is `Copy` — `p` is owned on the stack, so we can
    // hold `&p.bytes[..plen]` across the `frame.*` mutations
    // without needing a `.to_vec()` (mirror of `insert_into_prefix`'s
    // borrow-only descent).
    let p = read_prefix(frame.as_ref(), pfx_off)?;
    let plen = p.prefix_len as usize;
    let prefix_bytes = &p.bytes[..plen];
    let child_off = child_offset(p.child as u16);

    if !key.range_eq(depth, prefix_bytes) {
        return Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        }));
    }

    let r = erase_at_step(
        frame,
        child_off,
        key,
        depth + plen,
        condition,
        allow_crossing,
    )?;
    let EraseStep::Done(r) = r else {
        return Ok(r);
    };
    match r.signal {
        EraseSignal::Unchanged => Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: r.mutated,
        })),
        EraseSignal::Replaced(new_child_off) => {
            set_prefix_child(frame, pfx_off, new_child_off)?;
            Ok(EraseStep::Done(EraseReturn {
                signal: EraseSignal::Unchanged,
                mutated: r.mutated,
            }))
        }
        EraseSignal::SubtreeGone => {
            // Abandon-on-free: the Prefix is unreachable once the
            // parent drops this edge; reclaimed at next compaction.
            frame.note_abandoned(pfx_off);
            Ok(EraseStep::Done(EraseReturn {
                signal: EraseSignal::SubtreeGone,
                mutated: r.mutated,
            }))
        }
    }
}

#[allow(clippy::too_many_arguments)] // mirrors erase_at_step's call shape
fn erase_at_inner_step(
    frame: &mut BlobFrame<'_>,
    inner_off: u32,
    ntype: NodeType,
    key: SearchKey<'_>,
    depth: usize,
    condition: EraseCondition,
    allow_crossing: bool,
) -> Result<EraseStep> {
    let Some(byte) = key.byte_at(depth) else {
        return Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        }));
    };
    let Some(child_off) = inner_find_child(frame, inner_off, ntype, byte)? else {
        return Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        }));
    };

    let r = erase_at_step(frame, child_off, key, depth + 1, condition, allow_crossing)?;
    let EraseStep::Done(r) = r else {
        return Ok(r);
    };
    match r.signal {
        EraseSignal::Unchanged => Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: r.mutated,
        })),
        EraseSignal::Replaced(new_child_off) => {
            inner_update_child(frame, inner_off, ntype, byte, new_child_off)?;
            Ok(EraseStep::Done(EraseReturn {
                signal: EraseSignal::Unchanged,
                mutated: r.mutated,
            }))
        }
        EraseSignal::SubtreeGone => {
            let sig = inner_remove_child_and_collapse(frame, inner_off, ntype, byte)?;
            Ok(EraseStep::Done(EraseReturn {
                signal: sig,
                mutated: r.mutated,
            }))
        }
    }
}

/// Remove `byte` from `slot`'s child set. After removal:
/// - `count == 0` → free the inner node, signal `SubtreeGone`.
/// - `count == 1` → free the inner node, wrap the lone child in a
///   `Prefix([surviving_byte])` so descendant depth indexing stays
///   valid, signal `Replaced(prefix_slot)`.
/// - `count` dropped to the shrink threshold for the current
///   `NodeType` → allocate the next-smaller variant
///   (`Node256→Node48`, `Node48→Node16`, `Node16→Node4`), copy the
///   remaining children across, free the old slot, signal
///   `Replaced(new_slot)`. Thresholds (12, 37, 3) leave hysteresis
///   so a single re-insert doesn't immediately grow back.
/// - otherwise → rewrite the body in place, signal `Unchanged`.
///
/// The `Prefix` wrap on lone-child collapse is load-bearing: an
/// inner-node child sits one byte deeper in the descent than its
/// parent, so dropping the inner node without re-inserting its
/// pointing-byte breaks every leaf below it.
#[allow(clippy::too_many_lines)] // intentional — one match over 4 NodeTypes
fn inner_remove_child_and_collapse(
    frame: &mut BlobFrame<'_>,
    off: u32,
    ntype: NodeType,
    byte: u8,
) -> Result<EraseSignal> {
    match ntype {
        NodeType::Node4 => {
            let mut n = read_node4(frame.as_ref(), off)?;
            let count = (n.count as usize).min(4);
            let mut idx = None;
            for i in 0..count {
                if n.keys[i] == byte {
                    idx = Some(i);
                    break;
                }
            }
            let i = idx.ok_or(Error::node_corrupt(
                "inner_remove_child_and_collapse: byte not present (Node4)",
            ))?;
            for j in i..count - 1 {
                n.keys[j] = n.keys[j + 1];
                n.children[j] = n.children[j + 1];
            }
            n.keys[count - 1] = 0;
            n.children[count - 1] = 0;
            n.count -= 1;
            finish_inner_with_sorted(frame, off, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node16 => {
            let mut n = read_node16(frame.as_ref(), off)?;
            let count = (n.count as usize).min(16);
            let i = simd::node16_find_byte(&n.keys, n.count, byte)
                .map(usize::from)
                .ok_or(Error::node_corrupt(
                    "inner_remove_child_and_collapse: byte not present (Node16)",
                ))?;
            for j in i..count - 1 {
                n.keys[j] = n.keys[j + 1];
                n.children[j] = n.children[j + 1];
            }
            n.keys[count - 1] = 0;
            n.children[count - 1] = 0;
            n.count -= 1;

            // Try shrinking to Node4 before the count<=1 paths. The old
            // Node16 is abandoned (abandon-on-free); the Prefix-wrap
            // below handles count==1.
            if n.count >= 2 && n.count <= SHRINK_NODE16_TO_NODE4_AT {
                let shrunk = shrink_node16_to_node4(frame, n)?;
                frame.note_abandoned(off);
                return Ok(EraseSignal::Replaced(shrunk));
            }
            finish_inner_with_sorted(frame, off, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node48 => {
            let mut n = read_node48(frame.as_ref(), off)?;
            let ci = n.index[byte as usize];
            if ci == 0 {
                return Err(Error::node_corrupt(
                    "inner_remove_child_and_collapse: byte not present (Node48)",
                ));
            }
            n.children[(ci as usize) - 1] = 0;
            n.index[byte as usize] = 0;
            n.count -= 1;

            if n.count == 0 {
                frame.note_abandoned(off); // abandon-on-free
                return Ok(EraseSignal::SubtreeGone);
            }
            if n.count == 1 {
                let (surviving_byte, surviving_child_enc) = {
                    let b = simd::find_next_nonzero_byte(&n.index, 0).ok_or(
                        Error::node_corrupt("inner_remove_child_and_collapse: empty Node48"),
                    )?;
                    (b as u8, n.children[(n.index[b] as usize) - 1])
                };
                let new_off = write_prefix_chain(
                    frame,
                    &[surviving_byte],
                    child_offset(surviving_child_enc),
                )?;
                frame.note_abandoned(off); // abandon-on-free: collapsed Node48
                return Ok(EraseSignal::Replaced(new_off));
            }
            if n.count <= SHRINK_NODE48_TO_NODE16_AT {
                let shrunk = shrink_node48_to_node16(frame, n)?;
                frame.note_abandoned(off);
                return Ok(EraseSignal::Replaced(shrunk));
            }
            write_struct_at(frame, off, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        NodeType::Node256 => {
            let mut n = read_node256(frame.as_ref(), off)?;
            if n.children[byte as usize] == 0 {
                return Err(Error::node_corrupt(
                    "inner_remove_child_and_collapse: byte not present (Node256)",
                ));
            }
            n.children[byte as usize] = 0;
            n.count = n.count.saturating_sub(1);

            if n.count == 0 {
                frame.note_abandoned(off); // abandon-on-free
                return Ok(EraseSignal::SubtreeGone);
            }
            if n.count == 1 {
                let (surviving_byte, surviving_child_enc) = {
                    let b = simd::find_next_nonzero_u16(&n.children, 0).ok_or(
                        Error::node_corrupt("inner_remove_child_and_collapse: empty Node256"),
                    )?;
                    (b as u8, n.children[b])
                };
                let new_off = write_prefix_chain(
                    frame,
                    &[surviving_byte],
                    child_offset(surviving_child_enc),
                )?;
                frame.note_abandoned(off); // abandon-on-free: collapsed Node256
                return Ok(EraseSignal::Replaced(new_off));
            }
            if n.count <= SHRINK_NODE256_TO_NODE48_AT {
                let shrunk = shrink_node256_to_node48(frame, n)?;
                frame.note_abandoned(off);
                return Ok(EraseSignal::Replaced(shrunk));
            }
            write_struct_at(frame, off, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        _ => Err(Error::node_corrupt(
            "inner_remove_child_and_collapse: not an inner node",
        )),
    }
}
