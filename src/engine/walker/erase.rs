//! Erase path — `erase` / `erase_multi` + recursive `erase_at`
//! dispatch + per-NodeType arms + collapse-on-lone-child rewiring.

use crate::api::errors::{Error, Result};
use crate::layout::{BlobNode, NodeType, BLOB_MAX_INLINE};
use std::sync::Arc;

use super::cast;
use super::lookup::lookup_at;
use super::readers::{
    ntype_of, read_leaf_key_ref, read_node16, read_node256, read_node4, read_node48, read_prefix,
};
use super::route::{pin_route_parent, route_pin_not_found, validate_route_edge};
use super::types::{EraseCondition, EraseOutcome, EraseReturn, EraseSignal, LookupResult};
use super::writers::{
    finish_inner_with_sorted, inner_find_child, inner_update_child, set_prefix_child,
    shrink_node16_to_node4, shrink_node256_to_node48, shrink_node48_to_node16, write_prefix_chain,
    write_struct_to_slot, SHRINK_NODE16_TO_NODE4_AT, SHRINK_NODE256_TO_NODE48_AT,
    SHRINK_NODE48_TO_NODE16_AT,
};
use super::SearchKey;
use crate::engine::RouteCache;
use crate::store::BlobWriteGuard;
use crate::store::{BlobFrame, BlobFrameRef, BufferManager, CachedBlob};

// ---------- public entry points ----------

/// Single-blob erase. Surfaces [`Error::NotYetImplemented`] if the
/// descent reaches a [`NodeType::Blob`] crossing — callers wanting
/// cross-blob erase should use [`erase_multi`].
///
/// Updates `header.root_slot` in place.
#[cfg(test)]
pub(super) fn erase(frame: &mut BlobFrame<'_>, root_slot: u16, key: &[u8]) -> Result<EraseOutcome> {
    let r = erase_at(frame, root_slot, key, 0)?;
    let root_dirty = r.mutated || !matches!(r.signal, EraseSignal::Unchanged);
    let new_root = resolve_new_root_after_erase(frame, root_slot, &r.signal)?;
    frame.header_mut().root_slot = new_root;
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
                let child_pin = bm.pin(crossing.child_guid)?;
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
        &mut blob_hops,
        &mut max_cross_blob_depth,
    );
    if outcome.is_ok() {
        bm.note_walker_blob_hops(blob_hops, max_cross_blob_depth);
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
        Err(e) if route_pin_not_found(&e) => {
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
        Err(e) if route_pin_not_found(&e) => {
            drop(parent_guard);
            cache.invalidate(key, route);
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
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
        let root_slot = frame.header().root_slot;
        erase_at_step(&mut frame, root_slot, key, depth, condition, true)
            .map_err(|e| e.with_blob_guid(current_guid))?
    };

    let r = match step {
        EraseStep::Done(r) => r,
        EraseStep::Crossing(crossing) => {
            let child_pin = bm.pin(crossing.child_guid)?;
            child_pin.prefetch_header();
            let child_guard = child_pin.write();
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
        let root_slot = frame.header().root_slot;
        let child_touched = !matches!(r.signal, EraseSignal::Unchanged) || r.mutated;
        if child_touched {
            let new_root = resolve_new_root_after_erase(&mut frame, root_slot, &r.signal)?;
            frame.header_mut().root_slot = new_root;
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
    root_slot: u16,
    signal: &EraseSignal,
) -> Result<u16> {
    match signal {
        EraseSignal::Unchanged => Ok(root_slot),
        EraseSignal::Replaced(s) => Ok(*s),
        EraseSignal::SubtreeGone => {
            // The whole tree is empty — re-seed the EmptyRoot
            // sentinel so subsequent lookups return NotFound and
            // subsequent inserts replace the sentinel cleanly.
            let out = frame.alloc_node(NodeType::EmptyRoot)?;
            Ok(out.slot)
        }
    }
}

// ---------- recursive dispatch ----------

#[cfg(test)]
pub(super) fn erase_at(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    match erase_at_step(
        frame,
        slot,
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
    slot: u16,
    key: SearchKey<'_>,
    depth: usize,
    condition: EraseCondition,
    allow_crossing: bool,
) -> Result<EraseStep> {
    let ntype = ntype_of(frame.as_ref(), slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::erase_at: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot => Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        })),
        NodeType::Leaf => erase_at_leaf(frame, slot, key, condition).map(EraseStep::Done),
        NodeType::Prefix => {
            erase_at_prefix_step(frame, slot, key, depth, condition, allow_crossing)
        }
        NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
            erase_at_inner_step(frame, slot, ntype, key, depth, condition, allow_crossing)
        }
        NodeType::Blob => {
            if allow_crossing {
                blob_node_erase_step(frame, slot, key, depth)
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
    slot: u16,
    key: SearchKey<'_>,
    depth: usize,
) -> Result<EraseStep> {
    let body = frame.body_of_slot(slot).ok_or(Error::node_corrupt(
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
    }))
}

/// Soft-delete a leaf in place: flip its `tombstone` byte and bump
/// the blob's `tombstone_leaf_cnt`. The leaf body stays in its slot
/// (so the parent never sees the deletion) and the extent bytes
/// stay allocated until [`super::compact_blob`] rebuilds the blob.
///
/// Returns `EraseSignal::Unchanged` so descending callers do not
/// rewire parents — structural collapse is now a compaction-time
/// responsibility.
///
/// Replaying an erase against an already-tombstoned leaf is a
/// no-op and the counter is not double-bumped.
fn erase_at_leaf(
    frame: &mut BlobFrame<'_>,
    leaf_slot: u16,
    key: SearchKey<'_>,
    condition: EraseCondition,
) -> Result<EraseReturn> {
    // Always read the existing key; the value bytes are not needed
    // for delete.
    let leaf = {
        let (existing_key, leaf) = read_leaf_key_ref(frame.as_ref(), leaf_slot)?;
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
    let mut new_leaf = leaf;
    new_leaf.tombstone = 1;
    write_struct_to_slot(frame, leaf_slot, &new_leaf)?;
    let h = frame.header_mut();
    h.tombstone_leaf_cnt = h.tombstone_leaf_cnt.saturating_add(1);
    Ok(EraseReturn {
        signal: EraseSignal::Unchanged,
        mutated: true,
    })
}

#[allow(clippy::too_many_arguments)] // mirrors erase_at_step's call shape
fn erase_at_prefix_step(
    frame: &mut BlobFrame<'_>,
    pfx_slot: u16,
    key: SearchKey<'_>,
    depth: usize,
    condition: EraseCondition,
    allow_crossing: bool,
) -> Result<EraseStep> {
    // `Prefix` is `Copy` — `p` is owned on the stack, so we can
    // hold `&p.bytes[..plen]` across the `frame.*` mutations
    // without needing a `.to_vec()` (mirror of `insert_into_prefix`'s
    // borrow-only descent).
    let p = read_prefix(frame.as_ref(), pfx_slot)?;
    let plen = p.prefix_len as usize;
    let prefix_bytes = &p.bytes[..plen];
    let child_slot = p.child as u16;

    if !key.range_eq(depth, prefix_bytes) {
        return Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        }));
    }

    let r = erase_at_step(
        frame,
        child_slot,
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
        EraseSignal::Replaced(new_child) => {
            set_prefix_child(frame, pfx_slot, u32::from(new_child))?;
            Ok(EraseStep::Done(EraseReturn {
                signal: EraseSignal::Unchanged,
                mutated: r.mutated,
            }))
        }
        EraseSignal::SubtreeGone => {
            frame.free_node(pfx_slot)?;
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
    inner_slot: u16,
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
    let Some(child) = inner_find_child(frame, inner_slot, ntype, byte)? else {
        return Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: false,
        }));
    };

    let r = erase_at_step(frame, child, key, depth + 1, condition, allow_crossing)?;
    let EraseStep::Done(r) = r else {
        return Ok(r);
    };
    match r.signal {
        EraseSignal::Unchanged => Ok(EraseStep::Done(EraseReturn {
            signal: EraseSignal::Unchanged,
            mutated: r.mutated,
        })),
        EraseSignal::Replaced(new_child) => {
            inner_update_child(frame, inner_slot, ntype, byte, u32::from(new_child))?;
            Ok(EraseStep::Done(EraseReturn {
                signal: EraseSignal::Unchanged,
                mutated: r.mutated,
            }))
        }
        EraseSignal::SubtreeGone => {
            let sig = inner_remove_child_and_collapse(frame, inner_slot, ntype, byte)?;
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
    slot: u16,
    ntype: NodeType,
    byte: u8,
) -> Result<EraseSignal> {
    match ntype {
        NodeType::Node4 => {
            let mut n = read_node4(frame.as_ref(), slot)?;
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
            finish_inner_with_sorted(frame, slot, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node16 => {
            let mut n = read_node16(frame.as_ref(), slot)?;
            let count = (n.count as usize).min(16);
            let mut idx = None;
            for i in 0..count {
                if n.keys[i] == byte {
                    idx = Some(i);
                    break;
                }
            }
            let i = idx.ok_or(Error::node_corrupt(
                "inner_remove_child_and_collapse: byte not present (Node16)",
            ))?;
            for j in i..count - 1 {
                n.keys[j] = n.keys[j + 1];
                n.children[j] = n.children[j + 1];
            }
            n.keys[count - 1] = 0;
            n.children[count - 1] = 0;
            n.count -= 1;

            // Try shrinking to Node4 before the count<=1 paths so
            // that the freed Node16 slot is the only old slot we
            // hand back to the free list (the Prefix-wrap below
            // already does that for count==1).
            if n.count >= 2 && n.count <= SHRINK_NODE16_TO_NODE4_AT {
                let shrunk = shrink_node16_to_node4(frame, slot, n)?;
                return Ok(EraseSignal::Replaced(shrunk));
            }
            finish_inner_with_sorted(frame, slot, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node48 => {
            let mut n = read_node48(frame.as_ref(), slot)?;
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
                let new_slot =
                    write_prefix_chain(frame, &[surviving_byte], surviving_child as u16)?;
                return Ok(EraseSignal::Replaced(new_slot));
            }
            if n.count <= SHRINK_NODE48_TO_NODE16_AT {
                let shrunk = shrink_node48_to_node16(frame, slot, n)?;
                return Ok(EraseSignal::Replaced(shrunk));
            }
            write_struct_to_slot(frame, slot, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        NodeType::Node256 => {
            let mut n = read_node256(frame.as_ref(), slot)?;
            if n.children[byte as usize] == 0 {
                return Err(Error::node_corrupt(
                    "inner_remove_child_and_collapse: byte not present (Node256)",
                ));
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
                let new_slot =
                    write_prefix_chain(frame, &[surviving_byte], surviving_child as u16)?;
                return Ok(EraseSignal::Replaced(new_slot));
            }
            if n.count <= SHRINK_NODE256_TO_NODE48_AT {
                let shrunk = shrink_node256_to_node48(frame, slot, n)?;
                return Ok(EraseSignal::Replaced(shrunk));
            }
            write_struct_to_slot(frame, slot, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        _ => Err(Error::node_corrupt(
            "inner_remove_child_and_collapse: not an inner node",
        )),
    }
}
