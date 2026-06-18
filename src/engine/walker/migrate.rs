//! Deep-clone primitives — `make_blob_from_node` (spillover) and
//! `compact_blob` (in-place repack). Share the same recursive
//! `clone_subtree` machinery; both produce a fresh, packed image
//! containing a deep copy of a source subtree.
//!
//! `clone_subtree` runs in two modes:
//!
//! - **preserve** (`filter_tombstones = false`) — copies every byte
//!   verbatim, tombstones included. The result is always `Some`.
//!   Used by `make_blob_from_node` to migrate a subtree wholesale
//!   into a fresh blob without changing its observable shape.
//! - **filter** (`filter_tombstones = true`) — drops tombstoned
//!   leaves and collapses inner nodes whose live-child count falls
//!   below the natural threshold (lone-child → `Prefix([byte])`;
//!   smaller-tier `NodeType` if the count slips below its grow
//!   point). Returns `None` only when the whole subtree under
//!   `src_slot` has no live leaves. Used by `compact_blob` to
//!   reclaim tombstone leaves + bump-area waste in one rebuild.

use crate::api::errors::{Error, Result};
use crate::layout::{
    size_of_node, BlobGuid, BlobNode, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix,
    BLOB_MAX_INLINE, DATA_AREA_START, MAX_SLOTS, PAGE_SIZE, PREFIX_MAX_INLINE,
};
use crate::store::blob_store::AlignedBlobBuf;
use crate::store::{
    bloom_byte_len, decode_child_off, encode_child_off, page_align_up, BlobFrame, BlobFrameRef,
    BloomBuilder, BufferManager, BLOOM_BITS_PER_KEY, PAGE_4K, SPILLOVER_RESERVATION,
};

use super::cast;
use super::cow::child_is_snapshot_shared;
use super::readers::child_offset;
use super::types::MakeBlobOutcome;
use super::writers::{write_prefix_chain, write_struct_at};

/// Conservative bump-area headroom kept free during a merge.
///
/// Larger than `SPILLOVER_RESERVATION` (128 B) so the parent
/// retains room for slot-table growth + a future emergency
/// spillover after the merge completes. Tuning past 4 KB rarely
/// helps; smaller leaves merges flaky under realistic workloads.
const MERGE_RESERVE: u32 = 0x1000;

/// Deep-clone the subtree rooted at `src_slot` of `src_frame` into
/// a fresh 512 KB blob keyed by `new_guid`.
///
/// Used by spillover: when an insert into a blob overflows, the
/// caller migrates a subtree out via this primitive, installs a
/// [`BlobNode`] placeholder where the subtree used to live, and
/// writes both blobs back.
///
/// **Leaf bodies are deep-copied verbatim** — each leaf is one
/// contiguous, self-describing node (`[16B header][key][value]`), so
/// the clone bump-allocates a same-size leaf in the new blob's data
/// area and copies the bytes across (no offset to repoint). The
/// original blob is untouched; freeing the migrated slots is the
/// caller's responsibility (typical pattern is one
/// `BlobFrame::free_node` per migrated slot).
///
/// Migration is **preserve-mode** — tombstones in the source travel
/// to the destination verbatim. Compaction (in either blob) is the
/// place to drop them.
#[cfg(test)]
pub fn make_blob_from_node(
    src_frame: &BlobFrame<'_>,
    src_off: u32,
    new_guid: BlobGuid,
) -> Result<MakeBlobOutcome> {
    make_blob_from_node_with_buf(AlignedBlobBuf::zeroed(), src_frame, src_off, new_guid)
}

/// Same as [`make_blob_from_node`], but allocates the destination
/// from the buffer manager's store-preferred allocator. Spillover
/// uses this path so fresh child blobs enter the cache backed by
/// registered `io_uring` buffers when the persistent store has a
/// fixed-buffer pool.
pub fn make_blob_from_node_in(
    bm: &BufferManager,
    src_frame: &BlobFrame<'_>,
    src_off: u32,
    new_guid: BlobGuid,
) -> Result<MakeBlobOutcome> {
    make_blob_from_node_with_buf(bm.alloc_blob_buf_zeroed(), src_frame, src_off, new_guid)
}

fn make_blob_from_node_with_buf(
    mut buf: AlignedBlobBuf,
    src_frame: &BlobFrame<'_>,
    src_off: u32,
    new_guid: BlobGuid,
) -> Result<MakeBlobOutcome> {
    let cloned_root_off;
    {
        let mut new_frame = BlobFrame::init(buf.as_mut_slice(), new_guid)?;
        // Spillover builds a fresh legacy blob (routed = false); the
        // dummy cursor is unused. Freshly-spilled blobs are born legacy
        // and get the routing layout at their first compaction.
        let mut leaf_cursor = 0u32;
        cloned_root_off = clone_subtree(
            src_frame,
            &mut new_frame,
            src_off,
            false,
            false,
            &mut leaf_cursor,
        )?
        .expect("preserve mode never returns None");
        // The EmptyRoot sentinel `BlobFrame::init` seeded at
        // DATA_AREA_START is now unreachable (the cloned root sits
        // after it); abandon-on-free leaves its 8 bytes to be reclaimed
        // by a future compaction of this fresh blob. Record the cloned
        // root's encoded offset.
        new_frame.header_mut().root_slot = encode_child_off(cloned_root_off);
    }
    Ok(MakeBlobOutcome { buf })
}

/// Threshold of abandoned (`dead_bytes`) weight that, on its own,
/// makes a rebuild worth the 512 KB scratch alloc + memcpy. Roughly
/// 6% of the data-area capacity — small enough to keep a churny blob
/// from bloating, large enough not to compact on a couple of node
/// grows.
const DEAD_BYTES_COMPACT_THRESHOLD: u32 = (PAGE_SIZE - DATA_AREA_START) / 16;

/// Cheap header-level predicate for whether `compact_blob` can
/// reclaim anything worthwhile.
///
/// v4 uses abandon-on-free: structural ops (node grow/shrink/collapse,
/// leaf value-grow realloc, EmptyRoot replacement, prefix split) don't
/// return their old node to a free list — they leave it unreachable
/// and bump `header.dead_bytes`. The per-NodeType free lists are no
/// longer populated by the walker, so the old free-list trigger is
/// dead; the dead-bytes counter is the churn signal now. Tombstoned
/// leaves keep their (contiguous) bodies until compaction too, so a
/// non-zero `tombstone_leaf_cnt` is also a trigger.
#[must_use]
pub fn blob_needs_compaction(frame: BlobFrameRef<'_>) -> bool {
    let h = frame.header();
    h.tombstone_leaf_cnt != 0 || h.dead_bytes >= DEAD_BYTES_COMPACT_THRESHOLD
}

/// Whether compacting `frame` **now** would produce a routed
/// (page-granular cold-read) layout.
///
/// A read-only pass-0 measurement ([`routing_budget`] +
/// [`routing_geometry`]) — no allocation, no mutation. The maintenance
/// scheduler uses it to lazily route blobs that settled write-cold
/// without ever accruing the tombstone / dead-byte churn that
/// [`blob_needs_compaction`] keys on — e.g. a bulk-loaded, write-once
/// dataset whose blobs are born legacy and would otherwise stay legacy
/// forever, pinning a full 512 KB frame on every cold point read.
///
/// Returns `false` for a blob that is **already routed**
/// (`routing_len != 0`) — recompacting it gains nothing — and for any
/// blob [`routing_geometry`] would steer to the legacy layout
/// (degenerate tree, below `ROUTE_MIN_LEAF_BYTES`, or doesn't fit the
/// two-arena layout). That "false unless it will actually route"
/// property is what keeps the scheduler from re-compacting the same
/// won't-route blob every maintenance cycle.
#[must_use]
pub fn blob_would_route(frame: BlobFrameRef<'_>) -> bool {
    let h = frame.header();
    if h.routing_len != 0 {
        return false; // already routed — nothing to gain
    }
    if h.routing_unfit != 0 {
        // A prior compaction intended to route this blob but its clone
        // overran the measured budget and fell back to legacy. Don't
        // re-schedule it forever; a later churn compaction (which clears
        // the flag by rebuilding from zero) re-evaluates routability.
        return false;
    }
    let root_off = decode_child_off(h.root_slot);
    match routing_budget(frame, root_off) {
        Ok(budget) => routing_geometry(budget).is_some(),
        // Undecodable source — leave it to `blob_needs_compaction`;
        // never schedule a routing rewrite we can't measure.
        Err(_) => false,
    }
}

/// Repack `buf` in place, discarding all unreachable bytes plus
/// every tombstoned leaf.
///
/// Builds a fresh `BlobFrame` image in a scratch `AlignedBlobBuf`,
/// deep-clones the live subtree from `buf` into it under
/// **filter-mode** (tombstones dropped, inner-node collapse
/// applied wherever a live-child count falls below its
/// `NodeType`'s threshold), then memcpys the scratch image back
/// over `buf`.
///
/// Post-conditions on the rebuilt blob:
///
/// - Packed data area. In the legacy layout every byte in
///   `DATA_AREA_START .. space_used` is live. In the routing layout
///   (`routing_len != 0`), internal nodes occupy
///   `[routing_off, routing_off + routing_len)` and leaves are
///   page-aligned at `[leaf_region_start, space_used)`; the gap between
///   `routing_off + routing_len` and `leaf_region_start` is an expected
///   dead page-alignment span (≤ one 4 KB page), NOT reclaimable churn.
/// - Empty free lists (no leftover stale slot entries).
/// - `tombstone_leaf_cnt = 0` (every survivor is by definition live).
/// - `compact_times` bumped by one.
/// - `gap_space` reset to whatever fresh allocations report.
/// - Original `blob_guid` preserved.
/// - If every leaf in the source was tombstoned, the root becomes
///   the freshly-allocated `EmptyRoot` sentinel.
///
/// **What this reclaims:** leaf bodies whose slots returned to the
/// class-0 free list (alloc-fresh same-key updates), dead node bodies
/// whose slots returned to a per-NodeType free list but whose
/// `NodeType` isn't being allocated any more, and every (contiguous)
/// leaf body whose `tombstone` byte was set.
///
/// **What this costs:** one scratch `AlignedBlobBuf` (512 KB on
/// the heap, lives for the duration of the call) plus one full
/// blob memcpy at the end. Roughly tens of µs on a modern machine.
pub fn compact_blob(buf: &mut AlignedBlobBuf) -> Result<()> {
    let (blob_guid, old_root_off, old_compact_times) = {
        let old_frame = BlobFrame::wrap(buf.as_mut_slice());
        let h = old_frame.header();
        (h.blob_guid, decode_child_off(h.root_slot), h.compact_times)
    };

    // Pass 0: measure the live subtree so the page-aligned leaf region
    // is fixed BEFORE the clone. With `leaf_region_start` known up
    // front, the clone places each leaf at its final offset and the
    // post-order back-patch in `clone_*` is unchanged. `None` ⇒ keep
    // the legacy whole-frame layout (degenerate tree, or it doesn't fit
    // the two-arena layout).
    let (routed_lrs, bloom_leaf_count) = {
        let old_frame = BlobFrame::wrap(buf.as_mut_slice());
        let budget = routing_budget(old_frame.as_ref(), old_root_off)?;
        let leaf_count = budget.as_ref().map_or(0, |b| b.leaf_count);
        (routing_geometry(budget), leaf_count)
    };

    let mut new_buf = buf.zeroed_like();
    let mut lrs_opt = routed_lrs;
    loop {
        let overran = {
            let mut new_frame = BlobFrame::init(new_buf.as_mut_slice(), blob_guid)?;
            let old_frame = BlobFrame::wrap(buf.as_mut_slice());
            let mut leaf_cursor = lrs_opt.unwrap_or(0);
            let cloned = clone_subtree(
                &old_frame,
                &mut new_frame,
                old_root_off,
                true,
                lrs_opt.is_some(),
                &mut leaf_cursor,
            )?;
            let entry_off = match cloned {
                Some(off) => off,
                None => {
                    // Every leaf below the old root was tombstoned — the
                    // new tree is empty. The EmptyRoot sentinel `init`
                    // already seeded at DATA_AREA_START (encoded root 1)
                    // is the new root; reuse it.
                    decode_child_off(1)
                }
            };

            // Release-safe guard: in a routed build internal nodes bump
            // `space_used` (the routing arena). If it crossed
            // `leaf_region_start`, pass-0 under-counted (a budget/clone
            // drift bug) and the internals overlap the leaves just
            // written — the image is unusable. Fall back to a legacy
            // rebuild rather than persist corruption.
            let overran = lrs_opt.is_some_and(|lrs| new_frame.header().space_used > lrs);
            if !overran {
                let internal_end = new_frame.header().space_used;
                {
                    let h = new_frame.header_mut();
                    h.root_slot = encode_child_off(entry_off);
                    h.tombstone_leaf_cnt = 0;
                    h.dead_bytes = 0;
                    h.compact_times = old_compact_times.saturating_add(1);
                }
                if let Some(lrs) = lrs_opt {
                    {
                        // `routing_len` is the ACTUAL post-collapse internal
                        // byte count (not the budget, not incl. the
                        // alignment gap). `space_used` becomes the leaf-arena
                        // high-water so later in-place appends land above the
                        // live leaves.
                        let h = new_frame.header_mut();
                        h.routing_off = DATA_AREA_START;
                        h.routing_len = internal_end - DATA_AREA_START;
                        h.leaf_region_start = lrs;
                        h.space_used = leaf_cursor;
                    }
                    // Stage 6: build the per-blob bloom over the freshly
                    // cloned leaves and place it at the routing-region tail
                    // ([internal_end, lrs)), so the cold read loads it for
                    // free with the routing region. Routed-without-bloom is
                    // a fine fallback if it doesn't fit.
                    if let Some((boff, blen)) = build_routing_bloom(
                        &mut new_frame,
                        internal_end,
                        lrs,
                        leaf_cursor,
                        bloom_leaf_count,
                    ) {
                        let h = new_frame.header_mut();
                        h.bloom_off = boff;
                        h.bloom_len = blen;
                        h.bloom_bits_per_key = u32::from(BLOOM_BITS_PER_KEY);
                    }
                } else if routed_lrs.is_some() {
                    // We INTENDED to route (`routed_lrs` was `Some`) but
                    // the routed clone overran the measured budget and we
                    // fell back to this legacy rebuild. Record that so the
                    // maintenance scheduler (`blob_would_route`) won't keep
                    // re-selecting this blob for routing every cycle — a
                    // budget/clone drift would otherwise recompact a
                    // settled blob forever. Cleared automatically on the
                    // next compaction (the frame is rebuilt from zero).
                    new_frame.header_mut().routing_unfit = 1;
                }
            }
            overran
        };
        if !overran {
            break;
        }
        debug_assert!(
            false,
            "compact_blob: routing budget under-counted the internal arena; \
             falling back to legacy layout",
        );
        new_buf = buf.zeroed_like();
        lrs_opt = None;
    }

    buf.as_mut_slice().copy_from_slice(new_buf.as_slice());

    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "holt::engine::compact",
        blob_guid = ?&blob_guid[..4],
        compact_times = old_compact_times.saturating_add(1),
        "compact_blob: in-place rebuild complete",
    );

    Ok(())
}

// ---------- merge primitives ----------

/// Decide whether the child blob beneath `parent_bn_slot` is safe
/// to fold back into the parent in a single pass.
///
/// Returns `true` when **all** of:
///
/// 1. The combined data-area usage fits in `PAGE_SIZE` with
///    `MERGE_RESERVE` headroom (rephrased below as a
///    child-fits-into-parent-remaining test).
/// 2. The combined slot-table usage stays under `MAX_SLOTS`.
/// 3. The child has **no** own `BlobNode` crossings
///    (`child.num_ext_blobs == 0`) — the merge pass doesn't
///    unfold nested crossings; a child whose subtree itself
///    spans multiple blobs needs that handled by a separate
///    pass first.
/// 4. The child has no tombstoned leaves (`tombstone_leaf_cnt == 0`).
///    Compact the child first if the workload has just churned
///    deletes through it; merging tombstone weight is wasted work.
pub fn is_mergeable(
    bm: &BufferManager,
    parent_frame: &BlobFrame<'_>,
    parent_bn_off: u32,
) -> Result<bool> {
    let bn = read_blob_node(parent_frame, parent_bn_off)?;
    let child_pin = bm.pin(bn.child_blob_guid)?;
    if child_is_snapshot_shared(bm, child_pin.as_ref()) {
        return Ok(false);
    }
    let guard = child_pin.read();
    let child_frame = BlobFrameRef::wrap(guard.as_slice());

    let parent_h = parent_frame.header();
    let child_h = child_frame.header();

    let parent_remaining = PAGE_SIZE
        .saturating_sub(parent_h.space_used)
        .saturating_sub(MERGE_RESERVE);
    let child_data_bytes = child_h.space_used.saturating_sub(DATA_AREA_START);
    let space_ok = child_data_bytes <= parent_remaining;

    let combined_slots = u32::from(parent_h.num_slots) + u32::from(child_h.num_slots);
    let slots_ok = combined_slots <= MAX_SLOTS;

    let no_grandchild = child_h.num_ext_blobs == 0;
    let no_tombstones = child_h.tombstone_leaf_cnt == 0;

    Ok(space_ok && slots_ok && no_grandchild && no_tombstones)
}

/// Inline a child blob's subtree back into its parent, replacing
/// the cross-blob `BlobNode` crossing with the child's contents.
///
/// Reads the child via an exclusive guard, deep-clones the child's
/// entry-point subtree into `parent_frame` (preserve-mode — caller
/// should compact the child first if dropping tombstones matters),
/// optionally wraps the cloned root in the `BlobNode`'s inline
/// prefix, frees the parent's `BlobNode` slot, and drops the child
/// blob from the BM. Returns the slot in `parent_frame` where the
/// inlined subtree's root now lives.
///
/// **The caller's responsibility**: rewire the grandparent's
/// pointer to the returned slot. Typical pattern: a recursive
/// merge walker that returns the new slot up the chain so each
/// parent rewires its own child pointer; if `parent_bn_slot` was
/// the parent's `root_slot`, the caller writes the new slot back
/// into the parent's header.
///
/// `is_mergeable(bm, parent_frame, parent_bn_slot)` should return
/// `true` before this is called. Calling without that check risks
/// `OutOfSpace` mid-clone on a too-big merge or wasted work on a
/// merge that violates the no-nested-crossings precondition.
/// `seq` is stamped on the deferred-delete entry the merged
/// child generates. Callers from a user op path should pass the
/// op's WAL seq so the W2D protocol can pair the manifest delete
/// with a real WAL record. Internal callers (compact, the
/// checkpoint round's merge pass) pass
/// [`crate::store::STRUCTURAL_SEQ`] — the merge
/// has no WAL record and shouldn't pin the trim watermark.
pub fn merge_blob(
    bm: &BufferManager,
    parent_frame: &mut BlobFrame<'_>,
    parent_bn_off: u32,
    seq: u64,
) -> Result<u32> {
    let bn = read_blob_node(parent_frame, parent_bn_off)?;
    let child_guid = bn.child_blob_guid;
    let plen = (bn.prefix_len as usize).min(BLOB_MAX_INLINE);
    let prefix_bytes: Vec<u8> = bn.bytes[..plen].to_vec();

    let new_subtree_root = {
        let child_pin = bm.pin(child_guid)?;
        let mut child_guard = child_pin.write();
        let child_frame = child_guard.frame();
        let child_root_off = decode_child_off(child_frame.header().root_slot);
        // Preserve-mode legacy build into the existing parent (routed =
        // false); the dummy cursor is unused.
        let mut leaf_cursor = 0u32;
        clone_subtree(
            &child_frame,
            parent_frame,
            child_root_off,
            false,
            false,
            &mut leaf_cursor,
        )?
        .expect("preserve mode never returns None")
    };

    let inlined_root = if prefix_bytes.is_empty() {
        new_subtree_root
    } else {
        write_prefix_chain(parent_frame, &prefix_bytes, new_subtree_root)?
    };

    // Abandon-on-free: the parent's BlobNode is unreachable now that
    // the grandparent will be repointed at `inlined_root`.
    parent_frame.note_abandoned(parent_bn_off);
    // Keep external-blob accounting correct — the BlobNode is gone.
    {
        let h = parent_frame.header_mut();
        h.num_ext_blobs = h.num_ext_blobs.saturating_sub(1);
        // A merge appends the cloned subtree's internals + leaves via the
        // single `space_used` cursor, so any prior routing layout on the
        // parent is now stale (internals would sit above a stale
        // `leaf_region_start`). Demote to legacy/full-pin; the next
        // compaction re-routes the parent.
        h.routing_len = 0;
    }
    // Hand the now-orphaned child blob to the deferred-delete
    // protocol. An inline `bm.delete_blob` here would push the
    // manifest mutation to in-memory before the caller's WAL
    // flush (or, for internal callers like compact / merge_pass,
    // before the next checkpoint round's Sync); on crash that
    // would leave the parent in cache pointing at no child
    // while manifest persistence raced ahead through any
    // unrelated `store.flush`.
    bm.mark_for_delete(child_guid, seq);

    #[cfg(feature = "tracing")]
    tracing::debug!(
        target: "holt::engine::merge",
        child_guid = ?&child_guid[..4],
        parent_bn_off = parent_bn_off,
        inlined_root = inlined_root,
        "merge_blob: folded child into parent + queued delete",
    );

    Ok(inlined_root)
}

fn read_blob_node(frame: &BlobFrame<'_>, off: u32) -> Result<BlobNode> {
    let body = frame.body_at_offset(off).ok_or(Error::node_corrupt(
        "read_blob_node: body resolution failed",
    ))?;
    Ok(*cast::<BlobNode>(body))
}

// ---------- routing-region pass-0 (measure) ----------

/// Byte size [`pack_inner_node`] emits for a surviving-child count.
/// `None` ⇒ the branch is dropped (0 survivors).
///
/// This is the authoritative tier table; `pack_inner_node` and
/// [`routing_budget`] MUST agree with it (a `debug_assert` in
/// `pack_inner_node` and the `routing == full` proptest gate bind
/// them). NOTE the 1-survivor tier emits a `Prefix` (128 B) — *larger*
/// than a source `Node4` (16 B) / `Node16` (56 B) — so the routing
/// budget must use THIS size, never the source node's, or it
/// under-counts and an internal node would land past the leaf region.
fn packed_inner_size(survivors: usize) -> Option<u32> {
    match survivors {
        0 => None,
        1 => Some(size_of_node(NodeType::Prefix)),
        2..=4 => Some(size_of_node(NodeType::Node4)),
        5..=16 => Some(size_of_node(NodeType::Node16)),
        17..=48 => Some(size_of_node(NodeType::Node48)),
        _ => Some(size_of_node(NodeType::Node256)),
    }
}

/// Exact internal + surviving-leaf byte totals a filter-mode
/// [`clone_subtree`] will emit for a subtree.
struct BudgetNode {
    /// Bytes the internal nodes (routing arena) will occupy — exact,
    /// via [`packed_inner_size`] so it tracks `pack_inner_node`'s tier
    /// collapse rather than the source node sizes.
    routing_bytes: u32,
    /// Bytes the surviving leaves will occupy.
    leaf_bytes: u32,
    /// Count of surviving leaves — drives the per-blob bloom size
    /// reserved in the routing region (stage 6).
    leaf_count: u32,
}

/// Read-only **pass 0** of the routing-aware compaction: measure the
/// live subtree at `src_off` so [`compact_blob`] can fix the
/// page-aligned `leaf_region_start` BEFORE the real clone. With the
/// leaf region fixed up front, the clone places each leaf at its final
/// offset, so the post-order back-patch in `clone_*` stays unchanged
/// (no placeholder, no relocation pass).
///
/// Mirrors filter-mode `clone_subtree` arm-for-arm and
/// `pack_inner_node`'s tier collapse, so the totals are EXACT. Returns
/// `None` when the whole subtree filters away (matches `clone_subtree`
/// returning `None`). `EmptyRoot` reports 0 routing bytes so the caller
/// keeps the empty / bare-leaf-root degenerate cases in the legacy
/// layout.
///
/// **Keep in lockstep with `clone_subtree` (below) and
/// `pack_inner_node`.**
fn routing_budget(src: BlobFrameRef<'_>, src_off: u32) -> Result<Option<BudgetNode>> {
    let ntype = src
        .ntype_at(src_off)
        .ok_or(Error::node_corrupt("routing_budget: undecodable src ntype"))?;
    let body = src.body_at_offset(src_off).ok_or(Error::node_corrupt(
        "routing_budget: src body resolution failed",
    ))?;

    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "routing_budget: NodeType::Invalid in source",
        )),
        // EmptyRoot is only ever the root of an empty tree; report 0
        // routing bytes so the caller steers it to the legacy layout.
        NodeType::EmptyRoot => Ok(Some(BudgetNode {
            routing_bytes: 0,
            leaf_bytes: 0,
            leaf_count: 0,
        })),
        NodeType::Leaf => {
            let leaf = *cast::<Leaf>(&body[..std::mem::size_of::<Leaf>()]);
            if leaf.tombstone != 0 {
                return Ok(None);
            }
            Ok(Some(BudgetNode {
                routing_bytes: 0,
                leaf_bytes: body.len() as u32,
                leaf_count: 1,
            }))
        }
        NodeType::Prefix => {
            let p = *cast::<Prefix>(body);
            Ok(
                routing_budget(src, child_offset(p.child as u16))?.map(|c| BudgetNode {
                    routing_bytes: size_of_node(NodeType::Prefix) + c.routing_bytes,
                    leaf_bytes: c.leaf_bytes,
                    leaf_count: c.leaf_count,
                }),
            )
        }
        NodeType::Blob => Ok(Some(BudgetNode {
            routing_bytes: size_of_node(NodeType::Blob),
            leaf_bytes: 0,
            leaf_count: 0,
        })),
        NodeType::Node4 => {
            let n = *cast::<Node4>(body);
            let count = (n.count as usize).min(4);
            budget_inner(src, (0..count).map(|i| child_offset(n.children[i])))
        }
        NodeType::Node16 => {
            let n = *cast::<Node16>(body);
            let count = (n.count as usize).min(16);
            budget_inner(src, (0..count).map(|i| child_offset(n.children[i])))
        }
        NodeType::Node48 => {
            let n = *cast::<Node48>(body);
            budget_inner(
                src,
                (0..256usize)
                    .map(|b| n.index[b])
                    .filter(|&idx| idx != 0)
                    .map(|idx| child_offset(n.children[idx as usize - 1])),
            )
        }
        NodeType::Node256 => {
            let n = *cast::<Node256>(body);
            budget_inner(
                src,
                n.children
                    .iter()
                    .filter(|&&c| c != 0)
                    .map(|&c| child_offset(c)),
            )
        }
    }
}

/// Sum a filter-mode inner node's surviving children and add the
/// `pack_inner_node` tier size. Mirrors the inner-node arms of
/// `clone_subtree` (survivor count) + `pack_inner_node` (tier).
fn budget_inner(
    src: BlobFrameRef<'_>,
    child_offs: impl Iterator<Item = u32>,
) -> Result<Option<BudgetNode>> {
    let mut survivors = 0usize;
    let mut routing_bytes = 0u32;
    let mut leaf_bytes = 0u32;
    let mut leaf_count = 0u32;
    for coff in child_offs {
        if let Some(c) = routing_budget(src, coff)? {
            survivors += 1;
            routing_bytes += c.routing_bytes;
            leaf_bytes += c.leaf_bytes;
            leaf_count += c.leaf_count;
        }
    }
    Ok(packed_inner_size(survivors).map(|self_size| BudgetNode {
        routing_bytes: routing_bytes + self_size,
        leaf_bytes,
        leaf_count,
    }))
}

/// Minimum surviving-leaf bytes for a blob to be worth routing.
///
/// The routed layout inserts an up-to-`PAGE_4K` page-alignment gap
/// between the routing arena and the leaf region. For a blob with only
/// a handful of leaves that gap dominates `space_used` — it would make
/// a compaction that genuinely reclaimed bytes *look* like growth, and
/// such tiny blobs barely benefit from page-granular cold reads anyway.
/// Below this threshold a blob keeps the legacy whole-frame layout.
/// (The cold-read win for larger blobs is independent of this — it
/// always replaces a 512 KB frame read with a routing region + one leaf
/// page.)
const ROUTE_MIN_LEAF_BYTES: u32 = 2 * PAGE_4K;

/// Turn a pass-0 budget into the page-aligned `leaf_region_start`, or
/// `None` to keep the blob in the legacy whole-frame layout.
///
/// `None` for degenerate trees (no internal nodes — empty or a
/// bare-leaf root), for blobs below [`ROUTE_MIN_LEAF_BYTES`], and for
/// blobs that would not fit the page-aligned two-arena layout (the
/// up-to-4 KB alignment gap can push a near-full blob over; falling
/// back to legacy is always correct). `routing_len == 0` is then the
/// sole "not routed" sentinel.
fn routing_geometry(budget: Option<BudgetNode>) -> Option<u32> {
    let b = budget?;
    if b.routing_bytes == 0 {
        return None; // empty / bare-leaf root → legacy
    }
    if b.leaf_bytes < ROUTE_MIN_LEAF_BYTES {
        return None; // too small to amortize the page-alignment gap
    }
    // The fresh frame's init `EmptyRoot` sentinel sits at DATA_AREA_START,
    // ahead of the cloned internals, so the routing arena actually ends at
    // DATA_AREA_START + sentinel + routing_bytes. Account for it here, or a
    // routing_bytes that lands DATA_AREA_START + routing_bytes exactly on a
    // page boundary would put `space_used` one sentinel past
    // leaf_region_start and trip the overrun guard.
    //
    // The per-blob bloom (stage 6) lives at the tail of the routing
    // region, between the internal nodes and the page-aligned leaf
    // region, so reserve its bytes here too. It is read for free with the
    // routing region. `bloom_reserve_bytes` is recomputed identically in
    // `compact_blob` from the same `leaf_count`, so placement and
    // reservation stay in lockstep.
    let bloom_len = bloom_reserve_bytes(b.leaf_count);
    let routing_end =
        DATA_AREA_START + size_of_node(NodeType::EmptyRoot) + b.routing_bytes + bloom_len;
    let lrs = page_align_up(routing_end);
    let fits = u64::from(lrs) + u64::from(b.leaf_bytes) + u64::from(SPILLOVER_RESERVATION)
        <= u64::from(PAGE_SIZE);
    fits.then_some(lrs)
}

/// Bytes to reserve (and later fill) for the per-blob bloom over
/// `leaf_count` live leaves. A blob with no leaves gets no bloom. Keep
/// in lockstep between [`routing_geometry`] (reservation) and
/// [`compact_blob`] (placement + fill).
#[inline]
fn bloom_reserve_bytes(leaf_count: u32) -> u32 {
    if leaf_count == 0 {
        return 0;
    }
    bloom_byte_len(leaf_count as usize, BLOOM_BITS_PER_KEY) as u32
}

/// Build the per-blob bloom over the freshly-cloned leaves and write it
/// at the routing-region tail (`[internal_end, leaf_region_start)`),
/// returning `(bloom_off, bloom_len)` to stamp into the header.
///
/// Hashes each leaf's **stored** key bytes (`[16B hdr][key]…`, key incl.
/// the ART terminator) — the exact bytes the cold read reconstructs from
/// its `SearchKey` via `write_to_slice`, so a present key can never be a
/// bloom false negative. Returns `None` (routed-without-bloom, a fine
/// fallback) when there are no leaves or the reserved span can't hold the
/// filter — never fails the compaction.
fn build_routing_bloom(
    frame: &mut BlobFrame<'_>,
    internal_end: u32,
    leaf_region_start: u32,
    leaf_cursor: u32,
    leaf_count: u32,
) -> Option<(u32, u32)> {
    let bloom_len = bloom_reserve_bytes(leaf_count);
    if bloom_len == 0 {
        return None;
    }
    let bloom_off = internal_end;
    // Must fit between the internal nodes and the page-aligned leaves.
    if bloom_off.checked_add(bloom_len)? > leaf_region_start {
        return None;
    }
    // Walk the packed leaf arena, hashing each leaf's stored key.
    let bytes = {
        let view = frame.as_ref();
        let mut builder = BloomBuilder::new(leaf_count as usize, BLOOM_BITS_PER_KEY);
        let hdr = std::mem::size_of::<Leaf>();
        let mut off = leaf_region_start;
        while off < leaf_cursor {
            let body = view.body_at_offset(off)?;
            let leaf = *cast::<Leaf>(&body[..hdr]);
            let key_end = hdr + leaf.key_len as usize;
            if key_end > body.len() {
                return None; // malformed → routed-without-bloom
            }
            builder.add(&body[hdr..key_end]);
            off = off.checked_add(body.len() as u32)?;
        }
        builder.into_bytes()
    };
    debug_assert_eq!(bytes.len() as u32, bloom_len);
    let dst = frame.bytes_at_mut(bloom_off, bloom_len)?;
    dst.copy_from_slice(&bytes);
    Some((bloom_off, bloom_len))
}

// ---------- clone primitives ----------

/// Recursively clone the subtree at `src_slot` into `dst`.
///
/// When `filter_tombstones` is false the result is always `Some`
/// — the entire source subtree is copied byte-for-byte. When true,
/// tombstoned leaves are dropped, prefix wrappers over dead
/// children collapse upward, and inner-node arms whose live-child
/// count slips into a smaller `NodeType`'s range re-allocate as
/// the smaller variant. A `None` return means the subtree had no
/// live leaves — caller decides what to substitute (typically
/// `EmptyRoot` at the root, or just "drop this branch" further
/// down).
/// `routed` + `leaf_cursor` drive the routing-aware (two-arena) build
/// used by `compact_blob`: when `routed`, leaves are placed at
/// `leaf_cursor` (the page-aligned leaf arena) instead of bumping
/// `space_used`, while internal nodes keep bumping `space_used` (the
/// routing arena). Legacy callers (spillover/merge) pass `routed =
/// false` + a dummy cursor and behave byte-identically to before. Only
/// `clone_leaf` consults the cursor; the inner-node arms just forward
/// it down the recursion.
fn clone_subtree(
    src: &BlobFrame<'_>,
    dst: &mut BlobFrame<'_>,
    src_off: u32,
    filter_tombstones: bool,
    routed: bool,
    leaf_cursor: &mut u32,
) -> Result<Option<u32>> {
    let ntype = src
        .ntype_at(src_off)
        .ok_or(Error::node_corrupt("clone_subtree: undecodable src ntype"))?;
    let body = src.body_at_offset(src_off).ok_or(Error::node_corrupt(
        "clone_subtree: src body resolution failed",
    ))?;

    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "clone_subtree: NodeType::Invalid in source",
        )),
        NodeType::EmptyRoot => {
            let out = dst.alloc_node(NodeType::EmptyRoot)?;
            let off = dst
                .offset_of_slot(out.slot)
                .ok_or(Error::node_corrupt("clone_subtree: EmptyRoot offset"))?;
            // Stamp the self-describing node_type byte so the clone is
            // offset-resolvable like the init sentinel.
            if let Some(b) = dst.bytes_at_mut(off, 8) {
                b[1] = NodeType::EmptyRoot.as_u8();
            }
            Ok(Some(off))
        }
        NodeType::Leaf => clone_leaf(body, dst, filter_tombstones, routed, leaf_cursor),
        NodeType::Prefix => clone_prefix(src, body, dst, filter_tombstones, routed, leaf_cursor),
        NodeType::Node4 => clone_node4(src, body, dst, filter_tombstones, routed, leaf_cursor),
        NodeType::Node16 => clone_node16(src, body, dst, filter_tombstones, routed, leaf_cursor),
        NodeType::Node48 => clone_node48(src, body, dst, filter_tombstones, routed, leaf_cursor),
        NodeType::Node256 => clone_node256(src, body, dst, filter_tombstones, routed, leaf_cursor),
        NodeType::Blob => clone_blob_node(body, dst),
    }
}

/// Clone a leaf verbatim. A leaf is one contiguous, self-describing
/// node (`[16B header][key][value]`), so the clone bump-allocates a
/// same-size leaf in the destination and copies the whole body across
/// — no key_offset to repoint. The tombstone byte travels with the
/// body in preserve-mode; filter-mode drops tombstoned survivors.
fn clone_leaf(
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
    routed: bool,
    leaf_cursor: &mut u32,
) -> Result<Option<u32>> {
    // `src_body` is the full leaf body (sized by `body_at_offset` from
    // the header's key_len/value_len). Decode only the 16-byte header.
    let src_leaf = *cast::<Leaf>(&src_body[..std::mem::size_of::<Leaf>()]);
    if filter_tombstones && src_leaf.tombstone != 0 {
        return Ok(None);
    }
    let total = src_body.len() as u32;
    debug_assert_eq!(
        total,
        crate::layout::leaf_body_size(u32::from(src_leaf.key_len), u32::from(src_leaf.value_len),)
    );
    // Routed build: place the leaf at the caller's page-aligned leaf
    // cursor (its final offset, so the parent's `encode_child_off`
    // back-patch is unchanged) without touching `space_used`. Legacy:
    // bump `space_used` exactly as before.
    let dst_off = if routed {
        let off = *leaf_cursor;
        dst.alloc_leaf_at(off, total)?;
        *leaf_cursor += total;
        off
    } else {
        let out = dst.alloc_leaf(total)?;
        dst.offset_of_slot(out.slot)
            .ok_or(Error::node_corrupt("clone_leaf: dst slot offset"))?
    };
    {
        // The freshly-allocated dst leaf body's header is still zero,
        // so address it by byte offset and copy the source body
        // verbatim (`[header][key][value]`, incl. the `node_type @ +1`
        // byte) — the dst becomes self-describing once the bytes land.
        let dst_body = dst
            .bytes_at_mut(dst_off, total)
            .ok_or(Error::node_corrupt("clone_leaf: dst body out of range"))?;
        debug_assert_eq!(dst_body.len(), src_body.len());
        dst_body.copy_from_slice(src_body);
    }
    Ok(Some(dst_off))
}

fn clone_prefix(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
    routed: bool,
    leaf_cursor: &mut u32,
) -> Result<Option<u32>> {
    let p = *cast::<Prefix>(src_body);
    let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
    let Some(new_child_off) = clone_subtree(
        src,
        dst,
        child_offset(p.child as u16),
        filter_tombstones,
        routed,
        leaf_cursor,
    )?
    else {
        return Ok(None);
    };
    let out = dst.alloc_node(NodeType::Prefix)?;
    let off = dst
        .offset_of_slot(out.slot)
        .ok_or(Error::node_corrupt("clone_prefix: dst offset"))?;
    let new_p = Prefix::new(&p.bytes[..plen], u32::from(encode_child_off(new_child_off)));
    write_struct_at(dst, off, &new_p)?;
    Ok(Some(off))
}

fn clone_node4(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
    routed: bool,
    leaf_cursor: &mut u32,
) -> Result<Option<u32>> {
    let src_n = *cast::<Node4>(src_body);
    let count = (src_n.count as usize).min(4);
    if filter_tombstones {
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(count);
        for i in 0..count {
            if let Some(new_child) = clone_subtree(
                src,
                dst,
                child_offset(src_n.children[i]),
                true,
                routed,
                leaf_cursor,
            )? {
                survivors.push((src_n.keys[i], new_child));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 4];
        for (i, child) in new_children.iter_mut().enumerate().take(count) {
            let cloned = clone_subtree(
                src,
                dst,
                child_offset(src_n.children[i]),
                false,
                routed,
                leaf_cursor,
            )?
            .expect("preserve mode never returns None");
            *child = encode_child_off(cloned);
        }
        let out = dst.alloc_node(NodeType::Node4)?;
        let off = dst
            .offset_of_slot(out.slot)
            .ok_or(Error::node_corrupt("clone_node4: dst offset"))?;
        let mut new_n = Node4::empty();
        new_n.count = src_n.count;
        new_n.keys = src_n.keys;
        new_n.children = new_children;
        write_struct_at(dst, off, &new_n)?;
        Ok(Some(off))
    }
}

fn clone_node16(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
    routed: bool,
    leaf_cursor: &mut u32,
) -> Result<Option<u32>> {
    let src_n = *cast::<Node16>(src_body);
    let count = (src_n.count as usize).min(16);
    if filter_tombstones {
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(count);
        for i in 0..count {
            if let Some(new_child) = clone_subtree(
                src,
                dst,
                child_offset(src_n.children[i]),
                true,
                routed,
                leaf_cursor,
            )? {
                survivors.push((src_n.keys[i], new_child));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 16];
        for (i, child) in new_children.iter_mut().enumerate().take(count) {
            let cloned = clone_subtree(
                src,
                dst,
                child_offset(src_n.children[i]),
                false,
                routed,
                leaf_cursor,
            )?
            .expect("preserve mode never returns None");
            *child = encode_child_off(cloned);
        }
        let out = dst.alloc_node(NodeType::Node16)?;
        let off = dst
            .offset_of_slot(out.slot)
            .ok_or(Error::node_corrupt("clone_node16: dst offset"))?;
        let mut new_n = Node16::empty();
        new_n.count = src_n.count;
        new_n.keys = src_n.keys;
        new_n.children = new_children;
        write_struct_at(dst, off, &new_n)?;
        Ok(Some(off))
    }
}

fn clone_node48(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
    routed: bool,
    leaf_cursor: &mut u32,
) -> Result<Option<u32>> {
    let src_n = *cast::<Node48>(src_body);
    if filter_tombstones {
        // Iterate bytes 0..256 in order — naturally yields survivors
        // sorted by byte, which `pack_inner_node` requires.
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(48);
        for b in 0..256usize {
            let idx = src_n.index[b];
            if idx == 0 {
                continue;
            }
            let ci = idx as usize - 1;
            if ci >= 48 {
                return Err(Error::node_corrupt("clone_node48: index out of range"));
            }
            if let Some(new_child) = clone_subtree(
                src,
                dst,
                child_offset(src_n.children[ci]),
                true,
                routed,
                leaf_cursor,
            )? {
                survivors.push((b as u8, new_child));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 48];
        for (i, child) in new_children.iter_mut().enumerate() {
            if src_n.children[i] != 0 {
                let cloned = clone_subtree(
                    src,
                    dst,
                    child_offset(src_n.children[i]),
                    false,
                    routed,
                    leaf_cursor,
                )?
                .expect("preserve mode never returns None");
                *child = encode_child_off(cloned);
            }
        }
        let out = dst.alloc_node(NodeType::Node48)?;
        let off = dst
            .offset_of_slot(out.slot)
            .ok_or(Error::node_corrupt("clone_node48: dst offset"))?;
        let mut new_n = Node48::empty();
        new_n.count = src_n.count;
        new_n.index = src_n.index;
        new_n.children = new_children;
        write_struct_at(dst, off, &new_n)?;
        Ok(Some(off))
    }
}

fn clone_node256(
    src: &BlobFrame<'_>,
    src_body: &[u8],
    dst: &mut BlobFrame<'_>,
    filter_tombstones: bool,
    routed: bool,
    leaf_cursor: &mut u32,
) -> Result<Option<u32>> {
    let src_n = *cast::<Node256>(src_body);
    if filter_tombstones {
        let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(64);
        for (b, &child_enc) in src_n.children.iter().enumerate() {
            if child_enc == 0 {
                continue;
            }
            if let Some(new_child) =
                clone_subtree(src, dst, child_offset(child_enc), true, routed, leaf_cursor)?
            {
                survivors.push((b as u8, new_child));
            }
        }
        pack_inner_node(dst, &survivors)
    } else {
        let mut new_children = [0u16; 256];
        for (i, child) in new_children.iter_mut().enumerate() {
            if src_n.children[i] != 0 {
                let cloned = clone_subtree(
                    src,
                    dst,
                    child_offset(src_n.children[i]),
                    false,
                    routed,
                    leaf_cursor,
                )?
                .expect("preserve mode never returns None");
                *child = encode_child_off(cloned);
            }
        }
        let out = dst.alloc_node(NodeType::Node256)?;
        let off = dst
            .offset_of_slot(out.slot)
            .ok_or(Error::node_corrupt("clone_node256: dst offset"))?;
        let mut new_n = Node256::empty();
        new_n.count = src_n.count;
        new_n.children = new_children;
        write_struct_at(dst, off, &new_n)?;
        Ok(Some(off))
    }
}

fn clone_blob_node(src_body: &[u8], dst: &mut BlobFrame<'_>) -> Result<Option<u32>> {
    let src_b = *cast::<BlobNode>(src_body);
    let plen = (src_b.prefix_len as usize).min(BLOB_MAX_INLINE);
    let new_b = BlobNode::new(&src_b.bytes[..plen], src_b.child_blob_guid);
    let out = dst.alloc_node(NodeType::Blob)?;
    let off = dst
        .offset_of_slot(out.slot)
        .ok_or(Error::node_corrupt("clone_blob_node: dst offset"))?;
    write_struct_at(dst, off, &new_b)?;
    Ok(Some(off))
}

/// Pack `survivors` into the smallest inner-node variant that fits.
///
/// Used during filter-mode cloning to collapse inner nodes whose
/// live-child count has shrunk into a smaller `NodeType`'s range:
///
/// - 0 children → `None` (drop the branch).
/// - 1 child → `Prefix([byte])` wrapping the child slot; this
///   preserves the descent depth invariant (the parent expected
///   one byte of routing here, and `Prefix` consumes it).
/// - 2–4 → `Node4`; 5–16 → `Node16`; 17–48 → `Node48`; 49+ → `Node256`.
///
/// `survivors` must be byte-sorted ascending — `Node4` / `Node16`
/// store keys in sorted order and their lookup paths break out
/// early on `keys[i] > byte`, so out-of-order entries would corrupt
/// future descents.
fn pack_inner_node(dst: &mut BlobFrame<'_>, survivors: &[(u8, u32)]) -> Result<Option<u32>> {
    debug_assert!(
        survivors.windows(2).all(|w| w[0].0 < w[1].0),
        "pack_inner_node: survivors must be byte-sorted ascending"
    );
    // `survivors` carry child *byte offsets* in the destination blob;
    // the helper encodes them into the `u16` child fields.
    let alloc = |dst: &mut BlobFrame<'_>, nt: NodeType| -> Result<(u16, u32)> {
        let out = dst.alloc_node(nt)?;
        let off = dst
            .offset_of_slot(out.slot)
            .ok_or(Error::node_corrupt("pack_inner_node: dst offset"))?;
        Ok((out.slot, off))
    };
    match survivors.len() {
        0 => Ok(None),
        1 => {
            let (byte, child_off) = survivors[0];
            let off = write_prefix_chain(dst, &[byte], child_off)?;
            Ok(Some(off))
        }
        2..=4 => {
            let (_, off) = alloc(dst, NodeType::Node4)?;
            let mut n = Node4::empty();
            n.count = survivors.len() as u8;
            for (i, &(b, c)) in survivors.iter().enumerate() {
                n.keys[i] = b;
                n.children[i] = encode_child_off(c);
            }
            write_struct_at(dst, off, &n)?;
            Ok(Some(off))
        }
        5..=16 => {
            let (_, off) = alloc(dst, NodeType::Node16)?;
            let mut n = Node16::empty();
            n.count = survivors.len() as u8;
            for (i, &(b, c)) in survivors.iter().enumerate() {
                n.keys[i] = b;
                n.children[i] = encode_child_off(c);
            }
            write_struct_at(dst, off, &n)?;
            Ok(Some(off))
        }
        17..=48 => {
            let (_, off) = alloc(dst, NodeType::Node48)?;
            let mut n = Node48::empty();
            n.count = survivors.len() as u8;
            for (ci, &(b, c)) in survivors.iter().enumerate() {
                n.children[ci] = encode_child_off(c);
                n.index[b as usize] = (ci as u8) + 1;
            }
            write_struct_at(dst, off, &n)?;
            Ok(Some(off))
        }
        _ => {
            let (_, off) = alloc(dst, NodeType::Node256)?;
            let mut n = Node256::empty();
            // `count: u8` wraps to 0 at 256 children; tolerate that
            // — the lookup path only consults `children[byte]` so
            // the count field is informational.
            n.count = survivors.len() as u8;
            for &(b, c) in survivors {
                n.children[b as usize] = encode_child_off(c);
            }
            write_struct_at(dst, off, &n)?;
            Ok(Some(off))
        }
    }
}

#[cfg(test)]
mod budget_tests {
    use super::{pack_inner_node, packed_inner_size, routing_geometry, BudgetNode};
    use crate::layout::{size_of_node, NodeType, DATA_AREA_START};
    use crate::store::blob_store::AlignedBlobBuf;
    use crate::store::{BlobFrame, PAGE_4K};

    /// `leaf_region_start` must leave room for the fresh frame's init
    /// `EmptyRoot` sentinel that precedes the cloned internals — even
    /// when `DATA_AREA_START + routing_bytes` lands exactly on a 4 KB
    /// boundary. Otherwise the routed clone's `space_used`
    /// (= DATA_AREA_START + sentinel + routing_bytes) overruns
    /// `leaf_region_start` by the sentinel and trips the overrun guard.
    /// Regression for a concurrent-compaction panic.
    #[test]
    fn routing_geometry_reserves_the_init_sentinel_on_a_page_boundary() {
        let sentinel = size_of_node(NodeType::EmptyRoot);
        // DATA_AREA_START is itself a 4 KB multiple, so any routing_bytes
        // that is a page multiple lands the arena start on a boundary —
        // the case that used to overrun.
        for routing_bytes in [PAGE_4K, 2 * PAGE_4K, 3 * PAGE_4K] {
            let lrs = routing_geometry(Some(BudgetNode {
                routing_bytes,
                leaf_bytes: 16 * PAGE_4K,
                // leaf_count 0 ⇒ no bloom reservation, so this isolates
                // the init-sentinel page-boundary case the test targets.
                leaf_count: 0,
            }))
            .expect("substantial routed blob");
            let space_used_after_clone = DATA_AREA_START + sentinel + routing_bytes;
            assert!(
                space_used_after_clone <= lrs,
                "routing_bytes={routing_bytes}: space_used {space_used_after_clone:#x} \
                 overruns leaf_region_start {lrs:#x}",
            );
            assert_eq!(lrs % PAGE_4K, 0, "leaf_region_start must be page-aligned");
        }
    }

    /// The routing budget is exact only if `packed_inner_size` predicts
    /// the SAME byte size `pack_inner_node` actually emits for every
    /// surviving-child count. Bind them directly at each tier boundary
    /// (including the 1-survivor → `Prefix` inflation, where the emitted
    /// node is *larger* than the source). Drift here would make pass-0
    /// under-count and trip `compact_blob`'s overrun guard.
    #[test]
    fn packed_inner_size_matches_pack_inner_node() {
        let mut ab = AlignedBlobBuf::zeroed();
        for &n in &[1usize, 2, 3, 4, 5, 16, 17, 48, 49, 100, 256] {
            // Re-init zeroes the frame, so each tier starts clean.
            let mut frame = BlobFrame::init(ab.as_mut_slice(), [0u8; 16]).unwrap();
            // `n` survivors with valid in-data child offsets and
            // distinct, ascending bytes (pack requires byte-sorted).
            let mut survivors: Vec<(u8, u32)> = Vec::with_capacity(n);
            for i in 0..n {
                let out = frame.alloc_leaf(16).unwrap();
                let off = frame.offset_of_slot(out.slot).unwrap();
                survivors.push((i as u8, off));
            }
            let off = pack_inner_node(&mut frame, &survivors)
                .unwrap()
                .expect("non-empty survivors pack to Some");
            let nt = frame.ntype_at(off).unwrap();
            assert_eq!(
                Some(size_of_node(nt)),
                packed_inner_size(n),
                "tier mismatch at n={n}: pack emitted {nt:?}",
            );
        }
        assert_eq!(packed_inner_size(0), None);
    }
}
