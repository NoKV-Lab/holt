# Per-node-latch milestone — Phase 2.B design spec

Status: pending implementation (queued for next v0.3 work session).
Phases 1 + 2.A have landed (commits `3222330` + `7d14697`); this doc
specifies the protocol Phase 2.B will execute, so the next session
opens directly to executable code rather than re-deriving the
contract.

## Goal

Make the walker hold **at most one blob's exclusive latch at any
time** during a cross-blob mutation. Today the writer holds parent +
child latches simultaneously while descending into a child blob; this
serialises all writers that share any ancestor blob even if their
mutations target disjoint subtrees.

After Phase 2.B + Phase 3 (which together complete the lock-coupling
protocol) the only writer-vs-writer serialisation is the `wal.lock`
that brackets the WAL append (#8 will then attack that separately).

## Non-goals for Phase 2.B

* Removing the `wal.lock` — that's #8.
* Per-slot reader optimistic-restart granularity — separate
  follow-up; current blob-level `guard.validate()` stays for reads.
* Touching the spillover / merge / compact lock ordering — Phase 2.B
  scopes the change to the **normal** cross-blob insert / erase
  path; structural ops keep their current "take parent + child
  latches simultaneously" behavior. Phase 4 attacks structural ops.

## Protocol — narrative

For an insert (erase mirrors):

1. Caller (`insert_multi`) takes root blob's exclusive guard.
2. Walker (`insert_at`) descends in root frame. If it reaches a
   leaf / inner-node update / prefix update without crossing into a
   different blob, it returns `InsertHop::Done(InsertReturn)` and
   `insert_multi` is finished.
3. If walker hits a `NodeType::Blob` slot, it captures the
   `BlobNode` body + the parent's per-slot version (via
   `BlobFrame::slot_version`) and returns
   `InsertHop::CrossBlobHop { bn_slot, bn_snapshot, parent_bn_version,
   child_guid, child_entry_slot, child_depth }`. **The walker does
   not pin or latch the child blob.**
4. `insert_multi` receives the hop, **drops the root guard**, then
   pins + write-guards the child blob and recurses (i.e. calls
   `insert_at` on the child frame at `child_entry_slot`).
5. The child descent may itself produce another `CrossBlobHop` — in
   that case `insert_multi` pushes the parent context onto its hop
   stack and descends one more level. The stack records
   `(parent_pin, bn_slot, bn_snapshot, parent_bn_version)` per hop.
6. When the deepest descent returns `Done(InsertReturn)`,
   `insert_multi` unwinds the stack:
   a. Update the current blob's `header.root_slot` if
      `slot_after` differs.
   b. `mark_dirty(current_guid, seq)`.
   c. Drop current guard.
   d. Pop the top of `hop_stack` → re-acquire the parent guard.
   e. `Acquire`-load `parent_pin.slot_version(bn_slot)`. If it
      doesn't equal the captured `parent_bn_version` → **restart
      from root** (clear stack, `continue 'restart`).
   f. Validate-after-lock: re-load slot_version with the latch
      held (defensive against any pre-lock drift). Same restart on
      mismatch.
   g. If `child_slot_after != bn_snapshot.child_entry_ptr`, write
      back the updated `BlobNode` via `write_struct_to_slot` (which
      bumps parent's slot version on the same slot, signalling the
      next observer correctly).
   h. Drop parent guard, set current = parent, loop.
7. When the hop stack is empty, return the final
   `InsertReturn { slot_after: <root-level bn_slot or final
   in-frame slot>, previous }`.

## Type changes — `src/engine/walker/types.rs`

```rust
/// Phase 2.B: outcome a single-blob descent of `insert_at` produces.
/// Either it completed in-frame (`Done`) or it needs the caller to
/// hop into a child blob (`CrossBlobHop`).
#[derive(Debug)]
pub(super) enum InsertHop {
    Done(InsertReturn),
    CrossBlobHop(CrossBlobInsertHop),
}

#[derive(Debug)]
pub(super) struct CrossBlobInsertHop {
    /// Slot in the parent frame holding the `BlobNode` we hit.
    pub(super) bn_slot: u16,
    /// Copy of the parent's `BlobNode` body at descent time.
    /// Carried so `insert_multi` can patch `child_entry_ptr` on
    /// the way back without re-reading the parent frame.
    pub(super) bn_snapshot: BlobNode,
    /// `Acquire`-loaded value of `parent_pin.slot_version(bn_slot)`
    /// at descent time. Re-loaded on the way back; mismatch =
    /// "writer raced into our parent, restart from root."
    pub(super) parent_bn_version: u64,
    /// Where the child descent should resume.
    pub(super) child_guid: BlobGuid,
    pub(super) child_entry_slot: u16,
    pub(super) child_depth: usize,
}

// Same shape for erase:
pub(super) enum EraseHop {
    Done(EraseReturn),
    CrossBlobHop(CrossBlobEraseHop),
}

pub(super) struct CrossBlobEraseHop {
    pub(super) bn_slot: u16,
    pub(super) bn_snapshot: BlobNode,
    pub(super) parent_bn_version: u64,
    pub(super) child_guid: BlobGuid,
    pub(super) child_entry_slot: u16,
    pub(super) child_depth: usize,
}
```

## Walker changes — single-blob arms

`insert_at` signature changes from `Result<InsertReturn>` to
`Result<InsertHop>`. All in-frame arms wrap their result:

```rust
NodeType::EmptyRoot => insert_into_empty_root(...).map(InsertHop::Done),
NodeType::Leaf      => insert_into_leaf(...).map(InsertHop::Done),
NodeType::Node4 | Node16 | Node48 | Node256 =>
                       insert_into_inner(bm, frame, slot, ntype, ...),
NodeType::Prefix    => insert_into_prefix(bm, frame, slot, ...),
NodeType::Blob      => capture_blob_node_for_hop(frame, slot, ...),
```

`insert_into_inner` / `insert_into_prefix` recurse via `insert_at`.
They receive `InsertHop` and forward `CrossBlobHop` unchanged; only
the `Done` branch runs the post-recursion in-frame work (e.g.
`inner_update_child` when the recursive descent's `slot_after`
differs).

**Why post-recursion work is safe to skip on `CrossBlobHop`:** the
hop's parent-side endpoint is the `BlobNode` slot in the inner
node's `children[]`. That slot index doesn't change as a result of
cross-blob work (the BN slot itself stays put). So
`inner_update_child` would be a no-op anyway. Forwarding the hop
unchanged is correct.

`capture_blob_node_for_hop` is a tiny new helper that:

1. Reads `BlobNode` from `frame[slot]`.
2. Validates inline prefix matches `key[depth..depth + plen]` (the
   current `insert_at_blob_node` already does this).
3. Captures `parent_bn_version = frame.slot_version(slot)`.
4. Returns `Ok(InsertHop::CrossBlobHop(...))`.

The current `insert_at_blob_node` body — the pin + write-guard + the
spillover-retry loop + the `header.root_slot` patch + the write-back
to parent's `BlobNode` — **moves to `insert_multi`**. Nothing about
that work needs the walker's recursion context; it's all driven by
the hop's pre-captured info.

## Walker changes — `insert_multi` state machine

```rust
pub fn insert_multi(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    key: &[u8], value: &[u8],
    seq: u64, wants_prev: bool,
) -> Result<InsertOutcome> {
    'restart: loop {
        // Initial descent target: root blob's root_slot.
        let mut current_pin = Arc::clone(root_pin);
        let mut current_guid = ROOT_BLOB_GUID; // tracked alongside pin
        let mut current_slot = read_root_slot(&current_pin);
        let mut current_depth = 0usize;

        // Stack of pending parent contexts (innermost-first on
        // pop). On each hop we push the parent we just descended
        // *from*.
        let mut hop_stack: Vec<HopContext> = Vec::new();

        // Phase A: descend, doing the in-frame work in the deepest
        // blob we reach. Each iteration takes the guard for
        // `current_pin`, runs the walker with a spillover-retry
        // loop, and either breaks with the `Done` result (leaf
        // installed) or pushes the parent context + advances to
        // the next child blob.
        let final_inframe = 'descend: loop {
            let hop = run_blob_attempts(
                bm, &current_pin, current_slot, current_depth,
                key, value, seq, wants_prev,
            )?;
            match hop {
                InsertHop::Done(ret) => break 'descend ret,
                InsertHop::CrossBlobHop(h) => {
                    hop_stack.push(HopContext {
                        parent_pin: Arc::clone(&current_pin),
                        parent_guid: current_guid,
                        bn_slot: h.bn_slot,
                        bn_snapshot: h.bn_snapshot,
                        parent_bn_version: h.parent_bn_version,
                    });
                    current_pin = bm.pin(h.child_guid)?;
                    current_guid = h.child_guid;
                    current_slot = h.child_entry_slot;
                    current_depth = h.child_depth;
                }
            }
        };

        // Phase B: patch the deepest blob's `header.root_slot`
        // + mark it dirty.
        {
            let mut guard = current_pin.write();
            guard.frame().header_mut().root_slot = final_inframe.slot_after;
        }
        bm.mark_dirty(current_guid, seq);

        // Phase C: unwind, validating each parent's slot version.
        let mut child_slot_after = final_inframe.slot_after;
        let mut prev_value = final_inframe.previous;
        while let Some(ctx) = hop_stack.pop() {
            // Optimistic recheck before re-locking — cheap path
            // out of the lock-acquire.
            if ctx.parent_pin.slot_version(ctx.bn_slot) != ctx.parent_bn_version {
                bm.note_optimistic_restart();
                continue 'restart;
            }
            let mut parent_guard = ctx.parent_pin.write();
            // Defensive re-validate under the lock — closes the
            // race window between optimistic recheck and the
            // exclusive acquire.
            if ctx.parent_pin.slot_version(ctx.bn_slot) != ctx.parent_bn_version {
                drop(parent_guard);
                bm.note_optimistic_restart();
                continue 'restart;
            }
            // Write back the BlobNode if child_entry_ptr changed.
            // The write_struct_to_slot bumps the parent's slot
            // version so subsequent observers see the change.
            if u32::from(child_slot_after) != ctx.bn_snapshot.child_entry_ptr {
                let mut new_bn = ctx.bn_snapshot;
                new_bn.child_entry_ptr = u32::from(child_slot_after);
                write_struct_to_slot(&mut parent_guard.frame(), ctx.bn_slot, &new_bn)?;
            }
            drop(parent_guard);
            bm.mark_dirty(ctx.parent_guid, seq);
            // For the next unwind level, `child_slot_after` is the
            // BN slot in the parent (the slot the grandparent's
            // descent reached us at).
            child_slot_after = ctx.bn_slot;
        }

        // Root level — return the final result.
        return Ok(InsertOutcome {
            new_root_slot: child_slot_after,
            previous: prev_value,
        });
    }
}
```

`run_blob_attempts` is the per-blob spillover-retry loop (extracted
from the current `insert_at` top of body for root + the current
`insert_at_blob_node` loop for child — now a single shared helper):

```rust
fn run_blob_attempts(
    bm: &BufferManager,
    pin: &Arc<CachedBlob>,
    entry_slot: u16,
    depth: usize,
    key: &[u8], value: &[u8],
    seq: u64, wants_prev: bool,
) -> Result<InsertHop> {
    let mut last_err: Option<Error> = None;
    for _attempt in 0..MAX_SPILLOVER_ATTEMPTS {
        let mut guard = pin.write();
        let mut frame = guard.frame();
        match insert_at(Some(bm), &mut frame, entry_slot, key, value, depth, seq, wants_prev) {
            Ok(hop) => return Ok(hop),
            Err(Error::Alloc(crate::store::AllocError::OutOfSpace { .. })) => {
                spillover_blob(bm, &mut frame, seq)?;
                drop(frame); drop(guard);
                {
                    let mut guard2 = pin.write();
                    compact_blob(&mut guard2)?;
                }
                // After compact, re-pick entry_slot from
                // header.root_slot (slot indices were renumbered).
                let guard3 = pin.read();
                entry_slot = BlobFrameRef::wrap(guard3.as_slice()).header().root_slot;
            }
            Err(e) => { last_err = Some(e); break; }
        }
    }
    Err(last_err.unwrap_or(Error::NotYetImplemented(
        "insert spillover retry loop exhausted",
    )))
}
```

(`entry_slot` is `mut` because compact may renumber it.)

## Erase mirror

Identical shape, with `EraseHop` / `CrossBlobEraseHop`. The unwind
phase has three branches for the deepest blob's `EraseSignal`:

* `Unchanged`: skip header.root_slot update (no in-cache change to
  the blob image); skip mark_dirty unless `r.mutated` (existing
  contract from Phase 1.A.2). Walk up the stack with
  `child_slot_after = ctx.bn_slot` for each level — the unwind
  validates parent versions but doesn't patch the parent BN (no
  change to write back).
* `Replaced(new_entry)`: patch the deepest blob's
  `header.root_slot = new_entry`. Walk up validating; if parent
  BN's `child_entry_ptr` differs from `new_entry`, write back.
* `SubtreeGone`: the deepest blob is now empty. **Skip the deepest
  blob's header.root_slot patch**; instead, after the deepest blob
  is dropped from cache, walk up: the *innermost* parent must
  `parent_frame.free_node(bn_slot)` and propagate the new
  `EraseSignal::SubtreeGone` to *its* parent. This recursive
  collapse is currently inside `erase_at`'s in-frame arms;
  Phase 2.B's unwind has to mirror it across the hop boundary.

The cross-hop SubtreeGone case is the trickiest part of the spec
— allow extra time for that arm + dedicated test coverage.

## Concurrent-writer correctness

Once parent latch is dropped before child pin, another writer can
take the parent latch. Cases to consider:

1. **Other writer does an unrelated in-frame mutation in parent**
   (e.g. inserts a leaf under a different inner-node child). That
   writer doesn't touch `bn_slot` → parent's slot_version[bn_slot]
   stays the same → our re-validate passes → we proceed safely.

2. **Other writer does a structural op on parent's BN slot**
   (spillover that moves the BN, compact that renumbers slots,
   merge that folds child back into parent). All of these touch
   `bn_slot`'s version (or `bn_slot` no longer exists post-compact).
   Re-validate fails → we restart from root.

3. **Other writer descends into the same child blob.** They'd pin
   + write-guard the child. We hold child's write guard during our
   work → they block. After our work, they take child guard, see
   the new state, proceed. No correctness issue.

4. **`compact_blob` runs on the parent.** Compact takes parent's
   exclusive latch. Compact may *renumber* slot indices, so the
   `bn_slot` index we captured may now point to a different node
   entirely. Per-slot version bumps on every slot rewrite during
   compact ensure our re-validate fails. We restart from root.

5. **Concurrent erase that calls `parent_frame.free_node(bn_slot)`.**
   The `free_node` bumps the slot version (Phase 1 invariant). Our
   re-validate fails → restart.

6. **Race-free under stress?** Yes if (a) per-slot version is
   bumped on **every** mutation that touches the slot's tag or
   body, and (b) the version is `Acquire`-loaded on read and
   `Release`-stored on write. Phase 1's `bump_slot_version` is
   `Release`; Phase 2.A's `slot_version()` reader is `Acquire`.
   The validate-after-lock branch closes the small race window
   where another writer could have completed between our
   optimistic recheck and our exclusive acquire.

## W2D invariant preservation

The W2D barrier (WAL record durable before any data byte / manifest
mutation reaches backend) still goes through `wal.lock` held by
`Tree::put_inner` / `delete_inner`. `insert_multi` is called from
*inside* `wal.lock`'s critical section. All the `mark_dirty`
calls — including the new per-hop marks during unwind — happen
inside `wal.lock`, so the checkpoint round still sees a coherent
dirty set after the snapshot.

The checkpoint round itself doesn't change.

## Restart semantics

On parent slot version mismatch, the walker restarts from root.
But by this point in Phase 2.B, the writer has *already mutated*
the leaf-level blob and marked it dirty. Those bytes are still in
the cache + the dirty map even though the parent's `BlobNode`
pointer is now stale.

**Is this a leak?** Yes, but bounded:

* On restart, the walker re-descends from root, potentially
  landing on a different child blob (because the parent's BN moved).
  It writes the leaf there — second copy.
* The first copy is in a blob that's still reachable from cache
  (and the dirty map keeps it from being silently dropped) but
  unreachable from any tree-descent path.
* The next `compact()` over the parent's subtree will sweep up the
  orphan during reachability analysis.

For Phase 2.B we accept the bounded leak. Phase 5's stress tests
should verify (a) compact eventually reclaims orphans, and (b)
orphans don't accumulate fast enough to OOM under realistic
contention.

Alternative is "tentative write" / shadow paging in the leaf blob,
which is heavier than the bounded-leak tradeoff. Defer that to a
later optimisation if compact-doesn't-keep-up materialises in
practice.

## Test plan

1. **Existing tests pass** — all integration / property / failpoint
   tests should continue to pass. The cross-blob hop is exercised
   by `cross_blob_writes_replay_correctly_through_wal_without_checkpoint`
   among others.
2. **New: validate-mismatch restart** — a unit test that:
   - constructs a tree with a known parent BN
   - simulates a "writer descended, dropped parent guard, then
     someone bumped the BN slot version" scenario via direct
     `bump_slot_version` calls
   - confirms `insert_multi` restarts from root and produces the
     correct final state
3. **New: hop stack depth ≥ 2** — write a key that crosses two
   cross-blob hops; verify the unwind correctly validates each
   level.
4. **New: SubtreeGone propagation across hop boundary** — erase
   the last leaf in a deeply-nested child blob; verify all parents
   on the path correctly collapse their BN slots.
5. **Concurrent stress** — already-existing
   `tests/concurrent_writers_*` tests run with the new walker.
6. **Failpoint** — failpoint between Phase B (deepest mark_dirty)
   and Phase C (parent BN writeback) — verify recovery sees a
   consistent tree.

## Effort estimate

* Type definitions: ~30 lines, ~15 min.
* `insert_at` + in-frame arms refactor: ~150 lines net, ~1 hour.
* `insert_at_blob_node` → `capture_blob_node_for_hop` + move logic
  to `insert_multi`: ~200 lines moved, ~1.5 hours.
* `insert_multi` state machine: ~150 lines, ~1.5 hours.
* Erase mirror with SubtreeGone propagation: ~250 lines, ~2 hours.
* Tests: ~150 lines, ~1 hour.
* Debugging + fixing tests + clippy + fmt: ~2 hours.

**Total: ~9-10 hours of focused work.** Probably spread across two
fresh sessions to keep the diff reviewable.

## After Phase 2.B

Phase 3 closes the remaining edges (compact/merge lock ordering),
Phase 4 wires the structural-op coverage, Phase 5 stress-tests
everything. Then #8 (WAL group commit) becomes the next visible
perf lever.

## Open design gap — header.root_slot tracking across the hop

**Discovered during a follow-up implementation attempt; documented
here so it's solved before any Phase 2.B PR is opened.**

During Phase C unwind, the popped parent context's
`bn_snapshot.child_entry_ptr` must be compared against the **child
blob's CURRENT `header.root_slot`** to decide whether a writeback
is needed. For the deepest blob this is trivially the value we
just wrote in Phase B (`final_inframe.slot_after`). For
intermediate blobs (those that returned `CrossBlobHop` from
`run_blob_attempts`) the value is whatever each blob's
`header.root_slot` was at the moment its walker call returned —
typically unchanged from descent time, **but possibly bumped by an
in-blob `compact_blob` run inside the spillover-retry loop**.

The protocol as written in this doc captures only `bn_slot`,
`bn_snapshot`, `parent_bn_version`, `child_guid`,
`child_entry_slot`, `child_depth`. It does not record the
intermediate blob's `header.root_slot` post-descent. So at unwind
time we can't tell whether to write back the parent's
`BlobNode.child_entry_ptr` — we'd need to re-acquire a guard on the
intermediate blob (just to read its header), and even then a
concurrent writer running `compact_blob` could renumber the
intermediate's slot table between our read and our parent-side
writeback.

The naive fixes each fail for a different reason:

1. **Capture the intermediate's `header.root_slot` at push time
   into the hop context.** Stale by unwind time if a concurrent
   writer ran compact on the intermediate between our descent and
   our return — we'd write a stale value into the parent's BN.

2. **Re-read intermediate's `header.root_slot` during unwind under
   a brief read guard.** The race window between the read and the
   parent-side writeback is smaller but still exists.

3. **Hold the intermediate's exclusive guard all the way through
   the unwind.** Recreates Phase 2.A's "parent + child both held"
   pathology — defeats the whole point of lock-coupling.

4. **Treat the intermediate's `header.root_slot` as locked behind
   `bn_slot`'s version.** Doesn't work: `bn_slot`'s version
   tracks mutations to the parent's BN slot body, not to the
   child's header.

The genuine fix is one of:

- **(A) Version the child blob's header.root_slot.** Add a
  separate `header_root_slot_version: AtomicU64` to `CachedBlob`
  (or repurpose a slot-version slot 0). Bump on every
  `header.root_slot` mutation (compact / write-back). On unwind:
  capture-at-push-time + re-load + compare. Mismatch → restart
  from root. This is the LeanStore-equivalent of versioning the
  parent-pointer target.
- **(B) Lock-coupling that keeps the intermediate latched until
  parent's BN is patched.** Hand-over-hand: when we descend into
  a deeper child, take the deeper child's guard *before* dropping
  the intermediate's. This bounds the held-latch window to "two
  blobs during the hop boundary" rather than "all ancestors
  throughout the descent". Less concurrency but correct without
  versioning the header. The Phase 2.B doc's "release parent
  before child" then becomes "release grandparent before
  pinning grandchild" — the immediate parent stays held until
  the writeback completes.
- **(C) Avoid touching parent's BN.child_entry_ptr during normal
  ops.** Currently spillover/compact in the child can change
  child's root_slot. If we restructure so the child's compact
  is coordinated with parent's BN update (e.g. compact takes a
  "pending compact" reservation that the next parent-side write
  picks up), the cross-blob pointer stays stable except at
  explicitly synchronised moments. Heavier protocol change.

**Recommendation for next session:** start with (B)
hand-over-hand — preserves correctness without the protocol churn
of (A) or (C), still meaningfully more concurrent than Phase 2.A
(any writer NOT on the same grandparent-of-parent can proceed).
(A) is the long-term endgame but adds one more atomic per blob
write — defer until #8 makes the bench numbers worth optimising.

The InsertHop / CrossBlobInsertHop type definitions sketched
above are **still useful** but need the `current_blob_root_slot`
field added (option A or B both want it).

The design doc was published *before* this gap was discovered.
Next session's first task: pick (A) or (B) explicitly and edit
this section + the protocol sketch above accordingly.
