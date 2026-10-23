# DB view capture and COW debt

Scope: optimize DB::view capture and transient COW/reclamation cost;
keep the public read, atomic, acknowledgement, and persistence contracts unchanged.

## Implementation and invariants

1. Preserve the pre-change release executable and the existing workspace workload.
2. Batch snapshot registration under the coordinated tree freeze, avoid redundant
   root initialization and empty delta flushes, and avoid dirtying named-tree
   roots merely to persist the clock. DB checkpoints stage one catalog-root
   epoch summary, preserving monotonicity and exhausted-clock reopen behavior;
   DB open still checks all reachable frame epochs. Standalone trees retain
   their root epoch high-water update.
3. Track an orphan's closed visibility interval, from its creation to detachment.
   Queue it against the newest live epoch inside that interval. Later snapshots
   cannot prolong its lifetime. Retiring an epoch examines only its own waiters.
4. Discard bounded batches of unreachable, never-persisted COW images only when
   no pin, checkpoint claim, or deletion fence owns them. Preserve the replaced
   image's dirty WAL low-watermark in its successor. Persisted GUIDs still wait
   for a clean durable frontier or reachability GC; no inline disk deletion.
   Earlier checkpoint parents can depend on a different retired child: exclude
   pruning while any flushing claim remains, and share an exclusion with the
   complete dirty capture. At an idle capture, omit the retirement-set image
   atomically so dead parents cannot reference elided children. Structural
   promotion also carries the child's WAL debt into the rewritten parent.
   Wake the planner only on the reclaim queue's empty-to-nonempty transition.
5. GC-owned snapshot root pins retain epoch protection until the walk releases
   them. Escaped views, ranges and cursors retain their existing leases and scan
   fences. No on-disk format or durability policy changes.
6. Add targeted lifecycle, concurrent checkpoint, epoch exhaustion, duplicate
   scope, and reopen tests, run the existing regression suite, then interleave
   baseline/patched runs of the unchanged agent-workspace workflow. Report
   throughput, tail latency, peak RSS, COW debt, disk footprint and final drain.

## Deferred alternatives

A new persistent root descriptor, per-tree frame ownership tags, and a new
on-disk format are not required for this iteration. Removing scan fences or
substituting point reads would weaken semantics and is not an optimization here.

## Acceptance

Full ordered-domain fingerprints must match across builds and reopen. Atomic
multi-tree captures must remain consistent. Pinned/escaped readers and in-flight
checkpoint images must never be discarded. Existing crash/replay and W2D tests
remain mandatory. Performance claims use process-level balanced A/B repetitions,
not the earlier baseline alone.
