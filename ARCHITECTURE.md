# holt — architecture

This document describes the live design. For the milestone view
see [ROADMAP.md](ROADMAP.md); for what changed when see
[CHANGELOG.md](CHANGELOG.md) and `git log`.

## 1. Data layout — one 512 KB blob frame

Every `Tree` is backed by one or more 512 KB blob frames. Each blob is
self-describing and walkable in isolation:

```
+------------------------ 524288 bytes ----------------------+
| BlobHeader (4096 B)                                         |
|   - blob_guid, num_slots, root_slot, space_used, gap_space  |
|   - free_list_head[8]u16  ← per-NodeType free LIFO          |
+-------------------------------------------------------------+
| Slot table (40 KB = 10240 × u32, 1-based)                   |
|   Each entry is bit-packed:                                 |
|     bits 0..16  (17 bits) = byte_offset / 8                 |
|     bits 17..31 (15 bits) = NodeType (live) OR              |
|                              next free slot (freed)         |
+-------------------------------------------------------------+
| Data area (~468 KB, bump-allocated)                         |
|   Node bodies (8 to 520 bytes each, NodeType-tagged)        |
|   Leaf nodes inline their key + value after a 16 B header   |
+-------------------------------------------------------------+
```

Every field offset is pinned at compile time via `const _: () =
assert!(offset_of!(...) == ...)` blocks. Drift in `BlobHeader`
or any per-NodeType body fails the build, not a runtime test.

The tree grows through `BlobNode` crossings — when an insert can't
find room in the current blob, the walker materializes a subtree
into a fresh blob and installs a Blob-type node in the parent that
says "the walk continues in blob X." The child blob's own
`header.root_slot` is the authoritative entry point. This
composes recursively: each child blob is itself a full ART frame,
so the same walker code descends across blob boundaries without
special-casing the crossing.

## 2. NodeType variants

| ntype | Name      | Size       | Purpose                                        |
|------:|-----------|-----------:|------------------------------------------------|
|     1 | Leaf      |    16 B*   | `[16 B header: key_fp, tombstone, key_len, value_len, seq]` + inline key + value — one self-describing, variable-size node |
|     2 | Prefix    |   128 B    | Path-compressed segment (≤112 inline bytes)    |
|     3 | Blob      |   128 B    | Cross-blob crossing (target_guid + ≤104 inline prefix bytes) |
|     4 | Node4     |    16 B    | 1..4 children, linear scan                     |
|     5 | Node16    |    56 B    | 5..16 children, SSE2 / NEON byte search        |
|     6 | Node48    |   360 B    | 17..48 children, byte→slot index               |
|     7 | Node256   |   520 B    | 49..256 children, direct array                 |
|     8 | EmptyRoot |     8 B    | All-zero sentinel for an empty tree            |

\* `Leaf` is variable-size: the 16 B is the header only; the true body
is `16 + key_len + value_len`, recovered from the header. The four
internal nodes store child slot indices as `u16` (slots are 1-based
into a ≤10240 table), which halves their footprint versus a `u32`
child array — hence 16 / 56 / 360 / 520 rather than 24 / 88 / 456 / 1032.

The walker descends through `Prefix` and `Node{4,16,48,256}` based
on the next key byte; terminates at `Leaf`; crosses blobs at
`Blob`.

## 3. Walker mechanics

```text
walk(slot, key, depth) {
    loop {
        match nodeType(slot) {
            Leaf      -> compare full key, return value or NotFound
            EmptyRoot -> NotFound
            Prefix    -> match prefix vs key[depth..],
                         advance depth, descend
            Node*     -> use key[depth] to pick child, descend
            Blob      -> match inline prefix, pin target blob,
                         descend at child.header.root_slot
        }
    }
}
```

Insert adds a Leaf at the divergence point and lazily grows inner
nodes: `Node4` promotes to `Node16` at 5 children, then to
`Node48` at 17, then to `Node256` at 49.

Erase removes the Leaf and contracts on the way up: at the
hysteresis thresholds 37 / 12 / 3, a `Node256` / `48` / `16`
shrinks to the next smaller variant; `Node4` with one child
collapses into a `Prefix([byte])` that gets merged with any
surrounding prefix. A fully drained tree contracts back to the
`EmptyRoot` sentinel.

In-place leaf-value update on same-size writes skips both the
allocator and free-list paths — common for inode-metadata-style
updates where size doesn't change.

## 4. Cross-blob: spillover, compact, merge

Three primitives keep multi-blob trees healthy.

**`splitBlob` (spillover)** runs in-band when `insert_at` returns
`AllocError::OutOfSpace`. The v0.3 picker is occupancy-aware: it
skips existing `BlobNode` crossings, descends into overfull
path-shaped branches, and chooses a subtree near the target child
fill band instead of blindly peeling off the largest direct child.
The selected subtree is deep-cloned into a fresh blob via
`make_blob_from_node`, staged in the buffer manager's dirty set,
removed from the source blob, and replaced by a `BlobNode`
crossing. The walker then retries the insert; the descent now
follows the new child blob's `header.root_slot`.

To keep spillover from itself OOM'ing, `alloc_node` (non-`Blob`)
refuses to consume the last 128 bytes
of the data area — exactly one `BlobNode` body's worth.
`alloc_node(Blob)` is exempt and may consume that reservation,
guaranteeing spillover can always install its emergency crossing.
The same 128-byte pair also serves a cross-type fallback:
`alloc_node(Blob)` reuses a freed `Prefix` slot body when the
`Blob` free list is empty, and vice versa.

**`compactBlob`** deep-clones the live tree out of a blob into a
scratch buffer (same `clone_subtree` machinery as splitBlob) and
copies the packed image back. The result is a contiguous data
area with empty free lists and the original GUID preserved.
Reclaims:

- Dead leaf nodes (variable-size, no free list — they accumulate
  on resized-value update / delete / spillover-migrate).
- Stale slot-entry / body-byte pairs whose NodeType free list
  has no live demand.

The walker pairs splitBlob + compactBlob on every retry: spillover
returns nodes to the per-type free lists, then compact does the
byte-level repack so the next walker pass sees a fresh bump
cursor.

**`mergeBlob`** is the inverse of splitBlob — a child blob's
subtree gets inlined back into its parent at the `BlobNode` slot
(preserving the BlobNode's inline prefix as a wrapping `Prefix`),
then the child blob is deleted. Guarded by `is_mergeable`:
combined space + slots fit, no nested crossings, no tombstones.

`Tree::compact` is online and candidate-driven. Foreground churn
queues blob-local compaction candidates and parent-merge
candidates; a cold manual call seeds those queues only when no
hints exist. Blob-local compaction runs on the shared side of the
tree-wide `maintenance_gate` under that blob's latch. Parent
merge/delete takes the exclusive side only around the one edge
being folded. Parents do not store child entry slots; after a
child compacts, its own `header.root_slot` remains the only
cross-blob entry token.

## 5. Concurrency

### Per-blob HybridLatch (LeanStore 3-mode)

Every cached blob lives behind a `HybridLatch` wrapping the 512
KB buffer in `UnsafeCell`:

| Mode       | Cost                | Used by                              |
|------------|---------------------|--------------------------------------|
| Optimistic | atomic load + check | `Tree::get` walker (wait-free)       |
| Shared     | brief CAS spin loop | `Tree::stats`, checkpoint snapshot   |
| Exclusive  | brief CAS spin loop | `Tree::put` / `delete` / spillover   |

State encoding (single `AtomicU32`):
- `0` = idle, `1..(WRITER-1)` = N shared readers,
- `WRITER = u32::MAX` = exclusive.

Plus an `AtomicU64` version counter bumped on every exclusive
release. Optimistic readers snapshot version → walk → revalidate;
on a torn read the lookup restarts from the root (the parent's
BlobNode may have moved too, so re-entry has to be the tree root).

### Writer synchronisation — per-blob latches + publish gates

`Tree::put` / `Tree::delete` enter the shared side of
`maintenance_gate` while they may cross `BlobNode` boundaries. That
prevents a maintenance merge from deleting a child after a
foreground walker has observed the parent edge but before it pins
the child. Blob-local conflicts are handled by the per-blob
`HybridLatch`: disjoint child blobs can mutate concurrently.

`Tree::atomic` takes the exclusive side of the same gate for its short
preflight + apply window. That makes logical failures invisible to
concurrent readers/writers: rename, conditional, and
prefix-emptiness guards are checked before any walker mutation, and
no ordinary operation can observe the intermediate state while the
committed batch is being applied.

Persistent writers also enter the writer side of `CommitGate`
while they mutate cached blobs, publish dirty/pending-delete
state, and submit an already-encoded WAL record to the journal
worker. The checkpoint path takes the checkpoint side of the same
gate while draining dirty state, flushing the journal, and cloning
bytes. This is the W2D boundary: any store image written by a
checkpoint has a WAL record admitted and flushed before the bytes
are copied for write-through.

The journal worker owns the `WalWriter`. Callers with
`Durability::Wal { sync: true }` wait outside `CommitGate`; sync requests
arriving in the short group window share one `sync_data`.

`Tree::rename` takes a separate `Mutex<()>` `rename_lock` around
its multi-step `lookup → erase → insert` so other renames see it
atomically. `put` / `delete` / `get` never take `rename_lock`.

## 6. Persistence + crash safety

### WAL — logical redo log of WalOps

Mutations emit encoded `WalOp` records to an append-only
`journal.wal` file via the journal worker. The durable variants
are the logical API mutations: `Insert`, `Erase`, `RenameObject`,
and `Batch`. Blob-shape changes (`splitBlob`, `mergeBlob`,
`compactBlob`) are recovered either by replaying those logical
records or by loading checkpointed blob images; they are not
standalone WAL records. Each record is

```text
MAGIC | LEN | SEQ | TY | BODY | CRC32
```

with hardware-accelerated CRC32 (`crc32fast`, dispatching to
PCLMULQDQ on x86_64 + ARM-CRC32 on AArch64). The writer's pending
buffer auto-drains to the OS page cache at 64 KB. `Journal::flush`
and durable group-commit batches are the `sync_data` boundaries.

Replay walks the journal forward, validating CRC + magic +
variant tag on each record. Torn tails (mid-write power loss)
are recovered gracefully — the scanner reports the offset where
it stopped; real mid-file corruption surfaces as
`Error::ReplaySanityFailed` with the bad record's offset.

### `BufferManager` — cache + dirty/pending tracking

`BufferManager` wraps any `BlobStore` and itself implements
`BlobStore` (drop-in for the write path). Backed by a sharded
`DashMap<BlobGuid, Arc<CachedBlob>, GuidBuildHasher>` so concurrent
`pin` / `get_cached` on different blobs hit different shards instead
of contending on a single mutex. The map is keyed with a cheap
custom hasher (`guid_hash::GuidBuildHasher`) rather than the default
SipHash13: a `BlobGuid` is already 16 high-entropy bytes, so a
two-lane multiply + splitmix avalanche distributes as well at ~2.5×
the speed. Since a multi-blob lookup pays one cache hash per
`BlobNode` crossing, this measurably shrinks the per-crossing cost
(see below).

It tracks "newer than store" state in dirty state plus deferred
delete state:

- `dirty.dirty: HashMap<BlobGuid, u64>` — guid → lowest
  unflushed WAL seq not yet claimed by a checkpoint round. Entry
  exists iff the cached image is newer than store.
- `dirty.flushing: HashMap<BlobGuid, u64>` — entries drained by
  a checkpoint round whose cached image must remain unevictable
  until `write_through` completes. Eviction treats both dirty maps
  as protected.
- `pending_deletes: Mutex<HashMap<BlobGuid, u64>>` — blobs the
  erase walker's `SubtreeGone` path unlinked from their parent
  in cache, queued for `store.delete_blob` at the next
  checkpoint round so the manifest mutation can't race ahead of
  the WAL record covering the unlink.

LRU eviction uses a `clock_tick` / `last_touched` mechanism:
the inline overflow path walks the cache for the oldest tick
whose `Arc::strong_count == 1` (no outstanding pin), so pinned
blobs are skipped until the pinning walker drops its handle.
The same primitive drives the background eviction sweep.

Observability paths (`Tree::stats`, metrics scrapes) use
`pin_silent` / `get_cached_silent` so a scrape does not bump
cache hit/miss counters or refresh the LRU tick — the call
must not inflate the very counters it's about to report.

### Cold-read path — page-granular reads instead of full-frame pins

For write-cold data a point lookup does not need the whole 512 KB
frame. Four layers cut the cold read down:

1. **In-blob routing region.** On compaction a blob's internal nodes
   are clustered into `[routing_off, leaf_region_start)` and its leaves
   are page-aligned at/after `leaf_region_start` (`layout/header.rs`).
   A cold routed read fetches the header page + routing region + the one
   leaf page its descent reaches — ~18 KB mean, ~27× less cold I/O than
   pinning the frame. `routing_len == 0` is the legacy whole-frame layout.
2. **Per-blob bloom** (`bloom_off` / `bloom_len`, at the routing-region
   tail, so it is read for free with the region). A negative cold lookup
   returns `NotFound` without the leaf-page read. Built from stored keys,
   so it never false-negatives; it only ever *skips* a leaf read.
3. **Bounded resident routing cache** (`store/routing_cache.rs`), keyed by
   `(guid, compact_times)` under a fixed byte budget — repeat cold reads of
   a hot blob skip the routing-region read, paying header + leaf page only.
4. **Cold-scan read-ahead.** `BufferManager::pin_scan_many` probes the
   cache for upcoming children, batches the misses through
   `BlobStore::read_blobs`, and inserts them before the walker arrives —
   so scans read children at device queue depth, not one round-trip each.

Structural write-path mutations de-route a blob (`routing_len = 0`); the
maintenance pass re-routes write-cold blobs lazily so write-once data
regains the page-granular path without a foreground cost.

### Crossing cost — why no pointer swizzling

A LeanStore-style optimization would *swizzle* the `BlobNode`
crossing: cache the resolved `Arc<CachedBlob>` (or a raw pointer)
on the parent edge so a crossing skips the `BufferManager::pin`
(the `DashMap` lookup + `Arc` clone). We measured the ceiling
before committing to that complexity, on a ~20k-key object-store
metadata tree (7 blobs, depth-1 forest, ~0.94 pins per lookup):

- A crossing's removable work is exactly one `pin`. Everything
  else on the path — the child `HybridLatch` acquire, the
  `header.root_slot` re-read, the optimistic `validate` — is
  irreducible.
- The full-swizzle **ceiling is ~9–16 %** of a multi-blob point
  read (machine-dependent; higher on a faster box where the fixed
  pin is a larger share). That is an *upper bound* — a real swizzle
  must add back parent + child `content_version` validation, and
  it pays nothing on single-blob reads.
- Against that, swizzling adds a large correctness surface unique
  to this buffer manager: there is **no per-slot generation**, so a
  cached pointer cannot tell "same blob, unchanged" from "evicted
  and reloaded into a fresh `CachedBlob` (version reset to 0)" — a
  classic ABA. `install_new_blob` can also replace a GUID's slot
  while an old `Arc` is still pinned (split-brain), and CoW fork /
  pending-delete each invalidate a cached edge. The only safe form
  retains the `Arc` (so `strong_count > 1` blocks eviction), which
  pins blobs in memory and fights the cache.

So instead of swizzling we took the part of `pin` that is *purely*
cost with no correctness surface — the cache hash — and made it
cheap (`GuidBuildHasher`, above). That recovered a large fraction
of the same ceiling (the real `pin` dropped ~37 %, a multi-blob
get ~9 % on x86) with zero new invariants, and it benefits every
`pin` (writes, scans, range) rather than only the hot read path.
Crucially, lowering the pin cost also *shrinks* the swizzle
ceiling itself, making the high-risk path even less worthwhile.
Holt already has the coarse, safe analogue of swizzling — the
parent-validated `route_cache` — which collapses a deep walk to a
single prefix-anchor edge for hot prefixes without caching raw
pointers.

### Checkpoint — durable dependency protocol

`Tree::checkpoint` and the background `Checkpointer` share the same W2D
ordering, while the background path pipelines complete `CheckpointEpoch`
objects through a bounded planner → I/O queue:

1. **Capture intent** under `CommitGate` checkpoint mode: drain dirty and
   pending logical-delete state, install in-flight protection, record content
   versions, and force the matching WAL watermark durable. A failure restores
   the complete capture.
2. **Clone version-matched bytes** outside the commit gate. A racing mutation
   restores the stale item instead of letting old bytes retire newer debt.
3. **Build dependency waves** from physical BlobNode edges. Missing children,
   self-cycles, and longer cycles fail closed; children are always submitted
   before parents, including no-WAL checkpoints.
4. **Write each wave with CAS-on-version.** Successful writes retire only their
   matching dirty/flushing ownership. External dirty children defer their
   parents to a later epoch rather than publishing a dangling durable edge.
5. **Pre-delete sync** persists data and manifest additions before any logical
   delete can remove an old child mapping.
6. **Apply pending logical deletes**, then run a second sync if any manifest
   delete applied. Structural merge children never enter this queue: they stay
   in parent-scoped orphan staging until the rewritten parent is dirty and a
   clean durable frontier is proven.
7. **Retire epochs in FIFO order.** An epoch error restores all unreported
   dirty/delete ownership, and a later epoch cannot advance the WAL watermark
   past it.
8. **Conditional WAL truncate** requires an empty pipeline plus zero dirty,
   flushing, pending-delete, orphan-staging, and write-delta debt and no store
   flush debt.
9. **Bounded exact reclaim** drains retired COW/structural GUIDs while the
   maintenance fence still excludes topology changes. Pinned GUIDs return to
   the FIFO; crash leftovers are handled by an explicit full reachability GC.

The background checkpointer has three threads: a planner/orchestrator, an I/O
worker that executes complete dependency-ordered epochs from a bounded
`crossbeam-channel`, and an independent cold-blob eviction sweep. `Drop` joins
the planner and runs a final synchronous round so state admitted after the last
idle round is not lost.

`Tree::compact` is online with respect to point reads and
foreground writers through `maintenance_gate`. Range iterators
keep a versioned traversal stack between `next()` calls. If a
writer rewrites any blob on that path, the iterator invalidates the
stack and seeks from the last emitted key / delimiter boundary
instead of continuing through stale `(blob_guid, slot)` state.

## 7. Range iteration

`Tree::range()` and `Tree::scan(p)` return a
`RangeBuilder` → `RangeIter` yielding `RangeEntry::{Key,
CommonPrefix}` items in lex order. `Key` entries carry key, value,
and the live `RecordVersion` from the same leaf emit.
`Tree::range_keys()` and `Tree::scan_keys(p)` return the key-only
companion `KeyRangeBuilder` → `KeyRangeIter`; it uses the same
cursor and delimiter machinery but emits `KeyRangeEntry` without
materialising value bytes. The builders chain:

- `.prefix(p)` — marker-aware lower-bound seek to the prefix range;
  no full-tree scan.
- `.start_after(k)` — strict-greater lower bound for pagination;
  combined with `.prefix(p)` as `max(prefix, marker)`.
- `.delimiter(b)` — S3-style rollup; folds every leaf under a
  common prefix into a single `CommonPrefix` emission and
  fast-forwards the descent stack past that subtree so the cost
  is `O(distinct_rollups)`, not `O(leaves_under_prefix)`.

Cross-blob descent is transparent — the same path stack used
for in-blob traversal also crosses `BlobNode` boundaries via
shared read guards on each child blob. The projection is chosen at
iterator construction time, so full-record and key-only scans share
the same restart and delimiter correctness path.

Forward-only, restart-on-conflict cursor — writers can interleave
between `next()` calls, but any observed blob-version change on the
cursor path forces a rebuild from the monotonic lower bound. This
is stronger than the upstream-style "invalid iterator" surface
because stale paths are handled internally. It is still not MVCC:
a long scan can observe keys committed after iterator creation if
they sort after the current cursor. Each cursor step holds the shared
maintenance/mutation gates only for that step; an iterator paused in
caller code does not block structural maintenance.

For stable read transactions, `Tree::view(prefix, |view| ...)`
copies one root frame and shares descendants with the live tree.
Subsequent live writes fork only snapshot-visible frames. Cloned views
and owned cursors keep the snapshot epoch lease alive after the callback
or main snapshot handle returns, until the final derived handle drops.

## 8. BlobStore abstraction

```rust
pub trait BlobStore: Send + Sync {
    fn alloc_blob_buf_zeroed(&self) -> AlignedBlobBuf;
    fn alloc_blob_buf_uninit(&self) -> AlignedBlobBuf;
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()>;
    fn read_blobs(&self, guids: &[BlobGuid], dsts: &mut [AlignedBlobBuf]) -> Vec<Result<()>>;
    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()>;
    fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()>;
    fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()>;
    fn delete_blob(&self, guid: BlobGuid) -> Result<()>;
    fn list_blobs(&self) -> Result<Vec<BlobGuid>>;
    fn flush(&self) -> Result<()>;
    fn needs_flush(&self) -> bool;
    fn has_blob(&self, guid: BlobGuid) -> Result<bool>;
}
```

`AlignedBlobBuf` is a 4 KB-aligned 512 KB heap buffer — safe to
hand directly to `O_DIRECT` or to register with `io_uring`.
`flush()` blocks until every previously-returned `write_blob` is
durable on the underlying medium.

Implementations:

- **`MemoryBlobStore`** — `RwLock<HashMap<BlobGuid,
  AlignedBlobBuf>>`. For tests, micro-benches, and ephemeral
  workloads.
- **`FileBlobStore`** — single packed `blobs.dat` (blob N at
  byte offset `N × PAGE_SIZE`) plus `manifest.bin` snapshot and
  append-only `manifest.log` deltas. Opens with `O_DIRECT` on
  Linux, `F_NOCACHE` (`fcntl`) on macOS.
- **`io_uring` fast path** (`cfg(target_os = "linux") + feature
  = "io-uring"`) — `FileBlobStore` routes reads, batched
  writes, and data-file fsync through a per-store ring with a
  fixed file and a bounded registered-buffer pool. Non-Linux builds
  with the feature flag still get the syscall path.

`BlobStore` is part of the public API surface so users can plug in
custom storage; everything else (`BufferManager`, `BlobFrame`,
the walker guards) is `pub(crate)`.

## 9. Threading model

`Tree` is `Send + Sync`. Concurrency is per-blob:

- Operations on different blobs run truly in parallel.
- Operations on the same blob serialise at that blob's
  `HybridLatch`.
- Persistent writers share the `CommitGate` publish window only
  while dirty state and journal submission are made visible to
  checkpoint. Durable fsync waiting is handled by the journal
  worker outside that gate.
- Reads take optimistic latches first; only escalate to shared /
  exclusive when needed.

The library does not manage a thread pool. The caller supplies
threads (`std::thread`, `tokio`, `rayon`, whatever). The
background checkpointer is the only thread holt itself owns, and
only when `CheckpointConfig::enabled = true`.

## 10. Failure modes

| What | Behaviour |
|---|---|
| Crash mid-write | WAL replay restores the tree to the last durable record. Uncommitted partial writes drop. |
| WAL torn tail | Replay yields every complete record before the chop, reports the byte offset where it stopped. Real mid-file corruption surfaces as `Error::ReplaySanityFailed`. |
| Partial `store.flush` | Manifest deltas are appended and fsync'd before becoming the recovery contract; full manifest snapshots use tmp+rename. Data file writes are O_DIRECT aligned (atomic at 4 KB on NVMe). |
| Out of disk space | BlobStore write errors propagate as `Error::BlobStoreIo`; the dirty entry stays in BM for retry, no state corruption. |
| OOM in buffer pool | Clock-tick eviction reclaims cold blobs; pinned blobs are skipped until released. |
| Checkpoint mid-failure | Each phase restores any drained state on error return so the next round retries cleanly (see §6 phase 1-7). |

For the supported user API surface and the SemVer contract, see
the top-level re-exports in `src/lib.rs` and the `Breaking`
sections of [CHANGELOG.md](CHANGELOG.md).
