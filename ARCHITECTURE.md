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
|   Node bodies (8 to 1032 bytes each, NodeType-tagged)       |
|   Leaf key/value extents (raw bytes, ref'd by Leaf.key_off) |
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
|     1 | Leaf      |    16 B    | `(value_size, tombstone, key_offset, seq)` + bump-allocated extent for key+value bytes |
|     2 | Prefix    |   128 B    | Path-compressed segment (≤112 inline bytes)    |
|     3 | Blob      |   128 B    | Cross-blob crossing (target_guid + ≤104 inline prefix bytes) |
|     4 | Node4     |    24 B    | 1..4 children, linear scan                     |
|     5 | Node16    |    88 B    | 5..16 children, SSE2 / NEON byte search        |
|     6 | Node48    |   456 B    | 17..48 children, byte→slot index               |
|     7 | Node256   |  1032 B    | 49..256 children, direct array                 |
|     8 | EmptyRoot |     8 B    | All-zero sentinel for an empty tree            |

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

To keep spillover from itself OOM'ing, `alloc_node` /
`alloc_extent` (non-`Blob`) refuse to consume the last 128 bytes
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

- Leaf key/value extents (no free list — they accumulate on
  update / delete / spillover-migrate).
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

The journal worker owns the `WalWriter`. Callers using
`WalCommit::Write` or `WalCommit::Sync` wait outside `CommitGate`;
sync requests arriving in the short group window share one
`sync_data`.

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
`DashMap<BlobGuid, Arc<CachedBlob>>` so concurrent `pin` /
`get_cached` on different blobs hit different shards instead of
contending on a single mutex.

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

### Checkpoint — 7-phase protocol

Both `Tree::checkpoint` (synchronous, caller-driven) and the
background `Checkpointer` round follow the same seven-phase
protocol, strictly ordered around the W2D invariant:

1. **Snapshot + journal flush** under `CommitGate` checkpoint
   mode. Drain live dirty entries into the checkpoint snapshot,
   move them into the in-flight `flushing` protection set, drain
   pending deletes, force the journal durable, and clone the
   snapshotted blob bytes before releasing the gate. Flush failure
   restores both snapshots.
2. **Batched per-blob write-through** with CAS-on-seq.
   Successful writes release the matching `flushing` protection;
   racing writers' fresh dirty entries survive for the next round.
   Failures are restored to live dirty.
3. **Pre-delete sync** — `store.flush` (data file fdatasync +
   manifest persist) so the writes from phase 2 are stable.
   Failure restores `pending`.
4. **Abort-on-dirty-failure gate.** Any phase-2 failure restores
   `pending` and returns without touching the manifest. A failed
   parent write must not propagate to its child's manifest
   delete — that would leave the on-disk parent referencing a
   slot the manifest no longer has, and WAL replay's walker
   descent through the BlobNode crossing would fail.
5. **Apply pending deletes** — manifest mutation in-memory.
6. **Post-delete sync** if any delete actually applied. Failure
   restores the already-applied entries (`execute_pending_delete`
   is idempotent on retry).
7. **Conditional WAL truncate** — only if `dirty_count == 0`
   AND `pending_delete_count == 0` *now*. A racing writer or a
   restored failure keeps the WAL alive until a future round.

The background checkpointer is 3 threads: a planner running the
seven phases, a dedicated I/O worker processing
`IoTask::FlushBatchAndSync` + `IoTask::Sync` from a bounded
`crossbeam-channel` queue, and a cold-blob eviction sweep on the
same `clock_tick` mechanism.
`Drop` joins the planner and runs one final synchronous round on
the calling thread so dirty state between the last bg round and
shutdown doesn't get lost.

`Tree::compact` is online with respect to point reads and
foreground writers through `maintenance_gate`. Range iterators
keep a versioned traversal stack between `next()` calls. If a
writer rewrites any blob on that path, the iterator invalidates the
stack and seeks from the last emitted key / delimiter boundary
instead of continuing through stale `(blob_guid, slot)` state.

## 7. Range iteration

`Tree::range()` and `Tree::scan_prefix(p)` return a
`RangeBuilder` → `RangeIter` yielding `RangeEntry::{Key,
CommonPrefix}` items in lex order. `Key` entries carry key, value,
and the live `RecordVersion` from the same leaf emit. The builder
chains:

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
shared read guards on each child blob.

Forward-only, restart-on-conflict cursor — writers can interleave
between `next()` calls, but any observed blob-version change on the
cursor path forces a rebuild from the monotonic lower bound. This
is stronger than the upstream-style "invalid iterator" surface
because stale paths are handled internally. It is still not MVCC:
a long scan can observe keys committed after iterator creation if
they sort after the current cursor.

## 8. BlobStore abstraction

```rust
pub trait BlobStore: Send + Sync {
    fn alloc_blob_buf_zeroed(&self) -> AlignedBlobBuf;
    fn alloc_blob_buf_uninit(&self) -> AlignedBlobBuf;
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()>;
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
