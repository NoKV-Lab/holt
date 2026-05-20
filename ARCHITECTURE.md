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
says "the walk continues in blob X at slot Y." This composes
recursively: each child blob is itself a full ART frame, so the
same walker code descends across blob boundaries without
special-casing the crossing.

## 2. NodeType variants

| ntype | Name      | Size       | Purpose                                        |
|------:|-----------|-----------:|------------------------------------------------|
|     1 | Leaf      |    16 B    | `(value_size, tombstone, key_offset, seq)` + bump-allocated extent for key+value bytes |
|     2 | Prefix    |   128 B    | Path-compressed segment (≤112 inline bytes)    |
|     3 | Blob      |   128 B    | Cross-blob crossing (target_guid + entry_slot) |
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
            Blob      -> pin target blob, descend at entry slot
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
`AllocError::OutOfSpace`. It picks the largest non-`Blob` subtree
under the current blob's first branching node, deep-clones it
into a fresh blob via `make_blob_from_node`, persists the new
blob, frees the source slots, and installs a `BlobNode` crossing
in the parent. The walker then retries the insert; the descent
now follows the new BlobNode.

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

`Tree::compact` (today, quiescent-only — see §6) walks every
blob, runs `compact_blob` per blob, repairs the
`BlobNode.child_entry_ptr == child.header.root_slot` invariant
via `refresh_blob_node_pointers` (compact_blob rewrites a child's
root in isolation; parents need a separate sweep to catch up),
then folds every direct-mergeable `BlobNode` child via a
single-pass merge sweep.

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

### Writer synchronisation — `wal.lock` is the per-op barrier

`Tree::put` / `Tree::delete` for persistent trees take
`wal.lock()` (the `Mutex<WalWriter>`) **once** at the top and
hold it across:

1. Walker descent (including all cross-blob hops, spillover, and
   compact retries — these take the per-blob `HybridLatch`
   exclusive as needed).
2. `bm.mark_dirty(root_guid, seq)` (plus any
   `mark_dirty(child_guid, seq)` / `mark_for_delete(...)` the
   walker issued internally).
3. `wal.append_*` for the op's record.
4. Optional `wal.flush()` if `wal_sync_on_commit` is set.

The wal-lock-around-the-whole-op shape is **load-bearing**: any
dirty / pending-delete entry visible to a checkpoint round has
its corresponding WAL record already buffered, because both
`Tree::checkpoint` and the bg round snapshot under the same
`wal.lock`. This is the W2D-strict invariant: every backend
mutation is preceded by a durable WAL record describing it.

Concurrent writers serialise on `wal.lock`. That's the current
barrier, not the end state. The v0.3 performance track replaces
the coarse write shape with slot-versioned cross-blob
lock-coupling and a journal worker / group-commit pipeline, so
writers on disjoint child blobs do not keep an ancestor guard or
the WAL mutex for the full mutation.

`Tree::rename` takes a separate `Mutex<()>` `rename_lock` around
its multi-step `lookup → erase → insert` so other renames see it
atomically. `put` / `delete` / `get` never take `rename_lock`.

## 6. Persistence + crash safety

### WAL — physiological log of TxnOps

Mutations emit a `TxnOp` to an append-only `journal.wal` file.
Eleven variants today: `Insert`, `Erase`, `Split`, `Merge`,
`Compact`, `RenameObject`, `Rename`, `NewTree`, `RmTree`,
`MemMarker`, `Batch`. Each record is

```text
MAGIC | LEN | SEQ | TY | BODY | CRC32
```

with hardware-accelerated CRC32 (`crc32fast`, dispatching to
PCLMULQDQ on x86_64 + ARM-CRC32 on AArch64). The writer's pending
buffer auto-drains to the OS page cache at 64 KB; explicit
`wal.flush()` is the `sync_data` durability boundary.

Replay walks the journal forward, validating CRC + magic +
variant tag on each record. Torn tails (mid-write power loss)
are recovered gracefully — the scanner reports the offset where
it stopped; real mid-file corruption surfaces as
`Error::ReplaySanityFailed` with the bad record's offset.

### `BufferManager` — cache + dirty/pending tracking

`BufferManager` wraps any `Backend` and itself implements
`Backend` (drop-in for the write path). Backed by a sharded
`DashMap<BlobGuid, Arc<CachedBlob>>` so concurrent `pin` /
`get_cached` on different blobs hit different shards instead of
contending on a single mutex.

It tracks "newer than backend" state in two parallel sets:

- `dirty: Mutex<HashMap<BlobGuid, u64>>` — guid → lowest
  unflushed WAL seq. Entry exists iff the cached image is newer
  than backend.
- `pending_deletes: Mutex<HashMap<BlobGuid, u64>>` — blobs the
  erase walker's `SubtreeGone` path unlinked from their parent
  in cache, queued for `backend.delete_blob` at the next
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

1. **Snapshot + WAL flush** under `wal.lock`. Drain `dirty` +
   `pending` sets and `wal.flush()`. Flush failure restores both
   snapshots.
2. **Per-blob write-through** with CAS-on-seq. Successful writes
   retire the dirty entry; failures stay in `dirty`.
3. **Pre-delete sync** — `backend.flush` (data file fdatasync +
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
seven phases, a dedicated I/O worker processing `IoTask::Flush`
+ `IoTask::Sync` from a bounded `crossbeam-channel` queue, and a
cold-blob eviction sweep on the same `clock_tick` mechanism.
`Drop` joins the planner and runs one final synchronous round on
the calling thread so dirty state between the last bg round and
shutdown doesn't get lost.

`Tree::compact` is currently documented **NOT online-safe** —
running concurrently with reads or writes can torn-read across
`BlobNode` crossings. The v0.3 performance track moves this
toward incremental online maintenance guarded by structure
versions, not a stop-the-world compact pass.

## 7. Range iteration

`Tree::range()` and `Tree::scan_prefix(p)` return a
`RangeBuilder` → `RangeIter` yielding `RangeEntry::{Key,
CommonPrefix}` items in lex order. The builder chains:

- `.prefix(p)` — anchored descent, no full-tree scan.
- `.start_after(k)` — strict-greater lower bound (for pagination).
- `.delimiter(b)` — S3-style rollup; folds every leaf under a
  common prefix into a single `CommonPrefix` emission and
  fast-forwards the descent stack past that subtree so the cost
  is `O(distinct_rollups)`, not `O(leaves_under_prefix)`.

Cross-blob descent is transparent — the same path stack used
for in-blob traversal also crosses `BlobNode` boundaries via
shared read guards on each child blob.

Forward-only, best-effort snapshot — writers can interleave
between `next()` calls (same failure mode as the upstream
algorithm's "invalid iterator" warning); for a strict snapshot,
pause writes externally.

## 8. Backend abstraction

```rust
pub trait Backend: Send + Sync {
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()>;
    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()>;
    fn delete_blob(&self, guid: BlobGuid) -> Result<()>;
    fn list_blobs(&self) -> Result<Vec<BlobGuid>>;
    fn flush(&self) -> Result<()>;
    fn has_blob(&self, guid: BlobGuid) -> Result<bool>;
}
```

`AlignedBlobBuf` is a 4 KB-aligned 512 KB heap buffer — safe to
hand directly to `O_DIRECT` or to register with `io_uring`.
`flush()` blocks until every previously-returned `write_blob` is
durable on the underlying medium.

Implementations:

- **`MemoryBackend`** — `RwLock<HashMap<BlobGuid,
  AlignedBlobBuf>>`. For tests, micro-benches, and ephemeral
  workloads.
- **`PersistentBackend`** — single packed `blobs.dat` (blob N at
  byte offset `N × PAGE_SIZE`) + atomic-rename `manifest.bin`.
  Opens with `O_DIRECT` on Linux, `F_NOCACHE` (`fcntl`) on macOS.
- **`io_uring` fast path** (`cfg(target_os = "linux") + feature
  = "io-uring"`) — `PersistentBackend` routes reads / writes
  through a per-backend `IoUring` (depth 8) instead of `pread` /
  `pwrite`. Non-Linux builds with the feature flag still get the
  syscall path.

`Backend` is part of the public API surface so users can plug in
custom storage; everything else (`BufferManager`, `BlobFrame`,
the walker guards) is `pub(crate)`.

## 9. Threading model

`Tree` is `Send + Sync`. Concurrency is per-blob:

- Operations on different blobs run truly in parallel.
- Operations on the same blob serialise at that blob's
  `HybridLatch` (and at `wal.lock` for the WAL-append window).
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
| Partial `backend.flush` | Manifest rewrite is atomic-rename — old manifest stays intact if the rename doesn't complete. Data file writes are O_DIRECT aligned (atomic at 4 KB on NVMe). |
| Out of disk space | Backend write errors propagate as `Error::BackendIo`; the dirty entry stays in BM for retry, no state corruption. |
| OOM in buffer pool | Clock-tick eviction reclaims cold blobs; pinned blobs are skipped until released. |
| Checkpoint mid-failure | Each phase restores any drained state on error return so the next round retries cleanly (see §6 phase 1-7). |

For the supported user API surface and the SemVer contract, see
the top-level re-exports in `src/lib.rs` and the `Breaking`
sections of [CHANGELOG.md](CHANGELOG.md).
