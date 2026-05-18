# artisan

> A carefully crafted **adaptive radix tree** for path-shaped metadata.

`artisan` is an embedded Rust library for storing **hierarchical
keys** — file paths, S3 object names, multi-tenant namespaces,
time-bucketed identifiers — with sub-microsecond lookups, per-blob
concurrency, and crash-safe persistence.

It is **not** a general-purpose KV store. It targets the workloads
where:

- Keys are **hierarchical / path-shaped** (so prefix compression pays).
- The dominant access pattern is **point lookup + prefix range scan**.
- Concurrency is **high** (many readers + writers across disjoint
  subtrees).
- Latency is **micro-critical** — no LSM compaction stalls, no
  single-writer locks.

If you need full-text search or vector similarity, use a different
tool. If you need exactly this shape, artisan should beat
LMDB / RocksDB / SQLite on its target workload.

## When to reach for artisan

| Engine          | Data structure | Persistence       | Concurrency        | Notes                                                       |
|-----------------|----------------|-------------------|--------------------|-------------------------------------------------------------|
| LMDB            | B+tree         | mmap              | Single-writer MVCC | Battle-tested; cross-page chasing for short hot keys.       |
| RocksDB         | LSM            | SST + WAL         | MVCC               | Compaction stalls; large hot dataset is RAM-heavy.          |
| SQLite          | B-tree         | File              | Single writer      | Convenient, but writer is the bottleneck under load.        |
| Sled            | Hybrid LSM     | Log-structured    | Lock-free          | Rust-native, but largely unmaintained.                      |
| **artisan**     | **Adaptive Radix Tree** | **512 KB blobs** | **Per-blob 3-mode latch** | **Path compression + lookup is O(key.len)**     |

ART's lookup cost is `O(key.len)`, not `O(log N)`. For short hot keys
(say, < 64 bytes), that's faster than any tree-based competitor. The
per-blob HybridLatch lets N readers traverse different subtrees in
parallel without coordinating with each other.

## Project status

**v0.1 in active development.** 159 tests pass; `cargo bench --bench main`
runs a side-by-side comparison with RocksDB (memory + persistent
variants, both showing artisan **~3.5–5× faster** on small-metadata
workloads — see [benches/README.md](benches/README.md)).

Done — algorithm core:

- Layout (9 NodeTypes, 4 KB BlobHeader, bit-packed slot table)
- Walker insert / lookup / erase / rename — **all four with full
  cross-blob support** (Stage 2d phase A / B / C)
- SIMD Node16 byte search + longest-common-prefix (SSE2 / NEON /
  scalar)
- `splitBlob` auto-spillover via `make_blob_from_node`
- `compactBlob` — in-place extent reclaim, paired with splitBlob
  on every OOM so churn workloads (insert + delete + reinsert)
  stay in fewer blobs
- 128-byte bump-area reservation so spillover always has room to
  install its emergency BlobNode
- Child-blob auto-reclaim when an erase empties it
- Strict-prefix support (terminator byte)
- In-place leaf-value update on same-size writes
- `MemoryBackend` + cross-platform `PersistentBackend`
  (Linux `O_DIRECT`, macOS `F_NOCACHE`)
- `BufferManager` LRU cache wrapping any backend (Stage 6 phase 1)
  — `TreeConfig::buffer_pool_size` drives capacity, per-blob
  `RwLock` lets N readers share a single blob in parallel
- Zero-copy `Tree::get` via `BufferManager::pin` + `BlobFrameRef`
  (Stage 6 phase 2a) — read paths walk the cached buffer in place,
  no 512 KB memcpy per cross-blob hop
- Zero-copy `Tree::put` / `delete` / `rename` via
  `BufferManager::pin` + `commit` (Stage 6 phase 2c) — mutations
  also operate in place against the BM-owned buffer; `Tree` no
  longer keeps a separate `state.root_buf`
- **`HybridLatch` wired into `CachedBlob`** (Stage 6 phase 2b) —
  3-mode latch (optimistic / shared / exclusive) over an
  `UnsafeCell<AlignedBlobBuf>`. `Tree::get`'s walker runs
  **wait-free**: snapshot the latch version, walk, validate;
  restart from the root on a torn read. `put` / `delete` no
  longer take any Tree-wide lock — per-blob exclusive on the
  root serialises them. `rename` keeps a small `rename_lock`
  for its multi-step atomicity
- Walker split into focused submodules under `engine/walker/`
  (types / readers / writers / lookup / insert / erase /
  spillover / migrate / tests) — ten files under ~700 LOC each
  rather than one ~3400-LOC blob
- **WAL record codec** (Stage 5a) — binary `TxnOp` format with
  CRC32 `sanity_info`. 10 variants encode / decode round-trip;
  4 corruption catch-cases (CRC, magic, truncation, unknown tag)
  surface as `Error::ReplaySanityFailed`. See
  [`src/journal/codec.rs`](src/journal/codec.rs).
- **WAL writer + replay scanner** (Stage 5b) —
  [`WalWriter`](src/journal/writer.rs) is append-only with explicit
  `flush()`-for-durability (`sync_data`); [`replay()`](src/journal/reader.rs)
  is a forward scanner that handles a torn tail gracefully and
  patches in the byte offset when reporting mid-file corruption.

Queued — see [ROADMAP.md](ROADMAP.md):
- `Tree::open` replays the log; `put` / `delete` / `rename` emit
  records; `Tree::checkpoint` trims (Stage 5c)
- `Tree::range` / `Tree::txn` iterators
- io_uring submission on the persistent backend (Stage 7)
- `mergeBlob` (child-blob → parent inverse of splitBlob)
- Erase-time node shrinkage (Node256 → 48 → 16 → 4)

## Quick taste

```rust
use artisan::{Tree, TreeBuilder, TreeConfig};

// Persistent (default).
let tree = TreeBuilder::new("/var/lib/myapp/meta.artisan")
    .buffer_pool_size(128)
    .open()?;

// Or in-memory:
let tree = Tree::open(TreeConfig::memory())?;

tree.put(b"img/01.jpg", b"rgb_data_blob_id_abc")?;
let value = tree.get(b"img/01.jpg")?.unwrap();
tree.delete(b"img/01.jpg")?;

// Atomic rename (force=true overwrites dst).
tree.rename(b"old/path", b"new/path", false)?;

tree.checkpoint()?;   // flush root blob + manifest
```

## Architecture at a glance

```
┌────────────────────────────────────────────────────────────┐
│ Public API: Tree, Iter, Txn, TreeBuilder                    │
├────────────────────────────────────────────────────────────┤
│ Engine: insert / lookup / erase / scan / rename / compact   │
├────────────────────────────────────────────────────────────┤
│ Concurrency: HybridLatch (3-mode) + lock-coupling           │
├────────────────────────────────────────────────────────────┤
│ Journal: physiological WAL + replay + checkpoint            │
├────────────────────────────────────────────────────────────┤
│ Store: BufferManager + BlobFrame (512 KB, bump alloc)       │
├────────────────────────────────────────────────────────────┤
│ Layout: NodeType variants + SlotEntry + BlobHeader          │
├────────────────────────────────────────────────────────────┤
│ Backend: file / mmap / memory (pluggable trait)             │
└────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the deep dive.

## Design notes

- **Adaptive Radix Tree** core: four internal node sizes
  (4 / 16 / 48 / 256 children) chosen at runtime to keep the tree
  dense. Lookup walks one byte of the key per level.
- **Path compression** via a dedicated `Prefix` node variant — long
  shared paths cost one node, not one per byte.
- **Multi-blob via in-tree crossings**: when a 512 KB blob fills,
  a subtree is migrated to a fresh blob and a `BlobNode` crossing is
  written into the parent. Trees grow to arbitrary size in 512 KB
  increments.
- **Per-blob `HybridLatch`** (LeanStore-style 3-mode lock):
  optimistic readers take a version snapshot and validate; shared
  readers take a reader counter; writers take exclusive. Reader
  fast path is wait-free under no contention.
- **Crash safety** via a physiological WAL with 13+ TxnOp variants
  and a synchronous (or eventually asynchronous) checkpointer.
- **Per-blob free-list** lets recycled slot indices feed back into
  the bump allocator with zero overhead.

The design is informed by:

- Leis et al., "The Adaptive Radix Tree: ARTful Indexing for
  Main-Memory Databases" (ICDE 2013) — the four-node-size scheme.
- Leis et al., "LeanStore: In-Memory Data Management Beyond Main
  Memory" (ICDE 2018) — the HybridLatch contract.

## What this is NOT

To avoid surprise:

- **Not a SQL database.** No joins, no aggregates, no query planner.
- **Not a vector DB.** No kNN, no embeddings, no similarity search.
- **Not a full-text index.** No tokenization, no inverted index.
- **Not a replication / consensus layer.** The library is single-node
  + persistent. Replication is a layer above this.
- **Not a network server.** This is a library you embed; bring your
  own RPC.

For these, combine artisan with a domain-appropriate engine:

- artisan + FAISS / Qdrant / pgvector → AI workspace metadata + vectors
- artisan + Tantivy → FS metadata + full-text
- artisan + custom Raft → distributed deployments

## License

Dual-licensed under Apache-2.0 OR MIT.
