# artisan — roadmap

## Where things stand

The algorithm core (Stage 2), the storage cache (Stage 6
phase 1+2a+2b+2c), and the WAL record codec (Stage 5a) are all
done. The tree walks insert / lookup / erase / rename across
arbitrarily many 512 KB blobs, auto-splits when any blob fills,
has SIMD-accelerated Node16 byte search, and ships a criterion
bench that runs **~3.5–5× faster than RocksDB** on small-metadata
workloads (both `memory` and `persistent` variants).

Concurrency model is settled: per-blob `HybridLatch` (LeanStore
3-mode) gives wait-free optimistic reads + per-blob exclusive
writes with **no Tree-wide writer mutex**. 159 tests + a
4-readers × 1-writer concurrent stress test all green.

The remaining v0.1 cuts are around **WAL persistence** (Stage 5b/5c
— writer/replay/integration) and **shrink-on-erase + tombstone
GC**. The sections below are the live status — see
[ARCHITECTURE.md](ARCHITECTURE.md) for design and `git log` for
what changed when.

The goal is **v0.1: a usable embedded library** for path-shaped
metadata, single-node + persistent + crash-safe. After that we
extend (background checkpointer, async backends, MVCC snapshots,
etc.).

## v0.1 — Usable embedded library

Required for the v0.1 tag:

### Core engine

- [x] `NodeType` enum + all per-NodeType struct layouts
      (Leaf 16 B, Prefix 128 B, Blob 128 B, Node{4,16,48,256},
      EmptyRoot 8 B)
- [x] 4096-byte `BlobHeader` with compile-time-asserted field
      offsets (num_slots, root_slot, space_used, gap_space,
      free_list_head, blob_guid)
- [x] Bit-packed `SlotEntry` (`ntype << 17 | offset / 8`)
- [x] `BlobFrame` bump allocator with per-NodeType free list
- [x] Cross-type free-list fallback (Prefix ↔ Blob, both 128 B)
      so spillover can install its own BlobNode without bump room
- [x] 3-mode `HybridLatch` (optimistic / shared / exclusive)
- [x] Recursive walker: insert / lookup / erase / rename
  - [x] Leaf + EmptyRoot arms
  - [x] Node4 arm + promotion to Node16
  - [x] Node16 + promotion to Node48
  - [x] Node48 + promotion to Node256
  - [x] Prefix arm (full-match descent + mismatch split)
  - [x] BlobNode lookup arm (Stage 2d phase A)
  - [x] BlobNode insert arm with auto-spillover (Stage 2d phase B)
  - [x] BlobNode erase arm + child-blob delete-on-empty
        (Stage 2d phase C)
  - [ ] Shrink chain on erase (Node256 → 48 → 16 → 4) — collapse
        currently always wraps surviving child in `Prefix([byte])`
        to preserve descent invariants
  - [ ] Tombstone + lazy reclaim
- [x] `make_blob_from_node` deep-clone primitive
- [x] `splitBlob` automatic spillover trigger (in-band on OOM)
- [x] `free_subtree` (recursive slot reclaim post-migration)
- [x] `compactBlob` — in-place reclaim of leaf-extent leaks via
      clone-into-scratch + memcpy-back. Wired into the multi-blob
      insert OOM recovery loop alongside `splitBlob` (the two run
      back-to-back so spillover frees a subtree and compact then
      reclaims its bump-area bytes).
- [x] `SPILLOVER_RESERVATION` (128 B bump headroom) — walker
      `alloc_node`/`alloc_extent` (non-Blob) leave one BlobNode's
      worth of bump area for spillover's emergency placeholder
- [ ] `mergeBlob` (compaction inverse — child blob → parent)
- [x] In-place leaf-value update when new value fits existing
      extent footprint (zero alloc, zero extent leak)
- [x] SIMD Node16 byte search (SSE2 / NEON / scalar fallback)
- [x] SIMD `longest_common_prefix` for leaf-split / prefix-split
      hot paths
- [x] Single-Mutex Tree write lock (Stage 5 swaps for per-blob
      HybridLatch)
- [x] Strict-prefix support via Tree-layer terminator byte
- [x] Atomic rename (single-blob and cross-blob; both flavours
      run erase_multi + insert_multi under the write_lock;
      Stage 5 will swap for a dedicated RenameTxnOp so the
      child-blob writes between erase and insert commit as one
      WAL record)
- [ ] Stateful iterator with prefix + start_after + delimiter

### Persistence + crash safety

- [x] 10 `TxnOp` variants enumerated (`Insert`, `Erase`, `Split`,
      `Merge`, `Compact`, `RenameObject`, `Rename`, `NewTree`,
      `RmTree`, `MemMarker`)
- [x] **Binary record codec** (Stage 5a) — fixed header
      (`MAGIC | LEN | SEQ | TY`) + variant body + CRC32 footer.
      All variants round-trip; corruption (CRC, magic, truncation,
      unknown tag) surfaces as `Error::ReplaySanityFailed`. See
      [`journal::codec`](src/journal/codec.rs).
- [x] **`WalWriter`** (Stage 5b) — append-only file with
      buffered I/O and `sync_data`-on-flush; explicit `flush()` is
      the durability boundary. `discard_pending()` for bail-out
      paths. `open_or_create()` validates the `tree_id` of an
      existing log.
- [x] **`replay()` forward scanner** (Stage 5b) — yields every
      record to a callback in order; stops cleanly at a torn tail
      and reports its byte offset in `ReplayStats`. Real
      corruption mid-file (CRC mismatch, magic mismatch, unknown
      variant tag) surfaces as `Error::ReplaySanityFailed` with
      the bad record's offset patched in.
- [ ] Tree integration — `put` / `delete` / `rename` emit ops;
      `Tree::open` replays the log; `Tree::checkpoint` trims past
      the last durable blob commit (Stage 5c)

### Storage backends

- [x] `Backend` trait (blob-granular I/O, 4 KB-aligned via
      `AlignedBlobBuf`)
- [x] `MemoryBackend` — `RwLock<HashMap<_, AlignedBlobBuf>>`
- [x] `PersistentBackend` (cross-platform) — `O_DIRECT` on Linux,
      `F_NOCACHE` on macOS, single packed `blobs.dat` + atomic-
      rename `manifest.bin` + `fdatasync` on flush
- [ ] `io_uring` submission/completion hot path on Linux
      (Stage 7 — currently `pread`/`pwrite`)
- [x] `TreeBuilder` + single `Tree::open(TreeConfig)` entry

### Concurrency

- [x] `HybridLatch` 3-mode primitive (LeanStore-style: optimistic
      / shared / exclusive)
- [x] `BufferManager` — LRU-bounded cache wrapping any Backend.
      Implements Backend itself (transparent drop-in).
- [x] `BlobFrameRef<'a>` + `BufferManager::pin(guid)` —
      pin-and-operate read path. `Tree::get` walks each blob via
      a shared read-guard with **no 512 KB memcpy per hop**
      (Stage 6 phase 2a).
- [x] Walker insert/erase use `pin` + `BufferManager::commit`
      throughout — `Tree::state.root_buf` removed, root + every
      cross-blob hop operate in place against the BM-owned buffer
      (Stage 6 phase 2c).
- [x] **`HybridLatch` wired into `CachedBlob`** —
      `RwLock<AlignedBlobBuf>` replaced by `HybridLatch +
      UnsafeCell<AlignedBlobBuf>`. Three guards exposed:
      `read_optimistic()` (wait-free snapshot+validate),
      `read()` (shared), `write()` (exclusive).
      `Tree::get`'s walker runs in optimistic mode and restarts
      from the root on a torn read; `put` / `delete` never take
      a Tree-wide mutex (per-blob exclusive on the root
      serialises them) — Stage 6 phase 2b.
- [ ] Cross-blob lock-coupling (`BlobNode` descent acquires the
      target blob's latch)
- [x] MVCC seq counter bumped on writes (carried in Leaf body;
      not yet read by lookup)
- [ ] Per-blob `ext_bfs_latch` (second-tier latch for the ext-blob
      cache)

### Public API

- [x] `Tree::open(TreeConfig)` — single entry; `TreeConfig::new(dir)`
      = persistent (default), `TreeConfig::memory()` = volatile
- [x] `Tree::put / get / delete / rename` (with cross-blob lookup
      + auto-spillover insert; delete + rename cross-blob queued)
- [ ] `Tree::range(prefix)` + `.delimiter(b'/')` + `.start_after(key)`
      + `.take(n)`
- [ ] `Tree::txn(|t| { ... })` for batch ops under one WAL record
- [x] `Tree::checkpoint()` (flushes cached root + backend flush)
- [ ] `Tree::stats()` — per-blob compact_times, tombstone count,
      slot utilization
- [x] `TreeBuilder` (chainable: `.memory()`, `.buffer_pool_size(n)`,
      `.wal_sync_on_commit(bool)`, `.checkpoint_byte_interval(b)`)
- [x] Typed errors (`Error::{BackendIo, Alloc, Free, KeyTooLong,
      ValueTooLong, NotYetImplemented, NodeCorrupt,
      ReplaySanityFailed, NotFound, DstExists}`)

### Testing + benchmarks

- [x] Unit tests for every NodeType arm of the walker
- [x] Multi-blob auto-spillover end-to-end test (~2000 keys ×
      200 B values forces spillover, every key still readable)
- [ ] Property-based tests (random key insertion, random erase,
      verify lookup consistency)
- [ ] Recovery tests (insert, kill process mid-write, recover,
      verify) — pairs with WAL (Stage 5)
- [x] Concurrent stress test (8 threads × 25 puts each, all
      readable after; single-Mutex so writes serialise)
- [x] Criterion benchmarks: KV / objstore-metadata / fs-metadata
      shapes × get / put / mixed, side-by-side with RocksDB
      (no-WAL parity). See [benches/README.md](benches/README.md).

### Docs + examples

- [ ] `examples/basic_kv.rs` — minimal "open, put, get, close"
- [ ] `examples/filesystem_meta.rs` — artisan as the metadata layer
      for a toy POSIX filesystem
- [ ] `examples/session_store.rs` — multi-tenant chat session storage
- [ ] `examples/s3_metadata.rs` — artisan as an S3-compatible object
      metadata backend
- [ ] Rendered docs.rs documentation (every public type + method)
- [ ] `docs/benchmarks.md` with numbers vs LMDB / RocksDB / Sled

### Polish

- [ ] CI (cargo test + clippy + rustfmt + miri on a subset)
- [ ] Cross-platform (Linux + macOS + Windows tier-1)
- [ ] MSRV bump policy
- [ ] Versioning policy
- [ ] CHANGELOG.md
- [ ] CONTRIBUTING.md
- [ ] CODE_OF_CONDUCT.md

## v0.2 — Performance

- Async checkpointer (3 background threads: checkpoint / io / eviction)
- io_uring backend (Linux, behind feature flag)
- SIMD-accelerated Node16 keys[] scan (vpcmpeqb)
- Lock-free reader fast path (validated optimistic snapshot)
- Buffer-pool tuning + adaptive eviction
- Metrics export (Prometheus + OpenTelemetry traces)

## v0.3 — Advanced features

- Full MVCC snapshots (read at a specific seq, snapshot iteration)
- Online compaction (background, doesn't block writers)
- Change feed / subscription API (consumers receive a stream of
  TxnOps)
- Column families (multiple independent ARTs in one Tree)
- Encryption-at-rest (per-blob AES-GCM)
- Compression (per-blob Zstd, transparent)

## v1.0 — Production-ready

- Comprehensive feature set covered.
- Multi-platform stability (Linux + macOS + Windows + BSDs).
- Real production deployments + case studies.
- Long-term API stability commitment.

## Not on the roadmap

The library deliberately stays single-node. Things outside scope:

- **Replication / consensus**: build it above this. We expose hooks
  (change feed, snapshot transfer) but don't implement Raft.
- **Network server**: this is a library. Wrap it in your gRPC /
  HTTP / whatever.
- **SQL**: not the right abstraction for this data shape.
- **Vector search**: combine with a dedicated vector DB.
- **Full-text search**: combine with Tantivy / Lucene-rs.

## Contributing

We're at very early stage; ideas + design feedback most welcome.
PRs welcome too, but please open an issue first for non-trivial
changes — the architecture is being shaped and we want to avoid
churn.
