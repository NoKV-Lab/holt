# holt — roadmap

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
writes with **no Tree-wide writer mutex**. 202 tests (including a
property-based suite and a 4-readers × 1-writer optimistic-read
stress) all green; zero clippy / rustdoc warnings under
`-D warnings`; CI matrix (ubuntu + macOS) wired.

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
  - [x] **Shrink chain on erase** (Node256 → 48 → 16 → 4) —
        thresholds 37 / 12 / 3 give hysteresis vs the grow
        thresholds 48 / 16 / 4. Below the shrink point the
        smaller variant is allocated, the surviving children
        are copied across, the old slot freed, and the parent's
        child pointer rewired via `EraseSignal::Replaced`. The
        terminal `Node4 → Prefix([byte])` lone-child collapse
        is unchanged.
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
- [x] **`mergeBlob`** (compaction inverse — child blob → parent) —
      [`engine::merge_blob`](src/engine/walker/migrate.rs) inlines
      a child's subtree back into its parent at the `BlobNode`
      slot (preserving the inline-prefix wrap), then deletes the
      child blob. Guarded by `engine::is_mergeable` (combined
      space + slots fit, child has no own crossings, no
      tombstones). [`engine::try_merge_children`](src/engine/walker/merge.rs)
      is the tree-walker fold: [`Tree::compact`](src/api/tree.rs)
      runs `compact_blob` per blob, then
      `refresh_blob_node_pointers` repairs the
      `BlobNode.child_entry_ptr == child.header.root_slot`
      invariant `compact_blob` couldn't keep in lock-step, then a
      single-pass merge sweep folds every direct mergeable
      `BlobNode` child. Nested crossings (mergeable-child-of-
      mergeable-child) are deferred to a second `Tree::compact`
      pass.
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
- [x] **Tree integration** (Stage 5c) — every `put` / `delete` /
      `rename` emits a `TxnOp` to the WAL. Per-op `sync_data`
      is opt-in via `TreeConfig::wal_sync_on_commit` (default
      `false`, matching RocksDB's `sync=false` default — high
      throughput, durable past a process crash but not a power
      loss until `Tree::checkpoint`). `Tree::open` replays the
      durable WAL onto the BM-cached blob and resumes `next_seq`
      past every replayed record. `Tree::checkpoint` writes the
      BM root through to the backend, flushes, then atomically
      truncates the WAL to header-only.
- [x] **Group-commit auto-flush** (Stage 5d) — once the
      `WalWriter`'s pending buffer crosses 64 KB the bytes
      drain to the OS page cache via `write_all` (no `sync_data`).
      Bounds the in-memory buffer regardless of how long the
      caller waits between `checkpoint()` calls; `flush()` is
      still the durability boundary for `sync_data`.
- [x] **WAL encode fast path** (Stage 5e) — 256-entry
      compile-time CRC32 table (≈3× faster than the bitwise
      reference impl) + reference-based
      `WalWriter::append_insert / append_erase /
      append_rename_object` methods that skip the `TxnOp` enum's
      three `Vec` clones. Persistent put latency:
      ≈1.74 µs → ≈735 ns (kv) on the bench microcase.

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
- [x] **`Tree::range()` stateful iterator** —
      [`engine::RangeBuilder`](src/engine/walker/range.rs) /
      [`engine::RangeIter`](src/engine/walker/range.rs) /
      [`engine::RangeEntry`](src/engine/walker/range.rs).
      Modelled on the upstream `fa_iter` shape (8 log strings give
      the contract: `path` stack of `(blob_guid, slot)`, materialised
      `curr_key`, exclusive `marker`, char `delimiter`, per-node
      resume cursor `start_index_in_node`). Builder chains
      `.prefix(p)` (anchored descent — no full-tree scan),
      `.start_after(k)` (strict-greater lower bound), `.delimiter(b)`
      (S3-style rollup with dedup of `CommonPrefix` emits). Walks
      transparently across `BlobNode` crossings, holding one shared
      read guard per stack frame. Forward-only — no `findPrev`.
      Best-effort snapshot: between `next()` calls writers can
      interleave (same failure mode as the upstream's
      "invalid iterator(#1)" warning); for strict snapshot, pause
      writes externally. Caller can stop via `.take(n)` /
      collect-with-limit on the `Iterator` trait. Returns
      `Iterator<Item = Result<RangeEntry>>`, where `RangeEntry::Key`
      / `RangeEntry::CommonPrefix` distinguish leaf vs rollup
      emissions.
- [x] **`Tree::txn(|b| { ... })` for batch ops under one WAL record** —
      [`api::TxnBatch`](src/api/txn.rs) buffers `put` / `delete` /
      `rename`; on closure return,
      [`Tree::txn`](src/api/tree.rs) takes `rename_lock`, applies
      each op in order, and emits one [`TxnOp::Batch`](src/journal/txn_op.rs)
      record (new `TY_BATCH = 10` tag). Inner ops carry derived
      seqs (`base + i`) via a contiguous reservation, so neither
      encoder nor decoder needs per-inner seq bytes. Replay
      transparently flattens the batch into per-inner callbacks
      ([`journal::reader::replay_bytes`](src/journal/reader.rs)).
      Crash atomicity: all-or-nothing across restarts. Runtime
      isolation: best-effort — see the contract on
      [`Tree::txn`](src/api/tree.rs).
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
- [x] Concurrent stress test (8 threads × 25 puts each, all
      readable after; single-Mutex so writes serialise)
- [x] Optimistic-readers-vs-writer stress test
      (4 readers × 500 gets + 1 writer × 200 puts, no torn data)
- [x] Property-based tests (`proptest`) — random put / delete /
      rename traces cross-checked against a `HashMap` oracle,
      both memory and persistent (drop-without-checkpoint +
      reopen via WAL replay) modes
- [x] Criterion benchmarks: KV / objstore-metadata / fs-metadata
      shapes × get / put / mixed × memory / persistent, comparing
      against RocksDB **and** SQLite. See
      [benches/README.md](benches/README.md) for methodology +
      how-to-read-the-numbers.

### Docs + examples

- [x] `examples/basic_kv.rs` — minimal "open, put, get, close"
- [x] `examples/filesystem_meta.rs` — holt as the metadata layer
      for a toy POSIX filesystem
- [x] `examples/session_store.rs` — multi-tenant chat session storage
- [x] `examples/s3_metadata.rs` — holt as an S3-compatible object
      metadata backend
- [x] `cargo doc` renders with zero warnings under `-D warnings`
- [x] `benches/README.md` rollup of the criterion methodology +
      how-to-read (kv = ART anti-pattern; objstore / fs = design
      target). LMDB comparison queued (needs a separate dev-dep
      wiring); Sled skipped (largely unmaintained).

### Polish

- [x] **CI** — GitHub Actions matrix (ubuntu + macOS) × build /
      test / doctest + lint (`cargo fmt --check`, `cargo clippy
      -- -D warnings`) + docs (`cargo doc -- -D warnings`) +
      MSRV (Rust 1.82). See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
- [x] **Zero clippy warnings** under `-D warnings`. The vetted
      `#![allow]` block in `src/lib.rs` documents the categories
      where `clippy::pedantic` fires for intentional design
      choices.
- [x] **MSRV policy** — Rust 1.82, gated by the `msrv` CI job
      (library-only build; dev-dependencies routinely require a
      newer toolchain than the library surface itself does)
- [ ] Versioning policy (semver from v0.1.0 onwards)
- [x] **CHANGELOG.md** (this release)
- [x] **CONTRIBUTING.md** (build / test / commit-style guide)
- [x] **CODE_OF_CONDUCT.md** (Contributor Covenant 2.1)

## v0.2 — Performance + concurrency upgrades

v0.2 is **scoped to the metadata-engine core**. No new public API
surface (those land in v0.3). The bench surfaces from v0.1 are
the success criteria — `objstore_get` should stay sub-200 ns,
`*_list` should drop another 30-50 %, `*_list_dir` should drop
by `~leaves_per_rollup` once fast-forward lands.

### Hot-path performance

- [ ] **`io_uring` persistent-backend submission/completion**
      (Linux, behind a `feature = "io-uring"` flag). The current
      backend uses `pread`/`pwrite`; uring removes the per-call
      syscall + lets the BM batch reads on cold-cache walks.
      Stage 7 of the original Stage chain.
- [ ] **SIMD CRC32** — PCLMULQDQ on x86_64, CRC32 intrinsic on
      AArch64. The 256-entry table from v0.1's WAL fast-path
      gets ≈1.5 GB/s; SIMD variants push past 8 GB/s. Drops
      persistent-put cost another ≈80 ns.
- [x] **Cached `Tree.root_pin`** (commit `a6f5c78`) — every
      `get` / `put` / `delete` keeps the root pinned via
      `Arc<CachedBlob>` and skips `BufferManager`'s
      `Mutex<HashMap>` lookup on the root hop. ≈300 ns / op
      saving on the hot path.
- [x] **`RangeIter` fast-forward in delimiter mode** (commit
      `861dba9`) — after emitting a `CommonPrefix(C)`, ascend the
      descent stack past `C`'s subtree instead of scanning every
      leaf under it to dedup. `*_list_dir` is now
      `O(distinct_rollups)` instead of `O(leaves_under_prefix)`.
- [ ] **Sharded `BufferManager` state** — replace the single
      `Mutex<BMState>` with sharded buckets (DashMap-style) so
      concurrent pins on different blobs don't contend.

### Concurrency primitive upgrades

- [ ] **Per-node `HybridLatch`** — currently lives on
      `BlobFrame.Header` (per-blob granularity); any write in a
      blob invalidates every optimistic reader in that blob.
      Moving the latch to `NodeHeader` lets readers and writers
      of disjoint subtrees inside one blob run truly in parallel.
- [ ] **Cross-blob lock-coupling** — acquire the child blob's
      latch before stepping into it (currently happens inside
      `resolveNodePtr`). Tightens descent's visible-state window
      and prepares the ground for ext-blob latching.
- [ ] **Multi-reader stress bench** — extend `benches/main.rs`
      with a concurrent-reader scenario so the HybridLatch
      wait-free claim has a number behind it (currently the
      bench is single-threaded).

### Durability + background work

- [x] **Async checkpoint thread** — one thread,
      round-driven via [`CheckpointConfig`]. Each round
      (1) folds mergeable child blobs back into parents,
      (2) snapshots the BM dirty set + flushes WAL,
      (3) commits each blob to backend, (4) `fdatasync`s the
      backend, (5) atomically truncates the WAL when no racing
      writer has re-dirtied. Default `enabled = false`
      (opt-in until v0.3 promotes it on by default). Final
      synchronous round runs in `Checkpointer::Drop` so the
      window between the last bg round and Tree shutdown
      doesn't lose dirty cache state.
- [ ] **Free-list retry/backoff subsystem** — under heap
      exhaustion, the per-NodeType LIFO chains today fail-fast.
      The original ancestor has a backoff path
      (`free_list is {} sleep 10ms retry={}`); adding it would
      buy resilience under stress.
- [ ] **Adaptive buffer-pool eviction** — touched-recently scan,
      not strict LRU. The current LRU `VecDeque` does an O(n)
      remove on every pin's touch; with `buffer_pool_size > 128`
      this starts to bite.

### Observability

- [ ] **Metrics export** — Prometheus counters + OpenTelemetry
      traces. Hooks for: put / get / delete / rename throughput,
      spillover / merge / compact counts, cache hit rate, WAL
      flush latency, optimistic-read restart count.
- [ ] **Tracing spans** — `tracing` crate spans on `spillover`,
      `merge_blob`, `compact_blob`, WAL flush, BM evict — lets
      downstream apps correlate holt's internal events with
      their own request traces.
- [ ] **Extended `Tree::stats`** — expose free-list health
      (chains length, gap_space distribution), cache hit rate,
      latch contention counters.

### Ergonomics + diagnostics

- [ ] **Richer `Error::NodeCorrupt`** — currently carries only a
      `&'static str` context; add `blob_guid` + `slot` so a
      caller's bug-report has actionable detail without us
      needing to instrument.
- [ ] **`Tree::scan_prefix(p)` shorthand** for the common
      `tree.range().prefix(p)` case — cosmetic but tightens the
      90% query.
- [ ] **Property tests for `txn` + `range` semantics** — extend
      `tests/properties.rs`'s `HashMap` oracle to cover batch
      transactions and range queries.

### Polish

- [ ] **`cargo deny check` in CI** — wire `deny.toml` into the
      `.github/workflows/ci.yml` matrix so license drift /
      RustSec advisories fail the build.
- [ ] **PGO build profile docs** — measure + document the
      `cargo pgo` gain on `objstore_get` / `*_list`. Probably
      worth 10-15 % on hot paths.

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
- Multi-platform stability across the supported Unix targets
  (Linux + macOS, optional BSDs).
- Real production deployments + case studies.
- Long-term API stability commitment.

## Not on the roadmap

The library is **a metadata engine**, period. Single-node, embed-in-
your-process, Unix-only. Out of scope:

- **Windows support**: holt is Unix-only by design — the persistent
  backend rides `O_DIRECT` on Linux and `F_NOCACHE` on macOS, and
  there is no Windows analog this project wants to maintain. The
  crate `compile_error!`s on Windows targets.
- **Object-storage frontend / S3 layer**: the upstream that
  inspired holt's algorithm core wrapped its ART in an S3-style
  RPC server (PUT/GET/LIST inode handlers, multi-tenant bucket
  registry, RPC worker pool, distributed checkpointer with a BSS
  client). holt does **not** reproduce any of that. The
  alignment-with-upstream effort is bounded to the **metadata
  engine** (ART core, blob layout, WAL, latching, range iterator).
  TxnOp variants holt journals (`NewTree`, `RmTree`,
  `RenameObject`, `Rename`, `MemMarker`) carry the same wire
  shape as the upstream so a future RPC layer could re-use the
  format, but holt itself ships no multi-root registry, no
  bucket namespace, no RPC dispatcher, no `SplitMemOp` /
  `MergeMemOp` post-replay-ack reconciliation twins.
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
