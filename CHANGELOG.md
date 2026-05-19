# Changelog

All notable changes to **holt** are documented in this file. Format
adapted from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

For design background see [ARCHITECTURE.md](ARCHITECTURE.md);
fine-grained per-commit history is in `git log`.

## [0.2.1] — 2026-05-20

### Fixed — durability (silent data loss path)

- **`BufferManager::try_evict_lru` was evicting dirty cache
  images.** The inline-overflow eviction picked victims based on
  `Arc::strong_count == 1` alone — it did not check the dirty
  map. A blob that had been mutated (`pin → write → mark_dirty →
  drop pin`) could be picked as a victim by the next cache-miss
  load, leaving the dirty entry orphaned (cache image gone, dirty
  map still pointing at the now-missing guid). Downstream the
  next checkpoint's `snapshot_bytes(guid)` returned `None` and
  the round / `Tree::checkpoint` silently `continue`-d past it;
  in memory mode the cache mutation was lost outright, in
  persistent mode the WAL truncate gate stuck closed forever
  (dirty_count never reached zero).

  `try_evict_lru` now matches `try_evict_cold`'s contract: skip
  any entry whose guid is in `dirty` or `pending_deletes`. Both
  the victim-selection loop and the `remove_if` predicate
  re-check under the relevant lock, guarding against a fresh
  `mark_dirty` landing between scan and remove.

- **Checkpoint paths no longer silently drop a missing cache
  image.** `Tree::checkpoint` and the background round's phase 2
  used to `if let Some(bytes) = snapshot_bytes(guid) { ... }`
  and silently fall through on `None`. They now treat that case
  as the invariant-I1 violation it is: restore both drained
  snapshots and return `Error::Internal("checkpoint: dirty
  entry lost cache image — invariant I1 violated")`. Better to
  fail loud than truncate the WAL while data is still pending.

- Regression test: `lru_eviction_skips_dirty_entries` in
  `src/store/buffer_manager.rs` exercises capacity-2 cache with
  one dirty + one clean entry, asserts the clean entry is the
  victim of inline overflow and the dirty cache image survives.

### Internal

- `release.yml`: dropped the `release-notes/v$VERSION.md`
  curated-note branch — CHANGELOG is now the single source for
  GitHub Release body content.

## [0.2.0] — 2026-05-20

### Breaking

- **Public API surface closure.** `holt::layout`, `holt::journal`,
  `holt::store` are now `pub(crate)`. The supported `holt::*`
  surface is `Tree`, `TreeBuilder`, `TreeConfig`, `Storage`,
  `Error`, `Result`, `RangeBuilder`, `RangeEntry`, `RangeIter`,
  `BlobStats`, `TreeStats`, `CheckpointerStats`, `TxnBatch`,
  `CheckpointConfig`, `Backend`, `MemoryBackend`,
  `PersistentBackend`, `AlignedBlobBuf`, `BlobGuid`. The
  `metrics::render_prometheus` renderer is part of the
  `metrics`-feature surface.
- **`pub use holt::BufferManager` removed**; `BufferManager` is
  internal.
- **`BlobGuid` now re-exported at the crate root** for custom
  `Backend` implementations.
- **`RangeBuilder::new` is `pub(crate)`** — use `Tree::range()` /
  `Tree::scan_prefix()`.
- **`TreeConfig::checkpoint_byte_interval` field +
  `TreeBuilder::checkpoint_byte_interval` method removed.** The
  field was reserved and never read.
- **`AllocOutcome` shrunk to `{ slot }`; `ExtentAllocOutcome`
  shrunk to `{ byte_offset }`.** The other fields were dead.
- **`encode_record` returns `()` instead of `Result<()>`** — no
  fallible step.
- **`BufferManager::capacity()` / `clear()` removed.** Dead code.
- **`TreeConfig::flush_on_write` renamed to
  `memory_flush_on_write`** — the field had no effect on
  persistent trees; the v0.1 name suggested per-write fsync, which
  it never was.
- **`Error::NodeCorrupt` is a struct variant with optional
  `blob_guid` + `slot` fields.** Construct via
  `Error::node_corrupt(ctx)` and enrich via `.with_blob_guid(g)`
  / `.with_slot(s)`. Pattern-matchers must spread the new fields
  (`NodeCorrupt { context, .. }`).

### Fixed — durability (W2D-strict)

- **Checkpoint error paths no longer drop drained state.** Manual
  `Tree::checkpoint` and the background round now restore every
  snapshot they drained on every error return — WAL flush
  failure, I/O worker channel-closed, and pre-delete `Sync`
  failure paths previously left `dirty` / `pending` partially
  drained, allowing the next round to truncate the WAL with cache
  state still pending. See ARCHITECTURE.md §6 for the seven-phase
  protocol.
- **Abort-on-dirty-failure gate before pending-delete.** A failed
  parent `write_through` no longer propagates to the dependent
  child's manifest delete (which would have left the on-disk
  parent referencing a slot the manifest no longer had). The pre-
  delete sync still runs to fsync the writes that did succeed;
  the pending set is restored and the next round retries the
  parent + child together.
- **Writer ↔ background-checkpoint W2D race.** Pending-delete
  snapshot now drains inside the same `wal.lock` critical section
  as `snapshot_dirty` + `wal.flush`, closing the inversion window
  where a writer could land a fresh blob between the two drains.
- **`scan.rs::refresh_blob_node_pointers` inline `bm.commit`**
  replaced with `bm.mark_dirty(parent_guid, STRUCTURAL_SEQ)` so
  the post-compact pointer repair stages through the unified
  dirty-set protocol instead of pushing cache state straight to
  backend.
- **`Tree::compact` documented `NOT online-safe`** — running
  concurrently with reads or writes can torn-read across
  `BlobNode` crossings. The v0.3 maintenance latch will lift this.

### Added

- **`io-uring` feature flag** (Linux only). `PersistentBackend`
  reads/writes route through a per-backend `io_uring` (depth 8)
  instead of `pread`/`pwrite`.
- **`tracing` feature flag** (off by default). Structured
  `tracing` events on `checkpoint` round complete, `spillover`,
  `merge`, `compact`, WAL truncate, and eviction sweeps. Zero-
  cost when the feature is off.
- **`metrics` feature flag** (off by default). Renders
  `TreeStats` into Prometheus text format. Gauges
  (`holt_slots`, `holt_tombstones`, `holt_compactions`) follow
  the convention of dropping the `_total` suffix.
- **3-thread background checkpointer** — planner + dedicated I/O
  worker + cold-blob eviction sweep, parked between rounds via
  `park_timeout(idle_interval)`. Default disabled; opt in via
  `TreeBuilder::checkpoint(CheckpointConfig::default()
  .enabled(true))`. `Drop` runs one final synchronous round on
  the calling thread.
- **`Tree::scan_prefix(p)`** — one-line wrapper for
  `tree.range().prefix(p)`.
- **`Tree::stats` extended** with `bm_dirty_count`,
  `bm_pending_delete_count`, `bm_cache_hits` / `bm_cache_misses`,
  `bm_optimistic_restarts`, and an `Option<CheckpointerStats>`.
- **Silent observability reads** — `pin_silent` /
  `get_cached_silent` / `collect_blob_guids_silent` don't bump
  cache counters or refresh the LRU tick, so `Tree::stats` and
  metrics scrapes don't pollute the counters they report.
- **`Error::Internal(&'static str)`** variant for invariant-
  violation paths (previously `Error::NotYetImplemented`, now
  reserved for genuine walker-arm feature gaps). Non-breaking
  thanks to `Error`'s `#[non_exhaustive]` marker.

### Changed

- **Sharded `BufferManager` cache** — v0.1's
  `Mutex<HashMap<BlobGuid, _>>` + `VecDeque<BlobGuid>` LRU
  replaced by `DashMap<BlobGuid, Arc<CachedBlob>>` with
  `clock_tick` / `last_touched` eviction; concurrent pins on
  different blobs hit different shards instead of contending on
  a single mutex.
- **Cached `Tree.root_pin`** — every `get` / `put` / `delete`
  keeps the root pinned via `Arc<CachedBlob>` and skips the BM
  hash lookup on the root hop (~300 ns/op on the hot path).
- **`RangeIter` delimiter fast-forward** — after emitting a
  `CommonPrefix(C)`, ascend the descent stack past `C`'s subtree
  instead of scanning every leaf. `*_list_dir` is now
  `O(distinct_rollups)`.
- **Hardware-accelerated CRC32** via `crc32fast` — auto-detects
  PCLMULQDQ on x86_64 and ARM-CRC32 on AArch64. Drops per-record
  WAL cost from ~110 ns to ~20 ns on supported hardware.
- **SIMD Node48 / Node256 range-iter scans** — `vpcmpeqb` / NEON
  byte search for `Node48::index[256]`, slot-index scan for
  `Node256::children[256]`. Worth ~80-120 ns per `next()` on
  wide branch nodes; matters most for `*_list_dir`.

### Benchmarks

- **Group B — scale curve** across kv / objstore / fs × four
  dataset sizes (`{ 20 k, 100 k, 500 k, 2 M }`). The 500 k tier
  already exceeds the default 32 MB buffer pool; the 2 M tier
  (~192 MB payload) forces full eviction churn. **Get** scales
  beautifully on all three workloads (holt wins every cell with
  the lead vs RocksDB widening to 5.4× / 2.8× / 2.2× at 2 M).
  **Put** wins at 20 k / 100 k / 500 k, ties RocksDB at 2 M kv,
  but loses 8-22 % to RocksDB / SQLite at 2 M on objstore / fs
  — the regime where LSM-style write amortization is the right
  choice and ART-over-blobs isn't competitive; cross-blob lock-
  coupling is queued for v0.3 to close the gap.
- **Group C — p95/p99 under maintenance interference**
  (`tests/bench_contention_p95.rs`, `#[ignore]`). 4 writer
  threads + 5 ms-cadence background checkpointer + concurrent
  `Tree::compact()`; tracks every `put` latency via
  `hdrhistogram`. M3 Pro: 307k ops/s sustained, p50 = 2 µs,
  p99 = 108 µs.
- **PGO build profile docs** in [`PGO.md`](PGO.md).

Full numbers in [`benches/RESULTS.md`](benches/RESULTS.md).

## [0.1.0] — 2026-05-19

First crates.io release. The v0.1 cycle built the engine end-to-
end on a single Unix-only stack: ART core, multi-blob `splitBlob`
/ `mergeBlob` / `compactBlob`, `PersistentBackend` (`O_DIRECT`
Linux + `F_NOCACHE` macOS), physiological WAL with replay,
S3-style range iteration with delimiter rollup. 203 tests on
ubuntu + macOS CI.

### Algorithm core

- 9-NodeType ART layout (`Leaf` 16 B, `Prefix` 128 B, `Blob`
  128 B, `Node{4,16,48,256}`, `EmptyRoot` 8 B). Every field
  offset pinned at compile time via `offset_of!` asserts.
- 4 KB `BlobHeader` + bit-packed `SlotEntry`
  (`ntype << 17 | offset / 8`); 10 240-slot table per 512 KB
  blob.
- Recursive walker (insert / lookup / erase / rename) crossing
  blobs transparently via `BlobNode`.
- `splitBlob` in-band spillover, `compactBlob` in-place repack,
  `mergeBlob` inverse fold (with `is_mergeable` guard +
  `refresh_blob_node_pointers` post-compact invariant repair).
- 128 B `SPILLOVER_RESERVATION` + `Prefix` ↔ `Blob` cross-type
  free-list fallback — spillover can always install its
  emergency BlobNode.
- Erase-time node shrink (Node256 → 48 → 16 → 4 at hysteresis
  thresholds 37 / 12 / 3) + terminal lone-child
  `Node4 → Prefix([byte])` collapse.
- In-place leaf-value update on same-size writes — zero allocator
  activity.
- SIMD `node16_find_byte` (SSE2 + NEON + scalar) and SIMD
  `longest_common_prefix` for leaf-split / prefix-split hot
  paths.

### Concurrency

- 3-mode `HybridLatch` (LeanStore: optimistic / shared /
  exclusive) wired into `CachedBlob` over
  `UnsafeCell<AlignedBlobBuf>`.
- Wait-free `Tree::get` walker — optimistic snapshots with
  validate-after, restart from root on torn read. No Tree-wide
  reader lock.
- `put` / `delete` serialise on `wal.lock` (not a global writer
  mutex); `rename` keeps a separate `rename_lock` for its
  multi-step atomicity.

### Persistence

- `MemoryBackend` and `PersistentBackend` (single packed
  `blobs.dat` + atomic-rename `manifest.bin`, `O_DIRECT` Linux,
  `F_NOCACHE` macOS).
- `Backend` trait + `AlignedBlobBuf` 4 KB-aligned zero-copy
  buffer.
- 10-variant `TxnOp` codec (`MAGIC | LEN | SEQ | TY | BODY |
  CRC32`); torn-tail-tolerant forward replay scanner.
- `WalWriter` with `sync_data`-on-flush durability + 64 KB
  group-commit auto-flush.
- `Tree::checkpoint` flushes WAL + commits BM + truncates WAL
  conditionally; replay reapplies records onto the BM-cached
  blob and resumes `next_seq` past every replayed record.
- `TxnOp::Batch` (`TY_BATCH = 10`) carries N primitive ops under
  one record with shared CRC and derived seqs; replay
  transparently flattens to per-inner callbacks.

### Public API

- `Tree::open(TreeConfig)` single entry, `TreeBuilder` chainable
  config.
- `Tree::put / get / delete / rename` (cross-blob via
  `lookup_multi` / `insert_multi` / `erase_multi`).
- `Tree::range()` stateful iterator — `.prefix(p)`,
  `.start_after(k)`, `.delimiter(b)` (S3-style rollup with
  `CommonPrefix` dedup). Forward-only, best-effort snapshot.
- `Tree::txn(|batch| { ... })` — batched mutations under one
  `TxnOp::Batch` WAL record. Crash-atomic, runtime isolation is
  best-effort.
- `Tree::checkpoint()`, `Tree::stats()`.
- Typed `Error` (`BackendIo` / `Alloc` / `Free` / `KeyTooLong`
  / `ValueTooLong` / `NotYetImplemented` / `NodeCorrupt` /
  `ReplaySanityFailed` / `NotFound` / `DstExists`).
  `#[non_exhaustive]` so new variants are non-breaking in minor
  releases.

### Tests + benches + tooling

- 202 tests: unit + property (`proptest` vs `HashMap` oracle, in
  memory and crash-and-replay persistent modes) + multi-reader
  stress + multi-blob auto-spillover end-to-end.
- Criterion benches vs RocksDB across three workload shapes
  (`kv` / `objstore` / `fs`) × get / put / mixed × memory /
  persistent.
- Four examples: `basic_kv`, `filesystem_meta`, `session_store`,
  `s3_metadata`.
- GitHub Actions CI matrix (ubuntu + macOS) × build / test /
  doctest + lint (`cargo fmt`, `cargo clippy -D warnings`) +
  docs (`cargo doc -D warnings`) + MSRV (1.82).
- Windows targets fire a top-of-crate `compile_error!` — the
  `O_DIRECT` / `F_NOCACHE` fast paths have no Windows analog
  worth maintaining.
- MIT license, MSRV pinned to Rust 1.82.
