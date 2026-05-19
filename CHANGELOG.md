# Changelog

All notable changes to **holt** are documented in this file. Format
adapted from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — I/O backend

- **`io-uring` feature flag** (Linux only). When enabled,
  `PersistentBackend::{read_blob, write_blob}` route through a
  single per-backend `io_uring` ring instead of `pread`/`pwrite`.
  Default builds + macOS / other Unix builds are unaffected.
  Behind a `Mutex<UringContext>` so concurrent callers serialise
  on the submission queue; with the new I/O worker thread the
  lock is uncontended on the hot path.

### Added — durability + background work

- **Background checkpointer** (`pub(crate) mod checkpoint` + opt-in
  `CheckpointConfig`). One thread, round-driven, parked between
  rounds via `park_timeout(idle_interval)`. Each round:
  (1) folds mergeable child blobs back into parents
  (`engine::try_merge_children`),
  (2) snapshots the `BufferManager` dirty set,
  (3) flushes the WAL writer (`sync_data`),
  (4) commits each snapshotted blob to backend,
  (5) `fdatasync`s the backend,
  (6) atomically truncates the WAL when `dirty_count == 0` under
  the WAL lock.
  `Drop` joins the bg thread then runs one final synchronous round
  on the calling thread to close the window between the last bg
  round and Tree shutdown.
- **`BufferManager` dirty-tracking** (`mark_dirty` / `snapshot_dirty`
  / `restore_dirty` / `min_unflushed_txn`). Per-blob lowest unflushed
  WAL seq, drained atomically via `mem::take` so concurrent writers'
  `mark_dirty` calls land in the fresh empty map. `commit` drains
  on success and restores on failure. The WAL trim watermark falls
  out as `min(dirty.values()) − 1`.
- **`CheckpointConfig`** + **`TreeBuilder::checkpoint(cfg)`** —
  user-facing opt-in for the background thread. Default is
  `enabled = false`; flipping it on flips on `auto_merge = true` by
  default. `idle_interval` / `dirty_blob_threshold` knobs match the
  fjall / sled flusher conventions.

### Changed — public surface

- **`engine` + `concurrency` are now `pub(crate)`** (commit
  `3cfa80f`). Their public types (`RangeBuilder` / `RangeEntry` /
  `RangeIter`) are now curated by new `api::range` and `api::stats`
  re-export modules; the top-level `holt::*` flat surface is
  unchanged from a user's perspective.
- **`api::stats`** is the canonical home for `BlobStats` /
  `TreeStats` (moved here from `api::tree`).
- **`lib.rs` re-exports grouped + commented** (Core / Range / Stats /
  Txn / Backend / Checkpoint sections).

### Removed — dead code surfaced by the lockdown

- `HybridLatch::try_upgrade`, `Guard::{state, upgrade_to_shared,
  upgrade_to_exclusive}`, `LatchMode` — no callers post-lockdown.
- `engine::walker::types::CompactStats` — `compact_blob` now returns
  `Result<()>`. Test sites that read the stats counters read
  `space_used` straight off the frame header instead.

### Performance

- **Cached `Tree.root_pin`** (commit `a6f5c78`) — every
  `get` / `put` / `delete` keeps the root pinned via
  `Arc<CachedBlob>` and skips the BM `Mutex<HashMap>` lookup on the
  root hop. ≈300 ns / op on the hot path.
- **`RangeIter` delimiter fast-forward** (commit `861dba9`) — after
  emitting a `CommonPrefix(C)`, ascend the descent stack past `C`'s
  subtree instead of scanning every leaf under it to dedup.
  `*_list_dir` is now `O(distinct_rollups)`.

### Internal

- Drop dead `journal/checkpoint.rs` stub and `engine/compact.rs`
  shim; fold `CompactReason` into `journal::txn_op`.
- Move the cross-blob descent unit test from `tests/tree_smoke.rs`
  into the walker's internal `tests` module — the
  `make_blob_from_node` primitive it pokes is now crate-private.

## [0.1.0] — 2026-05-19

First crates.io release. The v0.1 cycle was "build the engine
end-to-end" — all algorithmic + API surfaces below are landed:
ART core, multi-blob `splitBlob` / `mergeBlob` / `compactBlob`,
persistent backend (Linux `O_DIRECT` + macOS `F_NOCACHE`),
physiological WAL with batched transactions, S3-style range
iteration with delimiter rollup. 203 tests on Ubuntu + macOS
CI. See [`ROADMAP.md`](ROADMAP.md) for what's queued for v0.2.

### Added — algorithm core

- **9-NodeType ART layout** with compile-time-asserted byte offsets
  (`Leaf` 16 B, `Prefix` 128 B, `Blob` 128 B, `Node4` 24 B, `Node16`
  88 B, `Node48` 456 B, `Node256` 1032 B, `EmptyRoot` 8 B, `Invalid`).
- **4 KB `BlobHeader`** with bit-packed `SlotEntry` (`ntype << 17 |
  offset / 8`); 10 240-slot table per 512 KB blob.
- **Recursive walker**: `insert` / `lookup` / `erase` / `rename` —
  every arm cross-blob via `BlobNode` crossings.
- **`splitBlob` auto-spillover** on `OutOfSpace`; victim heuristic
  picks the largest non-`Blob` subtree under the root's first
  branching node.
- **`compactBlob` in-place repack** via deep-clone-into-scratch +
  memcpy-back; paired with `splitBlob` on every retry so churn
  workloads (insert + delete + reinsert) stay in fewer blobs.
- **`make_blob_from_node` deep-clone primitive** + `free_subtree`
  recursive slot reclaim.
- **`mergeBlob` inverse of splitBlob** — `engine::merge_blob`
  inlines a child blob's subtree back into its parent at the
  `BlobNode` slot, preserves the BlobNode's inline prefix as a
  `Prefix` chain over the inlined root, and deletes the child
  blob. `engine::is_mergeable` guards the fold (combined data
  area + slot count fit, child has no nested crossings, no
  tombstones). `engine::try_merge_children` walks a parent's
  tree and folds every direct mergeable `BlobNode` child.
  `Tree::compact` runs it after the per-blob compact pass —
  heavy-erase workloads collapse multi-blob trees back toward a
  single root.
- **`refresh_blob_node_pointers` post-compact invariant repair**
  — `compact_blob` rebuilds a child's `header.root_slot` in
  isolation, breaking the lock-step
  `BlobNode.child_entry_ptr == child.header.root_slot`
  invariant that insert / erase keep inline.
  `Tree::compact` runs `refresh_blob_node_pointers` between the
  per-blob compact pass and the merge pass to walk every
  `BlobNode` crossing and re-point it at the child's current
  root slot.
- **`SPILLOVER_RESERVATION = 128 B`** bump-area headroom so
  `spillover_blob` always has room to allocate its emergency
  `BlobNode` placeholder.
- **Cross-type free-list fallback** (`Prefix` ↔ `Blob`, both 128 B).
- **Erase-time node shrink** (Node256 → 48 → 16 → 4) with hysteresis
  thresholds 37 / 12 / 3.
- **`Node4 → Prefix([byte])` lone-child collapse** preserves descent-
  depth invariants when an inner node empties to a single child.
- **Strict-prefix support** via a Tree-layer terminator byte.
- **In-place leaf-value update on same-size writes** — zero
  allocator activity when an update fits the existing extent.
- **SIMD Node16 byte search** (SSE2 + NEON + scalar fallback).
- **SIMD `longest_common_prefix`** (SSE2 + NEON + scalar) for
  leaf-split / prefix-split hot paths.

### Added — concurrency

- **3-mode `HybridLatch`** (LeanStore: optimistic / shared /
  exclusive) wired into `CachedBlob` over an
  `UnsafeCell<AlignedBlobBuf>`.
- **Wait-free `Tree::get`** — walks every blob under an optimistic
  snapshot, restarts from the root on a torn read. No Tree-wide
  reader lock.
- **No Tree-wide writer mutex** — `put` / `delete` serialise on the
  root blob's per-blob exclusive latch; mutations on disjoint child
  blobs proceed in parallel. `rename` keeps a small `rename_lock`
  scoped to its multi-step atomicity.

### Added — buffer manager

- **`BufferManager`** — LRU-bounded blob cache wrapping any
  `Backend`, transparent (itself implements `Backend`).
  `TreeConfig::buffer_pool_size` (default 64) sets capacity.
- **`pin` / `commit` API** with the three-guard family
  (`OptimisticGuard`, `BlobReadGuard`, `BlobWriteGuard`) — pin-and-
  operate for zero-copy reads and writes against the cached buffer.

### Added — persistence

- **`MemoryBackend`** for in-memory trees and tests.
- **`PersistentBackend`** — single packed `blobs.dat` + atomic-
  rename `manifest.bin`; `O_DIRECT` on Linux, `F_NOCACHE` on macOS.
- **`AlignedBlobBuf`** — 4 KB-aligned heap buffer required by
  `O_DIRECT`.

### Added — WAL (Stage 5a-5e)

- **`TxnOp` record codec** — 10 variants (`Insert` / `Erase` /
  `Split` / `Merge` / `Compact` / `RenameObject` / `Rename` /
  `NewTree` / `RmTree` / `MemMarker`) encoded as
  `MAGIC | LEN | SEQ | TY | BODY | CRC32`.
- **CRC32 (table-driven, IEEE 802.3)** with a 256-entry compile-
  time table — ≈1.5 GB/s, ~110 ns per typical 175-byte record.
- **`WalWriter`** — append-only file with `sync_data`-on-flush
  durability boundary + 64 KB group-commit auto-flush.
- **`replay()` forward scanner** — torn-tail-tolerant; real mid-
  file corruption surfaces as `Error::ReplaySanityFailed` with the
  bad record's byte offset.
- **Tree ↔ WAL integration** — `put` / `delete` / `rename` emit
  ops; `Tree::open` replays onto the BM-cached blob; `Tree::checkpoint`
  flushes WAL + commits BM + atomically truncates the WAL.
- **Reference-based `WalWriter::append_insert` / `append_erase` /
  `append_rename_object`** fast paths — skip the three `Vec` clones
  the `TxnOp` enum's owned-data shape would force.
- **`TxnOp::Batch` + `TY_BATCH = 10`** — single WAL record carrying
  N primitive ops (`Insert` / `Erase` / `RenameObject` today);
  nested batches are rejected at encode + decode. Inner ops share
  the outer record's CRC and derive their seqs from
  `outer_seq + index` via a contiguous WAL-level seq reservation.
  Replay (`journal::reader::replay_bytes`) transparently flattens
  a `Batch` into per-inner callbacks so existing consumers don't
  need a new arm.

### Added — public API

- **`Tree::open(TreeConfig)`** — single entry point;
  `TreeConfig::new(dir)` opens persistent (default),
  `TreeConfig::memory()` is volatile.
- **`Tree::put / get / delete / rename`** — bytes-in, bytes-out.
- **`Tree::checkpoint`** — flush WAL + commit BM + truncate WAL.
- **`Tree::range()` stateful iterator** — `RangeBuilder` /
  `RangeIter` / `RangeEntry`. Builder chains `.prefix(p)`
  (anchored descent, no full-tree scan), `.start_after(k)`
  (strict-greater lower bound), `.delimiter(b)` (S3-style rollup
  with `CommonPrefix` dedup). Walks transparently across `BlobNode`
  crossings. Forward-only. Best-effort snapshot — writers can
  interleave between `next()` calls; mirrors the upstream
  `fa_iter` contract extracted from binary log strings.
- **`Tree::txn(|batch| { ... })`** — closure-based batched
  transaction. [`TxnBatch`] buffers `put` / `delete` / `rename`;
  on closure return, holt takes `rename_lock`, applies each op in
  order, then emits one `TxnOp::Batch` WAL record. Crash atomicity:
  on recovery the whole batch is either replayed or lost. Runtime
  isolation is best-effort — concurrent `put` / `delete` on
  disjoint blobs are not blocked, so the contract is "crash-
  atomic, not serialisable." Mid-batch failure (e.g., rename
  `NotFound`) leaves earlier ops applied to the BM but skips the
  WAL emit; documented on the method.
- **`TreeConfig::wal_sync_on_commit`** — opt-in per-op WAL fsync
  (default `false`, matching RocksDB's `sync=false` baseline).
- **`TreeBuilder`** — chainable config (`memory()`,
  `buffer_pool_size(n)`, `wal_sync_on_commit(bool)`,
  `checkpoint_byte_interval(b)`).
- **Typed `Error`** — `BackendIo` / `Alloc` / `Free` /
  `KeyTooLong` / `ValueTooLong` / `NotYetImplemented` /
  `NodeCorrupt` / `ReplaySanityFailed` / `NotFound` / `DstExists`.

### Added — tests / benches / examples

- **202 tests**: 117 unit + 51 tree_smoke + 15 wal_round_trip + 12
  wal_tree_integration + 2 property-based + 5 layout-invariants.
- **Property-based tests** (`proptest`) — random put / delete /
  rename traces cross-checked against a `HashMap` oracle in memory
  mode + crash-and-replay in persistent mode.
- **Criterion benchmarks** vs RocksDB across three workload shapes
  (`kv` / `objstore` / `fs`) × three ops (get / put / mixed) ×
  two storage modes (memory / persistent) = 18 microbenchmarks.
  See [`benches/README.md`](benches/README.md) for headline
  numbers.
- **Four examples**: `basic_kv`, `filesystem_meta`, `session_store`,
  `s3_metadata`. Each `cargo run --example` prints golden output.

### Added — tooling / project polish

- **GitHub Actions CI** — matrix of ubuntu + macOS × build / test /
  doctest + lint (`cargo fmt --check`, `cargo clippy -D warnings`)
  + docs (`cargo doc -D warnings`) + MSRV (1.82) build.
- **Platform scope locked**: holt is Unix-only by design. Building
  on Windows fires a top-of-crate `compile_error!`; the persistent
  backend's `O_DIRECT` (Linux) / `F_NOCACHE` (macOS) fast path has
  no Windows analog worth carrying.
- **Zero clippy / rustdoc warnings** under `-D warnings`. The
  curated `#![allow]` block in `src/lib.rs` lists the
  `clippy::pedantic` lints we've reviewed and judged either
  intentional or noise.
- **`CONTRIBUTING.md`** / **`CODE_OF_CONDUCT.md`** / this changelog.

### Notes

- The crate is pinned to MSRV **1.82**.
- License: MIT.
- v0.2 will add the 3-thread async checkpointer, `io_uring` backend
  (Linux), SIMD CRC32, and the buffer-pool tuning knobs.
