# Changelog

All notable changes to **holt** are documented in this file. Format
adapted from [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

For design background see [ARCHITECTURE.md](ARCHITECTURE.md);
fine-grained per-commit history is in `git log`.

## [Unreleased]

### Added

- Added lightweight conditional-write API for metadata-style
  compare-and-set flows: `Record`, `RecordVersion`,
  `Tree::get_record`, `Tree::get_version`,
  `Tree::put_if_absent`, `Tree::compare_and_put`, and
  `Tree::delete_if_version`. Versions are current leaf sequence
  tokens, not MVCC snapshot timestamps.
- Added conditional `AtomicBatch` operations:
  `put_if_absent`, `compare_and_put`, `delete_if_version`, and
  read-only `assert_version`.
- Added prefix-emptiness guards for metadata delete flows:
  `Tree::is_prefix_empty` for read checks and
  `AtomicBatch::assert_prefix_empty` for atomic batch preflight.
- `RangeEntry::Key` now includes the live `RecordVersion` so
  list-then-CAS metadata workflows do not need a second lookup.
- Added a `cargo-fuzz` harness (`fuzz/fuzz_targets/atomic_model.rs`)
  that checks persistent reopen/checkpoint, range scans,
  conditional writes, and atomic batches against a `BTreeMap`
  oracle.

### Changed

- Renamed the public custom storage surface around the actual
  blob-granular contract: `Backend` → `BlobStore`,
  `MemoryBackend` → `MemoryBlobStore`, `PersistentBackend` →
  `FileBlobStore`, `Tree::open_with_backend` /
  `TreeBuilder::open_with_backend` → `open_with_blob_store`,
  `Storage::Persistent` → `Storage::File`, and
  `Error::BackendIo` → `Error::BlobStoreIo`. Internally,
  `src/store/backend/persistent` is now
  `src/store/blob_store/file`.
- WAL `WalOp` is now a logical redo surface only: `Insert`,
  `Erase`, `RenameObject`, and `Batch`. Removed draft
  structural / multi-tree variants that production never emitted
  and replay previously treated as successful no-ops; their old
  draft tags now fail decode as unsupported records.
- `Tree::atomic` now returns `Result<bool>` and preflights logical
  failures before mutation. Rename errors publish no partial batch;
  failed conditional guards return `Ok(false)`.

### Fixed

- CI and release test gates no longer execute the Criterion
  benchmark harness through `cargo test --all-targets`; benchmark
  coverage stays in the dedicated bench job.

## [0.3.0] — 2026-05-21

The v0.3 milestone ships the API split, walker hot-path
optimizations, WAL format cleanup, batch-WAL encoding, recursive
cross-blob latch coupling, journal group commit, candidate-driven
online maintenance, Linux fixed-buffer `io_uring` checkpoint I/O,
and release benchmark coverage focused on metadata workloads.

The three breaking-but-surgical wins below land first; the
extreme metadata-engine performance track builds on them.

### Breaking — WAL format v3 (drops dead audit fields)

`WalOp::Insert.prev_value` and `WalOp::Erase.value` are gone.
Both were documented as "for replay reversibility" but replay is
an idempotent forward redo that only consumes `(key, value)` for
Insert and `key` for Erase — the prior-value slots were dead
weight on every returning `Tree::insert` / `Tree::remove` (the
blind variants already wrote `None`).

**File format version bumped 2 → 3.** A binary built from the
earlier v0.3 draft opening a v3 WAL fails with
`Error::ReplaySanityFailed` /
`"WAL file format version unsupported"` rather than mis-parsing
the absent optional-bytes slot as a length prefix. **Upgrade
path for local data written by the earlier v0.3 draft:
`Tree::checkpoint()` before swapping in the new binary** —
checkpoint truncates the WAL to header-only, so the next open
writes a v0.3 (= v3-format) header.

Concretely:
- WAL `Insert` record body shrinks by one `optional_bytes` slot
  (`u8` presence tag + `u32` length + `prev_value.len()` bytes
  on the returning path).
- WAL `Erase` record body shrinks by one `optional_bytes` slot
  (same shape as Insert).
- Returning `Tree::insert` / `Tree::remove` no longer clone or
  serialise the prior value into the WAL buffer; the caller still
  receives it through the return path (walker `leaf_extent` read,
  same as the earlier v0.3 API split).

### Breaking — public lookup surface stays owned

The draft `Tree::get_with` closure API was removed before v0.3.
The engine still uses an internal zero-copy lookup walker
for existence probes and rename preflight, but the public API stays
small:

```rust
pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
```

Reason: a closure borrowing live cache bytes is a lifetime and
contention contract, not just a convenience method. For the
metadata-engine surface we want to stabilize now, owned `Vec<u8>`
lookups plus range iteration are the right external boundary.

### Performance — `atomic()` batch bypasses `WalOp` enum

`Tree::atomic` (the batched-multi-op API) now uses a streaming
`BatchEncoder` that writes inner-op bytes directly from `&[u8]`
refs into the WAL pending buffer. The previous v0.3 draft path
constructed `WalOp::Insert { key: Vec<u8>, value: Vec<u8>, .. }`
(and similarly for `Erase` / `RenameObject`) per inner op, then
encoded the enum-of-`Vec`s — two extra clones per op the WAL never
needed.

Mid-batch error handling is preserved: if a walker call returns
`Err` partway through, the encoder's `Drop` rolls the partial
record bytes off the WAL pending buffer (truncate to where
`begin` started). On `Ok`, the encoder backpatches the inner
count + body length and appends the CRC — byte-identical to
what the old `encode_record(&WalOp::Batch { .. })` path would
have produced (verified by `batch_encoder_wire_matches_encode_record`
in `src/journal/codec.rs::tests`).

### Performance — rename's wasted `wants_prev`

`Tree::rename`'s two walker calls (`erase_multi` on src,
`insert_multi` on dst) were passing `wants_prev=true` even
though the rename already read the src value in the pre-walker
lookup and the dst-existence check (or `force=true`) gates the
insert side. The walker-materialised previous values were
dropped on the floor. Flipped to `wants_prev=false` on both;
same change applied inside the batch-path rename arm of
`apply_batch_walker_inline`.

### Internal — codec cleanup

- Dropped `write_optional_bytes` / `read_optional_bytes` helpers
  (no callers after Insert/Erase shed their optional slots).
- `WalWriter::append` (the generic `&WalOp` entry) is now
  test-only. Foreground mutation paths encode owned WAL records
  with the per-variant codec helpers and hand those bytes to the
  journal worker.
- `apply_put_inner` / `apply_delete_inner` / `apply_rename_inner`
  Tree helpers folded into the new `apply_batch_walker_inline`.

### Breaking — BlobNode format + cross-blob latch coupling

Implements the first real concurrency cut for split metadata
trees and makes the on-disk `BlobNode` contract smaller. `put` /
`insert` and `delete` / `remove` no longer hold a root or
ancestor blob's exclusive latch while mutating a descendant blob
on the key path. The walker reads the `BlobNode`, pins the child
blob, releases the parent, and repeats that handoff recursively
for every deeper `BlobNode`.

The correctness boundary also changes: parent
`BlobNode.child_entry_ptr` is removed. The child blob's own
`header.root_slot` is the only cross-blob entry. Child-local
splits, collapses, and post-compact root-slot changes therefore
do not require re-acquiring and rewriting the parent `BlobNode`.

Concretely:

- `BlobNode` stays 128 bytes, but the removed child-entry field
  is reclaimed for inline prefix payload. `BLOB_MAX_INLINE`
  grows from 96 to 104 bytes.
- This is an on-disk format break for multi-blob trees written by
  earlier v0.3 draft binaries. Rebuild from checkpointed data or
  export/import before moving existing draft data onto this
  layout.
- `BlobWriteGuard::frame()` remains the writer-side convenience
  for in-place mutation, but it now returns a plain
  `BlobFrame::wrap(...)`; there is no per-slot version sidecar.
- `walker::insert::insert_multi` first tries the recursive
  lock-coupled path. If the root blob is full, `spillover_blob`
  + `compact_blob` run as before, then the walker re-checks the
  key path; if the target subtree moved behind a new `BlobNode`,
  the retry immediately hands off to the child instead of holding
  the root latch across child mutation.
- `walker::erase::erase_multi` mirrors the same recursive
  handoff. The lock-coupled delete path leaves an emptied child
  blob reachable with an `EmptyRoot` sentinel; structural pruning
  remains a maintenance concern rather than a foreground delete
  latch chain.
- `walker::lookup`, `range`, and merge paths follow
  `child.header.root_slot` when crossing blobs, so parent
  BlobNodes never need child-entry repair after child compaction.
- The old recursive `insert_at_blob_node` /
  `erase_at_blob_node` parent-held fallback arms are removed.

### Performance / Correctness — online maintenance gate + shape counters

The remaining v0.3 concurrency cleanup is now in place:

- Foreground mutation paths (`put` / `insert` / `delete` /
  `remove` / `rename` / `atomic`) enter the shared side of a narrow
  atomic `maintenance_gate` while they may cross `BlobNode`
  boundaries.
  `Tree::compact()` runs blob-local compaction on the shared side,
  skips clean stale candidates after a shared-latch header check,
  and both manual compact plus the background checkpointer's merge
  pass enter the exclusive side only around one parent
  merge/delete window at a time.
- Online maintenance is now candidate-driven. Deletes and
  leaf-slot churn enqueue blob-local compaction candidates;
  spillovers enqueue parent-merge candidates. `Tree::compact()`
  cold-seeds the queues only when no hints exist, then drains a
  bounded batch instead of sweeping every blob on every call.
  Background auto-merge drains the same queued parent candidates,
  and too-large parent candidates are consumed until fresh shape
  debt requeues them, so idle checkpoint rounds avoid a large-tree
  merge scan.
- Point reads (`get`) also take the shared
  maintenance gate so a merge cannot delete a child after a reader
  observes the parent `BlobNode`. Blob-local reads still use
  per-blob optimistic validation; ordinary readers and writers
  remain mutually concurrent.
- `Tree::compact()` is no longer documented as quiescent-only.
  It is safe against active point reads and foreground writers;
  range iterators remain best-effort snapshots because they keep a
  raw `(blob_guid, slot)` stack across calls.
- `Tree::stats()` and `holt::metrics::render_prometheus` now
  expose walker/shape counters: mutation walker ops, total and
  average blob hops, max blob hops, max cross-blob boundary depth,
  foreground spillovers, and child-blob merges.
- Cross-blob `put` no longer takes the root's exclusive latch just
  to read a `BlobNode`. The root is held shared while the child
  write latch is acquired, then the mutation proceeds from the
  child down. Cross-blob updates also return a precise
  `root_dirty` bit so child-only writes do not mark the root dirty
  or take the dirty-map mutex for an unchanged blob.
- Checkpoint dirty snapshots now move drained blobs into an
  in-flight `flushing` protection set until `write_through`
  completes. Eviction skips both live dirty and flushing entries,
  closing the pressure-window where a background sweep could drop
  the only cached image after `snapshot_dirty()` drained the dirty
  map but before the checkpoint planner copied the bytes.
- Dirty / flushing / pending-delete bookkeeping is sharded by
  `BlobGuid` (64 shards). `mark_dirty` and `mark_for_delete` now
  take one per-guid shard lock instead of one global dirty mutex,
  which removes the next persistent-write contention point after
  `CommitGate`.
- Fresh spillover blobs keep a local `Arc` pin alive until their
  dirty entry is published. This closes the complementary I1
  window where a background eviction sweep could see a just-created
  child blob as clean, remove its cache image, and leave checkpoint
  with a dirty entry but no bytes to flush, without introducing a
  mutation-shard / cache-shard lock-order inversion.

### Performance / Correctness — journal group commit

- Persistent trees now own a dedicated `Journal` worker instead of
  sharing `Arc<Mutex<WalWriter>>` directly.
- Foreground writers encode a complete WAL record into owned bytes,
  enter the writer-shared `CommitGate` only for walker mutation +
  dirty publish + journal submission, then wait for the journal
  acknowledgement outside that gate.
- `WalCommit::Sync` writers are batched by a short group
  window; the journal worker appends every queued record and calls
  `sync_data` once for all sync waiters in the batch.
- Manual and background checkpoint rounds use the same
  `CommitGate` on its exclusive side while draining dirty/pending
  sets, flushing the journal, and cloning snapshotted bytes. This
  prevents checkpoint I/O from copying bytes from a writer whose
  WAL record was not in the durable snapshot without serialising
  ordinary writers against each other.
- `Tree::stats()` / Prometheus metrics expose journal appends,
  append batches, and sync counts.
- Short-key padding now uses the 256-byte inline path without
  clearing the full stack buffer on every operation, and tree
  sequence allocation uses relaxed atomics because ordering is
  provided by WAL/dirty/latch synchronization rather than by the
  counter itself.

### Breaking — API redesign (split returning from blind)

The v0.2 `put` / `delete` returned `Option<Vec<u8>>` by default,
forcing every caller to pay the read-old-leaf + value-clone cost
even when the prior value wasn't needed. This worked but
contradicted the "metadata hot path" design goal — for a storage
engine, the HashMap-style "give me the old value for free" contract
is anything but free. Aligned with RocksDB / LevelDB / FoundationDB
convention: blind by default, returning by explicit opt-in.

The new surface:

```rust
// blind hot paths — no leaf-extent value read
put(&self, k: &[u8], v: &[u8]) -> Result<()>
delete(&self, k: &[u8]) -> Result<bool>

// returning variants — pay the read + clone explicitly
insert(&self, k: &[u8], v: &[u8]) -> Result<Option<Vec<u8>>>
remove(&self, k: &[u8]) -> Result<Option<Vec<u8>>>
```

Migrating from v0.2.x:
- `tree.put(k, v).unwrap()` → unchanged (returns `()` now; `.unwrap()` works the same).
- `let prev = tree.put(k, v).unwrap();` → `let prev = tree.insert(k, v).unwrap();`
- `tree.delete(k).unwrap().is_some()` → `tree.delete(k).unwrap()` (already a `bool`).
- `let prev = tree.delete(k).unwrap();` → `let prev = tree.remove(k).unwrap();`

### Breaking — WAL format

`WalOp::Erase.value` changed from `Vec<u8>` (always present) to
`Option<Vec<u8>>`: `Some(prev)` on the returning `Tree::remove`
path, `None` on the blind `Tree::delete` path. Wire shape: the
trailing `bytes(value)` became `optional_bytes(value)`.

**File format version bumped 1 → 2.** A v0.3 binary opening a
v0.2 WAL fails with `Error::ReplaySanityFailed` /
`"WAL file format version unsupported"` rather than mis-decoding.
**Upgrade path: run `Tree::checkpoint()` on the v0.2 tree
before swapping in the v0.3 binary** — checkpoint truncates the
WAL to header-only, so the next open writes a v0.3 header.

### Performance — walker hot-path optimizations

The walker now threads a `wants_prev: bool` flag through
`insert_at` / `erase_at` and all their arms. Concrete savings on
the blind path:

- **`read_leaf_key_only`** (new helper): same-key check reads
  only the leaf's key bytes, not value. Saves a per-op
  `value_size`-byte clone on every same-key `put` / `delete`.
- **`insert_into_prefix` + `erase_at_prefix` borrow-only
  descent**: `Prefix` is `Copy` so `let p = read_prefix(...)`
  is an owned stack value; the inline prefix bytes can be held
  via `&p.bytes[..plen]` across the subsequent `frame.*`
  mutations without the previous `.to_vec()` allocation. Hot on
  path-shaped workloads (objstore / fs) where prefix chains are
  long.
- **WAL `Insert.prev_value` encoded as `None`** on blind put;
  **WAL `Erase.value` encoded as `None`** on blind delete. Both
  skip the `Vec` clone + bytes copy that the v0.2.x always-encoded
  path paid.

Linux v0.3 release run after the cross-blob latch-coupling and
`BlobNode` format break:
- **kv put @ 2 M**: 1 866 ns (vs RocksDB 2 001 ns, SQLite 2 336 ns).
- **objstore put @ 2 M**: 1 707 ns (vs RocksDB 1 994 ns, SQLite 2 222 ns).
- **fs put @ 2 M**: 1 796 ns (vs RocksDB 1 969 ns, SQLite 2 199 ns).

At 2 M vs RocksDB: kv is **1.07×** ahead, objstore is **1.17×**
ahead, and fs is **1.10×** ahead. Full table in
[benches/RESULTS.md](benches/RESULTS.md). Point writes are now
competitive at large scale; the release headline remains
metadata-native mixes and delimiter directory rollup.

### Changed — internal types

- **`EraseOutcome` and the walker-internal `EraseReturn`** gain a
  `mutated: bool` field separate from `previous: Option<Vec<u8>>`.
  `mutated` is the authoritative "did the walker tombstone a
  leaf" signal regardless of whether the caller asked for the
  prior value; previously this was inferred from
  `previous.is_some()`, which conflated "no mutation" with
  "blind erase".

### Internal

- `BufferManager` and other crate-private types unchanged in
  shape; only the walker entry-point signatures and WAL codec
  changed.

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
  `BlobStats`, `TreeStats`, `CheckpointerStats`, `AtomicBatch`,
  `CheckpointConfig`, `BlobStore`, `MemoryBlobStore`,
  `FileBlobStore`, `AlignedBlobBuf`, `BlobGuid`. The
  `metrics::render_prometheus` renderer is part of the
  `metrics`-feature surface.
- **`pub use holt::BufferManager` removed**; `BufferManager` is
  internal.
- **`BlobGuid` now re-exported at the crate root** for custom
  `BlobStore` implementations.
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
- **Writer ↔ background-checkpoint W2D race.** Dirty and
  pending-delete snapshots now drain inside the same
  commit-publish critical section as journal flush and byte
  snapshotting, closing both the pending-delete inversion window
  and the "snapshot cloned a newer unflushed mutation" window.
- **Parent-side BlobNode pointer repair removed.** Compact keeps
  each child blob self-describing through `header.root_slot`; the
  parent no longer stores a child entry slot and therefore no
  post-compact repair pass is needed.
- **`Tree::compact` documented `NOT online-safe`** — running
  concurrently with reads or writes can torn-read across
  `BlobNode` crossings. Future online-maintenance work needs
  structure-version protection before this can run with traffic.

### Added

- **`io-uring` feature flag** (Linux only). `FileBlobStore`
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
  well on all three workloads (holt wins every cell with the lead
  vs RocksDB widening to 6.4× / 3.3× / 3.0× at 2 M).
  **Put** now wins every point in the current scale-put run;
  the hard cell is 2 M kv put, where holt is only **1.07×**
  ahead of RocksDB and should be treated as parity rather than a
  decisive write win.
- **Metadata-native release claim.** `objstore_metadata_mix` is
  43× faster than RocksDB and 29× faster than SQLite;
  `fs_metadata_mix` is 66× / 53× faster. `objstore_list_dir` is
  151× / 139× faster and `fs_list_dir` is 268× / 244× faster.
- **PGO build profile docs** in [`PGO.md`](PGO.md).

Full numbers in [`benches/RESULTS.md`](benches/RESULTS.md).

## [0.1.0] — 2026-05-19

First crates.io release. The v0.1 cycle built the engine end-to-
end on a single Unix-only stack: ART core, multi-blob `splitBlob`
/ `mergeBlob` / `compactBlob`, `FileBlobStore` (`O_DIRECT`
Linux + `F_NOCACHE` macOS), logical WAL with replay,
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
- Persistent `put` / `delete` / `rename` / `atomic` publish dirty
  state and journal records through writer-shared `CommitGate`;
  durable fsync waits happen outside that gate through the
  group-commit worker.
  `rename` keeps a separate `rename_lock` for its multi-step
  atomicity.

### Persistence

- `MemoryBlobStore` and `FileBlobStore` (single packed
  `blobs.dat` + atomic-rename `manifest.bin`, `O_DIRECT` Linux,
  `F_NOCACHE` macOS).
- `BlobStore` trait + `AlignedBlobBuf` 4 KB-aligned zero-copy
  buffer.
- 10-variant `WalOp` codec (`MAGIC | LEN | SEQ | TY | BODY |
  CRC32`); torn-tail-tolerant forward replay scanner.
- `WalWriter` with `sync_data`-on-flush durability + 64 KB
  buffered auto-drain, driven by a dedicated journal
  group-commit worker in persistent trees.
- `Tree::checkpoint` flushes WAL + commits BM + truncates WAL
  conditionally; replay reapplies records onto the BM-cached
  blob and resumes `next_seq` past every replayed record.
- `WalOp::Batch` (`TY_BATCH = 10`) carries N primitive ops under
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
- `Tree::atomic(|batch| { ... })` — batched mutations under one
  `WalOp::Batch` WAL record. Crash-atomic, runtime isolation is
  best-effort.
- `Tree::checkpoint()`, `Tree::stats()`.
- Typed `Error` (`BlobStoreIo` / `Alloc` / `Free` / `KeyTooLong`
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
