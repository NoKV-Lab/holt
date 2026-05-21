# holt — roadmap

## Where things stand

holt is a single-node, Unix-only ART-over-blobs metadata engine.
The algorithm core, persistent backend, WAL + replay, sharded
buffer manager, 3-thread background checkpointer, and the curated
public API are all in place. 260+ tests (unit + property + crash +
failpoint + multi-reader stress); zero clippy / rustdoc warnings
under `-D warnings`; ubuntu + macOS CI; `cargo deny` supply-chain
job.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and
[CHANGELOG.md](CHANGELOG.md) for what changed when. The fine-
grained shipping log lives in `git log`; this file tracks
milestones, not individual features.

## v0.1 — Usable embedded library (shipped, 2026-05-19)

Goal: build the engine end-to-end so a path-shaped-metadata
workload can use it on a single node.

Delivered: 9-NodeType ART layout pinned at compile time, recursive
walker (insert / lookup / erase / rename), cross-blob `splitBlob`
/ `mergeBlob` / `compactBlob`, `PersistentBackend` (`O_DIRECT`
Linux + `F_NOCACHE` macOS), physiological WAL with replay,
`Tree::range` iterator with prefix + start-after + S3-style
delimiter rollup, `Tree::txn` batch transactions under one WAL
record, four examples (`basic_kv` / `filesystem_meta` /
`session_store` / `s3_metadata`), property-based tests against a
`HashMap` oracle, criterion benches vs RocksDB + SQLite.

## v0.2 — Performance + concurrency upgrades (shipped)

Goal: scope the metadata-engine core for production-shaped
workloads — no new public API surface, the bench numbers from v0.1
are the success criteria.

Delivered: `io_uring` persistent-backend fast path (Linux,
feature-gated), `crc32fast` SIMD CRC32, sharded `BufferManager`
(`DashMap` replacing the v0.1 `Mutex<HashMap>` + `VecDeque` LRU),
cached `Tree.root_pin`, range-iter fast-forward in delimiter
mode, 3-thread background checkpointer (planner + I/O worker +
eviction) under a W2D-strict protocol, adaptive tick-based
eviction, observability (`Tree::stats`, structured `tracing`,
Prometheus text-format renderer behind the `metrics` feature,
silent-pin reads so scrapes don't pollute cache counters),
diagnostics (`Tree::scan_prefix`, range-iter tombstone fix,
structured `Error::NodeCorrupt`), PGO docs, `cargo deny check`
in CI, scale-curve + p95/p99 contention benches.

Concurrency primitives: per-blob `HybridLatch` (LeanStore 3-mode,
wait-free optimistic reads) inside one WAL-writer critical section
per write. Slot-versioned cross-blob lock-coupling was considered
for v0.2 but deferred to v0.3 — it needs structural changes the
v0.2 scope didn't budget for, and the per-slot version array is
best designed alongside the cross-blob descent flattening rather
than retro-fitted.

Public API surface closed before crates.io publication: the
v0.1 `pub mod layout / journal / store` exposure shipped the
on-disk layout, WAL codec, and buffer-manager guards as part of
SemVer; v0.2 tightened those to `pub(crate)` so the engine is
free to evolve internally without minor-version breaks. See
[CHANGELOG.md](CHANGELOG.md) for the supported user surface.

## v0.3 — Extreme metadata-engine performance

v0.3 is now scoped as the performance ceiling milestone for an
embedded metadata engine. Feature work waits. The goal is to push
the Fractal-ART-inspired kernel as far as practical on modern
Linux/macOS: cache-resident reads, path-shaped writes, prefix/list
walks, WAL durability, checkpoint I/O, and large-tree behavior.

The target design borrows the strongest parts of Fractal ART,
LeanStore, and modern NVMe engines while keeping holt's product
boundary narrow: one embedded library, opaque byte values, no RPC
server, no replication, no distributed object-store layer.

### P0 — Remove the write-path serial choke points (done)

v0.3's concurrency cut is implemented in the codebase:

- **Cross-blob lock coupling.** The old parent-held fallback is
  gone. `insert_multi` / `erase_multi` read the parent `BlobNode`,
  pin the child, release the parent guard, and continue in the
  child. The child blob's own `header.root_slot` is the
  authoritative entry, so child-local splits, collapses, and
  compaction do not require parent rewrites.
- **No slot-version sidecar.** The earlier per-slot version plan
  would have added ~80 KB of out-of-line state per cached blob.
  The shipped shape avoids that memory tax by removing
  `BlobNode.child_entry_ptr` and making the child header the only
  cross-blob root token.
- **Scoped walker instead of unsafe state machine.** The mutation
  walker still uses recursive Rust scopes, but it no longer
  retains ancestor guards across child mutation. That gives the
  same latch lifetime as the flattened state-machine design
  without self-referential guard plumbing.
- **Maintenance is separated.** Foreground writers enter only the
  shared side of a narrow atomic `maintenance_gate` while they may
  cross `BlobNode` boundaries. Deletes and leaf-slot churn enqueue
  blob-local compaction candidates; spillovers enqueue parent-merge
  candidates. `compact()` cold-seeds only when no hints exist,
  drains bounded candidate batches, skips clean stale candidates
  after a shared-latch header check, and `compact()` plus the
  background merge pass enter the exclusive side only around the
  parent edge currently being folded/deleted.
  Point reads also take the shared side, but blob-local access
  remains optimistic; ordinary readers and writers still run
  concurrently with each other.
- **Large-tree puts avoid root write-latch traffic.** Cross-blob
  puts route through the root under a shared latch, acquire the
  child write latch while the edge is stable, then mutate only
  from the child down. Child-only writes also return a precise
  `root_dirty = false`, so the caller does not mark the root dirty
  or take the dirty-map mutex for a blob it did not change.
- **Shape counters are exposed.** `Tree::stats()` now reports
  mutation walker ops, total/average/max blob hops, max
  cross-blob boundary depth, spillovers, merges, and optimistic
  read restarts. Prometheus export includes the same counters.
- **Checkpoint/eviction interlock is closed.** Dirty entries
  drained by a checkpoint round stay protected as in-flight
  `flushing` entries until their snapshotted bytes complete
  `write_through`, so eviction cannot drop the cache image in the
  gap between `snapshot_dirty()` and the planner's byte copy.
  Dirty / flushing / pending-delete bookkeeping is sharded by
  `BlobGuid`, and fresh spillover blobs keep a local `Arc` pin
  alive until their dirty entry is visible, so background eviction
  cannot observe a new child blob as clean before checkpoint can
  flush it.

Still intentionally not in P0: per-op latch wait histograms. The
current `HybridLatch` API has no timed acquisition boundary, and
adding timing to every latch acquire should be driven by a
contention benchmark rather than added speculatively.

### P1 — Journal group commit (done, with one explicit boundary)

Durable group commit is implemented:

- Persistent trees own a dedicated journal worker instead of an
  exposed `Arc<Mutex<WalWriter>>`.
- Writers encode complete WAL records into owned buffers, submit
  them to the worker, and wait outside the commit-publish critical
  section.
- `wal_sync_on_commit=true` writers share one `sync_data` when
  they arrive inside the short group window. `Tree::stats()` and
  Prometheus export journal appends, append batches, and sync
  counts so the batching ratio is observable.
- The old global `commit_lock` is replaced with `CommitGate`:
  foreground writers enter the shared side while they mutate blobs,
  publish dirty state, and submit journal records; checkpoint takes
  the exclusive side only while draining dirty/pending sets,
  flushing the journal, and cloning dirty blob bytes.
- This keeps the W2D proof intact while removing writer-vs-writer
  serialization from the persistent write hot path. Writers on
  disjoint child blobs now contend on per-blob latches and the
  mutation-bookkeeping shard they actually touch, not a global
  commit mutex or global dirty mutex.

### P2 — NVMe-grade checkpoint I/O

The background checkpointer already has planner / I/O / eviction
threads. v0.3 makes the I/O side worth that structure:

- Submit dirty blobs as batches, not one synchronous write at a
  time: `Backend::write_blobs` is the checkpoint write-through
  primitive.
- The default Unix backend sorts and coalesces slot-contiguous
  512 KB blob writes with `pwritev`; Linux `io_uring` keeps
  ring-depth batched SQE submission with fixed-file registration.
- Next Linux-only step: registered aligned buffers and optional
  `SQPOLL` / `IOPOLL` for direct NVMe.
- Manifest persistence now uses a base snapshot plus append-only
  `manifest.log` set/delete deltas. Checkpoint rounds append and
  fsync only the current delta batch; full `manifest.bin`
  rewrites happen only as log compaction. Deleted slots become
  reusable only after the manifest delta is durable, and reopen
  reconstructs reusable slots from final manifest holes using
  compact ranges rather than expanding sparse high-water files
  into one `u64` per free slot.
- Backends expose a conservative `needs_flush` hint. Clean manual
  checkpoints and background idle rounds skip the data-file Sync
  only when the backend has no outstanding data or manifest work,
  so a previous failed Sync still forces the next retry.
- `tests/bench_manifest_checkpoint.rs` isolates this path with
  path-shaped insert/delete/compact/checkpoint rounds and reports
  checkpoint latency percentiles plus manifest/WAL/data sizes.
- The WAL worker tracks whether the current log has uncheckpointed
  records, including records found when reopening a nonempty WAL.
  Clean checkpoints now skip empty WAL flush/truncate work instead
  of paying a `sync_data` or temp-file rename on every idle round.
  It also tracks the durable WAL frontier, so checkpoint does not
  re-fsync records that already passed through durable group
  commit. A nonempty WAL discovered on reopen is treated as
  replayable but not proven fsync-durable, so the first checkpoint
  after replay still forces a WAL flush before backend durability.
  Background checkpoint rounds also handle WAL-only truncate work
  without paying an unrelated backend Sync. WAL truncate itself is
  an in-place `ftruncate` + `sync_data`, avoiding the older
  temp-file write, rename, and fd reopen after every checkpoint.

### P3 — CPU hot-path work

Most of the structural CPU hot path is already in place:

- `SearchKey` uses a virtual terminator, so point lookups and
  mutations no longer allocate a padded key per operation.
- Same-key insert/delete comparison borrows the stored leaf key
  directly. The API allocates only when it must return the old
  value or materialise split-prefix bytes.
- SIMD is already used for Node16 byte search, longest-common
  prefix, Node48 child scans, and Node256 child scans. CRC32 uses
  `crc32fast`.
- Node bodies stay packed in 512 KB blob frames; the hot path does
  not allocate heap node objects or chase per-node boxes.

Remaining P3 work should be profile-driven only: simplify
branches that show up in the flamegraph, then consider targeted
SIMD/copy-repack kernels where profiles prove the scalar path is
still material.

### P4 — Large-tree shape control

- Recursive occupancy-aware spillover is implemented. The picker
  skips existing `BlobNode` crossings, descends inside overfull
  path branches, and chooses a victim near the target child fill
  band instead of blindly peeling off the largest direct child.
- `BlobNode` inline-prefix divergence split is implemented: a
  bad blob boundary now recovers locally into
  `Prefix? -> Node4 -> {old BlobNode, new Leaf}` instead of
  failing the insert.
- Make merge/rebalance incremental. `compact()` and background
  merge are now candidate-driven and online with respect to
  foreground writers through the atomic `maintenance_gate`; the
  remaining large-tree work is policy quality (when to
  split/merge/rebalance), not basic safety.
- Targeted large-tree shape probe is in
  `tests/bench_large_tree_shape.rs`: skewed prefixes, hot
  directories, delete-heavy churn, and a tiny-buffer-pool
  persistent read probe report blob count, space/gap/tombstones,
  spillovers, merges, and blob-hop counters.

### Deferred until after the performance core

These are useful features, but they do not define the metadata
engine's ceiling and should not compete with the v0.3 hot path:

- Full MVCC snapshots.
- Change feed / subscription API.
- Column families.
- Encryption-at-rest.
- Compression.
- OpenTelemetry bridge.

## v1.0 — Production-ready

- v0.3 feature set covered.
- Multi-platform stability across Linux + macOS (optional BSDs
  if anyone needs them).
- Real production deployments + case studies.
- Long-term API stability commitment — `holt::*` surface frozen,
  `#[non_exhaustive]` markers in place so additive changes stay
  non-breaking.

## Not on the roadmap

The library is **a metadata engine**, period. Single-node,
embed-in-your-process, Unix-only. Out of scope:

- **Windows support** — `O_DIRECT` (Linux) and `F_NOCACHE` (macOS)
  have no Windows analog this project wants to maintain. The
  crate `compile_error!`s on Windows targets.
- **Object-storage frontend / S3 layer** — the upstream that
  inspired holt's algorithm core wrapped its ART in an S3-style
  RPC server (PUT/GET/LIST inode handlers, multi-tenant bucket
  registry, RPC worker pool). holt does not reproduce any of
  that. The alignment is bounded to the **metadata engine**: ART
  core, blob layout, WAL, latching, range iterator. `TxnOp`
  variants holt journals share wire shape with the upstream so a
  future RPC layer could re-use the format, but holt itself ships
  no multi-root registry, no bucket namespace, no RPC dispatcher.
- **Replication / consensus** — build it above this. We expose
  hooks (change feed in v0.3) but don't implement Raft.
- **Network server** — this is a library. Wrap it in your gRPC /
  HTTP / whatever.
- **SQL** — not the right abstraction for this data shape.
- **Vector search** — combine with a dedicated vector DB.
- **Full-text search** — combine with Tantivy / Lucene-rs.

## Contributing

Early-stage project; design feedback most welcome. PRs welcome
too, but please open an issue first for non-trivial changes —
the architecture is still being shaped and we want to avoid
churn.
