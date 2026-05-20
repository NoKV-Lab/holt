# holt — roadmap

## Where things stand

holt is a single-node, Unix-only ART-over-blobs metadata engine.
The algorithm core, persistent backend, WAL + replay, sharded
buffer manager, 3-thread background checkpointer, and the curated
public API are all in place. 251 tests (unit + property + crash +
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
wait-free optimistic reads) inside one `wal.lock` critical section
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

### P0 — Remove the write-path serial choke points

- **Slot-versioned lock coupling.** Keep the per-blob
  `HybridLatch`, but use per-slot version counters as the
  validation token for cross-blob descent. Writers release parent
  guards before doing child work when the parent edge can be
  revalidated; mismatch returns an internal restart, mirroring
  LeanStore / Fractal ART's `OptLockNeedsRestart` shape.
- **Flatten recursive mutation walkers.** Convert insert / erase
  cross-blob recursion into an explicit blob-hop state machine.
  The walker should hold the smallest possible latch set, avoid
  ancestor retention, and make restart points obvious.
- **Separate normal mutation from structural maintenance.**
  Ordinary leaf update / insert / tombstone paths should not pay
  for split / merge / compact machinery unless they actually hit
  a full or degenerate blob.
- **Track the right counters.** Add `restart_count`,
  `max_blob_hops`, `avg_blob_hops`, `max_art_depth`,
  `spillover_count`, `merge_count`, and per-op latch wait stats.

### P1 — Real journal group commit

The current WAL batches bytes in a per-writer buffer, but every
persistent mutation still serialises through `wal.lock`. The next
shape is a dedicated journal worker:

- Writers encode records into owned buffers and enqueue
  `JournalRequest { seq, bytes, durability, notify }`.
- The journal worker batches by byte threshold and short time
  window, writes with `writev` / `pwritev`, and calls `fdatasync`
  once for all durable waiters in the batch.
- Publish dirty / pending-delete state only after the WAL record
  is admitted to the journal pipeline, preserving W2D without
  holding a global mutex around the whole walker mutation.
- Expose durability classes internally: `visible` (accepted by
  the journal worker), `durable` (fsync-completed), and
  `checkpointed` (blob image written and WAL truncate-safe).

### P2 — NVMe-grade checkpoint I/O

The background checkpointer already has planner / I/O / eviction
threads. v0.3 makes the I/O side worth that structure:

- Submit dirty blobs as batches, not one synchronous write at a
  time.
- On Linux, upgrade the `io_uring` path from
  `submit_and_wait(1)` to a real queue: larger ring depth,
  batched SQEs, CQE polling, fixed file, registered aligned
  buffers, and optional `SQPOLL` / `IOPOLL` for direct NVMe.
- Keep macOS on `F_NOCACHE` + `pwrite`, but keep the abstraction
  shaped around batched writes so Linux is not held back.
- Stop letting manifest persistence dominate checkpoint rounds:
  add slot reuse first, then evaluate an append-only manifest log
  if full rewrite shows up in profiles.

### P3 — CPU hot-path work

- Remove per-op key padding allocation. Replace `pad_key` with a
  virtual terminator or a small-stack key view.
- Make same-key leaf comparison borrow the stored key directly;
  avoid allocating a `Vec` just to compare.
- Push SIMD beyond Node16 where profiles justify it: prefix
  compare, Node48 index scan, Node256 child scan, CRC32, and
  copy/repack loops.
- Keep node bodies packed and cache-line friendly. Avoid object
  indirection or heap nodes on the hot path.
- Use PGO/LTO as a release profile, not as a substitute for
  simpler branch structure.

### P4 — Large-tree shape control

- Replace "largest non-Blob child" spillover with an
  occupancy-aware split policy. The goal is bounded blob hops and
  healthy parent/child fill ratios, not just freeing any space.
- Implement `BlobNode` inline-prefix divergence split so a bad
  blob boundary can recover into a high-fanout structure.
- Make merge/rebalance incremental and online. `compact()` should
  become a maintenance operation guarded by structure versions,
  not a quiescent-only stop-the-world pass.
- Add targeted benchmarks for skewed path prefixes, hot
  directories, delete-heavy churn, and working sets larger than
  the buffer pool.

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
