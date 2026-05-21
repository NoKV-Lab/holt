# holt — roadmap

## Where things stand

holt is a single-node, Unix-only ART-over-blobs metadata engine.
The algorithm core, persistent backend, WAL + replay, sharded
buffer manager, 3-thread background checkpointer, and the curated
public API are all in place. The release test surface is unit,
property, crash/replay, failpoint, and integration coverage; the
public benchmark surface is the RocksDB/SQLite metadata comparator
in `benches/main.rs`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and
[CHANGELOG.md](CHANGELOG.md) for what changed when. The fine-
grained shipping log lives in `git log`; this file tracks
milestones, not individual features.

## v0.1 — Usable embedded library (shipped, 2026-05-19)

Goal: build the engine end-to-end so a path-shaped-metadata
workload can use it on a single node.

Delivered: 9-NodeType ART layout pinned at compile time, recursive
walker (insert / lookup / erase / rename), cross-blob `splitBlob`
/ `mergeBlob` / `compactBlob`, `FileBlobStore` (`O_DIRECT`
Linux + `F_NOCACHE` macOS), logical WAL with replay,
`Tree::range` iterator with prefix + start-after + S3-style
delimiter rollup, `Tree::atomic` atomic batches under one WAL
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

## v0.3 — Extreme metadata-engine performance (shipped, 2026-05-21)

v0.3 shipped as the performance ceiling milestone for an
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

### P2 — NVMe-grade checkpoint I/O (done)

The background checkpointer already has planner / I/O / eviction
threads. v0.3 makes the I/O side worth that structure:

- Submit dirty blobs as batches, not one synchronous write at a
  time: `BlobStore::write_blobs` is the checkpoint write-through
  primitive.
- The default Unix backend sorts and coalesces slot-contiguous
  512 KB blob writes with `pwritev`; Linux `io_uring` keeps
  ring-depth batched SQE submission with fixed-file and
  fixed-buffer registration.
- Experimental `SQPOLL` / `IOPOLL` / linked-fsync modes are not
  part of the v0.3 path: on Holt's short checkpoint batches they
  did not improve throughput, and keeping them would make the I/O
  completion path harder to audit. The next real Linux step is a
  dedicated async I/O scheduler, not more ring flags.
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
  Release benchmark coverage stays focused on the public
  RocksDB/SQLite metadata comparison. Checkpoint-specific probes
  should be reintroduced only when they are promoted to either a
  public comparator or a correctness integration test.

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
- Large-tree release coverage is the public `{20 k, 100 k,
  500 k, 2 M}` scale curve in `benches/main.rs`. Additional
  shape probes should be profile-driven and promoted only if they
  become part of the release comparison surface.

## v0.4 — Scale-stable metadata performance

Goal: keep the v0.3 cache-resident read/list advantages while making
large-tree writes and negative metadata probes stay flat as the
namespace grows from millions to tens or hundreds of millions of
records. v0.4 should be judged by scale curves, not by small-tree
microbench wins.

The main risk at this point is not one more syscall on the durable
path. It is shape drift: overfull hot prefixes, too many cross-blob
hops, negative lookups walking full descents, checkpoint bursts
competing with writers, and bulk creates paying one full ART descent
per key. The v0.4 work is therefore workload-shaped rather than
generic KV tuning.

### P0 — Prefix shape policy and write-stability guardrails

- Add explicit shape-control metrics to release benches:
  blob fill distribution, blob-hop histogram, spillover/merge rate,
  checkpoint dirty backlog, and WAL sync batching ratio at each scale
  point.
- Promote prefix/subtree boundaries deliberately. The generic API
  should be a shape hint such as `promote_prefix(prefix)`, not an
  S3-specific "bucket" concept. Object-store users can map one
  bucket or tenant to a promoted prefix; filesystem users can map hot
  directories or mount roots.
- Make spill/merge policy backlog-aware. When writers are flooding a
  hot prefix, the engine should prefer bounded child splits and defer
  cosmetic merges instead of oscillating between split/compact/merge.
- Add write-stability benches that model untar/rsync/S3-sync bursts:
  sorted path creates, random creates under one hot prefix, mixed
  create/delete churn, and large-prefix list while writes continue.

Success criteria:

- `put` and `atomic` throughput should degrade smoothly with scale:
  no cliff when moving from 2M to larger trees.
- Average blob hops should stay low and p99 blob hops should be
  explainable by explicit promoted boundaries, not accidental shape
  drift.
- Checkpoint backlog should drain without forcing foreground writers
  through long global waits.

### P1 — Negative lookup filter, in-memory first

Metadata workloads are unusually negative-heavy: `open/stat/head`
often ask for keys that do not exist. Holt should exploit that, but
without changing the disk format prematurely.

- Add a per-cached-blob negative-lookup sidecar filter first. It must
  never introduce false negatives: on mutation uncertainty, mark the
  filter stale and fall back to the full walker.
- Use the filter only at cross-blob boundaries and within hot prefixes
  where avoiding another blob hop is measurable.
- Rebuild filters from blob contents on cache fill. Persisting the
  filter in the blob header is a later on-disk-format decision, not
  the first implementation.

Success criteria:

- Negative point lookup latency improves on path-shaped workloads
  without regressing positive lookups.
- Filter memory overhead is bounded and visible in `Tree::stats`.

### P2 — Explicit bulk-load path

Bulk create is a metadata-specific write pattern. Treating an archive
restore as N unrelated `put`s leaves performance on the table.

- Add an explicit sorted bulk-load API or builder rather than hidden
  timing heuristics. Callers that know they are loading a directory,
  bucket prefix, image layer, or archive can opt in.
- Build final ART shapes directly for sorted keys: avoid repeated
  Node4 -> Node16 -> Node48 -> Node256 promotion and repeated descent
  through the same prefix.
- Commit each bulk batch as one atomic WAL envelope with bounded byte
  and key-count limits, so crash replay and memory use stay simple.

Success criteria:

- Mass-create workloads improve by multiples, not percentages.
- Point-write latency for ordinary single-key writes is unchanged.

### P3 — Async checkpoint scheduler, not more io_uring flags

The v0.3 Linux backend already has the useful fixed-file /
fixed-buffer / batched-ring foundation. The next I/O gain is not
`SQPOLL` or linked-fsync by default; those did not help Holt's short
checkpoint batches enough to justify the complexity. The next gain is
execution-model work:

- Keep dirty-byte and dirty-age watermarks so checkpoint starts before
  the dirty set becomes a foreground-write problem.
- Let the I/O worker keep multiple write batches in flight when the
  backend can absorb queue depth, while preserving the WAL-before-data
  and data-before-manifest ordering boundaries.
- Add foreground write throttling only as a last resort, driven by
  dirty backlog and checkpoint lag, not by raw operation count.
- Keep the default Unix `pwritev` path simple and auditable; Linux
  `io_uring` should remain an internal backend optimization with the
  same correctness contract.

Success criteria:

- Under sustained writes, p99 put latency should remain bounded while
  checkpoint makes progress.
- Idle and lightly loaded trees should not pay extra scheduler cost.

## v0.5 — Stronger metadata semantics without MVCC tax

Goal: add the missing semantics that real metadata servers need while
preserving Holt's core design: path-shaped keys, opaque byte values,
single-node embedded library, no always-on MVCC version chains.

### P0 — Snapshot view for list/readdir

- Add a `view` / snapshot-list path that copies the blobs needed for a
  prefix scan into private memory, then scans the copy without holding
  writer latches.
- This gives stable list/readdir semantics without per-write MVCC
  overhead. The cost is paid by the list operation proportional to the
  prefix it asks to observe.
- Keep ordinary `range()` as the fast best-effort iterator only if the
  API clearly distinguishes the two modes. If the public API must stay
  minimal, prefer one strong iterator and optimize it.

### P1 — WAL tail / change feed with explicit retention

- Expose a change-feed API only after WAL retention is designed.
  Today's checkpoint path truncates WAL, so `subscribe(from_seq)`
  cannot pretend old history is always available.
- The first version should support live-tail subscription and bounded
  retained segments for cache invalidation, audit, and follower-style
  experiments.
- Follower mode or replication should remain outside the core release
  until the single-node durability contract has a crisp tail-retention
  story.

### P2 — Sealed immutable subtree

- Add an optional `seal_prefix(prefix)` mode for image layers,
  archives, and package indexes that become read-only after build.
- Start with semantic enforcement and latch elision. Huge pages,
  mmap tiers, and TLB-focused placement are later profile-driven
  optimizations.

## v0.6+ — Optional adapters and research-grade hardware work

These are useful, but they should not pollute the core engine or the
0.4/0.5 scale-stability work.

- POSIX dirent/inode separation belongs in an adapter/example layer:
  e.g. `d/<path>` for dirents and `i/<inode>` for inode metadata.
- ACL inheritance, S3 bucket semantics, lifecycle policy, and object
  versioning are schema conventions above Holt's opaque byte values.
- Value compression should be an optional codec layer, not a core
  schema-aware dictionary unless benchmarks prove opaque values are
  the limiting factor.
- Prefix interning changes node layout and adds read indirection; do
  it only after path-memory profiles show it beats the existing ART
  prefix compression.
- NUMA, CXL.mem, ZNS, and lockless rename are server-appliance or
  research tracks. They are valid experiments, but not release
  blockers for an embedded metadata engine.

## Non-goals for the core engine

- Do not bake S3 object schemas, inode structs, ACL formats, JSON,
  protobuf, chunk lists, or lifecycle rules into the storage core.
  Holt stores opaque values; higher layers own schema.
- Do not add full MVCC until a concrete workload proves copy-on-list
  and atomic CAS batches are insufficient.
- Do not chase io_uring flags that make the backend harder to audit
  without moving a measured bottleneck.
- Do not add compatibility aliases for renamed public APIs. The crate
  is still young enough that clean API shape matters more than
  transition glue.

## v1.0 — Production-ready

- Post-v0.5 public surface and persistence format stabilized.
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
  core, blob layout, WAL, latching, range iterator. `WalOp`
  variants holt journals share wire shape with the upstream so a
  future RPC layer could re-use the format, but holt itself ships
  no multi-root registry, no bucket namespace, no RPC dispatcher.
- **Replication / consensus** — build it above this. Future hooks
  can support it, but holt itself does not implement Raft.
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
