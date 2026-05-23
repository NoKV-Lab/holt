# Benchmarks

Criterion-based microbenchmarks comparing **holt** against
**RocksDB**, **SQLite**, and **sled** across three shapes of
metadata workload — `kv` (anti-pattern baseline), `objstore`,
and `fs` (holt's design target).

## Scenarios

| Group | Key shape | Value shape | Models |
|---|---|---|---|
| `kv` | 32-byte random | 64-byte random | Anonymous KV baseline — **pessimal for ART** (no prefix sharing, no key locality). |
| `objstore` | `bucket-NN/path/sub/file-NNNN.bin` | `{"size":...,"etag":...,"class":"STD"}` (~60 B fixed) | S3-style object metadata. |
| `fs` | `/usr/local/share/category-N/file-NNNN` | 32-byte packed inode (size + mtime + mode + uid + gid + nlink) | POSIX filesystem metadata. |

Each scenario runs three point-access operations:

- `*_get` — random key lookup over a pre-loaded dataset
- `*_put` — random key replacement (in-place update)
- `*_mixed` — 50% get / 50% put, key chosen at random

The `objstore` + `fs` scenarios additionally run
**metadata-native** operations — the common operations that a
metadata engine actually serves beyond blind point overwrite:

- `*_list` — marker-aware prefix range scan, `take(100)` entries
- `*_list_dir` — S3-style delimiter rollup, take 8 distinct
  `CommonPrefix` entries (holt does the dedup in the engine;
  RocksDB, SQLite, and sled get the same logic done at the
  bench's app layer, since they do not expose a native
  `?delimiter=` API)
- `*_create_delete` — create a scratch metadata entry, then
  delete it to keep the benchmark state bounded
- `*_rename` — atomic rename round-trip. Holt uses `Tree::rename`;
  RocksDB and sled use write batches; SQLite uses an explicit
  transaction.
- `*_metadata_mix` — weighted objstore/fs metadata mix:
  45% stat/get, 20% metadata update, 10% plain list, 10%
  delimiter list-dir, 10% create+delete, 5% rename round-trip.

The Criterion release harness keeps `*_list` as a full key/value
prefix scan because that is the historical published metric. The
large-tree stress harness additionally splits key-only `list` from
full-record `list_records`, matching the public Holt API:
`scan_keys` for name-only metadata listings and `range` when the
caller needs value bytes.

`N_KEYS = 20 000` for the baseline scenarios — large enough that
the data spreads across **multiple holt blobs** (~6–8 × 512 KB),
so the bench exercises `BlobNode` crossings + cross-blob
spillover/compact retries, not just single-blob descent.

A second group — **scale curve** (`kv_scale_get` / `kv_scale_put`)
— parameterizes over `{ 20 000, 100 000, 500 000, 2 000 000 }`
keys. The 500 k tier (~48 MB payload) already exceeds the
default 32 MB buffer pool; the 2 M tier is the large-tree
pressure case used to judge path-put scalability.

## Running

```sh
# Full criterion sweep (~5 min on M3 Pro):
cargo bench --bench main

# Quick smoke pass (~1 minute):
cargo bench --bench main -- --quick --noplot

# Scale curve only (Group B):
cargo bench --bench main -- kv_scale

# A single scenario:
cargo bench --bench main -- kv_get

# Just the range scans (the load-bearing metadata-engine test):
cargo bench --bench main -- _list

# Just the metadata-native mutation/mix groups:
cargo bench --bench main -- _create_delete
cargo bench --bench main -- _rename
cargo bench --bench main -- _metadata_mix

# Large-tree stress harness: fixed 20M preload, million-scale ops.
# Run this explicitly; it is intentionally not part of Criterion.
HOLT_STRESS_N=20000000 \
HOLT_STRESS_POINT_OPS=1000000 \
HOLT_STRESS_LIST_OPS=1000000 \
cargo bench --bench stress -- objstore

HOLT_STRESS_N=20000000 \
HOLT_STRESS_POINT_OPS=1000000 \
HOLT_STRESS_LIST_OPS=1000000 \
cargo bench --bench stress -- fs
```

HTML criterion reports land in `target/criterion/`.

## Large-Tree Stress Harness

`benches/stress.rs` keeps the dataset fixed at 20M keys by
default, then runs high-pressure operation loops:

- `get`: `HOLT_STRESS_POINT_OPS` random point reads
- `put`: `HOLT_STRESS_POINT_OPS` same-size metadata updates
- `mixed`: 50% get / 50% put over the same sampled key stream
- `list`: `HOLT_STRESS_LIST_OPS` bounded key-only prefix scans
- `list_records`: `HOLT_STRESS_LIST_OPS` bounded full key/value
  prefix scans
- `list_dir`: `HOLT_STRESS_LIST_OPS` delimiter rollups

The stress harness prints `ns/op`, `Mops/s`, and Holt shape
telemetry (`blobs`, cross-blob `edges`, `max_depth`, `avg_depth`,
blob fill ratios, walker hop counters). Those shape metrics are
the main signal for diagnosing whether 20M writes remain stable
or start paying extra blob-hop / spillover cost.

For `list_dir`, the stress harness gives RocksDB and SQLite a
fair app-layer fast-forward implementation: after emitting one
rolled-up prefix, it seeks to that prefix's lexicographic
successor instead of scanning every key below the directory.
sled uses the same app-layer fast-forward shape when selected.

Useful knobs:

```sh
# Run only Holt while tuning tree shape.
HOLT_STRESS_ENGINES=holt cargo bench --bench stress -- objstore

# Add sled as a Rust embedded-KV peer.
HOLT_STRESS_ENGINES=holt,rocksdb,sled cargo bench --bench stress -- objstore

# Smoke-test the harness quickly.
HOLT_STRESS_N=10000 \
HOLT_STRESS_POINT_OPS=1000 \
HOLT_STRESS_LIST_OPS=100 \
cargo bench --bench stress -- fs

# Change list fanout without changing the 20M preload.
HOLT_STRESS_LIST_TAKE=1000 \
HOLT_STRESS_DIR_TAKE=32 \
cargo bench --bench stress -- objstore
```

Profile: single-threaded, warm-service, file-backed persistent
engines with WAL enabled and no per-op fsync. Holt uses
`TreeConfig::new(tempdir)` with `WalCommit::Write`; RocksDB uses
WAL on with `sync = false`; SQLite uses a file-backed WAL database
with `synchronous = OFF`. This is the fair persistent hot path:
WAL bytes are written before the operation returns, but power-loss
durability is not forced per operation.
sled can be selected as an embedded-KV peer, but its flush controls
do not map exactly to this WAL-on/no-fsync matrix; the harness runs
it in high-throughput mode with background flush disabled during
the timed section.

## Concurrent Harness

`benches/concurrent.rs` measures multi-thread throughput for one
shared file-backed engine. It currently compares Holt and RocksDB;
SQLite is intentionally excluded from the main table because WAL
mode still serializes writers, so it is a different concurrency
model.

```sh
HOLT_CONCURRENT_N=2000000 \
HOLT_CONCURRENT_OPS_PER_THREAD=100000 \
HOLT_CONCURRENT_THREADS=1,2,4,8 \
HOLT_CONCURRENT_OPS=get,put,mixed90,mixed50,list_dir \
cargo bench --bench concurrent -- objstore

HOLT_CONCURRENT_N=2000000 \
HOLT_CONCURRENT_OPS_PER_THREAD=100000 \
HOLT_CONCURRENT_THREADS=1,2,4,8 \
HOLT_CONCURRENT_OPS=get,put,mixed90,mixed50,list_dir \
cargo bench --bench concurrent -- fs
```

Profile: warm-service, file-backed persistent engines with WAL
enabled and no per-op fsync. Reported throughput is wall-clock
multi-thread throughput; sampled latency is approximate and exists
to expose tail regressions, not to replace a production latency
study.

## Methodology — apples-to-apples

Two comparison modes, with Holt/RocksDB/SQLite tuned to the same
durability profile. sled is included as a Rust embedded-KV peer,
but its flush/durability knobs are not an exact match, so sled rows
should keep that caveat attached.

### Memory / no-WAL mode (`*_get` / `*_put` / `*_mixed`)

Engine algorithm cost only — durability disabled where the engine
exposes that mode:

- **holt**: `TreeConfig::memory()` with `memory_flush_on_write =
  false`. Mutations stay in the BufferManager-pinned blobs.
- **RocksDB**: temp-dir DB, `disable_wal = true`, `sync = false`,
  64 MB memtable, compression disabled.
- **SQLite**: `:memory:` DB, `journal_mode=MEMORY`,
  `synchronous=OFF`, 64 MB page cache, `WITHOUT ROWID` schema.
- **sled**: temporary DB, `Mode::HighThroughput`, 64 MB cache,
  `flush_every_ms(None)`. sled does not expose a direct
  `disable_wal` equivalent.

### Hot persistent mode (`*_persist_get` / `*_persist_put` / `*_persist_mixed`)

All three engines disk-backed with WAL on, per-op durability to
the OS page cache (not fsync) — the "you survive a process
crash, not a power failure" mode high-throughput services target.
The service is warm: the Holt BufferManager, RocksDB cache/memtable,
and SQLite page cache may all contain data touched during preload
or Criterion warmup. This is a foreground WAL/cache benchmark, not
a cold data-file I/O benchmark:

- **holt**: `TreeConfig::new(tempdir)` (FileBlobStore with
  `F_NOCACHE` on macOS / `O_DIRECT` on Linux) and
  `WalCommit::Write`. Every mutation waits until its encoded WAL
  record reaches the OS page cache; blobs only hit disk at
  checkpoint.
- **RocksDB**: temp-dir DB, `disable_wal = false`, `sync = false`.
  Each `put` appends to the WAL (buffered) plus the memtable.
- **SQLite**: file-backed DB, `journal_mode=WAL`,
  `synchronous=OFF`, 64 MB page cache.
- **sled**: temp-dir DB, `Mode::HighThroughput`, 64 MB cache,
  `flush_every_ms(None)`. Its foreground acknowledgement semantics
  are not identical to the WAL-on engines above; treat the row as
  peer context rather than a strict durability-equivalent cell.

Shared settings: 20 000 unique keys preloaded; bench iterates a
seeded permutation of that set; `cargo bench` builds with
`lto="thin"`, `codegen-units=1`, `opt-level=3`; single-threaded.

### Metadata-native groups

`*_create_delete`, `*_rename`, and `*_metadata_mix` currently run
in the memory/no-WAL profile. They are meant to isolate operation
semantics and data-structure cost:

- create/delete is a bounded create+unlink pair, not a growing
  insert-only workload.
- rename is held to atomic move semantics for every engine.
- metadata_mix is deliberately heterogeneous; one iteration is
  one sampled metadata operation, and the operation mix is fixed
  by seed and percentage buckets.

## How to read the numbers

The `objstore` + `fs` scenarios are the **right** test for what
holt is designed to do. The `kv` scenario is the **wrong** test,
included on purpose — it tells you how badly an ART degrades when
the workload violates its assumptions.

| Scenario | What it actually measures | Expected outcome |
|---|---|---|
| `kv` (random 32-byte keys) | ART without prefix sharing or metadata semantics | anti-pattern baseline; useful mainly for checking constants and scale |
| `objstore` (path keys) | ART on hierarchical keys, plus S3 list/rename/create semantics | holt should win most clearly on list_dir and metadata_mix |
| `fs` (POSIX paths) | Long common prefixes, directory list, rename/create/delete | holt should win most clearly on directory/list-heavy mixes |

Pick the engine that matches your **key shape**. holt is for
hierarchical, prefix-rich keys; if your keys are random bytes
(hashes, UUIDs without a path prefix), reach for RocksDB / SQLite.

## Results

This README defines the workload surface and methodology only.
Concrete release numbers live in [`RESULTS.md`](RESULTS.md), so
there is one source of truth for quoted performance data.

When reading those results, keep the profiles separate:

- Memory/no-WAL rows isolate ART/data-structure and
  metadata-operation semantics.
- Hot persistent rows use disk-backed engines with WAL enabled and
  no per-op fsync.
- The Criterion harness has both memory/no-WAL rows and explicit
  persistent rows. The large-tree stress harness is persistent by
  default because the release story should match Holt's file-backed
  deployment shape.

Plain prefix scans (`*_list`) model `readdir` / `ListObjects` with
a bounded prefix range. The Criterion release table reports the
full key/value form; the stress harness reports key-only `list`
and full-record `list_records` separately. Delimiter rollup
(`*_list_dir`) is the S3-style listing test: Holt emits
`CommonPrefix` inside the engine and fast-forwards past the
rolled-up subtree; RocksDB and SQLite use generic ordered
iteration plus app-layer dedup because neither exposes a native
delimiter-list API.
sled is reported the same way as RocksDB here: ordered iteration
plus app-layer prefix math.

## Caveats

1. **Single-threaded latency, not throughput.** Point reads use
   optimistic per-blob latching; range scans use shared guards plus
   versioned cursor validation. The public benchmark surface
   measures single-thread latency, not multi-core throughput.
2. **No fsync in persistent rows.** Persistent rows use
   `sync=off`-equivalent semantics: WAL bytes reach the OS page
   cache before the operation returns, but no per-op `fsync` is
   forced. Memory rows are volatile by definition. sled does not
   expose the same WAL/no-WAL split, so sled rows are not strict
   durability-equivalent cells.
3. **Delimiter rollup is still an engine-specific semantic.** Holt's
   `Tree::range` ascends the descent stack past a rolled-up
   subtree after emitting its `CommonPrefix`, so the cost is
   `O(distinct_rollups)`. The stress harness gives RocksDB,
   SQLite, and sled an app-layer fast-forward cursor, but the
   semantic is not native to those engines; it is still built from
   ordered iteration plus application-side prefix math.
4. **Bench numbers are machine-dependent.** Don't take any
   absolute throughput claim from this README at face value —
   re-run on your hardware. The relative ordering (holt wins on
   path-shaped point lookup, metadata-native mixes, and
   delimiter rollup; point put is a smaller win at large scale) is
   the load-bearing observation.
5. **Range is restart-on-conflict, not MVCC.** `Tree::range` and
   `Tree::range_keys` store blob versions in their cursor path and
   seek from the last emitted lower bound if a writer invalidates
   that path. A long scan can still observe keys committed after
   iterator creation if they sort after the current cursor. Use
   `Tree::view(prefix, ...)` when a benchmark needs stable read
   transaction semantics.

This bench is the right comparison for **metadata-engine
workloads** with bounded per-tree dataset and hierarchical keys —
directory listings, S3 metadata, inode tables, AI artefact
catalogs. It is not the right comparison for "100M-key analytics
datastore" workloads or "random UUID hot-path" workloads.
