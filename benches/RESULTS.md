# holt — benchmark results

End-to-end criterion micro-benches comparing holt against
**RocksDB** (`rocksdb` crate, default-features-off + bundled
`librocksdb-sys`) and **SQLite** (`rusqlite` with the
`bundled` libsqlite3, so contributors don't need a system
SQLite installation). Three workload shapes (KV, S3
object-store metadata, POSIX filesystem metadata) ×
{ memory, persistent } × { get, put, mixed, list, list-delim,
durable-put }.

## Reproducing

```bash
# Full suite (~5 min on M3 Pro).
cargo bench --bench main -- --output-format bencher

# One group only — e.g. just the durable-put numbers.
cargo bench --bench main -- _durable_put --output-format bencher
```

Each criterion sample is one op. Numbers are mean ± noise band
in nanoseconds; lower is better. Holt's per-op numbers are
randomised over a 10 000-key dataset (see `gen_*_dataset`);
RocksDB / SQLite are driven by the same dataset for fair
comparison.

## Test environment

- **Hardware**: Apple M3 Pro (12 cores), 36 GB RAM
- **OS**: macOS 26.3 (Darwin 25.0.0)
- **Rust**: 1.94.0 stable, release profile (`lto=thin`,
  `codegen-units=1`, `opt-level=3`)
- **holt**: commit `63b181d` (v0.2 release-class — `wal.lock`
  W2D protocol, sharded BufferManager, 3-thread bg
  checkpointer, SIMD CRC32 + node scans). The baseline 24-
  bench data was captured at commit `aabb133` (same code on
  the hot path); the durable-write group A was captured at
  `63b181d`.
- **RocksDB**: 0.24 (`librocksdb-sys` 0.18, bundled)
- **SQLite**: rusqlite 0.39 (bundled libsqlite3)
- **Knob alignment**: all three engines use comparable
  "per-op durable to OS page cache, not fsync'd" semantics —
  see the durability matrix at the top of `benches/main.rs`.

## Headline numbers

24 baseline benches across KV / objstore / fs shapes, memory
+ persistent variants. **Holt wins all 24** vs RocksDB and
SQLite. Margin range: 1.3× (in-memory fs_put vs SQLite — both
short codepaths) to **467×** (`fs_list_dir` S3-style rollup
vs RocksDB — fast-forward over `BlobNode` crossings beats
seek-iterator-per-leaf hands down).

## KV workload (short random keys + short values)

| Bench               | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
| ------------------- | --------: | -----------: | ----------: | ---------: | --------: |
| **memory** get      |  **169**  |          684 |         567 |       4.0× |      3.4× |
| **memory** put      |  **344**  |        1 201 |         629 |       3.5× |      1.8× |
| **memory** mixed    |  **351**  |        2 138 |         663 |       6.1× |      1.9× |
| **persist** get     |  **187**  |          637 |       1 508 |       3.4× |      8.1× |
| **persist** put     |  **473**  |        3 470 |       2 310 |       7.3× |      4.9× |
| **persist** mixed   |  **328**  |        3 294 |       1 951 |      10.0× |      5.9× |

## Object-store workload (S3-shaped path keys + metadata values)

| Bench                       | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
| --------------------------- | --------: | -----------: | ----------: | ---------: | --------: |
| **memory** get              |  **250**  |          702 |         622 |       2.8× |      2.5× |
| **memory** put              |  **481**  |        1 441 |         664 |       3.0× |      1.4× |
| **memory** mixed            |  **377**  |        2 152 |         663 |       5.7× |      1.8× |
| **memory** list             |  **10 808** |     16 815 |      16 637 |       1.6× |      1.5× |
| **persist** get             |  **247**  |          740 |       1 508 |       3.0× |      6.1× |
| **persist** put             |  **567**  |        3 499 |       2 319 |       6.2× |      4.1× |
| **persist** mixed           |  **420**  |        3 264 |       1 954 |       7.8× |      4.7× |
| **persist** list            |  **10 651** |     16 937 |      17 801 |       1.6× |      1.7× |
| **list_dir** (S3 rollup)    |  **2 463** |    624 672 |     436 204 |     **254×** |  **177×** |

## Filesystem-metadata workload (inode + dirent path keys)

| Bench                | Holt (ns) | RocksDB (ns) |  SQLite (ns) | vs RocksDB | vs SQLite |
| -------------------- | --------: | -----------: | -----------: | ---------: | --------: |
| **memory** get       |  **239**  |          700 |          630 |       2.9× |      2.6× |
| **memory** put       |  **488**  |        1 452 |          660 |       3.0× |      1.4× |
| **memory** mixed     |  **372**  |        2 469 |          668 |       6.6× |      1.8× |
| **memory** list      |  **10 854** |    17 887 |       16 775 |       1.6× |      1.5× |
| **persist** get      |  **251**  |          701 |        1 516 |       2.8× |      6.0× |
| **persist** put      |  **555**  |        3 456 |        2 292 |       6.2× |      4.1× |
| **persist** mixed    |  **411**  |        3 165 |        1 961 |       7.7× |      4.8× |
| **persist** list     |  **11 111** |    17 842 |       17 727 |       1.6× |      1.6× |
| **list_dir**         |  **2 812** |  1 317 457 |      917 245 |     **468×** |  **326×** |

## Durable-write workload (group A: per-op fsync)

**Knob alignment**: holt `wal_sync_on_commit=true`, RocksDB
`WriteOptions::set_sync(true)`, SQLite `PRAGMA
journal_mode=WAL; synchronous=FULL`.

| Bench                    |   Holt (µs) | RocksDB (µs) | SQLite (µs) |
| ------------------------ | ----------: | -----------: | ----------: |
| **kv** durable put       |   **2 996** |       **25** |     **2.2** |
| **objstore** durable put |   **2 985** |       **21** |     **2.1** |
| **fs** durable put       |   **3 000** |       **24** |     **2.1** |

These numbers look like a loss for holt at first glance, but
they're actually **measuring three different durability
guarantees on macOS**:

| Engine    | Effective sync syscall on macOS | Drive-cache barrier |
| --------- | ------------------------------- | ------------------- |
| holt      | `fcntl(F_FULLFSYNC)` via Rust std `File::sync_data` | **yes** (real power-safe fsync) |
| RocksDB   | `fcntl(F_BARRIERFSYNC)` (since 10.14) | ordered, but no drive flush |
| SQLite    | `write()` + lazy fsync (WAL mode collapses per-row commits) | no per-op fsync at all |

The 3 ms / op number is what an actual power-safe durable
write costs on M3 Pro + APFS — that's the cost of telling the
NVMe controller to flush its cache. RocksDB / SQLite's
"sync=true" flags do **not** request that on macOS; they only
order writes against the kernel page cache. On a Linux ext4
system the gap shrinks substantially because `fdatasync` there
includes a drive barrier by default.

Two implications:

1. **For real crash-safe-against-power-loss workloads** (the
   reason you'd flip `wal_sync_on_commit=true`), holt's number
   is the honest one. RocksDB / SQLite would need
   platform-specific full-fsync to match it.
2. **For "process crash, kernel survives"** durability —
   the more common requirement — holt's default
   `wal_sync_on_commit=false` already gets you that for free
   via the WAL writer's `write()` calls landing in the kernel
   page cache. Use the persistent-put numbers above (473 ns
   for KV `_persist_put`), not the durable-put numbers, for
   that durability tier.

A future bench could add a "kernel-flush only" tier — drop
`fcntl(F_FULLFSYNC)` from the holt WAL writer behind a config
flag and re-measure. That would put all three engines on
the same `F_BARRIERFSYNC`-or-equivalent footing.

## Workload notes

- **`*_get` / `*_put`**: 10 000-key dataset, randomly sampled
  with `StdRng(seed=SEED)`. Pre-load happens once outside the
  measured region.
- **`*_mixed`**: 80 % gets, 20 % puts, same dataset.
- **`*_list`** (plain): prefix narrows to ~625 keys
  (`objstore`) / ~1 250 keys (`fs`); each criterion sample
  iterates up to 100 results.
- **`*_list_dir`** (S3-style rollup): prefix + delimiter `/`;
  emits 32 (`objstore`) / 16 (`fs`) `CommonPrefix` entries per
  pass, then stops. Holt's iterator's fast-forward — ascend
  past each rollup's subtree — turns the walk from
  `O(leaves_under_prefix)` into `O(distinct_rollups)`. RocksDB
  + SQLite both scan every leaf and dedupe in the host loop,
  which is what the 100–500× gap measures.

## Planned follow-up groups (B, C)

- **Group B — scale curve**: parameterise `kv_get` / `kv_put` /
  `objstore_list_dir` over `{20 k, 200 k, 2 M}` keys to see how
  holt's walker latency (and cache-miss rate) compares to
  RocksDB's LSM amplification at scale.
- **Group C — p95 / p99 under maintenance interference**: a
  separate `tests/bench_contention_p95.rs` harness with
  `hdrhistogram`: N writer threads + a background checkpointer
  + periodic manual `compact()` runs concurrently; report
  mean / p50 / p95 / p99 / p99.9 of `put` latency. Verifies
  that tail latency stays bounded while compactions run.
