# holt benchmark results

End-to-end Criterion microbenchmarks comparing **holt** against
**RocksDB** (`rocksdb` crate, bundled `librocksdb-sys`) and
**SQLite** (`rusqlite`, bundled libsqlite3). The public benchmark
surface is intentionally small: one harness, three workload
shapes, and operation mixes that reflect metadata engines rather
than generic key/value stores.

The Criterion tables below are the v0.3.0 Linux release snapshot.
The 50 M large-tree stress section is a v0.3.1 local macOS
snapshot. Keep those environments separate when quoting numbers.

## Reproducing

```bash
cargo bench --features io-uring --bench main -- --output-format bencher
```

Use filters for shorter runs:

```bash
cargo bench --bench main -- _metadata_mix --output-format bencher
cargo bench --bench main -- _list_dir --output-format bencher
cargo bench --bench main -- _scale_ --output-format bencher
```

Each sample is one logical operation. Lower is better. All three
engines receive the same generated dataset and the same seeded
operation stream.

## Test environment

- **Hardware**: GCP `c2-standard-8`
- **OS**: Linux on local SSD
- **Rust**: 1.95.0
- **holt**: v0.3.0, `--features io-uring`
- **RocksDB**: 0.24 (`librocksdb-sys` 0.18, bundled)
- **SQLite**: rusqlite 0.39 (bundled libsqlite3)
- **Durability alignment**: `*_persist_*` groups are hot
  persistent / WAL-on measurements. They model "durable to OS page
  cache, not per-op fsync" semantics, with each engine's cache
  allowed to stay warm. They are not cold-reopen I/O numbers.
- **Coverage**: the 20 k point-operation tables include both
  memory/no-WAL and hot persistent/WAL-on rows. Plain prefix
  `list` also has hot persistent rows. The current
  `metadata_mix`, create/delete, rename, delimiter `list_dir`, and
  20 k→2 M scale-curve groups are memory/no-WAL unless explicitly
  labeled `hot persist`.

## Workloads

| Workload | Key shape | Value shape | Why it exists |
|---|---|---|---|
| `kv` | random 32-byte keys | random 64-byte values | Anti-pattern baseline for ART: little prefix sharing. |
| `objstore` | `bucket-NN/path/sub/file-NNNN.bin` | fixed JSON-ish metadata | S3-style object metadata. |
| `fs` | `/usr/local/share/category-N/file-NNNN` | packed inode-ish bytes | Filesystem metadata. |

`objstore` and `fs` also run metadata-native operations:
`list`, `list_dir` delimiter rollup, create/delete, rename, and a
weighted metadata mix.

## Headline

Holt is strongest where the workload is actually metadata-shaped.
The headline metadata-native numbers below are memory/no-WAL
operation-semantics measurements; hot persistent point-op and
plain-list rows are reported in the tables that follow.

- `objstore_metadata_mix`: **2.251 us** vs RocksDB **98.064 us**
  and SQLite **66.394 us**.
- `fs_metadata_mix`: **2.462 us** vs RocksDB **162.939 us** and
  SQLite **131.395 us**.
- `objstore_list_dir`: **4.204 us** vs RocksDB **638.397 us** and
  SQLite **584.494 us**.
- `fs_list_dir`: **4.898 us** vs RocksDB **1.316 ms** and SQLite
  **1.197 ms**.

The tight cells are large-scale point writes. At 2 M keys, Holt
still wins every point-put cell in this Linux run, but only by
**1.07-1.17x** against RocksDB. Treat point put as competitive,
not as the main marketing claim.

## Baseline point operations, 20 k keys

### KV

| Bench | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|
| memory get | **272** | 749 | 920 | 2.8x | 3.4x |
| memory put | **421** | 1,796 | 996 | 4.3x | 2.4x |
| memory mixed | **348** | 3,045 | 979 | 8.8x | 2.8x |
| hot persist get | **274** | 743 | 2,317 | 2.7x | 8.5x |
| hot persist put | **922** | 3,446 | 3,535 | 3.7x | 3.8x |
| hot persist mixed | **542** | 3,978 | 2,933 | 7.3x | 5.4x |

### Object-store metadata

| Bench | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|
| memory get | **310** | 691 | 892 | 2.2x | 2.9x |
| memory put | **478** | 2,016 | 995 | 4.2x | 2.1x |
| memory mixed | **402** | 3,011 | 950 | 7.5x | 2.4x |
| hot persist get | **314** | 709 | 2,262 | 2.3x | 7.2x |
| hot persist put | **829** | 3,714 | 3,460 | 4.5x | 4.2x |
| hot persist mixed | **617** | 3,807 | 2,963 | 6.2x | 4.8x |
| memory list | **21,013** | 23,825 | 32,941 | 1.1x | 1.6x |
| hot persist list | **20,816** | 23,875 | 34,634 | 1.1x | 1.7x |
| list_dir | **4,204** | 638,397 | 584,494 | 151.9x | 139.0x |

### Filesystem metadata

| Bench | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|
| memory get | **374** | 700 | 885 | 1.9x | 2.4x |
| memory put | **560** | 1,858 | 972 | 3.3x | 1.7x |
| memory mixed | **480** | 2,740 | 960 | 5.7x | 2.0x |
| hot persist get | **377** | 695 | 2,245 | 1.8x | 6.0x |
| hot persist put | **891** | 3,802 | 3,467 | 4.3x | 3.9x |
| hot persist mixed | **649** | 3,758 | 2,929 | 5.8x | 4.5x |
| memory list | **21,822** | 24,008 | 32,457 | 1.1x | 1.5x |
| hot persist list | **21,986** | 24,190 | 34,393 | 1.1x | 1.6x |
| list_dir | **4,898** | 1,316,044 | 1,196,702 | 268.7x | 244.3x |

## Metadata-native operations

| Bench | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|
| objstore create/delete | **473** | 1,250 | 5,478 | 2.6x | 11.6x |
| objstore rename | **1,961** | 4,668 | 24,866 | 2.4x | 12.7x |
| objstore metadata_mix | **2,251** | 98,064 | 66,394 | 43.6x | 29.5x |
| fs create/delete | **780** | 1,260 | 5,499 | 1.6x | 7.1x |
| fs rename | **2,529** | 4,819 | 24,903 | 1.9x | 9.8x |
| fs metadata_mix | **2,462** | 162,939 | 131,395 | 66.2x | 53.4x |

The metadata mix is the best single summary for the project: it
combines stat/get, metadata update, plain list, delimiter
list-dir, create/delete, and rename. Generic KV baselines must
reconstruct delimiter rollup above the engine; Holt implements it
inside the ART walker and fast-forwards past rolled-up subtrees.

## Scale curve: 20 k to 2 M keys

### Get

| Workload | Keys | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|---:|
| kv | 20 k | **283** | 729 | 903 | 2.6x | 3.2x |
| kv | 100 k | **505** | 1,202 | 1,301 | 2.4x | 2.6x |
| kv | 500 k | **803** | 2,018 | 1,828 | 2.5x | 2.3x |
| kv | 2 M | **1,406** | 9,006 | 2,327 | 6.4x | 1.7x |
| objstore | 20 k | **314** | 732 | 913 | 2.3x | 2.9x |
| objstore | 100 k | **559** | 1,057 | 1,287 | 1.9x | 2.3x |
| objstore | 500 k | **962** | 1,554 | 1,739 | 1.6x | 1.8x |
| objstore | 2 M | **1,457** | 4,765 | 2,093 | 3.3x | 1.4x |
| fs | 20 k | **370** | 738 | 891 | 2.0x | 2.4x |
| fs | 100 k | **588** | 1,043 | 1,213 | 1.8x | 2.1x |
| fs | 500 k | **1,053** | 1,520 | 1,706 | 1.4x | 1.6x |
| fs | 2 M | **1,262** | 3,797 | 2,088 | 3.0x | 1.7x |

### Put

| Workload | Keys | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|---:|
| kv | 20 k | **410** | 1,803 | 1,015 | 4.4x | 2.5x |
| kv | 100 k | **800** | 1,956 | 1,439 | 2.4x | 1.8x |
| kv | 500 k | **1,183** | 1,958 | 1,950 | 1.7x | 1.6x |
| kv | 2 M | **1,866** | 2,001 | 2,336 | 1.1x | 1.3x |
| objstore | 20 k | **474** | 1,946 | 984 | 4.1x | 2.1x |
| objstore | 100 k | **783** | 2,071 | 1,403 | 2.6x | 1.8x |
| objstore | 500 k | **1,220** | 2,071 | 1,882 | 1.7x | 1.5x |
| objstore | 2 M | **1,707** | 1,994 | 2,222 | 1.2x | 1.3x |
| fs | 20 k | **562** | 1,911 | 976 | 3.4x | 1.7x |
| fs | 100 k | **768** | 1,984 | 1,339 | 2.6x | 1.7x |
| fs | 500 k | **1,264** | 2,081 | 1,844 | 1.6x | 1.5x |
| fs | 2 M | **1,796** | 1,969 | 2,199 | 1.1x | 1.2x |

## Large-tree stress: 50 M keys

This is a local stress snapshot, not the Linux v0.3.0 release
baseline above. It was run on an Apple M3 Pro macOS development
machine, so Linux `io_uring` is not active. The profile is still
the same fair hot-persistent comparison: all three engines are
file-backed with WAL enabled, no per-op fsync, and a warm service
after preload.

Command shape:

```bash
TMPDIR="/Volumes/mac Ds - Data/tmp/holt-stress-50m" \
HOLT_STRESS_N=50000000 \
HOLT_STRESS_POINT_OPS=2000000 \
HOLT_STRESS_LIST_OPS=500000 \
HOLT_STRESS_LIST_TAKE=100 \
HOLT_STRESS_DIR_TAKE=8 \
cargo bench --bench stress -- objstore

TMPDIR="/Volumes/mac Ds - Data/tmp/holt-stress-50m" \
HOLT_STRESS_N=50000000 \
HOLT_STRESS_POINT_OPS=2000000 \
HOLT_STRESS_LIST_OPS=500000 \
HOLT_STRESS_LIST_TAKE=100 \
HOLT_STRESS_DIR_TAKE=8 \
cargo bench --bench stress -- fs
```

### Object-store metadata stress

| Bench | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|
| get | **2,282** | 155,025 | 283,215 | 68.0x | 124.1x |
| put | 4,957 | **3,891** | 264,627 | 0.78x | 53.4x |
| mixed | **3,982** | 4,162 | 73,826 | 1.05x | 18.5x |
| list keys, take 100 | 10,302 | 15,028 | **8,864** | 1.46x | 0.86x |
| list records, take 100 | **12,156** | n/a | n/a | n/a | n/a |
| list_dir, take 8 | **3,558** | 22,866 | 11,964 | 6.4x | 3.4x |

Holt shape after 50 M preload + 3 M point mutations:

| blobs | edges | max depth | avg depth | avg hops | max hops | avg fill | max fill |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 39,952 | 39,951 | 4 | 2.90 | 2.78 | 5 | 0.381 | 0.970 |

### Filesystem metadata stress

| Bench | Holt (ns) | RocksDB (ns) | SQLite (ns) | vs RocksDB | vs SQLite |
|---|---:|---:|---:|---:|---:|
| get | **1,822** | 66,632 | 174,126 | 36.6x | 95.6x |
| put | 3,958 | **3,905** | 226,390 | 0.99x | 57.2x |
| mixed | **3,072** | 4,962 | 127,024 | 1.6x | 41.3x |
| list keys, take 100 | 10,292 | 12,980 | **8,340** | 1.3x | 0.81x |
| list records, take 100 | **12,053** | n/a | n/a | n/a | n/a |
| list_dir, take 8 | **3,437** | 15,335 | 10,976 | 4.5x | 3.2x |

Holt shape after 50 M preload + 3 M point mutations:

| blobs | edges | max depth | avg depth | avg hops | max hops | avg fill | max fill |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 34,846 | 34,845 | 4 | 2.71 | 2.51 | 5 | 0.317 | 0.944 |

The 50 M run is useful because it stresses tree shape, route cache
coverage, and cross-blob traversal beyond the Criterion scale
curve. The good signal is that Holt's tree height stays bounded
(`max_depth=4`, `max_hops=5`) after 50 M preloaded records and 3 M
foreground point mutations. The honest limitation is that point
put is only competitive with RocksDB here, not a clear win. The
strong stress result is read scalability and metadata-native
`list_dir`; plain key-only `list` is not Holt's strongest cell and
SQLite is faster on this local warm-cache run.

## Interpretation

Holt is not a generic RocksDB replacement. It is a persistent ART
metadata engine for hierarchical keys.

- The strongest release claim is metadata semantics:
  `list_dir` and `metadata_mix` are tens to hundreds of times
  faster because Holt can skip whole subtrees after emitting a
  delimiter rollup.
- Point reads remain strong at 2 M keys across all three shapes.
  RocksDB suffers most at this tier; SQLite stays closer but still
  trails Holt.
- Point writes are competitive but are the current bottleneck. At
  2 M keys Holt wins every point-put cell in this run, but the
  margin against RocksDB is only 1.07-1.17x. The limiting work is
  CPU-side path descent, cross-blob traversal, dirty bookkeeping,
  and WAL submission, not data-file I/O.
- Plain prefix `list` is only a small win. Generic iterators are
  already good at "take 100 under a prefix"; Holt's structural
  advantage shows up when the operation has metadata semantics
  such as delimiter rollup.
