# holt

[![Crates.io](https://img.shields.io/crates/v/holt.svg)](https://crates.io/crates/holt)
[![Docs.rs](https://docs.rs/holt/badge.svg)](https://docs.rs/holt)
[![CI](https://github.com/feichai0017/holt/actions/workflows/ci.yml/badge.svg)](https://github.com/feichai0017/holt/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.82-blue.svg)](https://github.com/feichai0017/holt/blob/main/Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A carefully crafted **adaptive radix tree** for path-shaped metadata.

> ⚠️ **Pre-1.0 (v0.5.3 released).** The public API is now narrow and
> SemVer-stable inside a minor release, but minor releases may
> still break source compatibility before 1.0. Pin the exact
> published version in your `Cargo.toml` (`holt = "=0.5.3"`) until 1.0
> stabilises the surface.

`holt` is an embedded Rust library for storing **hierarchical
keys** — file paths, S3 object names, multi-tenant namespaces,
time-bucketed identifiers — with sub-microsecond lookups, per-blob
concurrency, and crash-safe persistence.

It targets workloads where:

- Keys are **hierarchical / path-shaped** (so prefix compression pays).
- The dominant access is **point lookup + prefix range scan**.
- Concurrency is **high** (many readers + writers across disjoint
  subtrees).
- Latency is **micro-critical** — no LSM compaction stalls, no
  single-writer locks.

It is **not** a general-purpose KV store; if you need full-text
or vector similarity, reach for the right tool. For this shape,
holt's strongest wins are metadata-native operations: delimiter
directory rollup and mixed metadata workloads. In the v0.3 Linux
release run, `objstore_list_dir` is **151×** faster than RocksDB
and `fs_list_dir` is **268×** faster; `objstore_metadata_mix` is
**43×** faster than RocksDB and `fs_metadata_mix` is **66×**
faster. Point reads stay ahead through 2 M keys; point writes are
competitive but intentionally not the headline claim — see
[`benches/RESULTS.md`](benches/RESULTS.md).

## Why "holt"?

A **holt** (Old English *holt*) is a small grove or copse — a
self-contained collection of trees on a single piece of ground.
That maps directly to the design: each `holt::Tree` is **one** ART
made of **many** 512 KB blob frames, bounded and self-contained,
grown by repeated `splitBlob`. Short to type, distinct from other
crates, easy to say.

## When to reach for holt

| Engine        | Data structure        | Persistence       | Concurrency        | Notes                                                |
|---------------|-----------------------|-------------------|--------------------|------------------------------------------------------|
| LMDB          | B+tree                | mmap              | Single-writer MVCC | Battle-tested; page chasing for short hot keys.      |
| RocksDB       | LSM                   | SST + WAL         | MVCC               | Compaction stalls; large hot dataset is RAM-heavy.   |
| SQLite        | B-tree                | File              | Single writer      | Convenient, but writer is the bottleneck under load. |
| Sled          | Hybrid LSM            | Log-structured    | Lock-free          | Rust-native, largely unmaintained.                   |
| **holt**      | **Adaptive Radix Tree** | **512 KB blobs** | **Per-blob 3-mode latch** | **Path compression + lookup is O(key.len)** |

ART's lookup cost is `O(key.len)`, not `O(log N)`. For short hot keys
(< 64 bytes), that beats any tree-based competitor. The per-blob
HybridLatch lets N readers traverse disjoint subtrees in parallel
without coordinating.

## Project status

**Pre-1.0, actively maintained.** The metadata-engine core is landed:
put / get / delete / rename / range / atomic / compact, multi-blob
crossings, online maintenance gates, persistent store with `O_DIRECT`
and optional Linux `io_uring`, logical WAL with group commit, sharded
buffer manager, 3-thread background checkpointer, SIMD CRC32 + node
scans, copy-on-write snapshots, and stateful `Tree::range` with prefix,
`start_after`, and S3 delimiter rollup.

**0.5** adds a two-axis durability model (`Durability::Wal` /
`Durability::StateMachine`, orthogonal to storage) and the pieces a
*replicated* metadata service needs: crash-consistent on-disk recovery
with **no WAL** when an external log owns durability
(`DB::commit_durable` / `durable_applied_index`), whole-DB checkpoint
export/install, and the metadata-shaped fast paths `DB::scatter`,
`Tree::put_many_if_absent`, and per-scan `ScanStats`. See the
[Durability](#durability) section below.

See [`CHANGELOG.md`](CHANGELOG.md) for the release notes and
[`ROADMAP.md`](ROADMAP.md) for direction.

`cargo bench --manifest-path benches/Cargo.toml --bench main`
runs a side-by-side comparison with RocksDB, SQLite, and sled
across three metadata workload shapes — see
[`benches/README.md`](benches/README.md) for the methodology and
headline numbers.

## Usage

Add holt to your `Cargo.toml`:

```toml
[dependencies]
holt = "0.5"
```

The supported user surface is deliberately small:
`DB`, `DBAtomicBatch`, `DBView`, `Scatter`, `TreeBuilder`, `Tree`, `TreeConfig`,
`Storage`, `Durability`, `RangeBuilder`, `RangeEntry`, `RangeIter`,
`KeyRangeBuilder`, `KeyRangeEntry`, `KeyRangeIter`, `ScanStats`, `Snapshot`,
`View`, `AtomicBatch`, `Record`, `RecordVersion`, `PutOutcome`,
`KeyPathBuf`, `KeyPrefixBuf`, `KeyPathError`, `CheckpointConfig`,
`CheckpointImage`, `TreeStats` / `DBStats` / related stats structs,
`Error` / `Result`, and the custom-store surface (`BlobStore`,
`MemoryBlobStore`, `FileBlobStore`, `AlignedBlobBuf`, `BlobGuid`,
`DurableManifest`). Internal layout, WAL, walker, and buffer-manager
modules are not public API.

### Open a tree

Two storage modes; same `TreeBuilder`, one knob switches between them:

```rust
use holt::{Durability, TreeBuilder};

// File-backed production mode, Unix-only — Linux `O_DIRECT`,
// macOS `F_NOCACHE`. The directory is created if missing.
let tree = TreeBuilder::new("/var/lib/myapp/meta.holt")
    .buffer_pool_size(512)                          // optional: 512 blobs = 256 MiB
    .durability(Durability::Wal { sync: false })    // async group-commit WAL (default)
    .open()?;

// In-memory — volatile, dies with the last handle. The path
// argument becomes informational once `.memory()` flips the
// mode. Good for tests, sidecar caches, ephemeral session stores.
let tree = TreeBuilder::new("scratch").memory().open()?;
```

File-backed trees default to 256 resident 512 KB blobs (128 MiB).
In-memory trees keep the smaller 64-blob default (32 MiB).

### Path-shaped keys

The core API takes byte keys. For object-store and filesystem-like
metadata, `KeyPathBuf` builds canonical slash-separated keys without
hand-written string formatting. It is optional: callers with opaque
keys can keep passing raw bytes.

```rust
use holt::KeyPathBuf;

let mut key = KeyPathBuf::with_namespace(b"o")?;
key.push(b"photos")?;
key.push(b"users")?;
key.push(b"alice")?;
key.push(b"01.jpg")?;

tree.put(key.as_bytes(), b"object_meta")?;

let mut prefix = KeyPathBuf::with_namespace(b"o")?;
prefix.push(b"photos")?;
prefix.push(b"users")?;
let prefix = prefix.into_prefix();

for entry in tree.scan_keys(prefix.as_bytes()).delimiter(b'/') {
    println!("{:?}", entry?);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Single-key CRUD

Bytes in, bytes out. `put` and `delete` are the hot paths: they write
or tombstone without reading the existing leaf. `rename` is atomic and
errors if `dst` exists unless `force = true`.

```rust
tree.put(b"img/01.jpg", b"rgb_data_blob_id_abc")?;

let value: Option<Vec<u8>> = tree.get(b"img/01.jpg")?;
assert_eq!(value.as_deref(), Some(&b"rgb_data_blob_id_abc"[..]));

let existed: bool = tree.delete(b"img/01.jpg")?;
assert!(existed);

tree.put(b"old/path", b"v")?;
tree.rename(b"old/path", b"new/path", /*force=*/ false)?;
```

### Conditional writes

`RecordVersion` is Holt's lightweight compare-and-set token. It is
not an MVCC timestamp and does not provide historical snapshot
reads; it only says "this live record still has the same leaf seq".

```rust
tree.put_if_absent(b"users/alice", b"{...}")?;
let record = tree.get_record(b"users/alice")?.unwrap();

let updated: bool = tree.compare_and_put(
    b"users/alice",
    record.version,
    b"{\"tier\":\"hot\"}",
)?;
assert!(updated);

let current = tree.get_version(b"users/alice")?.unwrap();
let deleted: bool = tree.delete_if_version(b"users/alice", current)?;
assert!(deleted);

// Point-in-time read helper for rmdir-style metadata checks.
assert!(tree.is_prefix_empty(b"users/alice/")?);
```

### Atomic batch

Multiple ops under one WAL record. Holt preflights rename and
conditional-write / prefix-emptiness guards before mutating,
applies the batch behind the tree-wide mutation gate, then emits
one Batch WAL record.
`Ok(true)` means committed, `Ok(false)` means a conditional guard
failed and nothing was published, and `Err` reports hard failures
such as a missing rename source or destination collision.

```rust
let template = tree.get_record(b"users/template")?.unwrap();
let committed = tree.atomic(|batch| {
    batch.assert_version(b"users/template", template.version);
    batch.put(b"users/alice", b"{...}");
    batch.put(b"users/bob",   b"{...}");
    batch.put_if_absent(b"users/new", b"{...}");
    batch.delete(b"users/legacy-account");
    batch.assert_prefix_empty(b"users/temp/");
    batch.rename(b"users/temp", b"users/permanent", true);
})?;
assert!(committed);
```

### Range scans and S3 delimiter rollup

`Tree::range` yields full records: key, value, and
`RecordVersion`. Use it when the list response needs metadata
bytes. `Tree::range_keys` / `Tree::scan_keys` use the same cursor
and delimiter semantics but skip value materialisation; use them
for name-only directory and object listings. Chain `.prefix()` to
anchor the scan, `.start_after()` for paging, and `.delimiter()`
for S3-style `?delimiter=/` rollup.

```rust
use holt::{KeyRangeEntry, RangeEntry};

// Full prefix scan — `take(50)` for a paged "first 50" view
// when the caller needs metadata bytes.
let first_50: Vec<_> = tree
    .range()
    .prefix(b"users/")
    .into_iter()
    .take(50)
    .map(|r| r.map(|e| match e {
        RangeEntry::Key { key, value, version } => (key, value, version),
        _ => unreachable!("no delimiter set"),
    }))
    .collect::<Result<_, _>>()?;

// S3-style "list one level" without copying values — leaves under
// `/img/` get emitted as Key; deeper paths roll up to a single
// `CommonPrefix` per distinct subdir.
for entry in tree.scan_keys(b"img/").delimiter(b'/') {
    match entry? {
        KeyRangeEntry::Key { key, .. }      => println!("file {key:?}"),
        KeyRangeEntry::CommonPrefix(prefix) => println!("dir  {prefix:?}/"),
        _ => {} // `KeyRangeEntry` is `#[non_exhaustive]`
    }
}
```

### Snapshot reads

`Tree::range` and `Tree::range_keys` are the hot restart-on-conflict
iterators. Use `Tree::view(prefix, |view| { ... })` when multiple
reads must observe one stable prefix snapshot. A view copies the
reachable blob frames for `prefix`, releases the live tree, then
runs point reads and scans against that private frame set.

```rust
tree.view(b"img/", |view| {
    let meta = view.get_record(b"img/01.jpg")?;
    let first_page: Vec<_> = view
        .range_keys()
        .delimiter(b'/')
        .into_iter()
        .take(100)
        .collect::<Result<_, _>>()?;
    Ok(())
})?;
```

### Durability

Durability is a policy orthogonal to storage — pick who owns the durable
record with `TreeConfig::durability` / `TreeBuilder::durability`:

**`Durability::Wal { sync }` — holt owns durability (single node).** Each write
updates the BufferManager cache and appends one logical WAL record. `sync: false`
(the default) returns after the group-commit worker queues the record; `sync: true`
waits for `sync_data`, and concurrent writers share one fsync. Disk-truth advances at:

- **Background checkpoint** — enabled by default for file-backed trees. It flushes
  the WAL, writes dirty blobs through to the store, syncs, applies pending deletes,
  and truncates the WAL once the pipeline is clean.
- **`Tree::checkpoint()`** — the same protocol, run synchronously.
- **WAL auto-flush** — drains the pending buffer to the OS page cache past 64 KB,
  bounding in-memory buffering even when checkpoints are rare.

```rust
tree.checkpoint()?;   // flush WAL + write through + truncate
```

**`Durability::StateMachine` — an external log owns durability (replicated).**
For a replicated state machine (e.g. a Raft apply loop): holt attaches **no WAL**
— the external log is the durable, replicated record. Writes apply to the
BufferManager; you periodically pin a crash-consistent on-disk checkpoint with
`commit_durable`, and on restart replay only the log *tail* past the recovered index:

```rust
use holt::{Durability, TreeConfig, DB};

let mut cfg = TreeConfig::new("/var/lib/meta");
cfg.durability = Durability::StateMachine;
let db = DB::open(cfg.clone())?;

// ... apply committed log entries to the state machine, then pin a checkpoint:
db.commit_durable(applied_index)?;          // atomic, crash-consistent on-disk point

// After a crash, reopen and replay only the tail past the recovered index:
let db = DB::open(cfg)?;
let replay_from = db.durable_applied_index()?;   // re-apply entries > replay_from
```

`commit_durable` takes a copy-on-write snapshot, persists its frame closure, and
atomically commits a manifest recording the durable roots plus `applied_index`
and the resume sequence — the manifest rename is the commit point, so a crash
lands on the previous or the new checkpoint, never a torn state (fault-injection
and SIGKILL-soak tested). Two StateMachine-only fast paths exploit the external
log already owning write order: `DB::scatter` (independent cross-family CAS with
no atomic barrier, each on its own per-key concurrent path) and atomic batches
that take the mutation gate *shared* instead of fencing concurrent range scans.

### Validation and observability

Holt has four validation layers:

- Unit and integration tests cover WAL replay, checkpoint recovery,
  range/view semantics, conditional atomic batches, and checkpoint
  failpoints.
- Property tests compare random operation streams against an oracle.
- `fuzz/` contains a `cargo-fuzz` target for the atomic/WAL/range
  model.
- `verified/` contains Verus specs for ART node shape, grow/shrink,
  prefix split, delimiter rollup bounds, virtual terminators, and leaf
  alignment.

Long lifecycle campaigns live in the explicit soak tool:

```sh
cargo run --manifest-path tools/soak/Cargo.toml --locked -- \
  --mode normal --dir target/holt-soak --reset \
  --duration-secs 3600 --keys 10000000 --ops 10000000 \
  --threads 8 --buffer-pool 256 --wal-sync false

cargo run --manifest-path tools/soak/Cargo.toml --locked -- \
  --mode crash --dir target/holt-soak-crash --reset \
  --duration-secs 21600 --keys 100000 --ops 1000000 \
  --buffer-pool 64 --wal-sync true \
  --kill-min-ms 100 --kill-max-ms 5000
```

With the `metrics` feature, `holt::metrics::render_prometheus` exposes
cache hit/miss, eviction/admission, route-cache, WAL work/debt,
checkpoint debt, dirty/pending-delete counts, and reopen WAL replay
time. The caller owns the HTTP endpoint; Holt only renders the
Prometheus text payload.

The GitHub workflows are split by cost:

- normal CI runs unit/integration/property tests, a bounded fuzz smoke,
  coverage, and a short soak smoke;
- nightly validation runs checkpoint/WAL fault tests, longer normal and
  crash soak campaigns, and a time-bounded fuzz campaign;
- Verus is available as a manual `Nightly Validation` option because
  hosted runners do not ship a Verus binary.

See [`TESTING.md`](TESTING.md) for the full test matrix and release-gate
commands.

See [`examples/`](examples/) for full programs:
[`basic_kv`](examples/basic_kv.rs),
[`filesystem_meta`](examples/filesystem_meta.rs),
[`session_store`](examples/session_store.rs),
[`s3_metadata`](examples/s3_metadata.rs).

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the deep dive.
The design draws on Leis et al.'s ART paper (ICDE 2013) for the
four-node-size scheme and LeanStore (ICDE 2018) for the HybridLatch
contract.

## Not on the roadmap

holt is **just the metadata engine** — single-node, embed-in-
your-process, Unix-only. Out of scope:

- **Windows** — `compile_error!`s the crate (Unix `O_DIRECT` /
  `F_NOCACHE` has no Windows analog worth carrying).
- **Object-storage frontend / S3 layer** — no RPC server, no
  multi-tenant bucket registry, no distributed checkpointer.
- **SQL / vector / full-text** — combine with a domain-appropriate
  engine (`+ FAISS` for vectors, `+ Tantivy` for full-text).
- **Replication / consensus** — build above; we'll expose hooks
  (change feed, snapshot transfer) but won't ship Raft.
- **Network server** — this is a library; wrap it in your own RPC.

## License

Licensed under the [MIT licence](LICENSE).
