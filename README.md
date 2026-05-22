# holt

[![Crates.io](https://img.shields.io/crates/v/holt.svg)](https://crates.io/crates/holt)
[![Docs.rs](https://docs.rs/holt/badge.svg)](https://docs.rs/holt)
[![CI](https://github.com/feichai0017/holt/actions/workflows/ci.yml/badge.svg)](https://github.com/feichai0017/holt/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.82-blue.svg)](https://github.com/feichai0017/holt/blob/main/Cargo.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A carefully crafted **adaptive radix tree** for path-shaped metadata.

> ⚠️ **Pre-1.0 (v0.3.1 released).** The public API is now narrow and
> SemVer-stable inside a minor release, but minor releases may
> still break source compatibility before 1.0. Pin the exact
> published version in your `Cargo.toml` (`holt = "=0.3.1"`) until 1.0
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

**Pre-1.0, actively maintained.** The v0.3 metadata-engine core is
landed: insert / lookup / erase / rename / range / atomic / compact,
multi-blob crossings, online maintenance gates, persistent store
with `O_DIRECT` and optional Linux `io_uring`, logical WAL
with group commit, sharded buffer manager, 3-thread background
checkpointer, SIMD CRC32 + node scans, and stateful `Tree::range`
with prefix, `start_after`, and S3 delimiter rollup.

See [`CHANGELOG.md`](CHANGELOG.md) for the v0.3 release notes and
[`ROADMAP.md`](ROADMAP.md) for post-0.3 direction.

`cargo bench --bench main` runs a side-by-side comparison with
RocksDB and SQLite across three metadata workload shapes — see
[`benches/README.md`](benches/README.md) for the methodology and
headline numbers.

## Usage

Add holt to your `Cargo.toml`:

```toml
[dependencies]
holt = "=0.3.1"
```

The supported user surface is deliberately small:
`TreeBuilder`, `Tree`, `TreeConfig`, `Storage`, `RangeBuilder`,
`RangeEntry`, `RangeIter`, `KeyRangeBuilder`, `KeyRangeEntry`,
`KeyRangeIter`, `AtomicBatch`, `Record`, `RecordVersion`,
`CheckpointConfig`,
`TreeStats` / related stats structs, `Error` / `Result`, and the
custom-store surface (`BlobStore`, `MemoryBlobStore`,
`FileBlobStore`, `AlignedBlobBuf`, `BlobGuid`). Internal
layout, WAL, walker, and buffer-manager modules are not public API.

### Open a tree

Two storage modes; same `TreeBuilder`, one knob switches between them:

```rust
use holt::{TreeBuilder, WalCommit};

// File-backed production mode, Unix-only — Linux `O_DIRECT`,
// macOS `F_NOCACHE`. The directory is created if missing.
let tree = TreeBuilder::new("/var/lib/myapp/meta.holt")
    .buffer_pool_size(128)        // pinned 512 KB blobs (default 64)
    .wal_commit(WalCommit::Enqueue) // default async journal acknowledgement
    .open()?;

// In-memory — volatile, dies with the last handle. The path
// argument becomes informational once `.memory()` flips the
// mode. Good for tests, sidecar caches, ephemeral session stores.
let tree = TreeBuilder::new("scratch").memory().open()?;
```

### Single-key CRUD

Bytes in, bytes out. `put` and `delete` are the **blind hot paths** —
they write or tombstone without reading the existing leaf, and return
`()` / `bool` respectively. `insert` and `remove` are the **explicit
returning variants** that pay one extra leaf read to hand you the
prior value. Use `put`/`delete` by default; reach for `insert`/`remove`
only where you actually consume the previous bytes. `rename` is
atomic and errors if `dst` exists unless `force = true`.

```rust
// Blind hot paths — recommended default.
tree.put(b"img/01.jpg", b"rgb_data_blob_id_abc")?;

let value: Option<Vec<u8>> = tree.get(b"img/01.jpg")?;
assert_eq!(value.as_deref(), Some(&b"rgb_data_blob_id_abc"[..]));

let existed: bool = tree.delete(b"img/01.jpg")?;
assert!(existed);

// Returning variants — pay the read-back cost only when you need it.
let prev: Option<Vec<u8>> = tree.insert(b"img/01.jpg", b"v2")?;
assert!(prev.is_none()); // we just deleted it above
let dropped: Option<Vec<u8>> = tree.remove(b"img/01.jpg")?;
assert_eq!(dropped.as_deref(), Some(&b"v2"[..]));

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

### Durability

Per-op writes land in the journal worker + BufferManager cache.
The WAL acknowledgement boundary is explicit:

- **`WalCommit::Enqueue`** — default. Return after the journal
  worker queue accepts the encoded record.
- **`WalCommit::Write`** — return after WAL bytes reach the OS
  page cache, with no per-op `sync_data`. This is the benchmark
  profile matching RocksDB `WAL on, sync=false`.
- **`WalCommit::Sync`** — return after `sync_data`; concurrent
  writers can share one fsync through group commit.

Disk-truth advances at:

- **`Tree::checkpoint()`** — flush the journal (`sync_data`),
  write dirty blobs through to the store, `fdatasync` the store,
  truncate the WAL. Call this at your own application checkpoint
  cadence.
- **WAL auto-flush** — once the WAL writer's pending buffer
  crosses 64 KB it drains to the OS page cache (no `sync_data`).
  Bounds in-memory buffering even if `checkpoint` is rare.

```rust
tree.checkpoint()?;   // flush WAL + write through + truncate
```

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
