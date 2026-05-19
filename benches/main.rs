//! Criterion benchmarks comparing holt against RocksDB **and**
//! SQLite across three realistic shapes of metadata workload.
//!
//! ## Scenarios
//!
//! 1. **General KV** — 32-byte random keys, 64-byte random values.
//!    Baseline "anonymous bytes" workload.
//! 2. **Object storage metadata** — path-like keys
//!    (`bucket-NN/path/sub/file-NNNN.bin`) and small JSON-ish
//!    values carrying size / etag / storage class. Models the S3
//!    metadata tier (a holt-target workload).
//! 3. **Filesystem metadata** — `/usr/local/share/...` paths +
//!    32-byte packed inode bodies (size + mtime + mode + uid + gid).
//!    Models a POSIX metadata server.
//!
//! Each scenario runs three operations:
//! - **get**: random lookup over a pre-loaded dataset.
//! - **put**: random key replacement (in-place update).
//! - **mixed**: 50% get / 50% put, key chosen at random.
//!
//! The dataset size is intentionally large enough
//! (`N_KEYS = 20 000`) to spread across **multiple holt blobs**
//! (~5–7 × 512 KB), so the bench exercises `BlobNode` crossings
//! rather than single-blob descent.
//!
//! ## Fairness
//!
//! All three engines run in their "no-WAL, batched flush" mode
//! for the memory variant, and "WAL on, no per-op fsync" for the
//! persistent variant:
//!
//! | Mode       | holt                                        | RocksDB                              | SQLite                                              |
//! |------------|---------------------------------------------|--------------------------------------|-----------------------------------------------------|
//! | memory     | `TreeConfig::memory()`, `memory_flush_on_write=false` | `disable_wal=true`, `sync=false`     | `journal_mode=MEMORY`, `synchronous=OFF`, `:memory:` |
//! | persistent | `TreeConfig::new(dir)`, `memory_flush_on_write=false` | `WAL=on`, `sync=false`               | `journal_mode=WAL`, `synchronous=NORMAL`, file-backed |
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench main                     # full ~5 min sweep
//! cargo bench --bench main -- --quick --noplot # ~1 min smoke
//! cargo bench --bench main -- kv_get           # single scenario
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rusqlite::{params, Connection};
use tempfile::TempDir;

use holt::{RangeEntry, Tree, TreeConfig};
use rocksdb::{Direction, IteratorMode, Options, WriteOptions, DB};

// ---------------------------------------------------------------
// Workload configuration
// ---------------------------------------------------------------

/// Dataset size — large enough to spread across ≈ 5–7 holt blobs
/// for the kv/objstore/fs shapes (≈ 100 B/leaf amortised).
const N_KEYS: usize = 20_000;
const KV_KEY_LEN: usize = 32;
const KV_VAL_LEN: usize = 64;

const OBJSTORE_BUCKETS: usize = 32;
const OBJSTORE_FILES_PER_BUCKET: usize = N_KEYS / OBJSTORE_BUCKETS;

const FS_DIRS: usize = 16;
const FS_FILES_PER_DIR: usize = N_KEYS / FS_DIRS;

const SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

// ---------------------------------------------------------------
// Dataset generators
// ---------------------------------------------------------------

fn gen_kv_dataset() -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..N_KEYS)
        .map(|_| {
            let mut k = vec![0u8; KV_KEY_LEN];
            let mut v = vec![0u8; KV_VAL_LEN];
            rng.fill_bytes(&mut k);
            rng.fill_bytes(&mut v);
            (k, v)
        })
        .collect()
}

fn gen_objstore_dataset() -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut pairs = Vec::with_capacity(N_KEYS);
    for b in 0..OBJSTORE_BUCKETS {
        for f in 0..OBJSTORE_FILES_PER_BUCKET {
            let key = format!("bucket-{b:02}/path/sub/file-{f:04}.bin").into_bytes();
            // Fixed-length (~60 bytes) JSON-ish metadata — zero-padded
            // numeric fields so every value rounds to the same
            // extent footprint (lets in-place updates re-use the
            // existing leaf extent without leaking).
            let value = format!(
                "{{\"size\":{:08},\"etag\":\"{:08x}\",\"class\":\"STD\"}}",
                f * 1000 + b * 100,
                (b * 1000 + f) as u32,
            )
            .into_bytes();
            pairs.push((key, value));
        }
    }
    pairs
}

fn gen_fs_dataset() -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut pairs = Vec::with_capacity(N_KEYS);
    for d in 0..FS_DIRS {
        for f in 0..FS_FILES_PER_DIR {
            let key = format!("/usr/local/share/category-{d}/file-{f:04}").into_bytes();
            // Packed inode body: size(8) + mtime(8) + mode(4) +
            // uid(4) + gid(4) + nlink(4) = 32 bytes.
            let mut value = Vec::with_capacity(32);
            value.extend_from_slice(&((f as u64) * 1024).to_le_bytes());
            value.extend_from_slice(&(1_700_000_000u64 + f as u64).to_le_bytes());
            value.extend_from_slice(&0o644u32.to_le_bytes());
            value.extend_from_slice(&1000u32.to_le_bytes());
            value.extend_from_slice(&1000u32.to_le_bytes());
            value.extend_from_slice(&1u32.to_le_bytes());
            pairs.push((key, value));
        }
    }
    pairs
}

// ---------------------------------------------------------------
// Engine setup
// ---------------------------------------------------------------

fn make_holt() -> Tree {
    let mut cfg = TreeConfig::memory();
    cfg.memory_flush_on_write = false; // batched flushes; matches RocksDB / SQLite no-WAL mode
    Tree::open(cfg).expect("holt open")
}

/// Persistent holt on a temp dir. Each `put` lands in the WAL
/// writer's buffer + BufferManager cache; the persistent
/// backend only gets a `pwrite` at spillover or `checkpoint()`.
/// Matches RocksDB's `WAL=on, sync=false` (per-op durable to OS
/// page cache, not fsync'd) and SQLite's `WAL + synchronous=NORMAL`.
fn make_holt_persistent() -> (Tree, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let cfg = TreeConfig::new(dir.path());
    let tree = Tree::open(cfg).expect("holt persistent open");
    (tree, dir)
}

fn make_rocksdb() -> (DB, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(64 * 1024 * 1024);
    opts.set_max_write_buffer_number(2);
    opts.set_compression_type(rocksdb::DBCompressionType::None);
    let db = DB::open(&opts, dir.path()).expect("rocksdb open");
    (db, dir)
}

fn rocksdb_write_opts() -> WriteOptions {
    let mut wo = WriteOptions::default();
    wo.disable_wal(true);
    wo.set_sync(false);
    wo
}

/// Same as `rocksdb_write_opts` but with the WAL enabled — the
/// per-op durability profile we compare holt's persistent
/// backend against (`WAL=on, sync=false`).
fn rocksdb_write_opts_persistent() -> WriteOptions {
    let mut wo = WriteOptions::default();
    wo.disable_wal(false);
    wo.set_sync(false);
    wo
}

/// `:memory:` SQLite with the journal off — matches our "no-WAL,
/// batched flush" memory bench mode.
fn make_sqlite_memory() -> Connection {
    let conn = Connection::open_in_memory().expect("sqlite open");
    conn.execute_batch(
        "PRAGMA journal_mode = MEMORY;\n\
         PRAGMA synchronous = OFF;\n\
         PRAGMA cache_size = -65536;\n\
         CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID;",
    )
    .expect("sqlite pragmas + schema");
    conn
}

/// File-backed SQLite with WAL on and `synchronous = NORMAL` —
/// matches RocksDB's `WAL=on, sync=false` durability profile.
fn make_sqlite_persistent() -> (Connection, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let conn = Connection::open(dir.path().join("bench.db")).expect("sqlite open");
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;\n\
         PRAGMA synchronous = NORMAL;\n\
         PRAGMA cache_size = -65536;\n\
         CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID;",
    )
    .expect("sqlite pragmas + schema");
    (conn, dir)
}

fn preload_holt(tree: &Tree, pairs: &[(Vec<u8>, Vec<u8>)]) {
    for (k, v) in pairs {
        tree.put(k, v).expect("holt put");
    }
}

fn preload_rocksdb(db: &DB, pairs: &[(Vec<u8>, Vec<u8>)]) {
    let wo = rocksdb_write_opts();
    for (k, v) in pairs {
        db.put_opt(k, v, &wo).expect("rocksdb put");
    }
}

fn preload_sqlite(conn: &Connection, pairs: &[(Vec<u8>, Vec<u8>)]) {
    // Bulk-load inside one transaction — without this SQLite's
    // per-statement implicit transactions dominate setup time at
    // 20k rows.
    let tx = conn.unchecked_transaction().expect("tx");
    {
        let mut stmt = tx
            .prepare("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
            .expect("prep");
        for (k, v) in pairs {
            stmt.execute(params![k.as_slice(), v.as_slice()])
                .expect("insert");
        }
    }
    tx.commit().expect("commit");
}

// ---------------------------------------------------------------
// Per-scenario benches
// ---------------------------------------------------------------

fn bench_scenario(c: &mut Criterion, name: &str, pairs: &[(Vec<u8>, Vec<u8>)]) {
    let key_count = pairs.len();

    // ---- get ----
    {
        let mut group = c.benchmark_group(format!("{name}_get"));
        group.throughput(Throughput::Elements(1));

        let holt = make_holt();
        preload_holt(&holt, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        group.bench_function("holt", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                black_box(holt.get(black_box(k)).unwrap());
            });
        });

        let (db, _dir) = make_rocksdb();
        preload_rocksdb(&db, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        group.bench_function("rocksdb", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                black_box(db.get(black_box(k)).unwrap());
            });
        });

        let conn = make_sqlite_memory();
        preload_sqlite(&conn, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        group.bench_function("sqlite", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                let mut stmt = conn.prepare_cached("SELECT v FROM kv WHERE k = ?").unwrap();
                let v: Vec<u8> = stmt
                    .query_row(params![k.as_slice()], |row| row.get(0))
                    .unwrap();
                black_box(v);
            });
        });

        group.finish();
    }

    // ---- put (update) ----
    {
        let mut group = c.benchmark_group(format!("{name}_put"));
        group.throughput(Throughput::Elements(1));

        let holt = make_holt();
        preload_holt(&holt, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 2);
        group.bench_function("holt", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                black_box(holt.put(black_box(k), black_box(v)).unwrap());
            });
        });

        let (db, _dir) = make_rocksdb();
        preload_rocksdb(&db, pairs);
        let wo = rocksdb_write_opts();
        let mut rng = StdRng::seed_from_u64(SEED + 2);
        group.bench_function("rocksdb", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                let _: () = db.put_opt(black_box(k), black_box(v), &wo).unwrap();
                black_box(());
            });
        });

        let conn = make_sqlite_memory();
        preload_sqlite(&conn, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 2);
        group.bench_function("sqlite", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                let mut stmt = conn
                    .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
                    .unwrap();
                stmt.execute(params![k.as_slice(), v.as_slice()]).unwrap();
                black_box(());
            });
        });

        group.finish();
    }

    // ---- mixed (50% get / 50% put) ----
    {
        let mut group = c.benchmark_group(format!("{name}_mixed"));
        group.throughput(Throughput::Elements(1));

        let holt = make_holt();
        preload_holt(&holt, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        group.bench_function("holt", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    black_box(holt.get(black_box(k)).unwrap());
                } else {
                    black_box(holt.put(black_box(k), black_box(v)).unwrap());
                }
            });
        });

        let (db, _dir) = make_rocksdb();
        preload_rocksdb(&db, pairs);
        let wo = rocksdb_write_opts();
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        group.bench_function("rocksdb", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    black_box(db.get(black_box(k)).unwrap());
                } else {
                    let _: () = db.put_opt(black_box(k), black_box(v), &wo).unwrap();
                    black_box(());
                }
            });
        });

        let conn = make_sqlite_memory();
        preload_sqlite(&conn, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        group.bench_function("sqlite", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    let mut stmt = conn.prepare_cached("SELECT v FROM kv WHERE k = ?").unwrap();
                    let v: Vec<u8> = stmt
                        .query_row(params![k.as_slice()], |row| row.get(0))
                        .unwrap();
                    black_box(v);
                } else {
                    let mut stmt = conn
                        .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
                        .unwrap();
                    stmt.execute(params![k.as_slice(), v.as_slice()]).unwrap();
                    black_box(());
                }
            });
        });

        group.finish();
    }
}

// Persistent variant: all three engines on disk with WAL/durability
// on (each at the `sync=off` profile — durable past a process crash
// but not a power loss).
fn bench_scenario_persistent(c: &mut Criterion, name: &str, pairs: &[(Vec<u8>, Vec<u8>)]) {
    let key_count = pairs.len();

    // ---- get ----
    {
        let mut group = c.benchmark_group(format!("{name}_persist_get"));
        group.throughput(Throughput::Elements(1));

        let (holt, _dir) = make_holt_persistent();
        preload_holt(&holt, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 11);
        group.bench_function("holt", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                black_box(holt.get(black_box(k)).unwrap());
            });
        });

        let (db, _dir) = make_rocksdb();
        let wo = rocksdb_write_opts_persistent();
        for (k, v) in pairs {
            db.put_opt(k, v, &wo).expect("rocksdb preload");
        }
        let mut rng = StdRng::seed_from_u64(SEED + 11);
        group.bench_function("rocksdb", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                black_box(db.get(black_box(k)).unwrap());
            });
        });

        let (conn, _dir) = make_sqlite_persistent();
        preload_sqlite(&conn, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 11);
        group.bench_function("sqlite", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                let mut stmt = conn.prepare_cached("SELECT v FROM kv WHERE k = ?").unwrap();
                let v: Vec<u8> = stmt
                    .query_row(params![k.as_slice()], |row| row.get(0))
                    .unwrap();
                black_box(v);
            });
        });

        group.finish();
    }

    // ---- put ----
    {
        let mut group = c.benchmark_group(format!("{name}_persist_put"));
        group.throughput(Throughput::Elements(1));

        let (holt, _dir) = make_holt_persistent();
        preload_holt(&holt, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 12);
        group.bench_function("holt", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                black_box(holt.put(black_box(k), black_box(v)).unwrap());
            });
        });

        let (db, _dir) = make_rocksdb();
        let wo = rocksdb_write_opts_persistent();
        for (k, v) in pairs {
            db.put_opt(k, v, &wo).expect("rocksdb preload");
        }
        let mut rng = StdRng::seed_from_u64(SEED + 12);
        group.bench_function("rocksdb", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                let _: () = db.put_opt(black_box(k), black_box(v), &wo).unwrap();
                black_box(());
            });
        });

        let (conn, _dir) = make_sqlite_persistent();
        preload_sqlite(&conn, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 12);
        group.bench_function("sqlite", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                let mut stmt = conn
                    .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
                    .unwrap();
                stmt.execute(params![k.as_slice(), v.as_slice()]).unwrap();
                black_box(());
            });
        });

        group.finish();
    }

    // ---- mixed ----
    {
        let mut group = c.benchmark_group(format!("{name}_persist_mixed"));
        group.throughput(Throughput::Elements(1));

        let (holt, _dir) = make_holt_persistent();
        preload_holt(&holt, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 13);
        group.bench_function("holt", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    black_box(holt.get(black_box(k)).unwrap());
                } else {
                    black_box(holt.put(black_box(k), black_box(v)).unwrap());
                }
            });
        });

        let (db, _dir) = make_rocksdb();
        let wo = rocksdb_write_opts_persistent();
        for (k, v) in pairs {
            db.put_opt(k, v, &wo).expect("rocksdb preload");
        }
        let mut rng = StdRng::seed_from_u64(SEED + 13);
        group.bench_function("rocksdb", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    black_box(db.get(black_box(k)).unwrap());
                } else {
                    let _: () = db.put_opt(black_box(k), black_box(v), &wo).unwrap();
                    black_box(());
                }
            });
        });

        let (conn, _dir) = make_sqlite_persistent();
        preload_sqlite(&conn, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 13);
        group.bench_function("sqlite", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    let mut stmt = conn.prepare_cached("SELECT v FROM kv WHERE k = ?").unwrap();
                    let v: Vec<u8> = stmt
                        .query_row(params![k.as_slice()], |row| row.get(0))
                        .unwrap();
                    black_box(v);
                } else {
                    let mut stmt = conn
                        .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
                        .unwrap();
                    stmt.execute(params![k.as_slice(), v.as_slice()]).unwrap();
                    black_box(());
                }
            });
        });

        group.finish();
    }
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// DURABLE-WRITE benches (Group A)
// ---------------------------------------------------------------
//
// Same shape as `*_persist_put` but every engine flips its
// strongest "per-op durability" flag:
//
// | engine   | knob set                                            |
// | -------- | --------------------------------------------------- |
// | holt     | `wal_sync_on_commit = true`                         |
// | rocksdb  | `WriteOptions::set_sync(true)`                      |
// | sqlite   | `PRAGMA journal_mode = WAL; synchronous = FULL`     |
//
// All three now `sync_data`/`fsync` per op, so the bench is
// dominated by storage-stack syscall latency. This is the
// honest "what about fsync workloads" answer to the
// `*_persist_put` baseline numbers (which use `sync=false` to
// match RocksDB / SQLite defaults).

/// Persistent holt with per-op WAL fsync.
fn make_holt_durable() -> (Tree, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut cfg = TreeConfig::new(dir.path());
    cfg.wal_sync_on_commit = true;
    let tree = Tree::open(cfg).expect("holt durable open");
    (tree, dir)
}

/// RocksDB `WriteOptions` with `sync=true` (every put fsyncs WAL).
fn rocksdb_write_opts_durable() -> WriteOptions {
    let mut wo = WriteOptions::default();
    wo.disable_wal(false);
    wo.set_sync(true);
    wo
}

/// SQLite file-backed with `synchronous=FULL` (every commit
/// fsyncs both data + WAL pages).
fn make_sqlite_durable() -> (Connection, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let conn = Connection::open(dir.path().join("bench.db")).expect("sqlite open");
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;\n\
         PRAGMA synchronous = FULL;\n\
         PRAGMA cache_size = -65536;\n\
         CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID;",
    )
    .expect("sqlite pragmas + schema");
    (conn, dir)
}

fn bench_durable_put(c: &mut Criterion, name: &str, pairs: &[(Vec<u8>, Vec<u8>)]) {
    let key_count = pairs.len();
    let mut group = c.benchmark_group(format!("{name}_durable_put"));
    group.throughput(Throughput::Elements(1));
    // Per-op fsync brings the wall clock to hundreds of µs on a
    // typical NVMe + ~ms on a slow SSD. Criterion's default 100
    // samples × 5 s warm-up explodes from there; cap it.
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(10));

    let (holt, _dir) = make_holt_durable();
    preload_holt(&holt, pairs);
    let mut rng = StdRng::seed_from_u64(SEED + 30);
    group.bench_function("holt", |b| {
        b.iter(|| {
            let idx = (rng.next_u32() as usize) % key_count;
            let (k, v) = &pairs[idx];
            black_box(holt.put(black_box(k), black_box(v)).unwrap());
        });
    });

    let (db, _dir) = make_rocksdb();
    let wo = rocksdb_write_opts_durable();
    for (k, v) in pairs {
        db.put_opt(k, v, &wo).expect("rocksdb preload");
    }
    let mut rng = StdRng::seed_from_u64(SEED + 30);
    group.bench_function("rocksdb", |b| {
        b.iter(|| {
            let idx = (rng.next_u32() as usize) % key_count;
            let (k, v) = &pairs[idx];
            db.put_opt(black_box(k), black_box(v), &wo).unwrap();
            black_box(());
        });
    });

    let (conn, _dir) = make_sqlite_durable();
    preload_sqlite(&conn, pairs);
    let mut rng = StdRng::seed_from_u64(SEED + 30);
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let idx = (rng.next_u32() as usize) % key_count;
            let (k, v) = &pairs[idx];
            let mut stmt = conn
                .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
                .unwrap();
            stmt.execute(params![k.as_slice(), v.as_slice()]).unwrap();
            black_box(());
        });
    });

    group.finish();
}

// LIST / range-scan benches
// ---------------------------------------------------------------
//
// These are the load-bearing test for the metadata-engine claim:
// `readdir(dir)` / S3 `LIST ?prefix=foo/&delimiter=/` is the
// dominant access pattern beyond raw point lookup. holt's
// `Tree::range` does an anchored descent + sequential leaf walk;
// RocksDB uses a seek + prefix-bounded iterator; SQLite uses a
// `WHERE k >= ? AND k < ?` range scan over the B-tree primary key.
// `kv` (random keys) has no prefix structure, so list benches are
// only meaningful for objstore + fs.

/// Smallest byte-string strictly greater than every string with
/// `prefix`. Used to bound SQLite range queries
/// (`WHERE k >= prefix AND k < prefix_upper(prefix)`). Caller must
/// guarantee the last byte of `prefix` is < `0xFF`.
fn prefix_upper(prefix: &[u8]) -> Vec<u8> {
    let mut u = prefix.to_vec();
    let last = u.last_mut().expect("prefix must be non-empty");
    *last = last
        .checked_add(1)
        .expect("prefix's last byte must be < 0xFF for this helper");
    u
}

fn bench_list_plain(
    c: &mut Criterion,
    group_name: &str,
    pairs: &[(Vec<u8>, Vec<u8>)],
    prefix: &[u8],
    take: usize,
) {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(take as u64));

    let holt = make_holt();
    preload_holt(&holt, pairs);
    group.bench_function("holt", |b| {
        b.iter(|| {
            let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(take);
            for entry in holt.range().prefix(black_box(prefix)) {
                match entry.unwrap() {
                    RangeEntry::Key { key, value } => out.push((key, value)),
                    RangeEntry::CommonPrefix(_) => unreachable!("no delimiter set"),
                    _ => unreachable!("RangeEntry got a new variant"),
                }
                if out.len() >= take {
                    break;
                }
            }
            black_box(out);
        });
    });

    let (db, _dir) = make_rocksdb();
    preload_rocksdb(&db, pairs);
    group.bench_function("rocksdb", |b| {
        b.iter(|| {
            let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(take);
            for item in db.iterator(IteratorMode::From(prefix, Direction::Forward)) {
                let (k, v) = item.unwrap();
                if !k.starts_with(prefix) {
                    break;
                }
                out.push((k.to_vec(), v.to_vec()));
                if out.len() >= take {
                    break;
                }
            }
            black_box(out);
        });
    });

    let conn = make_sqlite_memory();
    preload_sqlite(&conn, pairs);
    let upper = prefix_upper(prefix);
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut stmt = conn
                .prepare_cached("SELECT k, v FROM kv WHERE k >= ? AND k < ? ORDER BY k LIMIT ?")
                .unwrap();
            let rows = stmt
                .query_map(params![prefix, upper.as_slice(), take as i64], |row| {
                    Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .unwrap();
            let out: Vec<(Vec<u8>, Vec<u8>)> = rows.collect::<Result<_, _>>().unwrap();
            black_box(out);
        });
    });

    group.finish();
}

fn bench_list_plain_persistent(
    c: &mut Criterion,
    group_name: &str,
    pairs: &[(Vec<u8>, Vec<u8>)],
    prefix: &[u8],
    take: usize,
) {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(take as u64));

    let (holt, _dir) = make_holt_persistent();
    preload_holt(&holt, pairs);
    group.bench_function("holt", |b| {
        b.iter(|| {
            let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(take);
            for entry in holt.range().prefix(black_box(prefix)) {
                match entry.unwrap() {
                    RangeEntry::Key { key, value } => out.push((key, value)),
                    RangeEntry::CommonPrefix(_) => unreachable!("no delimiter set"),
                    _ => unreachable!("RangeEntry got a new variant"),
                }
                if out.len() >= take {
                    break;
                }
            }
            black_box(out);
        });
    });

    let (db, _dir) = make_rocksdb();
    let wo = rocksdb_write_opts_persistent();
    for (k, v) in pairs {
        db.put_opt(k, v, &wo).expect("rocksdb preload");
    }
    group.bench_function("rocksdb", |b| {
        b.iter(|| {
            let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(take);
            for item in db.iterator(IteratorMode::From(prefix, Direction::Forward)) {
                let (k, v) = item.unwrap();
                if !k.starts_with(prefix) {
                    break;
                }
                out.push((k.to_vec(), v.to_vec()));
                if out.len() >= take {
                    break;
                }
            }
            black_box(out);
        });
    });

    let (conn, _dir) = make_sqlite_persistent();
    preload_sqlite(&conn, pairs);
    let upper = prefix_upper(prefix);
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut stmt = conn
                .prepare_cached("SELECT k, v FROM kv WHERE k >= ? AND k < ? ORDER BY k LIMIT ?")
                .unwrap();
            let rows = stmt
                .query_map(params![prefix, upper.as_slice(), take as i64], |row| {
                    Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .unwrap();
            let out: Vec<(Vec<u8>, Vec<u8>)> = rows.collect::<Result<_, _>>().unwrap();
            black_box(out);
        });
    });

    group.finish();
}

/// S3-style `LIST` with delimiter rollup. holt has the dedup in
/// the engine via `RangeEntry::CommonPrefix`; RocksDB and SQLite
/// have to do app-level dedup over the raw range scan. None of
/// the three currently fast-forward past a rolled-up subtree
/// (holt v0.2 backlog item), so all three scan every leaf to find
/// the next distinct rollup — this is fair, just slow.
fn bench_list_delim(
    c: &mut Criterion,
    group_name: &str,
    pairs: &[(Vec<u8>, Vec<u8>)],
    prefix: &[u8],
    delim: u8,
    take: usize,
) {
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(take as u64));

    let holt = make_holt();
    preload_holt(&holt, pairs);
    group.bench_function("holt", |b| {
        b.iter(|| {
            let mut out: Vec<RangeEntry> = Vec::with_capacity(take);
            for entry in holt.range().prefix(black_box(prefix)).delimiter(delim) {
                out.push(entry.unwrap());
                if out.len() >= take {
                    break;
                }
            }
            black_box(out);
        });
    });

    let (db, _dir) = make_rocksdb();
    preload_rocksdb(&db, pairs);
    group.bench_function("rocksdb", |b| {
        b.iter(|| {
            let mut out: Vec<Vec<u8>> = Vec::with_capacity(take);
            let mut last_common: Option<Vec<u8>> = None;
            for item in db.iterator(IteratorMode::From(prefix, Direction::Forward)) {
                let (k, _v) = item.unwrap();
                if !k.starts_with(prefix) {
                    break;
                }
                let rest = &k[prefix.len()..];
                let emit: Vec<u8> = if let Some(idx) = rest.iter().position(|b| *b == delim) {
                    k[..=prefix.len() + idx].to_vec()
                } else {
                    k.to_vec()
                };
                if last_common.as_deref() != Some(emit.as_slice()) {
                    last_common = Some(emit.clone());
                    out.push(emit);
                    if out.len() >= take {
                        break;
                    }
                }
            }
            black_box(out);
        });
    });

    let conn = make_sqlite_memory();
    preload_sqlite(&conn, pairs);
    let upper = prefix_upper(prefix);
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let mut stmt = conn
                .prepare_cached("SELECT k FROM kv WHERE k >= ? AND k < ? ORDER BY k")
                .unwrap();
            let rows = stmt
                .query_map(params![prefix, upper.as_slice()], |row| {
                    row.get::<_, Vec<u8>>(0)
                })
                .unwrap();
            let mut out: Vec<Vec<u8>> = Vec::with_capacity(take);
            let mut last_common: Option<Vec<u8>> = None;
            for row in rows {
                let k = row.unwrap();
                let rest = &k[prefix.len()..];
                let emit: Vec<u8> = if let Some(idx) = rest.iter().position(|b| *b == delim) {
                    k[..=prefix.len() + idx].to_vec()
                } else {
                    k
                };
                if last_common.as_deref() != Some(emit.as_slice()) {
                    last_common = Some(emit.clone());
                    out.push(emit);
                    if out.len() >= take {
                        break;
                    }
                }
            }
            black_box(out);
        });
    });

    group.finish();
}

fn kv_benches(c: &mut Criterion) {
    let pairs = gen_kv_dataset();
    bench_scenario(c, "kv", &pairs);
    bench_scenario_persistent(c, "kv", &pairs);
    bench_durable_put(c, "kv", &pairs);
    // kv has no prefix structure — no list benches.
}

fn objstore_benches(c: &mut Criterion) {
    let pairs = gen_objstore_dataset();
    bench_scenario(c, "objstore", &pairs);
    bench_scenario_persistent(c, "objstore", &pairs);
    bench_durable_put(c, "objstore", &pairs);
    // Single-bucket listing: prefix narrows to ~625 files.
    bench_list_plain(c, "objstore_list", &pairs, b"bucket-05/", 100);
    bench_list_plain_persistent(c, "objstore_persist_list", &pairs, b"bucket-05/", 100);
    // S3-style top-level dir rollup: prefix `bucket-` + delim `/`
    // yields 32 distinct `bucket-NN/` common prefixes.
    bench_list_delim(c, "objstore_list_dir", &pairs, b"bucket-", b'/', 8);
}

fn fs_benches(c: &mut Criterion) {
    let pairs = gen_fs_dataset();
    bench_scenario(c, "fs", &pairs);
    bench_scenario_persistent(c, "fs", &pairs);
    bench_durable_put(c, "fs", &pairs);
    // Single-dir listing: prefix narrows to ~1250 files.
    bench_list_plain(c, "fs_list", &pairs, b"/usr/local/share/category-5/", 100);
    bench_list_plain_persistent(
        c,
        "fs_persist_list",
        &pairs,
        b"/usr/local/share/category-5/",
        100,
    );
    // Parent rollup: prefix `/usr/local/share/` + delim `/` yields
    // 16 distinct `/usr/local/share/category-N/` common prefixes.
    bench_list_delim(c, "fs_list_dir", &pairs, b"/usr/local/share/", b'/', 8);
}

criterion_group!(benches, kv_benches, objstore_benches, fs_benches);
criterion_main!(benches);
