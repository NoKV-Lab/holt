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
//! Each scenario runs point-access operations:
//! - **get**: random lookup over a pre-loaded dataset.
//! - **put**: random key replacement (in-place update).
//! - **mixed**: 50% get / 50% put, key chosen at random.
//!
//! The objstore/fs scenarios add metadata-native operations:
//! create+delete of a scratch entry, atomic rename round-trip,
//! plain prefix list, delimiter list-dir, and a weighted metadata
//! mix that combines stat/update/list/create/delete/rename.
//!
//! The dataset size is intentionally large enough
//! (`N_KEYS = 20 000`) to spread across **multiple holt blobs**
//! (~5–7 × 512 KB), so the bench exercises `BlobNode` crossings
//! rather than single-blob descent.
//!
//! ## Fairness
//!
//! All three engines run in their "no-WAL, batched flush" mode
//! for the memory variant, and "hot WAL, no per-op fsync" for the
//! persistent variant:
//!
//! | Mode       | holt                                        | RocksDB                              | SQLite                                              |
//! |------------|---------------------------------------------|--------------------------------------|-----------------------------------------------------|
//! | memory     | `TreeConfig::memory()`, `memory_flush_on_write=false` | `disable_wal=true`, `sync=false`     | `journal_mode=MEMORY`, `synchronous=OFF`, `:memory:` |
//! | persistent | `TreeConfig::new(dir)`, `WalCommit::Write` | `WAL=on`, `sync=false`               | `journal_mode=WAL`, `synchronous=OFF`, file-backed |
//!
//! The `*_persist_*` groups are intentionally hot-service
//! measurements. They do **not** claim to measure cold data-file
//! I/O after reopen.
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

use holt::{RangeEntry, Tree, TreeConfig, WalCommit};
use rocksdb::{Direction, IteratorMode, Options, WriteBatch, WriteOptions, DB};

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

/// Hot persistent holt on a temp dir. Each `put` lands in the
/// WAL file (OS page cache, no fsync) + BufferManager cache; the
/// persistent data file only gets a `pwrite` at spillover or
/// checkpoint. Matches RocksDB's `WAL=on, sync=false` and SQLite's
/// `WAL + synchronous=OFF` as a hot service profile, not as a cold
/// data-file I/O profile.
fn make_holt_persistent() -> (Tree, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut cfg = TreeConfig::new(dir.path());
    cfg.wal_commit = WalCommit::Write;
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

/// File-backed SQLite with WAL on and `synchronous = OFF` —
/// matches RocksDB's `WAL=on, sync=false` durability profile.
fn make_sqlite_persistent() -> (Connection, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let conn = Connection::open(dir.path().join("bench.db")).expect("sqlite open");
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;\n\
         PRAGMA synchronous = OFF;\n\
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
                holt.put(black_box(k), black_box(v)).unwrap();
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
                    holt.put(black_box(k), black_box(v)).unwrap();
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

// Hot persistent variant: all three engines are disk-backed with
// WAL on and per-op fsync off. This isolates foreground WAL/cache
// cost under a warm service state, not cold data-file I/O after
// reopen.
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
                holt.put(black_box(k), black_box(v)).unwrap();
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
                    holt.put(black_box(k), black_box(v)).unwrap();
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
// SCALE-CURVE benches (Group B)
// ---------------------------------------------------------------
//
// Runs get + put across four dataset sizes (20 k, 100 k, 500 k,
// 2 M) to expose how each engine's hot-path scales.
// The 500 k tier is intentionally chosen to exceed holt's
// default 64-blob buffer pool (≈ 32 MB resident), so we see
// real cache-miss + spillover behaviour rather than a
// fully-resident microbench.
//
// One representative workload per scale to keep total runtime
// bounded (Criterion's default 100 samples × 5 s warm-up over 4
// sizes × 3 engines × 2 ops = 24 sub-benches).
//
// At 2M the dataset is ~192 MB — 6× the default buffer pool —
// so every miss pays the full read_blob + descent cost, and the
// numbers reflect a working set the cache cannot hold.

const SCALE_SIZES: &[usize] = &[20_000, 100_000, 500_000, 2_000_000];

fn gen_kv_dataset_sized(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..n)
        .map(|_| {
            let mut k = vec![0u8; KV_KEY_LEN];
            let mut v = vec![0u8; KV_VAL_LEN];
            rng.fill_bytes(&mut k);
            rng.fill_bytes(&mut v);
            (k, v)
        })
        .collect()
}

/// S3-shape dataset of arbitrary size, preserving the
/// 32-buckets fan-out so the prefix-sharing characteristic
/// stays constant across tiers (more files per bucket at
/// larger tiers = deeper subtrees, but the same number of
/// distinct top-level prefixes).
fn gen_objstore_dataset_sized(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let buckets = OBJSTORE_BUCKETS;
    let files_per_bucket = n.div_ceil(buckets);
    let mut pairs = Vec::with_capacity(buckets * files_per_bucket);
    for b in 0..buckets {
        for f in 0..files_per_bucket {
            let key = format!("bucket-{b:02}/path/sub/file-{f:06}.bin").into_bytes();
            let value = format!(
                "{{\"size\":{:08},\"etag\":\"{:08x}\",\"class\":\"STD\"}}",
                f * 1000 + b * 100,
                (b.wrapping_mul(1000).wrapping_add(f)) as u32,
            )
            .into_bytes();
            pairs.push((key, value));
        }
    }
    pairs.truncate(n);
    pairs
}

/// POSIX-fs-shape dataset of arbitrary size, 16 directories
/// fan-out preserved.
fn gen_fs_dataset_sized(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    let dirs = FS_DIRS;
    let files_per_dir = n.div_ceil(dirs);
    let mut pairs = Vec::with_capacity(dirs * files_per_dir);
    for d in 0..dirs {
        for f in 0..files_per_dir {
            let key = format!("/usr/local/share/category-{d}/file-{f:06}").into_bytes();
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
    pairs.truncate(n);
    pairs
}

fn bench_scale_get_workload(
    c: &mut Criterion,
    name: &str,
    gen: impl Fn(usize) -> Vec<(Vec<u8>, Vec<u8>)>,
) {
    use criterion::BenchmarkId;

    let mut group = c.benchmark_group(format!("{name}_scale_get"));
    group.throughput(Throughput::Elements(1));

    for &n in SCALE_SIZES {
        let pairs = gen(n);
        let key_count = pairs.len();

        let holt = make_holt();
        preload_holt(&holt, &pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 40);
        group.bench_with_input(
            BenchmarkId::new("holt", format_n(n)),
            &key_count,
            |b, &kc| {
                b.iter(|| {
                    let (k, _) = &pairs[(rng.next_u32() as usize) % kc];
                    black_box(holt.get(black_box(k)).unwrap());
                });
            },
        );

        let (db, _dir) = make_rocksdb();
        preload_rocksdb(&db, &pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 40);
        group.bench_with_input(
            BenchmarkId::new("rocksdb", format_n(n)),
            &key_count,
            |b, &kc| {
                b.iter(|| {
                    let (k, _) = &pairs[(rng.next_u32() as usize) % kc];
                    black_box(db.get(black_box(k)).unwrap());
                });
            },
        );

        let conn = make_sqlite_memory();
        preload_sqlite(&conn, &pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 40);
        group.bench_with_input(
            BenchmarkId::new("sqlite", format_n(n)),
            &key_count,
            |b, &kc| {
                b.iter(|| {
                    let (k, _) = &pairs[(rng.next_u32() as usize) % kc];
                    let mut stmt = conn.prepare_cached("SELECT v FROM kv WHERE k = ?").unwrap();
                    let _: Vec<u8> = stmt.query_row(params![k.as_slice()], |r| r.get(0)).unwrap();
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn bench_scale_put_workload(
    c: &mut Criterion,
    name: &str,
    gen: impl Fn(usize) -> Vec<(Vec<u8>, Vec<u8>)>,
) {
    use criterion::BenchmarkId;

    let mut group = c.benchmark_group(format!("{name}_scale_put"));
    group.throughput(Throughput::Elements(1));

    for &n in SCALE_SIZES {
        let pairs = gen(n);
        let key_count = pairs.len();

        let holt = make_holt();
        preload_holt(&holt, &pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 41);
        group.bench_with_input(
            BenchmarkId::new("holt", format_n(n)),
            &key_count,
            |b, &kc| {
                b.iter(|| {
                    let (k, v) = &pairs[(rng.next_u32() as usize) % kc];
                    holt.put(black_box(k), black_box(v)).unwrap();
                });
            },
        );

        let (db, _dir) = make_rocksdb();
        preload_rocksdb(&db, &pairs);
        let wo = rocksdb_write_opts();
        let mut rng = StdRng::seed_from_u64(SEED + 41);
        group.bench_with_input(
            BenchmarkId::new("rocksdb", format_n(n)),
            &key_count,
            |b, &kc| {
                b.iter(|| {
                    let (k, v) = &pairs[(rng.next_u32() as usize) % kc];
                    db.put_opt(black_box(k), black_box(v), &wo).unwrap();
                    black_box(());
                });
            },
        );

        let conn = make_sqlite_memory();
        preload_sqlite(&conn, &pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 41);
        group.bench_with_input(
            BenchmarkId::new("sqlite", format_n(n)),
            &key_count,
            |b, &kc| {
                b.iter(|| {
                    let (k, v) = &pairs[(rng.next_u32() as usize) % kc];
                    let mut stmt = conn
                        .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
                        .unwrap();
                    stmt.execute(params![k.as_slice(), v.as_slice()]).unwrap();
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

#[inline]
fn format_n(n: usize) -> String {
    // "20k", "100k", "500k", "2M"
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else {
        format!("{}k", n / 1_000)
    }
}

// LIST / range-scan benches
// ---------------------------------------------------------------
//
// These are the load-bearing test for the metadata-engine claim:
// `readdir(dir)` / S3 `LIST ?prefix=foo/&delimiter=/` is the
// dominant access pattern beyond raw point lookup. holt's
// `Tree::range` does marker-aware lower-bound seek + sequential leaf walk;
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
                    RangeEntry::Key { key, value, .. } => out.push((key, value)),
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
                    RangeEntry::Key { key, value, .. } => out.push((key, value)),
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
/// have to do app-level dedup over the raw range scan. Holt also
/// fast-forwards past the rolled-up subtree after emitting each
/// `CommonPrefix`; RocksDB and SQLite deliberately stay on the
/// generic iterator/query shape because they expose no native
/// delimiter-list API.
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

// ---------------------------------------------------------------
// Metadata-native operation benches
// ---------------------------------------------------------------
//
// Point `put` answers "how fast is a same-size value update?".
// Objstore/fs metadata engines also live or die on create/unlink,
// rename/move, prefix list, and S3/POSIX directory rollups. These
// groups keep the comparison at the same no-WAL memory profile as
// `*_get` / `*_put`, but exercise operations a KV baseline has to
// synthesize at the application layer.

#[derive(Clone, Copy)]
struct MetadataBenchSpec<'a> {
    list_prefix: &'a [u8],
    dir_prefix: &'a [u8],
    delimiter: u8,
    list_take: usize,
    dir_take: usize,
    create_key: &'a [u8],
    rename_dst: &'a [u8],
}

fn bench_metadata_ops(
    c: &mut Criterion,
    name: &str,
    pairs: &[(Vec<u8>, Vec<u8>)],
    spec: MetadataBenchSpec<'_>,
) {
    bench_metadata_create_delete(c, &format!("{name}_create_delete"), pairs, spec);
    bench_metadata_rename(c, &format!("{name}_rename"), pairs, spec);
    bench_metadata_mix(c, &format!("{name}_metadata_mix"), pairs, spec);
}

fn bench_metadata_create_delete(
    c: &mut Criterion,
    group_name: &str,
    pairs: &[(Vec<u8>, Vec<u8>)],
    spec: MetadataBenchSpec<'_>,
) {
    let value = &pairs[0].1;
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(1));

    let holt = make_holt();
    preload_holt(&holt, pairs);
    group.bench_function("holt", |b| {
        b.iter(|| {
            holt.put(black_box(spec.create_key), black_box(value))
                .unwrap();
            assert!(holt.delete(black_box(spec.create_key)).unwrap());
        });
    });

    let (db, _dir) = make_rocksdb();
    preload_rocksdb(&db, pairs);
    let wo = rocksdb_write_opts();
    group.bench_function("rocksdb", |b| {
        b.iter(|| {
            db.put_opt(black_box(spec.create_key), black_box(value), &wo)
                .unwrap();
            db.delete_opt(black_box(spec.create_key), &wo).unwrap();
        });
    });

    let conn = make_sqlite_memory();
    preload_sqlite(&conn, pairs);
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            conn.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                params![spec.create_key, value.as_slice()],
            )
            .unwrap();
            conn.execute("DELETE FROM kv WHERE k = ?", params![spec.create_key])
                .unwrap();
        });
    });

    group.finish();
}

fn bench_metadata_rename(
    c: &mut Criterion,
    group_name: &str,
    pairs: &[(Vec<u8>, Vec<u8>)],
    spec: MetadataBenchSpec<'_>,
) {
    let src = &pairs[pairs.len() / 3].0;
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(1));

    let holt = make_holt();
    preload_holt(&holt, pairs);
    group.bench_function("holt", |b| {
        b.iter(|| {
            holt.rename(black_box(src), black_box(spec.rename_dst), false)
                .unwrap();
            holt.rename(black_box(spec.rename_dst), black_box(src), false)
                .unwrap();
        });
    });

    let (db, _dir) = make_rocksdb();
    preload_rocksdb(&db, pairs);
    let wo = rocksdb_write_opts();
    group.bench_function("rocksdb", |b| {
        b.iter(|| {
            rocksdb_rename_roundtrip(&db, &wo, src, spec.rename_dst);
        });
    });

    let conn = make_sqlite_memory();
    preload_sqlite(&conn, pairs);
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            sqlite_rename_roundtrip(&conn, src, spec.rename_dst);
        });
    });

    group.finish();
}

fn bench_metadata_mix(
    c: &mut Criterion,
    group_name: &str,
    pairs: &[(Vec<u8>, Vec<u8>)],
    spec: MetadataBenchSpec<'_>,
) {
    let key_count = pairs.len();
    let rename_src = &pairs[pairs.len() / 3].0;
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(1));

    let holt = make_holt();
    preload_holt(&holt, pairs);
    let mut rng = StdRng::seed_from_u64(SEED + 71);
    group.bench_function("holt", |b| {
        b.iter(|| {
            let r = rng.next_u32();
            let (k, v) = &pairs[(r as usize) % key_count];
            match r % 100 {
                0..=44 => {
                    black_box(holt.get(black_box(k)).unwrap());
                }
                45..=64 => {
                    holt.put(black_box(k), black_box(v)).unwrap();
                }
                65..=74 => {
                    black_box(holt_list_plain(&holt, spec.list_prefix, spec.list_take));
                }
                75..=84 => {
                    black_box(holt_list_dir(
                        &holt,
                        spec.dir_prefix,
                        spec.delimiter,
                        spec.dir_take,
                    ));
                }
                85..=94 => {
                    holt.put(black_box(spec.create_key), black_box(v)).unwrap();
                    assert!(holt.delete(black_box(spec.create_key)).unwrap());
                }
                _ => {
                    holt.rename(black_box(rename_src), black_box(spec.rename_dst), false)
                        .unwrap();
                    holt.rename(black_box(spec.rename_dst), black_box(rename_src), false)
                        .unwrap();
                }
            }
        });
    });

    let (db, _dir) = make_rocksdb();
    preload_rocksdb(&db, pairs);
    let wo = rocksdb_write_opts();
    let mut rng = StdRng::seed_from_u64(SEED + 71);
    group.bench_function("rocksdb", |b| {
        b.iter(|| {
            let r = rng.next_u32();
            let (k, v) = &pairs[(r as usize) % key_count];
            match r % 100 {
                0..=44 => {
                    black_box(db.get(black_box(k)).unwrap());
                }
                45..=64 => {
                    db.put_opt(black_box(k), black_box(v), &wo).unwrap();
                    black_box(());
                }
                65..=74 => {
                    black_box(rocksdb_list_plain(&db, spec.list_prefix, spec.list_take));
                }
                75..=84 => {
                    black_box(rocksdb_list_dir(
                        &db,
                        spec.dir_prefix,
                        spec.delimiter,
                        spec.dir_take,
                    ));
                }
                85..=94 => {
                    db.put_opt(black_box(spec.create_key), black_box(v), &wo)
                        .unwrap();
                    db.delete_opt(black_box(spec.create_key), &wo).unwrap();
                }
                _ => rocksdb_rename_roundtrip(&db, &wo, rename_src, spec.rename_dst),
            }
        });
    });

    let conn = make_sqlite_memory();
    preload_sqlite(&conn, pairs);
    let upper = prefix_upper(spec.list_prefix);
    let dir_upper = prefix_upper(spec.dir_prefix);
    let mut rng = StdRng::seed_from_u64(SEED + 71);
    group.bench_function("sqlite", |b| {
        b.iter(|| {
            let r = rng.next_u32();
            let (k, v) = &pairs[(r as usize) % key_count];
            match r % 100 {
                0..=44 => {
                    let mut stmt = conn.prepare_cached("SELECT v FROM kv WHERE k = ?").unwrap();
                    let v: Vec<u8> = stmt
                        .query_row(params![k.as_slice()], |row| row.get(0))
                        .unwrap();
                    black_box(v);
                }
                45..=64 => {
                    conn.execute(
                        "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                        params![k.as_slice(), v.as_slice()],
                    )
                    .unwrap();
                    black_box(());
                }
                65..=74 => {
                    black_box(sqlite_list_plain(
                        &conn,
                        spec.list_prefix,
                        &upper,
                        spec.list_take,
                    ));
                }
                75..=84 => {
                    black_box(sqlite_list_dir(
                        &conn,
                        spec.dir_prefix,
                        &dir_upper,
                        spec.delimiter,
                        spec.dir_take,
                    ));
                }
                85..=94 => {
                    conn.execute(
                        "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                        params![spec.create_key, v.as_slice()],
                    )
                    .unwrap();
                    conn.execute("DELETE FROM kv WHERE k = ?", params![spec.create_key])
                        .unwrap();
                }
                _ => sqlite_rename_roundtrip(&conn, rename_src, spec.rename_dst),
            }
        });
    });

    group.finish();
}

fn holt_list_plain(tree: &Tree, prefix: &[u8], take: usize) -> usize {
    let mut seen = 0;
    for entry in tree.range().prefix(prefix) {
        match entry.unwrap() {
            RangeEntry::Key { .. } => seen += 1,
            RangeEntry::CommonPrefix(_) => unreachable!("no delimiter set"),
            _ => unreachable!("RangeEntry got a new variant"),
        }
        if seen >= take {
            break;
        }
    }
    seen
}

fn holt_list_dir(tree: &Tree, prefix: &[u8], delim: u8, take: usize) -> usize {
    let mut seen = 0;
    for entry in tree.range().prefix(prefix).delimiter(delim) {
        match entry.unwrap() {
            RangeEntry::CommonPrefix(_) | RangeEntry::Key { .. } => seen += 1,
            _ => unreachable!("RangeEntry got a new variant"),
        }
        if seen >= take {
            break;
        }
    }
    seen
}

fn rocksdb_list_plain(db: &DB, prefix: &[u8], take: usize) -> usize {
    let mut seen = 0;
    for item in db.iterator(IteratorMode::From(prefix, Direction::Forward)) {
        let (k, _v) = item.unwrap();
        if !k.starts_with(prefix) {
            break;
        }
        seen += 1;
        if seen >= take {
            break;
        }
    }
    seen
}

fn rocksdb_list_dir(db: &DB, prefix: &[u8], delim: u8, take: usize) -> usize {
    let mut seen = 0;
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
            last_common = Some(emit);
            seen += 1;
            if seen >= take {
                break;
            }
        }
    }
    seen
}

fn sqlite_list_plain(conn: &Connection, prefix: &[u8], upper: &[u8], take: usize) -> usize {
    let mut stmt = conn
        .prepare_cached("SELECT k FROM kv WHERE k >= ? AND k < ? ORDER BY k LIMIT ?")
        .unwrap();
    let rows = stmt
        .query_map(params![prefix, upper, take as i64], |row| {
            row.get::<_, Vec<u8>>(0)
        })
        .unwrap();
    rows.count()
}

fn sqlite_list_dir(
    conn: &Connection,
    prefix: &[u8],
    upper: &[u8],
    delim: u8,
    take: usize,
) -> usize {
    let mut stmt = conn
        .prepare_cached("SELECT k FROM kv WHERE k >= ? AND k < ? ORDER BY k")
        .unwrap();
    let rows = stmt
        .query_map(params![prefix, upper], |row| row.get::<_, Vec<u8>>(0))
        .unwrap();
    let mut seen = 0;
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
            last_common = Some(emit);
            seen += 1;
            if seen >= take {
                break;
            }
        }
    }
    seen
}

fn rocksdb_rename_roundtrip(db: &DB, wo: &WriteOptions, src: &[u8], dst: &[u8]) {
    rocksdb_rename(db, wo, src, dst);
    rocksdb_rename(db, wo, dst, src);
}

fn rocksdb_rename(db: &DB, wo: &WriteOptions, src: &[u8], dst: &[u8]) {
    let value = db.get(src).unwrap().expect("rename source exists");
    assert!(db.get(dst).unwrap().is_none(), "rename destination absent");
    let mut batch = WriteBatch::default();
    batch.delete(src);
    batch.put(dst, value);
    db.write_opt(batch, wo).unwrap();
}

fn sqlite_rename_roundtrip(conn: &Connection, src: &[u8], dst: &[u8]) {
    sqlite_rename(conn, src, dst);
    sqlite_rename(conn, dst, src);
}

fn sqlite_rename(conn: &Connection, src: &[u8], dst: &[u8]) {
    let tx = conn.unchecked_transaction().unwrap();
    let value: Vec<u8> = tx
        .query_row("SELECT v FROM kv WHERE k = ?", params![src], |row| {
            row.get(0)
        })
        .expect("rename source exists");
    let dst_exists: i64 = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM kv WHERE k = ?)",
            params![dst],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(dst_exists, 0, "rename destination absent");
    tx.execute("DELETE FROM kv WHERE k = ?", params![src])
        .unwrap();
    tx.execute(
        "INSERT INTO kv (k, v) VALUES (?, ?)",
        params![dst, value.as_slice()],
    )
    .unwrap();
    tx.commit().unwrap();
}

fn kv_benches(c: &mut Criterion) {
    let pairs = gen_kv_dataset();
    bench_scenario(c, "kv", &pairs);
    bench_scenario_persistent(c, "kv", &pairs);
    // kv has no prefix structure — no list benches.
}

fn objstore_benches(c: &mut Criterion) {
    let pairs = gen_objstore_dataset();
    bench_scenario(c, "objstore", &pairs);
    bench_scenario_persistent(c, "objstore", &pairs);
    bench_metadata_ops(
        c,
        "objstore",
        &pairs,
        MetadataBenchSpec {
            list_prefix: b"bucket-05/",
            dir_prefix: b"bucket-",
            delimiter: b'/',
            list_take: 100,
            dir_take: 8,
            create_key: b"bucket-31/path/sub/__holt_create_delete__.bin",
            rename_dst: b"bucket-31/path/sub/__holt_rename_dst__.bin",
        },
    );
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
    bench_metadata_ops(
        c,
        "fs",
        &pairs,
        MetadataBenchSpec {
            list_prefix: b"/usr/local/share/category-5/",
            dir_prefix: b"/usr/local/share/",
            delimiter: b'/',
            list_take: 100,
            dir_take: 8,
            create_key: b"/usr/local/share/category-15/__holt_create_delete__",
            rename_dst: b"/usr/local/share/category-15/__holt_rename_dst__",
        },
    );
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

fn scale_benches(c: &mut Criterion) {
    // kv = random 32-byte keys (ART anti-pattern, no prefix sharing)
    bench_scale_get_workload(c, "kv", gen_kv_dataset_sized);
    bench_scale_put_workload(c, "kv", gen_kv_dataset_sized);
    // objstore = S3-shape path keys with ~30-byte shared prefix per bucket
    bench_scale_get_workload(c, "objstore", gen_objstore_dataset_sized);
    bench_scale_put_workload(c, "objstore", gen_objstore_dataset_sized);
    // fs = POSIX paths with very long common prefix per directory
    bench_scale_get_workload(c, "fs", gen_fs_dataset_sized);
    bench_scale_put_workload(c, "fs", gen_fs_dataset_sized);
}

criterion_group!(
    benches,
    kv_benches,
    objstore_benches,
    fs_benches,
    scale_benches
);
criterion_main!(benches);
