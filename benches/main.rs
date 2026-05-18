//! Criterion benchmarks comparing artisan against RocksDB across
//! three realistic shapes of metadata workload.
//!
//! ## Scenarios
//!
//! 1. **General KV** — 32-byte random keys, 64-byte random values.
//!    Baseline "anonymous bytes" workload.
//! 2. **Object storage metadata** — path-like keys
//!    (`bucket-NN/path/sub/file-NNNN.bin`) and small JSON-ish
//!    values carrying size / etag / storage class. Models the S3
//!    metadata tier (an artisan/NSS-target workload).
//! 3. **Filesystem metadata** — `/usr/local/share/...` paths +
//!    32-byte packed inode bodies (size + mtime + mode + uid + gid).
//!    Models a POSIX metadata server.
//!
//! Each scenario runs three operations:
//! - **get**: random lookup over a pre-loaded 2000-key dataset.
//! - **put**: random key replacement (in-place update).
//! - **mixed**: 50% get / 50% put, key chosen at random.
//!
//! ## Fairness
//!
//! Both engines run in their "no-WAL, batched flush" mode:
//! - artisan: `TreeConfig::memory()` with `flush_on_write = false`.
//!   Mutations stay in the in-memory cached root blob; `checkpoint()`
//!   flushes through the backend.
//! - RocksDB: temp-dir database, `disable_wal=true`, `sync=false`,
//!   64 MB memtable, compression disabled. Equivalent to "memtable-
//!   only writes during the bench window."
//!
//! ## Caveat: artisan single-blob cap
//!
//! As of Stage 2d phase A, artisan auto-spillover (multi-blob
//! insertion) is not yet wired — the working set must fit in a
//! single 512 KB blob. N=2000 keys with the sizes above
//! comfortably fits (~200-250 KB). Phase B unlocks larger workloads.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench main
//! # Pick a single group:
//! cargo bench --bench main -- kv_get
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use tempfile::TempDir;

use artisan::{Tree, TreeConfig};
use rocksdb::{Options, WriteOptions, DB};

// ---------------------------------------------------------------
// Workload configuration
// ---------------------------------------------------------------

const N_KEYS: usize = 2000;
const KV_KEY_LEN: usize = 32;
const KV_VAL_LEN: usize = 64;

const OBJSTORE_BUCKETS: usize = 16;
const OBJSTORE_FILES_PER_BUCKET: usize = N_KEYS / OBJSTORE_BUCKETS;

const FS_DIRS: usize = 8;
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

fn make_artisan() -> Tree {
    let mut cfg = TreeConfig::memory();
    cfg.flush_on_write = false; // batched flushes; matches RocksDB no-WAL mode
    Tree::open(cfg).expect("artisan open")
}

/// Persistent artisan on a temp dir. `flush_on_write = false` so
/// each `put` lands in the BufferManager cache; the persistent
/// backend only gets a `pwrite` at spillover or `checkpoint()`.
/// Matches RocksDB's `WAL=on, sync=false` (per-op durable to OS
/// page cache, not fsync'd).
fn make_artisan_persistent() -> (Tree, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut cfg = TreeConfig::new(dir.path());
    cfg.flush_on_write = false;
    let tree = Tree::open(cfg).expect("artisan persistent open");
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
/// per-op durability profile we compare artisan's persistent
/// backend against (`WAL=on, sync=false`).
fn rocksdb_write_opts_persistent() -> WriteOptions {
    let mut wo = WriteOptions::default();
    wo.disable_wal(false);
    wo.set_sync(false);
    wo
}

fn preload_artisan(tree: &Tree, pairs: &[(Vec<u8>, Vec<u8>)]) {
    for (k, v) in pairs {
        tree.put(k, v).expect("artisan put");
    }
}

fn preload_rocksdb(db: &DB, pairs: &[(Vec<u8>, Vec<u8>)]) {
    let wo = rocksdb_write_opts();
    for (k, v) in pairs {
        db.put_opt(k, v, &wo).expect("rocksdb put");
    }
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

        let artisan = make_artisan();
        preload_artisan(&artisan, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        group.bench_function("artisan", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                black_box(artisan.get(black_box(k)).unwrap());
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

        group.finish();
    }

    // ---- put (update) ----
    {
        let mut group = c.benchmark_group(format!("{name}_put"));
        group.throughput(Throughput::Elements(1));

        let artisan = make_artisan();
        preload_artisan(&artisan, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 2);
        group.bench_function("artisan", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                black_box(artisan.put(black_box(k), black_box(v)).unwrap());
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

        group.finish();
    }

    // ---- mixed (50% get / 50% put) ----
    {
        let mut group = c.benchmark_group(format!("{name}_mixed"));
        group.throughput(Throughput::Elements(1));

        let artisan = make_artisan();
        preload_artisan(&artisan, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        group.bench_function("artisan", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    black_box(artisan.get(black_box(k)).unwrap());
                } else {
                    black_box(artisan.put(black_box(k), black_box(v)).unwrap());
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

        group.finish();
    }
}

// Persistent variant: both engines on disk with WAL/durability on
// (RocksDB: WAL enabled, fsync off; artisan: PersistentBackend,
// flush_on_write = false — each `put` stays in the BM cache,
// only spillover + `checkpoint()` hit disk).
fn bench_scenario_persistent(c: &mut Criterion, name: &str, pairs: &[(Vec<u8>, Vec<u8>)]) {
    let key_count = pairs.len();

    // ---- get ----
    {
        let mut group = c.benchmark_group(format!("{name}_persist_get"));
        group.throughput(Throughput::Elements(1));

        let (artisan, _dir) = make_artisan_persistent();
        preload_artisan(&artisan, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 11);
        group.bench_function("artisan", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, _) = &pairs[idx];
                black_box(artisan.get(black_box(k)).unwrap());
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

        group.finish();
    }

    // ---- put ----
    {
        let mut group = c.benchmark_group(format!("{name}_persist_put"));
        group.throughput(Throughput::Elements(1));

        let (artisan, _dir) = make_artisan_persistent();
        preload_artisan(&artisan, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 12);
        group.bench_function("artisan", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (k, v) = &pairs[idx];
                black_box(artisan.put(black_box(k), black_box(v)).unwrap());
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

        group.finish();
    }

    // ---- mixed ----
    {
        let mut group = c.benchmark_group(format!("{name}_persist_mixed"));
        group.throughput(Throughput::Elements(1));

        let (artisan, _dir) = make_artisan_persistent();
        preload_artisan(&artisan, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 13);
        group.bench_function("artisan", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (k, v) = &pairs[idx];
                if r & 1 == 0 {
                    black_box(artisan.get(black_box(k)).unwrap());
                } else {
                    black_box(artisan.put(black_box(k), black_box(v)).unwrap());
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

        group.finish();
    }
}

fn kv_benches(c: &mut Criterion) {
    let pairs = gen_kv_dataset();
    bench_scenario(c, "kv", &pairs);
    bench_scenario_persistent(c, "kv", &pairs);
}

fn objstore_benches(c: &mut Criterion) {
    let pairs = gen_objstore_dataset();
    bench_scenario(c, "objstore", &pairs);
    bench_scenario_persistent(c, "objstore", &pairs);
}

fn fs_benches(c: &mut Criterion) {
    let pairs = gen_fs_dataset();
    bench_scenario(c, "fs", &pairs);
    bench_scenario_persistent(c, "fs", &pairs);
}

criterion_group!(benches, kv_benches, objstore_benches, fs_benches);
criterion_main!(benches);
