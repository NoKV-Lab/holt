//! Fixed-size stress harness for large metadata datasets.
//!
//! This is intentionally not a Criterion benchmark. The default
//! run preloads 20M keys and then executes million-scale point and
//! list workloads, so a single run can take minutes and should be
//! invoked explicitly:
//!
//! ```sh
//! HOLT_STRESS_N=20000000 \
//! HOLT_STRESS_POINT_OPS=1000000 \
//! HOLT_STRESS_LIST_OPS=1000000 \
//! cargo bench --manifest-path benches/Cargo.toml --bench stress -- objstore
//! ```
//!
//! Profile: single-threaded, warm service, file-backed WAL enabled,
//! no per-op fsync by default, and the background checkpointer on.

use std::env;
use std::hint::black_box;
use std::os::unix::fs::MetadataExt;
#[cfg(feature = "comparators")]
use std::ops::Bound;
use std::time::{Duration, Instant};

#[cfg(feature = "comparators")]
use heed::types::Bytes;
#[cfg(feature = "comparators")]
use heed::{Database as LmdbDatabase, Env as LmdbEnv, EnvFlags, EnvOpenOptions};
use holt::{KeyRangeEntry, RangeEntry, Tree, TreeConfig};
use rand::{rngs::StdRng, RngCore, SeedableRng};
#[cfg(feature = "comparators")]
use rocksdb::{BlockBasedOptions, Cache, Options, WriteBatch, WriteOptions, DB};
#[cfg(feature = "comparators")]
use rusqlite::{params, Connection, OptionalExtension};
#[cfg(feature = "comparators")]
use sled::{Db as SledDb, Mode as SledMode};
use tempfile::TempDir;

const DEFAULT_N_KEYS: usize = 20_000_000;
const DEFAULT_POINT_OPS: usize = 1_000_000;
const DEFAULT_LIST_OPS: usize = 1_000_000;
const DEFAULT_LIST_TAKE: usize = 100;
const DEFAULT_DIR_TAKE: usize = 8;
const DEFAULT_BUFFER_POOL: usize = 64;
#[cfg(feature = "comparators")]
const PRELOAD_BATCH: usize = 10_000;
const SEED: u64 = 0xD15E_A5ED_2000_0001;
#[cfg(feature = "comparators")]
const DEFAULT_ENGINES: &str = "holt,rocksdb,sqlite";
#[cfg(not(feature = "comparators"))]
const DEFAULT_ENGINES: &str = "holt";

#[derive(Debug, Clone, Copy)]
enum Workload {
    Objstore,
    Fs,
}

impl Workload {
    fn parse(s: &str) -> Self {
        match s {
            "objstore" | "object" | "s3" => Self::Objstore,
            "fs" | "filesystem" | "posix" => Self::Fs,
            other => panic!("unknown workload `{other}`; use `objstore` or `fs`"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Objstore => "objstore",
            Self::Fs => "fs",
        }
    }

    fn key(self, idx: usize, n_keys: usize) -> Vec<u8> {
        match self {
            Self::Objstore => {
                let buckets = 256usize;
                let files_per_bucket = n_keys.div_ceil(buckets).max(1);
                let bucket = idx / files_per_bucket;
                let file = idx % files_per_bucket;
                format!(
                    "bucket-{bucket:03}/tenant-{tenant:02}/path/sub/file-{file:08}.bin",
                    tenant = bucket % 32
                )
                .into_bytes()
            }
            Self::Fs => {
                let dirs = 512usize;
                let files_per_dir = n_keys.div_ceil(dirs).max(1);
                let dir = idx / files_per_dir;
                let file = idx % files_per_dir;
                format!("/usr/local/share/category-{dir:03}/file-{file:08}").into_bytes()
            }
        }
    }

    fn value(self, idx: usize, revision: u64) -> Vec<u8> {
        match self {
            Self::Objstore => {
                let size = idx as u64 * 1024 + revision;
                let etag = (idx as u64)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add(revision);
                format!("{{\"size\":{size:016},\"etag\":\"{etag:016x}\",\"class\":\"STD\"}}")
                    .into_bytes()
            }
            Self::Fs => {
                let mut value = Vec::with_capacity(32);
                value.extend_from_slice(&((idx as u64) * 4096 + revision).to_le_bytes());
                value.extend_from_slice(&(1_700_000_000u64 + idx as u64 + revision).to_le_bytes());
                value.extend_from_slice(&0o644u32.to_le_bytes());
                value.extend_from_slice(&1000u32.to_le_bytes());
                value.extend_from_slice(&1000u32.to_le_bytes());
                value.extend_from_slice(&1u32.to_le_bytes());
                value
            }
        }
    }

    fn list_prefix(self) -> &'static [u8] {
        match self {
            Self::Objstore => b"bucket-005/",
            Self::Fs => b"/usr/local/share/category-005/",
        }
    }

    fn dir_prefix(self) -> &'static [u8] {
        match self {
            Self::Objstore => b"bucket-",
            Self::Fs => b"/usr/local/share/",
        }
    }

    fn delimiter(self) -> u8 {
        b'/'
    }
}

fn parse_bool_env(name: &str, s: &str) -> bool {
    match s {
        "1" | "true" | "yes" | "on" | "sync" | "fsync" => true,
        "0" | "false" | "no" | "off" | "enqueue" | "async" => false,
        other => panic!("unknown {name} value `{other}`; use true/false"),
    }
}

#[derive(Debug)]
struct StressConfig {
    workload: Workload,
    n_keys: usize,
    point_ops: usize,
    list_ops: usize,
    list_take: usize,
    dir_take: usize,
    buffer_pool_size: usize,
    wal_sync: bool,
    reopen_after_preload: bool,
    compact_after_preload: bool,
    rocksdb_direct: bool,
    get_threads: usize,
    drop_caches: bool,
    negative: bool,
    /// Persistent base dir for each engine's store (`<dir>/<engine>`) instead
    /// of a fresh TempDir. Lets a preload run and a later read-only run share
    /// the same on-disk store across processes (matched-memory cold reads).
    data_dir: Option<std::path::PathBuf>,
    /// Open the existing store and go straight to the timed phase — no
    /// preload, checkpoint, or compaction. Pairs with `data_dir`: preload once
    /// uncapped, then read under a tight cgroup memory cap.
    skip_preload: bool,
    engines: Vec<String>,
    ops: Vec<String>,
}

impl StressConfig {
    fn from_env() -> Self {
        let workload = env::args()
            .nth(1)
            .as_deref()
            .map(Workload::parse)
            .unwrap_or(Workload::Objstore);
        let engines = env::var("HOLT_STRESS_ENGINES")
            .unwrap_or_else(|_| DEFAULT_ENGINES.to_string())
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        let ops = env::var("HOLT_STRESS_OPS")
            .unwrap_or_else(|_| "get,put,mixed,list,list_records,list_dir".to_string())
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();

        Self {
            workload,
            n_keys: env_usize("HOLT_STRESS_N", DEFAULT_N_KEYS),
            point_ops: env_usize("HOLT_STRESS_POINT_OPS", DEFAULT_POINT_OPS),
            list_ops: env_usize("HOLT_STRESS_LIST_OPS", DEFAULT_LIST_OPS),
            list_take: env_usize("HOLT_STRESS_LIST_TAKE", DEFAULT_LIST_TAKE),
            dir_take: env_usize("HOLT_STRESS_DIR_TAKE", DEFAULT_DIR_TAKE),
            buffer_pool_size: env_usize("HOLT_STRESS_BUFFER_POOL", DEFAULT_BUFFER_POOL),
            wal_sync: env::var("HOLT_STRESS_WAL_SYNC")
                .as_deref()
                .map(|s| parse_bool_env("HOLT_STRESS_WAL_SYNC", s))
                .unwrap_or(false),
            reopen_after_preload: env::var("HOLT_STRESS_REOPEN_AFTER_PRELOAD")
                .as_deref()
                .map(|s| parse_bool_env("HOLT_STRESS_REOPEN_AFTER_PRELOAD", s))
                .unwrap_or(false),
            compact_after_preload: env::var("HOLT_STRESS_COMPACT_AFTER_PRELOAD")
                .as_deref()
                .map(|s| parse_bool_env("HOLT_STRESS_COMPACT_AFTER_PRELOAD", s))
                .unwrap_or(false),
            // Fair-comparison mode: make rocksdb ALSO bypass the OS page
            // cache (direct reads) and cap its block cache to holt's BM
            // size, so neither engine gets free RAM the other doesn't.
            // Without this, rocksdb's buffered I/O keeps the whole dataset
            // warm in the OS page cache while holt (O_DIRECT, 32 MB BM)
            // hits disk — an apples-to-oranges cold-read comparison.
            rocksdb_direct: env::var("HOLT_STRESS_ROCKSDB_DIRECT")
                .as_deref()
                .map(|s| parse_bool_env("HOLT_STRESS_ROCKSDB_DIRECT", s))
                .unwrap_or(false),
            // Run the cold get phase across N cloned Tree handles. Holt's
            // read path is lock-free (positional pread, shared-read
            // manifest lock), so N concurrent cold gets = N concurrent
            // preads = device queue depth N for free. If get throughput
            // scales with this, the cold-read parallelism is already
            // there with no engine change.
            get_threads: env_usize("HOLT_STRESS_GET_THREADS", 1).max(1),
            // Cold regime: drop the OS page cache before the timed phase so
            // the page-cache engines (lmdb, sqlite, buffered rocksdb) read
            // from the device, like holt's O_DIRECT path. Needs a pre-warmed
            // sudo timestamp (`sudo -v`); we shell out with `sudo -n` and
            // never touch the password here.
            drop_caches: env::var("HOLT_STRESS_DROP_CACHES")
                .as_deref()
                .map(|s| parse_bool_env("HOLT_STRESS_DROP_CACHES", s))
                .unwrap_or(false),
            // Negative-lookup mode: query keys that share a real key's full
            // prefix but are absent (a trailing sentinel byte). Exercises the
            // descent + per-blob bloom reject (holt) / leaf-miss (others).
            negative: env::var("HOLT_STRESS_NEGATIVE")
                .as_deref()
                .map(|s| parse_bool_env("HOLT_STRESS_NEGATIVE", s))
                .unwrap_or(false),
            data_dir: env::var_os("HOLT_STRESS_DATA_DIR").map(std::path::PathBuf::from),
            skip_preload: env::var("HOLT_STRESS_SKIP_PRELOAD")
                .as_deref()
                .map(|s| parse_bool_env("HOLT_STRESS_SKIP_PRELOAD", s))
                .unwrap_or(false),
            engines,
            ops,
        }
    }

    fn selected(&self, engine: &str) -> bool {
        self.engines
            .iter()
            .any(|s| s == "all" || s.eq_ignore_ascii_case(engine))
    }

    fn selected_op(&self, op: &str) -> bool {
        self.ops
            .iter()
            .any(|s| s == "all" || s.eq_ignore_ascii_case(op))
    }
}

#[derive(Debug)]
struct OpSample {
    key: Vec<u8>,
    value: Vec<u8>,
}

fn main() {
    let cfg = StressConfig::from_env();
    println!(
        "stress workload={} n_keys={} point_ops={} list_ops={} list_take={} dir_take={} buffer_pool={} engines={} ops={}",
        cfg.workload.name(),
        cfg.n_keys,
        cfg.point_ops,
        cfg.list_ops,
        cfg.list_take,
        cfg.dir_take,
        cfg.buffer_pool_size,
        cfg.engines.join(","),
        cfg.ops.join(","),
    );
    println!(
        "profile=single_thread,warm_service,persistent_wal,wal_sync={},checkpoint=enabled,reopen_after_preload={},compact_after_preload={},rocksdb_direct={},get_threads={},drop_caches={},negative={},skip_preload={},data_dir={}",
        cfg.wal_sync, cfg.reopen_after_preload, cfg.compact_after_preload, cfg.rocksdb_direct, cfg.get_threads, cfg.drop_caches, cfg.negative, cfg.skip_preload,
        cfg.data_dir.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "tempdir".to_string())
    );

    let samples = make_samples(cfg.workload, cfg.n_keys, cfg.point_ops, cfg.negative);
    reject_missing_comparators(&cfg);
    if cfg.selected("holt") {
        run_holt(&cfg, &samples);
    }
    #[cfg(feature = "comparators")]
    if cfg.selected("rocksdb") {
        run_rocksdb(&cfg, &samples);
    }
    #[cfg(feature = "comparators")]
    if cfg.selected("sqlite") {
        run_sqlite(&cfg, &samples);
    }
    #[cfg(feature = "comparators")]
    if cfg.selected("sled") {
        run_sled(&cfg, &samples);
    }
    #[cfg(feature = "comparators")]
    if cfg.selected("lmdb") {
        run_lmdb(&cfg, &samples);
    }
}

#[cfg(not(feature = "comparators"))]
fn reject_missing_comparators(cfg: &StressConfig) {
    for engine in ["rocksdb", "sqlite", "sled", "lmdb"] {
        if cfg.selected(engine) {
            panic!("stress comparator `{engine}` requires the `comparators` feature");
        }
    }
}

#[cfg(feature = "comparators")]
fn reject_missing_comparators(_cfg: &StressConfig) {}

/// A store directory: either an ephemeral `TempDir` (cleaned on drop) or a
/// caller-provided persistent path (kept across runs, for matched-memory
/// preload-once-then-read-capped). `path()` works for both so callers are
/// unchanged; the held `TempDir` keeps the temp path alive.
struct StoreDir {
    path: std::path::PathBuf,
    #[allow(dead_code)]
    tmp: Option<TempDir>,
}

impl StoreDir {
    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

fn store_dir(cfg: &StressConfig, engine: &str) -> StoreDir {
    match &cfg.data_dir {
        Some(base) => {
            let path = base.join(engine);
            std::fs::create_dir_all(&path).expect("store data dir");
            StoreDir { path, tmp: None }
        }
        None => {
            let tmp = TempDir::new().expect("store tempdir");
            StoreDir {
                path: tmp.path().to_path_buf(),
                tmp: Some(tmp),
            }
        }
    }
}

fn run_holt(cfg: &StressConfig, samples: &[OpSample]) {
    let dir = store_dir(cfg, "holt");
    let mut tree_cfg = TreeConfig::new(dir.path());
    tree_cfg.durability = holt::Durability::Wal { sync: cfg.wal_sync };
    tree_cfg.buffer_pool_size = cfg.buffer_pool_size;
    let mut tree = Tree::open(tree_cfg.clone()).expect("holt open");
    if !cfg.skip_preload {
        preload_holt(&tree, cfg.workload, cfg.n_keys);
        print_holt_shape("preload", &tree);
        tree.checkpoint().expect("holt preload checkpoint");
        print_holt_shape("ready", &tree);
        if cfg.compact_after_preload {
            compact_holt_until_settled(&tree);
            print_holt_shape("routed", &tree);
        }
    }
    if cfg.reopen_after_preload {
        drop(tree);
        tree = Tree::open(tree_cfg).expect("holt reopen");
        println!("holt_shape reopened deferred_until_after_timing=1");
    }

    print_space("holt", dir.path(), cfg.n_keys);
    maybe_drop_caches(cfg, "holt");
    if cfg.selected_op("get") {
        let rb0 = proc_read_bytes();
        let elapsed = time_get_holt_concurrent(&tree, samples, cfg.get_threads);
        let rb1 = proc_read_bytes();
        report("holt", "get", samples.len(), elapsed);
        print_read_amp("holt", "get", samples.len(), rb1.saturating_sub(rb0));
    }
    if cfg.selected_op("put") {
        report("holt", "put", samples.len(), time_put_holt(&tree, samples));
    }
    if cfg.selected_op("mixed") {
        report(
            "holt",
            "mixed",
            samples.len(),
            time_mixed_holt(&tree, samples),
        );
    }
    if cfg.selected_op("list") {
        report(
            "holt",
            "list",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(holt_list_keys(
                    &tree,
                    cfg.workload.list_prefix(),
                    cfg.list_take,
                ));
            }),
        );
    }
    if cfg.selected_op("list_records") {
        report(
            "holt",
            "list_records",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(holt_list_records(
                    &tree,
                    cfg.workload.list_prefix(),
                    cfg.list_take,
                ));
            }),
        );
    }
    if cfg.selected_op("list_dir") {
        report(
            "holt",
            "list_dir",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(holt_list_dir(
                    &tree,
                    cfg.workload.dir_prefix(),
                    cfg.workload.delimiter(),
                    cfg.dir_take,
                ));
            }),
        );
    }
    print_holt_shape("final", &tree);
}

#[cfg(feature = "comparators")]
fn run_rocksdb(cfg: &StressConfig, samples: &[OpSample]) {
    let rocksdb_dir = store_dir(cfg, "rocksdb");
    let mut db = make_rocksdb(&rocksdb_dir, cfg);
    if !cfg.skip_preload {
        preload_rocksdb(&db, cfg.workload, cfg.n_keys);
    }
    if cfg.reopen_after_preload {
        db.flush().expect("rocksdb preload flush before reopen");
        drop(db);
        db = make_rocksdb(&rocksdb_dir, cfg);
    }
    let wo = rocksdb_write_opts(cfg.wal_sync);

    print_space("rocksdb", rocksdb_dir.path(), cfg.n_keys);
    maybe_drop_caches(cfg, "rocksdb");
    if cfg.selected_op("get") {
        let rb0 = proc_read_bytes();
        let elapsed = time_get_rocksdb(&db, samples);
        let rb1 = proc_read_bytes();
        report("rocksdb", "get", samples.len(), elapsed);
        print_read_amp("rocksdb", "get", samples.len(), rb1.saturating_sub(rb0));
    }
    if cfg.selected_op("put") {
        report(
            "rocksdb",
            "put",
            samples.len(),
            time_put_rocksdb(&db, &wo, samples),
        );
    }
    if cfg.selected_op("mixed") {
        report(
            "rocksdb",
            "mixed",
            samples.len(),
            time_mixed_rocksdb(&db, &wo, samples),
        );
    }
    if cfg.selected_op("list") {
        report(
            "rocksdb",
            "list",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(rocksdb_list_plain(
                    &db,
                    cfg.workload.list_prefix(),
                    cfg.list_take,
                ));
            }),
        );
    }
    if cfg.selected_op("list_dir") {
        report(
            "rocksdb",
            "list_dir",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(rocksdb_list_dir(
                    &db,
                    cfg.workload.dir_prefix(),
                    cfg.workload.delimiter(),
                    cfg.dir_take,
                ));
            }),
        );
    }
}

#[cfg(feature = "comparators")]
fn run_sqlite(cfg: &StressConfig, samples: &[OpSample]) {
    let sqlite_dir = store_dir(cfg, "sqlite");
    let mut conn = make_sqlite_persistent(&sqlite_dir, cfg);
    if !cfg.skip_preload {
        preload_sqlite(&conn, cfg.workload, cfg.n_keys);
    }
    if cfg.reopen_after_preload {
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .expect("sqlite preload checkpoint before reopen");
        drop(conn);
        conn = make_sqlite_persistent(&sqlite_dir, cfg);
    }

    print_space("sqlite", sqlite_dir.path(), cfg.n_keys);
    maybe_drop_caches(cfg, "sqlite");
    if cfg.selected_op("get") {
        let rb0 = proc_read_bytes();
        let elapsed = time_get_sqlite(&conn, samples);
        let rb1 = proc_read_bytes();
        report("sqlite", "get", samples.len(), elapsed);
        print_read_amp("sqlite", "get", samples.len(), rb1.saturating_sub(rb0));
    }
    if cfg.selected_op("put") {
        report(
            "sqlite",
            "put",
            samples.len(),
            time_put_sqlite(&conn, samples),
        );
    }
    if cfg.selected_op("mixed") {
        report(
            "sqlite",
            "mixed",
            samples.len(),
            time_mixed_sqlite(&conn, samples),
        );
    }
    if cfg.selected_op("list") {
        report(
            "sqlite",
            "list",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(sqlite_list_plain(
                    &conn,
                    cfg.workload.list_prefix(),
                    &prefix_upper(cfg.workload.list_prefix()),
                    cfg.list_take,
                ));
            }),
        );
    }
    if cfg.selected_op("list_dir") {
        report(
            "sqlite",
            "list_dir",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(sqlite_list_dir(
                    &conn,
                    cfg.workload.dir_prefix(),
                    &prefix_upper(cfg.workload.dir_prefix()),
                    cfg.workload.delimiter(),
                    cfg.dir_take,
                ));
            }),
        );
    }
}

#[cfg(feature = "comparators")]
fn run_sled(cfg: &StressConfig, samples: &[OpSample]) {
    let sled_dir = store_dir(cfg, "sled");
    let mut db = make_sled_persistent(&sled_dir);
    if !cfg.skip_preload {
        preload_sled(&db, cfg.workload, cfg.n_keys);
    }
    if cfg.reopen_after_preload {
        db.flush().expect("sled preload flush before reopen");
        drop(db);
        db = make_sled_persistent(&sled_dir);
    }

    if cfg.selected_op("get") {
        report("sled", "get", samples.len(), time_get_sled(&db, samples));
    }
    if cfg.selected_op("put") {
        report("sled", "put", samples.len(), time_put_sled(&db, samples));
    }
    if cfg.selected_op("mixed") {
        report(
            "sled",
            "mixed",
            samples.len(),
            time_mixed_sled(&db, samples),
        );
    }
    if cfg.selected_op("list") {
        report(
            "sled",
            "list",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(sled_list_plain(
                    &db,
                    cfg.workload.list_prefix(),
                    cfg.list_take,
                ));
            }),
        );
    }
    if cfg.selected_op("list_dir") {
        report(
            "sled",
            "list_dir",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(sled_list_dir(
                    &db,
                    cfg.workload.dir_prefix(),
                    cfg.workload.delimiter(),
                    cfg.dir_take,
                ));
            }),
        );
    }
}

/// LMDB (via the `heed` safe wrapper) — the read-optimized embedded
/// B+tree that is holt's closest niche peer. Unlike the others, LMDB has
/// no tunable cache: it is mmap-only and rides the OS page cache. The
/// cold regime (drop_caches before timing) is the only fair way to
/// compare it against holt's O_DIRECT reads — see docs/paper/eval-design.md.
#[cfg(feature = "comparators")]
fn run_lmdb(cfg: &StressConfig, samples: &[OpSample]) {
    let lmdb_dir = store_dir(cfg, "lmdb");
    let mut lmdb = make_lmdb(&lmdb_dir, cfg);
    if !cfg.skip_preload {
        preload_lmdb(&lmdb, cfg.workload, cfg.n_keys);
    }
    if cfg.reopen_after_preload {
        lmdb.env
            .force_sync()
            .expect("lmdb preload sync before reopen");
        drop(lmdb);
        lmdb = make_lmdb(&lmdb_dir, cfg);
    }

    print_space("lmdb", lmdb_dir.path(), cfg.n_keys);
    maybe_drop_caches(cfg, "lmdb");
    if cfg.selected_op("get") {
        let rb0 = proc_read_bytes();
        let elapsed = time_get_lmdb(&lmdb, samples);
        let rb1 = proc_read_bytes();
        report("lmdb", "get", samples.len(), elapsed);
        print_read_amp("lmdb", "get", samples.len(), rb1.saturating_sub(rb0));
    }
    if cfg.selected_op("put") {
        report("lmdb", "put", samples.len(), time_put_lmdb(&lmdb, samples));
    }
    if cfg.selected_op("mixed") {
        report(
            "lmdb",
            "mixed",
            samples.len(),
            time_mixed_lmdb(&lmdb, samples),
        );
    }
    if cfg.selected_op("list") {
        report(
            "lmdb",
            "list",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(lmdb_list_plain(
                    &lmdb,
                    cfg.workload.list_prefix(),
                    cfg.list_take,
                ));
            }),
        );
    }
    if cfg.selected_op("list_dir") {
        report(
            "lmdb",
            "list_dir",
            cfg.list_ops,
            time_repeated(cfg.list_ops, || {
                black_box(lmdb_list_dir(
                    &lmdb,
                    cfg.workload.dir_prefix(),
                    cfg.workload.delimiter(),
                    cfg.dir_take,
                ));
            }),
        );
    }
}

fn make_samples(workload: Workload, n_keys: usize, ops: usize, negative: bool) -> Vec<OpSample> {
    assert!(n_keys > 0, "HOLT_STRESS_N must be > 0");
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut out = Vec::with_capacity(ops);
    for op in 0..ops {
        let idx = (rng.next_u64() as usize) % n_keys;
        let mut key = workload.key(idx, n_keys);
        if negative {
            // Append a sentinel byte: the result shares the real key's full
            // prefix (same descent + same target blob) but is absent, so the
            // lookup misses at the leaf / per-blob bloom.
            key.push(0x7E);
        }
        out.push(OpSample {
            key,
            value: workload.value(idx, op as u64 + 1),
        });
    }
    out
}

fn preload_holt(tree: &Tree, workload: Workload, n_keys: usize) {
    progress("holt", "preload", 0, n_keys);
    for i in 0..n_keys {
        let key = workload.key(i, n_keys);
        let value = workload.value(i, 0);
        tree.put(&key, &value).expect("holt preload put");
        if should_progress(i + 1, n_keys) {
            progress("holt", "preload", i + 1, n_keys);
        }
    }
}

/// Drive maintenance compaction to a fixpoint so the bulk-loaded,
/// write-once dataset gets routed (page-granular cold reads) before the
/// timed phase. A write-once load never accrues tombstone/dead-byte
/// churn, so blobs are born legacy; the maintenance scheduler routes
/// each settled blob the first time a seed pass sees it (see
/// `Tree::blob_wants_compaction`).
///
/// `Tree::compact` only (re)seeds the candidate queues when BOTH the
/// compaction and merge queues are empty, and a write-once preload
/// leaves merge candidates queued inline (parent blobs with crossings).
/// So the first few passes just *drain* the merge queue without bumping
/// `total_compactions`; routing only engages once a later pass finds
/// both queues empty, seeds, and compacts the would-route leaves. We
/// therefore wait for `total_compactions` to stay flat across a window
/// of consecutive passes rather than exiting on the first flat pass.
/// `MAX_PASSES` is a safety bound against a pathological never-settles
/// loop, not the expected exit.
fn compact_holt_until_settled(tree: &Tree) {
    const MAX_PASSES: usize = 1024;
    const STABLE_WINDOW: usize = 4;
    let mut prev = tree.stats().expect("holt stats").total_compactions;
    let mut stable = 0usize;
    for pass in 0..MAX_PASSES {
        tree.compact().expect("holt compact-after-preload");
        tree.checkpoint().expect("holt compact-after-preload checkpoint");
        let now = tree.stats().expect("holt stats").total_compactions;
        if now == prev {
            stable += 1;
            if stable >= STABLE_WINDOW {
                eprintln!(
                    "holt compact-after-preload settled after {} passes (total_compactions={now})",
                    pass + 1
                );
                return;
            }
        } else {
            eprintln!(
                "holt compact-after-preload pass {}: total_compactions {prev} -> {now}",
                pass + 1
            );
            stable = 0;
            prev = now;
        }
    }
    eprintln!("holt compact-after-preload hit MAX_PASSES={MAX_PASSES} cap (not settled)");
}

#[cfg(feature = "comparators")]
fn preload_rocksdb(db: &DB, workload: Workload, n_keys: usize) {
    // Preload is always async/fast (a bulk load is not the durable-write
    // experiment); the timed put/mixed phase uses the configured profile.
    let wo = rocksdb_write_opts(false);
    let mut batch = WriteBatch::default();
    let mut in_batch = 0usize;
    progress("rocksdb", "preload", 0, n_keys);
    for i in 0..n_keys {
        let key = workload.key(i, n_keys);
        let value = workload.value(i, 0);
        batch.put(key, value);
        in_batch += 1;
        if in_batch == PRELOAD_BATCH {
            db.write_opt(batch, &wo).expect("rocksdb batch preload");
            batch = WriteBatch::default();
            in_batch = 0;
        }
        if should_progress(i + 1, n_keys) {
            progress("rocksdb", "preload", i + 1, n_keys);
        }
    }
    if in_batch != 0 {
        db.write_opt(batch, &wo).expect("rocksdb final preload");
    }
}

#[cfg(feature = "comparators")]
fn preload_sqlite(conn: &Connection, workload: Workload, n_keys: usize) {
    progress("sqlite", "preload", 0, n_keys);
    let tx = conn.unchecked_transaction().expect("sqlite preload tx");
    {
        let mut stmt = tx
            .prepare("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
            .expect("sqlite preload stmt");
        for i in 0..n_keys {
            let key = workload.key(i, n_keys);
            let value = workload.value(i, 0);
            stmt.execute(params![key.as_slice(), value.as_slice()])
                .expect("sqlite preload insert");
            if should_progress(i + 1, n_keys) {
                progress("sqlite", "preload", i + 1, n_keys);
            }
        }
    }
    tx.commit().expect("sqlite preload commit");
}

#[cfg(feature = "comparators")]
fn preload_sled(db: &SledDb, workload: Workload, n_keys: usize) {
    progress("sled", "preload", 0, n_keys);
    for i in 0..n_keys {
        let key = workload.key(i, n_keys);
        let value = workload.value(i, 0);
        db.insert(key, value).expect("sled preload insert");
        if should_progress(i + 1, n_keys) {
            progress("sled", "preload", i + 1, n_keys);
        }
    }
}

#[cfg(feature = "comparators")]
fn preload_lmdb(lmdb: &Lmdb, workload: Workload, n_keys: usize) {
    progress("lmdb", "preload", 0, n_keys);
    let mut wtxn = lmdb.env.write_txn().expect("lmdb preload wtxn");
    let mut in_batch = 0usize;
    for i in 0..n_keys {
        let key = workload.key(i, n_keys);
        let value = workload.value(i, 0);
        lmdb.db
            .put(&mut wtxn, key.as_slice(), value.as_slice())
            .expect("lmdb preload put");
        in_batch += 1;
        if in_batch == PRELOAD_BATCH {
            wtxn.commit().expect("lmdb preload batch commit");
            wtxn = lmdb.env.write_txn().expect("lmdb preload wtxn");
            in_batch = 0;
        }
        if should_progress(i + 1, n_keys) {
            progress("lmdb", "preload", i + 1, n_keys);
        }
    }
    wtxn.commit().expect("lmdb preload final commit");
}

fn time_get_holt(tree: &Tree, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        black_box(tree.get(black_box(&sample.key)).expect("holt get"));
    })
}

/// Time the get phase across `threads` cloned `Tree` handles (sharing
/// one backing store). Returns wall-clock for ALL `samples`, so
/// `report` shows aggregate throughput — the lever that exposes free
/// device queue depth on holt's lock-free cold-read path.
fn time_get_holt_concurrent(tree: &Tree, samples: &[OpSample], threads: usize) -> Duration {
    if threads <= 1 {
        return time_get_holt(tree, samples);
    }
    let chunk = samples.len().div_ceil(threads);
    let start = Instant::now();
    std::thread::scope(|scope| {
        for c in samples.chunks(chunk) {
            let t = tree.clone();
            scope.spawn(move || {
                for sample in c {
                    black_box(t.get(black_box(&sample.key)).expect("holt get"));
                }
            });
        }
    });
    start.elapsed()
}

fn time_put_holt(tree: &Tree, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        tree.put(black_box(&sample.key), black_box(&sample.value))
            .expect("holt put");
    })
}

fn time_mixed_holt(tree: &Tree, samples: &[OpSample]) -> Duration {
    time_samples_indexed(samples, |i, sample| {
        if i & 1 == 0 {
            black_box(tree.get(black_box(&sample.key)).expect("holt get"));
        } else {
            tree.put(black_box(&sample.key), black_box(&sample.value))
                .expect("holt put");
        }
    })
}

#[cfg(feature = "comparators")]
fn time_get_rocksdb(db: &DB, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        black_box(db.get(black_box(&sample.key)).expect("rocksdb get"));
    })
}

#[cfg(feature = "comparators")]
fn time_put_rocksdb(db: &DB, wo: &WriteOptions, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        db.put_opt(black_box(&sample.key), black_box(&sample.value), wo)
            .expect("rocksdb put");
    })
}

#[cfg(feature = "comparators")]
fn time_mixed_rocksdb(db: &DB, wo: &WriteOptions, samples: &[OpSample]) -> Duration {
    time_samples_indexed(samples, |i, sample| {
        if i & 1 == 0 {
            black_box(db.get(black_box(&sample.key)).expect("rocksdb get"));
        } else {
            db.put_opt(black_box(&sample.key), black_box(&sample.value), wo)
                .expect("rocksdb put");
        }
    })
}

#[cfg(feature = "comparators")]
fn time_get_sqlite(conn: &Connection, samples: &[OpSample]) -> Duration {
    let mut stmt = conn
        .prepare("SELECT v FROM kv WHERE k = ?")
        .expect("sqlite get stmt");
    time_samples(samples, |sample| {
        let v: Option<Vec<u8>> = stmt
            .query_row(params![sample.key.as_slice()], |row| row.get(0))
            .optional()
            .expect("sqlite get");
        black_box(v);
    })
}

#[cfg(feature = "comparators")]
fn time_put_sqlite(conn: &Connection, samples: &[OpSample]) -> Duration {
    let mut stmt = conn
        .prepare("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
        .expect("sqlite put stmt");
    time_samples(samples, |sample| {
        stmt.execute(params![sample.key.as_slice(), sample.value.as_slice()])
            .expect("sqlite put");
    })
}

#[cfg(feature = "comparators")]
fn time_mixed_sqlite(conn: &Connection, samples: &[OpSample]) -> Duration {
    let mut get_stmt = conn
        .prepare("SELECT v FROM kv WHERE k = ?")
        .expect("sqlite get stmt");
    let mut put_stmt = conn
        .prepare("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
        .expect("sqlite put stmt");
    time_samples_indexed(samples, |i, sample| {
        if i & 1 == 0 {
            let v: Option<Vec<u8>> = get_stmt
                .query_row(params![sample.key.as_slice()], |row| row.get(0))
                .optional()
                .expect("sqlite get");
            black_box(v);
        } else {
            put_stmt
                .execute(params![sample.key.as_slice(), sample.value.as_slice()])
                .expect("sqlite put");
        }
    })
}

#[cfg(feature = "comparators")]
fn time_get_sled(db: &SledDb, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        black_box(db.get(black_box(&sample.key)).expect("sled get"));
    })
}

#[cfg(feature = "comparators")]
fn time_put_sled(db: &SledDb, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        db.insert(black_box(&sample.key), black_box(sample.value.as_slice()))
            .expect("sled put");
    })
}

#[cfg(feature = "comparators")]
fn time_mixed_sled(db: &SledDb, samples: &[OpSample]) -> Duration {
    time_samples_indexed(samples, |i, sample| {
        if i & 1 == 0 {
            black_box(db.get(black_box(&sample.key)).expect("sled get"));
        } else {
            db.insert(black_box(&sample.key), black_box(sample.value.as_slice()))
                .expect("sled put");
        }
    })
}

// One read transaction reused across the whole get loop — the idiomatic
// LMDB read path (parallels sqlite's reused prepared statement). LMDB read
// txns are snapshot-cheap; a per-op txn would measure txn setup, not reads.
#[cfg(feature = "comparators")]
fn time_get_lmdb(lmdb: &Lmdb, samples: &[OpSample]) -> Duration {
    let rtxn = lmdb.env.read_txn().expect("lmdb get rtxn");
    time_samples(samples, |sample| {
        black_box(
            lmdb.db
                .get(&rtxn, black_box(sample.key.as_slice()))
                .expect("lmdb get"),
        );
    })
}

// Per-op write txn + commit = one durable write per op (commit honors the
// env's sync flag: NO_SYNC for async-durable, msync for sync-durable),
// matching rocksdb `put_opt`/sqlite per-row insert.
#[cfg(feature = "comparators")]
fn time_put_lmdb(lmdb: &Lmdb, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        let mut wtxn = lmdb.env.write_txn().expect("lmdb put wtxn");
        lmdb.db
            .put(&mut wtxn, sample.key.as_slice(), sample.value.as_slice())
            .expect("lmdb put");
        wtxn.commit().expect("lmdb put commit");
    })
}

#[cfg(feature = "comparators")]
fn time_mixed_lmdb(lmdb: &Lmdb, samples: &[OpSample]) -> Duration {
    time_samples_indexed(samples, |i, sample| {
        if i & 1 == 0 {
            let rtxn = lmdb.env.read_txn().expect("lmdb mixed rtxn");
            black_box(
                lmdb.db
                    .get(&rtxn, sample.key.as_slice())
                    .expect("lmdb get"),
            );
        } else {
            let mut wtxn = lmdb.env.write_txn().expect("lmdb mixed wtxn");
            lmdb.db
                .put(&mut wtxn, sample.key.as_slice(), sample.value.as_slice())
                .expect("lmdb put");
            wtxn.commit().expect("lmdb mixed commit");
        }
    })
}

fn time_samples(mut samples: &[OpSample], mut f: impl FnMut(&OpSample)) -> Duration {
    let start = Instant::now();
    while let Some((sample, rest)) = samples.split_first() {
        f(sample);
        samples = rest;
    }
    start.elapsed()
}

fn time_samples_indexed(samples: &[OpSample], mut f: impl FnMut(usize, &OpSample)) -> Duration {
    let start = Instant::now();
    for (i, sample) in samples.iter().enumerate() {
        f(i, sample);
    }
    start.elapsed()
}

fn time_repeated(n: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    start.elapsed()
}

fn holt_list_keys(tree: &Tree, prefix: &[u8], take: usize) -> usize {
    let mut seen = 0usize;
    for entry in tree.scan_keys(prefix) {
        match entry.expect("holt list") {
            KeyRangeEntry::Key { .. } => seen += 1,
            KeyRangeEntry::CommonPrefix(_) => unreachable!("plain list has no delimiter"),
            _ => unreachable!("KeyRangeEntry got a new variant"),
        }
        if seen >= take {
            break;
        }
    }
    seen
}

fn holt_list_records(tree: &Tree, prefix: &[u8], take: usize) -> usize {
    let mut seen = 0usize;
    for entry in tree.range().prefix(prefix) {
        match entry.expect("holt list") {
            RangeEntry::Key { .. } => seen += 1,
            RangeEntry::CommonPrefix(_) => unreachable!("plain list has no delimiter"),
            _ => unreachable!("RangeEntry got a new variant"),
        }
        if seen >= take {
            break;
        }
    }
    seen
}

fn holt_list_dir(tree: &Tree, prefix: &[u8], delim: u8, take: usize) -> usize {
    let mut seen = 0usize;
    tree.scan_keys(prefix)
        .delimiter(delim)
        .visit(take, |_| {
            seen += 1;
            Ok(())
        })
        .expect("holt list_dir");
    seen
}

#[cfg(feature = "comparators")]
fn rocksdb_list_plain(db: &DB, prefix: &[u8], take: usize) -> usize {
    let mut seen = 0usize;
    let mut iter = db.raw_iterator();
    iter.seek(prefix);
    while let Some(key) = iter.key() {
        if !key.starts_with(prefix) {
            break;
        }
        seen += 1;
        if seen >= take {
            break;
        }
        iter.next();
    }
    iter.status().expect("rocksdb list iterator status");
    seen
}

#[cfg(feature = "comparators")]
fn rocksdb_list_dir(db: &DB, prefix: &[u8], delim: u8, take: usize) -> usize {
    let mut seen = 0usize;
    let mut iter = db.raw_iterator();
    iter.seek(prefix);
    while let Some(key) = iter.key() {
        if !key.starts_with(prefix) {
            break;
        }
        let next_seek = delimiter_successor(key, prefix.len(), delim);
        seen += 1;
        if seen >= take {
            break;
        }
        if let Some(next) = next_seek {
            iter.seek(next);
        } else {
            iter.next();
        }
    }
    iter.status().expect("rocksdb list_dir iterator status");
    seen
}

#[cfg(feature = "comparators")]
fn sqlite_list_plain(conn: &Connection, prefix: &[u8], upper: &[u8], take: usize) -> usize {
    let mut stmt = conn
        .prepare_cached("SELECT k FROM kv WHERE k >= ? AND k < ? ORDER BY k LIMIT ?")
        .expect("sqlite list stmt");
    let mut rows = stmt
        .query(params![prefix, upper, take as i64])
        .expect("sqlite list query");
    let mut seen = 0usize;
    while rows.next().expect("sqlite list row").is_some() {
        seen += 1;
        if seen >= take {
            break;
        }
    }
    seen
}

#[cfg(feature = "comparators")]
fn sqlite_list_dir(
    conn: &Connection,
    prefix: &[u8],
    upper: &[u8],
    delim: u8,
    take: usize,
) -> usize {
    let mut stmt = conn
        .prepare_cached("SELECT k FROM kv WHERE k >= ? AND k < ? ORDER BY k LIMIT 1")
        .expect("sqlite list_dir stmt");
    let mut seen = 0usize;
    let mut cursor = prefix.to_vec();
    while cursor.as_slice() < upper {
        let found = stmt
            .query_row(params![cursor.as_slice(), upper], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .optional()
            .expect("sqlite list_dir query");
        let Some(key) = found else {
            break;
        };
        let next_seek = delimiter_successor(&key, prefix.len(), delim);
        seen += 1;
        if seen >= take {
            break;
        }
        if let Some(next) = next_seek {
            cursor = next;
        } else {
            cursor = key_successor(&key);
        }
    }
    seen
}

#[cfg(feature = "comparators")]
fn sled_list_plain(db: &SledDb, prefix: &[u8], take: usize) -> usize {
    let mut seen = 0usize;
    for item in db.scan_prefix(prefix) {
        let (_k, _v) = item.expect("sled list");
        seen += 1;
        if seen >= take {
            break;
        }
    }
    seen
}

#[cfg(feature = "comparators")]
fn sled_list_dir(db: &SledDb, prefix: &[u8], delim: u8, take: usize) -> usize {
    let mut seen = 0usize;
    let mut cursor = prefix.to_vec();
    let upper = prefix_upper(prefix);
    while cursor.as_slice() < upper.as_slice() {
        let found = db.range(cursor.as_slice()..upper.as_slice()).next();
        let Some(item) = found else {
            break;
        };
        let (key, _value) = item.expect("sled list_dir");
        let next_seek = delimiter_successor(&key, prefix.len(), delim);
        seen += 1;
        if seen >= take {
            break;
        }
        if let Some(next) = next_seek {
            cursor = next;
        } else {
            cursor = key_successor(&key);
        }
    }
    seen
}

#[cfg(feature = "comparators")]
fn lmdb_list_plain(lmdb: &Lmdb, prefix: &[u8], take: usize) -> usize {
    let rtxn = lmdb.env.read_txn().expect("lmdb list rtxn");
    let mut seen = 0usize;
    let iter = lmdb
        .db
        .prefix_iter(&rtxn, prefix)
        .expect("lmdb list prefix_iter");
    for item in iter {
        let _ = item.expect("lmdb list item");
        seen += 1;
        if seen >= take {
            break;
        }
    }
    seen
}

#[cfg(feature = "comparators")]
fn lmdb_list_dir(lmdb: &Lmdb, prefix: &[u8], delim: u8, take: usize) -> usize {
    let rtxn = lmdb.env.read_txn().expect("lmdb list_dir rtxn");
    let mut seen = 0usize;
    let mut cursor = prefix.to_vec();
    let upper = prefix_upper(prefix);
    while cursor.as_slice() < upper.as_slice() {
        let range = (
            Bound::Included(cursor.as_slice()),
            Bound::Excluded(upper.as_slice()),
        );
        let found = lmdb
            .db
            .range(&rtxn, &range)
            .expect("lmdb list_dir range")
            .next();
        let Some(item) = found else {
            break;
        };
        let (key, _value) = item.expect("lmdb list_dir item");
        let next_seek = delimiter_successor(key, prefix.len(), delim);
        seen += 1;
        if seen >= take {
            break;
        }
        cursor = next_seek.unwrap_or_else(|| key_successor(key));
    }
    seen
}

#[cfg(feature = "comparators")]
fn delimiter_successor(key: &[u8], prefix_len: usize, delim: u8) -> Option<Vec<u8>> {
    let rest = &key[prefix_len..];
    let idx = rest.iter().position(|b| *b == delim)?;
    Some(key_successor(&key[..=prefix_len + idx]))
}

#[cfg(feature = "comparators")]
fn prefix_upper(prefix: &[u8]) -> Vec<u8> {
    key_successor(prefix)
}

#[cfg(feature = "comparators")]
fn key_successor(key: &[u8]) -> Vec<u8> {
    let mut upper = key.to_vec();
    let last = upper.last_mut().expect("key must be non-empty");
    *last = last.checked_add(1).expect("key last byte must be < 0xff");
    upper
}

#[cfg(feature = "comparators")]
fn make_rocksdb(dir: &StoreDir, cfg: &StressConfig) -> DB {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(256 * 1024 * 1024);
    opts.set_max_write_buffer_number(4);
    opts.set_compression_type(rocksdb::DBCompressionType::None);
    if cfg.rocksdb_direct {
        // Fair cold-read comparison: bypass the OS page cache (as holt
        // does with O_DIRECT/F_NOCACHE) and bound the block cache to
        // holt's buffer-pool size, so both engines read cold from the
        // device at matched memory instead of rocksdb serving the whole
        // dataset from the OS page cache.
        opts.set_use_direct_reads(true);
        opts.set_use_direct_io_for_flush_and_compaction(true);
        const BLOB_FRAME_BYTES: usize = 512 * 1024; // holt's PAGE_SIZE
        let mut bbt = BlockBasedOptions::default();
        bbt.set_block_cache(&Cache::new_lru_cache(
            cfg.buffer_pool_size * BLOB_FRAME_BYTES,
        ));
        opts.set_block_based_table_factory(&bbt);
    }
    DB::open(&opts, dir.path()).expect("rocksdb open")
}

#[cfg(feature = "comparators")]
fn rocksdb_write_opts(sync: bool) -> WriteOptions {
    let mut wo = WriteOptions::default();
    wo.disable_wal(false);
    // sync-durable profile: fsync the WAL on every write (matches holt
    // Wal{sync:true}, sqlite synchronous=FULL, lmdb default msync).
    wo.set_sync(sync);
    wo
}

#[cfg(feature = "comparators")]
fn make_sqlite_persistent(dir: &StoreDir, cfg: &StressConfig) -> Connection {
    let conn = Connection::open(dir.path().join("sqlite.db")).expect("sqlite open");
    // sync-durable: synchronous=FULL fsyncs on each autocommit write (matches
    // holt Wal{sync:true} / rocksdb sync=true / lmdb default). async-durable:
    // synchronous=OFF leaves durability to the OS, like the others' async mode.
    let synchronous = if cfg.wal_sync { "FULL" } else { "OFF" };
    // Matched memory: sqlite's page cache (in its own heap, NOT the OS page
    // cache, so drop_caches cannot clear it) is capped to the same bytes as
    // holt's buffer pool — buffer_pool_size frames × 512 KiB, expressed as
    // negative KiB. Without this the cold regime would serve sqlite's working
    // set from its internal cache.
    let cache_kib = cfg.buffer_pool_size.max(1) * 512;
    conn.execute_batch(&format!(
        "PRAGMA journal_mode = WAL;\n\
         PRAGMA synchronous = {synchronous};\n\
         PRAGMA temp_store = MEMORY;\n\
         PRAGMA cache_size = -{cache_kib};\n\
         CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID;",
    ))
    .expect("sqlite pragmas + schema");
    conn
}

#[cfg(feature = "comparators")]
fn make_sled_persistent(dir: &StoreDir) -> SledDb {
    sled::Config::new()
        .path(dir.path())
        .mode(SledMode::HighThroughput)
        .cache_capacity(256 * 1024 * 1024)
        .flush_every_ms(None)
        .open()
        .expect("sled open")
}

/// LMDB environment + its main (unnamed) database, held together because
/// every op needs the env to open a txn and the handle to address rows.
#[cfg(feature = "comparators")]
struct Lmdb {
    env: LmdbEnv,
    db: LmdbDatabase<Bytes, Bytes>,
}

/// LMDB requires its memory map (the cap on total DB file size) to be sized
/// up front. The file is sparse, so over-provisioning only reserves virtual
/// address space — cheap on 64-bit. Budget 384 B/entry (≈50 B path key +
/// ≈60 B JSON value + B+tree page overhead), floor 1 GiB, page-aligned.
#[cfg(feature = "comparators")]
fn lmdb_map_size(n_keys: usize) -> usize {
    const PAGE: usize = 4096;
    let bytes = (n_keys.max(1) * 384).max(1 << 30);
    bytes.div_ceil(PAGE) * PAGE
}

#[cfg(feature = "comparators")]
fn make_lmdb(dir: &StoreDir, cfg: &StressConfig) -> Lmdb {
    let mut opts = EnvOpenOptions::new();
    opts.map_size(lmdb_map_size(cfg.n_keys));
    opts.max_dbs(1);
    if !cfg.wal_sync {
        // async-durable profile: don't fsync on commit (matches holt
        // Wal{sync:false}, rocksdb sync=false, sqlite synchronous=OFF). A
        // crash can lose the last commits but not corrupt the DB — hence
        // unsafe. The sync-durable profile (wal_sync=true) leaves the
        // default msync-on-commit behavior in place.
        unsafe {
            opts.flags(EnvFlags::NO_SYNC | EnvFlags::NO_META_SYNC);
        }
    }
    let env = unsafe { opts.open(dir.path()) }.expect("lmdb open");
    let mut wtxn = env.write_txn().expect("lmdb create wtxn");
    let db: LmdbDatabase<Bytes, Bytes> =
        env.create_database(&mut wtxn, None).expect("lmdb create db");
    wtxn.commit().expect("lmdb create commit");
    Lmdb { env, db }
}

fn print_holt_shape(label: &str, tree: &Tree) {
    let s = tree.stats().expect("holt stats");
    let (journal_appends, journal_batches, journal_syncs) = s
        .journal
        .as_ref()
        .map_or((0, 0, 0), |j| (j.appends, j.batches, j.syncs));
    println!(
        "holt_shape {label} blobs={} edges={} leaves={} max_depth={} avg_depth={:.2} avg_fill={:.3} max_fill={:.3} underfilled={} overfull={} bm_hits={} bm_misses={} bm_reads={} bm_read_bytes={} bm_point_reads={} bm_scan_reads={} bm_silent_reads={} avg_hops={:.2} max_hops={} spillovers={} merges={} route_resident={} route_demotions={} route_entries={} route_hits={} route_misses={} route_learns={} route_invalidations={} journal_appends={} journal_batches={} journal_syncs={}",
        s.blob_count,
        s.total_blob_edges,
        s.leaf_blob_count,
        s.max_blob_depth,
        s.avg_blob_depth(),
        s.avg_blob_fill_ratio(),
        s.max_blob_fill_ratio(),
        s.underfilled_child_blobs,
        s.overfull_child_blobs,
        s.bm_cache_hits,
        s.bm_cache_misses,
        s.bm_full_blob_reads,
        s.bm_full_blob_read_bytes,
        s.bm_point_full_blob_reads,
        s.bm_scan_full_blob_reads,
        s.bm_silent_full_blob_reads,
        s.bm_avg_blob_hops(),
        s.bm_max_blob_hops,
        s.bm_spillovers,
        s.bm_merges,
        s.bm_route_resident_count,
        s.bm_route_resident_demotions,
        s.route_cache.entries,
        s.route_cache.hits,
        s.route_cache.misses,
        s.route_cache.learns,
        s.route_cache.invalidations,
        journal_appends,
        journal_batches,
        journal_syncs,
    );
}

fn report(engine: &str, op: &str, ops: usize, elapsed: Duration) {
    let ns = elapsed.as_secs_f64() * 1_000_000_000.0 / ops.max(1) as f64;
    let mops = ops as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    println!("{engine:<8} {op:<8} {ns:>10.1} ns/op {mops:>8.3} Mops/s");
}

/// E7 space amplification: total on-disk bytes for the (settled) store.
/// Call after preload + checkpoint/reopen so each engine's data is merged
/// to its steady on-disk form (sqlite WAL checkpointed, rocksdb flushed,
/// holt WAL drained to blobs). holt's 512 KiB frames at ~0.38 fill make this
/// its honest weak axis.
fn dir_size_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => total += dir_size_bytes(&entry.path()),
                // Actual allocated blocks (du-equivalent), not apparent length:
                // LMDB's data file is a sparse map_size reservation, so len()
                // would report the whole reservation rather than used pages.
                Ok(_) => total += entry.metadata().map(|m| m.blocks() * 512).unwrap_or(0),
                Err(_) => {}
            }
        }
    }
    total
}

fn print_space(engine: &str, dir: &std::path::Path, n_keys: usize) {
    let bytes = dir_size_bytes(dir);
    let per_key = bytes as f64 / n_keys.max(1) as f64;
    println!("space engine={engine} bytes={bytes} n_keys={n_keys} bytes_per_key={per_key:.1}");
}

/// Device read amplification: bytes actually fetched from storage by this
/// process, from `/proc/self/io` (`read_bytes`). Sampled before/after a timed
/// phase, the delta is the cross-engine cold read-amp — counts O_DIRECT
/// (holt, rocksdb-direct), mmap page faults (lmdb), and buffered reads
/// (sqlite) uniformly. Returns 0 off Linux (e.g. the mac dev box).
fn proc_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("read_bytes:")
                    .and_then(|v| v.trim().parse::<u64>().ok())
            })
        })
        .unwrap_or(0)
}

fn print_read_amp(engine: &str, op: &str, ops: usize, bytes: u64) {
    let per_op = bytes as f64 / ops.max(1) as f64;
    println!("read_amp engine={engine} op={op} read_bytes={bytes} bytes_per_op={per_op:.1}");
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn should_progress(done: usize, total: usize) -> bool {
    total >= 1_000_000 && (done == total || done % 1_000_000 == 0)
}

fn progress(engine: &str, stage: &str, done: usize, total: usize) {
    if env::var("HOLT_STRESS_PROGRESS")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(total >= 1_000_000)
    {
        eprintln!("{engine} {stage}: {done}/{total}");
    }
}

/// Cold regime: flush dirty pages and drop the OS page cache (+ dentries
/// and inodes) so the next timed phase reads from the device. Holt's reads
/// are O_DIRECT (page-cache-independent), but the page-cache comparators
/// (lmdb, sqlite, buffered rocksdb) would otherwise serve the whole dataset
/// from RAM — an unfair cold comparison. No-op unless `HOLT_STRESS_DROP_CACHES`
/// is set; the password is never handled here.
fn maybe_drop_caches(cfg: &StressConfig, engine: &str) {
    if !cfg.drop_caches {
        return;
    }
    // Flush dirty pages, then drop the page cache via `sudo -n` (relies on a
    // sudo timestamp warmed in the same pty before the run; the password is
    // never handled here). Run the harness inside a pty (tmux / ssh -t) so a
    // child process can reuse the interactively-authenticated ticket.
    let _ = std::process::Command::new("sync").status();
    let status = std::process::Command::new("sudo")
        .args(["-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        .status();
    match status {
        Ok(s) if s.success() => eprintln!("{engine} drop_caches: ok"),
        Ok(s) => panic!(
            "drop_caches failed (exit {s}); warm the sudo timestamp (`sudo -v`) in the \
             same pty before running — see benches/eval_cold.sh"
        ),
        Err(e) => panic!("drop_caches: failed to spawn sudo: {e}"),
    }
}
