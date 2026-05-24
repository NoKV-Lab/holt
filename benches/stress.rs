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
use std::time::{Duration, Instant};

use holt::{KeyRangeEntry, RangeEntry, Tree, TreeConfig};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rocksdb::{Options, WriteBatch, WriteOptions, DB};
use rusqlite::{params, Connection, OptionalExtension};
use sled::{Db as SledDb, Mode as SledMode};
use tempfile::TempDir;

const DEFAULT_N_KEYS: usize = 20_000_000;
const DEFAULT_POINT_OPS: usize = 1_000_000;
const DEFAULT_LIST_OPS: usize = 1_000_000;
const DEFAULT_LIST_TAKE: usize = 100;
const DEFAULT_DIR_TAKE: usize = 8;
const DEFAULT_BUFFER_POOL: usize = 64;
const PRELOAD_BATCH: usize = 10_000;
const SEED: u64 = 0xD15E_A5ED_2000_0001;

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
            .unwrap_or_else(|_| "holt,rocksdb,sqlite".to_string())
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
        "profile=single_thread,warm_service,persistent_wal,wal_sync={},checkpoint=enabled,reopen_after_preload={}",
        cfg.wal_sync, cfg.reopen_after_preload
    );

    let samples = make_samples(cfg.workload, cfg.n_keys, cfg.point_ops);
    if cfg.selected("holt") {
        run_holt(&cfg, &samples);
    }
    if cfg.selected("rocksdb") {
        run_rocksdb(&cfg, &samples);
    }
    if cfg.selected("sqlite") {
        run_sqlite(&cfg, &samples);
    }
    if cfg.selected("sled") {
        run_sled(&cfg, &samples);
    }
}

fn run_holt(cfg: &StressConfig, samples: &[OpSample]) {
    let dir = TempDir::new().expect("holt tempdir");
    let mut tree_cfg = TreeConfig::new(dir.path());
    tree_cfg.wal_sync = cfg.wal_sync;
    tree_cfg.buffer_pool_size = cfg.buffer_pool_size;
    let mut tree = Tree::open(tree_cfg.clone()).expect("holt open");
    preload_holt(&tree, cfg.workload, cfg.n_keys);
    print_holt_shape("preload", &tree);
    tree.checkpoint().expect("holt preload checkpoint");
    print_holt_shape("ready", &tree);
    if cfg.reopen_after_preload {
        drop(tree);
        tree = Tree::open(tree_cfg).expect("holt reopen");
        print_holt_shape("reopened", &tree);
    }

    if cfg.selected_op("get") {
        report("holt", "get", samples.len(), time_get_holt(&tree, samples));
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

fn run_rocksdb(cfg: &StressConfig, samples: &[OpSample]) {
    let rocksdb_dir = TempDir::new().expect("rocksdb tempdir");
    let mut db = make_rocksdb(&rocksdb_dir);
    preload_rocksdb(&db, cfg.workload, cfg.n_keys);
    if cfg.reopen_after_preload {
        db.flush().expect("rocksdb preload flush before reopen");
        drop(db);
        db = make_rocksdb(&rocksdb_dir);
    }
    let wo = rocksdb_write_opts();

    if cfg.selected_op("get") {
        report(
            "rocksdb",
            "get",
            samples.len(),
            time_get_rocksdb(&db, samples),
        );
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

fn run_sqlite(cfg: &StressConfig, samples: &[OpSample]) {
    let sqlite_dir = TempDir::new().expect("sqlite tempdir");
    let mut conn = make_sqlite_persistent(&sqlite_dir);
    preload_sqlite(&conn, cfg.workload, cfg.n_keys);
    if cfg.reopen_after_preload {
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .expect("sqlite preload checkpoint before reopen");
        drop(conn);
        conn = make_sqlite_persistent(&sqlite_dir);
    }

    if cfg.selected_op("get") {
        report(
            "sqlite",
            "get",
            samples.len(),
            time_get_sqlite(&conn, samples),
        );
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

fn run_sled(cfg: &StressConfig, samples: &[OpSample]) {
    let sled_dir = TempDir::new().expect("sled tempdir");
    let mut db = make_sled_persistent(&sled_dir);
    preload_sled(&db, cfg.workload, cfg.n_keys);
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

fn make_samples(workload: Workload, n_keys: usize, ops: usize) -> Vec<OpSample> {
    assert!(n_keys > 0, "HOLT_STRESS_N must be > 0");
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut out = Vec::with_capacity(ops);
    for op in 0..ops {
        let idx = (rng.next_u64() as usize) % n_keys;
        out.push(OpSample {
            key: workload.key(idx, n_keys),
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

fn preload_rocksdb(db: &DB, workload: Workload, n_keys: usize) {
    let wo = rocksdb_write_opts();
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

fn time_get_holt(tree: &Tree, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        black_box(tree.get(black_box(&sample.key)).expect("holt get"));
    })
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

fn time_get_rocksdb(db: &DB, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        black_box(db.get(black_box(&sample.key)).expect("rocksdb get"));
    })
}

fn time_put_rocksdb(db: &DB, wo: &WriteOptions, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        db.put_opt(black_box(&sample.key), black_box(&sample.value), wo)
            .expect("rocksdb put");
    })
}

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

fn time_get_sqlite(conn: &Connection, samples: &[OpSample]) -> Duration {
    let mut stmt = conn
        .prepare("SELECT v FROM kv WHERE k = ?")
        .expect("sqlite get stmt");
    time_samples(samples, |sample| {
        let v: Vec<u8> = stmt
            .query_row(params![sample.key.as_slice()], |row| row.get(0))
            .expect("sqlite get");
        black_box(v);
    })
}

fn time_put_sqlite(conn: &Connection, samples: &[OpSample]) -> Duration {
    let mut stmt = conn
        .prepare("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
        .expect("sqlite put stmt");
    time_samples(samples, |sample| {
        stmt.execute(params![sample.key.as_slice(), sample.value.as_slice()])
            .expect("sqlite put");
    })
}

fn time_mixed_sqlite(conn: &Connection, samples: &[OpSample]) -> Duration {
    let mut get_stmt = conn
        .prepare("SELECT v FROM kv WHERE k = ?")
        .expect("sqlite get stmt");
    let mut put_stmt = conn
        .prepare("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
        .expect("sqlite put stmt");
    time_samples_indexed(samples, |i, sample| {
        if i & 1 == 0 {
            let v: Vec<u8> = get_stmt
                .query_row(params![sample.key.as_slice()], |row| row.get(0))
                .expect("sqlite get");
            black_box(v);
        } else {
            put_stmt
                .execute(params![sample.key.as_slice(), sample.value.as_slice()])
                .expect("sqlite put");
        }
    })
}

fn time_get_sled(db: &SledDb, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        black_box(db.get(black_box(&sample.key)).expect("sled get"));
    })
}

fn time_put_sled(db: &SledDb, samples: &[OpSample]) -> Duration {
    time_samples(samples, |sample| {
        db.insert(black_box(&sample.key), black_box(sample.value.as_slice()))
            .expect("sled put");
    })
}

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

fn delimiter_successor(key: &[u8], prefix_len: usize, delim: u8) -> Option<Vec<u8>> {
    let rest = &key[prefix_len..];
    let idx = rest.iter().position(|b| *b == delim)?;
    Some(key_successor(&key[..=prefix_len + idx]))
}

fn prefix_upper(prefix: &[u8]) -> Vec<u8> {
    key_successor(prefix)
}

fn key_successor(key: &[u8]) -> Vec<u8> {
    let mut upper = key.to_vec();
    let last = upper.last_mut().expect("key must be non-empty");
    *last = last.checked_add(1).expect("key last byte must be < 0xff");
    upper
}

fn make_rocksdb(dir: &TempDir) -> DB {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(256 * 1024 * 1024);
    opts.set_max_write_buffer_number(4);
    opts.set_compression_type(rocksdb::DBCompressionType::None);
    DB::open(&opts, dir.path()).expect("rocksdb open")
}

fn rocksdb_write_opts() -> WriteOptions {
    let mut wo = WriteOptions::default();
    wo.disable_wal(false);
    wo.set_sync(false);
    wo
}

fn make_sqlite_persistent(dir: &TempDir) -> Connection {
    let conn = Connection::open(dir.path().join("sqlite.db")).expect("sqlite open");
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;\n\
         PRAGMA synchronous = OFF;\n\
         PRAGMA temp_store = MEMORY;\n\
         PRAGMA cache_size = -262144;\n\
         CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID;",
    )
    .expect("sqlite pragmas + schema");
    conn
}

fn make_sled_persistent(dir: &TempDir) -> SledDb {
    sled::Config::new()
        .path(dir.path())
        .mode(SledMode::HighThroughput)
        .cache_capacity(256 * 1024 * 1024)
        .flush_every_ms(None)
        .open()
        .expect("sled open")
}

fn print_holt_shape(label: &str, tree: &Tree) {
    let s = tree.stats().expect("holt stats");
    let (journal_appends, journal_batches, journal_syncs) = s
        .journal
        .as_ref()
        .map_or((0, 0, 0), |j| (j.appends, j.batches, j.syncs));
    println!(
        "holt_shape {label} blobs={} edges={} leaves={} max_depth={} avg_depth={:.2} avg_fill={:.3} max_fill={:.3} underfilled={} overfull={} bm_hits={} bm_misses={} avg_hops={:.2} max_hops={} spillovers={} merges={} route_resident={} route_demotions={} route_entries={} route_hits={} route_misses={} route_learns={} route_invalidations={} journal_appends={} journal_batches={} journal_syncs={}",
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
