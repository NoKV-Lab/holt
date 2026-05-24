//! Concurrent throughput harness for large metadata datasets.
//!
//! This is intentionally separate from `stress.rs`: `stress` is a
//! single-thread latency harness, while this file measures how Holt
//! and RocksDB behave when multiple worker threads share one
//! file-backed engine.
//!
//! ```sh
//! HOLT_CONCURRENT_N=20000000 \
//! HOLT_CONCURRENT_THREADS=1,2,4,8 \
//! HOLT_CONCURRENT_OPS_PER_THREAD=250000 \
//! cargo bench --manifest-path benches/Cargo.toml --bench concurrent -- objstore
//! ```
//!
//! Profile: warm service, file-backed WAL enabled, no per-op fsync.
//! SQLite is deliberately not part of the main table: its WAL mode
//! still has a single writer, so it is not the same concurrency model
//! as Holt or RocksDB.

use std::env;
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use holt::{Tree, TreeConfig};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rocksdb::{Options, WriteBatch, WriteOptions, DB};
use tempfile::TempDir;

const DEFAULT_N_KEYS: usize = 1_000_000;
const DEFAULT_OPS_PER_THREAD: usize = 100_000;
const DEFAULT_THREADS: &str = "1,2,4,8";
const DEFAULT_ENGINES: &str = "holt,rocksdb";
const DEFAULT_OPS: &str = "get,put,mixed90,mixed50,list_dir";
const DEFAULT_BUFFER_POOL: usize = 64;
const DEFAULT_DIR_TAKE: usize = 8;
const PRELOAD_BATCH: usize = 10_000;
const LATENCY_SAMPLE_STRIDE: usize = 997;
const SEED: u64 = 0xC0A5_E77E_5000_0001;

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

#[derive(Debug, Clone, Copy)]
enum BenchOp {
    Get,
    Put,
    Mixed90,
    Mixed50,
    ListDir,
}

impl BenchOp {
    fn parse(s: &str) -> Self {
        match s {
            "get" => Self::Get,
            "put" => Self::Put,
            "mixed90" | "mixed_90_10" => Self::Mixed90,
            "mixed50" | "mixed_50_50" => Self::Mixed50,
            "list_dir" | "listdir" => Self::ListDir,
            other => panic!("unknown op `{other}`"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Get => "get",
            Self::Put => "put",
            Self::Mixed90 => "mixed90",
            Self::Mixed50 => "mixed50",
            Self::ListDir => "list_dir",
        }
    }
}

#[derive(Debug)]
struct Config {
    workload: Workload,
    n_keys: usize,
    ops_per_thread: usize,
    thread_counts: Vec<usize>,
    engines: Vec<String>,
    ops: Vec<BenchOp>,
    buffer_pool_size: usize,
    dir_take: usize,
}

impl Config {
    fn from_env() -> Self {
        Self {
            workload: workload_arg()
                .as_deref()
                .map(Workload::parse)
                .unwrap_or(Workload::Objstore),
            n_keys: env_usize("HOLT_CONCURRENT_N", DEFAULT_N_KEYS),
            ops_per_thread: env_usize("HOLT_CONCURRENT_OPS_PER_THREAD", DEFAULT_OPS_PER_THREAD),
            thread_counts: env_list("HOLT_CONCURRENT_THREADS", DEFAULT_THREADS, |s| {
                s.parse::<usize>().expect("thread count must be usize")
            }),
            engines: env_list("HOLT_CONCURRENT_ENGINES", DEFAULT_ENGINES, str::to_string),
            ops: env_list("HOLT_CONCURRENT_OPS", DEFAULT_OPS, BenchOp::parse),
            buffer_pool_size: env_usize("HOLT_CONCURRENT_BUFFER_POOL", DEFAULT_BUFFER_POOL),
            dir_take: env_usize("HOLT_CONCURRENT_DIR_TAKE", DEFAULT_DIR_TAKE),
        }
    }

    fn selected(&self, engine: &str) -> bool {
        self.engines
            .iter()
            .any(|s| s == "all" || s.eq_ignore_ascii_case(engine))
    }

    fn max_threads(&self) -> usize {
        *self
            .thread_counts
            .iter()
            .max()
            .expect("at least one thread count")
    }
}

fn workload_arg() -> Option<String> {
    env::args().skip(1).find(|arg| !arg.starts_with("--"))
}

#[derive(Clone, Debug)]
struct OpSample {
    key: Vec<u8>,
    value: Vec<u8>,
}

#[derive(Debug)]
struct RunResult {
    elapsed: Duration,
    latency_ns: Vec<u64>,
}

fn main() {
    let cfg = Config::from_env();
    assert!(cfg.n_keys > 0, "HOLT_CONCURRENT_N must be > 0");
    assert!(
        cfg.thread_counts.iter().all(|n| *n > 0),
        "thread counts must be > 0"
    );

    println!(
        "concurrent workload={} n_keys={} ops_per_thread={} threads={} dir_take={} buffer_pool={} engines={} ops={}",
        cfg.workload.name(),
        cfg.n_keys,
        cfg.ops_per_thread,
        join_usize(&cfg.thread_counts),
        cfg.dir_take,
        cfg.buffer_pool_size,
        cfg.engines.join(","),
        cfg.ops.iter().map(|op| op.name()).collect::<Vec<_>>().join(","),
    );
    println!(
        "profile=multi_thread,warm_service,persistent_wal,wal_sync=false,checkpoint=enabled,latency_sample_stride={LATENCY_SAMPLE_STRIDE}"
    );

    let samples = make_thread_samples(&cfg);
    if cfg.selected("holt") {
        run_holt(&cfg, &samples);
    }
    if cfg.selected("rocksdb") {
        run_rocksdb(&cfg, &samples);
    }
}

fn run_holt(cfg: &Config, samples: &[Arc<[OpSample]>]) {
    let dir = TempDir::new().expect("holt tempdir");
    let mut tree_cfg = TreeConfig::new(dir.path());
    tree_cfg.wal_sync = false;
    tree_cfg.buffer_pool_size = cfg.buffer_pool_size;
    let tree = Arc::new(Tree::open(tree_cfg).expect("holt open"));
    preload_holt(&tree, cfg.workload, cfg.n_keys);
    print_holt_shape("preload", &tree);

    for op in &cfg.ops {
        for threads in &cfg.thread_counts {
            let result = run_holt_threads(tree.clone(), cfg, samples, *op, *threads);
            report("holt", op.name(), *threads, cfg.ops_per_thread, &result);
        }
    }
    print_holt_shape("final", &tree);
}

fn run_rocksdb(cfg: &Config, samples: &[Arc<[OpSample]>]) {
    let dir = TempDir::new().expect("rocksdb tempdir");
    let db = Arc::new(make_rocksdb(&dir));
    preload_rocksdb(&db, cfg.workload, cfg.n_keys);

    for op in &cfg.ops {
        for threads in &cfg.thread_counts {
            let result = run_rocksdb_threads(db.clone(), cfg, samples, *op, *threads);
            report("rocksdb", op.name(), *threads, cfg.ops_per_thread, &result);
        }
    }
}

fn run_holt_threads(
    tree: Arc<Tree>,
    cfg: &Config,
    samples: &[Arc<[OpSample]>],
    op: BenchOp,
    threads: usize,
) -> RunResult {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);
    for tid in 0..threads {
        let tree = tree.clone();
        let barrier = barrier.clone();
        let samples = Arc::clone(&samples[tid]);
        let workload = cfg.workload;
        let dir_take = cfg.dir_take;
        handles.push(thread::spawn(move || {
            barrier.wait();
            time_thread(samples, |i, sample| match op {
                BenchOp::Get => {
                    black_box(tree.get(black_box(&sample.key)).expect("holt get"));
                }
                BenchOp::Put => {
                    tree.put(black_box(&sample.key), black_box(&sample.value))
                        .expect("holt put");
                }
                BenchOp::Mixed90 => {
                    if i % 10 == 0 {
                        tree.put(black_box(&sample.key), black_box(&sample.value))
                            .expect("holt put");
                    } else {
                        black_box(tree.get(black_box(&sample.key)).expect("holt get"));
                    }
                }
                BenchOp::Mixed50 => {
                    if i & 1 == 0 {
                        black_box(tree.get(black_box(&sample.key)).expect("holt get"));
                    } else {
                        tree.put(black_box(&sample.key), black_box(&sample.value))
                            .expect("holt put");
                    }
                }
                BenchOp::ListDir => {
                    black_box(holt_list_dir(
                        &tree,
                        workload.dir_prefix(),
                        workload.delimiter(),
                        dir_take,
                    ));
                }
            })
        }));
    }
    let start = Instant::now();
    barrier.wait();
    let mut latency_ns = Vec::new();
    for handle in handles {
        latency_ns.extend(handle.join().expect("holt worker panicked"));
    }
    RunResult {
        elapsed: start.elapsed(),
        latency_ns,
    }
}

fn run_rocksdb_threads(
    db: Arc<DB>,
    cfg: &Config,
    samples: &[Arc<[OpSample]>],
    op: BenchOp,
    threads: usize,
) -> RunResult {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let mut handles = Vec::with_capacity(threads);
    for tid in 0..threads {
        let db = db.clone();
        let barrier = barrier.clone();
        let samples = Arc::clone(&samples[tid]);
        let workload = cfg.workload;
        let dir_take = cfg.dir_take;
        handles.push(thread::spawn(move || {
            let wo = rocksdb_write_opts();
            barrier.wait();
            time_thread(samples, |i, sample| match op {
                BenchOp::Get => {
                    black_box(db.get(black_box(&sample.key)).expect("rocksdb get"));
                }
                BenchOp::Put => {
                    db.put_opt(black_box(&sample.key), black_box(&sample.value), &wo)
                        .expect("rocksdb put");
                }
                BenchOp::Mixed90 => {
                    if i % 10 == 0 {
                        db.put_opt(black_box(&sample.key), black_box(&sample.value), &wo)
                            .expect("rocksdb put");
                    } else {
                        black_box(db.get(black_box(&sample.key)).expect("rocksdb get"));
                    }
                }
                BenchOp::Mixed50 => {
                    if i & 1 == 0 {
                        black_box(db.get(black_box(&sample.key)).expect("rocksdb get"));
                    } else {
                        db.put_opt(black_box(&sample.key), black_box(&sample.value), &wo)
                            .expect("rocksdb put");
                    }
                }
                BenchOp::ListDir => {
                    black_box(rocksdb_list_dir(
                        &db,
                        workload.dir_prefix(),
                        workload.delimiter(),
                        dir_take,
                    ));
                }
            })
        }));
    }
    let start = Instant::now();
    barrier.wait();
    let mut latency_ns = Vec::new();
    for handle in handles {
        latency_ns.extend(handle.join().expect("rocksdb worker panicked"));
    }
    RunResult {
        elapsed: start.elapsed(),
        latency_ns,
    }
}

fn time_thread(samples: Arc<[OpSample]>, mut f: impl FnMut(usize, &OpSample)) -> Vec<u64> {
    let mut latency_ns = Vec::with_capacity(samples.len() / LATENCY_SAMPLE_STRIDE + 1);
    for (i, sample) in samples.iter().enumerate() {
        if i % LATENCY_SAMPLE_STRIDE == 0 {
            let start = Instant::now();
            f(i, sample);
            latency_ns.push(start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64);
        } else {
            f(i, sample);
        }
    }
    latency_ns
}

fn make_thread_samples(cfg: &Config) -> Vec<Arc<[OpSample]>> {
    let mut out = Vec::with_capacity(cfg.max_threads());
    for tid in 0..cfg.max_threads() {
        let mut rng = StdRng::seed_from_u64(SEED ^ ((tid as u64 + 1) << 32));
        let mut samples = Vec::with_capacity(cfg.ops_per_thread);
        for op in 0..cfg.ops_per_thread {
            let idx = (rng.next_u64() as usize) % cfg.n_keys;
            samples.push(OpSample {
                key: cfg.workload.key(idx, cfg.n_keys),
                value: cfg
                    .workload
                    .value(idx, ((tid * cfg.ops_per_thread + op) as u64) + 1),
            });
        }
        out.push(Arc::from(samples));
    }
    out
}

fn preload_holt(tree: &Tree, workload: Workload, n_keys: usize) {
    progress("holt", "preload", 0, n_keys);
    for i in 0..n_keys {
        tree.put(&workload.key(i, n_keys), &workload.value(i, 0))
            .expect("holt preload put");
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
        batch.put(workload.key(i, n_keys), workload.value(i, 0));
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

fn delimiter_successor(key: &[u8], prefix_len: usize, delim: u8) -> Option<Vec<u8>> {
    let rest = &key[prefix_len..];
    let idx = rest.iter().position(|b| *b == delim)?;
    Some(key_successor(&key[..=prefix_len + idx]))
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

fn print_holt_shape(label: &str, tree: &Tree) {
    let s = tree.stats().expect("holt stats");
    let (journal_appends, journal_batches, journal_syncs) = s
        .journal
        .as_ref()
        .map_or((0, 0, 0), |j| (j.appends, j.batches, j.syncs));
    println!(
        "holt_shape {label} blobs={} edges={} leaves={} max_depth={} avg_depth={:.2} avg_fill={:.3} max_fill={:.3} underfilled={} overfull={} avg_hops={:.2} max_hops={} spillovers={} merges={} route_resident={} route_demotions={} route_entries={} route_hits={} route_misses={} route_learns={} route_invalidations={} journal_appends={} journal_batches={} journal_syncs={}",
        s.blob_count,
        s.total_blob_edges,
        s.leaf_blob_count,
        s.max_blob_depth,
        s.avg_blob_depth(),
        s.avg_blob_fill_ratio(),
        s.max_blob_fill_ratio(),
        s.underfilled_child_blobs,
        s.overfull_child_blobs,
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

fn report(engine: &str, op: &str, threads: usize, ops_per_thread: usize, result: &RunResult) {
    let total_ops = threads * ops_per_thread;
    let ns = result.elapsed.as_secs_f64() * 1_000_000_000.0 / total_ops.max(1) as f64;
    let mops = total_ops as f64 / result.elapsed.as_secs_f64() / 1_000_000.0;
    let (p50, p95, p99) = percentiles(&result.latency_ns);
    println!(
        "{engine:<8} {op:<8} threads={threads:<2} {ns:>10.1} ns/op {mops:>8.3} Mops/s sampled_p50={p50:>8}ns sampled_p95={p95:>8}ns sampled_p99={p99:>8}ns samples={}",
        result.latency_ns.len()
    );
}

fn percentiles(values: &[u64]) -> (u64, u64, u64) {
    if values.is_empty() {
        return (0, 0, 0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    (
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
    )
}

fn percentile(sorted: &[u64], q: f64) -> u64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_list<T>(name: &str, default: &str, f: impl Fn(&str) -> T) -> Vec<T> {
    env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(f)
        .collect()
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn should_progress(done: usize, total: usize) -> bool {
    total >= 1_000_000 && (done == total || done % 1_000_000 == 0)
}

fn progress(engine: &str, stage: &str, done: usize, total: usize) {
    if env::var("HOLT_CONCURRENT_PROGRESS")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(total >= 1_000_000)
    {
        eprintln!("{engine} {stage}: {done}/{total}");
    }
}
