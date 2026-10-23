//! YCSB core-workload driver (workloads A–F) over the shared engine
//! adapters.
//!
//! This is a spec-faithful Rust port of the YCSB core workload
//! definitions (Cooper et al., SoCC'10; core properties from the
//! reference Java implementation):
//!
//! - keys: `user<fnv64(seq)>` (hashed insert order, the YCSB default)
//! - records: 1 KiB (the 10 × 100 B field default, stored as one value
//!   blob, the standard shape for KV-store bindings; updates rewrite
//!   the whole record)
//! - request distributions: scrambled-zipfian (theta 0.99), latest,
//!   and uniform scan lengths in [1, 100]
//! - workload mixes: A 50r/50u, B 95r/5u, C 100r, D 95r/5i (latest),
//!   E 95scan/5i, F 50r/50rmw
//!
//! Run sequence follows the official YCSB guidance: load once, run
//! A, B, C, F, D against the same store (D appends new records), then
//! reload fresh and run E.
//!
//! ```sh
//! HOLT_YCSB_RECORDS=1000000 HOLT_YCSB_OPS=1000000 \
//! HOLT_YCSB_ENGINES=holt,rocksdb,sqlite,sled,lmdb,redb \
//! cargo bench --manifest-path benches/Cargo.toml --bench ycsb
//! ```
//!
//! PostgreSQL rows additionally need `--features pg` and `HOLT_PG`.

mod workload_test_engines;

use std::time::Instant;

use workload_test_engines::{env_usize, open_engine, Batch, Engine, Opts};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const SEED: u64 = 0x5EED_0BAD_C0FF_EE01;
const ZIPF_THETA: f64 = 0.99;
const FIELD_COUNT: usize = 10;
const FIELD_BYTES: usize = 100;
const RECORD_BYTES: usize = FIELD_COUNT * FIELD_BYTES;
const LOAD_BATCH: usize = 1_000;
const LATENCY_SAMPLE_EVERY: usize = 16;
const MAX_SCAN_LEN: usize = 100;

fn main() {
    let records = env_usize("HOLT_YCSB_RECORDS", 1_000_000);
    let ops = env_usize("HOLT_YCSB_OPS", 1_000_000);
    let engines = env_list("HOLT_YCSB_ENGINES", "holt,rocksdb,sqlite,sled,lmdb,redb");
    let workloads = env_list("HOLT_YCSB_WORKLOADS", "a,b,c,f,d,e");
    let opts = Opts::from_env();

    println!(
        "ycsb records={records} ops={ops} value_bytes={RECORD_BYTES} cache_mb={} engines={} workloads={}",
        opts.cache_mb,
        engines.join(","),
        workloads.join(","),
    );
    println!(
        "profile=single_thread,persistent,async_durable (WAL/journal on, no per-op fsync); \
         keys=user<fnv64(seq)>; zipfian theta={ZIPF_THETA}; scan len uniform 1..={MAX_SCAN_LEN}"
    );

    // Official sequence: everything except E shares one loaded store;
    // E gets a fresh reload (its inserts + scans assume an untouched
    // keyspace).
    let (shared, fresh): (Vec<&String>, Vec<&String>) =
        workloads.iter().partition(|w| w.as_str() != "e");

    for engine_name in &engines {
        if !shared.is_empty() {
            let dir = TempDir::new().expect("tempdir");
            let mut engine = open_engine(engine_name, dir.path(), &opts);
            let mut state = KeySpace::new(records);
            load(engine.as_mut(), &mut state, records);
            for w in &shared {
                run_workload(engine.as_mut(), w, ops, &mut state);
            }
        }
        for w in &fresh {
            let dir = TempDir::new().expect("tempdir");
            let mut engine = open_engine(engine_name, dir.path(), &opts);
            let mut state = KeySpace::new(records);
            load(engine.as_mut(), &mut state, records);
            run_workload(engine.as_mut(), w, ops, &mut state);
        }
    }
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

// --------------------------------------------------------------------
// key space + generators
// --------------------------------------------------------------------

/// FNV-1a 64 over the little-endian bytes of `v` (YCSB `Utils.hash`
/// equivalent: stable, well-spread scramble of the sequence number).
fn fnv64(v: u64) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in v.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn key_for(seq: usize) -> Vec<u8> {
    format!("user{}", fnv64(seq as u64)).into_bytes()
}

struct KeySpace {
    /// Records present: sequences `0..count` exist.
    count: usize,
    zipf: Zipfian,
    rng: StdRng,
    value_counter: u64,
    template: Vec<u8>,
}

impl KeySpace {
    fn new(records: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut template = vec![0u8; RECORD_BYTES];
        rng.fill(template.as_mut_slice());
        Self {
            count: 0,
            zipf: Zipfian::new(records, ZIPF_THETA),
            rng,
            value_counter: 0,
            template,
        }
    }

    /// Unique-per-call 1 KiB record: the shared template stamped with a
    /// counter, so every insert/update writes distinct bytes without
    /// paying for 1 KiB of fresh randomness per op.
    fn next_value(&mut self) -> Vec<u8> {
        self.value_counter += 1;
        let mut v = self.template.clone();
        v[..8].copy_from_slice(&self.value_counter.to_le_bytes());
        v[8..16].copy_from_slice(&fnv64(self.value_counter).to_le_bytes());
        v
    }

    /// Scrambled-zipfian sample over the live keyspace.
    fn zipf_seq(&mut self) -> usize {
        if self.zipf.items < self.count {
            self.zipf.extend_to(self.count);
        }
        let z = self.zipf.next(&mut self.rng);
        (fnv64(z as u64) % self.count as u64) as usize
    }

    /// `latest` sample: zipfian-skewed towards the most recent insert.
    fn latest_seq(&mut self) -> usize {
        if self.zipf.items < self.count {
            self.zipf.extend_to(self.count);
        }
        let back = self.zipf.next(&mut self.rng);
        self.count - 1 - back.min(self.count - 1)
    }

    fn insert_seq(&mut self) -> usize {
        let seq = self.count;
        self.count += 1;
        seq
    }
}

/// YCSB's zipfian generator (Gray et al. "Quickly generating
/// billion-record synthetic databases" algorithm), with incremental
/// `zeta` extension when the item count grows.
struct Zipfian {
    items: usize,
    theta: f64,
    alpha: f64,
    zetan: f64,
    zeta2theta: f64,
    eta: f64,
}

impl Zipfian {
    fn new(items: usize, theta: f64) -> Self {
        assert!(items >= 2, "zipfian needs at least 2 items");
        let mut zetan = 0.0;
        for i in 1..=items {
            zetan += 1.0 / (i as f64).powf(theta);
        }
        let zeta2theta = 1.0 + 1.0 / 2f64.powf(theta);
        let alpha = 1.0 / (1.0 - theta);
        let eta = Self::eta_for(items, theta, zetan, zeta2theta);
        Self {
            items,
            theta,
            alpha,
            zetan,
            zeta2theta,
            eta,
        }
    }

    fn eta_for(items: usize, theta: f64, zetan: f64, zeta2theta: f64) -> f64 {
        (1.0 - (2.0 / items as f64).powf(1.0 - theta)) / (1.0 - zeta2theta / zetan)
    }

    fn extend_to(&mut self, items: usize) {
        for i in (self.items + 1)..=items {
            self.zetan += 1.0 / (i as f64).powf(self.theta);
        }
        self.items = items;
        self.eta = Self::eta_for(items, self.theta, self.zetan, self.zeta2theta);
    }

    fn next(&mut self, rng: &mut StdRng) -> usize {
        let u: f64 = rng.random();
        let uz = u * self.zetan;
        if uz < 1.0 {
            return 0;
        }
        if uz < 1.0 + 0.5f64.powf(self.theta) {
            return 1;
        }
        let v = (self.items as f64 * (self.eta * u - self.eta + 1.0).powf(self.alpha)) as usize;
        v.min(self.items - 1)
    }
}

// --------------------------------------------------------------------
// phases
// --------------------------------------------------------------------

fn load(engine: &mut dyn Engine, state: &mut KeySpace, records: usize) {
    let start = Instant::now();
    let mut batch = Batch::default();
    for _ in 0..records {
        let seq = state.insert_seq();
        batch.puts.push((key_for(seq), state.next_value()));
        if batch.puts.len() >= LOAD_BATCH {
            engine.batch(&batch);
            batch.puts.clear();
        }
    }
    if !batch.puts.is_empty() {
        engine.batch(&batch);
    }
    engine.finish_load();
    let elapsed = start.elapsed();
    println!(
        "ycsb engine={} phase=load records={} elapsed_s={:.2} kops={:.1} (batched {LOAD_BATCH}/txn, settle barrier included)",
        engine.name(),
        records,
        elapsed.as_secs_f64(),
        records as f64 / elapsed.as_secs_f64() / 1_000.0,
    );
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Op {
    Read,
    Update,
    Insert,
    Scan,
    Rmw,
}

impl Op {
    fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Update => "update",
            Self::Insert => "insert",
            Self::Scan => "scan",
            Self::Rmw => "rmw",
        }
    }
}

fn pick_op(workload: &str, roll: f64) -> Op {
    match workload {
        "a" => {
            if roll < 0.5 {
                Op::Read
            } else {
                Op::Update
            }
        }
        "b" => {
            if roll < 0.95 {
                Op::Read
            } else {
                Op::Update
            }
        }
        "c" => Op::Read,
        "d" => {
            if roll < 0.95 {
                Op::Read
            } else {
                Op::Insert
            }
        }
        "e" => {
            if roll < 0.95 {
                Op::Scan
            } else {
                Op::Insert
            }
        }
        "f" => {
            if roll < 0.5 {
                Op::Read
            } else {
                Op::Rmw
            }
        }
        other => panic!("unknown workload `{other}`; use a..f"),
    }
}

fn run_workload(engine: &mut dyn Engine, workload: &str, ops: usize, state: &mut KeySpace) {
    let mut lat: std::collections::HashMap<Op, Vec<u64>> = Default::default();
    let mut counts: std::collections::HashMap<Op, usize> = Default::default();
    let mut misses = 0usize;
    let mut op_rng = StdRng::seed_from_u64(SEED ^ 0xA11_0F_0C5);

    let start = Instant::now();
    for i in 0..ops {
        let roll: f64 = op_rng.random();
        let op = pick_op(workload, roll);
        let sampled = i % LATENCY_SAMPLE_EVERY == 0;
        let t0 = sampled.then(Instant::now);
        match op {
            Op::Read => {
                let seq = if workload == "d" {
                    state.latest_seq()
                } else {
                    state.zipf_seq()
                };
                if engine.get(&key_for(seq)).is_none() {
                    misses += 1;
                }
            }
            Op::Update => {
                let seq = state.zipf_seq();
                let value = state.next_value();
                engine.put(&key_for(seq), &value);
            }
            Op::Insert => {
                let seq = state.insert_seq();
                let value = state.next_value();
                engine.put(&key_for(seq), &value);
            }
            Op::Scan => {
                let seq = state.zipf_seq();
                let len = op_rng.random_range(1..=MAX_SCAN_LEN);
                engine.scan_from(&key_for(seq), len);
            }
            Op::Rmw => {
                let seq = state.zipf_seq();
                let key = key_for(seq);
                if engine.get(&key).is_none() {
                    misses += 1;
                }
                let value = state.next_value();
                engine.put(&key, &value);
            }
        }
        if let Some(t0) = t0 {
            lat.entry(op).or_default().push(t0.elapsed().as_nanos() as u64);
        }
        *counts.entry(op).or_default() += 1;
    }
    let elapsed = start.elapsed();

    let mut parts: Vec<String> = Vec::new();
    let mut ordered: Vec<(&Op, &Vec<u64>)> = lat.iter().collect();
    ordered.sort_by_key(|(op, _)| op.label());
    for (op, samples) in ordered {
        let mut s = samples.clone();
        s.sort_unstable();
        parts.push(format!(
            "{}(n={} p50={}ns p99={}ns)",
            op.label(),
            counts[op],
            percentile(&s, 0.50),
            percentile(&s, 0.99),
        ));
    }
    println!(
        "ycsb engine={} workload={} phase=run ops={} elapsed_s={:.2} kops={:.1} misses={} {}",
        engine.name(),
        workload,
        ops,
        elapsed.as_secs_f64(),
        ops as f64 / elapsed.as_secs_f64() / 1_000.0,
        misses,
        parts.join(" "),
    );
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}
