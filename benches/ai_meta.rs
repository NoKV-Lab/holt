//! AI metadata workload suite over the shared engine adapters.
//!
//! No official AI benchmark targets embedded metadata engines (MLPerf
//! Storage v3.0 measures filesystem/S3 bandwidth via DLIO; VectorDBBench
//! needs vector search), so this suite models the *metadata plane* of
//! the four AI storage workloads MLPerf Storage v3.0 standardized —
//! KV-cache paging, training-data loading, checkpointing — plus the
//! artifact-registry shape that serves model distribution:
//!
//! - `kvcache`  — chained LLM prefix-cache block index
//!   (Mooncake/LMCache-shaped): longest-prefix chain lookup, block
//!   fetch, chain extension + eviction.
//! - `manifest` — training-dataset shard/sample manifest: shuffled
//!   per-epoch point reads, shard listings, shard ingest.
//! - `ckpt`     — checkpoint shard catalog: atomic step commit +
//!   retention prune, step reads, step listing.
//! - `registry` — HF-style model/artifact registry: file metadata
//!   reads, revision listing with delimiter rollup, CAS updates,
//!   revision publish.
//!
//! Chain lookup is reported twice per engine: `lookup` uses the
//! engine's native longest-prefix path (holt: one `longest_prefix_record`
//! descent; others: the forward walk-until-miss a chained KV-cache
//! index performs on a flat KV store), and `lookup_bisect` gives every
//! engine the smarter O(log d) binary search over chain depths, so the
//! comparison does not rest on a strawman app layer.
//!
//! ```sh
//! HOLT_AI_ENGINES=holt,rocksdb,sqlite,sled,lmdb,redb \
//! cargo bench --manifest-path benches/Cargo.toml --bench ai_meta
//! ```
//!
//! PostgreSQL rows additionally need `--features pg` and `HOLT_PG`.

mod workload_test_engines;

use std::time::{Duration, Instant};

use workload_test_engines::{env_usize, open_engine, Batch, Engine, Opts};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const SEED: u64 = 0xA1_0E7A_0001;

fn main() {
    let engines = env_list("HOLT_AI_ENGINES", "holt,rocksdb,sqlite,sled,lmdb,redb");
    let scenarios = env_list("HOLT_AI_SCENARIOS", "kvcache,manifest,ckpt,registry,s3page");
    let point_ops = env_usize("HOLT_AI_POINT_OPS", 200_000);
    let list_ops = env_usize("HOLT_AI_LIST_OPS", 20_000);
    let batch_ops = env_usize("HOLT_AI_BATCH_OPS", 500);
    let mixed_ops = env_usize("HOLT_AI_MIXED_OPS", 100_000);
    let opts = Opts::from_env();

    println!(
        "ai_meta engines={} scenarios={} point_ops={point_ops} list_ops={list_ops} batch_ops={batch_ops} mixed_ops={mixed_ops} cache_mb={}",
        engines.join(","),
        scenarios.join(","),
        opts.cache_mb,
    );
    println!(
        "profile=single_thread,persistent,async_durable (WAL/journal on, no per-op fsync)"
    );

    let counts = OpCounts {
        point: point_ops,
        list: list_ops,
        batch: batch_ops,
        mixed: mixed_ops,
    };
    for engine_name in &engines {
        for scenario in &scenarios {
            let dir = TempDir::new().expect("tempdir");
            let mut engine = open_engine(engine_name, dir.path(), &opts);
            match scenario.as_str() {
                "kvcache" => kvcache(engine.as_mut(), &counts),
                "manifest" => manifest(engine.as_mut(), &counts),
                "ckpt" => ckpt(engine.as_mut(), &counts),
                "registry" => registry(engine.as_mut(), &counts),
                "s3page" => s3page(engine.as_mut(), &counts),
                other => panic!("unknown scenario `{other}`"),
            }
        }
    }
}

struct OpCounts {
    point: usize,
    list: usize,
    batch: usize,
    mixed: usize,
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn fnv64(v: u64) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in v.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn report(engine: &str, scenario: &str, op: &str, ops: usize, elapsed: Duration) {
    let ns = elapsed.as_secs_f64() * 1_000_000_000.0 / ops.max(1) as f64;
    let kops = ops as f64 / elapsed.as_secs_f64() / 1_000.0;
    println!(
        "ai_meta engine={engine:<8} scenario={scenario:<8} op={op:<14} ops={ops:>8} {ns:>12.1} ns/op {kops:>10.1} kops/s"
    );
}

fn time<F: FnMut()>(ops: usize, mut f: F) -> Duration {
    let start = Instant::now();
    for _ in 0..ops {
        f();
    }
    start.elapsed()
}

/// Binary search for the deepest stored chain prefix (prefix-closed key
/// set), charged through the engine's ordinary point-get path.
fn lpm_bisect(engine: &mut dyn Engine, query: &[u8], probes: &[usize]) -> usize {
    let mut lo = 0usize;
    let mut hi = probes.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if engine.get(&query[..probes[mid]]).is_some() {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        0
    } else {
        probes[lo - 1]
    }
}

// --------------------------------------------------------------------
// kvcache — chained LLM prefix-cache block index
// --------------------------------------------------------------------
//
// One cached chain per session: block i is keyed by the path of chain
// hashes `kv/llama70b/<h0>/<h1>/.../<hi>`, value = block descriptor
// (segment file, offset, length). The set is prefix-closed (a cached
// block implies its ancestors are cached; eviction trims suffixes), so
// "how much of this request's prefix is cached?" is a longest-prefix
// match over the chain path.

const KV_BASE: &str = "kv/llama70b/";
const KV_SEG_BYTES: usize = 9; // 8 hex chars + '/'
const KV_MIN_DEPTH: usize = 4;
const KV_MAX_DEPTH: usize = 32;
const KV_EXTEND: usize = 4;

fn kv_depth(chain: usize) -> usize {
    KV_MIN_DEPTH + (fnv64(chain as u64) as usize) % (KV_MAX_DEPTH - KV_MIN_DEPTH + 1)
}

fn kv_seg(chain: usize, level: usize) -> u32 {
    (fnv64((chain as u64) << 8 | level as u64) & 0xFFFF_FFFF) as u32
}

/// Chain path with `levels` segments; returns the key and the probe
/// table (byte length of each depth's key, ascending).
fn kv_path(chain: usize, levels: usize) -> (Vec<u8>, Vec<usize>) {
    let mut key = Vec::with_capacity(KV_BASE.len() + levels * KV_SEG_BYTES);
    key.extend_from_slice(KV_BASE.as_bytes());
    let mut probes = Vec::with_capacity(levels);
    for level in 0..levels {
        if level > 0 {
            key.push(b'/');
        }
        key.extend_from_slice(format!("{:08x}", kv_seg(chain, level)).as_bytes());
        probes.push(key.len());
    }
    (key, probes)
}

fn kv_value(chain: usize, level: usize) -> Vec<u8> {
    format!(
        "{{\"seg\":{},\"off\":{},\"len\":131072}}",
        fnv64(chain as u64) % 4096,
        level * 131072,
    )
    .into_bytes()
}

fn kvcache(engine: &mut dyn Engine, counts: &OpCounts) {
    let chains = env_usize("HOLT_AI_KV_CHAINS", 30_000);
    let name = engine.name().to_string();

    // Preload: every chain's blocks, one atomic batch per chain.
    let mut preloaded = 0usize;
    let t_load = Instant::now();
    for c in 0..chains {
        let depth = kv_depth(c);
        let (key, probes) = kv_path(c, depth);
        let mut batch = Batch::default();
        for (level, &len) in probes.iter().enumerate() {
            batch.puts.push((key[..len].to_vec(), kv_value(c, level)));
        }
        preloaded += depth;
        engine.batch(&batch);
    }
    engine.finish_load();
    println!(
        "ai_meta engine={name} scenario=kvcache preload chains={chains} blocks={preloaded} elapsed_s={:.2}",
        t_load.elapsed().as_secs_f64()
    );

    let mut rng = StdRng::seed_from_u64(SEED);

    // lookup: native longest-prefix path. 80% hit (existing chain, query
    // two phantom segments deeper), 20% miss (unknown chain).
    let mut i = 0usize;
    let elapsed = time(counts.point, || {
        let roll: f64 = rng.random();
        let (query, probes) = if roll < 0.8 {
            let c = i % chains;
            kv_path(c, kv_depth(c) + 2)
        } else {
            kv_path(chains + i, KV_MIN_DEPTH + 2)
        };
        std::hint::black_box(engine.longest_prefix(&query, &probes));
        i += 1;
    });
    report(&name, "kvcache", "lookup", counts.point, elapsed);

    // lookup_bisect: same queries, O(log d) point-get binary search.
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut i = 0usize;
    let elapsed = time(counts.point, || {
        let roll: f64 = rng.random();
        let (query, probes) = if roll < 0.8 {
            let c = i % chains;
            kv_path(c, kv_depth(c) + 2)
        } else {
            kv_path(chains + i, KV_MIN_DEPTH + 2)
        };
        std::hint::black_box(lpm_bisect(engine, &query, &probes));
        i += 1;
    });
    report(&name, "kvcache", "lookup_bisect", counts.point, elapsed);

    // block_get: random cached block descriptor fetch.
    let mut rng = StdRng::seed_from_u64(SEED ^ 1);
    let elapsed = time(counts.point, || {
        let c = rng.random_range(0..chains);
        let depth = kv_depth(c);
        let level = rng.random_range(0..depth);
        let (key, probes) = kv_path(c, level + 1);
        std::hint::black_box(engine.get(&key[..probes[level]]));
    });
    report(&name, "kvcache", "block_get", counts.point, elapsed);

    // extend_evict: admit KV_EXTEND new blocks onto a chain (atomic),
    // then evict the same suffix (atomic) — bounded cache churn.
    let mut rng = StdRng::seed_from_u64(SEED ^ 2);
    let elapsed = time(counts.batch, || {
        let c = rng.random_range(0..chains);
        let depth = kv_depth(c);
        let (key, probes) = kv_path(c, depth + KV_EXTEND);
        let mut admit = Batch::default();
        for level in depth..depth + KV_EXTEND {
            admit
                .puts
                .push((key[..probes[level]].to_vec(), kv_value(c, level)));
        }
        engine.batch(&admit);
        let mut evict = Batch::default();
        for level in depth..depth + KV_EXTEND {
            evict.deletes.push(key[..probes[level]].to_vec());
        }
        engine.batch(&evict);
    });
    report(&name, "kvcache", "extend_evict", counts.batch, elapsed);

    // mixed: 70% chain lookup, 15% block fetch, 15% extend+evict pair.
    let mut rng = StdRng::seed_from_u64(SEED ^ 3);
    let mut i = 0usize;
    let elapsed = time(counts.mixed, || {
        let roll: f64 = rng.random();
        if roll < 0.70 {
            let c = i % chains;
            let (query, probes) = kv_path(c, kv_depth(c) + 2);
            std::hint::black_box(engine.longest_prefix(&query, &probes));
        } else if roll < 0.85 {
            let c = rng.random_range(0..chains);
            let level = rng.random_range(0..kv_depth(c));
            let (key, probes) = kv_path(c, level + 1);
            std::hint::black_box(engine.get(&key[..probes[level]]));
        } else {
            let c = rng.random_range(0..chains);
            let depth = kv_depth(c);
            let (key, probes) = kv_path(c, depth + KV_EXTEND);
            let mut admit = Batch::default();
            for level in depth..depth + KV_EXTEND {
                admit
                    .puts
                    .push((key[..probes[level]].to_vec(), kv_value(c, level)));
            }
            engine.batch(&admit);
            let mut evict = Batch::default();
            for level in depth..depth + KV_EXTEND {
                evict.deletes.push(key[..probes[level]].to_vec());
            }
            engine.batch(&evict);
        }
        i += 1;
    });
    report(&name, "kvcache", "mixed", counts.mixed, elapsed);
}

// --------------------------------------------------------------------
// s3page — bucket listing pages, minio#17472 shape
// --------------------------------------------------------------------
//
// One bucket holding many "directories": a ListObjectsV2 page with
// delimiter=/ and maxKeys=1000 rolls up 1000 CommonPrefix entries per
// request. MinIO reports OOM-kill at ~200 such list calls/s over a
// 10M-object bucket (minio/minio#17472); Ceph RGW's hash-sharded
// index must merge-sort every shard per listing. This scenario
// measures exactly that page over an ordered metadata engine.

fn s3page_key(dir: usize, obj: usize) -> Vec<u8> {
    format!("bkt/dir-{dir:05}/obj-{obj:04}").into_bytes()
}

fn s3page_value(dir: usize, obj: usize) -> Vec<u8> {
    format!(
        "{{\"size\":{},\"etag\":\"{:016x}\",\"class\":\"STD\"}}",
        4096 + obj,
        fnv64((dir * 1000 + obj) as u64),
    )
    .into_bytes()
}

fn s3page(engine: &mut dyn Engine, counts: &OpCounts) {
    let dirs = env_usize("HOLT_AI_S3P_DIRS", 20_000);
    let objs = env_usize("HOLT_AI_S3P_OBJS", 100);
    let name = engine.name().to_string();

    let t_load = Instant::now();
    let mut batch = Batch::default();
    for d in 0..dirs {
        for o in 0..objs {
            batch.puts.push((s3page_key(d, o), s3page_value(d, o)));
            if batch.puts.len() >= 1_000 {
                engine.batch(&batch);
                batch.puts.clear();
            }
        }
    }
    if !batch.puts.is_empty() {
        engine.batch(&batch);
    }
    engine.finish_load();
    println!(
        "ai_meta engine={name} scenario=s3page preload dirs={dirs} objs_per_dir={objs} keys={} elapsed_s={:.2}",
        dirs * objs,
        t_load.elapsed().as_secs_f64()
    );

    // page_rollup: one ListObjectsV2 page — 1000 CommonPrefix rollups.
    let elapsed = time(counts.list, || {
        std::hint::black_box(engine.list_dir(b"bkt/", b'/', 1_000));
    });
    report(&name, "s3page", "page_rollup", counts.list, elapsed);

    // dir_page: one leaf-directory page (all objects of one prefix).
    let mut rng = StdRng::seed_from_u64(SEED ^ 40);
    let elapsed = time(counts.list, || {
        let d = rng.random_range(0..dirs);
        let prefix = format!("bkt/dir-{d:05}/");
        std::hint::black_box(engine.scan_prefix(prefix.as_bytes(), objs));
    });
    report(&name, "s3page", "dir_page", counts.list, elapsed);

    // obj_stat: HEAD-object metadata fetch.
    let mut rng = StdRng::seed_from_u64(SEED ^ 41);
    let elapsed = time(counts.point, || {
        let d = rng.random_range(0..dirs);
        let o = rng.random_range(0..objs);
        std::hint::black_box(engine.get(&s3page_key(d, o)));
    });
    report(&name, "s3page", "obj_stat", counts.point, elapsed);
}

// --------------------------------------------------------------------
// manifest — training-dataset shard/sample manifest
// --------------------------------------------------------------------

fn manifest_key(shard: usize, sample: usize) -> Vec<u8> {
    format!("ds/laion/shard-{shard:05}/sample-{sample:08}").into_bytes()
}

fn manifest_value(shard: usize, sample: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(24);
    v.extend_from_slice(&((shard * 1_000_000 + sample) as u64 * 4096).to_le_bytes());
    v.extend_from_slice(&(147_456u32).to_le_bytes());
    v.extend_from_slice(&(fnv64(sample as u64) as u32).to_le_bytes());
    v.extend_from_slice(&(1_760_000_000u64).to_le_bytes());
    v
}

fn manifest(engine: &mut dyn Engine, counts: &OpCounts) {
    let shards = env_usize("HOLT_AI_MANIFEST_SHARDS", 200);
    let per_shard = env_usize("HOLT_AI_MANIFEST_SAMPLES", 10_000);
    let total = shards * per_shard;
    let name = engine.name().to_string();

    let t_load = Instant::now();
    let mut batch = Batch::default();
    for s in 0..shards {
        for i in 0..per_shard {
            batch.puts.push((manifest_key(s, i), manifest_value(s, i)));
            if batch.puts.len() >= 1_000 {
                engine.batch(&batch);
                batch.puts.clear();
            }
        }
    }
    if !batch.puts.is_empty() {
        engine.batch(&batch);
    }
    engine.finish_load();
    println!(
        "ai_meta engine={name} scenario=manifest preload shards={shards} samples={total} elapsed_s={:.2}",
        t_load.elapsed().as_secs_f64()
    );

    // sample_get: shuffled per-epoch access — uniform random point reads.
    let mut rng = StdRng::seed_from_u64(SEED ^ 10);
    let elapsed = time(counts.point, || {
        let s = rng.random_range(0..shards);
        let i = rng.random_range(0..per_shard);
        std::hint::black_box(engine.get(&manifest_key(s, i)));
    });
    report(&name, "manifest", "sample_get", counts.point, elapsed);

    // shard_scan: data-loader readahead — first 100 records of a shard.
    let mut rng = StdRng::seed_from_u64(SEED ^ 11);
    let elapsed = time(counts.list, || {
        let s = rng.random_range(0..shards);
        let prefix = format!("ds/laion/shard-{s:05}/");
        std::hint::black_box(engine.scan_prefix(prefix.as_bytes(), 100));
    });
    report(&name, "manifest", "shard_scan", counts.list, elapsed);

    // ingest_shard: register a new 1000-sample shard, then retire it.
    let elapsed = time(counts.batch, || {
        let s = shards + 1;
        let mut admit = Batch::default();
        for i in 0..1_000 {
            admit.puts.push((manifest_key(s, i), manifest_value(s, i)));
        }
        engine.batch(&admit);
        let mut retire = Batch::default();
        for i in 0..1_000 {
            retire.deletes.push(manifest_key(s, i));
        }
        engine.batch(&retire);
    });
    report(&name, "manifest", "ingest_shard", counts.batch, elapsed);

    // mixed: 90% sample reads, 8% shard scans, 2% 100-sample ingest.
    let mut rng = StdRng::seed_from_u64(SEED ^ 12);
    let elapsed = time(counts.mixed, || {
        let roll: f64 = rng.random();
        if roll < 0.90 {
            let s = rng.random_range(0..shards);
            let i = rng.random_range(0..per_shard);
            std::hint::black_box(engine.get(&manifest_key(s, i)));
        } else if roll < 0.98 {
            let s = rng.random_range(0..shards);
            let prefix = format!("ds/laion/shard-{s:05}/");
            std::hint::black_box(engine.scan_prefix(prefix.as_bytes(), 100));
        } else {
            let s = shards + 2;
            let mut admit = Batch::default();
            for i in 0..100 {
                admit.puts.push((manifest_key(s, i), manifest_value(s, i)));
            }
            engine.batch(&admit);
            let mut retire = Batch::default();
            for i in 0..100 {
                retire.deletes.push(manifest_key(s, i));
            }
            engine.batch(&retire);
        }
    });
    report(&name, "manifest", "mixed", counts.mixed, elapsed);
}

// --------------------------------------------------------------------
// ckpt — checkpoint shard catalog
// --------------------------------------------------------------------

const CKPT_JOBS: usize = 4;
const CKPT_STEPS: usize = 200;
const CKPT_RANKS: usize = 256;

fn ckpt_key(job: usize, step: usize, rank: usize) -> Vec<u8> {
    format!("ckpt/job-{job}/step-{step:08}/shard-{rank:04}").into_bytes()
}

fn ckpt_value(job: usize, step: usize, rank: usize) -> Vec<u8> {
    format!(
        "{{\"path\":\"s3://ckpt/job-{job}/{step}/{rank}.distcp\",\"bytes\":{},\"etag\":\"{:016x}\"}}",
        17_179_869_184u64 + rank as u64,
        fnv64((step * CKPT_RANKS + rank) as u64),
    )
    .into_bytes()
}

fn ckpt(engine: &mut dyn Engine, counts: &OpCounts) {
    let name = engine.name().to_string();

    let t_load = Instant::now();
    for job in 0..CKPT_JOBS {
        for step in 0..CKPT_STEPS {
            let mut batch = Batch::default();
            for rank in 0..CKPT_RANKS {
                batch.puts.push((ckpt_key(job, step, rank), ckpt_value(job, step, rank)));
            }
            engine.batch(&batch);
        }
    }
    engine.finish_load();
    println!(
        "ai_meta engine={name} scenario=ckpt preload jobs={CKPT_JOBS} steps={CKPT_STEPS} ranks={CKPT_RANKS} entries={} elapsed_s={:.2}",
        CKPT_JOBS * CKPT_STEPS * CKPT_RANKS,
        t_load.elapsed().as_secs_f64()
    );

    // shard_get: resume path — fetch one rank's shard record.
    let mut rng = StdRng::seed_from_u64(SEED ^ 20);
    let elapsed = time(counts.point, || {
        let job = rng.random_range(0..CKPT_JOBS);
        let step = rng.random_range(0..CKPT_STEPS);
        let rank = rng.random_range(0..CKPT_RANKS);
        std::hint::black_box(engine.get(&ckpt_key(job, step, rank)));
    });
    report(&name, "ckpt", "shard_get", counts.point, elapsed);

    // step_read: full restore manifest — all shards of one step.
    let mut rng = StdRng::seed_from_u64(SEED ^ 21);
    let elapsed = time(counts.list, || {
        let job = rng.random_range(0..CKPT_JOBS);
        let step = rng.random_range(0..CKPT_STEPS);
        let prefix = format!("ckpt/job-{job}/step-{step:08}/");
        std::hint::black_box(engine.scan_prefix(prefix.as_bytes(), CKPT_RANKS));
    });
    report(&name, "ckpt", "step_read", counts.list, elapsed);

    // steps_list: "what checkpoints exist?" — delimiter rollup of step
    // directories under one job.
    let mut rng = StdRng::seed_from_u64(SEED ^ 22);
    let elapsed = time(counts.list, || {
        let job = rng.random_range(0..CKPT_JOBS);
        let prefix = format!("ckpt/job-{job}/");
        std::hint::black_box(engine.list_dir(prefix.as_bytes(), b'/', 20));
    });
    report(&name, "ckpt", "steps_list", counts.list, elapsed);

    // rotate: one checkpoint cycle — atomically commit a new step's 256
    // shard records and prune the oldest step (steady-state retention).
    let mut next_step = CKPT_STEPS;
    let elapsed = time(counts.batch, || {
        let job = 0usize;
        let new_step = next_step;
        let old_step = next_step - CKPT_STEPS;
        next_step += 1;
        let mut batch = Batch::default();
        for rank in 0..CKPT_RANKS {
            batch
                .puts
                .push((ckpt_key(job, new_step, rank), ckpt_value(job, new_step, rank)));
            batch.deletes.push(ckpt_key(job, old_step, rank));
        }
        engine.batch(&batch);
    });
    report(&name, "ckpt", "rotate", counts.batch, elapsed);

    // mixed: 60% shard reads, 20% step restores, 15% listings, 5% rotate.
    let mut rng = StdRng::seed_from_u64(SEED ^ 23);
    let elapsed = time(counts.mixed, || {
        let roll: f64 = rng.random();
        if roll < 0.60 {
            let job = rng.random_range(1..CKPT_JOBS);
            let step = rng.random_range(0..CKPT_STEPS);
            let rank = rng.random_range(0..CKPT_RANKS);
            std::hint::black_box(engine.get(&ckpt_key(job, step, rank)));
        } else if roll < 0.80 {
            let job = rng.random_range(1..CKPT_JOBS);
            let step = rng.random_range(0..CKPT_STEPS);
            let prefix = format!("ckpt/job-{job}/step-{step:08}/");
            std::hint::black_box(engine.scan_prefix(prefix.as_bytes(), CKPT_RANKS));
        } else if roll < 0.95 {
            let job = rng.random_range(1..CKPT_JOBS);
            let prefix = format!("ckpt/job-{job}/");
            std::hint::black_box(engine.list_dir(prefix.as_bytes(), b'/', 20));
        } else {
            let job = 0usize;
            let new_step = next_step;
            let old_step = next_step - CKPT_STEPS;
            next_step += 1;
            let mut batch = Batch::default();
            for rank in 0..CKPT_RANKS {
                batch
                    .puts
                    .push((ckpt_key(job, new_step, rank), ckpt_value(job, new_step, rank)));
                batch.deletes.push(ckpt_key(job, old_step, rank));
            }
            engine.batch(&batch);
        }
    });
    report(&name, "ckpt", "mixed", counts.mixed, elapsed);
}

// --------------------------------------------------------------------
// registry — HF-style model/artifact registry
// --------------------------------------------------------------------

const REG_ORGS: usize = 50;
const REG_REPOS: usize = 20;
const REG_REVS: usize = 4;
const REG_WEIGHT_FILES: usize = 36;
const REG_RUN_FILES: usize = 10;

fn reg_rev_prefix(org: usize, repo: usize, rev: usize) -> String {
    format!("reg/org-{org:03}/repo-{repo:03}/rev-{rev:02}/")
}

/// File paths within one revision: root metadata files, sharded
/// weights, and a nested `runs/` subtree so delimiter listing has
/// something to roll up.
fn reg_file(idx: usize) -> String {
    match idx {
        0 => ".gitattributes".to_string(),
        1 => "config.json".to_string(),
        2 => "tokenizer.json".to_string(),
        3 => "README.md".to_string(),
        i if i < 4 + REG_WEIGHT_FILES => format!(
            "model-{:05}-of-{REG_WEIGHT_FILES:05}.safetensors",
            i - 4 + 1
        ),
        i => format!("runs/evt-{:03}/log.json", i - 4 - REG_WEIGHT_FILES),
    }
}

const REG_FILES: usize = 4 + REG_WEIGHT_FILES + REG_RUN_FILES;

fn reg_key(org: usize, repo: usize, rev: usize, idx: usize) -> Vec<u8> {
    format!("{}{}", reg_rev_prefix(org, repo, rev), reg_file(idx)).into_bytes()
}

fn reg_value(seed: u64) -> Vec<u8> {
    format!(
        "{{\"size\":{},\"etag\":\"{:016x}\",\"lfs\":true}}",
        4_900_000_000u64 + (seed % 100_000_000),
        fnv64(seed),
    )
    .into_bytes()
}

fn registry(engine: &mut dyn Engine, counts: &OpCounts) {
    let name = engine.name().to_string();
    let total = REG_ORGS * REG_REPOS * REG_REVS * REG_FILES;

    let t_load = Instant::now();
    let mut batch = Batch::default();
    for org in 0..REG_ORGS {
        for repo in 0..REG_REPOS {
            for rev in 0..REG_REVS {
                for idx in 0..REG_FILES {
                    batch.puts.push((
                        reg_key(org, repo, rev, idx),
                        reg_value((org * 1000 + repo * 10 + rev + idx) as u64),
                    ));
                }
                engine.batch(&batch);
                batch.puts.clear();
            }
        }
    }
    engine.finish_load();
    println!(
        "ai_meta engine={name} scenario=registry preload orgs={REG_ORGS} repos={REG_REPOS} revs={REG_REVS} files={total} elapsed_s={:.2}",
        t_load.elapsed().as_secs_f64()
    );

    let mut rng = StdRng::seed_from_u64(SEED ^ 30);
    let elapsed = time(counts.point, || {
        let org = rng.random_range(0..REG_ORGS);
        let repo = rng.random_range(0..REG_REPOS);
        let rev = rng.random_range(0..REG_REVS);
        let idx = rng.random_range(0..REG_FILES);
        std::hint::black_box(engine.get(&reg_key(org, repo, rev, idx)));
    });
    report(&name, "registry", "meta_get", counts.point, elapsed);

    // rev_list_dir: repo page view — root entries + runs/ rollup.
    let mut rng = StdRng::seed_from_u64(SEED ^ 31);
    let elapsed = time(counts.list, || {
        let org = rng.random_range(0..REG_ORGS);
        let repo = rng.random_range(0..REG_REPOS);
        let rev = rng.random_range(0..REG_REVS);
        let prefix = reg_rev_prefix(org, repo, rev);
        std::hint::black_box(engine.list_dir(prefix.as_bytes(), b'/', 50));
    });
    report(&name, "registry", "rev_list_dir", counts.list, elapsed);

    // rev_scan: full revision manifest (download planner).
    let mut rng = StdRng::seed_from_u64(SEED ^ 32);
    let elapsed = time(counts.list, || {
        let org = rng.random_range(0..REG_ORGS);
        let repo = rng.random_range(0..REG_REPOS);
        let rev = rng.random_range(0..REG_REVS);
        let prefix = reg_rev_prefix(org, repo, rev);
        std::hint::black_box(engine.scan_prefix(prefix.as_bytes(), 100));
    });
    report(&name, "registry", "rev_scan", counts.list, elapsed);

    // cas_touch: guarded metadata update of one file record.
    let mut rng = StdRng::seed_from_u64(SEED ^ 33);
    let mut counter = 0u64;
    let elapsed = time(counts.point, || {
        let org = rng.random_range(0..REG_ORGS);
        let repo = rng.random_range(0..REG_REPOS);
        let rev = rng.random_range(0..REG_REVS);
        let idx = rng.random_range(0..REG_FILES);
        counter += 1;
        assert!(engine.cas(&reg_key(org, repo, rev, idx), &reg_value(counter)));
    });
    report(&name, "registry", "cas_touch", counts.point, elapsed);

    // publish_prune: push a full new revision atomically, then prune it.
    let elapsed = time(counts.batch, || {
        let (org, repo, rev) = (0usize, 0usize, REG_REVS + 1);
        let mut publish = Batch::default();
        for idx in 0..REG_FILES {
            publish.puts.push((reg_key(org, repo, rev, idx), reg_value(idx as u64)));
        }
        engine.batch(&publish);
        let mut prune = Batch::default();
        for idx in 0..REG_FILES {
            prune.deletes.push(reg_key(org, repo, rev, idx));
        }
        engine.batch(&prune);
    });
    report(&name, "registry", "publish_prune", counts.batch, elapsed);

    // mixed: 60% reads, 15% listings, 10% scans, 10% CAS, 5% publish.
    let mut rng = StdRng::seed_from_u64(SEED ^ 34);
    let elapsed = time(counts.mixed, || {
        let roll: f64 = rng.random();
        let org = rng.random_range(0..REG_ORGS);
        let repo = rng.random_range(0..REG_REPOS);
        let rev = rng.random_range(0..REG_REVS);
        if roll < 0.60 {
            let idx = rng.random_range(0..REG_FILES);
            std::hint::black_box(engine.get(&reg_key(org, repo, rev, idx)));
        } else if roll < 0.75 {
            let prefix = reg_rev_prefix(org, repo, rev);
            std::hint::black_box(engine.list_dir(prefix.as_bytes(), b'/', 50));
        } else if roll < 0.85 {
            let prefix = reg_rev_prefix(org, repo, rev);
            std::hint::black_box(engine.scan_prefix(prefix.as_bytes(), 100));
        } else if roll < 0.95 {
            let idx = rng.random_range(0..REG_FILES);
            counter += 1;
            std::hint::black_box(engine.cas(&reg_key(org, repo, rev, idx), &reg_value(counter)));
        } else {
            let rev = REG_REVS + 2;
            let mut publish = Batch::default();
            for idx in 0..REG_FILES {
                publish.puts.push((reg_key(org, repo, rev, idx), reg_value(idx as u64)));
            }
            engine.batch(&publish);
            let mut prune = Batch::default();
            for idx in 0..REG_FILES {
                prune.deletes.push(reg_key(org, repo, rev, idx));
            }
            engine.batch(&prune);
        }
    });
    report(&name, "registry", "mixed", counts.mixed, elapsed);
}
