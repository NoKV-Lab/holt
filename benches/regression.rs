//! Holt-only benchmark used as a lightweight regression guard.
//!
//! The public comparator harness lives in `main.rs`. Keeping CI on this
//! Holt-only target avoids compiling or initializing RocksDB, SQLite, and
//! sled in the push-time CI path.

use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use holt::{JournalAnchor, JournalEnvelope, Tree, TreeConfig, DB};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use tempfile::TempDir;

const SEED: u64 = 0xC10A_D15E_A5ED;
const KEY_COUNT: usize = 20_000;
const JOURNAL_SAMPLE_COMMIT_CAP: u64 = 4_096;

fn gen_kv_dataset() -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::with_capacity(KEY_COUNT);
    let mut rng = StdRng::seed_from_u64(SEED);
    for _ in 0..KEY_COUNT {
        let mut key = vec![0u8; 32];
        rng.fill_bytes(&mut key);
        let mut value = vec![0u8; 64];
        rng.fill_bytes(&mut value);
        out.push((key, value));
    }
    out
}

fn gen_objstore_dataset() -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..KEY_COUNT)
        .map(|i| {
            let bucket = i / 1000;
            let file = i % 1000;
            let key = format!(
                "bucket-{bucket:03}/tenant-{tenant:02}/path/file-{file:08}.bin",
                tenant = bucket % 32,
            )
            .into_bytes();
            let value = format!(
                "{{\"size\":{size:016},\"etag\":\"{etag:016x}\"}}",
                size = i as u64 * 1024,
                etag = (i as u64).wrapping_mul(0x9E37_79B9),
            )
            .into_bytes();
            (key, value)
        })
        .collect()
}

fn gen_fs_dataset() -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..KEY_COUNT)
        .map(|i| {
            let dir = i / 1000;
            let file = i % 1000;
            let key = format!("/usr/local/share/category-{dir:03}/file-{file:08}").into_bytes();
            let mut value = Vec::with_capacity(32);
            value.extend_from_slice(&((i as u64) * 4096).to_le_bytes());
            value.extend_from_slice(&(1_700_000_000u64 + i as u64).to_le_bytes());
            value.extend_from_slice(&0o644u32.to_le_bytes());
            value.extend_from_slice(&1000u32.to_le_bytes());
            value.extend_from_slice(&1000u32.to_le_bytes());
            value.extend_from_slice(&1u32.to_le_bytes());
            (key, value)
        })
        .collect()
}

fn make_tree() -> (Tree, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut cfg = TreeConfig::new(dir.path());
    cfg.durability = holt::Durability::Wal { sync: false };
    cfg.buffer_pool_size = 32;
    cfg.checkpoint.enabled = false;
    let tree = Tree::open(cfg).expect("holt open");
    (tree, dir)
}

fn make_db() -> (DB, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut cfg = TreeConfig::new(dir.path());
    cfg.durability = holt::Durability::Wal { sync: false };
    cfg.buffer_pool_size = 32;
    cfg.checkpoint.enabled = false;
    let db = DB::open(cfg).expect("holt DB open");
    db.create_tree("data").expect("create benchmark tree");
    db.checkpoint().expect("checkpoint benchmark setup");
    (db, dir)
}

fn journal_anchor(sequence: u64) -> JournalAnchor {
    let mut digest = [0u8; 32];
    digest[..8].copy_from_slice(&sequence.to_le_bytes());
    JournalAnchor::new(sequence, digest)
}

fn extrapolate_sample(elapsed: Duration, requested: u64, measured: u64) -> Duration {
    elapsed.mul_f64(requested as f64 / measured as f64)
}

fn preload(tree: &Tree, pairs: &[(Vec<u8>, Vec<u8>)]) {
    for (key, value) in pairs {
        tree.put(key, value).expect("holt preload");
    }
}

fn bench_persistent(c: &mut Criterion) {
    for (name, pairs) in [
        ("kv", gen_kv_dataset()),
        ("objstore", gen_objstore_dataset()),
        ("fs", gen_fs_dataset()),
    ] {
        bench_workload(c, name, &pairs);
    }
}

fn bench_workload(c: &mut Criterion, name: &str, pairs: &[(Vec<u8>, Vec<u8>)]) {
    let key_count = pairs.len();

    {
        let (tree, _dir) = make_tree();
        preload(&tree, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        let mut group = c.benchmark_group(format!("{name}_persist_get"));
        group.throughput(Throughput::Elements(1));
        group.bench_function("holt", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (key, _) = &pairs[idx];
                std::hint::black_box(tree.get(std::hint::black_box(key)).expect("holt get"));
            });
        });
        group.finish();
    }

    {
        let (tree, _dir) = make_tree();
        preload(&tree, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 2);
        let mut group = c.benchmark_group(format!("{name}_persist_put"));
        group.throughput(Throughput::Elements(1));
        group.bench_function("holt", |b| {
            b.iter(|| {
                let idx = (rng.next_u32() as usize) % key_count;
                let (key, value) = &pairs[idx];
                tree.put(std::hint::black_box(key), std::hint::black_box(value))
                    .expect("holt put");
            });
        });
        group.finish();
    }

    {
        let (tree, _dir) = make_tree();
        preload(&tree, pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 3);
        let mut group = c.benchmark_group(format!("{name}_persist_mixed"));
        group.throughput(Throughput::Elements(1));
        group.bench_function("holt", |b| {
            b.iter(|| {
                let r = rng.next_u32();
                let idx = (r as usize) % key_count;
                let (key, value) = &pairs[idx];
                if r & 1 == 0 {
                    std::hint::black_box(tree.get(std::hint::black_box(key)).expect("holt get"));
                } else {
                    tree.put(std::hint::black_box(key), std::hint::black_box(value))
                        .expect("holt put");
                }
            });
        });
        group.finish();
    }
}

fn bench_db_journal_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("db_journal_async");
    group.throughput(Throughput::Elements(1));

    {
        let (db, _dir) = make_db();
        let mut sequence = 0u64;
        group.bench_function("ordinary", |b| {
            b.iter_custom(|requested| {
                let measured = requested.clamp(1, JOURNAL_SAMPLE_COMMIT_CAP);
                let start = Instant::now();
                for _ in 0..measured {
                    sequence = sequence.wrapping_add(1);
                    let value = sequence.to_le_bytes();
                    db.atomic(|batch| batch.put("data", b"key", &value))
                        .expect("ordinary DB atomic");
                }
                let elapsed = start.elapsed();
                // Bound retained WAL bytes without charging checkpoint I/O to
                // the commit measurement returned to Criterion.
                db.checkpoint().expect("ordinary benchmark checkpoint");
                extrapolate_sample(elapsed, requested, measured)
            });
        });
    }

    for payload_len in [128usize, 4 * 1024] {
        let (db, _dir) = make_db();
        let genesis = journal_anchor(0);
        db.initialize_journal_stream(genesis)
            .expect("initialize benchmark stream");
        let mut previous = genesis;
        group.bench_with_input(
            BenchmarkId::new("attached", payload_len),
            &payload_len,
            |b, &payload_len| {
                b.iter_custom(|requested| {
                    let measured = requested.clamp(1, JOURNAL_SAMPLE_COMMIT_CAP);
                    let start = Instant::now();
                    for _ in 0..measured {
                        let current = journal_anchor(previous.sequence() + 1);
                        let mut payload = vec![0xA5; payload_len];
                        payload[..8].copy_from_slice(&current.sequence().to_le_bytes());
                        let value = current.sequence().to_le_bytes();
                        let envelope = JournalEnvelope::new(previous, current, payload)
                            .expect("valid benchmark envelope");
                        db.atomic_with_journal_envelope(envelope, |batch| {
                            batch.put("data", b"key", &value);
                        })
                        .expect("attached DB atomic");
                        previous = current;
                    }
                    let elapsed = start.elapsed();
                    db.checkpoint().expect("attached benchmark checkpoint");
                    extrapolate_sample(elapsed, requested, measured)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_persistent, bench_db_journal_paths);
criterion_main!(benches);
