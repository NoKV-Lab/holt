//! Copy-on-write snapshot benchmark.
//!
//! Compares the owned [`Tree::snapshot`] and scoped [`Tree::view`] API paths,
//! which use the same copy-on-write capture, and measures the cost of holding
//! a snapshot:
//!
//! - `cow_create`   — owned snapshot vs scoped-view capture; each copies one
//!                    root frame and initially shares descendants.
//! - `cow_write`    — put throughput with no snapshot vs one held (the
//!                    fork-on-write + route-cache-disabled overhead).
//! - `cow_read`     — point read on the live tree vs on a snapshot.
//!
//! ```sh
//! cargo bench --manifest-path benches/Cargo.toml --bench cow_snapshot
//! ```

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use holt::{Tree, TreeConfig};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use tempfile::TempDir;

const SEED: u64 = 0x5A5A_C0DE_1234_5678;
const KEY_COUNT: usize = 20_000;
const VALUE_LEN: usize = 200;

/// Path-shaped keys + fixed-size bodies, sized so the tree spans many
/// blob frames and captured reads cross shared descendants.
fn dataset() -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..KEY_COUNT)
        .map(|i| {
            let key = format!("bucket-{:03}/path/file-{i:08}.bin", i / 1000).into_bytes();
            let value = vec![(i & 0xFF) as u8; VALUE_LEN];
            (key, value)
        })
        .collect()
}

fn make_tree() -> (Tree, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let mut cfg = TreeConfig::new(dir.path());
    cfg.durability = holt::Durability::Wal { sync: false };
    cfg.buffer_pool_size = 256;
    cfg.checkpoint.enabled = false;
    let tree = Tree::open(cfg).expect("holt open");
    (tree, dir)
}

fn preload(tree: &Tree, pairs: &[(Vec<u8>, Vec<u8>)]) {
    for (key, value) in pairs {
        tree.put(key, value).expect("preload");
    }
}

fn bench_create(c: &mut Criterion) {
    let pairs = dataset();
    let (tree, _dir) = make_tree();
    preload(&tree, &pairs);

    let mut group = c.benchmark_group("cow_create");
    group.throughput(Throughput::Elements(1));
    group.bench_function("snapshot", |b| {
        b.iter(|| {
            // Create + drop (retire). No writes between, so the only
            // copy is the root frame.
            let snap = tree.snapshot(b"").expect("snapshot");
            std::hint::black_box(&snap);
        });
    });
    group.bench_function("view_scoped", |b| {
        b.iter(|| {
            tree.view(b"", |v| {
                std::hint::black_box(v);
                Ok(())
            })
            .expect("view");
        });
    });
    group.finish();
}

fn bench_write(c: &mut Criterion) {
    let pairs = dataset();
    let key_count = pairs.len();

    let mut group = c.benchmark_group("cow_write");
    group.throughput(Throughput::Elements(1));

    {
        let (tree, _dir) = make_tree();
        preload(&tree, &pairs);
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        group.bench_function("put_no_snapshot", |b| {
            b.iter(|| {
                let (key, value) = &pairs[(rng.next_u32() as usize) % key_count];
                tree.put(std::hint::black_box(key), std::hint::black_box(value))
                    .expect("put");
            });
        });
    }

    {
        let (tree, _dir) = make_tree();
        preload(&tree, &pairs);
        // Held for the whole measurement: the first write to each frame
        // forks it (one-time), then writes are in-place — but the route
        // cache stays disabled, so every put does a full root descent.
        let _snap = tree.snapshot(b"").expect("snapshot");
        let mut rng = StdRng::seed_from_u64(SEED + 1);
        group.bench_function("put_snapshot_held", |b| {
            b.iter(|| {
                let (key, value) = &pairs[(rng.next_u32() as usize) % key_count];
                tree.put(std::hint::black_box(key), std::hint::black_box(value))
                    .expect("put");
            });
        });
    }

    group.finish();
}

fn bench_read(c: &mut Criterion) {
    let pairs = dataset();
    let key_count = pairs.len();
    let (tree, _dir) = make_tree();
    preload(&tree, &pairs);
    let snap = tree.snapshot(b"").expect("snapshot");

    let mut group = c.benchmark_group("cow_read");
    group.throughput(Throughput::Elements(1));

    let mut rng = StdRng::seed_from_u64(SEED + 2);
    group.bench_function("live_get", |b| {
        b.iter(|| {
            let (key, _) = &pairs[(rng.next_u32() as usize) % key_count];
            std::hint::black_box(tree.get(std::hint::black_box(key)).expect("get"));
        });
    });

    let mut rng = StdRng::seed_from_u64(SEED + 2);
    group.bench_function("snapshot_get", |b| {
        b.iter(|| {
            let (key, _) = &pairs[(rng.next_u32() as usize) % key_count];
            std::hint::black_box(snap.get(std::hint::black_box(key)).expect("get"));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_create, bench_write, bench_read);
criterion_main!(benches);
