use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use holt::{Tree, TreeConfig};

const CHUNK_BYTES: usize = 33;
const VALUE: &[u8] = b"orbitkv-capsule-manifest";

fn capsule_key(chunk_count: usize) -> Vec<u8> {
    let mut key = b"orbitkv/capsule/v1\0tenant/model/plan".to_vec();
    key.reserve(chunk_count * CHUNK_BYTES);
    for chunk in 0..chunk_count {
        key.push(1);
        for byte in 0..32 {
            key.push(
                u8::try_from((chunk * 37 + byte * 13) % 251)
                    .expect("modulo result fits in u8"),
            );
        }
    }
    key
}

fn fallback(tree: &Tree, candidates: &[Vec<u8>]) {
    let record = candidates.iter().find_map(|key| {
        tree.get_record(key)
            .expect("fallback point lookup")
            .map(|record| (key, record))
    });
    black_box(record);
}

fn measure(
    tree: &Tree,
    query: &[u8],
    candidates: &[Vec<u8>],
    threads: usize,
    native: bool,
    operations_per_thread: usize,
) -> f64 {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let handles = (0..threads)
        .map(|_| {
            let tree = tree.clone();
            let query = query.to_vec();
            let candidates = candidates.to_vec();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..operations_per_thread {
                    if native {
                        black_box(
                            tree.longest_prefix_record(black_box(&query))
                                .expect("native lookup"),
                        );
                    } else {
                        fallback(&tree, black_box(&candidates));
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    barrier.wait();
    let start = Instant::now();
    for handle in handles {
        handle.join().expect("benchmark worker");
    }
    let elapsed = start.elapsed();
    let total = threads * operations_per_thread;
    total as f64 / elapsed.as_secs_f64()
}

fn main() {
    let query_chunks = 64;
    let tree = Tree::open(TreeConfig::memory()).expect("open benchmark tree");
    let query = capsule_key(query_chunks);
    tree.put(&capsule_key(1), VALUE)
        .expect("publish shallow capsule");
    let candidates = (1..=query_chunks)
        .rev()
        .map(capsule_key)
        .collect::<Vec<_>>();

    for _ in 0..10_000 {
        black_box(
            tree.longest_prefix_record(&query)
                .expect("native warmup lookup"),
        );
    }

    let operations_per_thread = std::env::var("HOLT_LPM_OPS_PER_THREAD")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(50_000);
    println!(
        "shape=orbitkv chunks={query_chunks} scenario=shallow_hit ops_per_thread={operations_per_thread}"
    );
    println!("threads,native_ops_s,fallback_ops_s,speedup");
    for threads in [1, 2, 4, 8] {
        let native = measure(
            &tree,
            &query,
            &candidates,
            threads,
            true,
            operations_per_thread,
        );
        thread::sleep(Duration::from_millis(100));
        let fallback = measure(
            &tree,
            &query,
            &candidates,
            threads,
            false,
            operations_per_thread,
        );
        println!(
            "{threads},{native:.0},{fallback:.0},{:.2}",
            native / fallback
        );
    }
}
