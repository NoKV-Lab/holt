use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use holt::{PrefixRecord, Tree, TreeConfig, View};

const CHUNK_BYTES: usize = 33;
const VALUE: &[u8] = b"orbitkv-capsule-manifest";

struct Case {
    tree: Tree,
    view: View,
    query_chunks: usize,
    query: Vec<u8>,
    candidates: Vec<Vec<u8>>,
    expected: Option<Vec<u8>>,
}

fn capsule_key(chunk_count: usize, salt: u8) -> Vec<u8> {
    let mut key = b"orbitkv/capsule/v1\0tenant/model/plan".to_vec();
    key.reserve(chunk_count * CHUNK_BYTES);
    for chunk in 0..chunk_count {
        key.push(1);
        for byte in 0..32 {
            key.push(
                salt.wrapping_add(
                    u8::try_from((chunk * 37 + byte * 13) % 251)
                        .expect("modulo result fits in u8"),
                ),
            );
        }
    }
    key
}

fn build_case(query_chunks: usize, match_chunks: Option<usize>) -> Case {
    let tree = Tree::open(TreeConfig::memory()).expect("open benchmark tree");
    let query = capsule_key(query_chunks, 7);
    let candidates = (1..=query_chunks)
        .rev()
        .map(|chunks| capsule_key(chunks, 7))
        .collect::<Vec<_>>();
    let expected = match_chunks.map(|chunks| capsule_key(chunks, 7));
    if let Some(key) = &expected {
        tree.put(key, VALUE).expect("publish matching capsule");
    }

    for salt in 32..96 {
        tree.put(&capsule_key(query_chunks, salt), VALUE)
            .expect("publish distractor capsule");
    }
    let view = tree.snapshot(b"").expect("capture benchmark view").view().clone();
    Case {
        tree,
        view,
        query_chunks,
        query,
        candidates,
        expected,
    }
}

fn fallback_tree(tree: &Tree, candidates: &[Vec<u8>]) -> Option<PrefixRecord> {
    candidates.iter().find_map(|key| {
        tree.get_record(key)
            .expect("fallback point lookup")
            .map(|record| PrefixRecord {
                key: key.clone(),
                value: record.value,
                version: record.version,
            })
    })
}

fn fallback_view(view: &View, candidates: &[Vec<u8>]) -> Option<PrefixRecord> {
    candidates.iter().find_map(|key| {
        view.get_record(key)
            .expect("fallback point lookup")
            .map(|record| PrefixRecord {
                key: key.clone(),
                value: record.value,
                version: record.version,
            })
    })
}

fn assert_case(case: &Case) {
    if case.query_chunks == 256 {
        assert!(
            case.tree
                .stats()
                .expect("inspect benchmark tree")
                .blob_count
                >= 2,
            "the 256-chunk tier must exercise cross-blob lookup"
        );
    }
    let native = case
        .tree
        .longest_prefix_record(&case.query)
        .expect("native tree lookup");
    let fallback = fallback_tree(&case.tree, &case.candidates);
    assert_eq!(native, fallback);
    assert_eq!(
        native.as_ref().map(|record| &record.key),
        case.expected.as_ref()
    );

    let native_view = case
        .view
        .longest_prefix_record(&case.query)
        .expect("native view lookup");
    let fallback_view = fallback_view(&case.view, &case.candidates);
    assert_eq!(native_view, fallback_view);
}

fn benchmark_longest_prefix(c: &mut Criterion) {
    for query_chunks in [4, 16, 64, 256] {
        for (scenario, match_chunks) in [
            ("deep_hit", Some(query_chunks)),
            ("half_hit", Some((query_chunks / 2).max(1))),
            ("shallow_hit", Some(1)),
            ("miss", None),
        ] {
            let case = build_case(query_chunks, match_chunks);
            assert_case(&case);
            let mut group = c.benchmark_group(format!("lpm/{scenario}"));
            group.measurement_time(Duration::from_secs(3));
            group.sample_size(50);
            group.throughput(Throughput::Elements(1));

            group.bench_with_input(
                BenchmarkId::new("tree_native", query_chunks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.tree
                                .longest_prefix_record(black_box(&case.query))
                                .expect("native tree lookup"),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("tree_fallback", query_chunks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(fallback_tree(
                            black_box(&case.tree),
                            black_box(&case.candidates),
                        ))
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("view_native", query_chunks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.view
                                .longest_prefix_record(black_box(&case.query))
                                .expect("native view lookup"),
                        )
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("view_fallback", query_chunks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(fallback_view(
                            black_box(&case.view),
                            black_box(&case.candidates),
                        ))
                    });
                },
            );
            group.finish();
        }
    }
}

criterion_group!(benches, benchmark_longest_prefix);
criterion_main!(benches);
