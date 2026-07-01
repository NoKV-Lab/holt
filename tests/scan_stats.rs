//! Per-scan [`ScanStats`] — visited / returned / rollup / restarts.

use holt::{KeyRangeEntry, Tree, TreeConfig};
use tempfile::tempdir;

#[test]
fn stats_count_returned_and_visited() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..50u32 {
        tree.put(format!("d/{i:04}").as_bytes(), b"v").unwrap();
    }
    let mut iter = tree.scan(b"d/").into_iter();
    let mut n = 0;
    for entry in &mut iter {
        entry.unwrap();
        n += 1;
    }
    let stats = iter.stats();
    assert_eq!(n, 50);
    assert_eq!(stats.returned, 50);
    assert_eq!(stats.visited, 50); // each live leaf examined once, no skips
    assert_eq!(stats.rollup, 0);
    assert_eq!(stats.restarts, 0);
}

#[test]
fn stats_count_rollups_under_delimiter() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for d in 0..3u32 {
        for f in 0..10u32 {
            tree.put(format!("dir{d}/f{f}").as_bytes(), b"v").unwrap();
        }
    }
    // delimiter '/' folds each dirN/ subtree into one CommonPrefix.
    let mut iter = tree.range_keys().delimiter(b'/').into_iter();
    let mut rollups = 0;
    for entry in &mut iter {
        if let KeyRangeEntry::CommonPrefix(_) = entry.unwrap() {
            rollups += 1;
        }
    }
    let stats = iter.stats();
    assert_eq!(rollups, 3);
    assert_eq!(stats.rollup, 3);
    assert_eq!(stats.returned, 0); // every leaf folded away
                                   // The delimiter path must fold at the subtree boundary, not scan all
                                   // 30 leaves and deduplicate afterwards.
    assert_eq!(stats.visited, stats.rollup);
}

#[test]
fn visit_terminal_returns_stats() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..20u32 {
        tree.put(format!("k{i:04}").as_bytes(), b"v").unwrap();
    }
    let mut seen = 0;
    let stats = tree
        .scan_keys(b"k")
        .visit(100, |_| {
            seen += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, 20);
    assert_eq!(stats.returned, 20);
    assert_eq!(stats.visited, 20);
    assert_eq!(stats.restarts, 0);
}

#[test]
fn cache_hit_reports_zero_visited() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..8u32 {
        tree.put(format!("p/{i}").as_bytes(), b"v").unwrap();
    }
    // First visit walks the tree and populates the prefix-list cache.
    let first = tree.scan_keys(b"p/").visit(16, |_| Ok(())).unwrap();
    assert!(first.visited > 0);
    assert_eq!(first.returned, 8);
    // Second identical visit (no writes between) is served from cache —
    // same entries, but visited == 0 because nothing was walked.
    let second = tree.scan_keys(b"p/").visit(16, |_| Ok(())).unwrap();
    assert_eq!(second.returned, 8);
    assert_eq!(second.visited, 0);
    assert_eq!(second.restarts, 0);
}

#[test]
fn visit_with_outcome_reports_cache_hits() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..8u32 {
        tree.put(format!("hot/{i}").as_bytes(), b"v").unwrap();
    }

    let first = tree
        .scan_keys(b"hot/")
        .visit_with_outcome(16, |_| Ok(()))
        .unwrap();
    assert!(!first.cache_hit);
    assert_eq!(first.stats.returned, 8);

    let second = tree
        .scan_keys(b"hot/")
        .visit_with_outcome(16, |_| Ok(()))
        .unwrap();
    assert!(second.cache_hit);
    assert_eq!(second.stats.returned, 8);
    assert_eq!(second.stats.visited, 0);
}

#[test]
fn delimiter_list_dir_cache_serves_rollups_without_leaf_walk() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for d in ["a", "b", "c"] {
        for i in 0..8u32 {
            tree.put(format!("bucket/{d}/file-{i:04}").as_bytes(), b"v")
                .unwrap();
        }
    }

    let first = tree
        .scan_keys(b"bucket/")
        .delimiter(b'/')
        .visit_with_outcome(8, |_| Ok(()))
        .unwrap();
    assert!(!first.cache_hit);
    assert_eq!(first.stats.rollup, 3);

    let second = tree
        .scan_keys(b"bucket/")
        .delimiter(b'/')
        .visit_with_outcome(8, |_| Ok(()))
        .unwrap();
    assert!(second.cache_hit);
    assert_eq!(second.stats.rollup, 3);
    assert_eq!(second.stats.visited, 0);
}

#[test]
fn indexed_delimiter_rollup_uses_blobnode_summary_without_scan_pin() {
    let dir = tempdir().unwrap();
    let cfg = TreeConfig::new(dir.path());
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        let value = vec![7u8; 128];
        for d in 0..8u32 {
            for i in 0..800u32 {
                tree.put(format!("bucket/dir-{d}/file-{i:04}").as_bytes(), &value)
                    .unwrap();
            }
        }
        tree.checkpoint().unwrap();
    }

    let tree = Tree::open(cfg).unwrap();
    let mut rollups = 0usize;
    let mut prefixes = Vec::new();
    let outcome = tree
        .scan_keys(b"bucket/")
        .delimiter(b'/')
        .visit_with_outcome(8, |entry| {
            if let holt::KeyRangeEntryRef::CommonPrefix(prefix) = entry {
                rollups += 1;
                prefixes.push(prefix.to_vec());
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(rollups, 8, "prefixes={prefixes:?}");
    assert_eq!(outcome.stats.rollup, 8);
    let full_blob_reads = tree.stats().unwrap().bm_scan_full_blob_reads;
    assert!(
        full_blob_reads < rollups as u64,
        "cold delimiter rollup should use BlobNode/read-index liveness instead of pinning every child blob; full_blob_reads={full_blob_reads}, rollups={rollups}",
    );
}

#[test]
fn indexed_component_summary_rolls_child_blob_directories_with_one_routing_pin() {
    let dir = tempdir().unwrap();
    let cfg = TreeConfig::new(dir.path());
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        let value = vec![9u8; 128];
        for d in 0..8u32 {
            for i in 0..900u32 {
                tree.put(format!("bucket/a{d}/file-{i:04}").as_bytes(), &value)
                    .unwrap();
            }
        }
        tree.checkpoint().unwrap();
    }

    let tree = Tree::open(cfg).unwrap();
    let mut prefixes = Vec::new();
    let outcome = tree
        .scan_keys(b"bucket/")
        .delimiter(b'/')
        .visit_with_outcome(8, |entry| {
            if let holt::KeyRangeEntryRef::CommonPrefix(prefix) = entry {
                prefixes.push(prefix.to_vec());
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(outcome.stats.rollup, 8, "prefixes={prefixes:?}");
    let stats = tree.stats().unwrap();
    assert!(
        stats.bm_scan_full_blob_reads <= 1,
        "component summary should emit child directory rollups with at most one routing pin; full_blob_reads={}",
        stats.bm_scan_full_blob_reads,
    );
    assert!(
        stats.bm_read_index_loads > 0 || stats.bm_read_index_cache_hits > 0,
        "component summary should be served from read-index liveness"
    );
}

#[test]
fn colon_component_summary_rolls_child_blob_directories_with_one_routing_pin() {
    let dir = tempdir().unwrap();
    let cfg = TreeConfig::new(dir.path());
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        let value = vec![5u8; 128];
        for d in 0..8u32 {
            for i in 0..900u32 {
                tree.put(format!("tenant:dir-{d}:file-{i:04}").as_bytes(), &value)
                    .unwrap();
            }
        }
        tree.checkpoint().unwrap();
    }

    let tree = Tree::open(cfg).unwrap();
    let mut prefixes = Vec::new();
    let outcome = tree
        .scan_keys(b"tenant:")
        .delimiter(b':')
        .visit_with_outcome(8, |entry| {
            if let holt::KeyRangeEntryRef::CommonPrefix(prefix) = entry {
                prefixes.push(prefix.to_vec());
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(outcome.stats.rollup, 8, "prefixes={prefixes:?}");
    let stats = tree.stats().unwrap();
    assert!(
        stats.bm_scan_full_blob_reads <= 1,
        "colon component summary should avoid child leaf scans; full_blob_reads={}",
        stats.bm_scan_full_blob_reads,
    );
}

#[test]
fn prefix_empty_does_not_pollute_limit_one_prefix_cache() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"dir/child", b"v").unwrap();

    assert!(!tree.is_prefix_empty(b"dir/").unwrap());

    let outcome = tree
        .scan_keys(b"dir/")
        .visit_with_outcome(1, |_| Ok(()))
        .unwrap();
    assert!(!outcome.cache_hit);
    assert_eq!(outcome.stats.returned, 1);
    assert_eq!(outcome.stats.visited, 1);
}

#[test]
fn prefix_empty_uses_read_index_liveness_after_reopen() {
    let dir = tempdir().unwrap();
    let cfg = TreeConfig::new(dir.path());
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        let value = vec![3u8; 128];
        for d in 0..8u32 {
            for i in 0..900u32 {
                tree.put(format!("bucket/a{d}/file-{i:04}").as_bytes(), &value)
                    .unwrap();
            }
        }
        tree.checkpoint().unwrap();
    }

    let tree = Tree::open(cfg).unwrap();
    assert!(!tree.is_prefix_empty(b"bucket/a3/").unwrap());
    assert!(tree.is_prefix_empty(b"bucket/z/").unwrap());
    let stats = tree.stats().unwrap();
    assert!(
        stats.bm_scan_full_blob_reads <= 1,
        "prefix liveness should use read-index summaries before pinning child blobs; full_blob_reads={}",
        stats.bm_scan_full_blob_reads,
    );
    assert!(
        stats.bm_read_index_loads > 0 || stats.bm_read_index_cache_hits > 0,
        "prefix liveness should consult read-index summaries"
    );
}

#[test]
fn prefix_count_can_stop_after_limit() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..20u32 {
        tree.put(format!("dir/{i:04}").as_bytes(), b"v").unwrap();
    }

    let bounded = tree.prefix_count(b"dir/", 10).unwrap();
    assert_eq!(bounded.count, 10);
    assert!(!bounded.exact);
    assert_eq!(bounded.stats.returned, 11);

    let exact = tree.prefix_count(b"dir/", 0).unwrap();
    assert_eq!(exact.count, 20);
    assert!(exact.exact);
}

#[test]
fn view_prefix_count_reads_captured_state() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"dir/a", b"1").unwrap();
    tree.put(b"dir/b", b"2").unwrap();

    tree.view(b"dir/", |view| {
        tree.put(b"dir/c", b"3").unwrap();
        let count = view.prefix_count(b"dir/", 0).unwrap();
        assert_eq!(count.count, 2);
        assert!(count.exact);
        Ok(())
    })
    .unwrap();
}
