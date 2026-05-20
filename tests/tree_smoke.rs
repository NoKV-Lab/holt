//! End-to-end smoke tests driving the public `Tree` API.
//!
//! Exercises only the public surface so signature breakage shows
//! up here first.

use std::sync::Arc;

use holt::{Backend, MemoryBackend, Tree, TreeBuilder, TreeConfig, TreeStats};

#[test]
fn open_memory_get_on_empty_tree_returns_none() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.get(b"anything").unwrap().is_none());
    assert!(tree.get(b"").unwrap().is_none());
}

#[test]
fn builder_memory_path() {
    let tree = TreeBuilder::new("scratch")
        .memory()
        .buffer_pool_size(32)
        .open()
        .unwrap();
    assert!(tree.get(b"x").unwrap().is_none());
}

#[test]
fn open_with_explicit_backend_round_trips_root_blob() {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let _t = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();
    let blobs_after_first = backend.list_blobs().unwrap().len();
    assert!(blobs_after_first >= 1, "root blob should be present");

    let _t2 = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();
    assert_eq!(
        backend.list_blobs().unwrap().len(),
        blobs_after_first,
        "re-open must not allocate a fresh root"
    );
}

#[test]
fn checkpoint_is_idempotent_on_memory_backend() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.checkpoint().unwrap();
    tree.checkpoint().unwrap();
    assert!(tree.get(b"k").unwrap().is_none());
}

// ----------------------------------------------------------------
// Put / Get
// ----------------------------------------------------------------

#[test]
fn put_then_get_round_trip() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.insert(b"hello", b"world").unwrap().is_none());
    assert_eq!(tree.get(b"hello").unwrap().as_deref(), Some(&b"world"[..]));
    assert!(tree.get(b"missing").unwrap().is_none());
}

#[test]
fn put_returns_previous_value_on_update() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.insert(b"k", b"v1").unwrap().is_none());
    assert_eq!(
        tree.insert(b"k", b"v2").unwrap().as_deref(),
        Some(&b"v1"[..])
    );
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v2"[..]));
}

#[test]
fn many_keys_all_readable_via_public_api() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..100u32)
        .map(|i| {
            (
                format!("img/{i:04}.jpg").into_bytes(),
                format!("blob#{i}").into_bytes(),
            )
        })
        .collect();
    for (k, v) in &pairs {
        tree.put(k, v).unwrap();
    }
    for (k, v) in &pairs {
        assert_eq!(tree.get(k).unwrap().as_deref(), Some(&v[..]));
    }
}

#[test]
fn concurrent_writers_serialised_by_internal_lock() {
    use std::thread;

    let tree = Arc::new(Tree::open(TreeConfig::memory()).unwrap());
    let handles: Vec<_> = (0..8u8)
        .map(|t| {
            let tree = tree.clone();
            thread::spawn(move || {
                for i in 0..25u32 {
                    let k = format!("t{t}/k{i:03}").into_bytes();
                    let v = format!("v{t}-{i}").into_bytes();
                    tree.put(&k, &v).unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    for t in 0..8u8 {
        for i in 0..25u32 {
            let k = format!("t{t}/k{i:03}").into_bytes();
            let v = format!("v{t}-{i}").into_bytes();
            assert_eq!(tree.get(&k).unwrap().as_deref(), Some(&v[..]));
        }
    }
}

#[test]
fn strict_prefix_key_pair_now_works() {
    // "abc" and "abcdef" — one is a strict prefix of the other.
    // Resolved at the Tree layer via the terminator-byte trick.
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"abc", b"v1").unwrap();
    tree.put(b"abcdef", b"v2").unwrap();
    assert_eq!(tree.get(b"abc").unwrap().as_deref(), Some(&b"v1"[..]));
    assert_eq!(tree.get(b"abcdef").unwrap().as_deref(), Some(&b"v2"[..]));
    // Other length within the chain stays NotFound.
    assert!(tree.get(b"abcd").unwrap().is_none());
}

#[test]
fn deeply_nested_strict_prefix_chain() {
    // The classic "filesystem path" workload: each level of the
    // path is a key.
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let paths: &[&[u8]] = &[b"/", b"/a", b"/a/b", b"/a/b/c", b"/a/b/c/d", b"/a/b/c/d/e"];
    for (i, p) in paths.iter().enumerate() {
        tree.put(p, format!("level{i}").as_bytes()).unwrap();
    }
    for (i, p) in paths.iter().enumerate() {
        assert_eq!(
            tree.get(p).unwrap().as_deref(),
            Some(format!("level{i}").as_bytes()),
        );
    }
    // Holes in the chain stay NotFound.
    assert!(tree.get(b"/a/b/c/d/e/f").unwrap().is_none());
}

#[test]
fn empty_key_round_trips() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"", b"empty-key-value").unwrap();
    assert_eq!(
        tree.get(b"").unwrap().as_deref(),
        Some(&b"empty-key-value"[..])
    );
    tree.put(b"a", b"other").unwrap();
    assert_eq!(
        tree.get(b"").unwrap().as_deref(),
        Some(&b"empty-key-value"[..])
    );
    assert_eq!(tree.get(b"a").unwrap().as_deref(), Some(&b"other"[..]));
}

// ----------------------------------------------------------------
// Delete (Stage 2c)
// ----------------------------------------------------------------

#[test]
fn delete_existing_key_returns_value_and_removes_it() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v").unwrap();
    assert_eq!(tree.remove(b"k").unwrap().as_deref(), Some(&b"v"[..]));
    assert!(tree.get(b"k").unwrap().is_none());
}

#[test]
fn delete_missing_key_is_noop() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.remove(b"missing").unwrap().is_none());
}

#[test]
fn delete_then_reinsert_round_trips() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v1").unwrap();
    assert_eq!(tree.remove(b"k").unwrap().as_deref(), Some(&b"v1"[..]));
    tree.put(b"k", b"v2").unwrap();
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v2"[..]));
}

#[test]
fn delete_all_keys_then_reinsert_works() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..50u32)
        .map(|i| {
            (
                format!("img/{i:03}").into_bytes(),
                format!("v{i}").into_bytes(),
            )
        })
        .collect();
    for (k, v) in &pairs {
        tree.put(k, v).unwrap();
    }
    for (k, v) in &pairs {
        assert_eq!(tree.remove(k).unwrap().as_deref(), Some(&v[..]));
    }
    for (k, _) in &pairs {
        assert!(tree.get(k).unwrap().is_none());
    }
    tree.put(b"fresh", b"V").unwrap();
    assert_eq!(tree.get(b"fresh").unwrap().as_deref(), Some(&b"V"[..]));
}

#[test]
fn delete_keeps_siblings_under_shared_prefix() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"img/01.jpg", b"a").unwrap();
    tree.put(b"img/02.jpg", b"b").unwrap();
    tree.put(b"img/03.jpg", b"c").unwrap();
    assert_eq!(
        tree.remove(b"img/02.jpg").unwrap().as_deref(),
        Some(&b"b"[..])
    );
    assert_eq!(tree.get(b"img/01.jpg").unwrap().as_deref(), Some(&b"a"[..]));
    assert!(tree.get(b"img/02.jpg").unwrap().is_none());
    assert_eq!(tree.get(b"img/03.jpg").unwrap().as_deref(), Some(&b"c"[..]));
}

// ----------------------------------------------------------------
// Rename (Stage 2c)
// ----------------------------------------------------------------

#[test]
fn rename_moves_value_to_new_key() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"old", b"v").unwrap();
    tree.rename(b"old", b"new", false).unwrap();
    assert!(tree.get(b"old").unwrap().is_none());
    assert_eq!(tree.get(b"new").unwrap().as_deref(), Some(&b"v"[..]));
}

#[test]
fn rename_missing_src_errors_not_found() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let r = tree.rename(b"nope", b"new", false);
    assert!(matches!(r, Err(holt::Error::NotFound)));
}

#[test]
fn rename_to_existing_dst_without_force_errors_dst_exists() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"a", b"v_a").unwrap();
    tree.put(b"b", b"v_b").unwrap();
    let r = tree.rename(b"a", b"b", false);
    assert!(matches!(r, Err(holt::Error::DstExists)));
    // Both keys still present, values unchanged.
    assert_eq!(tree.get(b"a").unwrap().as_deref(), Some(&b"v_a"[..]));
    assert_eq!(tree.get(b"b").unwrap().as_deref(), Some(&b"v_b"[..]));
}

#[test]
fn rename_force_overwrites_existing_dst() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"a", b"v_a").unwrap();
    tree.put(b"b", b"v_b").unwrap();
    tree.rename(b"a", b"b", true).unwrap();
    assert!(tree.get(b"a").unwrap().is_none());
    assert_eq!(tree.get(b"b").unwrap().as_deref(), Some(&b"v_a"[..]));
}

#[test]
fn rename_same_key_is_noop() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v").unwrap();
    tree.rename(b"k", b"k", false).unwrap();
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
}

#[test]
fn rename_through_shared_prefix() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"img/01.jpg", b"a").unwrap();
    tree.put(b"img/02.jpg", b"b").unwrap();
    tree.put(b"img/03.jpg", b"c").unwrap();
    tree.rename(b"img/02.jpg", b"img/02-renamed.jpg", false)
        .unwrap();
    assert_eq!(tree.get(b"img/01.jpg").unwrap().as_deref(), Some(&b"a"[..]));
    assert!(tree.get(b"img/02.jpg").unwrap().is_none());
    assert_eq!(
        tree.get(b"img/02-renamed.jpg").unwrap().as_deref(),
        Some(&b"b"[..]),
    );
    assert_eq!(tree.get(b"img/03.jpg").unwrap().as_deref(), Some(&b"c"[..]));
}

// ----------------------------------------------------------------
// Stage 2d phase A — multi-blob lookup
//
// The spillover trigger that creates multi-blob state automatically
// lands in phase B. For now we hand-construct a 2-blob layout via
// the engine's `make_blob_from_node` primitive + a directly-installed
// `BlobNode`, then verify `Tree::get` follows the crossing.
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// Stage 2d phase B — automatic multi-blob spillover
// ----------------------------------------------------------------

#[test]
fn auto_spillover_creates_child_blob_when_root_blob_fills() {
    // Insert enough data to overflow the root blob (~448 KB usable
    // data area). Walker `insert_multi` auto-triggers `splitBlob`
    // when any alloc hits `OutOfSpace`, migrating the largest non-
    // BlobNode subtree of the current frame to a fresh child blob,
    // then retries.
    //
    // **Workload note:** until Stage 6's `compactBlob` lands, leaf
    // extents leak after every same-size update; the bump cursor
    // is monotonic. So spillover only buys "slot table" room, not
    // bump-area room — once the root blob has many subtrees
    // migrated out, *every* subsequent insert routes through a
    // BlobNode into a child blob and the root blob stays at its
    // high-water-mark bump cursor.
    //
    // We pick a workload size that triggers at least one spillover
    // but doesn't push past the `MAX_SPILLOVER_ATTEMPTS` per-call
    // budget. ~2000 keys × ~250 B per leaf = ~500 KB → ~50 KB has
    // to spill out via splitBlob.
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();

    const N: u32 = 2000;
    let value = vec![0xAB; 200];
    for i in 0..N {
        let k = format!("k{i:08}").into_bytes();
        tree.put(&k, &value).unwrap();
    }

    // All keys readable through the multi-blob tree.
    for i in 0..N {
        let k = format!("k{i:08}").into_bytes();
        assert_eq!(
            tree.get(&k).unwrap().as_deref(),
            Some(&value[..]),
            "post-spillover lookup failed at key {k:?}; backend has {} blob(s)",
            backend.list_blobs().unwrap().len(),
        );
    }

    // Spillover should have created at least one child blob.
    let blobs = backend.list_blobs().unwrap();
    assert!(
        blobs.len() >= 2,
        "expected auto-spillover to allocate at least 1 child blob, got {} total blob(s)",
        blobs.len(),
    );
}

#[test]
fn concurrent_reads_across_multi_blob_tree_via_buffer_manager() {
    // Builds a multi-blob tree (forces auto-spillover), then
    // hammers `Tree::get` from N threads concurrently. The
    // BufferManager between Tree and the inner backend keeps the
    // child blobs cached after the first read; per-blob locks
    // mean concurrent reads on *different* blobs don't fight for
    // a single mutex.
    use std::thread;

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let tree = Arc::new(
        TreeBuilder::new("ignored")
            .open_with_backend(backend.clone())
            .unwrap(),
    );

    const N: u32 = 2000;
    let value = vec![0x66; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &value).unwrap();
    }
    // Multi-blob pre-cond.
    assert!(
        backend.list_blobs().unwrap().len() >= 2,
        "test pre-cond: {} keys × 200 B values should overflow into multiple blobs",
        N,
    );

    // 8 threads × 250 random gets each.
    let handles: Vec<_> = (0..8u32)
        .map(|t| {
            let tree = tree.clone();
            let value = value.clone();
            thread::spawn(move || {
                for r in 0..250u32 {
                    let i = (t * 250 + r * 7) % N;
                    let k = format!("k{i:08}").into_bytes();
                    let got = tree.get(&k).unwrap();
                    assert_eq!(got.as_deref(), Some(&value[..]));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn compact_then_insert_reclaims_extent_leak() {
    // Pure-mutation workload: insert N keys, delete half, insert
    // another N. Without compact reclaiming the deleted-leaf
    // extents, the bump cursor would push past blob capacity and
    // force spillover much sooner. With compact wired into the
    // OOM recovery path, the extent leak is recoverable and the
    // workload stays in many fewer blobs (often the single root).
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();

    let val = vec![0x42; 200];

    for i in 0..1500u32 {
        tree.put(format!("k{i:08}").as_bytes(), &val).unwrap();
    }
    // Delete the lower half — leaves ~750 keys live + ~750
    // leaked extents.
    for i in 0..750u32 {
        let prev = tree.remove(format!("k{i:08}").as_bytes()).unwrap();
        assert_eq!(prev.as_deref(), Some(&val[..]));
    }
    // Now insert another 1500 — would not fit without compact
    // reclaiming the deleted extents.
    for i in 1500..3000u32 {
        tree.put(format!("k{i:08}").as_bytes(), &val).unwrap();
    }

    // Spot-check a few keys per cohort.
    assert!(tree.get(b"k00000000").unwrap().is_none()); // deleted
    assert!(tree.get(b"k00000749").unwrap().is_none()); // deleted
    assert_eq!(tree.get(b"k00000750").unwrap().as_deref(), Some(&val[..])); // kept
    assert_eq!(tree.get(b"k00001499").unwrap().as_deref(), Some(&val[..])); // kept
    assert_eq!(tree.get(b"k00001500").unwrap().as_deref(), Some(&val[..])); // new
    assert_eq!(tree.get(b"k00002999").unwrap().as_deref(), Some(&val[..])); // new
}

#[test]
fn multi_blob_delete_round_trip() {
    // Insert past one-blob capacity → auto-spillover creates
    // child blobs. Delete a key that lives in a child blob;
    // verify it disappears from the tree (including across
    // crossings). Also verify the rest of the keys still resolve.
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();

    const N: u32 = 2000;
    let value = vec![0x42; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &value).unwrap();
    }
    assert!(
        backend.list_blobs().unwrap().len() >= 2,
        "test pre-cond: expected multi-blob state",
    );

    // Delete every 5th key.
    let mut deleted = 0u32;
    for i in 0..N {
        if i % 5 != 0 {
            continue;
        }
        let k = format!("k{i:08}").into_bytes();
        let prev = tree.remove(&k).unwrap();
        assert_eq!(
            prev.as_deref(),
            Some(&value[..]),
            "delete returned wrong prev for {k:?}"
        );
        deleted += 1;
    }

    // Survivors readable, deletions gone.
    for i in 0..N {
        let k = format!("k{i:08}").into_bytes();
        let got = tree.get(&k).unwrap();
        if i % 5 == 0 {
            assert!(got.is_none(), "deleted key {k:?} still present");
        } else {
            assert_eq!(
                got.as_deref(),
                Some(&value[..]),
                "survivor key {k:?} missing"
            );
        }
    }
    let _ = deleted;
}

#[test]
fn multi_blob_rename_round_trip() {
    // Builds a multi-blob tree, then exercises rename:
    //   1. force-overwrite onto an existing dst (no new leaf
    //      allocated, so spillover never re-triggers)
    //   2. DstExists guard with force=false
    //
    // Renaming to a brand-new key in the multi-blob state can
    // cascade further spillovers and stress the
    // MAX_SPILLOVER_ATTEMPTS budget — that case is gated on
    // Stage 6 compactBlob (which reclaims the extent leak that
    // makes the budget tight). Within-existing-keys renames are
    // the realistic metadata workload anyway (move foo/bar → foo/baz
    // where both directory entries exist).
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();

    const N: u32 = 2000;
    let value_a = vec![0x99; 200];
    let value_b = vec![0xAA; 200];
    for i in 0..N {
        // Half the keys get value_a, half get value_b — so the
        // force-overwrite assertion below has something to check.
        let v = if i % 2 == 0 { &value_a } else { &value_b };
        tree.put(format!("k{i:08}").as_bytes(), v).unwrap();
    }
    assert!(
        backend.list_blobs().unwrap().len() >= 2,
        "test pre-cond: expected multi-blob state",
    );

    // force-overwrite an existing dst. value_a's "k00000006" wins
    // over value_b's "k00000007".
    let src = format!("k{:08}", 6).into_bytes();
    let dst = format!("k{:08}", 7).into_bytes();
    tree.rename(&src, &dst, /*force=*/ true).unwrap();
    assert!(
        tree.get(&src).unwrap().is_none(),
        "src should be gone post-rename"
    );
    assert_eq!(
        tree.get(&dst).unwrap().as_deref(),
        Some(&value_a[..]),
        "dst should now carry src's old value",
    );

    // force=false on a still-occupied dst → DstExists.
    let live_src = format!("k{:08}", 100).into_bytes();
    let occupied_dst = format!("k{:08}", 101).into_bytes();
    let r = tree.rename(&live_src, &occupied_dst, /*force=*/ false);
    assert!(
        matches!(r, Err(holt::Error::DstExists)),
        "force=false to occupied dst must be DstExists",
    );

    // Unaffected keys still resolve.
    let untouched = format!("k{:08}", 500).into_bytes();
    assert!(tree.get(&untouched).unwrap().is_some());
}

#[test]
fn auto_spillover_preserves_data_across_reopen() {
    // After auto-spillover the tree is multi-blob. Closing and
    // reopening the same backend must surface the same key/value
    // mapping (root blob + all spilled child blobs are persisted).
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    {
        let tree = TreeBuilder::new("ignored")
            .open_with_backend(backend.clone())
            .unwrap();
        for i in 0..2000u32 {
            tree.put(format!("k{i:08}").as_bytes(), &[0xCD; 192])
                .unwrap();
        }
        tree.checkpoint().unwrap();
    }

    let tree = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();
    for i in 0..2000u32 {
        let k = format!("k{i:08}").into_bytes();
        let v = tree.get(&k).unwrap();
        assert!(
            v.is_some(),
            "post-reopen lookup failed at key {k:?}; backend has {} blob(s)",
            backend.list_blobs().unwrap().len(),
        );
    }
}

#[test]
fn optimistic_readers_dont_block_writers() {
    // Stage 6 phase 2b: BufferManager's per-blob HybridLatch lets
    // readers walk in optimistic mode (snapshot version → read →
    // validate → restart on torn) while writers take exclusive.
    // This test pounds the same tree from N readers + a writer
    // concurrently and verifies (1) readers never see stale data
    // that doesn't match the stable initial seed, and (2) the
    // writer's monotonic counter ends at the expected value.
    use std::sync::atomic::{AtomicU64, Ordering as AOrdering};
    use std::thread;

    let tree = Arc::new(Tree::open(TreeConfig::memory()).unwrap());
    // Seed 100 stable keys that the readers will hammer.
    let stable_value = b"stable-value".to_vec();
    for i in 0..100u32 {
        let k = format!("stable/{i:04}").into_bytes();
        tree.put(&k, &stable_value).unwrap();
    }

    let wrong = Arc::new(AtomicU64::new(0));

    // 4 reader threads × 500 random gets on the stable keys.
    let reader_handles: Vec<_> = (0..4u32)
        .map(|t| {
            let tree = tree.clone();
            let stable_value = stable_value.clone();
            let wrong = wrong.clone();
            thread::spawn(move || {
                for r in 0..500u32 {
                    let i = (t * 500 + r * 13) % 100;
                    let k = format!("stable/{i:04}").into_bytes();
                    match tree.get(&k).unwrap() {
                        Some(v) if v == stable_value => {}
                        other => {
                            wrong.fetch_add(1, AOrdering::Relaxed);
                            panic!("reader saw torn / wrong value for {k:?}: {other:?}");
                        }
                    }
                }
            })
        })
        .collect();

    // 1 writer thread churns 200 keys disjoint from the stable ones.
    let writer_handle = {
        let tree = tree.clone();
        thread::spawn(move || {
            for i in 0..200u32 {
                let k = format!("churn/{i:04}").into_bytes();
                let v = format!("v-{i}").into_bytes();
                tree.put(&k, &v).unwrap();
            }
        })
    };

    for h in reader_handles {
        h.join().unwrap();
    }
    writer_handle.join().unwrap();

    assert_eq!(wrong.load(AOrdering::Relaxed), 0);
    // All churn writes landed.
    for i in 0..200u32 {
        let k = format!("churn/{i:04}").into_bytes();
        let v = format!("v-{i}").into_bytes();
        assert_eq!(tree.get(&k).unwrap().as_deref(), Some(&v[..]));
    }
    // Stable keys unchanged.
    for i in 0..100u32 {
        let k = format!("stable/{i:04}").into_bytes();
        assert_eq!(tree.get(&k).unwrap().as_deref(), Some(&stable_value[..]));
    }
}

// ----------------------------------------------------------------
// Stats + Compact
// ----------------------------------------------------------------

#[test]
fn stats_on_fresh_tree_reports_root_blob_only() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let s = tree.stats().unwrap();
    assert_eq!(s.blob_count, 1, "fresh tree has exactly the root blob");
    assert_eq!(s.blobs.len(), 1);
    assert_eq!(s.total_compactions, 0);
    assert_eq!(s.total_tombstones, 0);
    // A freshly-initialised blob holds the EmptyRoot sentinel, so
    // it has consumed some bump-area bytes and at least one slot.
    assert!(s.total_space_used > 0);
    assert!(s.total_slots >= 1);
}

#[test]
fn stats_reflects_inserts() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let before = tree.stats().unwrap();
    for i in 0..16u32 {
        tree.put(format!("k{i:04}").as_bytes(), b"value-bytes-here")
            .unwrap();
    }
    let after = tree.stats().unwrap();
    assert!(after.total_space_used > before.total_space_used);
    assert!(after.total_slots > before.total_slots);
    assert_eq!(after.blob_count, 1, "16 small keys still fit in one blob");
}

#[test]
fn compact_on_empty_tree_bumps_compact_times() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let before = tree.stats().unwrap();
    assert_eq!(before.total_compactions, 0);
    tree.compact().unwrap();
    let after = tree.stats().unwrap();
    assert_eq!(
        after.total_compactions, 1,
        "one blob compacted once → compact_times = 1"
    );
}

#[test]
fn compact_after_writes_preserves_data_and_bumps_count() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..32u32 {
        tree.put(format!("k{i:04}").as_bytes(), format!("v-{i}").as_bytes())
            .unwrap();
    }
    let pre_space = tree.stats().unwrap().total_space_used;
    tree.compact().unwrap();
    let post = tree.stats().unwrap();
    assert_eq!(post.total_compactions, 1);
    // Re-read every key — compact must not lose data.
    for i in 0..32u32 {
        let got = tree.get(format!("k{i:04}").as_bytes()).unwrap();
        assert_eq!(got.as_deref(), Some(format!("v-{i}").as_bytes()));
    }
    // No tombstones yet (no deletes), so compact should not have
    // grown the blob. Allow equality (no churn to reclaim).
    assert!(post.total_space_used <= pre_space + 4096);
}

#[test]
fn erase_tombstones_leaf_without_freeing_and_bumps_counter() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"alpha", b"A").unwrap();
    let before = tree.stats().unwrap();
    assert_eq!(before.total_tombstones, 0);
    tree.delete(b"alpha").unwrap();
    let after = tree.stats().unwrap();
    assert_eq!(after.total_tombstones, 1, "delete should mark a tombstone");
    // The blob still holds the tombstoned slot — space_used has
    // **not** shrunk (no immediate reclaim).
    assert_eq!(after.total_space_used, before.total_space_used);
    // Lookup returns None because the read path skips tombstones.
    assert!(tree.get(b"alpha").unwrap().is_none());
}

#[test]
fn reinsert_at_tombstoned_key_resurrects_and_decrements_counter() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v1").unwrap();
    tree.delete(b"k").unwrap();
    assert_eq!(tree.stats().unwrap().total_tombstones, 1);
    // Re-insert at the same key should resurrect the leaf in place:
    // the prior value isn't visible (it was deleted), so `put`
    // returns None as the previous value, and the tombstone counter
    // drops back to zero.
    let prev = tree.insert(b"k", b"v2").unwrap();
    assert_eq!(prev, None, "resurrected leaf has no observable prior value");
    let stats = tree.stats().unwrap();
    assert_eq!(stats.total_tombstones, 0);
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v2"[..]));
}

#[test]
fn compact_drops_tombstoned_leaves_and_resets_counter() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    // 16 keys, delete half. compact should drop them.
    for i in 0..16u32 {
        tree.put(format!("k{i:04}").as_bytes(), b"value-bytes")
            .unwrap();
    }
    for i in 0..8u32 {
        tree.delete(format!("k{i:04}").as_bytes()).unwrap();
    }
    let before = tree.stats().unwrap();
    assert_eq!(before.total_tombstones, 8);
    let bytes_before_compact = before.total_space_used;

    tree.compact().unwrap();

    let after = tree.stats().unwrap();
    assert_eq!(
        after.total_tombstones, 0,
        "compact must reset tombstone_leaf_cnt"
    );
    assert!(
        after.total_space_used < bytes_before_compact,
        "compact must reclaim bytes from the 8 dropped leaves: \
         before={bytes_before_compact}, after={}",
        after.total_space_used
    );
    assert_eq!(after.total_compactions, 1);
    // Survivors still readable.
    for i in 8..16u32 {
        let v = tree.get(format!("k{i:04}").as_bytes()).unwrap();
        assert_eq!(v.as_deref(), Some(&b"value-bytes"[..]));
    }
    // Dropped keys still gone (their tombstones were swept).
    for i in 0..8u32 {
        assert!(tree.get(format!("k{i:04}").as_bytes()).unwrap().is_none());
    }
}

#[test]
fn compact_collapses_all_tombstoned_tree_to_empty_root() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..8u32 {
        tree.put(format!("k{i}").as_bytes(), b"v").unwrap();
    }
    for i in 0..8u32 {
        tree.delete(format!("k{i}").as_bytes()).unwrap();
    }
    let pre = tree.stats().unwrap();
    assert_eq!(pre.total_tombstones, 8);

    tree.compact().unwrap();

    let post = tree.stats().unwrap();
    assert_eq!(post.total_tombstones, 0);
    // No live keys anywhere.
    for i in 0..8u32 {
        assert!(tree.get(format!("k{i}").as_bytes()).unwrap().is_none());
    }
    // Subsequent puts work on the post-compact EmptyRoot.
    tree.put(b"fresh", b"data").unwrap();
    assert_eq!(tree.get(b"fresh").unwrap().as_deref(), Some(&b"data"[..]));
}

#[test]
fn compact_merges_shrunk_child_blob_back_into_parent() {
    // Force a spillover with large values, then erase most of the
    // keys. Compact phase 1 drops the tombstones (child blob's
    // space_used drops); phase 2 sees the now-small child and
    // folds it back into the parent, dropping blob_count to 1.
    let tree = TreeBuilder::new("ignored")
        .memory()
        .buffer_pool_size(16)
        .open()
        .unwrap();
    let big = vec![0xCDu8; 4 * 1024];
    for i in 0..256u32 {
        tree.put(format!("k{i:08}").as_bytes(), &big).unwrap();
    }
    let before = tree.stats().unwrap();
    assert!(
        before.blob_count >= 2,
        "the workload must spill across blobs first (got {} blobs)",
        before.blob_count,
    );

    // Drop most of the keys — only a handful of survivors stay.
    for i in 0..248u32 {
        tree.delete(format!("k{i:08}").as_bytes()).unwrap();
    }

    tree.compact().unwrap();

    let after = tree.stats().unwrap();
    assert_eq!(
        after.blob_count, 1,
        "compact must merge the shrunk child blobs back into the root, got blobs={}",
        after.blob_count,
    );
    assert_eq!(after.total_tombstones, 0);
    // Each surviving key still readable through the public API.
    for i in 248..256u32 {
        let v = tree.get(format!("k{i:08}").as_bytes()).unwrap();
        assert_eq!(v.as_deref(), Some(&big[..]));
    }
    // The dropped keys stay gone.
    for i in 0..248u32 {
        assert!(tree.get(format!("k{i:08}").as_bytes()).unwrap().is_none());
    }
}

#[test]
fn compact_skips_merge_when_child_blob_still_large() {
    // Same shape but stop the erases before the child shrinks
    // enough — combined parent + child should still exceed the
    // page-size threshold, so phase 2 leaves the BlobNode crossing
    // alone.
    let tree = TreeBuilder::new("ignored")
        .memory()
        .buffer_pool_size(16)
        .open()
        .unwrap();
    let big = vec![0xEFu8; 4 * 1024];
    for i in 0..256u32 {
        tree.put(format!("k{i:08}").as_bytes(), &big).unwrap();
    }
    let multi = tree.stats().unwrap();
    assert!(multi.blob_count >= 2);

    // Compact without any erases — child blobs are still near-full,
    // so merges should not fire.
    tree.compact().unwrap();

    let after = tree.stats().unwrap();
    assert_eq!(
        after.blob_count, multi.blob_count,
        "compact must not merge children that are still too large"
    );
    // All 256 keys still present.
    for i in 0..256u32 {
        let v = tree.get(format!("k{i:08}").as_bytes()).unwrap();
        assert_eq!(v.as_deref(), Some(&big[..]));
    }
}

#[test]
fn stats_aggregates_across_multi_blob_tree() {
    // Force the tree across blob boundaries with large values.
    let tree = TreeBuilder::new("ignored")
        .memory()
        .buffer_pool_size(8)
        .open()
        .unwrap();
    let big = vec![0xABu8; 4 * 1024];
    for i in 0..256u32 {
        tree.put(format!("k{i:08}").as_bytes(), &big).unwrap();
    }
    let s: TreeStats = tree.stats().unwrap();
    assert!(
        s.blob_count >= 2,
        "256×4 KB values must spill into multiple blobs (got {} blobs)",
        s.blob_count
    );
    // Aggregate is just the sum of per-blob counters.
    let sum_space: u64 = s.blobs.iter().map(|b| u64::from(b.space_used)).sum();
    assert_eq!(sum_space, s.total_space_used);
    let sum_slots: u64 = s.blobs.iter().map(|b| u64::from(b.num_slots)).sum();
    assert_eq!(sum_slots, s.total_slots);
}

#[test]
fn txn_applies_buffered_ops_in_order() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"seed", b"S").unwrap();

    tree.txn(|batch| {
        batch.put(b"a", b"1");
        batch.put(b"b", b"2");
        batch.delete(b"seed");
        batch.rename(b"a", b"aa", false);
    })
    .unwrap();

    assert!(tree.get(b"seed").unwrap().is_none());
    assert!(tree.get(b"a").unwrap().is_none());
    assert_eq!(tree.get(b"aa").unwrap().as_deref(), Some(&b"1"[..]));
    assert_eq!(tree.get(b"b").unwrap().as_deref(), Some(&b"2"[..]));
}

#[test]
fn txn_empty_batch_is_a_noop() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"unchanged", b"X").unwrap();

    tree.txn(|_batch| {}).unwrap();

    assert_eq!(tree.get(b"unchanged").unwrap().as_deref(), Some(&b"X"[..]));
}

#[test]
fn txn_returns_error_when_rename_src_missing_and_leaves_prior_ops_applied() {
    // Contract per Tree::txn doc: mid-batch failure leaves the
    // already-applied ops in the BM cache (they're not rolled
    // back), and the WAL record is NOT written — so a future
    // reopen-via-replay would lose the whole batch.
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let err = tree
        .txn(|batch| {
            batch.put(b"committed", b"C");
            batch.rename(b"missing-src", b"dst", false);
            batch.put(b"never-reached", b"N");
        })
        .unwrap_err();
    assert!(
        matches!(err, holt::Error::NotFound),
        "expected NotFound from rename of missing src, got {err:?}",
    );
    // The pre-failure put landed in the BM cache.
    assert_eq!(tree.get(b"committed").unwrap().as_deref(), Some(&b"C"[..]),);
    // The post-failure put never ran.
    assert!(tree.get(b"never-reached").unwrap().is_none());
}

// ----------------------------------------------------------------
// Tree::range
// ----------------------------------------------------------------

use holt::RangeEntry;

fn collect_keys(iter: impl IntoIterator<Item = Result<RangeEntry, holt::Error>>) -> Vec<Vec<u8>> {
    iter.into_iter()
        .map(|r| match r.unwrap() {
            RangeEntry::Key { key, .. } => key,
            RangeEntry::CommonPrefix(p) => p,
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect()
}

#[test]
fn range_empty_tree_yields_nothing() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let v: Vec<_> = tree.range().into_iter().collect();
    assert!(v.is_empty());
}

#[test]
fn range_no_filter_walks_all_keys_in_lex_order() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (b"banana", b"yellow"),
        (b"apple", b"red"),
        (b"cherry", b"dark"),
        (b"apricot", b"orange"),
    ];
    for (k, v) in &pairs {
        tree.put(k, v).unwrap();
    }
    let got: Vec<_> = tree
        .range()
        .into_iter()
        .map(|r| match r.unwrap() {
            RangeEntry::Key { key, value } => (key, value),
            RangeEntry::CommonPrefix(_) => panic!("no delimiter set"),
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (b"apple".to_vec(), b"red".to_vec()),
            (b"apricot".to_vec(), b"orange".to_vec()),
            (b"banana".to_vec(), b"yellow".to_vec()),
            (b"cherry".to_vec(), b"dark".to_vec()),
        ]
    );
}

#[test]
fn range_prefix_narrows_to_matching_subtree_only() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [
        &b"img/01.jpg"[..],
        b"img/02.jpg",
        b"img/03.jpg",
        b"video/1.mp4",
        b"video/2.mp4",
        b"doc/readme.md",
    ] {
        tree.put(k, b"v").unwrap();
    }
    let got = collect_keys(tree.range().prefix(b"img/"));
    assert_eq!(
        got,
        vec![
            b"img/01.jpg".to_vec(),
            b"img/02.jpg".to_vec(),
            b"img/03.jpg".to_vec(),
        ]
    );
    // Empty for a non-existent prefix.
    let none: Vec<_> = tree.range().prefix(b"music/").into_iter().collect();
    assert!(none.is_empty());
}

#[test]
fn range_start_after_is_strict_lower_bound() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..10u32 {
        tree.put(format!("k{i:02}").as_bytes(), b"v").unwrap();
    }
    let got = collect_keys(tree.range().start_after(b"k04"));
    assert_eq!(
        got,
        (5..10u32)
            .map(|i| format!("k{i:02}").into_bytes())
            .collect::<Vec<_>>(),
    );
    // Start after the last key — empty.
    let after_last: Vec<_> = tree.range().start_after(b"k09").into_iter().collect();
    assert!(after_last.is_empty());
}

#[test]
fn range_delimiter_rolls_up_common_prefixes_with_dedup() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [
        &b"img/01.jpg"[..],
        b"img/02.jpg",
        b"img/sub/a.jpg",
        b"img/sub/b.jpg",
        b"img/other/x.jpg",
    ] {
        tree.put(k, b"v").unwrap();
    }
    let mut keys_seen = Vec::new();
    let mut prefixes_seen = Vec::new();
    for r in tree.range().prefix(b"img/").delimiter(b'/').into_iter() {
        match r.unwrap() {
            RangeEntry::Key { key, .. } => keys_seen.push(key),
            RangeEntry::CommonPrefix(p) => prefixes_seen.push(p),
            _ => panic!("RangeEntry got a new variant"),
        }
    }
    // Lex order over leaves under img/:
    //   img/01.jpg            → Key (no `/` past prefix)
    //   img/02.jpg            → Key
    //   img/other/x.jpg       → CommonPrefix("img/other/")
    //   img/sub/a.jpg         → CommonPrefix("img/sub/")
    //   img/sub/b.jpg         → deduped, skipped
    assert_eq!(
        keys_seen,
        vec![b"img/01.jpg".to_vec(), b"img/02.jpg".to_vec()]
    );
    assert_eq!(
        prefixes_seen,
        vec![b"img/other/".to_vec(), b"img/sub/".to_vec()]
    );
}

#[test]
fn range_walks_across_blob_crossings() {
    // Force spillover so leaves are split across multiple blobs;
    // the iterator must descend through the BlobNode crossings
    // transparently and still produce keys in lex order.
    let tree = TreeBuilder::new("ignored")
        .memory()
        .buffer_pool_size(16)
        .open()
        .unwrap();
    let big = vec![0xABu8; 4 * 1024];
    for i in 0..256u32 {
        tree.put(format!("k{i:08}").as_bytes(), &big).unwrap();
    }
    assert!(
        tree.stats().unwrap().blob_count >= 2,
        "workload must spill into multiple blobs",
    );
    let got = collect_keys(tree.range());
    assert_eq!(got.len(), 256);
    let expected: Vec<Vec<u8>> = (0..256u32)
        .map(|i| format!("k{i:08}").into_bytes())
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn range_prefix_plus_start_after_combines() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [&b"img/01"[..], b"img/02", b"img/03", b"img/04", b"video/1"] {
        tree.put(k, b"v").unwrap();
    }
    let got = collect_keys(tree.range().prefix(b"img/").start_after(b"img/02"));
    assert_eq!(got, vec![b"img/03".to_vec(), b"img/04".to_vec()]);
}

#[test]
fn random_kv_insert_after_child_blob_compact_stays_consistent() {
    // Regression for the spillover NodeCorrupt at N≈10k random KV.
    //
    // Pattern: insert random 32-byte keys until the workload spills
    // across multiple blobs; eventually a child blob OOMs, the
    // `insert_at_blob_node` retry loop runs spillover_blob +
    // compact_blob on the child, the compact renumbers slots, and
    // before the fix the retry then walked through the stale
    // cached `child_entry` (from the parent's BlobNode) into a
    // slot that no longer exists -> "walker: invalid slot".
    //
    // The fix in `insert_at_blob_node` re-reads `child_entry` from
    // the child's freshly-rewritten `header.root_slot` after
    // compact_blob, so the retry descends through a live slot.
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_CAFE_BABE);
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..20_000)
        .map(|_| {
            let mut k = vec![0u8; 32];
            let mut v = vec![0u8; 64];
            rng.fill_bytes(&mut k);
            rng.fill_bytes(&mut v);
            (k, v)
        })
        .collect();

    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for (i, (k, v)) in pairs.iter().enumerate() {
        tree.put(k, v).unwrap_or_else(|e| {
            panic!("put #{i} failed: {e:?}");
        });
    }
    // Spot-check a few keys come back intact through the same
    // multi-blob tree.
    for &i in &[0usize, 5000, 12345, 19_999] {
        let (k, v) = &pairs[i];
        assert_eq!(tree.get(k).unwrap().as_deref(), Some(v.as_slice()));
    }
}

#[test]
fn range_delim_fast_forward_yields_every_distinct_common_prefix() {
    // Regression for v0.2 fast-forward in `Tree::range` delim mode.
    //
    // Builds an objstore-shaped tree with 32 distinct rollup buckets
    // (`bucket-00/...` through `bucket-31/...`), each holding 50
    // leaves. Without fast-forward the iterator dedup-scanned every
    // leaf to find 32 rollups (1600 leaf walks). With fast-forward
    // we ascend past each emitted rollup so the next descent
    // visits the next bucket's first leaf directly.
    //
    // Either way the **emitted set** must be exactly the 32
    // distinct `CommonPrefix` entries; this test fails (over-skip)
    // if the ascent over-pops past a Prefix node whose bytes span
    // the delimiter and re-anchors at a parent inner whose cursor
    // points past the byte that would have led to a still-unseen
    // bucket.
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for b in 0..32u32 {
        for f in 0..50u32 {
            let key = format!("bucket-{b:02}/path/sub/file-{f:04}.bin").into_bytes();
            tree.put(&key, b"v").unwrap();
        }
    }

    let mut rollups: Vec<Vec<u8>> = Vec::new();
    for entry in tree.range().prefix(b"bucket-").delimiter(b'/') {
        match entry.unwrap() {
            RangeEntry::CommonPrefix(p) => rollups.push(p),
            RangeEntry::Key { key, .. } => panic!(
                "expected only CommonPrefix entries, got Key({:?})",
                String::from_utf8_lossy(&key)
            ),
            _ => panic!("RangeEntry got a new variant"),
        }
    }

    assert_eq!(
        rollups.len(),
        32,
        "fast-forward must emit exactly one rollup per distinct bucket; \
         got {} of 32 (over-skip from a Prefix-spans-delimiter ascent)",
        rollups.len()
    );
    let expected: Vec<Vec<u8>> = (0..32u32)
        .map(|b| format!("bucket-{b:02}/").into_bytes())
        .collect();
    assert_eq!(rollups, expected, "rollups must come back in lex order");
}

#[test]
fn range_skips_tombstoned_leaves() {
    // Regression for a range-iter bug uncovered by the
    // `range_iteration_matches_oracle` property test:
    // `erase` soft-deletes leaves by flipping the `tombstone`
    // byte; the leaf body + extent stay allocated until
    // `compact_blob` rebuilds the blob. Range iteration walks
    // every reachable leaf — without a `leaf.tombstone == 0`
    // check it would emit the soft-deleted leaves and surface
    // phantom keys that `Tree::get` correctly says don't exist.
    //
    // The strict-prefix shape (`k` ⊂ `kk`) is the trigger: the
    // ART rewires the prefix chain on rename + insert and the
    // old `kk` leaf survives in the slot table tombstoned.
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"kk", b"B").unwrap();
    tree.rename(b"kk", b"k", false).unwrap();
    // `get` already agreed kk was gone — this exercises the
    // range iter specifically.
    assert!(tree.get(b"kk").unwrap().is_none());
    let actual = collect_keys(tree.range());
    assert_eq!(
        actual,
        vec![b"k".to_vec()],
        "range must skip tombstoned leaves; got {actual:?}",
    );

    // Same shape via explicit delete + reinsert.
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"foo", b"1").unwrap();
    tree.put(b"foobar", b"2").unwrap();
    tree.delete(b"foobar").unwrap();
    tree.put(b"foo", b"3").unwrap();
    let actual = collect_keys(tree.range());
    assert_eq!(actual, vec![b"foo".to_vec()]);
}
