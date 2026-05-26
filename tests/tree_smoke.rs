//! End-to-end smoke tests driving the public `Tree` API.
//!
//! Exercises only the public surface so signature breakage shows
//! up here first.

use std::sync::Arc;

use holt::{
    BlobStore, DBStats, MemoryBlobStore, RecordVersion, Tree, TreeBuilder, TreeConfig, TreeStats,
    DB,
};

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
fn open_with_explicit_blob_store_round_trips_root_blob() {
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let _t = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();
    let blobs_after_first = store.list_blobs().unwrap().len();
    assert!(blobs_after_first >= 1, "root blob should be present");

    let _t2 = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();
    assert_eq!(
        store.list_blobs().unwrap().len(),
        blobs_after_first,
        "re-open must not allocate a fresh root"
    );
}

#[test]
fn checkpoint_is_idempotent_on_memory_store() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.checkpoint().unwrap();
    tree.checkpoint().unwrap();
    assert!(tree.get(b"k").unwrap().is_none());
}

#[test]
fn db_named_trees_are_isolated_but_share_handles() {
    let db = DB::open(TreeConfig::memory()).unwrap();
    let objects = db.create_tree("objects").unwrap();
    let inodes = db.create_tree("inodes").unwrap();
    assert!(matches!(
        db.create_tree("objects"),
        Err(holt::Error::TreeExists { .. })
    ));

    objects.put(b"same/key", b"object").unwrap();
    inodes.put(b"same/key", b"inode").unwrap();

    assert_eq!(
        objects.get(b"same/key").unwrap().as_deref(),
        Some(&b"object"[..])
    );
    assert_eq!(
        inodes.get(b"same/key").unwrap().as_deref(),
        Some(&b"inode"[..])
    );

    let objects_again = db.open_tree("objects").unwrap();
    assert_eq!(
        objects_again.get(b"same/key").unwrap().as_deref(),
        Some(&b"object"[..])
    );
    assert_eq!(db.list_trees().unwrap(), vec!["inodes", "objects"]);
    assert!(matches!(
        db.open_tree("missing"),
        Err(holt::Error::TreeNotFound { .. })
    ));
    let lazy = db.open_or_create_tree("lazy").unwrap();
    lazy.put(b"k", b"v").unwrap();
    assert_eq!(
        db.open_or_create_tree("lazy")
            .unwrap()
            .get(b"k")
            .unwrap()
            .as_deref(),
        Some(&b"v"[..])
    );
}

#[test]
fn db_atomic_commits_and_aborts_across_trees() {
    let db = DB::open(TreeConfig::memory()).unwrap();
    let default = db.create_tree("mvcc/default").unwrap();
    let lock = db.create_tree("mvcc/lock").unwrap();

    assert!(matches!(
        db.atomic(|batch| {
            batch.put("missing", b"k", b"v");
        }),
        Err(holt::Error::TreeNotFound { .. })
    ));

    assert!(db
        .atomic(|batch| {
            batch.put("mvcc/default", b"user-key", b"value");
            batch.put("mvcc/lock", b"user-key", b"lock");
        })
        .unwrap());
    assert_eq!(
        default.get(b"user-key").unwrap().as_deref(),
        Some(&b"value"[..])
    );
    assert_eq!(
        lock.get(b"user-key").unwrap().as_deref(),
        Some(&b"lock"[..])
    );

    assert!(!db
        .atomic(|batch| {
            batch.assert_version(
                "mvcc/default",
                b"user-key",
                RecordVersion::from_raw(u64::MAX),
            );
            batch.put("mvcc/lock", b"should-not-appear", b"x");
        })
        .unwrap());
    assert!(lock.get(b"should-not-appear").unwrap().is_none());
}

#[test]
fn db_view_captures_explicit_tree_scopes() {
    let db = DB::open(TreeConfig::memory()).unwrap();
    let objects = db.create_tree("objects").unwrap();
    let inodes = db.create_tree("inodes").unwrap();

    objects.put(b"tenant-a/file", b"old-object").unwrap();
    objects.put(b"tenant-b/file", b"outside-scope").unwrap();
    inodes.put(b"42", b"old-inode").unwrap();

    let scopes: [(&str, &[u8]); 2] = [("objects", b"tenant-a/"), ("inodes", b"")];
    db.view(&scopes, |view| {
        objects.put(b"tenant-a/file", b"new-object")?;
        objects.put(b"tenant-a/new", b"new-object")?;
        inodes.put(b"42", b"new-inode")?;

        assert_eq!(view.len(), 2);
        assert!(view.tree("missing").is_none());

        let object_view = view.tree("objects").unwrap();
        assert_eq!(
            object_view.get(b"tenant-a/file")?.as_deref(),
            Some(&b"old-object"[..])
        );
        assert!(object_view.get(b"tenant-a/new")?.is_none());
        assert!(matches!(
            object_view.get(b"tenant-b/file"),
            Err(holt::Error::OutsideViewScope { .. })
        ));

        let inode_view = view.tree("inodes").unwrap();
        assert_eq!(inode_view.get(b"42")?.as_deref(), Some(&b"old-inode"[..]));
        Ok(())
    })
    .unwrap();
}

#[test]
fn db_stats_reports_shared_resources() {
    let db = DB::open(TreeConfig::memory()).unwrap();
    assert_eq!(db.stats().open_tree_count, 0);

    let objects = db.create_tree("objects").unwrap();
    let inodes = db.create_tree("inodes").unwrap();
    objects.put(b"bucket/a", b"etag").unwrap();
    inodes.put(b"42", b"mode").unwrap();

    let stats: DBStats = db.stats();
    assert_eq!(stats.open_tree_count, 2);
    assert!(
        stats.bm_walker_ops >= 2,
        "shared DB stats should include writes from every named tree"
    );
    assert!(stats.journal.is_none());
}

// ----------------------------------------------------------------
// Put / Get
// ----------------------------------------------------------------

#[test]
fn put_then_get_round_trip() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"hello", b"world").unwrap();
    assert_eq!(tree.get(b"hello").unwrap().as_deref(), Some(&b"world"[..]));
    assert!(tree.get(b"missing").unwrap().is_none());
}

#[test]
fn put_overwrites_existing_value() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v1").unwrap();
    tree.put(b"k", b"v2").unwrap();
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v2"[..]));
}

#[test]
fn blind_same_size_put_updates_in_place() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"aaaa").unwrap();
    let before_record = tree.get_record(b"k").unwrap().unwrap();
    let before_stats = tree.stats().unwrap();

    tree.put(b"k", b"bbbb").unwrap();
    let after_record = tree.get_record(b"k").unwrap().unwrap();
    let after_stats = tree.stats().unwrap();

    assert_eq!(after_record.value, b"bbbb");
    assert!(
        after_record.version > before_record.version,
        "same-size update must still publish a fresh record version",
    );
    assert_eq!(after_stats.total_space_used, before_stats.total_space_used);
    assert_eq!(after_stats.total_gap_space, before_stats.total_gap_space);
    assert_eq!(after_stats.total_slots, before_stats.total_slots);
}

#[test]
fn conditional_same_size_put_updates_in_place() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"aaaa").unwrap();
    let before_record = tree.get_record(b"k").unwrap().unwrap();
    let before_stats = tree.stats().unwrap();

    assert!(tree
        .compare_and_put(b"k", before_record.version, b"bbbb")
        .unwrap());
    let after_record = tree.get_record(b"k").unwrap().unwrap();
    let after_stats = tree.stats().unwrap();

    assert_eq!(after_record.value, b"bbbb");
    assert!(after_record.version > before_record.version);
    assert_eq!(after_stats.total_space_used, before_stats.total_space_used);
    assert_eq!(after_stats.total_gap_space, before_stats.total_gap_space);
    assert_eq!(after_stats.total_slots, before_stats.total_slots);
}

#[test]
fn conditional_puts_use_record_versions() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.get_version(b"k").unwrap().is_none());

    assert!(tree.put_if_absent(b"k", b"v1").unwrap());
    assert!(!tree.put_if_absent(b"k", b"blocked").unwrap());
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v1"[..]));

    let v1 = tree.get_version(b"k").unwrap().unwrap();
    assert!(tree.compare_and_put(b"k", v1, b"v2").unwrap());
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v2"[..]));

    assert!(
        !tree.compare_and_put(b"k", v1, b"stale").unwrap(),
        "stale version must not overwrite the newer value",
    );
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v2"[..]));
}

#[test]
fn get_record_returns_value_and_version_together() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v1").unwrap();

    let record = tree.get_record(b"k").unwrap().unwrap();
    assert_eq!(record.value, b"v1");
    assert_eq!(
        tree.get_version(b"k").unwrap().unwrap(),
        record.version,
        "get_record must return the live CAS token from the same lookup",
    );
}

#[test]
fn conditional_delete_uses_record_versions() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v1").unwrap();
    let v1 = tree.get_version(b"k").unwrap().unwrap();

    tree.put(b"k", b"v2").unwrap();
    assert!(!tree.delete_if_version(b"k", v1).unwrap());
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v2"[..]));

    let v2 = tree.get_version(b"k").unwrap().unwrap();
    assert!(tree.delete_if_version(b"k", v2).unwrap());
    assert!(tree.get(b"k").unwrap().is_none());
    assert!(tree.get_version(b"k").unwrap().is_none());

    assert!(tree.put_if_absent(b"k", b"v3").unwrap());
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v3"[..]));
}

#[test]
fn conditional_put_reaches_cross_blob_children() {
    let tree = TreeBuilder::new("scratch")
        .memory()
        .buffer_pool_size(8)
        .open()
        .unwrap();
    let big = vec![0xCDu8; 4 * 1024];
    for i in 0..256u32 {
        tree.put(format!("obj/{i:04}/meta").as_bytes(), &big)
            .unwrap();
    }

    let key = b"obj/0128/meta";
    let version = tree.get_version(key).unwrap().unwrap();
    assert!(tree.compare_and_put(key, version, b"small").unwrap());
    assert_eq!(tree.get(key).unwrap().as_deref(), Some(&b"small"[..]));

    assert!(!tree.compare_and_put(key, version, b"stale").unwrap());
    assert_eq!(tree.get(key).unwrap().as_deref(), Some(&b"small"[..]));
}

#[test]
fn failed_compare_and_put_on_absent_prefix_path_is_noop() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"img/aaaa", b"a").unwrap();
    tree.put(b"img/aaab", b"b").unwrap();

    let before = tree.stats().unwrap();
    assert!(!tree
        .compare_and_put(
            b"img/aazz",
            holt::RecordVersion::from_raw(u64::MAX),
            b"nope",
        )
        .unwrap(),);
    let after = tree.stats().unwrap();

    assert!(tree.get(b"img/aazz").unwrap().is_none());
    assert_eq!(before.total_slots, after.total_slots);
    assert_eq!(before.total_space_used, after.total_space_used);
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
fn concurrent_writers_round_trip() {
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
fn delete_existing_key_returns_true_and_removes_it() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v").unwrap();
    assert!(tree.delete(b"k").unwrap());
    assert!(tree.get(b"k").unwrap().is_none());
}

#[test]
fn delete_missing_key_is_noop() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(!tree.delete(b"missing").unwrap());
}

#[test]
fn delete_then_reinsert_round_trips() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v1").unwrap();
    assert!(tree.delete(b"k").unwrap());
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
    for (k, _) in &pairs {
        assert!(tree.delete(k).unwrap());
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
    assert!(tree.delete(b"img/02.jpg").unwrap());
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
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
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
            "post-spillover lookup failed at key {k:?}; store has {} blob(s)",
            store.list_blobs().unwrap().len(),
        );
    }

    // Spillover should have created at least one child blob.
    let blobs = store.list_blobs().unwrap();
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
    // BufferManager between Tree and the inner store keeps the
    // child blobs cached after the first read; per-blob locks
    // mean concurrent reads on *different* blobs don't fight for
    // a single mutex.
    use std::thread;

    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = Arc::new(
        TreeBuilder::new("ignored")
            .open_with_blob_store(store.clone())
            .unwrap(),
    );

    const N: u32 = 2000;
    let value = vec![0x66; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &value).unwrap();
    }
    // Multi-blob pre-cond.
    assert!(
        store.list_blobs().unwrap().len() >= 2,
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
fn concurrent_writers_cross_blob_via_public_tree_api() {
    use std::sync::Barrier;
    use std::thread;

    let tree = Arc::new(Tree::open(TreeConfig::memory()).unwrap());
    let seed_value = vec![0x5A; 220];
    for dir in 0..8u32 {
        for file in 0..350u32 {
            let key = format!("tenant-{dir:02}/dir-{file:04}/seed").into_bytes();
            tree.put(&key, &seed_value).unwrap();
        }
    }
    assert!(
        tree.stats().unwrap().blob_count >= 2,
        "test precondition: seed workload must spill into child blobs",
    );

    const WRITERS: usize = 8;
    const PER_WRITER: u32 = 120;
    let barrier = Arc::new(Barrier::new(WRITERS));
    let handles: Vec<_> = (0..WRITERS)
        .map(|writer| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..PER_WRITER {
                    let key = format!("tenant-{writer:02}/hot/new-{i:04}").into_bytes();
                    let value = format!("writer-{writer}/value-{i}").into_bytes();
                    tree.put(&key, &value).unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    for writer in 0..WRITERS {
        for i in [0, PER_WRITER / 2, PER_WRITER - 1] {
            let key = format!("tenant-{writer:02}/hot/new-{i:04}").into_bytes();
            let value = format!("writer-{writer}/value-{i}").into_bytes();
            assert_eq!(tree.get(&key).unwrap().as_deref(), Some(value.as_slice()));
        }
    }
    let stats = tree.stats().unwrap();
    assert!(
        stats.bm_max_blob_hops >= 2,
        "writers should have crossed at least one BlobNode boundary; stats={stats:?}",
    );
}

#[test]
fn compact_then_insert_reclaims_extent_leak() {
    // Pure-mutation workload: insert N keys, delete half, insert
    // another N. Without compact reclaiming the deleted-leaf
    // extents, the bump cursor would push past blob capacity and
    // force spillover much sooner. With compact wired into the
    // OOM recovery path, the extent leak is recoverable and the
    // workload stays in many fewer blobs (often the single root).
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();

    let val = vec![0x42; 200];

    for i in 0..1500u32 {
        tree.put(format!("k{i:08}").as_bytes(), &val).unwrap();
    }
    // Delete the lower half — leaves ~750 keys live + ~750
    // leaked extents.
    for i in 0..750u32 {
        assert!(tree.delete(format!("k{i:08}").as_bytes()).unwrap());
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
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();

    const N: u32 = 2000;
    let value = vec![0x42; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &value).unwrap();
    }
    assert!(
        store.list_blobs().unwrap().len() >= 2,
        "test pre-cond: expected multi-blob state",
    );

    // Delete every 5th key.
    let mut deleted = 0u32;
    for i in 0..N {
        if i % 5 != 0 {
            continue;
        }
        let k = format!("k{i:08}").into_bytes();
        assert!(tree.delete(&k).unwrap(), "delete missed key {k:?}");
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
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
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
        store.list_blobs().unwrap().len() >= 2,
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
    // reopening the same store must surface the same key/value
    // mapping (root blob + all spilled child blobs are persisted).
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    {
        let tree = TreeBuilder::new("ignored")
            .open_with_blob_store(store.clone())
            .unwrap();
        for i in 0..2000u32 {
            tree.put(format!("k{i:08}").as_bytes(), &[0xCD; 192])
                .unwrap();
        }
        tree.checkpoint().unwrap();
    }

    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();
    for i in 0..2000u32 {
        let k = format!("k{i:08}").into_bytes();
        let v = tree.get(&k).unwrap();
        assert!(
            v.is_some(),
            "post-reopen lookup failed at key {k:?}; store has {} blob(s)",
            store.list_blobs().unwrap().len(),
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
    assert_eq!(s.total_blob_edges, 0);
    assert_eq!(s.leaf_blob_count, 1);
    assert_eq!(s.max_blob_depth, 0);
    assert_eq!(s.total_blob_depth, 0);
    assert_eq!(s.route_cache.entries, 0);
    assert_eq!(s.route_cache.hits, 0);
    assert_eq!(s.route_cache.misses, 0);
    // A freshly-initialised blob holds the EmptyRoot sentinel, so
    // it has consumed some bump-area bytes and at least one slot.
    assert!(s.total_space_used > 0);
    assert!(s.total_slots >= 1);
}

#[test]
fn stats_reflects_inserts() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let before = tree.stats().unwrap();
    assert_eq!(before.bm_walker_ops, 0);
    for i in 0..16u32 {
        tree.put(format!("k{i:04}").as_bytes(), b"value-bytes-here")
            .unwrap();
    }
    let after = tree.stats().unwrap();
    assert!(after.total_space_used > before.total_space_used);
    assert!(after.total_slots > before.total_slots);
    assert_eq!(after.blob_count, 1, "16 small keys still fit in one blob");
    assert!(after.bm_walker_ops >= 16);
    assert!(after.bm_walker_blob_hops >= after.bm_walker_ops);
    assert!(after.bm_max_blob_hops >= 1);
    assert!(after.bm_avg_blob_hops() >= 1.0);
}

#[test]
fn point_get_uses_route_cache_on_multi_blob_tree() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let value = vec![0xAB; 256];
    for i in 0..2400u32 {
        let key = format!("bucket-000/tenant-00/path/sub/file-{i:08}.bin");
        tree.put(key.as_bytes(), &value).unwrap();
    }
    assert!(
        tree.stats().unwrap().blob_count >= 2,
        "test precondition: path-shaped keys should spill into child blobs",
    );

    let before = tree.stats().unwrap().route_cache;
    assert_eq!(
        tree.get(b"bucket-000/tenant-00/path/sub/file-00000000.bin")
            .unwrap()
            .as_deref(),
        Some(&value[..]),
    );
    let after_first = tree.stats().unwrap().route_cache;
    assert!(
        after_first.hits > before.hits || after_first.learns > before.learns,
        "first read should either hit an existing route or learn one",
    );

    assert_eq!(
        tree.get(b"bucket-000/tenant-00/path/sub/file-00000001.bin")
            .unwrap()
            .as_deref(),
        Some(&value[..]),
    );
    let after_second = tree.stats().unwrap().route_cache;
    assert!(
        after_second.hits > after_first.hits,
        "second read under the same routed prefix should hit the route cache",
    );
}

#[test]
fn compact_on_empty_tree_skips_noop_rewrite() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let before = tree.stats().unwrap();
    assert_eq!(before.total_compactions, 0);
    tree.compact().unwrap();
    let after = tree.stats().unwrap();
    assert_eq!(
        after.total_compactions, 0,
        "clean root blob must not be rebuilt just to bump compact_times"
    );
}

#[test]
fn compact_after_writes_preserves_data_and_skips_clean_blobs() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..32u32 {
        tree.put(format!("k{i:04}").as_bytes(), format!("v-{i}").as_bytes())
            .unwrap();
    }
    let pre_space = tree.stats().unwrap().total_space_used;
    tree.compact().unwrap();
    let post = tree.stats().unwrap();
    assert_eq!(
        post.total_compactions, 0,
        "pure inserts create no tombstones/free-list garbage, so compact should skip"
    );
    // Re-read every key — compact must not lose data.
    for i in 0..32u32 {
        let got = tree.get(format!("k{i:04}").as_bytes()).unwrap();
        assert_eq!(got.as_deref(), Some(format!("v-{i}").as_bytes()));
    }
    // No tombstones or freed leaf slots yet, so compact should
    // leave the blob byte accounting untouched.
    assert_eq!(post.total_space_used, pre_space);
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
    // Re-insert at the same key should resurrect the leaf in place
    // and drop the tombstone counter back to zero.
    tree.put(b"k", b"v2").unwrap();
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
    assert!(
        before.bm_spillovers > 0,
        "multi-blob workload must have recorded at least one spillover"
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
    assert!(
        after.bm_merges > before.bm_merges,
        "compact should record folded child blobs"
    );
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
    assert!(
        s.bm_spillovers > 0,
        "stats should expose foreground spillover count"
    );
    assert!(s.bm_walker_blob_hops >= s.bm_walker_ops);
    assert!(s.bm_max_blob_hops >= 1);
    assert!(s.total_blob_edges >= 1);
    assert!(s.leaf_blob_count >= 1);
    assert!(s.max_blob_depth >= 1);
    assert!(s.total_blob_depth >= u64::from(s.max_blob_depth));
    assert!(s.avg_blob_depth() > 0.0);
    assert!(s.leaf_blob_ratio() > 0.0);
    assert!(s.avg_blob_fill_ratio() > 0.0);
    assert!(s.max_blob_fill_ratio() > 0.0);
    // Aggregate is just the sum of per-blob counters.
    let sum_space: u64 = s.blobs.iter().map(|b| u64::from(b.space_used)).sum();
    assert_eq!(sum_space, s.total_space_used);
    let sum_slots: u64 = s.blobs.iter().map(|b| u64::from(b.num_slots)).sum();
    assert_eq!(sum_slots, s.total_slots);
    let sum_edges: u64 = s.blobs.iter().map(|b| u64::from(b.num_ext_blobs)).sum();
    assert_eq!(sum_edges, s.total_blob_edges);
}

#[test]
fn atomic_applies_buffered_ops_in_order() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"seed", b"S").unwrap();

    assert!(tree
        .atomic(|batch| {
            batch.put(b"a", b"1");
            batch.put(b"b", b"2");
            batch.delete(b"seed");
            batch.rename(b"a", b"aa", false);
        })
        .unwrap());

    assert!(tree.get(b"seed").unwrap().is_none());
    assert!(tree.get(b"a").unwrap().is_none());
    assert_eq!(tree.get(b"aa").unwrap().as_deref(), Some(&b"1"[..]));
    assert_eq!(tree.get(b"b").unwrap().as_deref(), Some(&b"2"[..]));
}

#[test]
fn atomic_empty_batch_is_a_noop() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"unchanged", b"X").unwrap();

    assert!(tree.atomic(|_batch| {}).unwrap());

    assert_eq!(tree.get(b"unchanged").unwrap().as_deref(), Some(&b"X"[..]));
}

#[test]
fn atomic_returns_error_when_rename_src_missing_without_partial_publish() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let err = tree
        .atomic(|batch| {
            batch.put(b"committed", b"C");
            batch.rename(b"missing-src", b"dst", false);
            batch.put(b"never-reached", b"N");
        })
        .unwrap_err();
    assert!(
        matches!(err, holt::Error::NotFound),
        "expected NotFound from rename of missing src, got {err:?}",
    );
    assert!(
        tree.get(b"committed").unwrap().is_none(),
        "preflight failure must not publish earlier batch ops",
    );
    assert!(tree.get(b"never-reached").unwrap().is_none());
}

#[test]
fn atomic_conditional_guard_failure_is_invisible() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"guarded", b"v1").unwrap();
    let stale = tree.get_version(b"guarded").unwrap().unwrap();
    tree.put(b"guarded", b"v2").unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.put(b"before", b"B");
            batch.compare_and_put(b"guarded", stale, b"stale-write");
            batch.put(b"after", b"A");
        })
        .unwrap();

    assert!(!committed);
    assert_eq!(tree.get(b"guarded").unwrap().as_deref(), Some(&b"v2"[..]));
    assert!(tree.get(b"before").unwrap().is_none());
    assert!(tree.get(b"after").unwrap().is_none());
}

#[test]
fn atomic_conditional_ops_commit_in_order() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"update", b"old").unwrap();
    tree.put(b"delete", b"live").unwrap();
    let update_v = tree.get_record(b"update").unwrap().unwrap().version;
    let delete_v = tree.get_record(b"delete").unwrap().unwrap().version;

    assert!(tree
        .atomic(|batch| {
            batch.put_if_absent(b"create", b"new");
            batch.compare_and_put(b"update", update_v, b"newer");
            batch.delete_if_version(b"delete", delete_v);
        })
        .unwrap());

    assert_eq!(tree.get(b"create").unwrap().as_deref(), Some(&b"new"[..]));
    assert_eq!(tree.get(b"update").unwrap().as_deref(), Some(&b"newer"[..]));
    assert!(tree.get(b"delete").unwrap().is_none());
}

#[test]
fn atomic_insert_run_handles_blob_spillover() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let value = vec![0xCD; 4 * 1024];

    assert!(tree
        .atomic(|batch| {
            for i in 0..256u32 {
                let key = format!("bulk/path/object-{i:04}");
                batch.put(key.as_bytes(), &value);
            }
        })
        .unwrap());

    for i in 0..256u32 {
        let key = format!("bulk/path/object-{i:04}");
        assert_eq!(
            tree.get(key.as_bytes()).unwrap().as_deref(),
            Some(&value[..])
        );
    }
    let stats = tree.stats().unwrap();
    assert!(
        stats.blob_count >= 2,
        "large atomic insert run should force at least one spillover"
    );
}

#[test]
fn is_prefix_empty_tracks_live_keys() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.is_prefix_empty(b"dir/").unwrap());

    tree.put(b"dir/child", b"child").unwrap();
    assert!(!tree.is_prefix_empty(b"dir/").unwrap());
    assert!(tree.is_prefix_empty(b"missing/").unwrap());

    tree.delete(b"dir/child").unwrap();
    assert!(tree.is_prefix_empty(b"dir/").unwrap());
}

#[test]
fn atomic_assert_prefix_empty_failure_is_invisible() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"dir", b"meta").unwrap();
    tree.put(b"dir/child", b"child").unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.assert_prefix_empty(b"dir/");
            batch.delete(b"dir");
        })
        .unwrap();

    assert!(!committed);
    assert_eq!(tree.get(b"dir").unwrap().as_deref(), Some(&b"meta"[..]));
    assert_eq!(
        tree.get(b"dir/child").unwrap().as_deref(),
        Some(&b"child"[..])
    );
}

#[test]
fn atomic_assert_prefix_empty_observes_staged_deletes() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"dir", b"meta").unwrap();
    tree.put(b"dir/child", b"child").unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.delete(b"dir/child");
            batch.assert_prefix_empty(b"dir/");
            batch.delete(b"dir");
        })
        .unwrap();

    assert!(committed);
    assert!(tree.get(b"dir").unwrap().is_none());
    assert!(tree.get(b"dir/child").unwrap().is_none());
}

#[test]
fn atomic_assert_prefix_empty_still_sees_non_deleted_live_child() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"dir/a", b"a").unwrap();
    tree.put(b"dir/b", b"b").unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.delete(b"dir/a");
            batch.assert_prefix_empty(b"dir/");
            batch.put(b"marker", b"should-not-publish");
        })
        .unwrap();

    assert!(!committed);
    assert_eq!(tree.get(b"dir/a").unwrap().as_deref(), Some(&b"a"[..]));
    assert_eq!(tree.get(b"dir/b").unwrap().as_deref(), Some(&b"b"[..]));
    assert!(tree.get(b"marker").unwrap().is_none());
}

#[test]
fn atomic_assert_prefix_empty_observes_staged_puts() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.put(b"dir/new", b"new");
            batch.assert_prefix_empty(b"dir/");
        })
        .unwrap();

    assert!(!committed);
    assert!(tree.get(b"dir/new").unwrap().is_none());
}

#[test]
fn atomic_assert_version_copies_without_bumping_source() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"src", b"payload").unwrap();
    let src = tree.get_record(b"src").unwrap().unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.assert_version(b"src", src.version);
            batch.put(b"dst", &src.value);
        })
        .unwrap();

    assert!(committed);
    let src_after = tree.get_record(b"src").unwrap().unwrap();
    assert_eq!(
        src_after.version, src.version,
        "assert_version must not rewrite or bump the guarded source",
    );
    assert_eq!(tree.get(b"dst").unwrap().as_deref(), Some(&b"payload"[..]));
}

#[test]
fn atomic_assert_version_failure_is_invisible() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"src", b"v1").unwrap();
    let stale = tree.get_version(b"src").unwrap().unwrap();
    tree.put(b"src", b"v2").unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.assert_version(b"src", stale);
            batch.put(b"dst", b"should-not-publish");
        })
        .unwrap();

    assert!(!committed);
    assert_eq!(tree.get(b"src").unwrap().as_deref(), Some(&b"v2"[..]));
    assert!(tree.get(b"dst").unwrap().is_none());
}

#[test]
fn atomic_assert_version_observes_staged_deletes() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"src", b"v1").unwrap();
    let version = tree.get_version(b"src").unwrap().unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.delete(b"src");
            batch.assert_version(b"src", version);
            batch.put(b"dst", b"should-not-publish");
        })
        .unwrap();

    assert!(!committed);
    assert_eq!(tree.get(b"src").unwrap().as_deref(), Some(&b"v1"[..]));
    assert!(tree.get(b"dst").unwrap().is_none());
}

#[test]
fn atomic_assert_version_observes_staged_updates() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"src", b"v1").unwrap();
    let old = tree.get_version(b"src").unwrap().unwrap();

    let committed = tree
        .atomic(|batch| {
            batch.compare_and_put(b"src", old, b"v2");
            batch.assert_version(b"src", old);
            batch.put(b"side", b"should-not-publish");
        })
        .unwrap();

    assert!(!committed);
    assert_eq!(tree.get(b"src").unwrap().as_deref(), Some(&b"v1"[..]));
    assert!(tree.get(b"side").unwrap().is_none());
}

// ----------------------------------------------------------------
// Tree::range
// ----------------------------------------------------------------

use holt::{KeyRangeEntry, KeyRangeEntryRef, RangeEntry};

fn collect_keys(iter: impl IntoIterator<Item = Result<RangeEntry, holt::Error>>) -> Vec<Vec<u8>> {
    iter.into_iter()
        .map(|r| match r.unwrap() {
            RangeEntry::Key { key, .. } => key,
            RangeEntry::CommonPrefix(p) => p,
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect()
}

fn collect_key_range(
    iter: impl IntoIterator<Item = Result<KeyRangeEntry, holt::Error>>,
) -> Vec<Vec<u8>> {
    iter.into_iter()
        .map(|r| match r.unwrap() {
            KeyRangeEntry::Key { key, .. } => key,
            KeyRangeEntry::CommonPrefix(p) => p,
            _ => panic!("KeyRangeEntry got a new variant"),
        })
        .collect()
}

fn collect_key_entry_refs(tree: &Tree, prefix: &[u8], delimiter: u8, limit: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    tree.scan_keys(prefix)
        .delimiter(delimiter)
        .visit(limit, |entry| {
            match entry {
                KeyRangeEntryRef::Key { key, .. } => out.push(key.to_vec()),
                KeyRangeEntryRef::CommonPrefix(prefix) => out.push(prefix.to_vec()),
                _ => panic!("KeyRangeEntryRef got a new variant"),
            }
            Ok(())
        })
        .unwrap();
    out
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
            RangeEntry::Key { key, value, .. } => (key, value),
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
fn range_key_entries_expose_live_versions() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"img/a", b"v1").unwrap();
    tree.put(b"img/b", b"v2").unwrap();
    tree.put(b"img/a", b"v3").unwrap();
    let a = tree.get_record(b"img/a").unwrap().unwrap();
    let b = tree.get_record(b"img/b").unwrap().unwrap();

    let got: Vec<_> = tree
        .scan(b"img/")
        .into_iter()
        .map(|r| match r.unwrap() {
            RangeEntry::Key {
                key,
                value,
                version,
            } => (key, value, version),
            RangeEntry::CommonPrefix(_) => panic!("no delimiter set"),
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect();

    assert_eq!(
        got,
        vec![
            (b"img/a".to_vec(), b"v3".to_vec(), a.version),
            (b"img/b".to_vec(), b"v2".to_vec(), b.version),
        ],
    );
}

#[test]
fn range_keys_returns_keys_and_versions_without_values() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"img/a", b"v1").unwrap();
    tree.put(b"img/b", b"v2").unwrap();
    tree.put(b"img/a", b"v3").unwrap();
    let a = tree.get_record(b"img/a").unwrap().unwrap();
    let b = tree.get_record(b"img/b").unwrap().unwrap();

    let got: Vec<_> = tree
        .scan_keys(b"img/")
        .into_iter()
        .map(|r| match r.unwrap() {
            KeyRangeEntry::Key { key, version } => (key, version),
            KeyRangeEntry::CommonPrefix(_) => panic!("no delimiter set"),
            _ => panic!("KeyRangeEntry got a new variant"),
        })
        .collect();

    assert_eq!(
        got,
        vec![
            (b"img/a".to_vec(), a.version),
            (b"img/b".to_vec(), b.version),
        ],
    );
}

#[test]
fn range_keys_supports_start_after_and_delimiter_rollup() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [
        &b"img/01.jpg"[..],
        b"img/02.jpg",
        b"img/other/x.jpg",
        b"img/sub/a.jpg",
        b"img/sub/b.jpg",
        b"video/1.mp4",
    ] {
        tree.put(k, b"value-that-must-not-be-needed-for-listing")
            .unwrap();
    }

    let got = collect_key_range(
        tree.range_keys()
            .prefix(b"img/")
            .start_after(b"img/01.jpg")
            .delimiter(b'/'),
    );
    assert_eq!(
        got,
        vec![
            b"img/02.jpg".to_vec(),
            b"img/other/".to_vec(),
            b"img/sub/".to_vec(),
        ],
    );
}

#[test]
fn key_range_builder_visit_matches_key_range_iterator_order() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [
        &b"img/01.jpg"[..],
        b"img/02.jpg",
        b"img/other/x.jpg",
        b"img/sub/a.jpg",
        b"img/sub/b.jpg",
        b"video/1.mp4",
    ] {
        tree.put(k, b"value").unwrap();
    }

    let iter = collect_key_range(tree.scan_keys(b"img/").delimiter(b'/'));
    let visitor = collect_key_entry_refs(&tree, b"img/", b'/', usize::MAX);
    assert_eq!(visitor, iter);
}

#[test]
fn key_range_builder_visit_supports_start_after() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [b"a/0".as_slice(), b"a/1", b"a/2", b"a/sub/0"] {
        tree.put(k, b"value").unwrap();
    }

    let mut got = Vec::new();
    tree.scan_keys(b"a/")
        .start_after(b"a/0")
        .delimiter(b'/')
        .visit(8, |entry| {
            match entry {
                KeyRangeEntryRef::Key { key, .. } => got.push(key.to_vec()),
                KeyRangeEntryRef::CommonPrefix(prefix) => got.push(prefix.to_vec()),
                _ => panic!("KeyRangeEntryRef got a new variant"),
            }
            Ok(())
        })
        .unwrap();

    assert_eq!(
        got,
        vec![b"a/1".to_vec(), b"a/2".to_vec(), b"a/sub/".to_vec()]
    );
}

#[test]
fn key_range_builder_visit_honors_limit() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..8u32 {
        tree.put(format!("bucket-{i:02}/file").as_bytes(), b"value")
            .unwrap();
    }

    let got = collect_key_entry_refs(&tree, b"bucket-", b'/', 3);
    assert_eq!(
        got,
        vec![
            b"bucket-00/".to_vec(),
            b"bucket-01/".to_vec(),
            b"bucket-02/".to_vec(),
        ],
    );
}

// ----------------------------------------------------------------
// Tree::view
// ----------------------------------------------------------------

#[test]
fn view_get_reads_captured_state_after_live_mutation() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"tenant-a/file", b"old").unwrap();
    tree.put(b"tenant-b/file", b"outside").unwrap();

    tree.view(b"tenant-a/", |view| {
        assert_eq!(view.get(b"tenant-a/file")?.as_deref(), Some(&b"old"[..]));

        tree.put(b"tenant-a/file", b"new").unwrap();
        tree.put(b"tenant-a/created-after-view", b"new").unwrap();
        tree.delete(b"tenant-a/file").unwrap();

        assert_eq!(view.get(b"tenant-a/file")?.as_deref(), Some(&b"old"[..]));
        assert!(view.get(b"tenant-a/created-after-view")?.is_none());
        Ok(())
    })
    .unwrap();

    assert!(tree.get(b"tenant-a/file").unwrap().is_none());
    assert_eq!(
        tree.get(b"tenant-a/created-after-view").unwrap().as_deref(),
        Some(&b"new"[..])
    );
}

#[test]
fn view_range_and_key_visit_are_snapshot_consistent() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for key in [b"dir/a".as_slice(), b"dir/b", b"dir/sub/x"] {
        tree.put(key, b"value").unwrap();
    }

    tree.view(b"dir/", |view| {
        tree.put(b"dir/c", b"value").unwrap();
        tree.put(b"dir/sub/y", b"value").unwrap();

        let records = collect_keys(view.range().delimiter(b'/'));
        assert_eq!(
            records,
            vec![b"dir/a".to_vec(), b"dir/b".to_vec(), b"dir/sub/".to_vec()]
        );

        let mut keys = Vec::new();
        view.range_keys().delimiter(b'/').visit(16, |entry| {
            match entry {
                KeyRangeEntryRef::Key { key, .. } => keys.push(key.to_vec()),
                KeyRangeEntryRef::CommonPrefix(prefix) => keys.push(prefix.to_vec()),
                _ => panic!("KeyRangeEntryRef got a new variant"),
            }
            Ok(())
        })?;
        assert_eq!(keys, records);
        Ok(())
    })
    .unwrap();

    let live = collect_keys(tree.scan(b"dir/").delimiter(b'/'));
    assert_eq!(
        live,
        vec![
            b"dir/a".to_vec(),
            b"dir/b".to_vec(),
            b"dir/c".to_vec(),
            b"dir/sub/".to_vec(),
        ]
    );
}

#[test]
fn view_scoped_reads_expose_versions_and_bounds() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for (key, value) in [
        (b"dir/a".as_slice(), b"a".as_slice()),
        (b"dir/b", b"b"),
        (b"dir/sub/a", b"sub-a"),
        (b"other/a", b"other"),
    ] {
        tree.put(key, value).unwrap();
    }
    let live_b = tree.get_record(b"dir/b").unwrap().unwrap();

    tree.view(b"dir/", |view| {
        assert_eq!(view.scope(), b"dir/");
        assert_eq!(view.get_record(b"dir/b")?, Some(live_b.clone()));
        assert_eq!(view.get_version(b"dir/b")?, Some(live_b.version));
        assert!(!view.is_prefix_empty(b"dir/sub/")?);
        assert!(view.is_prefix_empty(b"dir/missing/")?);

        let scoped = collect_keys(view.scan(b"dir/")?.start_after(b"dir/a"));
        assert_eq!(scoped, vec![b"dir/b".to_vec(), b"dir/sub/a".to_vec()],);

        let scoped_keys = collect_key_range(view.scan_keys(b"dir/")?.start_after(b"dir/a"));
        assert_eq!(scoped_keys, scoped);
        Ok(())
    })
    .unwrap();
}

#[test]
fn view_empty_scope_reads_the_whole_tree() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"tenant-a/file", b"a").unwrap();
    tree.put(b"tenant-b/file", b"b").unwrap();

    tree.view(b"", |view| {
        assert_eq!(view.scope(), b"");
        assert_eq!(view.get(b"tenant-a/file")?.as_deref(), Some(&b"a"[..]));
        assert_eq!(view.get(b"tenant-b/file")?.as_deref(), Some(&b"b"[..]));

        let rolled = collect_key_range(view.scan_keys(b"tenant-")?.delimiter(b'/'));
        assert_eq!(rolled, vec![b"tenant-a/".to_vec(), b"tenant-b/".to_vec()]);
        Ok(())
    })
    .unwrap();
}

#[test]
fn view_rejects_reads_outside_captured_scope() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"tenant-a/file", b"a").unwrap();
    tree.put(b"tenant-b/file", b"b").unwrap();

    tree.view(b"tenant-a/", |view| {
        assert!(matches!(
            view.get(b"tenant-b/file"),
            Err(holt::Error::OutsideViewScope { .. })
        ));
        assert!(matches!(
            view.scan(b"tenant-b/"),
            Err(holt::Error::OutsideViewScope { .. })
        ));
        assert!(matches!(
            view.is_prefix_empty(b"tenant-"),
            Err(holt::Error::OutsideViewScope { .. })
        ));
        Ok(())
    })
    .unwrap();
}

#[test]
fn view_prefix_snapshot_spans_child_blobs() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    let value = vec![0x7A; 220];
    for i in 0..1800u32 {
        tree.put(format!("tenant-a/dir-{i:04}/file").as_bytes(), &value)
            .unwrap();
    }
    assert!(
        tree.stats().unwrap().blob_count >= 2,
        "test precondition: tenant-a prefix should spill into child blobs",
    );

    tree.view(b"tenant-a/", |view| {
        tree.put(b"tenant-a/dir-0000/file", b"updated").unwrap();
        tree.put(b"tenant-a/new-after-view", b"new").unwrap();

        for i in [0, 17, 511, 1023, 1799] {
            let key = format!("tenant-a/dir-{i:04}/file");
            assert_eq!(view.get(key.as_bytes())?.as_deref(), Some(&value[..]));
        }
        assert!(view.get(b"tenant-a/new-after-view")?.is_none());
        Ok(())
    })
    .unwrap();
}

#[test]
fn key_range_builder_visit_does_not_emit_fully_tombstoned_rollup() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"bucket-00/file", b"value").unwrap();
    tree.put(b"bucket-01/file", b"value").unwrap();
    assert!(tree.delete(b"bucket-00/file").unwrap());

    let got = collect_key_entry_refs(&tree, b"bucket-", b'/', usize::MAX);
    assert_eq!(got, vec![b"bucket-01/".to_vec()]);
}

#[test]
fn key_range_builder_visit_cache_is_invalidated_by_writes() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"bucket-00/file", b"value").unwrap();

    let first = collect_key_entry_refs(&tree, b"bucket-", b'/', 8);
    assert_eq!(first, vec![b"bucket-00/".to_vec()]);

    tree.put(b"bucket-01/file", b"value").unwrap();
    let second = collect_key_entry_refs(&tree, b"bucket-", b'/', 8);
    assert_eq!(second, vec![b"bucket-00/".to_vec(), b"bucket-01/".to_vec()]);
}

#[test]
fn key_range_builder_visit_cache_includes_start_after() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..3u32 {
        tree.put(format!("bucket-{i:02}/file").as_bytes(), b"value")
            .unwrap();
    }

    let all = collect_key_entry_refs(&tree, b"bucket-", b'/', 8);
    assert_eq!(
        all,
        vec![
            b"bucket-00/".to_vec(),
            b"bucket-01/".to_vec(),
            b"bucket-02/".to_vec(),
        ]
    );

    let mut after = Vec::new();
    tree.scan_keys(b"bucket-")
        .start_after(b"bucket-00/file")
        .delimiter(b'/')
        .visit(8, |entry| {
            match entry {
                KeyRangeEntryRef::Key { key, .. } => after.push(key.to_vec()),
                KeyRangeEntryRef::CommonPrefix(prefix) => after.push(prefix.to_vec()),
                _ => panic!("KeyRangeEntryRef got a new variant"),
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(after, vec![b"bucket-01/".to_vec(), b"bucket-02/".to_vec()]);
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
fn range_prefix_start_after_before_prefix_seeks_to_prefix() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [
        &b"doc/readme"[..],
        b"img/01",
        b"img/02",
        b"img/03",
        b"video/01",
    ] {
        tree.put(k, b"v").unwrap();
    }

    let got = collect_keys(tree.range().prefix(b"img/").start_after(b"aaa"));
    assert_eq!(
        got,
        vec![b"img/01".to_vec(), b"img/02".to_vec(), b"img/03".to_vec()],
    );
}

#[test]
fn range_prefix_start_after_past_prefix_successor_is_empty() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [&b"img/01"[..], b"img/02", b"img/sub/file", b"video/01"] {
        tree.put(k, b"v").unwrap();
    }

    let at_successor: Vec<_> = tree
        .range()
        .prefix(b"img/")
        .start_after(b"img0")
        .into_iter()
        .collect();
    assert!(at_successor.is_empty());

    let after_successor: Vec<_> = tree
        .range()
        .prefix(b"img/")
        .start_after(b"video/")
        .into_iter()
        .collect();
    assert!(after_successor.is_empty());
}

#[test]
fn range_prefix_start_after_inside_gap_resumes_at_next_key() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [&b"img/01"[..], b"img/02", b"img/03", b"img/04", b"video/1"] {
        tree.put(k, b"v").unwrap();
    }

    let got = collect_keys(tree.range().prefix(b"img/").start_after(b"img/02a"));
    assert_eq!(got, vec![b"img/03".to_vec(), b"img/04".to_vec()]);
}

#[test]
fn range_restarts_after_interleaved_insert_without_missing_key() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for k in [b"k00".as_slice(), b"k02", b"k03"] {
        tree.put(k, b"v").unwrap();
    }

    let mut iter = tree.range().into_iter();
    let first = match iter.next().unwrap().unwrap() {
        RangeEntry::Key { key, .. } => key,
        other => panic!("unexpected first range entry: {other:?}"),
    };
    assert_eq!(first, b"k00".to_vec());

    // Mutates the same blob after the iterator has a live cursor.
    // The next step must invalidate the old path and restart from
    // the last emitted key, otherwise a newly inserted key between
    // k00 and k02 can be skipped.
    let restarts_before = tree.stats().unwrap().bm_range_restarts;
    tree.put(b"k01", b"v").unwrap();

    let rest = collect_keys(iter);
    assert_eq!(
        rest,
        vec![b"k01".to_vec(), b"k02".to_vec(), b"k03".to_vec()]
    );
    assert!(tree.stats().unwrap().bm_range_restarts > restarts_before);
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
fn range_delimiter_restart_does_not_skip_new_rollup() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"bucket-00/path/file.bin", b"v").unwrap();
    tree.put(b"bucket-02/path/file.bin", b"v").unwrap();

    let mut iter = tree.range().prefix(b"bucket-").delimiter(b'/').into_iter();
    let first = match iter.next().unwrap().unwrap() {
        RangeEntry::CommonPrefix(p) => p,
        other => panic!("unexpected first range entry: {other:?}"),
    };
    assert_eq!(first, b"bucket-00/".to_vec());

    // Insert a rollup that sorts after the emitted prefix but
    // before the iterator's old cursor. Restart-on-conflict must
    // rebuild from the prefix successor of bucket-00/ and still
    // surface bucket-01/.
    let restarts_before = tree.stats().unwrap().bm_range_restarts;
    tree.put(b"bucket-01/path/file.bin", b"v").unwrap();

    let rest = collect_keys(iter);
    assert_eq!(rest, vec![b"bucket-01/".to_vec(), b"bucket-02/".to_vec()]);
    assert!(tree.stats().unwrap().bm_range_restarts > restarts_before);
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
fn range_start_after_seeks_across_blob_crossings() {
    let tree = TreeBuilder::new("ignored")
        .memory()
        .buffer_pool_size(16)
        .open()
        .unwrap();
    let big = vec![0xCDu8; 4 * 1024];
    for i in 0..256u32 {
        tree.put(format!("k{i:08}").as_bytes(), &big).unwrap();
    }
    assert!(
        tree.stats().unwrap().blob_count >= 2,
        "workload must spill into multiple blobs",
    );

    let got = collect_keys(tree.range().start_after(b"k00000127"));
    let expected: Vec<Vec<u8>> = (128..256u32)
        .map(|i| format!("k{i:08}").into_bytes())
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn range_keys_walks_across_blob_crossings() {
    let tree = TreeBuilder::new("ignored")
        .memory()
        .buffer_pool_size(16)
        .open()
        .unwrap();
    let big = vec![0xEFu8; 4 * 1024];
    for i in 0..256u32 {
        tree.put(format!("k{i:08}").as_bytes(), &big).unwrap();
    }
    assert!(
        tree.stats().unwrap().blob_count >= 2,
        "workload must spill into multiple blobs",
    );

    let got = collect_key_range(tree.range_keys().start_after(b"k00000127"));
    let expected: Vec<Vec<u8>> = (128..256u32)
        .map(|i| format!("k{i:08}").into_bytes())
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn range_key_versions_match_point_lookup_across_blob_crossings() {
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

    for entry in tree.range().start_after(b"k00000127").into_iter().take(16) {
        match entry.unwrap() {
            RangeEntry::Key {
                key,
                value,
                version,
            } => {
                assert_eq!(value.len(), big.len());
                let point = tree.get_record(&key).unwrap().unwrap();
                assert_eq!(point.version, version);
                assert_eq!(point.value, value);
            }
            other => panic!("unexpected range entry: {other:?}"),
        }
    }
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
    // across multiple blobs; eventually a child blob OOMs and the
    // lock-coupled insert path runs spillover_blob + compact_blob
    // on that child. The retry must re-enter via the child blob's
    // freshly-rewritten `header.root_slot`, not through any
    // parent-stored entry slot.
    use rand::{rngs::StdRng, Rng, SeedableRng};
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
