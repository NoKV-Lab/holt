//! End-to-end smoke tests driving the public `Tree` API.
//!
//! Exercises only the public surface so signature breakage shows
//! up here first.

use std::sync::Arc;

use holt::{AlignedBlobBuf, Backend, MemoryBackend, Tree, TreeBuilder, TreeConfig, TreeStats};

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
    assert!(tree.put(b"hello", b"world").unwrap().is_none());
    assert_eq!(tree.get(b"hello").unwrap().as_deref(), Some(&b"world"[..]));
    assert!(tree.get(b"missing").unwrap().is_none());
}

#[test]
fn put_returns_previous_value_on_update() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.put(b"k", b"v1").unwrap().is_none());
    assert_eq!(tree.put(b"k", b"v2").unwrap().as_deref(), Some(&b"v1"[..]));
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
    assert_eq!(tree.delete(b"k").unwrap().as_deref(), Some(&b"v"[..]));
    assert!(tree.get(b"k").unwrap().is_none());
}

#[test]
fn delete_missing_key_is_noop() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    assert!(tree.delete(b"missing").unwrap().is_none());
}

#[test]
fn delete_then_reinsert_round_trips() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"k", b"v1").unwrap();
    assert_eq!(tree.delete(b"k").unwrap().as_deref(), Some(&b"v1"[..]));
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
        assert_eq!(tree.delete(k).unwrap().as_deref(), Some(&v[..]));
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
        tree.delete(b"img/02.jpg").unwrap().as_deref(),
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
        let prev = tree.delete(format!("k{i:08}").as_bytes()).unwrap();
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
        let prev = tree.delete(&k).unwrap();
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
fn tree_get_follows_blob_node_crossing_across_two_blobs() {
    use holt::engine::make_blob_from_node;
    use holt::layout::{BlobNode, NodeType};
    use holt::store::BlobFrame;

    // Step 1: build a normal tree with some keys.
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    {
        let tree = TreeBuilder::new("ignored")
            .open_with_backend(backend.clone())
            .unwrap();
        for i in 0..10u32 {
            let k = format!("k{i:02}").into_bytes();
            let v = format!("v{i}").into_bytes();
            tree.put(&k, &v).unwrap();
        }
    }

    // Step 2: read root blob; deep-clone its subtree into a fresh
    // child blob.
    let root_guid = [0u8; 16];
    let child_guid = [0xAA; 16];

    let mut root_buf = AlignedBlobBuf::zeroed();
    backend.read_blob(root_guid, &mut root_buf).unwrap();

    let (saved_root_slot, child_outcome) = {
        let root_frame = BlobFrame::wrap(root_buf.as_mut_slice());
        let saved_root = root_frame.header().root_slot;
        let outcome = make_blob_from_node(&root_frame, saved_root, child_guid).unwrap();
        (saved_root, outcome)
    };

    // Step 3: write the child blob through the backend.
    backend.write_blob(child_guid, &child_outcome.buf).unwrap();

    // Step 4: rewrite root blob: allocate a BlobNode at a fresh
    // slot pointing at (child_guid, entry_slot), and re-point
    // header.root_slot there. The old saved_root subtree leaks
    // (its slot entries stay tagged live but unreachable) — fine
    // for this test; production spillover (phase B) will free
    // them via free_node walks.
    {
        let mut root_frame = BlobFrame::wrap(root_buf.as_mut_slice());
        let bn_out = root_frame.alloc_node(NodeType::Blob).unwrap();
        let bn = BlobNode::new(b"", child_guid, u32::from(child_outcome.entry_slot));
        // SAFETY: layout types are #[repr(C)] POD; body has the
        // right size; BlobFrame's bump allocator gives 8-byte
        // alignment.
        let body = root_frame.body_of_slot_mut(bn_out.slot).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                &bn as *const BlobNode as *const u8,
                body.as_mut_ptr(),
                std::mem::size_of::<BlobNode>(),
            );
        }
        root_frame.header_mut().root_slot = bn_out.slot;
        let _ = saved_root_slot; // intentionally orphaned in this test
    }
    backend.write_blob(root_guid, &root_buf).unwrap();

    // Step 5: open a fresh Tree against the same backend; verify
    // every original key is reachable through the BlobNode crossing.
    let tree = TreeBuilder::new("ignored")
        .open_with_backend(backend.clone())
        .unwrap();
    for i in 0..10u32 {
        let k = format!("k{i:02}").into_bytes();
        let v = format!("v{i}").into_bytes();
        assert_eq!(
            tree.get(&k).unwrap().as_deref(),
            Some(&v[..]),
            "post-crossing lookup failed for key {k:?}",
        );
    }
    // Missing keys still NotFound.
    assert!(tree.get(b"k99").unwrap().is_none());
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
