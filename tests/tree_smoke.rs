//! End-to-end smoke tests driving the public `Tree` API.
//!
//! Exercises only the public surface so signature breakage shows
//! up here first.

use std::sync::Arc;

use artisan::{Backend, MemoryBackend, Tree, TreeBuilder, TreeConfig};

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
        .map(|i| (format!("img/{i:04}.jpg").into_bytes(), format!("blob#{i}").into_bytes()))
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
    let paths: &[&[u8]] = &[
        b"/", b"/a", b"/a/b", b"/a/b/c", b"/a/b/c/d", b"/a/b/c/d/e",
    ];
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
    assert_eq!(tree.get(b"").unwrap().as_deref(), Some(&b"empty-key-value"[..]));
    tree.put(b"a", b"other").unwrap();
    assert_eq!(tree.get(b"").unwrap().as_deref(), Some(&b"empty-key-value"[..]));
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
        .map(|i| (format!("img/{i:03}").into_bytes(), format!("v{i}").into_bytes()))
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
    assert_eq!(tree.delete(b"img/02.jpg").unwrap().as_deref(), Some(&b"b"[..]));
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
    assert!(matches!(r, Err(artisan::Error::NotFound)));
}

#[test]
fn rename_to_existing_dst_without_force_errors_dst_exists() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"a", b"v_a").unwrap();
    tree.put(b"b", b"v_b").unwrap();
    let r = tree.rename(b"a", b"b", false);
    assert!(matches!(r, Err(artisan::Error::DstExists)));
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
    tree.rename(b"img/02.jpg", b"img/02-renamed.jpg", false).unwrap();
    assert_eq!(tree.get(b"img/01.jpg").unwrap().as_deref(), Some(&b"a"[..]));
    assert!(tree.get(b"img/02.jpg").unwrap().is_none());
    assert_eq!(
        tree.get(b"img/02-renamed.jpg").unwrap().as_deref(),
        Some(&b"b"[..]),
    );
    assert_eq!(tree.get(b"img/03.jpg").unwrap().as_deref(), Some(&b"c"[..]));
}
