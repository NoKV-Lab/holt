//! End-to-end tests for copy-on-write [`Tree::snapshot`].
//!
//! Exercises only the public surface. Stage 3 covers snapshot
//! creation, the scoped read path (including across blob-frame
//! boundaries), epoch advancement, and isolation from *root-local*
//! live writes — which hold without fork-on-write because the live
//! root frame is never shared (a snapshot takes a full copy of it).
//! Multi-blob isolation under mutation (the fork-on-write gate) is
//! added alongside that machinery.

use std::sync::Arc;

use holt::{BlobStore, Durability, Error, MemoryBlobStore, Tree, TreeBuilder, TreeConfig, DB};
use tempfile::tempdir;

#[test]
fn snapshot_isolates_root_local_writes() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..5u32 {
        tree.put(format!("k{i}").as_bytes(), format!("v{i}").as_bytes())
            .unwrap();
    }

    let snap = tree.snapshot(b"").unwrap();

    // Mutate the live tree after the snapshot. Both writes stay inside
    // the single root frame, which the snapshot copied — so the
    // snapshot must not observe either.
    tree.put(b"k0", b"OVERWRITTEN").unwrap();
    tree.put(b"k9", b"new").unwrap();

    assert_eq!(snap.get(b"k0").unwrap().as_deref(), Some(&b"v0"[..]));
    assert_eq!(snap.get(b"k9").unwrap(), None);
    for i in 1..5u32 {
        assert_eq!(
            snap.get(format!("k{i}").as_bytes()).unwrap().as_deref(),
            Some(format!("v{i}").as_bytes()),
        );
    }

    // The live tree reflects the new writes.
    assert_eq!(
        tree.get(b"k0").unwrap().as_deref(),
        Some(&b"OVERWRITTEN"[..]),
    );
    assert_eq!(tree.get(b"k9").unwrap().as_deref(), Some(&b"new"[..]));
}

#[test]
fn snapshot_reads_across_blob_boundaries() {
    // Enough keys to force auto-spillover into child blob frames, so
    // the snapshot's copied root crosses `BlobNode`s into shared child
    // frames on read.
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();

    const N: u32 = 5000;
    let value = vec![0xAB_u8; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &value).unwrap();
    }
    assert!(
        store.list_blobs().unwrap().len() >= 2,
        "test needs a multi-blob tree to be meaningful",
    );

    let snap = tree.snapshot(b"").unwrap();
    for i in 0..N {
        assert_eq!(
            snap.get(format!("k{i:08}").as_bytes()).unwrap().as_deref(),
            Some(&value[..]),
            "snapshot lost key {i} across a blob-frame boundary",
        );
    }
}

#[test]
fn snapshot_scope_restricts_reads() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"users/alice", b"1").unwrap();
    tree.put(b"users/bob", b"2").unwrap();
    tree.put(b"orders/x", b"9").unwrap();

    let snap = tree.snapshot(b"users/").unwrap();
    assert_eq!(
        snap.get(b"users/alice").unwrap().as_deref(),
        Some(&b"1"[..])
    );
    assert_eq!(snap.scope(), b"users/");

    let err = snap.get(b"orders/x").unwrap_err();
    assert!(
        matches!(err, Error::OutsideViewScope { .. }),
        "out-of-scope read should be rejected, got {err:?}",
    );
}

#[test]
fn snapshot_epochs_advance_and_retire() {
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    tree.put(b"a", b"1").unwrap();

    let s1 = tree.snapshot(b"").unwrap();
    let e1 = s1.epoch();
    let s2 = tree.snapshot(b"").unwrap();
    let e2 = s2.epoch();
    assert!(e2 > e1, "epochs must advance: {e1} then {e2}");

    s1.retire();
    drop(s2);

    // A fresh snapshot after all prior ones retire still advances the
    // monotonic epoch.
    let s3 = tree.snapshot(b"").unwrap();
    assert!(s3.epoch() > e2, "epoch must keep advancing past {e2}");
}

#[test]
fn snapshot_isolates_cross_blob_writes() {
    // The fork-on-write correctness gate: live writes that descend into
    // frames the snapshot still references must fork those frames, not
    // overwrite them. Uses a multi-blob tree so the writes cross into
    // shared child frames.
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();

    const N: u32 = 5000;
    let orig = vec![0xAB_u8; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &orig).unwrap();
    }
    assert!(
        store.list_blobs().unwrap().len() >= 2,
        "test needs a multi-blob tree",
    );

    let snap = tree.snapshot(b"").unwrap();

    // Mutations that must fork shared child frames: a spread of
    // different-size overwrites (forces leaf realloc), fresh inserts,
    // and a spread of deletes.
    for i in (0..N).step_by(4) {
        tree.put(format!("k{i:08}").as_bytes(), b"UPDATED").unwrap();
    }
    for i in N..N + 100 {
        tree.put(format!("k{i:08}").as_bytes(), b"brand-new")
            .unwrap();
    }
    for i in (2..N).step_by(7) {
        tree.delete(format!("k{i:08}").as_bytes()).unwrap();
    }

    // Snapshot unchanged: every original key still maps to the original
    // value, and no post-snapshot insert is visible.
    for i in 0..N {
        assert_eq!(
            snap.get(format!("k{i:08}").as_bytes()).unwrap().as_deref(),
            Some(&orig[..]),
            "snapshot key {i} changed under a live cross-blob write",
        );
    }
    for i in N..N + 100 {
        assert_eq!(snap.get(format!("k{i:08}").as_bytes()).unwrap(), None);
    }

    // Live tree reflects every mutation. Delete ran last, so a key that
    // was both updated and deleted ends up absent.
    for i in 0..N {
        let k = format!("k{i:08}");
        let live = tree.get(k.as_bytes()).unwrap();
        if i >= 2 && (i - 2) % 7 == 0 {
            assert_eq!(live, None, "live key {i} should be deleted");
        } else if i % 4 == 0 {
            assert_eq!(
                live.as_deref(),
                Some(&b"UPDATED"[..]),
                "live key {i} should be updated",
            );
        } else {
            assert_eq!(live.as_deref(), Some(&orig[..]), "live key {i} unchanged");
        }
    }
    for i in N..N + 100 {
        assert_eq!(
            tree.get(format!("k{i:08}").as_bytes()).unwrap().as_deref(),
            Some(&b"brand-new"[..]),
        );
    }
}

#[test]
fn nested_cross_blob_snapshots_each_isolated() {
    // Two overlapping snapshots over a multi-blob tree: each must see
    // its own generation while the live tree advances. Exercises the
    // multi-epoch fork barrier (a frame forked for snapshot 1 becomes a
    // shared frame that snapshot 2 in turn freezes).
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();

    const N: u32 = 5000;
    let v1 = vec![0x01_u8; 200];
    let v2 = vec![0x02_u8; 200];
    let v3 = vec![0x03_u8; 200];

    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &v1).unwrap();
    }
    assert!(store.list_blobs().unwrap().len() >= 2);

    let s1 = tree.snapshot(b"").unwrap();
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &v2).unwrap();
    }
    let s2 = tree.snapshot(b"").unwrap();
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &v3).unwrap();
    }

    for i in 0..N {
        let k = format!("k{i:08}");
        assert_eq!(
            s1.get(k.as_bytes()).unwrap().as_deref(),
            Some(&v1[..]),
            "s1 key {i}",
        );
        assert_eq!(
            s2.get(k.as_bytes()).unwrap().as_deref(),
            Some(&v2[..]),
            "s2 key {i}",
        );
        assert_eq!(
            tree.get(k.as_bytes()).unwrap().as_deref(),
            Some(&v3[..]),
            "live key {i}",
        );
    }
}

#[test]
fn snapshot_stable_under_randomized_churn() {
    use std::collections::HashMap;

    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store)
        .unwrap();

    // Deterministic LCG so the interleaving is reproducible.
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (lcg >> 33) as u32
    };

    // Seed a multi-blob tree and mirror it in a model map.
    let mut live: HashMap<String, Vec<u8>> = HashMap::new();
    for i in 0..1500u32 {
        let k = format!("key{i:06}");
        let v = vec![(i & 0xFF) as u8; 180];
        tree.put(k.as_bytes(), &v).unwrap();
        live.insert(k, v);
    }

    // Freeze the expected snapshot state, then churn the live tree.
    let snap = tree.snapshot(b"").unwrap();
    let frozen = live.clone();

    for _ in 0..6000 {
        // Keys 1500..1800 are never seeded ⇒ post-snapshot inserts.
        let k = format!("key{:06}", next() % 1800);
        if next() % 4 == 0 {
            tree.delete(k.as_bytes()).unwrap();
            live.remove(&k);
        } else {
            let vlen = 1 + (next() % 200) as usize;
            let v = vec![(next() & 0xFF) as u8; vlen];
            tree.put(k.as_bytes(), &v).unwrap();
            live.insert(k, v);
        }
    }

    // The snapshot is frozen at capture time regardless of the churn.
    for (k, v) in &frozen {
        assert_eq!(
            snap.get(k.as_bytes()).unwrap().as_deref(),
            Some(&v[..]),
            "snapshot drifted at {k}",
        );
    }
    for i in 1500..1800u32 {
        let k = format!("key{i:06}");
        assert_eq!(
            snap.get(k.as_bytes()).unwrap(),
            None,
            "snapshot saw post-snapshot key {k}",
        );
    }

    // The live tree matches the model after all churn.
    for (k, v) in &live {
        assert_eq!(
            tree.get(k.as_bytes()).unwrap().as_deref(),
            Some(&v[..]),
            "live tree drifted at {k}",
        );
    }
}

#[test]
fn retire_reclaims_forked_frames() {
    // Retiring a snapshot must free the frames it kept alive (the
    // forked-away originals + the snapshot root), returning the blob
    // count to the live working set — no leak.
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();

    const N: u32 = 5000;
    let orig = vec![0xAB_u8; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &orig).unwrap();
    }
    tree.checkpoint().unwrap();
    let baseline = store.list_blobs().unwrap().len();
    assert!(baseline >= 2, "need a multi-blob tree");

    {
        let snap = tree.snapshot(b"").unwrap();
        // Overwrite a spread of keys → forks the shared child frames
        // (same key set, smaller value, so no spillover: forks are 1:1
        // replacements of the originals).
        for i in (0..N).step_by(3) {
            tree.put(format!("k{i:08}").as_bytes(), b"x").unwrap();
        }
        tree.checkpoint().unwrap();
        let during = store.list_blobs().unwrap().len();
        assert!(
            during > baseline,
            "snapshot + forks should add blobs: {during} vs {baseline}",
        );
        assert_eq!(snap.get(b"k00000000").unwrap().as_deref(), Some(&orig[..]));
    } // snapshot dropped → retire → reclaim

    tree.checkpoint().unwrap();
    let after = store.list_blobs().unwrap().len();
    assert_eq!(
        after, baseline,
        "retire must reclaim every snapshot frame: {after} vs {baseline}",
    );

    // Live tree intact.
    for i in 0..N {
        let want: &[u8] = if i % 3 == 0 { b"x" } else { &orig };
        assert_eq!(
            tree.get(format!("k{i:08}").as_bytes()).unwrap().as_deref(),
            Some(want),
            "live key {i}",
        );
    }
}

#[test]
fn overlapping_snapshots_reclaim_after_last_retires() {
    // Two overlapping snapshots accumulate forked-away frames; the full
    // working set is reclaimed once the last one retires.
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();

    const N: u32 = 5000;
    let v = vec![0xAB_u8; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &v).unwrap();
    }
    tree.checkpoint().unwrap();
    let baseline = store.list_blobs().unwrap().len();
    assert!(baseline >= 2);

    let s1 = tree.snapshot(b"").unwrap();
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), b"a").unwrap();
    }
    let s2 = tree.snapshot(b"").unwrap();
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), b"b").unwrap();
    }
    tree.checkpoint().unwrap();
    assert!(store.list_blobs().unwrap().len() > baseline);

    // Retiring the older snapshot first, then the newer one.
    drop(s1);
    tree.checkpoint().unwrap();
    drop(s2);
    tree.checkpoint().unwrap();

    let after = store.list_blobs().unwrap().len();
    assert_eq!(
        after, baseline,
        "all snapshot frames reclaimed after the last retire: {after} vs {baseline}",
    );
    for i in 0..N {
        assert_eq!(
            tree.get(format!("k{i:08}").as_bytes()).unwrap().as_deref(),
            Some(&b"b"[..]),
            "live key {i}",
        );
    }
}

#[test]
fn snapshot_correct_after_reopen() {
    let dir = tempdir().unwrap();
    let cfg = || {
        let mut c = TreeConfig::new(dir.path());
        c.checkpoint.enabled = false;
        c.durability = Durability::Wal { sync: true };
        c
    };

    const N: u32 = 5000;
    let v1 = vec![0x01_u8; 200];
    let v2 = vec![0x02_u8; 200];
    let v3 = vec![0x03_u8; 200];

    // Session 1: write, then snapshot + fork + retire so the live child
    // frames end up with a created_epoch above 1, and checkpoint so they
    // persist into blobs.dat (not just the WAL — replay would re-stamp
    // them at epoch 1 and hide the bug).
    {
        let tree = Tree::open(cfg()).unwrap();
        for i in 0..N {
            tree.put(format!("k{i:06}").as_bytes(), &v1).unwrap();
        }
        {
            let snap = tree.snapshot(b"").unwrap();
            for i in 0..N {
                tree.put(format!("k{i:06}").as_bytes(), &v2).unwrap();
            }
            assert_eq!(snap.get(b"k000000").unwrap().as_deref(), Some(&v1[..]));
        } // retire
        tree.checkpoint().unwrap();
    }

    // Reopen.
    let tree = Tree::open(cfg()).unwrap();

    // Live data survives the reopen (forks included).
    for i in 0..N {
        assert_eq!(
            tree.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&v2[..]),
            "reopened live key {i}",
        );
    }

    // A NEW snapshot after reopen must isolate. If current_epoch reset to
    // 1 while the loaded frames carry created_epoch > 1, the walker would
    // wrongly treat them as private and overwrite them in place, leaking
    // v3 into the snapshot.
    let snap = tree.snapshot(b"").unwrap();
    for i in 0..N {
        tree.put(format!("k{i:06}").as_bytes(), &v3).unwrap();
    }
    for i in 0..N {
        assert_eq!(
            snap.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&v2[..]),
            "post-reopen snapshot key {i} was corrupted by a live write",
        );
        assert_eq!(
            tree.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&v3[..]),
            "post-reopen live key {i}",
        );
    }
}

/// Environment variable carrying the store directory into the
/// crash-session child processes below.
const CRASH_LEAK_DIR_ENV: &str = "HOLT_CRASH_LEAK_DIR";
/// Keys written by each crash session.
const CRASH_LEAK_N: u32 = 5000;

fn crash_leak_value() -> Vec<u8> {
    vec![0xAB_u8; 200]
}

fn crash_leak_cfg(dir: &std::path::Path) -> TreeConfig {
    let mut c = TreeConfig::new(dir);
    c.checkpoint.enabled = false;
    c.durability = Durability::Wal { sync: true };
    c
}

/// Run the named `#[ignore]` child test in a separate process.
///
/// The crash-leak tests simulate a crash with `mem::forget(snap)`.
/// Inside a single process that keeps the leaked instance — and its
/// exclusive store-directory lock — alive forever; a real crash ends
/// the process and the kernel drops the flock with it. A child
/// process reproduces the real semantics: leaked on-disk frames,
/// released lock.
fn run_crash_session(child_test: &str, dir: &std::path::Path) {
    let exe = std::env::current_exe().unwrap();
    let status = std::process::Command::new(exe)
        .args([child_test, "--exact", "--ignored", "--nocapture"])
        .env(CRASH_LEAK_DIR_ENV, dir)
        .status()
        .unwrap();
    assert!(status.success(), "crash-session child {child_test} failed");
}

/// Child body for [`gc_reclaims_crash_leaked_snapshot_frames`]:
/// snapshot + fork, checkpoint so the forks/orphans/snapshot root
/// persist, then "crash" — forget the snapshot so it never retires
/// and its in-memory orphan list dies with the process.
#[test]
#[ignore = "child-process body for gc_reclaims_crash_leaked_snapshot_frames"]
fn crash_leak_tree_session() {
    let Some(dir) = std::env::var_os(CRASH_LEAK_DIR_ENV) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let v = crash_leak_value();
    let tree = Tree::open(crash_leak_cfg(&dir)).unwrap();
    for i in 0..CRASH_LEAK_N {
        tree.put(format!("k{i:06}").as_bytes(), &v).unwrap();
    }
    tree.checkpoint().unwrap();
    let snap = tree.snapshot(b"").unwrap();
    for i in 0..CRASH_LEAK_N {
        tree.put(format!("k{i:06}").as_bytes(), b"new").unwrap();
    }
    tree.checkpoint().unwrap();
    std::mem::forget(snap);
}

#[test]
fn gc_reclaims_crash_leaked_snapshot_frames() {
    let dir = tempdir().unwrap();
    let cfg = || crash_leak_cfg(dir.path());

    const N: u32 = CRASH_LEAK_N;

    run_crash_session("crash_leak_tree_session", dir.path());

    // Reopen: the store still carries the leaked orphan frames + the
    // forgotten snapshot's root, unreachable from the live tree.
    let tree = Tree::open(cfg()).unwrap();
    for i in 0..N {
        assert_eq!(
            tree.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&b"new"[..]),
            "reopened live key {i}",
        );
    }

    let freed = tree.gc().unwrap();
    assert!(
        freed > 0,
        "gc should reclaim crash-leaked snapshot frames, freed {freed}",
    );
    // Idempotent: nothing unreachable remains.
    assert_eq!(tree.gc().unwrap(), 0, "second gc must be a no-op");
    // gc must not have touched live data.
    for i in 0..N {
        assert_eq!(
            tree.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&b"new"[..]),
            "live key {i} after gc",
        );
    }
}

/// Child body for [`db_gc_reclaims_leak_and_preserves_all_trees`]:
/// two trees; snapshot + fork + crash on t1.
#[test]
#[ignore = "child-process body for db_gc_reclaims_leak_and_preserves_all_trees"]
fn crash_leak_db_session() {
    let Some(dir) = std::env::var_os(CRASH_LEAK_DIR_ENV) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let v = crash_leak_value();
    let db = DB::open(crash_leak_cfg(&dir)).unwrap();
    let t1 = db.create_tree("t1").unwrap();
    let t2 = db.create_tree("t2").unwrap();
    for i in 0..CRASH_LEAK_N {
        t1.put(format!("k{i:06}").as_bytes(), &v).unwrap();
        t2.put(format!("k{i:06}").as_bytes(), &v).unwrap();
    }
    db.checkpoint().unwrap();
    let snap = t1.snapshot(b"").unwrap();
    for i in 0..CRASH_LEAK_N {
        t1.put(format!("k{i:06}").as_bytes(), b"new").unwrap();
    }
    db.checkpoint().unwrap();
    std::mem::forget(snap); // crash: t1's forked-away frames leak
}

#[test]
fn db_gc_reclaims_leak_and_preserves_all_trees() {
    let dir = tempdir().unwrap();
    let cfg = || crash_leak_cfg(dir.path());

    const N: u32 = CRASH_LEAK_N;
    let v = crash_leak_value();

    run_crash_session("crash_leak_db_session", dir.path());

    let db = DB::open(cfg()).unwrap();
    let freed = db.gc().unwrap();
    assert!(
        freed > 0,
        "db gc should reclaim crash-leaked frames, freed {freed}",
    );
    assert_eq!(db.gc().unwrap(), 0, "second db gc must be a no-op");

    // gc marked every tree's root, so both trees survive intact.
    let t1 = db.open_tree("t1").unwrap();
    let t2 = db.open_tree("t2").unwrap();
    for i in 0..N {
        assert_eq!(
            t1.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&b"new"[..]),
            "t1 key {i}",
        );
        assert_eq!(
            t2.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&v[..]),
            "t2 key {i}",
        );
    }
}

#[test]
fn gc_rejects_db_trees() {
    let dir = tempdir().unwrap();
    let db = DB::open(TreeConfig::new(dir.path())).unwrap();
    let tree = db.create_tree("t").unwrap();
    assert!(
        matches!(tree.gc(), Err(Error::GcRequiresStandaloneTree)),
        "gc on a DB tree must be rejected",
    );
}
