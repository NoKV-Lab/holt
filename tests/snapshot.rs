//! End-to-end tests for copy-on-write [`Tree::snapshot`].
//!
//! Exercises only the public surface. Snapshot tests cover
//! creation, the scoped read path (including across blob-frame
//! boundaries), epoch advancement, and isolation from both root-local
//! and cross-frame live writes. Capture copies the root frame; descendants
//! remain shared until a live mutation validates the parent edge and forks
//! the affected shared frame. Escaped views, builders, and cursors keep the
//! process-local epoch lease alive until the final handle is dropped.

use std::sync::Arc;

use holt::{
    BlobStore, Durability, Error, FileBlobStore, MemoryBlobStore, Tree, TreeBuilder, TreeConfig, DB,
};
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
    let tree = Tree::open_with_blob_store(TreeConfig::memory(), store.clone()).unwrap();

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
    let tree = Tree::open_with_blob_store(TreeConfig::memory(), store.clone()).unwrap();

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
fn retire_defers_forked_frame_reclamation_to_gc() {
    // Retiring a snapshot releases process-local ownership but must keep
    // persisted frames until a durability-barrier-protected GC proves them
    // unreachable. That fail-closed interval prevents a durable old parent
    // from losing a child.
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = Tree::open_with_blob_store(TreeConfig::memory(), store.clone()).unwrap();

    const N: u32 = 5000;
    let orig = vec![0xAB_u8; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &orig).unwrap();
    }
    tree.checkpoint().unwrap();
    let baseline = store.list_blobs().unwrap().len();
    assert!(baseline >= 2, "need a multi-blob tree");

    let during;
    {
        let snap = tree.snapshot(b"").unwrap();
        // Overwrite a spread of keys → forks the shared child frames
        // (same key set, smaller value, so no spillover: forks are 1:1
        // replacements of the originals).
        for i in (0..N).step_by(3) {
            tree.put(format!("k{i:08}").as_bytes(), b"x").unwrap();
        }
        tree.checkpoint().unwrap();
        during = store.list_blobs().unwrap().len();
        assert!(
            during > baseline,
            "snapshot + forks should add blobs: {during} vs {baseline}",
        );
        assert_eq!(snap.get(b"k00000000").unwrap().as_deref(), Some(&orig[..]));
    } // snapshot dropped → process-local retire only

    let after_retire = store.list_blobs().unwrap().len();
    assert_eq!(
        after_retire, during,
        "retire must preserve persisted COW frames until GC",
    );

    let freed = tree.gc().unwrap();
    assert_eq!(freed, during - baseline);
    let after_gc = store.list_blobs().unwrap().len();
    assert_eq!(
        after_gc, baseline,
        "GC must reclaim every retired snapshot frame: {after_gc} vs {baseline}",
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
fn overlapping_snapshots_reclaim_on_gc_after_last_retires() {
    // Two overlapping snapshots accumulate forked-away frames. Retirement
    // releases their process-local holds; durable space is reclaimed by GC.
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = Tree::open_with_blob_store(TreeConfig::memory(), store.clone()).unwrap();

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

    tree.gc().unwrap();
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
fn snapshot_correct_after_multi_blob_compact_checkpoint_reopen() {
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
        assert!(
            tree.stats().unwrap().blob_count > 1,
            "test must exercise compacted child frames",
        );
        {
            let snap = tree.snapshot(b"").unwrap();
            for i in 0..N {
                tree.put(format!("k{i:06}").as_bytes(), &v2).unwrap();
            }
            assert_eq!(snap.get(b"k000000").unwrap().as_deref(), Some(&v1[..]));
        } // retire
        tree.checkpoint().unwrap();

        // Rebuild the high-epoch COW frames before persisting them. If
        // compact_blob resets either generation field, reopen can allocate a
        // snapshot epoch older than a surviving child and a later live write
        // will mutate that child underneath the snapshot.
        let compactions_before = tree.stats().unwrap().total_compactions;
        for _ in 0..4 {
            tree.compact().unwrap();
        }
        let compactions_after = tree.stats().unwrap().total_compactions;
        assert!(
            compactions_after > compactions_before,
            "test must compact at least one high-epoch frame",
        );
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

#[test]
fn repeated_read_views_do_not_grow_file_store() {
    let dir = tempdir().unwrap();
    let blobs = dir.path().join("blobs.dat");
    let cfg = || {
        let mut c = TreeConfig::new(dir.path());
        c.checkpoint.enabled = false;
        c.durability = Durability::Wal { sync: false };
        c
    };

    let tree = Tree::open(cfg()).unwrap();
    for i in 0..128u32 {
        tree.put(format!("k{i:06}").as_bytes(), b"value").unwrap();
    }
    tree.checkpoint().unwrap();
    let before = std::fs::metadata(&blobs).unwrap().len();

    for _ in 0..128 {
        tree.view(b"", |view| {
            assert_eq!(view.get(b"k000000")?.as_deref(), Some(&b"value"[..]));
            Ok(())
        })
        .unwrap();
    }
    tree.checkpoint().unwrap();
    let after = std::fs::metadata(&blobs).unwrap().len();

    assert_eq!(
        after, before,
        "short-lived read views must not allocate persistent blob slots"
    );
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
/// process reproduces the real semantics: the read snapshot is lost,
/// the live tree remains authoritative, and GC must not find
/// snapshot-owned persistent garbage.
fn run_crash_session(child_test: &str, dir: &std::path::Path) {
    let exe = std::env::current_exe().unwrap();
    let status = std::process::Command::new(exe)
        .args([child_test, "--exact", "--ignored", "--nocapture"])
        .env(CRASH_LEAK_DIR_ENV, dir)
        .status()
        .unwrap();
    assert!(status.success(), "crash-session child {child_test} failed");
}

/// Child body for [`crash_leaked_tree_snapshot_does_not_leave_persistent_garbage`]:
/// snapshot + writes + checkpoint, then "crash" — forget the snapshot
/// so it never retires. Snapshot roots are in-memory only, so reopen
/// should see live data; the storage owner explicitly runs recovery GC.
#[test]
#[ignore = "child-process body for crash_leaked_tree_snapshot_does_not_leave_persistent_garbage"]
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
fn crash_leaked_tree_snapshot_does_not_leave_persistent_garbage() {
    let dir = tempdir().unwrap();
    let cfg = || crash_leak_cfg(dir.path());

    const N: u32 = CRASH_LEAK_N;

    run_crash_session("crash_leak_tree_session", dir.path());

    // Reopen itself preserves generic Holt startup semantics and does not run
    // a full store sweep. The forgotten snapshot was not a durable root, so
    // the storage owner's explicit recovery GC can reclaim its old frames.
    let tree = Tree::open(cfg()).unwrap();
    for i in 0..N {
        assert_eq!(
            tree.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&b"new"[..]),
            "reopened live key {i}",
        );
    }

    let freed = tree.gc().unwrap();
    assert!(freed > 0, "explicit recovery gc must reclaim crash orphans");
    // Idempotent: GC remains a no-op.
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

/// Child body for [`db_crash_leaked_snapshot_does_not_leave_persistent_garbage`]:
/// two trees; snapshot + writes + crash on t1.
#[test]
#[ignore = "child-process body for db_crash_leaked_snapshot_does_not_leave_persistent_garbage"]
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
    for i in [0, CRASH_LEAK_N / 2, CRASH_LEAK_N - 1] {
        assert_eq!(
            snap.get(format!("k{i:06}").as_bytes()).unwrap().as_deref(),
            Some(&v[..]),
            "deferred batch flush overwrote snapshot child {i}",
        );
    }
    assert!(
        db.stats().bm_gc_orphan_backlog_count > 0,
        "snapshot-shared batch updates must create COW orphan debt",
    );
    std::mem::forget(snap); // crash: read snapshot state is process-local
}

#[test]
fn db_crash_leaked_snapshot_does_not_leave_persistent_garbage() {
    let dir = tempdir().unwrap();
    let cfg = || crash_leak_cfg(dir.path());

    const N: u32 = CRASH_LEAK_N;
    let v = crash_leak_value();

    run_crash_session("crash_leak_db_session", dir.path());

    let db = DB::open(cfg()).unwrap();
    let freed = db.gc().unwrap();
    assert!(
        freed > 0,
        "explicit DB recovery gc must reclaim crash orphans",
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
fn db_reopen_recovers_epoch_from_surviving_other_family_frames() {
    let dir = tempdir().unwrap();
    let cfg = || crash_leak_cfg(dir.path());
    let keys = (400..528u32)
        .map(|i| format!("epoch/{i:08}").into_bytes())
        .collect::<Vec<_>>();

    {
        let db = DB::open(cfg()).unwrap();
        let data = db.create_tree("data").unwrap();
        let epoch_owner = db.create_tree("epoch-owner").unwrap();
        epoch_owner.put(b"anchor", b"owner").unwrap();
        let old = vec![0x31; 1024];
        for i in 0..1200u32 {
            data.put(format!("epoch/{i:08}").as_bytes(), &old).unwrap();
        }
        db.checkpoint().unwrap();
        assert!(data.stats().unwrap().blob_count > 1);

        // This snapshot advances the DB-global epoch on another family. Data
        // then forks high-epoch children while its own root high-water can
        // remain older.
        let owner_snapshot = epoch_owner.snapshot(b"").unwrap();
        for key in &keys {
            data.put(key, b"session-one").unwrap();
        }
        db.checkpoint().unwrap();
        assert!(db.stats().bm_gc_orphan_backlog_count > 0);
        drop(owner_snapshot);
        db.checkpoint().unwrap();

        // Remove the root that originally carried the maximum high-water.
        // Reopen must recover from the created_epoch on data's surviving
        // reachable frames rather than assuming that root still exists.
        db.drop_tree("epoch-owner").unwrap();
        drop(epoch_owner);
        db.gc().unwrap();
        db.checkpoint().unwrap();
        assert_eq!(db.list_trees().unwrap(), vec!["data"]);
    }

    let db = DB::open(cfg()).unwrap();
    let data = db.open_tree("data").unwrap();
    db.view(&[("data", b"")], |view| {
        for key in &keys {
            data.put(key, b"session-two")?;
        }
        db.checkpoint()?;

        let stable = view.tree("data").expect("captured data family");
        for key in &keys {
            assert_eq!(stable.get(key)?.as_deref(), Some(&b"session-one"[..]));
            assert_eq!(data.get(key)?.as_deref(), Some(&b"session-two"[..]));
        }
        assert!(
            db.stats().bm_gc_orphan_backlog_count > 0,
            "session-two writes must COW the reopened snapshot-shared child",
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn db_epoch_recovery_accepts_wal_replayed_cache_only_roots() {
    let dir = tempdir().unwrap();
    let cfg = || crash_leak_cfg(dir.path());

    {
        let db = DB::open(cfg()).unwrap();
        let tree = db.create_tree("wal-only").unwrap();
        tree.put(b"key", b"acked-before-checkpoint").unwrap();
        // Deliberately no checkpoint: catalog/root/data exist only through
        // the durable WAL plus replayed BM cache in the next session.
    }

    let db = DB::open(cfg()).unwrap();
    let tree = db.open_tree("wal-only").unwrap();
    assert_eq!(
        tree.get(b"key").unwrap().as_deref(),
        Some(&b"acked-before-checkpoint"[..])
    );
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

#[test]
fn db_gc_finishes_dropping_tree_protocol_before_global_sweep() {
    let dir = tempdir().unwrap();
    let cfg = || crash_leak_cfg(dir.path());

    {
        let db = DB::open(cfg()).unwrap();
        let live = db.create_tree("live").unwrap();
        let doomed = db.create_tree("doomed").unwrap();
        for i in 0..512u32 {
            live.put(format!("live/{i:04}").as_bytes(), b"kept")
                .unwrap();
            doomed
                .put(format!("doomed/{i:04}").as_bytes(), &[0xDD; 256])
                .unwrap();
        }
        db.checkpoint().unwrap();
        db.drop_tree("doomed").unwrap();
        drop(doomed);
        drop(live);

        db.gc().unwrap();
        assert_eq!(db.list_trees().unwrap(), vec!["live"]);
        assert!(matches!(
            db.open_tree("doomed"),
            Err(Error::TreeNotFound { .. })
        ));
    }

    let db = DB::open(cfg()).unwrap();
    assert_eq!(db.list_trees().unwrap(), vec!["live"]);
    assert!(matches!(
        db.open_tree("doomed"),
        Err(Error::TreeNotFound { .. })
    ));
    let live = db.open_tree("live").unwrap();
    for i in 0..512u32 {
        assert_eq!(
            live.get(format!("live/{i:04}").as_bytes())
                .unwrap()
                .as_deref(),
            Some(&b"kept"[..]),
        );
    }
}

#[test]
fn retained_dropped_tree_handle_does_not_block_checkpoint_or_gc() {
    use std::sync::mpsc;
    use std::time::Duration;

    let dir = tempdir().unwrap();
    let db = DB::open(crash_leak_cfg(dir.path())).unwrap();
    let doomed = db.create_tree("doomed").unwrap();
    for i in 0..2600u32 {
        doomed
            .put(format!("doomed/{i:08}").as_bytes(), &[0xD1; 240])
            .unwrap();
    }
    db.checkpoint().unwrap();
    db.drop_tree("doomed").unwrap();
    assert!(matches!(
        db.create_tree("doomed"),
        Err(Error::TreeExists { .. })
    ));
    assert!(matches!(
        doomed.get(b"doomed/00000000"),
        Err(Error::TreeDropped)
    ));

    let run_with_timeout = |label: &'static str, op: fn(&DB) -> holt::Result<usize>| {
        let worker_db = db.clone();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || tx.send(op(&worker_db)).unwrap());
        let result = rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap_or_else(|_| panic!("{label} did not return with a retained dropped handle"))
            .unwrap();
        worker.join().unwrap();
        result
    };

    let checkpoint = |db: &DB| db.checkpoint().map(|()| 0);
    assert_eq!(run_with_timeout("checkpoint", checkpoint), 0);
    assert_eq!(run_with_timeout("gc", DB::gc), 0);
    assert!(matches!(
        db.open_tree("doomed"),
        Err(Error::TreeNotFound { .. })
    ));

    drop(doomed);
    assert!(db.gc().unwrap() > 0);
    let replacement = db.create_tree("doomed").unwrap();
    assert_eq!(replacement.get(b"doomed/00000000").unwrap(), None);
    assert!(db.stats().bm_pending_delete_count == 0);
}

#[test]
fn dropped_tree_escaped_view_and_cursor_keep_descendants_live() {
    let dir = tempdir().unwrap();
    let db = DB::open(crash_leak_cfg(dir.path())).unwrap();
    let doomed = db.create_tree("doomed").unwrap();
    let original = vec![0x6D; 240];
    for i in 0..ESCAPED_READER_N {
        doomed
            .put(format!("k{i:08}").as_bytes(), &original)
            .unwrap();
    }
    db.checkpoint().unwrap();

    let snapshot = doomed.snapshot(b"").unwrap();
    let escaped_view = snapshot.view().clone();
    let mut escaped_cursor = snapshot.range().into_iter();
    drop(snapshot);
    db.drop_tree("doomed").unwrap();
    drop(doomed);

    db.checkpoint().unwrap();
    assert_eq!(db.gc().unwrap(), 0);
    assert_eq!(
        escaped_view.get(b"k00000000").unwrap().as_deref(),
        Some(&original[..])
    );
    let mut seen = 0u32;
    for entry in &mut escaped_cursor {
        match entry.unwrap() {
            holt::RangeEntry::Key { key, value, .. } => {
                assert_eq!(key, format!("k{seen:08}").into_bytes());
                assert_eq!(value, original);
                seen += 1;
            }
            other => panic!("unexpected escaped range entry: {other:?}"),
        }
    }
    assert_eq!(seen, ESCAPED_READER_N);

    drop(escaped_cursor);
    drop(escaped_view);
    assert!(db.gc().unwrap() > 0);
    assert!(matches!(
        db.open_tree("doomed"),
        Err(Error::TreeNotFound { .. })
    ));
}

#[test]
fn live_range_has_no_gap_or_duplicate_across_gc_epochs() {
    use std::sync::mpsc;

    let tree = Tree::open(TreeConfig::memory()).unwrap();
    for i in 0..ESCAPED_READER_N {
        tree.put(format!("k{i:08}").as_bytes(), &[0x7A; 240])
            .unwrap();
    }
    tree.checkpoint().unwrap();

    let scan_tree = tree.clone();
    let (first_tx, first_rx) = mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = mpsc::sync_channel(0);
    let scanner = std::thread::spawn(move || {
        let mut cursor = scan_tree.range().into_iter();
        let mut keys = Vec::new();
        match cursor.next().expect("first range entry").unwrap() {
            holt::RangeEntry::Key { key, .. } => keys.push(key),
            other => panic!("unexpected first live range entry: {other:?}"),
        }
        first_tx.send(()).unwrap();
        resume_rx.recv().unwrap();
        for entry in &mut cursor {
            match entry.unwrap() {
                holt::RangeEntry::Key { key, .. } => keys.push(key),
                other => panic!("unexpected live range entry: {other:?}"),
            }
        }
        (keys, cursor.stats())
    });

    first_rx.recv().unwrap();
    tree.gc().unwrap();
    resume_tx.send(()).unwrap();
    let (keys, stats) = scanner.join().unwrap();
    let expected = (0..ESCAPED_READER_N)
        .map(|i| format!("k{i:08}").into_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        keys, expected,
        "GC restart skipped or duplicated a range key"
    );
    assert!(stats.restarts > 0, "test did not cross a physical GC epoch");
}

const ESCAPED_READER_N: u32 = 2600;

fn escaped_reader_tree() -> (Arc<dyn BlobStore>, Tree, Vec<u8>) {
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let tree = Tree::open_with_blob_store(TreeConfig::memory(), store.clone()).unwrap();
    let original = vec![0x5A_u8; 240];
    for i in 0..ESCAPED_READER_N {
        tree.put(format!("k{i:08}").as_bytes(), &original).unwrap();
    }
    tree.checkpoint().unwrap();
    assert!(
        store.list_blobs().unwrap().len() >= 2,
        "escaped-reader tests require child blobs",
    );
    (store, tree, original)
}

fn mutate_escaped_reader_tree(tree: &Tree) {
    for i in (0..ESCAPED_READER_N).step_by(3) {
        tree.put(format!("k{i:08}").as_bytes(), b"live-generation")
            .unwrap();
    }
    tree.checkpoint().unwrap();
}

#[test]
fn escaped_view_keeps_epoch_and_descendants_live_through_gc() {
    let (_store, tree, original) = escaped_reader_tree();
    let snapshot = tree.snapshot(b"").unwrap();
    let escaped = snapshot.view().clone();
    drop(snapshot);

    mutate_escaped_reader_tree(&tree);
    tree.gc().unwrap();

    for i in 0..ESCAPED_READER_N {
        assert_eq!(
            escaped
                .get(format!("k{i:08}").as_bytes())
                .unwrap()
                .as_deref(),
            Some(&original[..]),
            "escaped view lost snapshot key {i}",
        );
    }

    drop(escaped);
    assert!(
        tree.gc().unwrap() > 0,
        "last escaped view drop must make retired COW frames reclaimable",
    );
}

#[test]
fn escaped_record_cursor_keeps_epoch_and_descendants_live_through_gc() {
    let (_store, tree, original) = escaped_reader_tree();
    let snapshot = tree.snapshot(b"").unwrap();
    let mut cursor = snapshot.range().into_iter();
    drop(snapshot);

    mutate_escaped_reader_tree(&tree);
    tree.gc().unwrap();

    let mut seen = 0u32;
    for entry in &mut cursor {
        match entry.unwrap() {
            holt::RangeEntry::Key { value, .. } => {
                assert_eq!(value, original);
                seen += 1;
            }
            holt::RangeEntry::CommonPrefix(_) => panic!("unexpected delimiter rollup"),
            _ => panic!("unexpected range entry variant"),
        }
    }
    assert_eq!(seen, ESCAPED_READER_N);

    drop(cursor);
    assert!(tree.gc().unwrap() > 0);
}

#[test]
fn escaped_key_cursor_keeps_epoch_and_descendants_live_through_gc() {
    let (_store, tree, _original) = escaped_reader_tree();
    let snapshot = tree.snapshot(b"").unwrap();
    let mut cursor = snapshot.range_keys().into_iter();
    drop(snapshot);

    mutate_escaped_reader_tree(&tree);
    tree.gc().unwrap();

    let mut seen = 0u32;
    for entry in &mut cursor {
        match entry.unwrap() {
            holt::KeyRangeEntry::Key { key, .. } => {
                assert_eq!(key, format!("k{seen:08}").into_bytes());
                seen += 1;
            }
            holt::KeyRangeEntry::CommonPrefix(_) => panic!("unexpected delimiter rollup"),
            _ => panic!("unexpected key-range entry variant"),
        }
    }
    assert_eq!(seen, ESCAPED_READER_N);

    drop(cursor);
    assert!(tree.gc().unwrap() > 0);
}

/// Store directory used by the crash-session child below.
const COW_RETIRE_CRASH_DIR_ENV: &str = "HOLT_COW_RETIRE_CRASH_DIR";

fn cow_retire_crash_cfg(dir: &std::path::Path) -> TreeConfig {
    let mut cfg = TreeConfig::new(dir);
    cfg.checkpoint.enabled = false;
    cfg.memory_flush_on_write = false;
    cfg
}

/// Create a durable multi-blob tree, fork its child frames under a snapshot,
/// retire that snapshot, then durably flush only the underlying store
/// manifest. The live parent and its replacement children deliberately stay
/// in the BufferManager's dirty cache. `process::exit` models a crash by
/// skipping every Rust destructor and therefore every orderly Tree shutdown.
#[test]
#[ignore = "child-process body for cow_retire_keeps_durable_parent_children_reopenable"]
fn cow_retire_before_parent_checkpoint_crash_session() {
    let Some(dir) = std::env::var_os(COW_RETIRE_CRASH_DIR_ENV) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let store = Arc::new(FileBlobStore::open(&dir).unwrap());
    let tree = Tree::open_with_blob_store(cow_retire_crash_cfg(&dir), store.clone()).unwrap();

    const N: u32 = 5000;
    let original = [0xAB_u8; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &original).unwrap();
    }
    tree.checkpoint().unwrap();
    assert!(
        store.list_blobs().unwrap().len() >= 2,
        "test requires a durable parent with child blobs",
    );

    let snapshot = tree.snapshot(b"").unwrap();
    for i in (0..N).step_by(3) {
        tree.put(format!("k{i:08}").as_bytes(), b"new").unwrap();
    }
    drop(snapshot);

    // Persist any manifest mutation issued by snapshot retirement without
    // checkpointing the dirty live parent or its replacement children.
    store.flush().unwrap();
    std::process::exit(0);
}

#[test]
fn cow_retire_keeps_durable_parent_children_reopenable() {
    let dir = tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();
    let status = std::process::Command::new(exe)
        .args([
            "cow_retire_before_parent_checkpoint_crash_session",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env(COW_RETIRE_CRASH_DIR_ENV, dir.path())
        .status()
        .unwrap();
    assert!(status.success(), "crash-session child failed: {status}");

    // The uncheckpointed live mutation is allowed to disappear. The last
    // durable parent must still be able to load every original child.
    let store = Arc::new(FileBlobStore::open(dir.path()).unwrap());
    let tree = Tree::open_with_blob_store(cow_retire_crash_cfg(dir.path()), store).unwrap();
    let original = [0xAB_u8; 200];
    for i in 0..5000u32 {
        assert_eq!(
            tree.get(format!("k{i:08}").as_bytes()).unwrap().as_deref(),
            Some(&original[..]),
            "durable key {i} was lost after snapshot retirement crash",
        );
    }
}

/// Store directory used by the GC crash-session child below.
const COW_GC_CRASH_DIR_ENV: &str = "HOLT_COW_GC_CRASH_DIR";

/// Leave the stable store at generation 1, advance the in-memory parent to
/// generation 2 through copy-on-write, and invoke public GC. GC must first
/// make generation 2 durable before it physically deletes generation 1's
/// children.
#[test]
#[ignore = "child-process body for gc_checkpoints_parent_before_physical_sweep"]
fn gc_before_parent_checkpoint_crash_session() {
    let Some(dir) = std::env::var_os(COW_GC_CRASH_DIR_ENV) else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let store = Arc::new(FileBlobStore::open(&dir).unwrap());
    let tree = Tree::open_with_blob_store(cow_retire_crash_cfg(&dir), store.clone()).unwrap();

    const N: u32 = 5000;
    let original = [0x11_u8; 200];
    for i in 0..N {
        tree.put(format!("k{i:08}").as_bytes(), &original).unwrap();
    }
    tree.checkpoint().unwrap();
    assert!(store.list_blobs().unwrap().len() >= 2);

    let snapshot = tree.snapshot(b"").unwrap();
    for i in (0..N).step_by(3) {
        tree.put(format!("k{i:08}").as_bytes(), b"generation-2")
            .unwrap();
    }
    drop(snapshot);

    tree.gc().unwrap();
    store.flush().unwrap();
    std::process::exit(0);
}

#[test]
fn gc_checkpoints_parent_before_physical_sweep() {
    let dir = tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();
    let status = std::process::Command::new(exe)
        .args([
            "gc_before_parent_checkpoint_crash_session",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env(COW_GC_CRASH_DIR_ENV, dir.path())
        .status()
        .unwrap();
    assert!(status.success(), "GC crash-session child failed: {status}");

    let store = Arc::new(FileBlobStore::open(dir.path()).unwrap());
    let tree = Tree::open_with_blob_store(cow_retire_crash_cfg(dir.path()), store).unwrap();
    let original = [0x11_u8; 200];
    for i in 0..5000u32 {
        let want: &[u8] = if i % 3 == 0 {
            b"generation-2"
        } else {
            &original
        };
        assert_eq!(
            tree.get(format!("k{i:08}").as_bytes()).unwrap().as_deref(),
            Some(want),
            "durable key {i} did not match the GC checkpoint generation",
        );
    }
}
