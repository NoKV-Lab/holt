//! Stage 5c integration tests — `Tree` <-> WAL hookup.
//!
//! Cover:
//! - Persistent put/delete/rename round-trip through reopen with
//!   `wal_sync_on_commit = true` (verifies WAL replay reconstructs
//!   the logical state on a crash-without-checkpoint).
//! - "Default mode without checkpoint loses unflushed" — under
//!   the default config (no per-op fsync) a drop without
//!   `checkpoint()` leaves the disk WAL empty and reopen sees
//!   the pre-mutation state.
//! - `checkpoint()` flushes everything and truncates the WAL.

use std::fs;
use std::path::{Path, PathBuf};

use tempfile::tempdir;

use artisan::{Tree, TreeConfig};

fn wal_path(dir: &Path) -> PathBuf {
    dir.join("journal.wal")
}

/// `TreeConfig::new(dir)` plus `wal_sync_on_commit = true` —
/// tests that simulate a crash without checkpoint need every
/// record on disk before the drop.
fn durable_cfg(dir: &std::path::Path) -> TreeConfig {
    let mut cfg = TreeConfig::new(dir);
    cfg.wal_sync_on_commit = true;
    cfg
}

#[test]
fn persistent_put_then_reopen_via_wal_replay() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    // Round 1: open, put, drop without checkpoint. Per-op WAL
    // fsync is on (`wal_sync_on_commit = true`), so every record
    // is on disk before the drop.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..50u32 {
            let k = format!("k{i:03}");
            let v = format!("v-{i}");
            tree.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
    } // tree dropped — no explicit checkpoint.

    // The WAL file exists and has bytes past the header.
    let wal_size_after_drop = fs::metadata(wal_path(dir.path())).unwrap().len();
    assert!(wal_size_after_drop > 32, "WAL should hold records");

    // Round 2: reopen. Replay rebuilds every key from the log.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..50u32 {
            let k = format!("k{i:03}");
            let v = format!("v-{i}");
            assert_eq!(
                tree.get(k.as_bytes()).unwrap().as_deref(),
                Some(v.as_bytes()),
                "WAL replay should have restored key {k}",
            );
        }
    }
}

#[test]
fn checkpoint_truncates_wal_and_keys_survive_reopen() {
    let dir = tempdir().unwrap();
    // Need per-op fsync so the WAL has bytes on disk before
    // the pre-checkpoint size assertion can trip.
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..20u32 {
            tree.put(format!("k{i:02}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
        }
        let wal_size_before = fs::metadata(wal_path(dir.path())).unwrap().len();
        assert!(wal_size_before > 32);
        tree.checkpoint().unwrap();
        let wal_size_after = fs::metadata(wal_path(dir.path())).unwrap().len();
        assert_eq!(
            wal_size_after, 32,
            "checkpoint should truncate WAL to header-only",
        );
    }

    // Reopen — everything still readable, but via the blob image
    // rather than WAL replay (the WAL is header-only).
    {
        let tree = Tree::open(cfg).unwrap();
        for i in 0..20u32 {
            let k = format!("k{i:02}");
            let v = format!("v{i}");
            assert_eq!(
                tree.get(k.as_bytes()).unwrap().as_deref(),
                Some(v.as_bytes()),
            );
        }
    }
}

#[test]
fn delete_through_wal_replays_correctly() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..10u32 {
            tree.put(format!("k{i}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
        }
        // Delete every even key.
        for i in 0..10u32 {
            if i % 2 == 0 {
                let prev = tree.delete(format!("k{i}").as_bytes()).unwrap();
                assert!(prev.is_some());
            }
        }
    }

    let tree = Tree::open(cfg).unwrap();
    for i in 0..10u32 {
        let got = tree.get(format!("k{i}").as_bytes()).unwrap();
        if i % 2 == 0 {
            assert_eq!(got, None, "k{i} should have been deleted");
        } else {
            assert_eq!(got.as_deref(), Some(format!("v{i}").as_bytes()));
        }
    }
}

#[test]
fn rename_through_wal_replays_correctly() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"a", b"v-a").unwrap();
        tree.put(b"b", b"v-b").unwrap();
        tree.rename(b"a", b"a2", false).unwrap();
    }

    let tree = Tree::open(cfg).unwrap();
    assert_eq!(tree.get(b"a").unwrap(), None);
    assert_eq!(tree.get(b"a2").unwrap().as_deref(), Some(&b"v-a"[..]));
    assert_eq!(tree.get(b"b").unwrap().as_deref(), Some(&b"v-b"[..]));
}

#[test]
fn default_mode_loses_writes_without_checkpoint_or_fsync() {
    // Under the default config (`wal_sync_on_commit = false`),
    // the WAL writer keeps records in its in-memory `Vec` /
    // auto-flushes them only when the buffer crosses 64 KB.
    // A short workload + drop-without-checkpoint = nothing
    // durable — exactly the high-throughput trade-off.
    let dir = tempdir().unwrap();
    let cfg = TreeConfig::new(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..50u32 {
            tree.put(
                format!("transient{i}").as_bytes(),
                format!("v{i}").as_bytes(),
            )
            .unwrap();
        }
        // Drop without `checkpoint()`. 50 records × ~80 B is
        // well below the 64 KB auto-flush threshold, so the WAL
        // file on disk is still header-only.
    }
    let wal_size = fs::metadata(wal_path(dir.path())).unwrap().len();
    assert_eq!(wal_size, 32);

    let tree = Tree::open(cfg).unwrap();
    for i in 0..50u32 {
        assert_eq!(
            tree.get(format!("transient{i}").as_bytes()).unwrap(),
            None,
            "transient{i} should have been lost",
        );
    }
}

#[test]
fn batched_mode_loses_writes_without_checkpoint() {
    let dir = tempdir().unwrap();
    let mut cfg = TreeConfig::new(dir.path());
    cfg.flush_on_write = false;

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..10u32 {
            tree.put(
                format!("transient{i}").as_bytes(),
                format!("v{i}").as_bytes(),
            )
            .unwrap();
        }
        // Drop without `checkpoint()`. Nothing was flushed:
        // - WAL records buffered in memory → lost
        // - BM root blob mutated in memory → lost
    }

    // Reopen — empty WAL, empty blob, no keys readable.
    let tree = Tree::open(cfg).unwrap();
    for i in 0..10u32 {
        assert_eq!(
            tree.get(format!("transient{i}").as_bytes()).unwrap(),
            None,
            "transient{i} should have been lost",
        );
    }
}

#[test]
fn batched_mode_with_checkpoint_persists_everything() {
    let dir = tempdir().unwrap();
    let mut cfg = TreeConfig::new(dir.path());
    cfg.flush_on_write = false;

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..30u32 {
            tree.put(
                format!("batch{i:02}").as_bytes(),
                format!("v{i}").as_bytes(),
            )
            .unwrap();
        }
        tree.checkpoint().unwrap();
        // After checkpoint, WAL is truncated and the blob image
        // is durable. Subsequent drops without further mutation
        // are safe.
    }

    let tree = Tree::open(cfg).unwrap();
    for i in 0..30u32 {
        let v = tree
            .get(format!("batch{i:02}").as_bytes())
            .unwrap()
            .expect("batch key survives via blob image");
        assert_eq!(v, format!("v{i}").into_bytes());
    }
}

#[test]
fn next_seq_resumes_past_replayed_records() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    // Round 1: write 5 keys; each consumes one seq.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..5u32 {
            tree.put(format!("k{i}").as_bytes(), b"v").unwrap();
        }
    }

    // Round 2: reopen. The replayed records carried seq 1..=5.
    // The first new `put` must use seq >= 6 — otherwise a leaf
    // built later could overwrite one rebuilt by replay.
    {
        let tree = Tree::open(cfg).unwrap();
        // The exact seq isn't exposed, but the round-trip
        // semantics imply: after a put, the value is readable.
        tree.put(b"after-replay", b"v").unwrap();
        assert_eq!(
            tree.get(b"after-replay").unwrap().as_deref(),
            Some(&b"v"[..])
        );
        // And every earlier key still readable too.
        for i in 0..5u32 {
            assert_eq!(
                tree.get(format!("k{i}").as_bytes()).unwrap().as_deref(),
                Some(&b"v"[..]),
            );
        }
    }
}

#[test]
fn open_with_backend_attaches_no_wal() {
    use artisan::{Backend, MemoryBackend, TreeBuilder};
    use std::sync::Arc;

    let dir = tempdir().unwrap();
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // open_with_backend deliberately bypasses WAL — `dir` here is
    // informational; the backend stores in memory.
    {
        let tree = TreeBuilder::new(dir.path())
            .open_with_backend(backend.clone())
            .unwrap();
        tree.put(b"k", b"v").unwrap();
    }

    // No WAL file should have been created.
    assert!(!wal_path(dir.path()).exists());
}

#[test]
fn many_round_trips_through_checkpoint_boundaries() {
    let dir = tempdir().unwrap();
    // The final batch isn't followed by a checkpoint — it relies
    // on per-op WAL fsync to survive the drop + reopen.
    let cfg = durable_cfg(dir.path());

    // Three rounds, each with a checkpoint mid-stream. Verifies
    // that records past a checkpoint are also durable (the WAL
    // truncate doesn't lose anything we already flushed through
    // the blob).
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..20u32 {
            tree.put(format!("a{i:02}").as_bytes(), b"A").unwrap();
        }
        tree.checkpoint().unwrap();
        for i in 0..20u32 {
            tree.put(format!("b{i:02}").as_bytes(), b"B").unwrap();
        }
        tree.checkpoint().unwrap();
        for i in 0..20u32 {
            tree.put(format!("c{i:02}").as_bytes(), b"C").unwrap();
        }
        // No checkpoint after c-batch — relies on WAL replay.
    }

    let tree = Tree::open(cfg).unwrap();
    for i in 0..20u32 {
        assert_eq!(
            tree.get(format!("a{i:02}").as_bytes()).unwrap().as_deref(),
            Some(&b"A"[..]),
        );
        assert_eq!(
            tree.get(format!("b{i:02}").as_bytes()).unwrap().as_deref(),
            Some(&b"B"[..]),
        );
        assert_eq!(
            tree.get(format!("c{i:02}").as_bytes()).unwrap().as_deref(),
            Some(&b"C"[..]),
        );
    }
}
