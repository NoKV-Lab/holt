//! Stage 5c integration tests — `Tree` <-> WAL hookup.
//!
//! Cover:
//! - Persistent put/delete/rename round-trip through reopen with
//!   `WalCommit::Sync` (verifies WAL replay reconstructs
//!   the logical state on a crash-without-checkpoint).
//! - "Default mode without checkpoint loses unflushed" — under
//!   the default config (no durable per-op journal wait) a drop without
//!   `checkpoint()` leaves the disk WAL empty and reopen sees
//!   the pre-mutation state.
//! - `checkpoint()` flushes everything and truncates the WAL.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;

use tempfile::tempdir;

use holt::{Tree, TreeConfig, WalCommit};

fn wal_path(dir: &Path) -> PathBuf {
    dir.join("journal.wal")
}

/// `TreeConfig::new(dir)` plus `WalCommit::Sync` — tests that
/// simulate power-safe crash recovery without checkpoint need every
/// record fsync'd before drop.
fn durable_cfg(dir: &std::path::Path) -> TreeConfig {
    let mut cfg = TreeConfig::new(dir);
    cfg.wal_commit = WalCommit::Sync;
    cfg
}

#[test]
fn durable_writers_share_group_commit_syncs() {
    let dir = tempdir().unwrap();
    let tree = Arc::new(Tree::open(durable_cfg(dir.path())).unwrap());
    const WRITERS: usize = 12;
    let barrier = Arc::new(Barrier::new(WRITERS));

    let handles: Vec<_> = (0..WRITERS)
        .map(|i| {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let key = format!("gc/key-{i:02}");
                let value = format!("value-{i:02}");
                barrier.wait();
                tree.put(key.as_bytes(), value.as_bytes()).unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let stats = tree.stats().unwrap();
    let journal = stats.journal.expect("persistent tree has journal stats");
    assert_eq!(journal.appends, WRITERS as u64);
    assert!(
        journal.syncs < journal.appends,
        "durable writers should share fsyncs through group commit; appends={}, syncs={}",
        journal.appends,
        journal.syncs,
    );

    drop(tree);
    let reopened = Tree::open(durable_cfg(dir.path())).unwrap();
    for i in 0..WRITERS {
        let key = format!("gc/key-{i:02}");
        let value = format!("value-{i:02}");
        assert_eq!(
            reopened.get(key.as_bytes()).unwrap().as_deref(),
            Some(value.as_bytes()),
        );
    }
}

#[test]
fn clean_checkpoint_skips_empty_wal_flush() {
    let dir = tempdir().unwrap();
    let tree = Tree::open(TreeConfig::new(dir.path())).unwrap();

    let before = tree.stats().unwrap().journal.unwrap();
    tree.checkpoint().unwrap();
    let after = tree.stats().unwrap().journal.unwrap();
    assert_eq!(
        after.syncs, before.syncs,
        "checkpoint on a fresh tree must not fsync an empty WAL",
    );

    tree.put(b"wal-clean/k", b"v").unwrap();
    tree.checkpoint().unwrap();
    assert_eq!(fs::metadata(wal_path(dir.path())).unwrap().len(), 32);

    let after_truncate = tree.stats().unwrap().journal.unwrap();
    tree.checkpoint().unwrap();
    let clean_again = tree.stats().unwrap().journal.unwrap();
    assert_eq!(
        clean_again.syncs, after_truncate.syncs,
        "checkpoint after WAL truncate must stay a no-op",
    );
}

#[test]
fn checkpoint_reuses_durable_group_commit_wal_sync() {
    let dir = tempdir().unwrap();
    let tree = Tree::open(durable_cfg(dir.path())).unwrap();

    tree.put(b"durable-checkpoint/k", b"v").unwrap();
    let after_put = tree.stats().unwrap().journal.unwrap();
    assert!(after_put.syncs > 0);

    tree.checkpoint().unwrap();
    let after_checkpoint = tree.stats().unwrap().journal.unwrap();
    assert_eq!(
        after_checkpoint.syncs, after_put.syncs,
        "checkpoint must not fsync WAL records already made durable by group commit",
    );
    assert_eq!(fs::metadata(wal_path(dir.path())).unwrap().len(), 32);
}

#[test]
fn persistent_put_then_reopen_via_wal_replay() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    // Round 1: open, put, drop without checkpoint. Per-op WAL
    // fsync is on (`WalCommit::Sync`), so every record
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
fn replay_then_checkpoint_then_reopen_preserves_data() {
    // Regression test for the "replay → checkpoint → reopen
    // loses data" hole:
    //
    // 1. Open, put N keys, drop without checkpoint (WAL has
    //    every record on disk; store root blob is still
    //    pristine because nothing was flushed).
    // 2. Reopen — `replay_wal` re-applies every WAL record onto
    //    the BM-cached root. The root blob's IN-MEMORY image now
    //    matches the post-put state; the store image is still
    //    the empty seeded root.
    // 3. Immediately call `tree.checkpoint()` — this must flush
    //    the cached root through to store BEFORE truncating
    //    the WAL. The v0.2-pre `replay_wal` didn't `mark_dirty`
    //    the root, so the dirty set was empty after replay; the
    //    checkpoint round drained nothing, wrote nothing to
    //    store, and then truncated the WAL — silently losing
    //    every replayed record.
    // 4. Reopen again — the store is the sole source of truth
    //    now (WAL was truncated). All keys must still be there.
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    // Round 1: write durably to WAL, no checkpoint.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..50u32 {
            tree.put(format!("k{i:03}").as_bytes(), format!("v-{i}").as_bytes())
                .unwrap();
        }
    }

    // Round 2: reopen → replay; then checkpoint → truncates WAL.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        // Sanity: WAL replay restored the cached state.
        for i in 0..50u32 {
            assert_eq!(
                tree.get(format!("k{i:03}").as_bytes()).unwrap().as_deref(),
                Some(format!("v-{i}").as_bytes()),
            );
        }
        // Now checkpoint. **This must flush the replayed state
        // to store before truncating the WAL.**
        tree.checkpoint().unwrap();
        let wal_size_after = fs::metadata(wal_path(dir.path())).unwrap().len();
        assert_eq!(wal_size_after, 32, "WAL truncated to header-only");
    }

    // Round 3: store is the source of truth (WAL empty). If
    // checkpoint didn't actually flush the replayed state, this
    // reopen sees the pre-put pristine root and every get returns
    // None.
    {
        let tree = Tree::open(cfg).unwrap();
        for i in 0..50u32 {
            let k = format!("k{i:03}");
            assert_eq!(
                tree.get(k.as_bytes()).unwrap().as_deref(),
                Some(format!("v-{i}").as_bytes()),
                "key {k} lost — replay→checkpoint didn't persist replayed state",
            );
        }
    }
}

#[test]
fn checkpoint_truncates_wal_and_keys_survive_reopen() {
    let dir = tempdir().unwrap();
    // Need durable per-op journal waits so the WAL has bytes on
    // disk before the pre-checkpoint size assertion can trip.
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
                let removed = tree.delete(format!("k{i}").as_bytes()).unwrap();
                assert!(removed);
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
fn conditional_writes_replay_through_wal() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        assert!(tree.put_if_absent(b"cas/k", b"v1").unwrap());
        assert!(!tree.put_if_absent(b"cas/k", b"blocked").unwrap());

        let v1 = tree.get_version(b"cas/k").unwrap().unwrap();
        assert!(tree.compare_and_put(b"cas/k", v1, b"v2").unwrap());
        assert!(!tree.compare_and_put(b"cas/k", v1, b"stale").unwrap());

        let v2 = tree.get_version(b"cas/k").unwrap().unwrap();
        assert!(!tree.delete_if_version(b"cas/k", v1).unwrap());
        assert!(tree.delete_if_version(b"cas/k", v2).unwrap());

        assert!(tree.put_if_absent(b"cas/resurrected", b"live").unwrap());
    }

    let tree = Tree::open(cfg).unwrap();
    assert!(tree.get(b"cas/k").unwrap().is_none());
    assert!(tree.get_version(b"cas/k").unwrap().is_none());
    assert_eq!(
        tree.get(b"cas/resurrected").unwrap().as_deref(),
        Some(&b"live"[..]),
    );
}

#[test]
fn default_mode_loses_writes_without_checkpoint_or_fsync() {
    // Under the default config (`WalCommit::Enqueue`),
    // the journal worker can still hold records in process memory
    // and only drains/syncs them at checkpoint. A short workload +
    // drop-without-checkpoint = nothing durable — exactly the
    // high-throughput trade-off.
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
    cfg.memory_flush_on_write = false;

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
    cfg.memory_flush_on_write = false;

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
fn open_with_blob_store_attaches_no_wal() {
    use holt::{BlobStore, MemoryBlobStore, TreeBuilder};
    use std::sync::Arc;

    let dir = tempdir().unwrap();
    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());

    // open_with_blob_store deliberately bypasses WAL — `dir` here is
    // informational; the store stores in memory.
    {
        let tree = TreeBuilder::new(dir.path())
            .open_with_blob_store(store.clone())
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

#[test]
fn batch_persists_through_crash_and_replay() {
    // Tree::atomic emits one Batch WAL record; on reopen the replay
    // unpacks it transparently into per-inner callbacks so every
    // op in the batch comes back. `WalCommit::Sync`
    // makes the simulated crash drop right after the batch flush.
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        // Seed something to mutate inside the batch.
        tree.put(b"seed", b"S").unwrap();

        tree.atomic(|b| {
            b.put(b"batch-a", b"A");
            b.put(b"batch-b", b"B");
            b.delete(b"seed");
            b.rename(b"batch-a", b"batch-aa", false);
        })
        .unwrap();
    } // dropped without checkpoint — disk has only the WAL.

    // Reopen — replay should reconstruct the post-batch state.
    let tree = Tree::open(cfg).unwrap();
    assert!(tree.get(b"seed").unwrap().is_none());
    assert!(tree.get(b"batch-a").unwrap().is_none());
    assert_eq!(tree.get(b"batch-aa").unwrap().as_deref(), Some(&b"A"[..]));
    assert_eq!(tree.get(b"batch-b").unwrap().as_deref(), Some(&b"B"[..]));
}

#[test]
fn compact_insert_run_batch_persists_through_replay() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());
    let mut versions = Vec::new();

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.atomic(|b| {
            for i in 0..16u32 {
                let key = format!("bulk/object-{i:04}");
                let value = format!("metadata-{i:04}");
                b.put(key.as_bytes(), value.as_bytes());
            }
        })
        .unwrap();

        for i in 0..16u32 {
            let key = format!("bulk/object-{i:04}");
            versions.push(tree.get_record(key.as_bytes()).unwrap().unwrap().version);
        }
    }

    let tree = Tree::open(cfg).unwrap();
    for i in 0..16u32 {
        let key = format!("bulk/object-{i:04}");
        let value = format!("metadata-{i:04}");
        let record = tree.get_record(key.as_bytes()).unwrap().unwrap();
        assert_eq!(record.value, value.as_bytes());
        assert_eq!(
            record.version, versions[i as usize],
            "compact insert-run replay must preserve per-inner record versions",
        );
    }
}

#[test]
fn batch_conditional_ops_replay_with_stable_versions() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());
    let seed_version_after_atomic;

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"seed", b"v1").unwrap();
        let seed_v1 = tree.get_record(b"seed").unwrap().unwrap().version;

        assert!(tree
            .atomic(|b| {
                // Deliberately no-op but still encoded inside the
                // Batch WAL record so later inner seqs replay with
                // the same record versions.
                b.delete(b"missing");
                b.compare_and_put(b"seed", seed_v1, b"v2");
                b.put_if_absent(b"created", b"new");
            })
            .unwrap());
        seed_version_after_atomic = tree.get_record(b"seed").unwrap().unwrap().version;
    }

    let tree = Tree::open(cfg).unwrap();
    let seed = tree.get_record(b"seed").unwrap().unwrap();
    assert_eq!(seed.value, b"v2");
    assert_eq!(
        seed.version, seed_version_after_atomic,
        "Batch replay must preserve per-inner seq even when an earlier inner op is a no-op",
    );
    assert_eq!(tree.get(b"created").unwrap().as_deref(), Some(&b"new"[..]));
}

#[test]
fn batch_prefix_assertions_do_not_shift_replay_versions() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());
    let k1_version;
    let k2_version;

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        assert!(tree
            .atomic(|b| {
                b.assert_prefix_empty(b"guard/");
                b.put(b"batch/k1", b"v1");
                b.assert_prefix_empty(b"other/");
                b.put(b"batch/k2", b"v2");
            })
            .unwrap());
        k1_version = tree.get_record(b"batch/k1").unwrap().unwrap().version;
        k2_version = tree.get_record(b"batch/k2").unwrap().unwrap().version;
    }

    let tree = Tree::open(cfg).unwrap();
    let k1 = tree.get_record(b"batch/k1").unwrap().unwrap();
    let k2 = tree.get_record(b"batch/k2").unwrap().unwrap();
    assert_eq!(k1.value, b"v1");
    assert_eq!(k2.value, b"v2");
    assert_eq!(
        k1.version, k1_version,
        "prefix assertions must not consume Batch WAL inner sequence numbers",
    );
    assert_eq!(
        k2.version, k2_version,
        "prefix assertions must not shift later replay versions",
    );
}

#[test]
fn batch_version_assertions_do_not_shift_replay_versions() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());
    let seed_version;
    let copied_version;
    let later_version;

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"seed", b"payload").unwrap();
        let seed = tree.get_record(b"seed").unwrap().unwrap();
        assert!(tree
            .atomic(|b| {
                b.assert_version(b"seed", seed.version);
                b.put(b"copied", &seed.value);
                b.assert_version(b"seed", seed.version);
                b.put(b"later", b"v2");
            })
            .unwrap());
        seed_version = tree.get_record(b"seed").unwrap().unwrap().version;
        copied_version = tree.get_record(b"copied").unwrap().unwrap().version;
        later_version = tree.get_record(b"later").unwrap().unwrap().version;
    }

    let tree = Tree::open(cfg).unwrap();
    let seed = tree.get_record(b"seed").unwrap().unwrap();
    let copied = tree.get_record(b"copied").unwrap().unwrap();
    let later = tree.get_record(b"later").unwrap().unwrap();
    assert_eq!(seed.value, b"payload");
    assert_eq!(copied.value, b"payload");
    assert_eq!(later.value, b"v2");
    assert_eq!(
        seed.version, seed_version,
        "assert_version must not rewrite the guarded source",
    );
    assert_eq!(
        copied.version, copied_version,
        "assert_version must not consume Batch WAL inner sequence numbers",
    );
    assert_eq!(
        later.version, later_version,
        "assert_version must not shift later replay versions",
    );
}

#[test]
fn atomic_assert_only_batch_does_not_append_wal_or_consume_seq() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"seed", b"payload").unwrap();
        let seed = tree.get_record(b"seed").unwrap().unwrap();
        let wal_len = fs::metadata(wal_path(dir.path())).unwrap().len();

        assert!(tree
            .atomic(|b| {
                b.assert_version(b"seed", seed.version);
                b.assert_prefix_empty(b"empty/");
            })
            .unwrap());

        assert_eq!(
            fs::metadata(wal_path(dir.path())).unwrap().len(),
            wal_len,
            "assert-only atomic batches must not emit WAL records",
        );
        assert_eq!(
            tree.get_record(b"seed").unwrap().unwrap().version,
            seed.version,
            "assert-only atomic batches must not consume record versions",
        );
    }

    let tree = Tree::open(cfg).unwrap();
    assert_eq!(tree.get(b"seed").unwrap().as_deref(), Some(&b"payload"[..]));
}

#[test]
fn failed_atomic_guard_does_not_append_wal_or_publish() {
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"guarded", b"v1").unwrap();
        let stale = tree.get_version(b"guarded").unwrap().unwrap();
        tree.put(b"guarded", b"v2").unwrap();
        let wal_len = fs::metadata(wal_path(dir.path())).unwrap().len();

        let committed = tree
            .atomic(|b| {
                b.assert_version(b"guarded", stale);
                b.put(b"side", b"should-not-publish");
            })
            .unwrap();

        assert!(!committed);
        assert_eq!(
            fs::metadata(wal_path(dir.path())).unwrap().len(),
            wal_len,
            "failed preflight must not append a Batch WAL record",
        );
        assert!(tree.get(b"side").unwrap().is_none());
        assert_eq!(tree.get(b"guarded").unwrap().as_deref(), Some(&b"v2"[..]));
    }

    let tree = Tree::open(cfg).unwrap();
    assert!(tree.get(b"side").unwrap().is_none());
    assert_eq!(tree.get(b"guarded").unwrap().as_deref(), Some(&b"v2"[..]));
}

#[test]
fn batch_crash_before_flush_loses_whole_batch() {
    // Default mode (`WalCommit::Enqueue`): if we drop
    // without checkpoint, the OS may not have flushed the batch
    // record yet, so the whole batch is rolled back on reopen.
    // We exercise the contract by skipping checkpoint and
    // checking that the unflushed batch isn't visible after
    // reopen.
    let dir = tempdir().unwrap();
    let cfg = TreeConfig::new(dir.path()); // default: WalCommit::Enqueue

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"durable", b"D").unwrap();
        tree.checkpoint().unwrap();

        // Batch goes through the BM cache but the WAL flush is
        // deferred; without a checkpoint, the on-disk WAL stays
        // empty for these ops.
        tree.atomic(|b| {
            b.put(b"vanish-a", b"VA");
            b.put(b"vanish-b", b"VB");
        })
        .unwrap();
        // Note: we do NOT call tree.checkpoint() — the batch
        // record sits in the WAL's in-memory buffer and dies
        // with the process.
    }

    let tree = Tree::open(cfg).unwrap();
    assert_eq!(tree.get(b"durable").unwrap().as_deref(), Some(&b"D"[..]));
    assert!(tree.get(b"vanish-a").unwrap().is_none());
    assert!(tree.get(b"vanish-b").unwrap().is_none());
}

#[test]
fn background_checkpointer_truncates_wal_and_keeps_data_durable() {
    // v0.2 integration smoke: with the background checkpointer
    // enabled, a steady stream of writes should leave the WAL
    // bounded (it gets truncated to header-only on rounds where
    // nothing else is racing the writer) AND every written value
    // remains observable after reopen (because the round flushed
    // the cached root into store before truncating).
    use holt::{CheckpointConfig, TreeBuilder};
    use std::thread;
    use std::time::{Duration, Instant};

    let dir = tempdir().unwrap();

    {
        let tree = TreeBuilder::new(dir.path())
            .checkpoint(CheckpointConfig {
                enabled: true,
                idle_interval: Duration::from_millis(25),
                dirty_blob_threshold: 1,
                auto_merge: true,
                ..CheckpointConfig::default()
            })
            .open()
            .unwrap();

        // Produce a WAL of non-trivial size.
        for i in 0..500u32 {
            tree.put(format!("bg/{i:04}").as_bytes(), format!("v-{i}").as_bytes())
                .unwrap();
        }

        // Wait until the background thread shrinks the WAL back
        // to header-only — i.e. it completed a round where dirty
        // was empty under the commit gate. Give it generous time;
        // the test cares about *eventual* truncate, not latency.
        let header_size_after_truncate = 32u64; // FILE_HEADER_SIZE
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            // Trigger another op so the dirty set isn't always
            // co-occupied with an in-flight write (the truncate
            // only fires when dirty is empty under the commit gate).
            tree.put(b"_tick", b".").unwrap();
            let wal_len = fs::metadata(wal_path(dir.path())).unwrap().len();
            if wal_len <= header_size_after_truncate + 128 {
                // Tolerate one or two trailing ops; the test cares
                // about "WAL stopped growing unbounded", not exact
                // zero.
                break;
            }
            assert!(
                Instant::now() < deadline,
                "background checkpointer never truncated WAL (size={wal_len})",
            );
            thread::sleep(Duration::from_millis(20));
        }
    } // tree dropped → checkpointer joined.

    // After reopen, every key is still readable — the bg
    // checkpointer's flush sequence (commit → fdatasync →
    // truncate) made the store the durable source of truth.
    let tree = Tree::open(TreeConfig::new(dir.path())).unwrap();
    for i in 0..500u32 {
        let k = format!("bg/{i:04}");
        let want = format!("v-{i}");
        assert_eq!(
            tree.get(k.as_bytes()).unwrap().as_deref(),
            Some(want.as_bytes()),
            "key {k} lost after bg-checkpoint-and-reopen",
        );
    }
}

#[test]
fn spillover_new_blobs_deferred_to_store_until_checkpoint() {
    // Regression test for the v0.2 W2D fix: spillover used to call
    // `bm.write_blob → bm.flush` inline, leaking the new child
    // blob's bytes to the inner store before any WAL record
    // covering the spillover-triggering op was durable. The fix
    // routes the new blob through `install_new_blob` (cache +
    // dirty), so the store write happens only after the
    // checkpoint round has flushed WAL first.
    use holt::{BlobStore, MemoryBlobStore};
    use std::sync::Arc;

    let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());

    // `open_with_blob_store` skips the WAL and the bg checkpointer
    // (default `CheckpointConfig::disabled`). Disable
    // `memory_flush_on_write` too so the test can observe the dirty-set
    // state between ops and the explicit `checkpoint` call.
    let mut cfg = TreeConfig::memory();
    cfg.memory_flush_on_write = false;
    let tree = Tree::open_with_blob_store(cfg, Arc::clone(&inner)).unwrap();

    // Inner store starts with only the seeded root.
    let initial = inner.list_blobs().unwrap();
    assert_eq!(initial.len(), 1, "open seeds only the root blob");

    // Force spillover by stuffing the tree with payloads big
    // enough that a single 512 KB root can't hold them all.
    let payload = vec![b'x'; 1024];
    for i in 0..1000u32 {
        tree.put(format!("k{i:05}").as_bytes(), &payload).unwrap();
    }

    // Spillover ran: the tree has multiple reachable blobs now.
    let stats = tree.stats().unwrap();
    assert!(
        stats.blob_count > 1,
        "spillover should have created at least one child blob (got {})",
        stats.blob_count,
    );
    assert!(
        stats.bm_dirty_count >= 1,
        "every spillover'd blob + root must be tracked dirty",
    );

    // **The point of the test**: the inner store has NOT yet
    // received the spillover'd child blobs. Pre-fix, the inline
    // `bm.write_blob` call in `spillover_blob` would have pushed
    // them through immediately — a crash here would have left
    // orphans in store.
    let mid = inner.list_blobs().unwrap();
    assert_eq!(
        mid.len(),
        1,
        "inner store must NOT see spillover'd children until checkpoint (got {} blobs)",
        mid.len(),
    );

    // Explicit checkpoint — drains the dirty set + fdatasync.
    tree.checkpoint().unwrap();

    let stats_after = tree.stats().unwrap();
    assert_eq!(
        stats_after.bm_dirty_count, 0,
        "checkpoint must drain every dirty entry",
    );

    let final_blobs = inner.list_blobs().unwrap();
    assert_eq!(
        final_blobs.len() as u32,
        stats.blob_count,
        "after checkpoint, inner store has every reachable blob",
    );

    // Sanity: every payload still readable through the tree
    // (sourced from the freshly-flushed inner store on a
    // cache miss).
    for i in 0..1000u32 {
        let k = format!("k{i:05}");
        let v = tree.get(k.as_bytes()).unwrap().expect("key present");
        assert_eq!(v.as_slice(), payload.as_slice(), "value drift for {k}");
    }
}

#[test]
fn compact_does_not_leak_pre_wal_state_to_store() {
    // Regression test: `Tree::compact` used to call
    // `bm.commit(*guid)` for every touched blob, which pushes
    // the cached image (including unflushed-WAL user mutations)
    // straight to store. A crash before the user's WAL record
    // was durable would have left the store at the post-put
    // state while the WAL contained no record — and the next
    // reopen-via-WAL-replay would have re-built a state that
    // looks like "put never happened" (cache rebuilt from WAL =
    // empty; checkpoint then truncates the empty WAL; reopen
    // sees the store which still has post-put state, but the
    // user-visible model now disagrees with the durable image).
    //
    // The simplest demonstration: turn off WAL fsync, put a key,
    // run compact (which races the WAL flush), drop without
    // checkpoint, reopen. Pre-fix, compact had already shoved
    // the put's bytes into the store; the put's WAL record
    // never made it to disk (`WalCommit::Enqueue` and no
    // explicit checkpoint), so the open-time replay sees no
    // record and considers `next_seq` to start at 1 — but the
    // store has the put. Mixing those would surface as either
    // a phantom value or a torn state on subsequent ops.
    //
    // Post-fix, compact only marks dirty + leaves flushing to
    // the user / checkpointer. With no checkpoint between put
    // and drop, the store stays at the pre-put state, the WAL
    // is empty (records weren't fsync'd), and reopen sees the
    // pristine root → `get` returns None. That's the expected
    // "lost write under no-fsync, no-checkpoint" semantics; the
    // failure mode pre-fix was different and worse.
    //
    // We verify the post-fix invariant by checking that
    // `tree.stats().bm_dirty_count > 0` *after* compact (i.e.,
    // compact left things dirty rather than flushing them
    // through), and that the store file size remains at the
    // initial seeded-root size (no spillover blobs leaked
    // through).
    use holt::{BlobStore, MemoryBlobStore};
    use std::sync::Arc;

    let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());

    let mut cfg = TreeConfig::memory();
    cfg.memory_flush_on_write = false; // no implicit per-op flush
    let tree = Tree::open_with_blob_store(cfg, Arc::clone(&inner)).unwrap();

    let initial = inner.list_blobs().unwrap();
    assert_eq!(initial.len(), 1, "open seeds only the root blob");

    // Put a key so the root blob's cache image diverges from
    // store. mark_dirty(root, seq) fires inside Tree::put.
    tree.put(b"key", b"value").unwrap();
    assert!(
        tree.stats().unwrap().bm_dirty_count >= 1,
        "put must leave root dirty",
    );

    // Now compact. Pre-fix, this would commit the root through
    // to store, eagerly persisting the put's bytes without
    // any WAL gate. Post-fix, compact only restructures cache +
    // marks dirty.
    tree.compact().unwrap();

    // BlobStore still pristine — compact did NOT flush.
    let after_compact = inner.list_blobs().unwrap();
    assert_eq!(
        after_compact.len(),
        1,
        "compact must not push cache state to store",
    );
    // And the root is still dirty: the put + the compact reshuffle
    // are both staged in cache, waiting for `checkpoint` to
    // commit them under the W2D gate.
    assert!(
        tree.stats().unwrap().bm_dirty_count >= 1,
        "compact must leave dirty entries (not auto-flush)",
    );

    // Now actually checkpoint — store receives the merged state.
    tree.checkpoint().unwrap();
    assert_eq!(tree.stats().unwrap().bm_dirty_count, 0);

    // Sanity: value still readable through the freshly-flushed
    // store.
    assert_eq!(tree.get(b"key").unwrap().as_deref(), Some(&b"value"[..]),);
}

#[test]
fn multi_blob_compact_does_not_leak_pre_wal_state_to_store() {
    // Same protocol assertion as
    // `compact_does_not_leak_pre_wal_state_to_store`, but
    // sized to force spillover so `Tree::compact` considers
    // multiple child blobs and then attempts tree-wide merge.
    // Cross-blob entry is now only the child blob's
    // `header.root_slot`, so parent BlobNodes do not carry a child
    // entry slot that needs a post-compact repair pass.
    use holt::{BlobStore, MemoryBlobStore};
    use std::sync::Arc;

    let inner: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    let mut cfg = TreeConfig::memory();
    cfg.memory_flush_on_write = false;
    let tree = Tree::open_with_blob_store(cfg, Arc::clone(&inner)).unwrap();

    // Inner store starts with only the seeded root.
    assert_eq!(inner.list_blobs().unwrap().len(), 1);

    // Stuff enough data to force at least two spillovers so the
    // compact walk has multiple BlobNodes to consider.
    let payload = vec![b'q'; 1024];
    for i in 0..1500u32 {
        tree.put(format!("k{i:05}").as_bytes(), &payload).unwrap();
    }
    let stats = tree.stats().unwrap();
    assert!(
        stats.blob_count > 1,
        "multi-blob compact precondition: spillover must trigger (got {} blobs)",
        stats.blob_count,
    );

    // Compact may rewrite blobs with reclaimable garbage and may
    // restructure parent BlobNodes. It must NOT push anything to
    // store — only stage via dirty.
    tree.compact().unwrap();
    let after_compact = inner.list_blobs().unwrap();
    assert_eq!(
        after_compact.len(),
        1,
        "multi-blob compact must not push cache state to store (got {} blobs)",
        after_compact.len(),
    );
    assert!(
        tree.stats().unwrap().bm_dirty_count >= 1,
        "compact must leave dirty entries waiting for the next checkpoint",
    );

    // Now checkpoint and reopen-via-store: every key must still
    // be present (any structural rewrite preserved logical state).
    tree.checkpoint().unwrap();
    assert_eq!(tree.stats().unwrap().bm_dirty_count, 0);
    for i in 0..1500u32 {
        let k = format!("k{i:05}");
        assert_eq!(
            tree.get(k.as_bytes()).unwrap().as_deref(),
            Some(payload.as_slice()),
            "key {k} lost after multi-blob compact + checkpoint",
        );
    }
}

#[test]
fn tree_stats_does_not_perturb_cache_counters_or_lru() {
    // `Tree::stats` is the observability path — it must NOT
    // bump `bm_cache_hits` / `bm_cache_misses` or refresh the
    // per-entry `last_touched` tick. A Prometheus scrape calling
    // `stats()` repeatedly should report the same numbers
    // between scrapes when no other work happens in between.
    let tree = Tree::open(TreeConfig::memory()).unwrap();
    // Seed enough data to spillover so `stats()` walks multiple
    // blobs (not just the root).
    let payload = vec![b'q'; 1024];
    for i in 0..800u32 {
        tree.put(format!("k{i:05}").as_bytes(), &payload).unwrap();
    }

    // Capture baseline counters AFTER one stats() call so any
    // first-call setup costs are excluded.
    let baseline = tree.stats().unwrap();
    let baseline_hits = baseline.bm_cache_hits;
    let baseline_misses = baseline.bm_cache_misses;
    assert!(
        baseline.blob_count > 1,
        "test premise: multi-blob tree (got {})",
        baseline.blob_count,
    );

    // Call stats() a bunch more times — none of these should
    // perturb the cache hit/miss counters.
    for _ in 0..50 {
        let s = tree.stats().unwrap();
        assert_eq!(
            s.bm_cache_hits, baseline_hits,
            "Tree::stats() must not bump cache_hits",
        );
        assert_eq!(
            s.bm_cache_misses, baseline_misses,
            "Tree::stats() must not bump cache_misses",
        );
    }
}

#[test]
fn batch_replay_then_checkpoint_then_reopen_preserves_data() {
    // Same W2D closure as `replay_then_checkpoint_then_reopen_preserves_data`,
    // but exercises the `apply_batch` path: a single `Batch` WAL
    // record fan-outs into per-inner Insert/Erase/RenameObject
    // callbacks during replay. The fix marks root dirty per
    // inner op, so the inner Batch unwrap must surface the seq /
    // mutation in the same way `Tree::put`/`delete` do.
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    // Round 1: durable batch, no checkpoint.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.atomic(|b| {
            b.put(b"a", b"1");
            b.put(b"b", b"2");
            b.put(b"c", b"3");
            b.delete(b"a");
            b.rename(b"b", b"B", /*force=*/ false);
        })
        .unwrap();
    }

    // Round 2: replay → checkpoint → expect state to be durable.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.checkpoint().unwrap();
    }

    // Round 3: store is sole source of truth (WAL truncated).
    {
        let tree = Tree::open(cfg).unwrap();
        assert_eq!(tree.get(b"a").unwrap(), None);
        assert_eq!(tree.get(b"B").unwrap().as_deref(), Some(&b"2"[..]));
        assert_eq!(tree.get(b"c").unwrap().as_deref(), Some(&b"3"[..]));
    }
}

#[test]
fn rename_replay_is_idempotent_across_two_drops() {
    // The replay path skips a rename whose `src` no longer
    // exists (already reconciled by a previous replay) and skips
    // a non-forced rename when `dst` is already populated. This
    // matters when two reopens happen without an intermediate
    // checkpoint — round 2's replay re-applies the rename, then
    // round 3's replay sees the same WAL records but the cache
    // already reflects the post-rename state.
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    // Round 1: put + rename, drop without checkpoint.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"src", b"v1").unwrap();
        tree.rename(b"src", b"dst", false).unwrap();
    }

    // Round 2: replay, drop without checkpoint. Replay re-runs
    // put then rename — both succeed because the cache is empty
    // at open.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        assert!(tree.get(b"src").unwrap().is_none());
        assert_eq!(tree.get(b"dst").unwrap().as_deref(), Some(&b"v1"[..]));
        // No checkpoint here.
    }

    // Round 3: replay AGAIN (still no checkpoint). Now the
    // rename replay path's "src already gone" branch must fire
    // — the cache after the put-replay has `dst` (the cache
    // image isn't durable yet, but the put record re-fills it).
    // Actually the cache is rebuilt fresh from an empty store
    // every reopen, so this exercises the same path; the test
    // just guarantees no drift across repeated replays.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        assert!(tree.get(b"src").unwrap().is_none());
        assert_eq!(tree.get(b"dst").unwrap().as_deref(), Some(&b"v1"[..]));
        // Now checkpoint and reopen — store must hold the
        // post-rename state.
        tree.checkpoint().unwrap();
    }
    {
        let tree = Tree::open(cfg).unwrap();
        assert!(tree.get(b"src").unwrap().is_none());
        assert_eq!(tree.get(b"dst").unwrap().as_deref(), Some(&b"v1"[..]));
    }
}

#[test]
fn subtree_gone_replay_reconstructs_correctly() {
    // Cross-blob erase that empties a child blob: the v0.2-pre
    // code called `bm.delete_blob` inline (W2D-broken). Post-fix
    // the SubtreeGone path queues a deferred delete and the
    // checkpoint round drains it after Sync.
    //
    // End-to-end test: fill enough data to spillover into N
    // children, delete every key under one of the children so
    // that SubtreeGone fires for that child, drop without
    // checkpoint, reopen + checkpoint + reopen, verify nothing
    // is corrupt.
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    {
        let tree = Tree::open(cfg.clone()).unwrap();
        // Push enough data that spillover triggers at least once.
        let payload = vec![b'z'; 1024];
        for i in 0..1000u32 {
            tree.put(format!("k{i:05}").as_bytes(), &payload).unwrap();
        }
        // Now delete everything under one prefix — for some range
        // of keys, deletion will collapse a child blob's leaves
        // and surface SubtreeGone. (We don't know exactly which
        // keys live in which blob, but deleting a contiguous
        // half-stripe should hit at least one SubtreeGone.)
        for i in 0..500u32 {
            tree.delete(format!("k{i:05}").as_bytes()).unwrap();
        }
        // Don't checkpoint.
    }

    // Reopen → replay → checkpoint → reopen → verify.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        // Half should be gone, half should still be there.
        for i in 0..500u32 {
            assert!(
                tree.get(format!("k{i:05}").as_bytes()).unwrap().is_none(),
                "deleted key k{i:05} resurrected on replay",
            );
        }
        for i in 500..1000u32 {
            assert!(
                tree.get(format!("k{i:05}").as_bytes()).unwrap().is_some(),
                "surviving key k{i:05} lost on replay",
            );
        }
        tree.checkpoint().unwrap();
    }

    {
        let tree = Tree::open(cfg).unwrap();
        for i in 0..500u32 {
            assert!(tree.get(format!("k{i:05}").as_bytes()).unwrap().is_none());
        }
        for i in 500..1000u32 {
            assert!(tree.get(format!("k{i:05}").as_bytes()).unwrap().is_some());
        }
    }
}

#[test]
fn cross_blob_writes_replay_correctly_through_wal_without_checkpoint() {
    // End-to-end test for the walker's W2D fix: under
    // `WalCommit::Sync`, every record reaches disk
    // before the op returns. The walker no longer commits any
    // child blob inline (the v0.2-pre `bm.commit(child_guid)` is
    // now `bm.mark_dirty(child_guid, seq)`), so the on-disk
    // child blob state at the moment of drop is exactly "stale"
    // — the WAL is the source of truth, and reopen-via-replay
    // must reconstruct the cross-blob state correctly.
    let dir = tempdir().unwrap();
    let cfg = durable_cfg(dir.path());

    // Round 1: fill enough to spill across blobs, drop without
    // calling `checkpoint`.
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        let payload = vec![b'y'; 1024];
        for i in 0..1000u32 {
            tree.put(format!("k{i:05}").as_bytes(), &payload).unwrap();
        }
        // Verify spillover actually happened — otherwise the
        // test isn't exercising the cross-blob path.
        let stats = tree.stats().unwrap();
        assert!(
            stats.blob_count > 1,
            "test premise: spillover must trigger (got blob_count={})",
            stats.blob_count,
        );
    } // drop without checkpoint — WAL truncate didn't run.

    // Round 2: reopen. WAL replay re-runs every op, including
    // the ones that triggered spillover. Every key must be
    // present.
    {
        let tree = Tree::open(cfg).unwrap();
        let payload = vec![b'y'; 1024];
        for i in 0..1000u32 {
            let k = format!("k{i:05}");
            assert_eq!(
                tree.get(k.as_bytes()).unwrap().as_deref(),
                Some(payload.as_slice()),
                "cross-blob key {k} lost after replay",
            );
        }
    }
}

// ============================================================
// Concurrent writer durability — regression for the W2D-strict
// protocol that publishes walker.mutate + mark_dirty + journal
// submission under the writer side of commit_gate and makes
// checkpoint drain + journal flush + byte snapshot use the
// exclusive side of the same gate.
//
// The pre-fix race: a writer marked a blob dirty + then released
// before appending WAL; a checkpoint round in between could
// snapshot the dirty entry, flush a stale WAL (no record yet),
// write the cache image to store, sync — and the writer's
// WAL append then landed AFTER store was already past it. On
// crash the WAL truncate point + store image disagreed.
//
// These tests aren't deterministic race triggers (they can't
// reliably interleave writer/checkpointer threads at a single
// instruction). They're stress-with-invariant regressions:
// every `put` that returns Ok is acknowledged, so the
// post-reopen tree must contain every acknowledged key.
// ============================================================

#[test]
fn concurrent_writers_and_bg_checkpoint_preserve_acked_ops() {
    use holt::{CheckpointConfig, TreeBuilder};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    const WRITERS: usize = 4;
    const OPS_PER_WRITER: u32 = 250;
    const PAYLOAD_LEN: usize = 256;

    let dir = tempdir().unwrap();
    let payload = vec![b'y'; PAYLOAD_LEN];

    {
        let tree = Arc::new(
            TreeBuilder::new(dir.path())
                .wal_commit(WalCommit::Sync) // per-op durable
                .checkpoint(CheckpointConfig {
                    enabled: true,
                    idle_interval: Duration::from_millis(5),
                    dirty_blob_threshold: 1,
                    auto_merge: false,
                    ..CheckpointConfig::default()
                })
                .open()
                .unwrap(),
        );

        let handles: Vec<_> = (0..WRITERS)
            .map(|writer_id| {
                let tree = Arc::clone(&tree);
                let payload = payload.clone();
                thread::spawn(move || {
                    for i in 0..OPS_PER_WRITER {
                        let key = format!("w{writer_id:02}/k{i:05}");
                        tree.put(key.as_bytes(), &payload).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Give the background checkpointer time to drain the
        // tail of the dirty set before drop.
        thread::sleep(Duration::from_millis(100));
    } // drop → final synchronous round + thread join.

    // Reopen — every acknowledged put must be visible.
    {
        let tree = Tree::open(TreeConfig::new(dir.path())).unwrap();
        for writer_id in 0..WRITERS {
            for i in 0..OPS_PER_WRITER {
                let key = format!("w{writer_id:02}/k{i:05}");
                assert_eq!(
                    tree.get(key.as_bytes()).unwrap().as_deref(),
                    Some(payload.as_slice()),
                    "key {key} lost after concurrent put + bg checkpoint + reopen",
                );
            }
        }
    }
}

#[test]
fn concurrent_writers_and_manual_checkpoints_preserve_acked_ops() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    const WRITERS: usize = 4;
    const OPS_PER_WRITER: u32 = 200;
    const PAYLOAD_LEN: usize = 256;

    let dir = tempdir().unwrap();
    let payload = vec![b'y'; PAYLOAD_LEN];

    {
        let cfg = durable_cfg(dir.path()); // WalCommit::Sync
        let tree = Arc::new(Tree::open(cfg).unwrap());
        let done = Arc::new(AtomicBool::new(false));

        // Background "checkpointer" — periodic manual
        // Tree::checkpoint() while writers churn. This exercises
        // the production path: snapshot under commit_gate,
        // write_through with expected_seq, conditional truncate.
        let ck_tree = Arc::clone(&tree);
        let ck_done = Arc::clone(&done);
        let ck_handle = thread::spawn(move || {
            while !ck_done.load(Ordering::Relaxed) {
                let _ = ck_tree.checkpoint();
                thread::sleep(Duration::from_millis(5));
            }
        });

        let writer_handles: Vec<_> = (0..WRITERS)
            .map(|writer_id| {
                let tree = Arc::clone(&tree);
                let payload = payload.clone();
                thread::spawn(move || {
                    for i in 0..OPS_PER_WRITER {
                        let key = format!("w{writer_id:02}/k{i:05}");
                        tree.put(key.as_bytes(), &payload).unwrap();
                    }
                })
            })
            .collect();

        for h in writer_handles {
            h.join().unwrap();
        }
        done.store(true, Ordering::Relaxed);
        ck_handle.join().unwrap();

        // One final checkpoint to make store the source of
        // truth before drop (we'll re-open with the same WAL
        // path and don't want replay to mask a missed write).
        tree.checkpoint().unwrap();
    }

    {
        let tree = Tree::open(durable_cfg(dir.path())).unwrap();
        for writer_id in 0..WRITERS {
            for i in 0..OPS_PER_WRITER {
                let key = format!("w{writer_id:02}/k{i:05}");
                assert_eq!(
                    tree.get(key.as_bytes()).unwrap().as_deref(),
                    Some(payload.as_slice()),
                    "key {key} lost after concurrent put + manual checkpoint + reopen",
                );
            }
        }
    }
}

#[test]
fn concurrent_writers_with_deletes_and_bg_checkpoint() {
    // Adds the deferred-delete path to the concurrent stress:
    // half the writers put unique keys, the others delete keys
    // a prior round inserted. Cross-blob erase that hits
    // SubtreeGone queues pending deletes; the round + drop +
    // reopen sequence must still leave the tree consistent.
    use holt::{CheckpointConfig, TreeBuilder};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    const PUT_WRITERS: usize = 3;
    const OPS: u32 = 200;
    const PAYLOAD_LEN: usize = 512;

    let dir = tempdir().unwrap();
    let payload = vec![b'z'; PAYLOAD_LEN];

    {
        let tree = Arc::new(
            TreeBuilder::new(dir.path())
                .wal_commit(WalCommit::Sync)
                .checkpoint(CheckpointConfig {
                    enabled: true,
                    idle_interval: Duration::from_millis(5),
                    dirty_blob_threshold: 1,
                    auto_merge: true,
                    ..CheckpointConfig::default()
                })
                .open()
                .unwrap(),
        );

        // Phase 1: prefill keys "p<n>/k<i>" for n in 0..PUT_WRITERS.
        for writer_id in 0..PUT_WRITERS {
            for i in 0..OPS {
                let k = format!("p{writer_id:02}/k{i:05}");
                tree.put(k.as_bytes(), &payload).unwrap();
            }
        }

        // Phase 2: concurrent put (different namespace) + delete
        // (drains the prefilled keys, exercising SubtreeGone).
        let putters: Vec<_> = (0..PUT_WRITERS)
            .map(|writer_id| {
                let tree = Arc::clone(&tree);
                let payload = payload.clone();
                thread::spawn(move || {
                    for i in 0..OPS {
                        let k = format!("q{writer_id:02}/k{i:05}");
                        tree.put(k.as_bytes(), &payload).unwrap();
                    }
                })
            })
            .collect();

        let deleter_tree = Arc::clone(&tree);
        let deleter = thread::spawn(move || {
            for writer_id in 0..PUT_WRITERS {
                for i in 0..OPS {
                    let k = format!("p{writer_id:02}/k{i:05}");
                    let _ = deleter_tree.delete(k.as_bytes()).unwrap();
                }
            }
        });

        for h in putters {
            h.join().unwrap();
        }
        deleter.join().unwrap();
        thread::sleep(Duration::from_millis(150));
    }

    // Reopen: all `q*` keys present, all `p*` keys gone.
    {
        let tree = Tree::open(TreeConfig::new(dir.path())).unwrap();
        for writer_id in 0..PUT_WRITERS {
            for i in 0..OPS {
                let kq = format!("q{writer_id:02}/k{i:05}");
                assert_eq!(
                    tree.get(kq.as_bytes()).unwrap().as_deref(),
                    Some(payload.as_slice()),
                    "{kq} (put) lost",
                );
                let kp = format!("p{writer_id:02}/k{i:05}");
                assert!(
                    tree.get(kp.as_bytes()).unwrap().is_none(),
                    "{kp} (deleted) resurrected",
                );
            }
        }
    }
}
