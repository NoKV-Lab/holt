//! Concurrent-write **correctness** stress harness.
//!
//! holt's write path is lock-coupled with exclusive blob latches, so
//! today concurrent writes are serialized (correct but they serialize
//! on the root blob's latch — see PERF_FINDINGS.md). This harness is
//! the *gate* for the planned optimistic-write-descent change: it
//! hammers the write path from many threads with the background
//! journal + checkpointer running and asserts no write is lost,
//! torn, or corrupted. It must stay green before and after that
//! change — a racy optimistic descent would surface here even though
//! the single-threaded proptest / failpoint suites cannot catch it.
//!
//! Run: `cargo test --test concurrent_stress -- --nocapture`
//! (or `--test-threads=1` to keep the three scenarios from contending
//! for the box; each scenario is itself multi-threaded).

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use holt::{CheckpointConfig, Tree, TreeBuilder};
use tempfile::TempDir;

/// A durable tree with the background checkpointer + journal running,
/// WAL without per-op fsync (the "hot service" profile the perf work
/// measured). Returns `(dir, tree)` with `dir` first so the `Tree`
/// (and its checkpointer thread) drops *before* the temp dir is
/// removed — avoiding the teardown race that the bench hit.
fn durable_tree() -> (TempDir, Arc<Tree>) {
    let dir = TempDir::new().expect("tempdir");
    let tree = TreeBuilder::new(dir.path())
        .checkpoint(CheckpointConfig {
            enabled: true,
            idle_interval: Duration::from_millis(20),
            dirty_blob_threshold: 1,
            auto_merge: true,
            ..CheckpointConfig::default()
        })
        .open()
        .expect("open durable tree");
    (dir, Arc::new(tree))
}

fn key(thread: usize, i: usize) -> Vec<u8> {
    // Path-shaped, thread-partitioned so the disjoint scenarios have a
    // deterministic expected final state.
    format!("t{thread:02}/shard/{i:06}").into_bytes()
}

fn val(thread: usize, i: usize, rev: u64) -> Vec<u8> {
    format!("v-{thread}-{i}-{rev}").into_bytes()
}

/// N threads write DISJOINT key ranges concurrently; afterwards every
/// key must read back exactly the value its owning thread wrote. A
/// lost or cross-thread-corrupted write fails here.
#[test]
fn concurrent_disjoint_writes_all_durable() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 4_000; // 32k keys -> spans several blobs

    let (_dir, tree) = durable_tree();
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let tree = tree.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..PER_THREAD {
                tree.put(&key(t, i), &val(t, i, 1)).expect("put");
            }
        }));
    }
    for h in handles {
        h.join().expect("thread");
    }

    for t in 0..THREADS {
        for i in 0..PER_THREAD {
            assert_eq!(
                tree.get(&key(t, i)).expect("get").as_deref(),
                Some(val(t, i, 1).as_slice()),
                "lost/corrupt write at t{t} i{i}",
            );
        }
    }
}

/// Disjoint ranges, but each thread put → overwrite → delete-evens.
/// Final state is deterministic: odd `i` present at revision 2, even
/// `i` absent. Stresses the realloc + tombstone write paths under
/// concurrency.
#[test]
fn concurrent_put_overwrite_delete_is_consistent() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 3_000;

    let (_dir, tree) = durable_tree();
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let tree = tree.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..PER_THREAD {
                tree.put(&key(t, i), &val(t, i, 1)).expect("put1");
            }
            for i in 0..PER_THREAD {
                // grow the value to force a realloc-not-in-place path
                tree.put(&key(t, i), &val(t, i, 2_000_000_000)).expect("put2");
            }
            for i in (0..PER_THREAD).step_by(2) {
                assert!(tree.delete(&key(t, i)).expect("delete"));
            }
        }));
    }
    for h in handles {
        h.join().expect("thread");
    }

    for t in 0..THREADS {
        for i in 0..PER_THREAD {
            let got = tree.get(&key(t, i)).expect("get");
            if i % 2 == 0 {
                assert!(got.is_none(), "deleted key present at t{t} i{i}");
            } else {
                assert_eq!(
                    got.as_deref(),
                    Some(val(t, i, 2_000_000_000).as_slice()),
                    "wrong value at t{t} i{i}",
                );
            }
        }
    }
}

/// All threads write the SAME shared keys (maximum latch contention on
/// the same blobs + the root) with self-describing values. Final
/// state need not be deterministic (last-writer-wins races), but every
/// key must hold a *valid, non-torn* value written by *some* thread —
/// catching torn/corrupted writes under contention.
#[test]
fn concurrent_overlapping_writes_no_torn_values() {
    const THREADS: usize = 8;
    const SHARED_KEYS: usize = 2_000;
    const ROUNDS: usize = 3;

    let (_dir, tree) = durable_tree();
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let tree = tree.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            for r in 0..ROUNDS {
                for i in 0..SHARED_KEYS {
                    // shared key (no thread in the key) -> all threads collide
                    let k = format!("shared/{i:06}").into_bytes();
                    tree.put(&k, &val(t, i, r as u64)).expect("put");
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread");
    }

    for i in 0..SHARED_KEYS {
        let k = format!("shared/{i:06}").into_bytes();
        let got = tree.get(&k).expect("get").expect("shared key must exist");
        // Value must parse as a well-formed v-<t>-<i>-<r> tag from some
        // writer for THIS key index — i.e. not torn across two writes.
        let s = String::from_utf8(got).expect("value not valid utf8 (torn write)");
        let parts: Vec<&str> = s.trim_start_matches("v-").split('-').collect();
        assert_eq!(parts.len(), 3, "malformed value {s:?} (torn write)");
        let wt: usize = parts[0].parse().expect("bad thread tag (torn)");
        let wi: usize = parts[1].parse().expect("bad index tag (torn)");
        let wr: u64 = parts[2].parse().expect("bad round tag (torn)");
        assert!(wt < THREADS, "impossible thread tag {wt}");
        assert_eq!(wi, i, "value belongs to a different key (corruption): {s:?}");
        assert!((wr as usize) < ROUNDS, "impossible round tag {wr}");
    }
}
