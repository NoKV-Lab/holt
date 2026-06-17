//! Diagnostic + regression: range scans must make progress under
//! concurrent writes. The optimistic cursor restarts whenever a writer
//! rewrites a blob on its descent path; without a bounded fallback that
//! fences the churn, a long scan can spin forever (the nightly db-normal
//! soak hung 45 min in `RangeIter::pin_scan_or_restart`). An in-test
//! watchdog fails the test instead of hanging the suite if the
//! regression returns.

use holt::{Durability, Tree, TreeConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

/// Soak-shaped key: spreads across 256 buckets so keys do not pack
/// pathologically into a single blob frame's slot table.
fn k(idx: u64) -> Vec<u8> {
    format!(
        "bucket-{:03}/tenant-{:02}/path/object-{:010}.bin",
        idx % 256,
        idx % 32,
        idx
    )
    .into_bytes()
}

const SEED: u64 = 60_000;
const PREFIX: &[u8] = b"bucket-";

fn seeded() -> (tempfile::TempDir, Arc<Tree>) {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = TreeConfig::new(dir.path());
    cfg.durability = Durability::Wal { sync: false };
    cfg.buffer_pool_size = 2048;
    let tree = Arc::new(Tree::open(cfg).unwrap());
    for i in 0..SEED {
        tree.put(&k(i), b"v").unwrap();
    }
    (dir, tree)
}

fn spawn_writers(
    tree: &Arc<Tree>,
    stop: &Arc<AtomicBool>,
    n: usize,
) -> Vec<thread::JoinHandle<()>> {
    (0..n)
        .map(|w| {
            let tree = Arc::clone(tree);
            let stop = Arc::clone(stop);
            thread::spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    // Atomic batch = exclusive `enter_batch`, matching the
                    // db-normal soak's write path (db.atomic).
                    let _ = tree.atomic(|b| b.put(&k(w as u64 * 9_000_000 + i), b"v"));
                    i += 1;
                }
            })
        })
        .collect()
}

/// Run `scan` on a worker thread; fail (don't hang) if it stalls.
fn with_watchdog<F: FnOnce() + Send + 'static>(label: &'static str, scan: F) {
    let (tx, rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        scan();
        let _ = tx.send(());
    });
    if rx.recv_timeout(Duration::from_secs(25)).is_err() {
        panic!("{label}: scans made no progress within 25s — livelock");
    }
    worker.join().unwrap();
}

#[test]
fn live_scan_progresses_under_writes() {
    let (_dir, tree) = seeded();
    let stop = Arc::new(AtomicBool::new(false));
    let writers = spawn_writers(&tree, &stop, 4);
    let t = Arc::clone(&tree);
    with_watchdog("live", move || {
        for _ in 0..20 {
            let mut n = 0u64;
            t.scan_keys(PREFIX)
                .visit(usize::MAX, |_| {
                    n += 1;
                    Ok(())
                })
                .unwrap();
            assert!(n > 0, "live scan returned nothing");
        }
    });
    stop.store(true, Ordering::Relaxed);
    for h in writers {
        h.join().unwrap();
    }
}

#[test]
fn view_scan_progresses_under_writes() {
    let (_dir, tree) = seeded();
    let stop = Arc::new(AtomicBool::new(false));
    let writers = spawn_writers(&tree, &stop, 4);
    let t = Arc::clone(&tree);
    with_watchdog("view", move || {
        for _ in 0..20 {
            let n: u64 = t
                .view(PREFIX, |v| {
                    let mut n = 0u64;
                    v.scan_keys(PREFIX)?.visit(usize::MAX, |_| {
                        n += 1;
                        Ok(())
                    })?;
                    Ok(n)
                })
                .unwrap();
            assert!(n > 0, "view scan returned nothing");
        }
    });
    stop.store(true, Ordering::Relaxed);
    for h in writers {
        h.join().unwrap();
    }
}
