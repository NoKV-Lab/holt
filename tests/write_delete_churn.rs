//! Regression: writes must not fail with `blob ... is pending delete`
//! (and must not hang) while concurrent delete churn drives merges that
//! de-route blobs. The merge pass marks a folded-away child for deferred
//! delete; if a concurrent write follows a stale route to that child it
//! surfaced the deferred-delete sentinel as a hard error instead of
//! restarting its descent (nightly db-normal soak). An out-of-thread
//! watchdog turns a hang into a test failure instead of a stuck suite.

use holt::{Durability, Tree, TreeConfig};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

fn k(idx: u64) -> Vec<u8> {
    format!(
        "bucket-{:03}/tenant-{:02}/path/object-{:010}.bin",
        idx % 256,
        idx % 32,
        idx
    )
    .into_bytes()
}

const N: u64 = 40_000;

fn run_churn() -> Result<u64, String> {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = TreeConfig::new(dir.path());
    cfg.durability = Durability::Wal { sync: false };
    // Small pool → eviction + the cold-read route cache are engaged, and
    // delete churn keeps the merge pass de-routing blobs.
    cfg.buffer_pool_size = 128;
    let tree = Arc::new(Tree::open(cfg).map_err(|e| format!("open: {e}"))?);
    for i in 0..N {
        tree.put(&k(i), b"v").map_err(|e| format!("seed: {e}"))?;
    }

    let stop = Arc::new(AtomicBool::new(false));
    let ops = Arc::new(AtomicU64::new(0));
    let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let mut handles = Vec::new();
    for w in 0..8u64 {
        let tree = Arc::clone(&tree);
        let stop = Arc::clone(&stop);
        let ops = Arc::clone(&ops);
        let failure = Arc::clone(&failure);
        handles.push(thread::spawn(move || {
            let mut i = w;
            while !stop.load(Ordering::Relaxed) {
                // Churn the seeded range so blobs empty and refill, which
                // makes them merge candidates (→ de-route → mark_for_delete).
                let key = k(i % N);
                let r = if (i / N) % 2 == 0 {
                    tree.atomic(|b| b.delete(&key))
                } else {
                    tree.atomic(|b| b.put(&key, b"v"))
                };
                if let Err(e) = r {
                    *failure.lock().unwrap() = Some(format!("{e}"));
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
                ops.fetch_add(1, Ordering::Relaxed);
                i += 8;
            }
        }));
    }

    thread::sleep(Duration::from_secs(15));
    stop.store(true, Ordering::Relaxed);
    // A livelocked worker stuck inside an op never observes `stop`; its
    // join hangs and the out-of-thread watchdog fails the test.
    for h in handles {
        h.join().unwrap();
    }
    if let Some(e) = failure.lock().unwrap().take() {
        return Err(e);
    }
    Ok(ops.load(Ordering::Relaxed))
}

#[test]
fn writes_progress_under_delete_churn() {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(run_churn());
    });
    match rx.recv_timeout(Duration::from_secs(50)) {
        Ok(Ok(ops)) => assert!(ops > 5_000, "writes stalled under churn: only {ops} ops"),
        Ok(Err(e)) => panic!("write failed under delete churn: {e}"),
        Err(_) => panic!("write/delete churn hung (a worker never completed)"),
    }
}
