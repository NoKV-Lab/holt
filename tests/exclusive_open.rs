//! End-to-end tests for exclusive store-directory locking.
//!
//! Two live instances on one data directory corrupt the
//! [`FileBlobStore`] manifest: each replays `manifest.log` into the
//! same `next_slot`, assigns the same slot to different blob GUIDs,
//! and appends conflicting set deltas — after which every later
//! open fails with `FileBlobStore::Manifest::duplicate slot` and
//! the store is permanently unreadable. Since 0.5.0 even read-only
//! snapshots write frozen root frames through the blob store, so
//! the overlap window of a handover (`store = reopen(path)`) is
//! enough to trip it.
//!
//! These tests pin the fix: a second opener waits for the previous
//! instance to drop (handover) or fails cleanly, and the store
//! replays cleanly afterwards.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use holt::{Error, TreeConfig, DB};
use tempfile::tempdir;

#[test]
fn open_waits_for_live_instance_to_drop() {
    let dir = tempdir().unwrap();
    {
        let db = DB::open(TreeConfig::new(dir.path())).unwrap();
        let tree = db.create_tree("t").unwrap();
        tree.put(b"k", b"v").unwrap();
    }

    let (ready_tx, ready_rx) = mpsc::channel();
    let path = dir.path().to_path_buf();
    let holder = thread::spawn(move || {
        let db = DB::open(TreeConfig::new(path.clone())).unwrap();
        let tree = db.open_tree("t").unwrap();
        // A snapshot read is the exact operation that, before the
        // lock, persisted a frozen root frame from each of two
        // overlapping instances into the same manifest slot.
        let snap = tree.snapshot(b"").unwrap();
        assert_eq!(snap.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
        ready_tx.send(()).unwrap();
        thread::sleep(Duration::from_secs(1));
    });

    ready_rx.recv().unwrap();
    // The previous instance is still live: this open must serialize
    // behind its drop instead of going live concurrently.
    let started = Instant::now();
    let db = DB::open(TreeConfig::new(dir.path())).unwrap();
    assert!(
        started.elapsed() >= Duration::from_millis(300),
        "second open went live while the first instance held the store"
    );
    holder.join().unwrap();

    let tree = db.open_tree("t").unwrap();
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
    drop(tree);
    drop(db);

    // The 0.5.x bug poisoned the manifest so every later open
    // failed; the store must keep replaying cleanly.
    let db = DB::open(TreeConfig::new(dir.path())).unwrap();
    let tree = db.open_tree("t").unwrap();
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
}

#[test]
fn concurrent_open_of_live_store_fails_cleanly() {
    let dir = tempdir().unwrap();
    let db = DB::open(TreeConfig::new(dir.path())).unwrap();
    let tree = db.create_tree("t").unwrap();
    tree.put(b"k", b"v").unwrap();

    // Waits out the full lock-acquire timeout, then must fail
    // instead of going live on a store another instance holds.
    let err = match DB::open(TreeConfig::new(dir.path())) {
        Err(e) => e,
        Ok(_) => panic!("second open went live on a store another instance holds"),
    };
    match err {
        Error::BlobStoreIo(e) => assert_eq!(
            e.kind(),
            std::io::ErrorKind::WouldBlock,
            "unexpected I/O error: {e}"
        ),
        other => panic!("unexpected error variant: {other}"),
    }

    // The held instance keeps working, and the rejected opener
    // left no trace behind.
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
    drop(tree);
    drop(db);
    let db = DB::open(TreeConfig::new(dir.path())).unwrap();
    let tree = db.open_tree("t").unwrap();
    assert_eq!(tree.get(b"k").unwrap().as_deref(), Some(&b"v"[..]));
}
