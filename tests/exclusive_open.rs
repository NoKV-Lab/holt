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

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::process::{Child, Command};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use holt::{Error, FileStoreReservation, Tree, TreeConfig, DB};
use tempfile::tempdir;

const EXEC_CHILD_PATH: &str = "HOLT_EXCLUSIVE_OPEN_CHILD_PATH";
const EXEC_CHILD_READY: &str = "HOLT_EXCLUSIVE_OPEN_CHILD_READY";
const EXEC_RELATIVE_BASE: &str = "HOLT_EXCLUSIVE_OPEN_RELATIVE_BASE";

#[derive(Debug, PartialEq, Eq)]
struct AuthorityEntrySnapshot {
    name: String,
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    bytes: Vec<u8>,
}

fn snapshot_authority_entries(path: &Path) -> Vec<AuthorityEntrySnapshot> {
    let mut entries = std::fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            AuthorityEntrySnapshot {
                name: entry.file_name().into_string().unwrap(),
                device: metadata.dev(),
                inode: metadata.ino(),
                mode: metadata.mode(),
                links: metadata.nlink(),
                bytes: std::fs::read(entry.path()).unwrap(),
            }
        })
        .collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    entries
}

fn authority_test_config(path: &Path) -> TreeConfig {
    let mut cfg = TreeConfig::new(path);
    cfg.checkpoint.enabled = false;
    cfg
}

fn move_authority_except_lock(source: &Path, target: &Path) {
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "store.lock" {
            std::fs::rename(entry.path(), target.join(entry.file_name())).unwrap();
        }
    }
}

fn remove_authority_except_lock(path: &Path) {
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "store.lock" {
            std::fs::remove_file(entry.path()).unwrap();
        }
    }
}

fn create_authority_file(path: &Path, bytes: &[u8]) {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
}

fn append_torn_suffix(path: &Path) -> u64 {
    let original_len = std::fs::metadata(path).unwrap().len();
    let mut file = OpenOptions::new().append(true).open(path).unwrap();
    file.write_all(b"torn").unwrap();
    file.sync_all().unwrap();
    original_len
}

fn restore_file_len(path: &Path, len: u64) {
    let file = OpenOptions::new().write(true).open(path).unwrap();
    file.set_len(len).unwrap();
    file.sync_all().unwrap();
}

fn spawn_exec_opener(path: &Path, ready: &Path) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("exec_child_opens_reserved_store")
        .arg("--nocapture")
        .env(EXEC_CHILD_PATH, path)
        .env(EXEC_CHILD_READY, ready)
        .spawn()
        .unwrap()
}

fn wait_for_path(path: &Path, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while !path.exists() {
        assert!(Instant::now() < deadline, "timed out waiting for {path:?}");
        thread::sleep(Duration::from_millis(10));
    }
}

fn assert_child_blocked(child: &mut Child, duration: Duration, stage: &str) {
    thread::sleep(duration);
    assert!(
        child.try_wait().unwrap().is_none(),
        "exec child escaped file-store exclusion during {stage}"
    );
}

fn wait_for_child_success(child: &mut Child, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "exec opener failed with {status}");
            return;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("exec opener stayed blocked after the final guard dropped");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn exec_child_opens_reserved_store() {
    let Ok(path) = std::env::var(EXEC_CHILD_PATH) else {
        return;
    };
    let ready = std::env::var(EXEC_CHILD_READY).unwrap();
    std::fs::write(ready, b"ready").unwrap();
    drop(DB::open(TreeConfig::new(path)).unwrap());
}

#[test]
fn exec_child_relative_reservation_survives_chdir() {
    let Ok(base) = std::env::var(EXEC_RELATIVE_BASE) else {
        return;
    };
    let base = std::path::PathBuf::from(base);
    let old_cwd = base.join("old-cwd");
    let new_cwd = base.join("new-cwd");
    std::fs::create_dir(&old_cwd).unwrap();
    std::fs::create_dir(&new_cwd).unwrap();
    std::env::set_current_dir(&old_cwd).unwrap();

    let relative = std::path::PathBuf::from("store");
    let mut reservation = FileStoreReservation::acquire_for_create(&relative).unwrap();
    std::env::set_current_dir(&new_cwd).unwrap();
    std::fs::create_dir(new_cwd.join("store")).unwrap();

    let db =
        DB::open_with_file_store_reservation(authority_test_config(&relative), &mut reservation)
            .expect("chdir rebound a reserved relative locator to the new cwd lure");
    db.validate_file_store_object_set()
        .expect("live validation used the caller's new cwd instead of the frozen locator");
    assert!(snapshot_authority_entries(&new_cwd.join("store")).is_empty());
    let tree = db.create_tree("objects").unwrap();
    tree.put(b"key", b"pending-checkpoint").unwrap();
    drop(tree);

    let held = old_cwd.join("held-store");
    std::fs::rename(old_cwd.join("store"), &held).unwrap();
    std::fs::create_dir(old_cwd.join("store")).unwrap();
    let replacement_before = snapshot_authority_entries(&old_cwd.join("store"));
    let lure_before = snapshot_authority_entries(&new_cwd.join("store"));

    db.validate_file_store_object_set()
        .expect_err("live validation accepted replacement of the frozen old-cwd locator");
    db.checkpoint()
        .expect_err("name-based durability mutation ignored the replaced frozen locator");
    assert_eq!(
        snapshot_authority_entries(&old_cwd.join("store")),
        replacement_before
    );
    assert_eq!(
        snapshot_authority_entries(&new_cwd.join("store")),
        lure_before
    );

    std::fs::remove_dir(old_cwd.join("store")).unwrap();
    std::fs::rename(&held, old_cwd.join("store")).unwrap();
    db.validate_file_store_object_set()
        .expect("restoring the frozen absolute locator did not recover validation");
    db.checkpoint()
        .expect("restoring the frozen locator did not make the failed checkpoint retryable");
    drop(db);
}

#[test]
fn relative_reservation_is_frozen_in_an_exec_child() {
    let sandbox = tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("exec_child_relative_reservation_survives_chdir")
        .arg("--nocapture")
        .env(EXEC_RELATIVE_BASE, sandbox.path())
        .status()
        .unwrap();
    assert!(
        status.success(),
        "relative-locator exec child failed: {status}"
    );
}

#[test]
fn create_and_existing_reservations_preserve_open_intent() {
    let parent = tempdir().unwrap();
    let existing_empty = parent.path().join("existing-empty");
    std::fs::create_dir(&existing_empty).unwrap();
    let error = FileStoreReservation::acquire_for_create(&existing_empty)
        .expect_err("fresh reservation silently adopted an existing directory");
    assert!(matches!(
        error,
        Error::BlobStoreIo(error) if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert!(!existing_empty.join("store.lock").exists());

    let path = parent.path().join("existing-store");
    let identity = {
        let db = DB::open(TreeConfig::new(&path)).unwrap();
        db.file_store_object_identity().unwrap()
    };
    let missing = parent.path().join("missing-existing-store");
    FileStoreReservation::acquire_existing(&missing, identity)
        .expect_err("existing reservation silently created a missing store");
    assert!(!missing.exists());

    let mut reservation = FileStoreReservation::acquire_existing(&path, identity).unwrap();
    let db = DB::open_with_file_store_reservation(
        TreeConfig::new(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .unwrap();
    assert_eq!(db.file_store_object_identity(), Some(identity));
    drop(db);
}

#[test]
fn fresh_reservation_rejects_foreign_store_without_side_effects() {
    let sandbox = tempdir().unwrap();
    let source = sandbox.path().join("source");
    {
        let db = DB::open(authority_test_config(&source)).unwrap();
        let tree = db.create_tree("objects").unwrap();
        tree.put(b"key", b"foreign-value").unwrap();
        db.checkpoint().unwrap();
    }

    let target = sandbox.path().join("fresh-target");
    let mut reservation = FileStoreReservation::acquire_for_create(&target).unwrap();
    move_authority_except_lock(&source, &target);
    let before = snapshot_authority_entries(&target);

    DB::open_with_file_store_reservation(authority_test_config(&target), &mut reservation)
        .expect_err("fresh reservation adopted a foreign durable store");

    assert!(reservation.is_ready());
    reservation.validate().unwrap();
    assert_eq!(snapshot_authority_entries(&target), before);
}

#[test]
fn existing_reservation_rejects_missing_recovery_truth_without_side_effects() {
    let sandbox = tempdir().unwrap();
    let path = sandbox.path().join("existing");
    let identity = {
        let db = DB::open(authority_test_config(&path)).unwrap();
        let tree = db.create_tree("objects").unwrap();
        tree.put(b"key", b"must-not-disappear").unwrap();
        db.checkpoint().unwrap();
        db.file_store_object_identity().unwrap()
    };
    remove_authority_except_lock(&path);
    let before = snapshot_authority_entries(&path);

    FileStoreReservation::acquire_existing(&path, identity)
        .expect_err("existing reservation accepted a store with no recovery truth");

    assert_eq!(snapshot_authority_entries(&path), before);
}

#[test]
fn existing_tree_adoption_refuses_a_different_store_incarnation() {
    let sandbox = tempdir().unwrap();
    let path = sandbox.path().join("database");
    let identity = {
        let db = DB::open(authority_test_config(&path)).unwrap();
        let tree = db.create_tree("objects").unwrap();
        tree.put(b"key", b"value").unwrap();
        db.checkpoint().unwrap();
        db.file_store_object_identity().unwrap()
    };
    let mut reservation = FileStoreReservation::acquire_existing(&path, identity).unwrap();
    let before = snapshot_authority_entries(&path);

    Tree::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect_err("standalone Tree adoption initialized a root in an existing DB incarnation");

    assert!(reservation.is_ready());
    assert_eq!(snapshot_authority_entries(&path), before);
    let db = DB::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect("the unchanged existing reservation was not retryable as its DB incarnation");
    assert!(!reservation.is_ready());
    drop(db);
}

#[test]
fn existing_db_adoption_refuses_a_standalone_tree_incarnation() {
    let sandbox = tempdir().unwrap();
    let path = sandbox.path().join("tree");
    let identity = {
        let tree = Tree::open(authority_test_config(&path)).unwrap();
        tree.put(b"key", b"value").unwrap();
        tree.checkpoint().unwrap();
        tree.file_store_object_identity().unwrap()
    };
    let mut reservation = FileStoreReservation::acquire_existing(&path, identity).unwrap();
    let before = snapshot_authority_entries(&path);

    DB::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect_err("DB adoption initialized a catalog root in a standalone Tree incarnation");

    assert!(reservation.is_ready());
    assert_eq!(snapshot_authority_entries(&path), before);
    let tree = Tree::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect("the unchanged existing reservation was not retryable as its Tree incarnation");
    assert_eq!(tree.get(b"key").unwrap().as_deref(), Some(&b"value"[..]));
    assert!(!reservation.is_ready());
    drop(tree);
}

#[test]
fn wrong_db_adoption_does_not_repair_tree_manifest_or_wal() {
    let sandbox = tempdir().unwrap();
    let path = sandbox.path().join("tree-with-torn-recovery");
    let identity = {
        let tree = Tree::open(authority_test_config(&path)).unwrap();
        tree.put(b"key", b"tree-value").unwrap();
        tree.checkpoint().unwrap();
        tree.file_store_object_identity().unwrap()
    };
    let mut reservation = FileStoreReservation::acquire_existing(&path, identity).unwrap();
    let manifest_log = path.join("manifest.log");
    let wal = path.join("journal.wal");
    let manifest_len = append_torn_suffix(&manifest_log);
    let wal_len = append_torn_suffix(&wal);
    let torn = snapshot_authority_entries(&path);

    DB::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect_err("wrong-kind DB adoption repaired a standalone Tree recovery tail");
    assert!(reservation.is_ready());
    assert_eq!(snapshot_authority_entries(&path), torn);

    restore_file_len(&manifest_log, manifest_len);
    restore_file_len(&wal, wal_len);
    let tree = Tree::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect("restored Tree recovery truth was not retryable with the correct API kind");
    assert_eq!(
        tree.get(b"key").unwrap().as_deref(),
        Some(&b"tree-value"[..])
    );
    assert!(!reservation.is_ready());
    drop(tree);
}

#[test]
fn wrong_tree_adoption_does_not_repair_db_manifest() {
    let sandbox = tempdir().unwrap();
    let path = sandbox.path().join("db-with-torn-manifest");
    let identity = {
        let db = DB::open(authority_test_config(&path)).unwrap();
        let tree = db.create_tree("objects").unwrap();
        tree.put(b"key", b"db-value").unwrap();
        db.checkpoint().unwrap();
        db.file_store_object_identity().unwrap()
    };
    let mut reservation = FileStoreReservation::acquire_existing(&path, identity).unwrap();
    let manifest_log = path.join("manifest.log");
    let manifest_len = append_torn_suffix(&manifest_log);
    let torn = snapshot_authority_entries(&path);

    Tree::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect_err("wrong-kind Tree adoption repaired a DB manifest tail");
    assert!(reservation.is_ready());
    assert_eq!(snapshot_authority_entries(&path), torn);

    restore_file_len(&manifest_log, manifest_len);
    let db = DB::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .expect("restored DB recovery truth was not retryable with the correct API kind");
    let tree = db.open_tree("objects").unwrap();
    assert_eq!(tree.get(b"key").unwrap().as_deref(), Some(&b"db-value"[..]));
    assert!(!reservation.is_ready());
    drop(tree);
    drop(db);
}

#[test]
fn existing_tree_reservation_reopens_a_valid_incarnation() {
    let sandbox = tempdir().unwrap();
    let path = sandbox.path().join("tree");
    let identity = {
        let tree = Tree::open(authority_test_config(&path)).unwrap();
        tree.put(b"key", b"value").unwrap();
        tree.checkpoint().unwrap();
        tree.file_store_object_identity().unwrap()
    };
    let mut reservation = FileStoreReservation::acquire_existing(&path, identity).unwrap();

    let tree = Tree::open_with_file_store_reservation(
        authority_test_config(&path).with_expected_file_store_identity(identity),
        &mut reservation,
    )
    .unwrap();

    assert_eq!(tree.get(b"key").unwrap().as_deref(), Some(&b"value"[..]));
    assert!(!reservation.is_ready());
    drop(tree);
}

#[test]
fn ready_existing_binds_accelerator_inodes_across_retry() {
    let sandbox = tempdir().unwrap();
    let path = sandbox.path().join("database");
    let identity = {
        let db = DB::open(authority_test_config(&path)).unwrap();
        let tree = db.create_tree("objects").unwrap();
        tree.put(b"key", b"value").unwrap();
        db.checkpoint().unwrap();
        db.file_store_object_identity().unwrap()
    };
    let mut reservation = FileStoreReservation::acquire_existing(&path, identity).unwrap();
    let cfg = || authority_test_config(&path).with_expected_file_store_identity(identity);

    let read_index = path.join("read.idx");
    let original_read_index = path.join("read.idx.reservation-original");
    std::fs::rename(&read_index, &original_read_index).unwrap();
    create_authority_file(&read_index, b"foreign read accelerator");
    let replaced_read_set = snapshot_authority_entries(&path);
    DB::open_with_file_store_reservation(cfg(), &mut reservation)
        .expect_err("ReadyExisting adopted a replacement read.idx inode");
    assert!(reservation.is_ready());
    assert_eq!(snapshot_authority_entries(&path), replaced_read_set);
    std::fs::remove_file(&read_index).unwrap();
    std::fs::rename(&original_read_index, &read_index).unwrap();

    let value_segment = path.join("value.seg");
    let original_value_segment = path.join("value.seg.reservation-original");
    std::fs::rename(&value_segment, &original_value_segment).unwrap();
    let missing_value_set = snapshot_authority_entries(&path);
    DB::open_with_file_store_reservation(cfg(), &mut reservation)
        .expect_err("ReadyExisting recreated a missing value.seg during adoption");
    assert!(reservation.is_ready());
    assert_eq!(snapshot_authority_entries(&path), missing_value_set);

    create_authority_file(&value_segment, b"inserted value accelerator");
    let inserted_value_set = snapshot_authority_entries(&path);
    DB::open_with_file_store_reservation(cfg(), &mut reservation)
        .expect_err("ReadyExisting adopted a newly inserted value.seg inode");
    assert!(reservation.is_ready());
    assert_eq!(snapshot_authority_entries(&path), inserted_value_set);
    std::fs::remove_file(&value_segment).unwrap();
    std::fs::rename(&original_value_segment, &value_segment).unwrap();

    let db = DB::open_with_file_store_reservation(cfg(), &mut reservation)
        .expect("restoring both exact accelerator inodes did not make the token retryable");
    assert!(!reservation.is_ready());
    drop(db);
}

#[test]
fn reservation_adopts_the_same_open_file_description() {
    let parent = tempdir().unwrap();
    let path = parent.path().join("store");
    let mut reservation = FileStoreReservation::acquire_for_create(&path).unwrap();
    let cfg = TreeConfig::new(&path);

    let (opened_tx, opened_rx) = mpsc::sync_channel(1);
    let ordinary_cfg = cfg.clone();
    let ordinary = thread::spawn(move || {
        opened_tx.send(DB::open(ordinary_cfg)).unwrap();
    });

    assert!(
        opened_rx.recv_timeout(Duration::from_millis(200)).is_err(),
        "ordinary open bypassed the live pre-open reservation"
    );

    let started = Instant::now();
    let db = DB::open_with_file_store_reservation(cfg, &mut reservation).unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "reservation adoption reopened and contended on its own lock"
    );
    assert!(
        opened_rx.recv_timeout(Duration::from_millis(200)).is_err(),
        "ordinary open succeeded while the adopted DB remained live"
    );

    drop(db);
    opened_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("ordinary open did not resume after the adopted DB dropped")
        .unwrap();
    ordinary.join().unwrap();
}

#[test]
fn reservation_rejects_parent_path_replacement_before_adoption() {
    let sandbox = tempdir().unwrap();
    let parent = sandbox.path().join("configured-parent");
    let held_parent = sandbox.path().join("held-parent");
    std::fs::create_dir(&parent).unwrap();
    let path = parent.join("store-with-private-locator");
    let mut reservation = FileStoreReservation::acquire_for_create(&path).unwrap();
    assert!(
        !format!("{reservation:?}").contains(path.to_string_lossy().as_ref()),
        "reservation Debug exposed the configured locator"
    );

    std::fs::rename(&parent, &held_parent).unwrap();
    std::fs::create_dir(&parent).unwrap();
    std::fs::create_dir(parent.join("store-with-private-locator")).unwrap();

    DB::open_with_file_store_reservation(TreeConfig::new(&path), &mut reservation)
        .expect_err("reservation adopted a root reached through a replaced parent path");
    assert!(reservation.is_ready());
    for name in [
        "blobs.dat",
        "read.idx",
        "value.seg",
        "manifest.bin",
        "manifest.log",
        "journal.wal",
    ] {
        assert!(
            !held_parent
                .join("store-with-private-locator")
                .join(name)
                .exists(),
            "failed adoption touched {name} through the held parent"
        );
        assert!(
            !parent
                .join("store-with-private-locator")
                .join(name)
                .exists(),
            "failed adoption touched {name} through the replacement parent"
        );
    }

    std::fs::remove_dir(parent.join("store-with-private-locator")).unwrap();
    std::fs::remove_dir(&parent).unwrap();
    std::fs::rename(&held_parent, &parent).unwrap();
    let db = DB::open_with_file_store_reservation(TreeConfig::new(&path), &mut reservation)
        .expect("restored locator should permit retry with the same reservation");
    assert!(!reservation.is_ready());
    drop(db);
}

#[test]
fn exec_opener_waits_through_reservation_db_and_journal_guards() {
    let parent = tempdir().unwrap();
    let path = parent.path().join("store");
    let ready = parent.path().join("exec.ready");
    let mut reservation = FileStoreReservation::acquire_for_create(&path).unwrap();
    let mut child = spawn_exec_opener(&path, &ready);
    wait_for_path(&ready, Duration::from_secs(5));
    assert_child_blocked(
        &mut child,
        Duration::from_millis(200),
        "pre-open reservation",
    );

    let db =
        DB::open_with_file_store_reservation(TreeConfig::new(&path), &mut reservation).unwrap();
    assert_child_blocked(&mut child, Duration::from_millis(200), "adopted DB guard");
    let tree = db.create_tree("journal-guard").unwrap();
    drop(db);
    assert_child_blocked(
        &mut child,
        Duration::from_millis(200),
        "adopted DB/Journal resource guard",
    );

    drop(tree);
    wait_for_child_success(&mut child, Duration::from_secs(5));
}

#[test]
fn failed_adoption_keeps_reservation_locked_and_retryable() {
    let parent = tempdir().unwrap();
    let path = parent.path().join("store");
    let ready = parent.path().join("failed-open-exec.ready");
    let mut reservation = FileStoreReservation::acquire_for_create(&path).unwrap();
    let manifest = path.join("manifest.bin");
    let mut corrupt = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&manifest)
        .unwrap();
    corrupt.write_all(b"injected invalid manifest").unwrap();
    corrupt.sync_all().unwrap();
    drop(corrupt);

    DB::open_with_file_store_reservation(TreeConfig::new(&path), &mut reservation)
        .expect_err("injected manifest corruption did not fail DB adoption");
    assert!(reservation.is_ready());
    reservation.validate().unwrap();

    let mut child = spawn_exec_opener(&path, &ready);
    wait_for_path(&ready, Duration::from_secs(5));
    assert_child_blocked(
        &mut child,
        Duration::from_millis(200),
        "failed adoption reservation recovery",
    );

    std::fs::remove_file(manifest).unwrap();
    let db = DB::open_with_file_store_reservation(TreeConfig::new(&path), &mut reservation)
        .expect("same reservation was not retryable after fixing the injected fault");
    assert!(!reservation.is_ready());
    assert_child_blocked(
        &mut child,
        Duration::from_millis(200),
        "retried DB/Journal resource guard",
    );
    drop(db);
    wait_for_child_success(&mut child, Duration::from_secs(5));
}

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
