use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use holt::{Durability, Error, Tree, TreeBuilder, TreeConfig, DB};
use tempfile::tempdir;

const HELPER_PATH: &str = "HOLT_READ_ONLY_HELPER_PATH";
const HELPER_MODE: &str = "HOLT_READ_ONLY_HELPER_MODE";
const HELPER_EXPECT_SUCCESS: &str = "HOLT_READ_ONLY_HELPER_EXPECT_SUCCESS";

const WAL_FILE_MAGIC: u32 = 0x414C_4157;
const WAL_RECORD_MAGIC: u32 = 0x5243_4552;
const LEGACY_WAL_FORMAT_VERSION: u32 = 3;
const CURRENT_WAL_FORMAT_VERSION: u32 = 4;
const LEGACY_WAL_HEADER_SIZE: usize = 32;
const CURRENT_WAL_HEADER_SIZE: usize = 4096;
const INSERT_RECORD_TAG: u8 = 0;
const FIRST_DB_USER_TREE_ID: u64 = 1;

fn writable_config(path: &Path) -> TreeConfig {
    let mut cfg = TreeConfig::new(path);
    cfg.durability = Durability::Wal { sync: true };
    cfg.checkpoint.enabled = false;
    cfg
}

fn snapshot_files(path: &Path) -> BTreeMap<OsString, Vec<u8>> {
    fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), fs::read(entry.path()).unwrap())
        })
        .collect()
}

fn legacy_wal_header(tree_id: u64, created_at: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(LEGACY_WAL_HEADER_SIZE);
    bytes.extend_from_slice(&WAL_FILE_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&LEGACY_WAL_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&tree_id.to_le_bytes());
    bytes.extend_from_slice(&created_at.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    assert_eq!(bytes.len(), LEGACY_WAL_HEADER_SIZE);
    bytes
}

fn legacy_wal_with_insert(tree_id: u64, key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut bytes = legacy_wal_header(0, 0x0102_0304_0506_0708);
    let record_start = bytes.len();
    let body_len = 8 + 4 + key.len() + 4 + value.len();
    bytes.extend_from_slice(&WAL_RECORD_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&(body_len as u32).to_le_bytes());
    bytes.extend_from_slice(&1_000u64.to_le_bytes());
    bytes.push(INSERT_RECORD_TAG);
    bytes.extend_from_slice(&tree_id.to_le_bytes());
    bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
    bytes.extend_from_slice(value);
    let checksum = crc32fast::hash(&bytes[record_start..]);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes
}

fn assert_nonempty_v3_error(error: &Error) {
    assert!(matches!(
        error,
        Error::ReplaySanityFailed {
            context: "nonempty WAL format 3 is replay-only; checkpoint it with a format-3 Holt binary before v4 writes",
            record_offset: 0,
        }
    ));
}

#[test]
fn read_only_open_replays_wal_without_changing_files() {
    let dir = tempdir().unwrap();
    {
        let tree = Tree::open(writable_config(dir.path())).unwrap();
        tree.put(b"objects/a", b"etag-a").unwrap();
    }

    let before = snapshot_files(dir.path());
    {
        let tree = TreeBuilder::new(dir.path()).read_only().open().unwrap();
        assert!(tree.config().is_read_only());
        assert_eq!(
            tree.get(b"objects/a").unwrap().as_deref(),
            Some(&b"etag-a"[..])
        );
        tree.view(b"objects/", |view| {
            assert_eq!(view.get(b"objects/a")?.as_deref(), Some(&b"etag-a"[..]));
            Ok(())
        })
        .unwrap();
        assert!(matches!(
            tree.put(b"objects/b", b"etag-b"),
            Err(Error::ReadOnly)
        ));
        assert!(matches!(tree.delete(b"objects/a"), Err(Error::ReadOnly)));
        assert!(matches!(tree.atomic(|_| {}), Err(Error::ReadOnly)));
        assert!(matches!(tree.checkpoint(), Err(Error::ReadOnly)));
        assert!(matches!(tree.compact(), Err(Error::ReadOnly)));
        assert!(matches!(tree.gc(), Err(Error::ReadOnly)));
        assert!(matches!(tree.vacuum(), Err(Error::ReadOnly)));
    }
    assert_eq!(snapshot_files(dir.path()), before);
}

#[test]
fn read_only_open_requires_existing_files() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("missing");
    let error = TreeBuilder::new(&missing).read_only().open().unwrap_err();
    assert!(matches!(error, Error::BlobStoreIo(_)));
    assert!(!missing.exists());
}

#[test]
fn read_only_open_allows_missing_optional_read_accelerators() {
    let dir = tempdir().unwrap();
    {
        let tree = Tree::open(writable_config(dir.path())).unwrap();
        tree.put(b"objects/a", b"etag-a").unwrap();
        tree.checkpoint().unwrap();
    }
    fs::remove_file(dir.path().join("read.idx")).unwrap();
    fs::remove_file(dir.path().join("value.seg")).unwrap();

    let before = snapshot_files(dir.path());
    {
        let tree = TreeBuilder::new(dir.path()).read_only().open().unwrap();
        assert_eq!(
            tree.get(b"objects/a").unwrap().as_deref(),
            Some(&b"etag-a"[..])
        );
    }
    assert_eq!(snapshot_files(dir.path()), before);
}

#[test]
fn read_only_database_replays_named_trees_and_rejects_mutations() {
    let dir = tempdir().unwrap();
    {
        let db = DB::open(writable_config(dir.path())).unwrap();
        let objects = db.create_tree("objects").unwrap();
        objects.put(b"a", b"etag-a").unwrap();
    }

    let before = snapshot_files(dir.path());
    let db = DB::open(writable_config(dir.path()).read_only()).unwrap();
    assert_eq!(db.list_trees().unwrap(), vec!["objects"]);
    let objects = db.open_tree("objects").unwrap();
    assert_eq!(objects.get(b"a").unwrap().as_deref(), Some(&b"etag-a"[..]));
    db.view(&[("objects", b"")], |view| {
        assert_eq!(
            view.tree("objects").unwrap().get(b"a")?.as_deref(),
            Some(&b"etag-a"[..])
        );
        Ok(())
    })
    .unwrap();
    let image = db.export_checkpoint().unwrap();

    assert!(matches!(objects.put(b"b", b"etag-b"), Err(Error::ReadOnly)));
    assert!(matches!(db.create_tree("new"), Err(Error::ReadOnly)));
    assert!(matches!(db.drop_tree("objects"), Err(Error::ReadOnly)));
    assert!(matches!(
        db.atomic(|_| {}),
        Err(Error::Atomic {
            kind: holt::AtomicErrorKind::DefinitelyNotApplied,
            source,
        }) if matches!(source.as_ref(), Error::ReadOnly)
    ));
    assert!(matches!(
        db.install_checkpoint(&image),
        Err(Error::ReadOnly)
    ));
    assert!(matches!(db.checkpoint(), Err(Error::ReadOnly)));
    assert!(matches!(db.compact(), Err(Error::ReadOnly)));
    assert!(matches!(db.gc(), Err(Error::ReadOnly)));
    assert!(matches!(db.vacuum(), Err(Error::ReadOnly)));
    drop(objects);
    drop(db);
    assert_eq!(snapshot_files(dir.path()), before);
}

#[test]
fn read_only_database_validates_and_scans_attached_journal() {
    let dir = tempdir().unwrap();
    let genesis = holt::JournalAnchor::new(0, [0x50; 32]);
    let first = holt::JournalAnchor::new(1, [0x51; 32]);
    let envelope =
        holt::JournalEnvelope::new(genesis, first, b"read-only recovery command".to_vec()).unwrap();

    {
        let db = DB::open(writable_config(dir.path())).unwrap();
        db.create_tree("metadata").unwrap();
        db.checkpoint().unwrap();
        db.initialize_journal_stream(genesis).unwrap();
        assert!(db
            .atomic_with_journal_envelope(envelope.clone(), |batch| {
                batch.put("metadata", b"key", b"value");
            })
            .unwrap());
    }

    let before = snapshot_files(dir.path());
    let db = DB::open(writable_config(dir.path()).read_only()).unwrap();
    assert_eq!(db.journal_state().unwrap().checkpoint(), genesis);
    assert_eq!(db.journal_state().unwrap().tail(), first);
    let page = db.journal_envelopes_after(genesis, 8, 4096).unwrap();
    assert_eq!(page.envelopes(), &[envelope]);
    assert_eq!(page.next(), first);
    assert!(!page.has_more());
    drop(db);
    assert_eq!(snapshot_files(dir.path()), before);
}

#[test]
fn nonempty_v3_tree_is_replay_only_and_writable_open_changes_nothing() {
    let dir = tempdir().unwrap();
    {
        let tree = Tree::open(writable_config(dir.path())).unwrap();
        tree.put(b"checkpointed", b"tree-value").unwrap();
        tree.checkpoint().unwrap();
    }

    fs::write(
        dir.path().join("journal.wal"),
        legacy_wal_with_insert(0, b"from-v3", b"tree-replay"),
    )
    .unwrap();
    let before = snapshot_files(dir.path());

    let error = Tree::open(writable_config(dir.path())).unwrap_err();
    assert_nonempty_v3_error(&error);
    assert_eq!(snapshot_files(dir.path()), before);

    let tree = TreeBuilder::new(dir.path()).read_only().open().unwrap();
    assert_eq!(
        tree.get(b"checkpointed").unwrap().as_deref(),
        Some(&b"tree-value"[..])
    );
    assert_eq!(
        tree.get(b"from-v3").unwrap().as_deref(),
        Some(&b"tree-replay"[..])
    );
    drop(tree);
    assert_eq!(snapshot_files(dir.path()), before);
}

#[test]
fn nonempty_v3_database_is_replay_only_and_read_only_export_includes_replay() {
    let dir = tempdir().unwrap();
    {
        let db = DB::open(writable_config(dir.path())).unwrap();
        let objects = db.create_tree("objects").unwrap();
        objects.put(b"checkpointed", b"db-value").unwrap();
        db.checkpoint().unwrap();
    }

    // A fresh DB assigns tree id 1 to its first named tree. That id is part
    // of this legacy wire fixture, not a public API assumption by callers.
    fs::write(
        dir.path().join("journal.wal"),
        legacy_wal_with_insert(FIRST_DB_USER_TREE_ID, b"from-v3", b"db-replay"),
    )
    .unwrap();
    let before = snapshot_files(dir.path());

    let error = DB::open(writable_config(dir.path())).unwrap_err();
    assert_nonempty_v3_error(&error);
    assert_eq!(snapshot_files(dir.path()), before);

    let db = DB::open(writable_config(dir.path()).read_only()).unwrap();
    let objects = db.open_tree("objects").unwrap();
    assert_eq!(
        objects.get(b"checkpointed").unwrap().as_deref(),
        Some(&b"db-value"[..])
    );
    assert_eq!(
        objects.get(b"from-v3").unwrap().as_deref(),
        Some(&b"db-replay"[..])
    );
    let image = db.export_checkpoint().unwrap();
    drop(objects);
    drop(db);
    assert_eq!(snapshot_files(dir.path()), before);

    let restored_dir = tempdir().unwrap();
    let restored = DB::open(writable_config(restored_dir.path())).unwrap();
    restored.install_checkpoint(&image).unwrap();
    let restored_objects = restored.open_tree("objects").unwrap();
    assert_eq!(
        restored_objects.get(b"checkpointed").unwrap().as_deref(),
        Some(&b"db-value"[..])
    );
    assert_eq!(
        restored_objects.get(b"from-v3").unwrap().as_deref(),
        Some(&b"db-replay"[..])
    );
}

#[test]
fn header_only_v3_tree_and_database_upgrade_to_v4_on_writable_open() {
    let tree_dir = tempdir().unwrap();
    {
        let tree = Tree::open(writable_config(tree_dir.path())).unwrap();
        tree.put(b"checkpointed", b"tree-value").unwrap();
        tree.checkpoint().unwrap();
    }
    let tree_wal = tree_dir.path().join("journal.wal");
    fs::write(&tree_wal, legacy_wal_header(0, 41)).unwrap();
    let tree_before = snapshot_files(tree_dir.path());

    let tree = Tree::open(writable_config(tree_dir.path())).unwrap();
    assert_eq!(
        tree.get(b"checkpointed").unwrap().as_deref(),
        Some(&b"tree-value"[..])
    );
    drop(tree);
    assert_v4_header_only_upgrade(&tree_wal, 41);
    assert_non_wal_files_unchanged(tree_dir.path(), &tree_before);

    let db_dir = tempdir().unwrap();
    {
        let db = DB::open(writable_config(db_dir.path())).unwrap();
        let objects = db.create_tree("objects").unwrap();
        objects.put(b"checkpointed", b"db-value").unwrap();
        db.checkpoint().unwrap();
    }
    let db_wal = db_dir.path().join("journal.wal");
    fs::write(&db_wal, legacy_wal_header(0, 43)).unwrap();
    let db_before = snapshot_files(db_dir.path());

    let db = DB::open(writable_config(db_dir.path())).unwrap();
    let objects = db.open_tree("objects").unwrap();
    assert_eq!(
        objects.get(b"checkpointed").unwrap().as_deref(),
        Some(&b"db-value"[..])
    );
    drop(objects);
    drop(db);
    assert_v4_header_only_upgrade(&db_wal, 43);
    assert_non_wal_files_unchanged(db_dir.path(), &db_before);
}

fn assert_v4_header_only_upgrade(wal_path: &Path, created_at: u64) {
    let bytes = fs::read(wal_path).unwrap();
    assert_eq!(bytes.len(), CURRENT_WAL_HEADER_SIZE);
    assert_eq!(
        u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        WAL_FILE_MAGIC
    );
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        CURRENT_WAL_FORMAT_VERSION
    );
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0);
    assert_eq!(
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        created_at
    );
    assert_eq!(
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        CURRENT_WAL_HEADER_SIZE as u64
    );
}

fn assert_non_wal_files_unchanged(path: &Path, before: &BTreeMap<OsString, Vec<u8>>) {
    let after = snapshot_files(path);
    for (name, bytes) in before {
        if name != "journal.wal" {
            assert_eq!(
                after.get(name),
                Some(bytes),
                "{} changed",
                name.to_string_lossy()
            );
        }
    }
}

#[test]
fn read_only_open_does_not_repair_a_torn_manifest_tail() {
    let dir = tempdir().unwrap();
    {
        let tree = Tree::open(writable_config(dir.path())).unwrap();
        tree.put(b"objects/a", b"etag-a").unwrap();
        tree.checkpoint().unwrap();
    }

    let log_path = dir.path().join("manifest.log");
    let valid_len = fs::metadata(&log_path).unwrap().len();
    OpenOptions::new()
        .append(true)
        .open(&log_path)
        .unwrap()
        .write_all(b"torn")
        .unwrap();
    let torn = fs::read(&log_path).unwrap();

    {
        let tree = TreeBuilder::new(dir.path()).read_only().open().unwrap();
        assert_eq!(
            tree.get(b"objects/a").unwrap().as_deref(),
            Some(&b"etag-a"[..])
        );
    }
    assert_eq!(fs::read(&log_path).unwrap(), torn);

    drop(Tree::open(writable_config(dir.path())).unwrap());
    assert_eq!(fs::metadata(&log_path).unwrap().len(), valid_len);
}

#[test]
fn read_only_open_does_not_repair_a_torn_wal_tail() {
    let dir = tempdir().unwrap();
    {
        let tree = Tree::open(writable_config(dir.path())).unwrap();
        tree.put(b"objects/a", b"etag-a").unwrap();
    }

    let wal_path = dir.path().join("journal.wal");
    let valid_len = fs::metadata(&wal_path).unwrap().len();
    OpenOptions::new()
        .append(true)
        .open(&wal_path)
        .unwrap()
        .write_all(b"torn")
        .unwrap();
    let torn = fs::read(&wal_path).unwrap();

    {
        let tree = TreeBuilder::new(dir.path()).read_only().open().unwrap();
        assert_eq!(
            tree.get(b"objects/a").unwrap().as_deref(),
            Some(&b"etag-a"[..])
        );
    }
    assert_eq!(fs::read(&wal_path).unwrap(), torn);

    drop(Tree::open(writable_config(dir.path())).unwrap());
    assert_eq!(fs::metadata(&wal_path).unwrap().len(), valid_len);
}

#[test]
fn process_lock_allows_readers_and_excludes_writers() {
    let dir = tempdir().unwrap();
    {
        let tree = Tree::open(writable_config(dir.path())).unwrap();
        tree.put(b"ready", b"1").unwrap();
        tree.checkpoint().unwrap();
    }

    let reader = TreeBuilder::new(dir.path()).read_only().open().unwrap();
    run_lock_helper(dir.path(), "read", true);
    run_lock_helper(dir.path(), "write", false);
    drop(reader);

    run_lock_helper(dir.path(), "write", true);

    let writer = Tree::open(writable_config(dir.path())).unwrap();
    run_lock_helper(dir.path(), "read", false);
    drop(writer);
}

fn run_lock_helper(path: &Path, mode: &str, expect_success: bool) {
    let output = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("lock_open_helper")
        .arg("--nocapture")
        .env(HELPER_PATH, path)
        .env(HELPER_MODE, mode)
        .env(
            HELPER_EXPECT_SUCCESS,
            if expect_success { "1" } else { "0" },
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "lock helper failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn lock_open_helper() {
    let Some(path) = std::env::var_os(HELPER_PATH) else {
        return;
    };
    let mode = std::env::var(HELPER_MODE).unwrap();
    let expect_success = std::env::var(HELPER_EXPECT_SUCCESS).unwrap() == "1";
    let result = match mode.as_str() {
        "read" => TreeBuilder::new(&path).read_only().open(),
        "write" => Tree::open(writable_config(Path::new(&path))),
        other => panic!("unknown helper mode: {other}"),
    };

    if expect_success {
        assert!(result.is_ok(), "open failed: {}", result.unwrap_err());
    } else {
        let error = result.unwrap_err().to_string();
        assert!(error.contains("incompatible live access mode"), "{error}");
    }
}
