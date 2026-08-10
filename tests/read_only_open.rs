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
    assert!(matches!(db.atomic(|_| {}), Err(Error::ReadOnly)));
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
