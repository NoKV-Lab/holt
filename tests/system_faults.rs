//! System-level file-store fault checks that are cheap and stable in CI.
//!
//! These complement `checkpoint_failpoint`: failpoints test protocol
//! recovery without relying on the host filesystem, while these tests
//! cover real file corruption and directory-level failures through the
//! public API.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use holt::{CheckpointConfig, Durability, Tree, TreeConfig};
use tempfile::tempdir;

fn cfg(path: &Path) -> TreeConfig {
    let mut cfg = TreeConfig::new(path);
    cfg.buffer_pool_size = 8;
    cfg.durability = Durability::Wal { sync: true };
    cfg.checkpoint = CheckpointConfig {
        enabled: false,
        ..CheckpointConfig::default()
    };
    cfg
}

fn build_checkpointed_tree(path: &Path) -> TreeConfig {
    let cfg = cfg(path);
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        for i in 0..256u32 {
            let key = format!("bucket/path/file-{i:04}");
            tree.put(key.as_bytes(), b"value").unwrap();
        }
        tree.checkpoint().unwrap();
    }
    cfg
}

fn flip_first_byte(path: &Path) {
    flip_byte_at(path, 0);
}

fn flip_byte_at(path: &Path, offset: u64) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut byte = [0u8; 1];
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 0xff;
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&byte).unwrap();
    file.sync_all().unwrap();
}

fn write_bytes_at(path: &Path, offset: u64, bytes: &[u8]) {
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
}

fn corrupt_blob_roots(path: &Path) {
    const BLOB_SIZE: u64 = 0x80000;
    const ROOT_SLOT_OFFSET: u64 = 0x56;

    let slots = fs::metadata(path).unwrap().len() / BLOB_SIZE;
    assert!(slots > 0, "checkpoint must write at least one blob frame");
    for slot in 0..slots {
        write_bytes_at(path, slot * BLOB_SIZE + ROOT_SLOT_OFFSET, &[0, 0]);
    }
}

fn remove_read_accelerators(path: &Path) {
    let _ = fs::remove_file(path.join("read.idx"));
    let _ = fs::remove_file(path.join("value.seg"));
}

fn first_nonempty(paths: &[PathBuf]) -> PathBuf {
    paths
        .iter()
        .find(|path| fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false))
        .cloned()
        .unwrap_or_else(|| panic!("no nonempty file among {paths:?}"))
}

#[test]
fn corrupt_manifest_or_delta_log_is_rejected_on_open() {
    let dir = tempdir().unwrap();
    let cfg = build_checkpointed_tree(dir.path());
    let victim = first_nonempty(&[
        dir.path().join("manifest.log"),
        dir.path().join("manifest.bin"),
    ]);
    flip_first_byte(&victim);

    assert!(
        Tree::open(cfg).is_err(),
        "corrupt manifest state must not be silently accepted",
    );
}

#[test]
fn corrupt_blob_image_is_rejected_on_open_or_read() {
    let dir = tempdir().unwrap();
    let cfg = build_checkpointed_tree(dir.path());
    remove_read_accelerators(dir.path());
    // Shadow rewrites can move the live GUID to any physical slot. Corrupt
    // every frame so the active manifest mapping cannot select a valid copy.
    corrupt_blob_roots(&dir.path().join("blobs.dat"));

    if let Ok(tree) = Tree::open(cfg) {
        assert!(
            tree.get(b"bucket/path/file-0001").is_err(),
            "corrupt blob image must not produce a trusted value",
        );
    }
}

#[test]
fn corrupt_read_index_falls_back_to_authoritative_blob() {
    let dir = tempdir().unwrap();
    let cfg = build_checkpointed_tree(dir.path());
    flip_first_byte(&dir.path().join("read.idx"));

    let tree = Tree::open(cfg).unwrap();
    assert_eq!(
        tree.get(b"bucket/path/file-0001").unwrap().as_deref(),
        Some(&b"value"[..]),
        "corrupt read index must fall back to the blob image",
    );
}

#[test]
fn corrupt_value_segment_falls_back_to_authoritative_blob() {
    let dir = tempdir().unwrap();
    let cfg = cfg(dir.path());
    let value = vec![0x5a; 1024];
    {
        let tree = Tree::open(cfg.clone()).unwrap();
        tree.put(b"bucket/path/large-value", &value).unwrap();
        tree.checkpoint().unwrap();
    }
    flip_first_byte(&dir.path().join("value.seg"));

    let tree = Tree::open(cfg).unwrap();
    assert_eq!(
        tree.get(b"bucket/path/large-value").unwrap().as_deref(),
        Some(value.as_slice()),
        "corrupt value segment must fall back to the blob image",
    );
}

#[cfg(unix)]
#[test]
fn removed_store_directory_surfaces_checkpoint_error() {
    let dir = tempdir().unwrap();
    let cfg = cfg(dir.path());
    let tree = Tree::open(cfg).unwrap();
    tree.put(b"before-remove", b"value").unwrap();
    fs::remove_dir_all(dir.path()).unwrap();

    assert!(
        tree.checkpoint().is_err(),
        "checkpoint must surface store-directory removal",
    );
}

#[cfg(unix)]
#[test]
fn permission_denied_store_directory_is_rejected_on_open() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempdir().unwrap();
    let path = dir.path();
    let original = fs::metadata(path).unwrap().permissions();
    fs::set_permissions(path, fs::Permissions::from_mode(0o000)).unwrap();
    let result = Tree::open(cfg(path));
    fs::set_permissions(path, original).unwrap();

    assert!(
        result.is_err(),
        "permission-denied store directory must fail open",
    );
}
