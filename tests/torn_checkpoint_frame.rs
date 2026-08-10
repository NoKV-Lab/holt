//! Power-loss safety of checkpoint frame rewrites.
//!
//! Crash model: the process dies while `pwrite`ing 512 KiB blob
//! frames into `blobs.dat`. A frame write is not power-loss atomic —
//! any subset of the in-flight bytes may persist — while everything
//! the last completed `BlobStore::flush` made durable (manifest
//! snapshot + delta log, WAL) survives. `flush` is a write barrier:
//! the engine only issues post-flush pwrites after it returns, so a
//! crash exposes at most the writes of one barrier-bracketed window.
//!
//! Two complementary invariants keep an acked write recoverable:
//!
//! 1. **Slot discipline** — between two flush barriers the store
//!    must never modify a slot the window-start durable manifest
//!    still references. The WAL can only redo onto a structurally
//!    complete base image; an in-place same-GUID rewrite (holt <=
//!    0.8.2) tears the only copy of that image.
//! 2. **Recovery isolation** — reopen must resolve reads purely
//!    through the durable manifest. Slots it does not reference may
//!    contain arbitrary bytes after a crash (torn shadow writes),
//!    and their content must never influence recovery.
//!
//! Both tests reconstruct crash states from live directory
//! snapshots through the public API, so the whole stack — frame
//! allocation, manifest ordering, WAL replay — is exercised exactly
//! as a real power cut would leave it on disk.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use holt::{
    AlignedBlobBuf, BlobStore, CheckpointConfig, Durability, FileBlobStore, Tree, TreeConfig,
};
use tempfile::tempdir;

/// One packed blob frame in `blobs.dat`.
const SLOT_BYTES: usize = 0x80000;

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

fn key(i: u32) -> Vec<u8> {
    format!("bucket/path/file-{i:06}").into_bytes()
}

/// Copy every file of a live store dir except the advisory lock.
/// Reading a live dir is sound here: the tests only rely on
/// kernel-visible file contents, which is what a crash persists.
fn snapshot_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name().to_string_lossy() == "store.lock" {
            continue;
        }
        let target = dst.join(entry.file_name());
        if entry.path().is_dir() {
            snapshot_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// GUID -> slot map of the durable manifest in a snapshot, decoded
/// from `manifest.bin` + `manifest.log` (see the format notes in
/// `src/store/blob_store/file/mod.rs`). Parsing the bytes directly
/// keeps the probe independent of the store implementation under
/// test — the point is to check what a *recovering* process would
/// consider referenced.
fn durable_manifest(dir: &Path) -> HashMap<[u8; 16], u64> {
    let mut entries = HashMap::new();
    if let Ok(buf) = fs::read(dir.join("manifest.bin")) {
        assert!(buf.len() >= 24, "manifest.bin header");
        assert_eq!(&buf[..8], b"ARTSNMNF", "manifest.bin magic");
        let count = u32::from_le_bytes(buf[10..14].try_into().unwrap()) as usize;
        let mut off = 24;
        for _ in 0..count {
            let mut guid = [0u8; 16];
            guid.copy_from_slice(&buf[off..off + 16]);
            let slot = u64::from_le_bytes(buf[off + 16..off + 24].try_into().unwrap());
            entries.insert(guid, slot);
            off += 24;
        }
    }
    if let Ok(buf) = fs::read(dir.join("manifest.log")) {
        let mut off = 0usize;
        while off + 9 <= buf.len() {
            let start = off;
            assert_eq!(&buf[start..start + 4], b"MLG1", "manifest.log magic");
            let body_len =
                u32::from_le_bytes(buf[start + 4..start + 8].try_into().unwrap()) as usize;
            let record_len = 9 + body_len + 4;
            if buf.len() - start < record_len {
                break; // torn tail is legal, replay stops here
            }
            let crc = u32::from_le_bytes(
                buf[start + 9 + body_len..start + record_len]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                crc,
                crc32fast::hash(&buf[start..start + 9 + body_len]),
                "manifest.log crc",
            );
            let body = &buf[start + 9..start + 9 + body_len];
            match buf[start + 8] {
                1 => {
                    let mut guid = [0u8; 16];
                    guid.copy_from_slice(&body[..16]);
                    let slot = u64::from_le_bytes(body[16..24].try_into().unwrap());
                    entries.insert(guid, slot);
                }
                2 => {
                    let mut guid = [0u8; 16];
                    guid.copy_from_slice(body);
                    entries.remove(&guid);
                }
                other => panic!("manifest.log unknown op {other}"),
            }
            off = start + record_len;
        }
    }
    entries
}

/// 512 KiB slot indexes whose bytes differ between two `blobs.dat`
/// images, over their overlapping length.
fn changed_slots(a: &[u8], b: &[u8]) -> Vec<u64> {
    let overlap = a.len().min(b.len()) / SLOT_BYTES;
    (0..overlap)
        .filter(|s| {
            a[s * SLOT_BYTES..(s + 1) * SLOT_BYTES] != b[s * SLOT_BYTES..(s + 1) * SLOT_BYTES]
        })
        .map(|s| s as u64)
        .collect()
}

/// Forwarding store that snapshots the live directory after every
/// completed `flush`, capturing each durable write barrier of the
/// crash model.
struct FenceSnapshotStore {
    inner: FileBlobStore,
    live_dir: PathBuf,
    snap_root: PathBuf,
    fences: AtomicUsize,
}

impl FenceSnapshotStore {
    fn new(inner: FileBlobStore, live_dir: PathBuf, snap_root: PathBuf) -> Self {
        Self {
            inner,
            live_dir,
            snap_root,
            fences: AtomicUsize::new(0),
        }
    }
    fn fence_count(&self) -> usize {
        self.fences.load(Ordering::SeqCst)
    }
    fn fence_path(&self, n: usize) -> PathBuf {
        self.snap_root.join(format!("fence-{n:04}"))
    }
}

impl BlobStore for FenceSnapshotStore {
    fn read_blob(&self, guid: holt::BlobGuid, dst: &mut AlignedBlobBuf) -> holt::Result<()> {
        self.inner.read_blob(guid, dst)
    }
    fn read_blobs(
        &self,
        guids: &[holt::BlobGuid],
        dsts: &mut [AlignedBlobBuf],
    ) -> Vec<holt::Result<()>> {
        self.inner.read_blobs(guids, dsts)
    }
    fn read_blob_range(
        &self,
        guid: holt::BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> holt::Result<()> {
        self.inner.read_blob_range(guid, byte_offset, dst)
    }
    fn read_index_range(
        &self,
        guid: holt::BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> holt::Result<bool> {
        self.inner.read_index_range(guid, byte_offset, dst)
    }
    fn read_value_segment_range(
        &self,
        guid: holt::BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> holt::Result<bool> {
        self.inner.read_value_segment_range(guid, byte_offset, dst)
    }
    fn publish_read_index(
        &self,
        guid: holt::BlobGuid,
        bytes: &[u8],
        value_bytes: &[u8],
    ) -> holt::Result<()> {
        self.inner.publish_read_index(guid, bytes, value_bytes)
    }
    fn delete_read_index(&self, guid: holt::BlobGuid) -> holt::Result<()> {
        self.inner.delete_read_index(guid)
    }
    fn write_blob(&self, guid: holt::BlobGuid, src: &AlignedBlobBuf) -> holt::Result<()> {
        self.inner.write_blob(guid, src)
    }
    fn write_blobs(&self, writes: &[(holt::BlobGuid, &AlignedBlobBuf)]) -> holt::Result<()> {
        self.inner.write_blobs(writes)
    }
    fn write_blobs_with_data_sync(
        &self,
        writes: &[(holt::BlobGuid, &AlignedBlobBuf)],
    ) -> holt::Result<()> {
        self.inner.write_blobs_with_data_sync(writes)
    }
    fn delete_blob(&self, guid: holt::BlobGuid) -> holt::Result<()> {
        self.inner.delete_blob(guid)
    }
    fn list_blobs(&self) -> holt::Result<Vec<holt::BlobGuid>> {
        self.inner.list_blobs()
    }
    fn flush(&self) -> holt::Result<()> {
        self.inner.flush()?;
        let n = self.fences.fetch_add(1, Ordering::SeqCst);
        snapshot_dir(&self.live_dir, &self.fence_path(n));
        Ok(())
    }
    fn needs_flush(&self) -> bool {
        self.inner.needs_flush()
    }
    fn has_blob(&self, guid: holt::BlobGuid) -> holt::Result<bool> {
        self.inner.has_blob(guid)
    }
    fn store_stats(&self) -> holt::StoreStats {
        self.inner.store_stats()
    }
    fn vacuum(&self) -> holt::Result<holt::VacuumStats> {
        self.inner.vacuum()
    }
}

/// Rewrite-heavy workload: three checkpoint rounds over overlapping
/// key ranges, so later rounds rewrite frames that earlier rounds
/// made durable, plus buffer-pool eviction write-through from a
/// small pool. `mutate` receives the tree and a put closure.
fn run_rewrite_workload(tree: &Tree, mut put: impl FnMut(&Tree, u32, u8)) {
    for i in 0..192 {
        put(tree, i, b'a');
    }
    tree.checkpoint().unwrap();
    for i in 0..96 {
        put(tree, i, b'b');
    }
    tree.checkpoint().unwrap();
    for i in 48..144 {
        put(tree, i, b'c');
    }
    for i in 192..224 {
        put(tree, i, b'c');
    }
    tree.checkpoint().unwrap();
}

/// Invariant 1: between two flush barriers no write may land in a
/// slot the window-start durable manifest references. This is the
/// exact property whose violation lets a power cut tear the only
/// complete copy of a checkpoint frame (in-place rewrites,
/// holt <= 0.8.2). Checked for every barrier window of the whole
/// workload, including eviction write-through and every internal
/// checkpoint flush.
#[test]
fn no_write_window_touches_durable_slots() {
    let live = tempdir().unwrap();
    let snaps = tempdir().unwrap();

    let store = Arc::new(FenceSnapshotStore::new(
        FileBlobStore::open(live.path()).unwrap(),
        live.path().to_path_buf(),
        snaps.path().to_path_buf(),
    ));
    {
        let tree =
            Tree::open_with_blob_store(cfg(live.path()), Arc::clone(&store) as Arc<dyn BlobStore>)
                .unwrap();
        run_rewrite_workload(&tree, |tree, i, fill| {
            tree.put(&key(i), &vec![fill; 4096]).unwrap();
        });
    }
    // Final live state closes the last window (writes after the last
    // flush barrier, e.g. drop-path work).
    let final_state = snaps.path().join("final");
    snapshot_dir(live.path(), &final_state);

    let fences = store.fence_count();
    assert!(fences >= 3, "workload must cross several flush barriers");
    let mut windows: Vec<PathBuf> = (0..fences).map(|n| store.fence_path(n)).collect();
    windows.push(final_state);

    let mut rewriting_windows = 0;
    for (k, pair) in windows.windows(2).enumerate() {
        let start_blobs = fs::read(pair[0].join("blobs.dat")).unwrap();
        let end_blobs = fs::read(pair[1].join("blobs.dat")).unwrap();
        let changed = changed_slots(&start_blobs, &end_blobs);
        if !changed.is_empty() {
            rewriting_windows += 1;
        }
        let referenced: HashMap<u64, [u8; 16]> = durable_manifest(&pair[0])
            .into_iter()
            .map(|(guid, slot)| (slot, guid))
            .collect();
        let violations: Vec<_> = changed
            .iter()
            .filter_map(|slot| referenced.get(slot).map(|guid| (*slot, *guid)))
            .collect();
        assert!(
            violations.is_empty(),
            "window {k}: slots {violations:02x?} were modified while still referenced \
             by the window-start durable manifest; a power cut inside this window \
             tears the only complete copy of those frames",
        );
    }
    assert!(
        rewriting_windows > 0,
        "workload must produce at least one window that writes into existing slots",
    );
}

/// Invariant 2: recovery resolves reads purely through the durable
/// manifest — slots it does not reference may hold arbitrary bytes
/// after a crash (torn shadow writes) without affecting a single
/// acked write. The tear overlays real next-checkpoint bytes, the
/// worst case short of random garbage: a structurally plausible
/// frame image that must still be ignored. Exercised for both a
/// half-frame tear and 4 KiB interleaving.
#[test]
fn recovery_ignores_torn_unreferenced_slots() {
    let live = tempdir().unwrap();
    let snaps = tempdir().unwrap();
    let pre = snaps.path().join("pre");
    let post = snaps.path().join("post");

    let mut expected: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    {
        // Journal-attached tree: acked writes are fsync'd in the WAL
        // and must survive the crash below.
        let tree = Tree::open(cfg(live.path())).unwrap();
        let mut put = |tree: &Tree, i: u32, fill: u8| {
            let value = vec![fill; 4096];
            tree.put(&key(i), &value).unwrap();
            expected.insert(key(i), value);
        };
        for i in 0..192 {
            put(&tree, i, b'a');
        }
        tree.checkpoint().unwrap();
        for i in 0..96 {
            put(&tree, i, b'b');
        }
        tree.checkpoint().unwrap();
        for i in 48..144 {
            put(&tree, i, b'c');
        }
        for i in 192..224 {
            put(&tree, i, b'c');
        }
        // Durable pre-crash state: acked WAL + last flushed manifest.
        snapshot_dir(live.path(), &pre);
        // The checkpoint the crash interrupts; its writes are the
        // in-flight bytes the tear below partially persists.
        tree.checkpoint().unwrap();
        snapshot_dir(live.path(), &post);
    }

    let pre_blobs = fs::read(pre.join("blobs.dat")).unwrap();
    let post_blobs = fs::read(post.join("blobs.dat")).unwrap();
    let referenced: HashMap<u64, [u8; 16]> = durable_manifest(&pre)
        .into_iter()
        .map(|(guid, slot)| (slot, guid))
        .collect();
    // Only slots the pre manifest does not reference may be torn: a
    // correct store confines the interrupted round's writes to such
    // slots (invariant 1), and their content is unconstrained after
    // a crash. Referenced slots that changed later did so behind
    // newer flush barriers the modelled crash never reached.
    let torn_slots: Vec<u64> = changed_slots(&pre_blobs, &post_blobs)
        .into_iter()
        .filter(|slot| !referenced.contains_key(slot))
        .collect();
    assert!(
        !torn_slots.is_empty(),
        "the interrupted round must have written at least one unreferenced slot",
    );

    for (label, torn_bytes) in [
        ("new-prefix", {
            // First half of each in-flight frame persisted.
            let mut bytes = pre_blobs.clone();
            for &slot in &torn_slots {
                let start = slot as usize * SLOT_BYTES;
                let mid = start + SLOT_BYTES / 2;
                bytes[start..mid].copy_from_slice(&post_blobs[start..mid]);
            }
            bytes
        }),
        ("interleave-4k", {
            // Alternating 4 KiB blocks persisted: device reordering.
            const BLOCK: usize = 4096;
            let mut bytes = pre_blobs.clone();
            for &slot in &torn_slots {
                let start = slot as usize * SLOT_BYTES;
                for (i, off) in (start..start + SLOT_BYTES).step_by(BLOCK).enumerate() {
                    if i % 2 == 0 {
                        bytes[off..off + BLOCK].copy_from_slice(&post_blobs[off..off + BLOCK]);
                    }
                }
            }
            bytes
        }),
    ] {
        let torn = snaps.path().join(format!("torn-{label}"));
        snapshot_dir(&pre, &torn);
        fs::write(torn.join("blobs.dat"), &torn_bytes).unwrap();

        let tree = Tree::open(cfg(&torn)).unwrap_or_else(|err| {
            panic!("reopen with torn unreferenced slots {torn_slots:?} ({label}) failed: {err:?}")
        });
        for (k, v) in &expected {
            let got = tree
                .get(k)
                .unwrap_or_else(|err| {
                    panic!(
                        "acked key {} unreadable ({label}, torn slots {torn_slots:?}): {err:?}",
                        String::from_utf8_lossy(k),
                    )
                })
                .unwrap_or_else(|| {
                    panic!(
                        "acked key {} lost ({label}, torn slots {torn_slots:?})",
                        String::from_utf8_lossy(k),
                    )
                });
            assert_eq!(
                &got,
                v,
                "acked key {} corrupted ({label}, torn slots {torn_slots:?})",
                String::from_utf8_lossy(k),
            );
        }
    }
}
