//! `FileBlobStore` — file-backed durable blob store.
//!
//! Available on every Unix platform. The Linux build opens the
//! packed data file with `O_DIRECT` so the kernel does not cache
//! blob frames (the buffer manager *is* the cache). The rebuildable
//! read index uses buffered positional I/O because it serves small
//! index records rather than authoritative blob frames.
//!
//! Layout on disk:
//!
//! ```text
//!   <data_dir>/
//!     blobs.dat      — single packed file, blob N lives at byte
//!                      offset N * PAGE_SIZE
//!     manifest.bin   — small file mapping BlobGuid → slot number
//!                      plus `next_slot`;
//!                      rewritten only when the manifest delta log
//!                      is compacted
//!     manifest.log   — append-only set/delete deltas replayed on
//!                      open; free slots are rebuilt from holes
//!     read.idx       — optional packed read indexes, one slot
//!                      per blob slot; rebuildable and never part of
//!                      recovery truth
//!     value.seg      — optional packed read value payloads, one
//!                      slot per blob slot; referenced only by
//!                      `read.idx` entries and rebuildable
//!     store.lock     — zero-byte advisory lock file; held
//!                      exclusively (flock) for the lifetime of an
//!                      open instance so a second opener cannot
//!                      corrupt the manifest
//! ```
//!
//! Design rationale:
//!
//! - **Single packed file** instead of one-file-per-blob: a buffer
//!   manager pinning thousands of blobs would otherwise need
//!   thousands of file descriptors. One fd + slot offsets keeps the
//!   kernel page tables and fs metadata trivial.
//! - **O_DIRECT / F_NOCACHE** bypasses the page cache for
//!   `blobs.dat`: ours *is* the cache. The buffer manager owns dirty
//!   pages and flushes through the store. The packed data file is
//!   preallocated in coarse chunks (`posix_fallocate` on Linux,
//!   `F_PREALLOCATE` on macOS) so checkpoint bursts do not
//!   repeatedly pay file-growth allocation latency.
//! - **4 KB-aligned I/O** (every offset is a multiple of `PAGE_SIZE`
//!   = 512 KB, every buffer is [`AlignedBlobBuf`] = 4 KB aligned) so
//!   `O_DIRECT` accepts every submission without `EINVAL`.
//! - **Manifest** holds the GUID → slot mapping. Checkpoint rounds
//!   append small set/delete deltas to `manifest.log` and fsync it
//!   instead of rewriting the whole map. When the log grows well
//!   past the snapshot size it compacts into `manifest.bin` via
//!   tmp+rename and truncates the log.
//! - **Read accelerators** are fixed-slot and rebuildable. `read.idx`
//!   and `value.seg` share the same manifest slot as `blobs.dat`.
//!   Rewriting or deleting a blob clears the read-index header before
//!   the authoritative frame changes; checkpoint publication writes
//!   value bytes first and the index header last. Slot reuse is the
//!   normal reclamation mechanism, avoiding an append-only value-segment
//!   GC. Explicit `vacuum` compacts live high-water slots into lower
//!   reusable holes, carries their advisory read accelerators with the
//!   blob slot, truncates the packed-file tail, and, on Linux, punches
//!   any remaining reusable middle-slot holes so sparse files can return
//!   blocks to the filesystem.
//!
//! ## I/O store
//!
//! Two code paths share the same `FileBlobStore` struct:
//!
//! - **`pread`/`pwritev`** (default): every Unix target, every build
//!   configuration. Reads use `FileExt::read_exact_at`; checkpoint
//!   write batches coalesce slot-contiguous blobs with `pwritev`.
//! - **`io_uring`** (`cfg(target_os = "linux")` + `feature =
//!   "io-uring"`): submits one SQE per read/write to a dedicated
//!   ring owned by the store. Eliminates the per-syscall entry/
//!   exit cost on Linux.
//!
//! Both paths share the same on-disk layout and the same
//! `BlobStore::flush` semantics (`sync_data` + manifest persist).
//! Switching between them is an internal performance toggle; no
//! caller-visible behaviour changes.

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod uring;

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
use std::os::unix::fs::FileExt;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::api::errors::{Error, Result};
use crate::api::stats::{StoreStats, VacuumStats};
use crate::layout::{BlobGuid, PAGE_SIZE};

#[cfg(all(target_os = "linux", feature = "io-uring"))]
use super::BlobBufPool;
use super::{AlignedBlobBuf, BlobStore};

#[cfg(all(target_os = "linux", feature = "io-uring"))]
use self::uring::UringContext;

/// Filename of the packed blob data file inside `data_dir`.
const DATA_FILENAME: &str = "blobs.dat";
/// Advisory lock file inside `data_dir`, flock'd exclusively for
/// the lifetime of an open instance.
const LOCK_FILENAME: &str = "store.lock";
/// How long `open` waits for a previous instance to release the
/// directory lock before failing. Covers the handover pattern where
/// a caller opens a new instance while the previous one is still
/// flushing its final checkpoint round on drop.
const DIR_LOCK_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(5);
/// Poll interval while waiting for the directory lock.
const DIR_LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(10);
/// Filename of the manifest inside `data_dir`.
const MANIFEST_FILENAME: &str = "manifest.bin";
/// Append-only manifest delta log inside `data_dir`.
const MANIFEST_LOG_FILENAME: &str = "manifest.log";
/// Packed rebuildable read-index file.
const READ_INDEX_FILENAME: &str = "read.idx";
/// Packed rebuildable read-value segment file.
const VALUE_SEGMENT_FILENAME: &str = "value.seg";
/// Filename used as the rename staging target for the manifest.
const MANIFEST_TMP_FILENAME: &str = "manifest.bin.tmp";
/// Conservative iovec chunk limit used by the non-uring batch
/// writer. POSIX guarantees at least 16; mainstream Unix kernels
/// support 1024, and chunking keeps us below the common cap.
#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
const PWRITEV_IOV_MAX: usize = 1024;
/// Packed-file reservation units. Small trees grow in 4 MiB
/// chunks; large trees switch to 32 MiB chunks so checkpoint bursts
/// don't pay file-growth allocation every few blobs.
const DATA_PREALLOC_SMALL_CHUNK_SLOTS: u64 = 8;
const DATA_PREALLOC_LARGE_CHUNK_SLOTS: u64 = 64;
const DATA_PREALLOC_LARGE_AT_SLOTS: u64 = 128;
/// Upper bound for `io_uring` fixed-buffer registration.
///
/// Each slot is one 512 KiB blob frame. Registering the whole cache
/// would pin `buffer_pool_size * 512 KiB` at open/reopen time, which
/// quickly dominates startup latency. Keep a bounded hot I/O pool
/// instead: resident cache entries and checkpoint snapshots try to
/// lease these fixed frames first, and fall back to normal aligned
/// heap buffers when the hot pool is exhausted.
const REGISTERED_BUFFER_MAX_SLOTS: usize = 32;

/// Manifest file magic — recognised on load to refuse bogus files.
const MANIFEST_MAGIC: [u8; 8] = *b"ARTSNMNF";
/// Manifest format version. Bumped on any breaking change.
///
/// Older-version files are refused on load — the on-disk format is
/// not migrated. v3 introduced the flattened single-encoding leaf
/// (one contiguous `[16B header][key][value]` node). v4 switches node
/// addressing from 1-based slot indices to body byte offsets: child
/// fields (`children[N]`, `Prefix.child`, `header.root`) now store a
/// biased `byte_offset/8` instead of a slot, and the Leaf header was
/// reordered to carry a self-describing `node_type @ +1` byte. Both
/// change the on-blob byte layout, so v3 files are refused on load.
/// v5 added a per-blob image `generation` to each manifest entry for
/// an older read-accelerator design. v6 drops it: the current packed
/// read index validates against the blob header stamp, so the
/// manifest generation field was dead weight. Older manifests (incl.
/// v5) are refused on load, not migrated.
const MANIFEST_VERSION: u16 = 6;
/// Per-record magic for `manifest.log`.
const MANIFEST_LOG_MAGIC: [u8; 4] = *b"MLG1";
const MANIFEST_LOG_TY_SET: u8 = 1;
const MANIFEST_LOG_TY_DELETE: u8 = 2;
const MANIFEST_LOG_HEADER_SIZE: usize = 4 + 4 + 1;
const MANIFEST_LOG_FOOTER_SIZE: usize = 4;
const MANIFEST_LOG_SET_BODY_SIZE: usize = 16 + 8;
const MANIFEST_LOG_DELETE_BODY_SIZE: usize = 16;
const MANIFEST_LOG_MIN_COMPACT_BYTES: u64 = 1024 * 1024;
const MANIFEST_LOG_COMPACT_RATIO: u64 = 4;
const READ_INDEX_IO_ALIGN: usize = 512;
const READ_INDEX_SLOT_BYTES: usize = PAGE_SIZE as usize;
const VALUE_SEGMENT_IO_ALIGN: usize = 512;
const VALUE_SEGMENT_SLOT_BYTES: usize = PAGE_SIZE as usize;

/// NVMe-backed, O_DIRECT, single-packed-file blob store.
///
/// Construct via [`FileBlobStore::open`]. Thread-safe; the
/// underlying file handle is shared and `pread`/`pwrite` are
/// atomic at the syscall boundary.
#[derive(Debug)]
pub struct FileBlobStore {
    data_dir: PathBuf,
    /// Exclusive advisory lock on `data_dir`, held for the lifetime
    /// of this instance. Two live instances on one directory would
    /// each replay `manifest.log` into the same `next_slot`, assign
    /// the same slot to different blob GUIDs, and append conflicting
    /// set deltas — permanently corrupting the manifest. The kernel
    /// releases the lock when this handle closes, so a crashed
    /// holder never leaves a stale lock behind.
    _dir_lock: File,
    data_file: File,
    read_index_file: File,
    value_segment_file: File,
    manifest: RwLock<Manifest>,
    /// Tracks whether `manifest.bin` needs a rewrite. Data-only
    /// overwrites of existing blobs leave this false, avoiding
    /// manifest I/O on pure data overwrites.
    manifest_dirty: AtomicBool,
    /// Monotonic counter bumped before each data-file write.
    /// `flush` syncs up to the observed epoch instead of clearing
    /// a single bool, so a racing writer cannot be hidden by a
    /// concurrent successful sync.
    data_write_epoch: AtomicU64,
    /// Highest data write epoch known to have survived
    /// `fdatasync` / `File::sync_data`.
    data_sync_epoch: AtomicU64,
    /// Serializes the durability boundary between slot assignment,
    /// data writes, data sync, and manifest persistence. This is
    /// not on the read path; checkpoint I/O already funnels through
    /// one worker, and Linux `io_uring` also has one SQ owner.
    data_io_lock: Mutex<()>,
    /// Protects physical slot remapping. Readers hold the shared
    /// side from GUID→slot resolution through the positional read.
    /// Vacuum/defrag holds the exclusive side while copying live
    /// slots and publishing the manifest remap.
    slot_io_lock: RwLock<()>,
    /// Highest slot count the packed data file has been
    /// best-effort preallocated to.
    preallocated_slots: AtomicU64,
    /// `io_uring` context — present iff Linux + `feature =
    /// "io-uring"`. Held behind a `Mutex` so concurrent callers
    /// serialise on the submission queue; with the single I/O
    /// worker thread this lock is uncontended on the hot path.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    uring: Mutex<UringContext>,
    /// Fixed-buffer pool registered with `uring`. Buffers allocated
    /// from this pool carry a stable `buf_index` so the Linux path
    /// can submit `READ_FIXED` / `WRITE_FIXED` without per-op
    /// registration.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    registered_buffers: Option<BlobBufPool>,
}

#[derive(Debug)]
struct Manifest {
    /// guid → packed-file slot.
    entries: HashMap<BlobGuid, ManifestEntry>,
    /// Next never-used slot to hand out when no reusable slot is
    /// available.
    next_slot: u64,
    /// Slots whose deletion is durable in the manifest and can be
    /// safely reused by future writes. Reopen stores contiguous
    /// holes as ranges so a sparse high-water manifest does not
    /// expand into one `u64` per free slot.
    reusable_slots: ReusableSlots,
    /// Slots removed from `slots` by `delete_blob` but not yet
    /// durable in `manifest.bin`. They become reusable only after
    /// `flush` successfully persists the manifest rewrite; reusing
    /// them earlier could corrupt crash recovery by overwriting a
    /// slot still referenced by the old on-disk manifest.
    pending_free_slots: Vec<u64>,
    /// Path to the manifest file (for tmp+rename writes).
    path: PathBuf,
    /// Path to the append-only manifest delta log.
    log_path: PathBuf,
    /// Bytes currently in `manifest.log`, used to decide when a
    /// full snapshot compaction is worth paying for.
    log_bytes: u64,
    /// Ordered set/delete records not yet durable in
    /// `manifest.log`. The in-memory `slots` map already reflects
    /// them; this queue is the recovery contract.
    pending_log: Vec<ManifestDelta>,
}

#[derive(Debug, Clone, Copy)]
enum ManifestDelta {
    Set { guid: BlobGuid, slot: u64 },
    Delete { guid: BlobGuid },
}

#[derive(Debug, Clone, Copy)]
struct ManifestEntry {
    slot: u64,
}

#[derive(Debug, Default)]
struct ReusableSlots {
    singles: Vec<u64>,
    ranges: Vec<FreeSlotRange>,
}

#[derive(Debug, Clone, Copy)]
struct FreeSlotRange {
    next: u64,
    end: u64,
}

impl FreeSlotRange {
    fn slot_count(self) -> u64 {
        self.end.saturating_sub(self.next).saturating_add(1)
    }
}

/// Acquire the exclusive advisory lock on `data_dir`, waiting up to
/// `timeout` for a previous instance to release it.
///
/// `flock(2)` locks are per open-file-description, so this also
/// rejects a second instance inside the same process — the scenario
/// `fcntl` record locks would silently allow. The polling wait lets
/// an open racing a previous instance's drop (the common handover
/// pattern `store = reopen(path)`) serialize instead of failing.
fn acquire_dir_lock(data_dir: &Path, timeout: Duration) -> Result<File> {
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .custom_flags(libc::O_CLOEXEC)
        .open(data_dir.join(LOCK_FILENAME))?;
    let deadline = Instant::now() + timeout;
    loop {
        let rc = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return Ok(lock_file);
        }
        let err = io::Error::last_os_error();
        match err.kind() {
            io::ErrorKind::Interrupted => {}
            io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(Error::BlobStoreIo(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        format!(
                            "blob store at {} is locked by another live instance \
                             (waited {timeout:?}); a second opener would corrupt \
                             the manifest",
                            data_dir.display()
                        ),
                    )));
                }
                thread::sleep(DIR_LOCK_RETRY_INTERVAL);
            }
            _ => return Err(Error::BlobStoreIo(err)),
        }
    }
}

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

impl FileBlobStore {
    /// Open or create a persistent store at `data_dir`.
    ///
    /// Creates the directory if missing. On Linux opens the packed
    /// data file with `O_DIRECT | O_CLOEXEC`; on other Unixes opens
    /// with `O_CLOEXEC` only (macOS additionally sets `F_NOCACHE`).
    /// Loads the manifest if present; otherwise starts empty.
    pub fn open<P: Into<PathBuf>>(data_dir: P) -> Result<Self> {
        Self::open_with_registered_buffer_capacity(data_dir, REGISTERED_BUFFER_MAX_SLOTS)
    }

    /// Open with a registered-buffer hot-pool hint derived from the
    /// caller's buffer-manager capacity. The actual pool is bounded
    /// by [`REGISTERED_BUFFER_MAX_SLOTS`] so large caches do not pin
    /// proportional memory at open/reopen time.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub(crate) fn open_with_buffer_pool_hint<P: Into<PathBuf>>(
        data_dir: P,
        buffer_pool_size: usize,
    ) -> Result<Self> {
        let slots = registered_buffer_slots(buffer_pool_size);
        Self::open_with_registered_buffer_capacity(data_dir, slots)
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn open_with_registered_buffer_capacity<P: Into<PathBuf>>(
        data_dir: P,
        registered_buffer_slots: usize,
    ) -> Result<Self> {
        let data_dir = data_dir.into();
        std::fs::create_dir_all(&data_dir)?;
        // Take the lock before touching any store file: manifest
        // replay (including torn-tail truncation) must not run
        // while another instance can still append deltas.
        let dir_lock = acquire_dir_lock(&data_dir, DIR_LOCK_ACQUIRE_TIMEOUT)?;

        let data_path = data_dir.join(DATA_FILENAME);
        let read_index_path = data_dir.join(READ_INDEX_FILENAME);
        let value_segment_path = data_dir.join(VALUE_SEGMENT_FILENAME);
        let manifest_path = data_dir.join(MANIFEST_FILENAME);
        let manifest_log_path = data_dir.join(MANIFEST_LOG_FILENAME);

        let data_flags = {
            #[cfg(target_os = "linux")]
            {
                libc::O_DIRECT | libc::O_CLOEXEC
            }
            #[cfg(not(target_os = "linux"))]
            {
                libc::O_CLOEXEC
            }
        };
        let index_flags = libc::O_CLOEXEC;
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(data_flags)
            .open(&data_path)?;
        let read_index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(index_flags)
            .open(&read_index_path)?;
        let value_segment_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(index_flags)
            .open(&value_segment_path)?;

        // macOS doesn't have O_DIRECT; F_NOCACHE on the fd is the
        // closest equivalent (tells the VFS not to populate the
        // unified buffer cache for this fd's I/O).
        #[cfg(target_os = "macos")]
        unsafe {
            let _ = libc::fcntl(data_file.as_raw_fd(), libc::F_NOCACHE, 1);
        }

        let manifest = Manifest::load_or_create(&manifest_path, &manifest_log_path)?;
        let file_slots = slots_for_len(data_file.metadata()?.len());
        let preallocated_slots = file_slots.max(manifest.next_slot);

        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let (uring, registered_buffers) = {
            let pool = BlobBufPool::new(registered_buffer_slots);
            match pool {
                Some(pool) => match UringContext::new(&data_file, Some(&pool)) {
                    Ok(ctx) => (Mutex::new(ctx), Some(pool)),
                    Err(_) => (Mutex::new(UringContext::new(&data_file, None)?), None),
                },
                None => (Mutex::new(UringContext::new(&data_file, None)?), None),
            }
        };

        Ok(Self {
            data_dir,
            _dir_lock: dir_lock,
            data_file,
            read_index_file,
            value_segment_file,
            manifest: RwLock::new(manifest),
            manifest_dirty: AtomicBool::new(false),
            data_write_epoch: AtomicU64::new(0),
            data_sync_epoch: AtomicU64::new(0),
            data_io_lock: Mutex::new(()),
            slot_io_lock: RwLock::new(()),
            preallocated_slots: AtomicU64::new(preallocated_slots),
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            uring,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            registered_buffers,
        })
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn open_with_registered_buffer_capacity<P: Into<PathBuf>>(
        data_dir: P,
        _registered_buffer_slots: usize,
    ) -> Result<Self> {
        let data_dir = data_dir.into();
        std::fs::create_dir_all(&data_dir)?;
        // Take the lock before touching any store file: manifest
        // replay (including torn-tail truncation) must not run
        // while another instance can still append deltas.
        let dir_lock = acquire_dir_lock(&data_dir, DIR_LOCK_ACQUIRE_TIMEOUT)?;

        let data_path = data_dir.join(DATA_FILENAME);
        let read_index_path = data_dir.join(READ_INDEX_FILENAME);
        let value_segment_path = data_dir.join(VALUE_SEGMENT_FILENAME);
        let manifest_path = data_dir.join(MANIFEST_FILENAME);
        let manifest_log_path = data_dir.join(MANIFEST_LOG_FILENAME);

        let data_flags = {
            #[cfg(target_os = "linux")]
            {
                libc::O_DIRECT | libc::O_CLOEXEC
            }
            #[cfg(not(target_os = "linux"))]
            {
                libc::O_CLOEXEC
            }
        };
        let index_flags = libc::O_CLOEXEC;
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(data_flags)
            .open(&data_path)?;
        let read_index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(index_flags)
            .open(&read_index_path)?;
        let value_segment_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(index_flags)
            .open(&value_segment_path)?;

        #[cfg(target_os = "macos")]
        unsafe {
            let _ = libc::fcntl(data_file.as_raw_fd(), libc::F_NOCACHE, 1);
        }

        let manifest = Manifest::load_or_create(&manifest_path, &manifest_log_path)?;
        let file_slots = slots_for_len(data_file.metadata()?.len());
        let preallocated_slots = file_slots.max(manifest.next_slot);

        Ok(Self {
            data_dir,
            _dir_lock: dir_lock,
            data_file,
            read_index_file,
            value_segment_file,
            manifest: RwLock::new(manifest),
            manifest_dirty: AtomicBool::new(false),
            data_write_epoch: AtomicU64::new(0),
            data_sync_epoch: AtomicU64::new(0),
            data_io_lock: Mutex::new(()),
            slot_io_lock: RwLock::new(()),
            preallocated_slots: AtomicU64::new(preallocated_slots),
        })
    }

    /// Directory holding `blobs.dat` and `manifest.bin`.
    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Number of blobs in the manifest.
    #[must_use]
    pub fn len(&self) -> usize {
        self.manifest.read().unwrap().entries.len()
    }

    /// True if the manifest is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifest.read().unwrap().entries.is_empty()
    }

    fn offset_of(&self, guid: BlobGuid) -> Result<u64> {
        Ok(self.entry_of(guid)?.slot * u64::from(PAGE_SIZE))
    }

    fn read_index_offset_of(&self, guid: BlobGuid) -> Option<u64> {
        let m = self.manifest.read().unwrap();
        m.entries
            .get(&guid)
            .map(|entry| entry.slot * u64::from(PAGE_SIZE))
    }

    fn value_segment_offset_of(&self, guid: BlobGuid) -> Option<u64> {
        let m = self.manifest.read().unwrap();
        m.entries
            .get(&guid)
            .map(|entry| entry.slot * u64::from(PAGE_SIZE))
    }

    fn enter_slot_read(&self) -> RwLockReadGuard<'_, ()> {
        self.slot_io_lock.read().unwrap()
    }

    fn enter_slot_write(&self) -> RwLockWriteGuard<'_, ()> {
        self.slot_io_lock.write().unwrap()
    }

    fn remove_read_accelerators_best_effort(&self, guid: BlobGuid) {
        let _ = self.clear_read_accelerator_slots(guid);
    }

    fn clear_read_accelerator_slots(&self, guid: BlobGuid) -> Result<()> {
        self.clear_read_index_slot(guid)?;
        self.clear_value_segment_slot(guid)
    }

    fn clear_read_index_slot(&self, guid: BlobGuid) -> Result<()> {
        let Some(offset) = self.read_index_offset_of(guid) else {
            return Ok(());
        };
        if offset >= file_len(&self.read_index_file) {
            return Ok(());
        }
        let zeros = [0u8; READ_INDEX_IO_ALIGN];
        self.write_read_index_aligned(offset, &zeros)
    }

    fn clear_value_segment_slot(&self, guid: BlobGuid) -> Result<()> {
        let Some(offset) = self.value_segment_offset_of(guid) else {
            return Ok(());
        };
        if offset >= file_len(&self.value_segment_file) {
            return Ok(());
        }
        let zeros = [0u8; VALUE_SEGMENT_IO_ALIGN];
        self.write_value_segment_aligned(offset, &zeros)
    }

    fn entry_of(&self, guid: BlobGuid) -> Result<ManifestEntry> {
        let m = self.manifest.read().unwrap();
        m.entries.get(&guid).copied().ok_or_else(|| {
            Error::BlobStoreIo(io::Error::new(
                io::ErrorKind::NotFound,
                format!("blob {:02x?} not in manifest", &guid[..4]),
            ))
        })
    }

    fn assign_write_entry(&self, guid: BlobGuid) -> ManifestEntry {
        let mut m = self.manifest.write().unwrap();
        let entry = m.assign_write_entry(guid);
        m.pending_log.push(ManifestDelta::Set {
            guid,
            slot: entry.slot,
        });
        self.manifest_dirty.store(true, Ordering::Release);
        entry
    }

    fn assign_write_entries(
        &self,
        guids: impl IntoIterator<Item = BlobGuid>,
    ) -> Vec<ManifestEntry> {
        let mut m = self.manifest.write().unwrap();
        let mut out = Vec::new();
        let mut dirty = false;
        for guid in guids {
            let entry = m.assign_write_entry(guid);
            m.pending_log.push(ManifestDelta::Set {
                guid,
                slot: entry.slot,
            });
            dirty = true;
            out.push(entry);
        }
        if dirty {
            self.manifest_dirty.store(true, Ordering::Release);
        }
        out
    }

    fn mark_data_write_started(&self) -> u64 {
        self.data_write_epoch.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn mark_data_synced(&self, epoch: u64) {
        self.data_sync_epoch.fetch_max(epoch, Ordering::AcqRel);
    }

    fn data_needs_sync(&self) -> Option<u64> {
        let written = self.data_write_epoch.load(Ordering::Acquire);
        let synced = self.data_sync_epoch.load(Ordering::Acquire);
        (synced < written).then_some(written)
    }

    fn prepare_blob_writes<'a>(
        &self,
        writes: &'a [(BlobGuid, &'a AlignedBlobBuf)],
    ) -> Result<Vec<PreparedBlobWrite<'a>>> {
        if writes.is_empty() {
            return Ok(Vec::new());
        }
        let entries = self.assign_write_entries(writes.iter().map(|(guid, _)| *guid));
        if let Some(required_slots) = entries
            .iter()
            .map(|entry| entry.slot.saturating_add(1))
            .max()
        {
            self.ensure_data_capacity(required_slots)?;
        }
        let mut io = Vec::with_capacity(writes.len());
        for ((_guid, src), entry) in writes.iter().zip(entries) {
            io.push(PreparedBlobWrite {
                offset: entry.slot * u64::from(PAGE_SIZE),
                src,
            });
        }
        Ok(io)
    }

    // ---------- I/O dispatch (uring vs pread/pwrite) ----------
    //
    // Two paired cfg-gated helpers per direction: the active one
    // compiles, the inactive one doesn't. Keeps `read_blob` /
    // `write_blob` clean of any conditional plumbing.

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn pread_at(&self, offset: u64, dst: &mut AlignedBlobBuf) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        ring.pread_at(offset, dst)?;
        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn pread_at(&self, offset: u64, dst: &mut AlignedBlobBuf) -> Result<()> {
        let dst = dst.as_mut_slice();
        self.data_file.read_exact_at(dst, offset)?;
        Ok(())
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn pwrite_at(&self, offset: u64, src: &AlignedBlobBuf) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        ring.pwrite_at(offset, src)?;
        Ok(())
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn pwrite_many_at(&self, writes: &[PreparedBlobWrite<'_>]) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        let io: Vec<_> = writes.iter().map(|w| (w.offset, w.src)).collect();
        ring.pwrite_many_at(&io)?;
        Ok(())
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn pwrite_many_and_sync_at(&self, writes: &[PreparedBlobWrite<'_>]) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        let io: Vec<_> = writes.iter().map(|w| (w.offset, w.src)).collect();
        ring.pwrite_many_and_sync_at(&io)?;
        Ok(())
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn sync_data_file(&self) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        ring.sync_data()?;
        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn pwrite_at(&self, offset: u64, src: &AlignedBlobBuf) -> Result<()> {
        self.data_file.write_all_at(src.as_slice(), offset)?;
        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn pwrite_many_at(&self, writes: &[PreparedBlobWrite<'_>]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }

        let mut ordered: Vec<_> = writes
            .iter()
            .enumerate()
            .map(|(order, write)| OrderedWrite {
                offset: write.offset,
                src: write.src.as_slice(),
                order,
            })
            .collect();
        ordered.sort_by(|a, b| a.offset.cmp(&b.offset).then(a.order.cmp(&b.order)));

        let mut start = 0usize;
        while start < ordered.len() {
            let mut end = start + 1;
            let mut next_offset = ordered[start].offset + ordered[start].src.len() as u64;
            while end < ordered.len() && ordered[end].offset == next_offset {
                next_offset += ordered[end].src.len() as u64;
                end += 1;
            }
            self.pwritev_contiguous(&ordered[start..end])?;
            start = end;
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn pwritev_contiguous(&self, writes: &[OrderedWrite<'_>]) -> Result<()> {
        debug_assert!(!writes.is_empty());
        for chunk in writes.chunks(PWRITEV_IOV_MAX) {
            let mut expected = 0usize;
            let mut iovecs = Vec::with_capacity(chunk.len());
            for write in chunk {
                expected += write.src.len();
                iovecs.push(libc::iovec {
                    iov_base: write.src.as_ptr() as *mut libc::c_void,
                    iov_len: write.src.len(),
                });
            }
            let offset = chunk[0].offset as libc::off_t;
            let written = loop {
                let written = unsafe {
                    libc::pwritev(
                        self.data_file.as_raw_fd(),
                        iovecs.as_ptr(),
                        iovecs.len() as libc::c_int,
                        offset,
                    )
                };
                if written >= 0 {
                    break written as usize;
                }
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(Error::BlobStoreIo(err));
            };
            if written != expected {
                return Err(Error::BlobStoreIo(io::Error::other(format!(
                    "short pwritev: wrote {written} of {expected}"
                ))));
            }
        }
        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn sync_data_file(&self) -> Result<()> {
        self.data_file.sync_data()?;
        Ok(())
    }

    fn write_read_index_aligned(&self, offset: u64, src: &[u8]) -> Result<()> {
        debug_assert_eq!(offset % READ_INDEX_IO_ALIGN as u64, 0);
        debug_assert_eq!(src.len() % READ_INDEX_IO_ALIGN, 0);
        use std::os::unix::fs::FileExt;
        self.read_index_file.write_all_at(src, offset)?;
        Ok(())
    }

    fn write_value_segment_aligned(&self, offset: u64, src: &[u8]) -> Result<()> {
        debug_assert_eq!(offset % VALUE_SEGMENT_IO_ALIGN as u64, 0);
        debug_assert_eq!(src.len() % VALUE_SEGMENT_IO_ALIGN, 0);
        use std::os::unix::fs::FileExt;
        self.value_segment_file.write_all_at(src, offset)?;
        Ok(())
    }

    fn ensure_data_capacity(&self, required_slots: u64) -> Result<()> {
        let current = self.preallocated_slots.load(Ordering::Acquire);
        if required_slots <= current {
            return Ok(());
        }
        let target = round_up_slots(required_slots);
        preallocate_data_file(&self.data_file, target.saturating_mul(u64::from(PAGE_SIZE)))?;
        self.preallocated_slots.fetch_max(target, Ordering::AcqRel);
        Ok(())
    }

    fn flush_locked(&self) -> Result<()> {
        // Order matters: data must be on disk before the manifest
        // promotes any new slot. Otherwise a crash could leave the
        // manifest pointing at a slot whose data is still in NVMe's
        // write cache.
        if let Some(epoch) = self.data_needs_sync() {
            self.sync_data_file()?;
            self.mark_data_synced(epoch);
        }

        if self.manifest_dirty.swap(false, Ordering::AcqRel) {
            let mut m = self.manifest.write().unwrap();
            if let Err(e) = m.persist_pending_deltas(&self.data_dir) {
                self.manifest_dirty.store(true, Ordering::Release);
                return Err(e);
            }
            m.pending_log.clear();
            m.publish_pending_free_slots();
        }
        Ok(())
    }

    fn shrink_packed_files(&self, slots: u64) -> Result<u64> {
        let len = slots.saturating_mul(u64::from(PAGE_SIZE));
        let mut bytes = 0;
        bytes += shrink_file_to_len(&self.data_file, len)?;
        bytes += shrink_file_to_len(&self.read_index_file, len)?;
        bytes += shrink_file_to_len(&self.value_segment_file, len)?;
        self.preallocated_slots.store(slots, Ordering::Release);
        Ok(bytes)
    }

    fn copy_relocated_slots(&self, plan: &[SlotMove]) -> Result<u64> {
        if plan.is_empty() {
            return Ok(0);
        }

        let mut data = self.alloc_blob_buf_zeroed();
        let mut aux = vec![0u8; PAGE_SIZE as usize];
        let mut bytes = 0u64;
        for item in plan {
            bytes = bytes.saturating_add(self.copy_data_slot(
                item.from_slot,
                item.to_slot,
                &mut data,
            )?);
            bytes = bytes.saturating_add(copy_advisory_slot(
                &self.read_index_file,
                item.from_slot,
                item.to_slot,
                &mut aux,
            )?);
            bytes = bytes.saturating_add(copy_advisory_slot(
                &self.value_segment_file,
                item.from_slot,
                item.to_slot,
                &mut aux,
            )?);
        }

        self.data_file.sync_all()?;
        self.read_index_file.sync_all()?;
        self.value_segment_file.sync_all()?;
        Ok(bytes)
    }

    fn copy_data_slot(
        &self,
        from_slot: u64,
        to_slot: u64,
        buf: &mut AlignedBlobBuf,
    ) -> Result<u64> {
        use std::os::unix::fs::FileExt;

        let from = from_slot.saturating_mul(u64::from(PAGE_SIZE));
        let to = to_slot.saturating_mul(u64::from(PAGE_SIZE));
        self.data_file.read_exact_at(buf.as_mut_slice(), from)?;
        self.data_file.write_all_at(buf.as_slice(), to)?;
        Ok(u64::from(PAGE_SIZE))
    }

    fn punch_reusable_slot_ranges(&self, ranges: &[FreeSlotRange]) -> Result<(u64, u64)> {
        let mut slots = 0u64;
        let mut bytes = 0u64;
        for range in ranges {
            let slot_count = range.slot_count();
            if slot_count == 0 {
                continue;
            }
            let offset = range.next.saturating_mul(u64::from(PAGE_SIZE));
            let len = slot_count.saturating_mul(u64::from(PAGE_SIZE));
            let range_bytes = punch_file_range(&self.data_file, offset, len)?
                .saturating_add(punch_file_range(&self.read_index_file, offset, len)?)
                .saturating_add(punch_file_range(&self.value_segment_file, offset, len)?);
            if range_bytes != 0 {
                slots = slots.saturating_add(slot_count);
                bytes = bytes.saturating_add(range_bytes);
            }
        }
        if bytes != 0 {
            self.data_file.sync_all()?;
            self.read_index_file.sync_all()?;
            self.value_segment_file.sync_all()?;
        }
        Ok((slots, bytes))
    }
}

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
#[derive(Clone, Copy)]
struct OrderedWrite<'a> {
    offset: u64,
    src: &'a [u8],
    order: usize,
}

fn slots_for_len(len: u64) -> u64 {
    let page = u64::from(PAGE_SIZE);
    len.saturating_add(page - 1) / page
}

fn file_len(file: &File) -> u64 {
    file.metadata().map_or(0, |m| m.len())
}

fn file_allocated_bytes(file: &File) -> u64 {
    use std::os::unix::fs::MetadataExt;

    file.metadata()
        .map_or(0, |m| m.blocks().saturating_mul(512))
}

fn reclaimable_tail_bytes(file: &File, target_len: u64) -> u64 {
    file_len(file).saturating_sub(target_len)
}

fn shrink_file_to_len(file: &File, len: u64) -> Result<u64> {
    let current = file_len(file);
    if current <= len {
        return Ok(0);
    }
    file.set_len(len)?;
    file.sync_all()?;
    Ok(current - len)
}

fn copy_advisory_slot(file: &File, from_slot: u64, to_slot: u64, buf: &mut [u8]) -> Result<u64> {
    use std::os::unix::fs::FileExt;

    debug_assert_eq!(buf.len(), PAGE_SIZE as usize);
    let from = from_slot.saturating_mul(u64::from(PAGE_SIZE));
    let to = to_slot.saturating_mul(u64::from(PAGE_SIZE));
    let source_len = file_len(file);
    buf.fill(0);

    if from < source_len {
        let available = (source_len - from).min(u64::from(PAGE_SIZE)) as usize;
        let mut filled = 0usize;
        while filled < available {
            match file.read_at(&mut buf[filled..available], from + filled as u64) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(Error::BlobStoreIo(err)),
            }
        }
        file.write_all_at(buf, to)?;
        Ok(u64::from(PAGE_SIZE))
    } else if to < source_len {
        let zeros = [0u8; READ_INDEX_IO_ALIGN];
        file.write_all_at(&zeros, to)?;
        Ok(READ_INDEX_IO_ALIGN as u64)
    } else {
        Ok(0)
    }
}

#[cfg(target_os = "linux")]
fn punch_file_range(file: &File, offset: u64, len: u64) -> Result<u64> {
    if len == 0 {
        return Ok(0);
    }

    let file_len = file_len(file);
    if offset >= file_len {
        return Ok(0);
    }
    let len = len.min(file_len - offset);
    let offset = libc::off_t::try_from(offset)
        .map_err(|_| Error::BlobStoreIo(io::Error::other("hole punch offset overflow")))?;
    let len = libc::off_t::try_from(len)
        .map_err(|_| Error::BlobStoreIo(io::Error::other("hole punch length overflow")))?;
    let mode = libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE;
    loop {
        let rc = unsafe { libc::fallocate(file.as_raw_fd(), mode, offset, len) };
        if rc == 0 {
            return Ok(len as u64);
        }
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        if hole_punch_unsupported(&err) {
            return Ok(0);
        }
        return Err(Error::BlobStoreIo(err));
    }
}

#[cfg(not(target_os = "linux"))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "non-Linux stub keeps the Linux fallible helper signature"
)]
fn punch_file_range(_file: &File, _offset: u64, _len: u64) -> Result<u64> {
    Ok(0)
}

fn round_up_slots(required_slots: u64) -> u64 {
    let chunk = if required_slots >= DATA_PREALLOC_LARGE_AT_SLOTS {
        DATA_PREALLOC_LARGE_CHUNK_SLOTS
    } else {
        DATA_PREALLOC_SMALL_CHUNK_SLOTS
    };
    required_slots.saturating_add(chunk - 1) / chunk * chunk
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
fn registered_buffer_slots(buffer_pool_size: usize) -> usize {
    buffer_pool_size.clamp(1, REGISTERED_BUFFER_MAX_SLOTS)
}

#[cfg(target_os = "linux")]
fn preallocate_data_file(file: &File, len: u64) -> Result<()> {
    let len = libc::off_t::try_from(len)
        .map_err(|_| Error::BlobStoreIo(io::Error::other("data file length exceeds off_t")))?;
    let rc = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len) };
    if rc == 0 {
        return Ok(());
    }
    let err = io::Error::from_raw_os_error(rc);
    if preallocate_unsupported(&err) {
        return Ok(());
    }
    Err(Error::BlobStoreIo(err))
}

#[cfg(target_os = "macos")]
fn preallocate_data_file(file: &File, len: u64) -> Result<()> {
    let current = file.metadata()?.len();
    if current >= len {
        return Ok(());
    }
    let reserve = libc::off_t::try_from(len - current)
        .map_err(|_| Error::BlobStoreIo(io::Error::other("data file length exceeds off_t")))?;
    let mut store = libc::fstore_t {
        fst_flags: libc::F_ALLOCATECONTIG,
        fst_posmode: libc::F_PEOFPOSMODE,
        fst_offset: 0,
        fst_length: reserve,
        fst_bytesalloc: 0,
    };
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_PREALLOCATE, &store) };
    if rc != 0 {
        store.fst_flags = libc::F_ALLOCATEALL;
        let fallback_rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_PREALLOCATE, &store) };
        if fallback_rc != 0 {
            let err = io::Error::last_os_error();
            if preallocate_unsupported(&err) {
                return Ok(());
            }
            return Err(Error::BlobStoreIo(err));
        }
    }

    file.set_len(len)?;
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn preallocate_data_file(_file: &File, _len: u64) -> Result<()> {
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn preallocate_unsupported(err: &io::Error) -> bool {
    let Some(raw) = err.raw_os_error() else {
        return false;
    };
    raw == libc::ENOSYS || raw == libc::EINVAL || raw == libc::EOPNOTSUPP || {
        #[cfg(target_os = "macos")]
        {
            raw == libc::ENOTSUP
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

#[cfg(target_os = "linux")]
fn hole_punch_unsupported(err: &io::Error) -> bool {
    let Some(raw) = err.raw_os_error() else {
        return false;
    };
    raw == libc::ENOSYS || raw == libc::EINVAL || raw == libc::EOPNOTSUPP || raw == libc::ENOTTY
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
impl FileBlobStore {
    pub(crate) unsafe fn alloc_blob_buf_uninit(&self) -> AlignedBlobBuf {
        if let Some(pool) = &self.registered_buffers {
            // SAFETY: this method's caller upholds the
            // initialization contract before reading the returned
            // buffer.
            if let Some(buf) = unsafe { AlignedBlobBuf::pooled_uninit(pool) } {
                return buf;
            }
        }
        // SAFETY: this method's caller upholds the initialization
        // contract before reading the returned buffer.
        unsafe { AlignedBlobBuf::uninit() }
    }
}

impl BlobStore for FileBlobStore {
    fn alloc_blob_buf_zeroed(&self) -> AlignedBlobBuf {
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        if let Some(pool) = &self.registered_buffers {
            if let Some(buf) = AlignedBlobBuf::pooled_zeroed(pool) {
                return buf;
            }
        }
        AlignedBlobBuf::zeroed()
    }

    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
        let _slot = self.enter_slot_read();
        let offset = self.offset_of(guid)?;
        self.pread_at(offset, dst)?;
        Ok(())
    }

    /// Batched full-frame read. On `io_uring` every read goes down a
    /// single ring submission (one `Mutex` acquire, queue depth =
    /// batch width) instead of N serialised `pread_at` calls; on the
    /// `pread` path the lock-free positional reads fan out across
    /// worker threads for the same device parallelism.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn read_blobs(&self, guids: &[BlobGuid], dsts: &mut [AlignedBlobBuf]) -> Vec<Result<()>> {
        debug_assert_eq!(guids.len(), dsts.len());
        let _slot = self.enter_slot_read();
        // Resolve offsets up front (lock-free manifest read). A guid
        // that doesn't resolve is reported in its own slot and left
        // out of the ring batch, so one bad guid can't sink the rest.
        let offsets: Vec<Result<u64>> = guids.iter().map(|g| self.offset_of(*g)).collect();

        let mut batch: Vec<(u64, &mut AlignedBlobBuf)> = Vec::with_capacity(dsts.len());
        for (off, dst) in offsets.iter().zip(dsts.iter_mut()) {
            if let Ok(off) = off {
                batch.push((*off, dst));
            }
        }

        // The ring batch is all-or-nothing on its first error. For
        // best-effort read-ahead that's fine: every resolved slot is
        // marked failed and the caller re-pins those guids one by one,
        // surfacing the real per-guid status there.
        let batch_result = if batch.is_empty() {
            Ok(())
        } else {
            let mut ring = self.uring.lock().unwrap();
            ring.pread_many_at(&mut batch)
        };
        drop(batch);

        offsets
            .into_iter()
            .map(|off| match off {
                Err(e) => Err(e),
                Ok(_) => match &batch_result {
                    Ok(()) => Ok(()),
                    Err(e) => Err(Error::BlobStoreIo(io::Error::other(format!(
                        "batched uring read failed: {e}"
                    )))),
                },
            })
            .collect()
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn read_blobs(&self, guids: &[BlobGuid], dsts: &mut [AlignedBlobBuf]) -> Vec<Result<()>> {
        const FANOUT: usize = 8;
        debug_assert_eq!(guids.len(), dsts.len());
        if guids.len() < 2 {
            return guids
                .iter()
                .zip(dsts.iter_mut())
                .map(|(g, d)| self.read_blob(*g, d))
                .collect();
        }
        // `read_blob` → `read_exact_at` is a lock-free positional read
        // with no shared state, so fanning the batch across worker
        // threads gives device queue depth = worker count — the same
        // parallelism the io_uring path gets from one batched ring
        // submission.
        let workers = guids.len().min(FANOUT);
        let chunk = guids.len().div_ceil(workers);
        let mut results: Vec<Result<()>> = Vec::with_capacity(guids.len());
        std::thread::scope(|scope| {
            let handles: Vec<_> = guids
                .chunks(chunk)
                .zip(dsts.chunks_mut(chunk))
                .map(|(gs, ds)| {
                    scope.spawn(move || {
                        gs.iter()
                            .zip(ds.iter_mut())
                            .map(|(g, d)| self.read_blob(*g, d))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in handles {
                results.extend(h.join().expect("read_blobs worker panicked"));
            }
        });
        results
    }

    /// Positional ranged read for page-granular cold lookups. `byte_offset`,
    /// `dst.len()`, and `dst`'s base must be 4 KB-aligned (whole pages) so the
    /// `O_DIRECT` / `F_NOCACHE` read is accepted; the buffer-manager paging
    /// layer guarantees this. Linux `io_uring` builds use the data-file ring;
    /// other Unix builds use a plain positional `pread`.
    fn read_blob_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<()> {
        debug_assert_eq!(
            byte_offset % 4096,
            0,
            "ranged read offset must be 4 KB-aligned"
        );
        debug_assert_eq!(
            dst.len() % 4096,
            0,
            "ranged read length must be a 4 KB multiple"
        );
        let _slot = self.enter_slot_read();
        let offset = self.offset_of(guid)? + byte_offset;

        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        {
            let mut ring = self.uring.lock().unwrap();
            ring.pread_slice_at(offset, dst)?;
        }

        #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
        {
            use std::os::unix::fs::FileExt;
            self.data_file.read_exact_at(dst, offset)?;
        }
        Ok(())
    }

    fn read_index_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<bool> {
        let _slot = self.enter_slot_read();
        let Some(base_offset) = self.read_index_offset_of(guid) else {
            return Ok(false);
        };
        if dst.is_empty() {
            return Ok(true);
        }
        let start = usize::try_from(byte_offset)
            .map_err(|_| Error::BlobStoreIo(io::Error::other("read index offset")))?;
        let Some(end) = start.checked_add(dst.len()) else {
            return Ok(false);
        };
        if end > READ_INDEX_SLOT_BYTES {
            return Ok(false);
        }
        let offset = base_offset + byte_offset;

        use std::os::unix::fs::FileExt;
        let mut filled = 0;
        while filled < dst.len() {
            match self
                .read_index_file
                .read_at(&mut dst[filled..], offset + filled as u64)
            {
                Ok(0) => {
                    dst[filled..].fill(0);
                    break;
                }
                Ok(n) => filled += n,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(Error::BlobStoreIo(err)),
            }
        }
        Ok(true)
    }

    fn read_value_segment_range(
        &self,
        guid: BlobGuid,
        byte_offset: u64,
        dst: &mut [u8],
    ) -> Result<bool> {
        let _slot = self.enter_slot_read();
        let Some(base_offset) = self.value_segment_offset_of(guid) else {
            return Ok(false);
        };
        if dst.is_empty() {
            return Ok(true);
        }
        let start = usize::try_from(byte_offset)
            .map_err(|_| Error::BlobStoreIo(io::Error::other("value segment offset")))?;
        let Some(end) = start.checked_add(dst.len()) else {
            return Ok(false);
        };
        if end > VALUE_SEGMENT_SLOT_BYTES {
            return Ok(false);
        }

        use std::os::unix::fs::FileExt;
        let offset = base_offset + byte_offset;
        let mut filled = 0;
        while filled < dst.len() {
            match self
                .value_segment_file
                .read_at(&mut dst[filled..], offset + filled as u64)
            {
                Ok(0) => {
                    dst[filled..].fill(0);
                    break;
                }
                Ok(n) => filled += n,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(Error::BlobStoreIo(err)),
            }
        }
        Ok(true)
    }

    fn publish_read_index(&self, guid: BlobGuid, bytes: &[u8], value_bytes: &[u8]) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
        let Some(base_offset) = self.read_index_offset_of(guid) else {
            return Ok(());
        };
        let Some(value_base_offset) = self.value_segment_offset_of(guid) else {
            return Ok(());
        };
        if bytes.is_empty()
            || bytes.len() > READ_INDEX_SLOT_BYTES
            || value_bytes.len() > VALUE_SEGMENT_SLOT_BYTES
        {
            self.clear_read_accelerator_slots(guid)?;
            return Ok(());
        }

        if !value_bytes.is_empty() {
            let aligned_value_len = align_up(value_bytes.len(), VALUE_SEGMENT_IO_ALIGN);
            let mut direct = AlignedBlobBuf::zeroed();
            direct.as_mut_slice()[..value_bytes.len()].copy_from_slice(value_bytes);
            self.write_value_segment_aligned(
                value_base_offset,
                &direct.as_mut_slice()[..aligned_value_len],
            )?;
        }

        let aligned_len = align_up(bytes.len(), READ_INDEX_IO_ALIGN);
        let mut direct = AlignedBlobBuf::zeroed();
        direct.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        if aligned_len > READ_INDEX_IO_ALIGN {
            self.write_read_index_aligned(
                base_offset + READ_INDEX_IO_ALIGN as u64,
                &direct.as_mut_slice()[READ_INDEX_IO_ALIGN..aligned_len],
            )?;
        }
        self.write_read_index_aligned(base_offset, &direct.as_mut_slice()[..READ_INDEX_IO_ALIGN])?;
        Ok(())
    }

    fn delete_read_index(&self, guid: BlobGuid) -> Result<()> {
        self.clear_read_accelerator_slots(guid)
    }

    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
        self.remove_read_accelerators_best_effort(guid);
        let entry = self.assign_write_entry(guid);
        let offset = entry.slot * u64::from(PAGE_SIZE);
        self.ensure_data_capacity(entry.slot.saturating_add(1))?;
        self.mark_data_write_started();
        self.pwrite_at(offset, src)?;
        Ok(())
    }

    fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
        let io = self.prepare_blob_writes(writes)?;
        if io.is_empty() {
            return Ok(());
        }
        for (guid, _) in writes {
            self.remove_read_accelerators_best_effort(*guid);
        }
        self.mark_data_write_started();
        self.pwrite_many_at(&io)?;
        Ok(())
    }

    fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }
        let _io = self.data_io_lock.lock().unwrap();

        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        {
            let io = self.prepare_blob_writes(writes)?;
            for (guid, _) in writes {
                self.remove_read_accelerators_best_effort(*guid);
            }
            let epoch = self.mark_data_write_started();
            self.pwrite_many_and_sync_at(&io)?;
            self.mark_data_synced(epoch);
            Ok(())
        }

        #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
        {
            let io = self.prepare_blob_writes(writes)?;
            for (guid, _) in writes {
                self.remove_read_accelerators_best_effort(*guid);
            }
            self.mark_data_write_started();
            self.pwrite_many_at(&io)?;
            Ok(())
        }
    }

    fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
        self.remove_read_accelerators_best_effort(guid);
        let mut m = self.manifest.write().unwrap();
        if let Some(entry) = m.entries.remove(&guid) {
            m.pending_free_slots.push(entry.slot);
            m.pending_log.push(ManifestDelta::Delete { guid });
            self.manifest_dirty.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
        let m = self.manifest.read().unwrap();
        Ok(m.entries.keys().copied().collect())
    }

    fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
        Ok(self.manifest.read().unwrap().entries.contains_key(&guid))
    }

    fn store_stats(&self) -> StoreStats {
        let (
            live_blobs,
            next_slot,
            reusable_slots,
            tail_reclaimable_slots,
            pending_free_slots,
            manifest_log_bytes,
        ) = {
            let m = self.manifest.read().unwrap();
            let reusable = m.reusable_slots.len();
            let tail = m.reusable_slots.tail_len(m.next_slot);
            (
                m.entries.len(),
                m.next_slot,
                reusable,
                tail,
                m.pending_free_slots.len() as u64,
                m.log_bytes,
            )
        };
        let high_water = next_slot.saturating_mul(u64::from(PAGE_SIZE));
        let tail_target = next_slot
            .saturating_sub(tail_reclaimable_slots)
            .saturating_mul(u64::from(PAGE_SIZE));
        let tail_reclaimable_bytes = reclaimable_tail_bytes(&self.data_file, tail_target)
            .saturating_add(reclaimable_tail_bytes(&self.read_index_file, tail_target))
            .saturating_add(reclaimable_tail_bytes(
                &self.value_segment_file,
                tail_target,
            ));
        StoreStats {
            live_blobs,
            live_slots: live_blobs as u64,
            next_slot,
            reusable_slots,
            pending_free_slots,
            data_file_bytes: file_len(&self.data_file),
            data_allocated_bytes: file_allocated_bytes(&self.data_file),
            data_high_water_bytes: high_water,
            read_index_file_bytes: file_len(&self.read_index_file),
            read_index_allocated_bytes: file_allocated_bytes(&self.read_index_file),
            read_index_high_water_bytes: high_water,
            value_segment_file_bytes: file_len(&self.value_segment_file),
            value_segment_allocated_bytes: file_allocated_bytes(&self.value_segment_file),
            value_segment_high_water_bytes: high_water,
            tail_reclaimable_slots,
            tail_reclaimable_bytes,
            middle_reusable_slots: reusable_slots.saturating_sub(tail_reclaimable_slots),
            manifest_log_bytes,
        }
    }

    fn flush(&self) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
        self.flush_locked()
    }

    fn needs_flush(&self) -> bool {
        self.data_needs_sync().is_some() || self.manifest_dirty.load(Ordering::Acquire)
    }

    fn vacuum(&self) -> Result<VacuumStats> {
        let _io = self.data_io_lock.lock().unwrap();
        self.flush_locked()?;
        let _slot = self.enter_slot_write();

        let plan = {
            let m = self.manifest.read().unwrap();
            m.relocation_plan()
        };
        let bytes_relocated = self.copy_relocated_slots(&plan)?;

        let (slots_trimmed, next_slot, free_ranges) = {
            let mut m = self.manifest.write().unwrap();
            let slots_trimmed = if plan.is_empty() {
                m.trim_trailing_free_slots()
            } else {
                m.apply_relocation_plan(&plan)?
            };
            if slots_trimmed != 0 || !plan.is_empty() {
                m.persist_snapshot(&self.data_dir)?;
                m.truncate_log()?;
                m.pending_log.clear();
                self.manifest_dirty.store(false, Ordering::Release);
            }
            (
                slots_trimmed,
                m.next_slot,
                m.reusable_slots.compact_ranges(),
            )
        };

        let bytes_truncated = if slots_trimmed == 0 {
            0
        } else {
            self.shrink_packed_files(next_slot)?
        };
        let (slots_punched, bytes_punched) = self.punch_reusable_slot_ranges(&free_ranges)?;
        Ok(VacuumStats {
            unreachable_blobs: 0,
            slots_trimmed,
            slots_relocated: plan.len() as u64,
            bytes_truncated,
            bytes_relocated,
            slots_punched,
            bytes_punched,
        })
    }
}

#[derive(Clone, Copy)]
struct PreparedBlobWrite<'a> {
    offset: u64,
    src: &'a AlignedBlobBuf,
}

#[derive(Debug, Clone, Copy)]
struct SlotMove {
    guid: BlobGuid,
    from_slot: u64,
    to_slot: u64,
}

impl Manifest {
    fn load_or_create(path: &Path, log_path: &Path) -> Result<Self> {
        let (mut entries, mut next_slot) = match File::open(path) {
            Ok(mut f) => Self::parse_snapshot(&mut f)?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => (HashMap::new(), 0),
            Err(e) => return Err(Error::BlobStoreIo(e)),
        };

        let replay = Self::replay_log(log_path, &mut entries, &mut next_slot)?;
        if replay.valid_bytes < replay.file_bytes {
            truncate_manifest_log(log_path, replay.valid_bytes)?;
        }
        let used_slots: Vec<_> = entries.values().map(|entry| entry.slot).collect();
        let reusable_slots = ReusableSlots::reconstruct(next_slot, &used_slots)?;

        Ok(Self {
            entries,
            next_slot,
            reusable_slots,
            pending_free_slots: Vec::new(),
            path: path.to_path_buf(),
            log_path: log_path.to_path_buf(),
            log_bytes: replay.valid_bytes,
            pending_log: Vec::new(),
        })
    }

    fn parse_snapshot(f: &mut File) -> Result<(HashMap<BlobGuid, ManifestEntry>, u64)> {
        // Header: magic 8 + version 2 + count 4 + reserved 2 + next_slot 8 = 24 B.
        let mut hdr = [0u8; 24];
        f.read_exact(&mut hdr)?;
        if hdr[..8] != MANIFEST_MAGIC {
            return Err(Error::node_corrupt("FileBlobStore::Manifest::magic"));
        }
        let version = u16::from_le_bytes([hdr[8], hdr[9]]);
        if version != MANIFEST_VERSION {
            return Err(Error::node_corrupt(
                "FileBlobStore::Manifest::version (older manifests are not migrated)",
            ));
        }
        let count = u32::from_le_bytes([hdr[10], hdr[11], hdr[12], hdr[13]]) as usize;
        // hdr[14..16] reserved (zero).
        let next_slot = u64::from_le_bytes(hdr[16..24].try_into().unwrap());

        let mut entries = HashMap::with_capacity(count);
        let mut used_slots = Vec::with_capacity(count);
        let mut entry = [0u8; 24];
        for _ in 0..count {
            f.read_exact(&mut entry)?;
            let mut g: BlobGuid = [0u8; 16];
            g.copy_from_slice(&entry[..16]);
            let s = u64::from_le_bytes(entry[16..24].try_into().unwrap());
            if entries.insert(g, ManifestEntry { slot: s }).is_some() {
                return Err(Error::node_corrupt(
                    "FileBlobStore::Manifest::duplicate guid",
                ));
            }
            used_slots.push(s);
        }
        ReusableSlots::reconstruct(next_slot, &used_slots)?;
        Ok((entries, next_slot))
    }

    fn assign_write_entry(&mut self, guid: BlobGuid) -> ManifestEntry {
        // Re-writing an existing blob reuses its slot (COW overwrite in
        // place); a new blob gets a fresh slot.
        if let Some(entry) = self.entries.get(&guid) {
            return *entry;
        }
        let entry = ManifestEntry {
            slot: self.allocate_slot(),
        };
        self.entries.insert(guid, entry);
        entry
    }

    fn allocate_slot(&mut self) -> u64 {
        self.reusable_slots.pop().unwrap_or_else(|| {
            let slot = self.next_slot;
            self.next_slot += 1;
            slot
        })
    }

    fn publish_pending_free_slots(&mut self) {
        if self.pending_free_slots.is_empty() {
            return;
        }
        self.reusable_slots
            .append_slots(&mut self.pending_free_slots);
    }

    fn trim_trailing_free_slots(&mut self) -> u64 {
        self.publish_pending_free_slots();
        self.reusable_slots.trim_trailing(&mut self.next_slot)
    }

    fn relocation_plan(&self) -> Vec<SlotMove> {
        let mut live: Vec<_> = self
            .entries
            .iter()
            .map(|(guid, entry)| (entry.slot, *guid))
            .collect();
        if live.is_empty() {
            return Vec::new();
        }
        live.sort_unstable_by_key(|(slot, _)| *slot);

        let mut free_ranges = self.reusable_slots.compact_ranges().into_iter();
        let Some(mut free) = free_ranges.next() else {
            return Vec::new();
        };

        let mut live_idx = live.len();
        let mut plan = Vec::new();
        while live_idx != 0 {
            while free.next > free.end {
                let Some(next) = free_ranges.next() else {
                    return plan;
                };
                free = next;
            }

            let (from_slot, guid) = live[live_idx - 1];
            if from_slot <= free.next {
                return plan;
            }

            plan.push(SlotMove {
                guid,
                from_slot,
                to_slot: free.next,
            });
            free.next = free.next.saturating_add(1);
            live_idx -= 1;
        }
        plan
    }

    fn apply_relocation_plan(&mut self, plan: &[SlotMove]) -> Result<u64> {
        if plan.is_empty() {
            return Ok(0);
        }

        for item in plan {
            let Some(entry) = self.entries.get_mut(&item.guid) else {
                return Err(Error::node_corrupt(
                    "FileBlobStore::Manifest::relocate guid",
                ));
            };
            if entry.slot != item.from_slot {
                return Err(Error::node_corrupt(
                    "FileBlobStore::Manifest::relocate slot",
                ));
            }
            entry.slot = item.to_slot;
        }

        let used_slots: Vec<_> = self.entries.values().map(|entry| entry.slot).collect();
        self.reusable_slots = ReusableSlots::reconstruct(self.next_slot, &used_slots)?;
        Ok(self.trim_trailing_free_slots())
    }

    fn persist_pending_deltas(&mut self, data_dir: &Path) -> Result<()> {
        if self.pending_log.is_empty() {
            return Ok(());
        }

        let log_created = !self.log_path.exists();
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        let mut buf = Vec::with_capacity(self.pending_log.len() * 40);
        for delta in &self.pending_log {
            encode_manifest_delta(*delta, &mut buf)?;
        }
        f.write_all(&buf)?;
        f.sync_data()?;
        drop(f);
        if log_created {
            sync_dir(data_dir)?;
        }

        self.log_bytes = self.log_bytes.saturating_add(buf.len() as u64);
        if self.should_compact_log() {
            self.persist_snapshot(data_dir)?;
            self.truncate_log()?;
        }
        Ok(())
    }

    fn should_compact_log(&self) -> bool {
        let snapshot_bytes = 24u64.saturating_add((self.entries.len() as u64).saturating_mul(32));
        self.log_bytes >= MANIFEST_LOG_MIN_COMPACT_BYTES
            && self.log_bytes >= snapshot_bytes.saturating_mul(MANIFEST_LOG_COMPACT_RATIO)
    }

    fn persist_snapshot(&self, data_dir: &Path) -> Result<()> {
        let tmp_path = data_dir.join(MANIFEST_TMP_FILENAME);
        let final_path = &self.path;

        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;

        let mut hdr = [0u8; 16];
        hdr[..8].copy_from_slice(&MANIFEST_MAGIC);
        hdr[8..10].copy_from_slice(&MANIFEST_VERSION.to_le_bytes());
        let count = u32::try_from(self.entries.len()).map_err(|_| {
            Error::BlobStoreIo(io::Error::other("manifest slot count exceeds u32::MAX"))
        })?;
        hdr[10..14].copy_from_slice(&count.to_le_bytes());
        // Bytes 14..16 reserved (zero).
        f.write_all(&hdr)?;
        f.write_all(&self.next_slot.to_le_bytes())?;

        for (g, entry) in &self.entries {
            f.write_all(g)?;
            f.write_all(&entry.slot.to_le_bytes())?;
        }

        f.sync_all()?;
        drop(f);

        std::fs::rename(&tmp_path, final_path)?;
        // Sync the parent directory so the rename itself is durable
        // (required by POSIX; ext4/xfs honour it).
        sync_dir(data_dir)?;
        Ok(())
    }

    fn truncate_log(&mut self) -> Result<()> {
        match OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.log_path)
        {
            Ok(f) => {
                f.sync_data()?;
                self.log_bytes = 0;
                Ok(())
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                self.log_bytes = 0;
                Ok(())
            }
            Err(e) => Err(Error::BlobStoreIo(e)),
        }
    }

    fn replay_log(
        log_path: &Path,
        entries: &mut HashMap<BlobGuid, ManifestEntry>,
        next_slot: &mut u64,
    ) -> Result<ManifestLogReplay> {
        let mut f = match File::open(log_path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(ManifestLogReplay {
                    file_bytes: 0,
                    valid_bytes: 0,
                });
            }
            Err(e) => return Err(Error::BlobStoreIo(e)),
        };
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let mut offset = 0usize;
        let mut valid_offset = 0usize;
        while offset < buf.len() {
            let remaining = buf.len() - offset;
            if remaining < MANIFEST_LOG_HEADER_SIZE {
                break;
            }
            let record_start = offset;
            if buf[offset..offset + 4] != MANIFEST_LOG_MAGIC {
                return Err(Error::node_corrupt("FileBlobStore::ManifestLog::magic"));
            }
            offset += 4;
            let body_len = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let ty = buf[offset];
            offset += 1;
            let record_len = MANIFEST_LOG_HEADER_SIZE
                .saturating_add(body_len)
                .saturating_add(MANIFEST_LOG_FOOTER_SIZE);
            if buf.len() - record_start < record_len {
                break;
            }
            let expected_crc = u32::from_le_bytes(
                buf[offset + body_len..offset + body_len + 4]
                    .try_into()
                    .unwrap(),
            );
            let actual_crc = crc32fast::hash(&buf[record_start..offset + body_len]);
            if expected_crc != actual_crc {
                return Err(Error::node_corrupt("FileBlobStore::ManifestLog::crc"));
            }
            let body = &buf[offset..offset + body_len];
            match ty {
                MANIFEST_LOG_TY_SET => {
                    if body.len() != MANIFEST_LOG_SET_BODY_SIZE {
                        return Err(Error::node_corrupt(
                            "FileBlobStore::ManifestLog::set length",
                        ));
                    }
                    let mut guid = [0u8; 16];
                    guid.copy_from_slice(&body[..16]);
                    let slot = u64::from_le_bytes(body[16..24].try_into().unwrap());
                    entries.insert(guid, ManifestEntry { slot });
                    *next_slot = (*next_slot).max(slot.saturating_add(1));
                }
                MANIFEST_LOG_TY_DELETE => {
                    if body.len() != MANIFEST_LOG_DELETE_BODY_SIZE {
                        return Err(Error::node_corrupt(
                            "FileBlobStore::ManifestLog::delete length",
                        ));
                    }
                    let mut guid = [0u8; 16];
                    guid.copy_from_slice(body);
                    entries.remove(&guid);
                }
                _ => {
                    return Err(Error::node_corrupt(
                        "FileBlobStore::ManifestLog::unknown op",
                    ));
                }
            }
            offset = record_start + record_len;
            valid_offset = offset;
        }
        Ok(ManifestLogReplay {
            file_bytes: buf.len() as u64,
            valid_bytes: valid_offset as u64,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ManifestLogReplay {
    file_bytes: u64,
    valid_bytes: u64,
}

fn encode_manifest_delta(delta: ManifestDelta, out: &mut Vec<u8>) -> Result<()> {
    let start = out.len();
    out.extend_from_slice(&MANIFEST_LOG_MAGIC);
    let len_pos = out.len();
    out.extend_from_slice(&[0u8; 4]);
    match delta {
        ManifestDelta::Set { guid, slot } => {
            out.push(MANIFEST_LOG_TY_SET);
            out.extend_from_slice(&guid);
            out.extend_from_slice(&slot.to_le_bytes());
        }
        ManifestDelta::Delete { guid } => {
            out.push(MANIFEST_LOG_TY_DELETE);
            out.extend_from_slice(&guid);
        }
    }
    let body_len = out.len() - start - MANIFEST_LOG_HEADER_SIZE;
    let body_len = u32::try_from(body_len)
        .map_err(|_| Error::BlobStoreIo(io::Error::other("manifest delta record too large")))?;
    out[len_pos..len_pos + 4].copy_from_slice(&body_len.to_le_bytes());
    let crc = crc32fast::hash(&out[start..]);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(())
}

fn sync_dir(path: &Path) -> Result<()> {
    let dir = File::open(path)?;
    dir.sync_all()?;
    Ok(())
}

fn truncate_manifest_log(path: &Path, valid_bytes: u64) -> Result<()> {
    let f = OpenOptions::new().write(true).open(path)?;
    f.set_len(valid_bytes)?;
    f.sync_all()?;
    Ok(())
}

impl ReusableSlots {
    fn pop(&mut self) -> Option<u64> {
        if let Some(slot) = self.singles.pop() {
            return Some(slot);
        }

        let idx = self.ranges.len().checked_sub(1)?;
        let (slot, exhausted) = {
            let range = &mut self.ranges[idx];
            let slot = range.next;
            let exhausted = range.next == range.end;
            if !exhausted {
                range.next += 1;
            }
            (slot, exhausted)
        };
        if exhausted {
            self.ranges.pop();
        }
        Some(slot)
    }

    fn append_slots(&mut self, slots: &mut Vec<u64>) {
        self.singles.append(slots);
    }

    fn trim_trailing(&mut self, next_slot: &mut u64) -> u64 {
        let original = *next_slot;
        self.singles.sort_unstable();

        while let Some(tail) = next_slot.checked_sub(1) {
            if self.singles.last().copied() == Some(tail) {
                self.singles.pop();
                *next_slot = tail;
                continue;
            }

            let Some(range_idx) = self
                .ranges
                .iter()
                .position(|range| range.next <= tail && tail <= range.end)
            else {
                break;
            };

            let lower = self.ranges[range_idx].next;
            self.ranges.swap_remove(range_idx);
            *next_slot = lower;
        }

        original.saturating_sub(*next_slot)
    }

    fn tail_len(&self, next_slot: u64) -> u64 {
        let mut tail = next_slot;
        for range in self.compact_ranges().iter().rev() {
            let Some(wanted) = tail.checked_sub(1) else {
                break;
            };
            if range.end < wanted {
                break;
            }
            if range.next <= wanted {
                tail = range.next;
            }
        }
        next_slot.saturating_sub(tail)
    }

    fn compact_ranges(&self) -> Vec<FreeSlotRange> {
        let mut ranges = Vec::with_capacity(self.ranges.len() + self.singles.len());
        ranges.extend(self.ranges.iter().copied());
        ranges.extend(self.singles.iter().copied().map(|slot| FreeSlotRange {
            next: slot,
            end: slot,
        }));
        if ranges.is_empty() {
            return ranges;
        }

        ranges.sort_unstable_by_key(|range| range.next);
        let mut compacted: Vec<FreeSlotRange> = Vec::with_capacity(ranges.len());
        for range in ranges {
            let Some(last) = compacted.last_mut() else {
                compacted.push(range);
                continue;
            };
            if range.next <= last.end.saturating_add(1) {
                last.end = last.end.max(range.end);
            } else {
                compacted.push(range);
            }
        }
        compacted
    }

    fn len(&self) -> u64 {
        let singles = self.singles.len() as u64;
        let ranges = self
            .ranges
            .iter()
            .map(|range| range.slot_count())
            .sum::<u64>();
        singles.saturating_add(ranges)
    }

    fn reconstruct(next_slot: u64, used_slots: &[u64]) -> Result<Self> {
        let mut sorted = used_slots.to_vec();
        sorted.sort_unstable();

        let mut previous = None;
        let mut lower = 0u64;
        let mut ranges = Vec::new();
        for &slot in &sorted {
            if slot >= next_slot {
                return Err(Error::node_corrupt(
                    "FileBlobStore::Manifest::slot past next_slot",
                ));
            }
            if previous == Some(slot) {
                return Err(Error::node_corrupt(
                    "FileBlobStore::Manifest::duplicate slot",
                ));
            }
            if lower < slot {
                ranges.push(FreeSlotRange {
                    next: lower,
                    end: slot - 1,
                });
            }
            lower = slot + 1;
            previous = Some(slot);
        }

        if lower < next_slot {
            ranges.push(FreeSlotRange {
                next: lower,
                end: next_slot - 1,
            });
        }
        ranges.reverse();

        Ok(Self {
            singles: Vec::new(),
            ranges,
        })
    }

    #[cfg(test)]
    fn single_count(&self) -> usize {
        self.singles.len()
    }

    #[cfg(test)]
    fn range_count(&self) -> usize {
        self.ranges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf_with(byte_at_100: u8) -> AlignedBlobBuf {
        let mut b = AlignedBlobBuf::zeroed();
        b.as_mut_slice()[100] = byte_at_100;
        b
    }

    #[test]
    fn data_preallocation_rounds_in_adaptive_chunks() {
        assert_eq!(round_up_slots(1), DATA_PREALLOC_SMALL_CHUNK_SLOTS);
        assert_eq!(
            round_up_slots(DATA_PREALLOC_SMALL_CHUNK_SLOTS + 1),
            DATA_PREALLOC_SMALL_CHUNK_SLOTS * 2,
        );
        assert_eq!(
            round_up_slots(DATA_PREALLOC_LARGE_AT_SLOTS),
            DATA_PREALLOC_LARGE_AT_SLOTS,
        );
        assert_eq!(
            round_up_slots(DATA_PREALLOC_LARGE_AT_SLOTS + 1),
            DATA_PREALLOC_LARGE_AT_SLOTS + DATA_PREALLOC_LARGE_CHUNK_SLOTS,
        );
    }

    /// Skip every test in this module when O_DIRECT isn't supported
    /// by the filesystem we landed on (e.g. tmpfs on some kernels,
    /// or macOS-mounted-via-CI). Returns the open store or `None`
    /// to skip cleanly.
    fn try_open(dir: &Path) -> Option<FileBlobStore> {
        match FileBlobStore::open(dir) {
            Ok(b) => Some(b),
            Err(Error::BlobStoreIo(e)) if e.raw_os_error() == Some(libc::EINVAL) => {
                eprintln!("skipping: O_DIRECT not supported on this fs");
                None
            }
            Err(e) => panic!("unexpected open error: {e}"),
        }
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    #[test]
    fn registered_buffer_allocator_returns_fixed_buffers_when_available() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        if b.registered_buffers.is_none() {
            eprintln!("skipping: io_uring fixed-buffer registration unavailable");
            return;
        }

        let mut src = b.alloc_blob_buf_zeroed();
        // SAFETY: read_blob below fills the full frame before the
        // test reads from `dst`.
        let mut dst = unsafe { b.alloc_blob_buf_uninit() };
        assert!(
            src.fixed_buffer_index().is_some(),
            "source buffer should come from the registered pool"
        );
        assert!(
            dst.fixed_buffer_index().is_some(),
            "destination buffer should come from the registered pool"
        );

        src.as_mut_slice()[100] = 0x5A;
        let g: BlobGuid = [0xF1; 16];
        b.write_blob(g, &src).unwrap();
        b.flush().unwrap();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 0x5A);
    }

    #[test]
    fn round_trip_single_blob() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0xAB; 16];
        b.write_blob(g, &buf_with(42)).unwrap();
        b.flush().unwrap();

        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 42);
    }

    #[test]
    fn open_holds_exclusive_dir_lock_until_drop() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };

        // While `b` is live, a second open must be rejected — on
        // 0.5.x a second instance replays the same manifest into
        // the same next_slot and corrupts it with duplicate-slot
        // set deltas.
        let second = acquire_dir_lock(dir.path(), Duration::from_millis(50));
        match second {
            Err(Error::BlobStoreIo(e)) => {
                assert_eq!(e.kind(), io::ErrorKind::WouldBlock, "unexpected error: {e}");
            }
            Err(e) => panic!("unexpected error variant: {e}"),
            Ok(_) => panic!("second open acquired the lock while the store is live"),
        }

        // The handover pattern: once the previous instance is fully
        // dropped, the kernel releases the flock and a fresh open
        // succeeds immediately.
        drop(b);
        let Some(_b2) = try_open(dir.path()) else {
            return;
        };
    }

    #[test]
    fn survives_reopen_after_flush() {
        let dir = tempfile::tempdir().unwrap();
        let g: BlobGuid = [0x55; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g, &buf_with(7)).unwrap();
            b.flush().unwrap();
        }
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 7);
    }

    #[test]
    fn write_replaces_existing_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x33; 16];
        b.write_blob(g, &buf_with(1)).unwrap();
        b.write_blob(g, &buf_with(2)).unwrap();
        b.flush().unwrap();
        assert_eq!(b.len(), 1);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2);
    }

    #[test]
    fn manifest_reuses_slot_on_rewrite_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        let g: BlobGuid = [0x34; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g, &buf_with(1)).unwrap();
            assert_eq!(b.entry_of(g).unwrap().slot, 0);
            b.write_blob(g, &buf_with(2)).unwrap();
            assert_eq!(b.entry_of(g).unwrap().slot, 0, "rewrite reuses the slot");
            b.flush().unwrap();
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(b.entry_of(g).unwrap().slot, 0);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2, "last write persists across reopen");
    }

    #[test]
    fn batch_duplicate_guid_last_write_wins() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x35; 16];
        let one = buf_with(1);
        let two = buf_with(2);
        let three = buf_with(3);

        b.write_blobs(&[(g, &one), (g, &two), (g, &three)]).unwrap();
        b.flush().unwrap();
        drop(b);

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(
            dst.as_slice()[100],
            3,
            "last write of a duplicate guid wins"
        );
    }

    #[test]
    fn needs_flush_tracks_data_and_manifest_work() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x44; 16];

        assert!(!b.needs_flush());
        b.write_blob(g, &buf_with(1)).unwrap();
        assert!(b.needs_flush());
        b.flush().unwrap();
        assert!(!b.needs_flush());

        b.delete_blob(g).unwrap();
        assert!(b.needs_flush());
        b.flush().unwrap();
        assert!(!b.needs_flush());
    }

    #[test]
    fn store_stats_track_slots_and_read_index_space() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g1: BlobGuid = [0x45; 16];
        let g2: BlobGuid = [0x46; 16];

        let empty = b.store_stats();
        assert_eq!(empty.live_blobs, 0);
        assert_eq!(empty.next_slot, 0);
        assert_eq!(empty.data_high_water_bytes, 0);

        b.write_blob(g1, &buf_with(1)).unwrap();
        b.flush().unwrap();
        let written = b.store_stats();
        assert_eq!(written.live_blobs, 1);
        assert_eq!(written.live_slots, 1);
        assert_eq!(written.next_slot, 1);
        assert_eq!(written.data_high_water_bytes, u64::from(PAGE_SIZE));
        assert!(written.data_file_bytes >= u64::from(PAGE_SIZE));

        b.delete_blob(g1).unwrap();
        let pending_delete = b.store_stats();
        assert_eq!(pending_delete.live_blobs, 0);
        assert_eq!(pending_delete.pending_free_slots, 1);
        assert_eq!(pending_delete.reusable_slots, 0);

        b.flush().unwrap();
        let reusable = b.store_stats();
        assert_eq!(reusable.pending_free_slots, 0);
        assert_eq!(reusable.reusable_slots, 1);

        b.write_blob(g2, &buf_with(2)).unwrap();
        b.flush().unwrap();
        b.publish_read_index(g2, &[0xAB; 512], &[0xCD; 512])
            .unwrap();
        let stats = b.store_stats();
        assert_eq!(stats.live_blobs, 1);
        assert_eq!(stats.next_slot, 1, "flushed free slot should be reused");
        assert!(stats.read_index_file_bytes >= 512);
        assert!(stats.value_segment_file_bytes >= 512);
        assert_eq!(stats.read_index_high_water_bytes, u64::from(PAGE_SIZE));
        assert_eq!(stats.value_segment_high_water_bytes, u64::from(PAGE_SIZE));
    }

    #[test]
    fn vacuum_trims_trailing_free_slots_and_accelerators() {
        let dir = tempfile::tempdir().unwrap();
        let g1: BlobGuid = [0x51; 16];
        let g2: BlobGuid = [0x52; 16];
        let g3: BlobGuid = [0x53; 16];

        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g1, &buf_with(1)).unwrap();
            b.write_blob(g2, &buf_with(2)).unwrap();
            b.write_blob(g3, &buf_with(3)).unwrap();
            b.flush().unwrap();
            b.publish_read_index(g3, &[0xAB; 512], &[0xCD; 512])
                .unwrap();

            let before = b.store_stats();
            assert_eq!(before.next_slot, 3);
            assert_eq!(before.tail_reclaimable_slots, 0);
            assert!(before.data_file_bytes >= 3 * u64::from(PAGE_SIZE));
            assert!(before.read_index_file_bytes > u64::from(PAGE_SIZE));
            assert!(before.value_segment_file_bytes > u64::from(PAGE_SIZE));

            b.delete_blob(g3).unwrap();
            b.delete_blob(g2).unwrap();
            b.flush().unwrap();
            let free = b.store_stats();
            assert_eq!(free.tail_reclaimable_slots, 2);
            assert_eq!(free.middle_reusable_slots, 0);
            assert!(free.tail_reclaimable_bytes >= 2 * u64::from(PAGE_SIZE));

            let vacuum = b.vacuum().unwrap();
            assert_eq!(vacuum.slots_trimmed, 2);
            assert!(vacuum.bytes_truncated >= 2 * u64::from(PAGE_SIZE));

            let after = b.store_stats();
            assert_eq!(after.next_slot, 1);
            assert_eq!(after.reusable_slots, 0);
            assert_eq!(after.tail_reclaimable_slots, 0);
            assert_eq!(after.middle_reusable_slots, 0);
            assert_eq!(after.data_file_bytes, u64::from(PAGE_SIZE));
            assert!(after.read_index_file_bytes <= u64::from(PAGE_SIZE));
            assert!(after.value_segment_file_bytes <= u64::from(PAGE_SIZE));
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g1, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);
        assert!(!b.has_blob(g2).unwrap());
        assert!(!b.has_blob(g3).unwrap());
    }

    #[test]
    fn vacuum_relocates_tail_live_slot_into_middle_hole() {
        let dir = tempfile::tempdir().unwrap();
        let g1: BlobGuid = [0x61; 16];
        let g2: BlobGuid = [0x62; 16];
        let g3: BlobGuid = [0x63; 16];
        let g4: BlobGuid = [0x64; 16];

        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g1, &buf_with(1)).unwrap();
            b.write_blob(g2, &buf_with(2)).unwrap();
            b.write_blob(g3, &buf_with(3)).unwrap();
            b.flush().unwrap();
            b.delete_blob(g2).unwrap();
            b.flush().unwrap();

            let free = b.store_stats();
            assert_eq!(free.next_slot, 3);
            assert_eq!(free.tail_reclaimable_slots, 0);
            assert_eq!(free.middle_reusable_slots, 1);

            let vacuum = b.vacuum().unwrap();
            assert_eq!(vacuum.slots_relocated, 1);
            assert_eq!(vacuum.slots_trimmed, 1);
            assert!(vacuum.bytes_relocated >= u64::from(PAGE_SIZE));
            assert!(vacuum.bytes_truncated >= u64::from(PAGE_SIZE));

            let stats = b.store_stats();
            assert_eq!(stats.next_slot, 2);
            assert_eq!(stats.reusable_slots, 0);
            assert_eq!(stats.tail_reclaimable_slots, 0);
            assert_eq!(stats.middle_reusable_slots, 0);
            assert_eq!(
                b.offset_of(g3).unwrap(),
                u64::from(PAGE_SIZE),
                "live tail slot should be relocated into the middle hole",
            );

            b.write_blob(g4, &buf_with(4)).unwrap();
            assert_eq!(
                b.offset_of(g4).unwrap(),
                2 * u64::from(PAGE_SIZE),
                "new writes append after the compacted live set",
            );
            b.flush().unwrap();

            let mut dst = AlignedBlobBuf::zeroed();
            b.read_blob(g1, &mut dst).unwrap();
            assert_eq!(dst.as_slice()[100], 1);
            b.read_blob(g3, &mut dst).unwrap();
            assert_eq!(dst.as_slice()[100], 3);
            b.read_blob(g4, &mut dst).unwrap();
            assert_eq!(dst.as_slice()[100], 4);
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g1, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);
        b.read_blob(g3, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 3);
        b.read_blob(g4, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 4);
    }

    #[test]
    fn deleted_slot_is_reused_only_after_manifest_flush() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g1: BlobGuid = [0x11; 16];
        let g2: BlobGuid = [0x22; 16];
        let g3: BlobGuid = [0x33; 16];

        b.write_blob(g1, &buf_with(1)).unwrap();
        b.flush().unwrap();
        assert_eq!(b.offset_of(g1).unwrap(), 0);

        b.delete_blob(g1).unwrap();
        b.write_blob(g2, &buf_with(2)).unwrap();
        assert_eq!(
            b.offset_of(g2).unwrap(),
            u64::from(PAGE_SIZE),
            "slot removed from manifest but not flushed yet must not be reused",
        );

        b.flush().unwrap();
        b.write_blob(g3, &buf_with(3)).unwrap();
        assert_eq!(
            b.offset_of(g3).unwrap(),
            0,
            "flushed manifest deletion makes slot reusable",
        );

        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g2, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2);
        b.read_blob(g3, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 3);
    }

    #[test]
    fn reusable_slots_are_reconstructed_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let g1: BlobGuid = [0xA1; 16];
        let g2: BlobGuid = [0xA2; 16];
        let g3: BlobGuid = [0xA3; 16];
        let g4: BlobGuid = [0xA4; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g1, &buf_with(1)).unwrap();
            b.write_blob(g2, &buf_with(2)).unwrap();
            b.write_blob(g3, &buf_with(3)).unwrap();
            b.flush().unwrap();
            assert_eq!(b.offset_of(g2).unwrap(), u64::from(PAGE_SIZE));

            b.delete_blob(g2).unwrap();
            b.flush().unwrap();
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        b.write_blob(g4, &buf_with(4)).unwrap();
        assert_eq!(
            b.offset_of(g4).unwrap(),
            u64::from(PAGE_SIZE),
            "reopen should rebuild free slot list from manifest holes",
        );

        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g1, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);
        b.read_blob(g3, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 3);
        b.read_blob(g4, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 4);
    }

    #[test]
    fn vacuum_relocates_single_live_tail_blob_to_lowest_slot() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g1: BlobGuid = [0x71; 16];
        let g2: BlobGuid = [0x72; 16];
        let g3: BlobGuid = [0x73; 16];

        b.write_blob(g1, &buf_with(1)).unwrap();
        b.write_blob(g2, &buf_with(2)).unwrap();
        b.write_blob(g3, &buf_with(3)).unwrap();
        b.flush().unwrap();

        b.delete_blob(g1).unwrap();
        b.delete_blob(g2).unwrap();
        b.flush().unwrap();
        let free = b.store_stats();
        assert_eq!(free.next_slot, 3);
        assert_eq!(free.tail_reclaimable_slots, 0);
        assert_eq!(free.middle_reusable_slots, 2);

        let vacuum = b.vacuum().unwrap();
        assert_eq!(vacuum.slots_relocated, 1);
        assert_eq!(vacuum.slots_trimmed, 2);

        let stats = b.store_stats();
        assert_eq!(stats.next_slot, 1);
        assert_eq!(stats.reusable_slots, 0);
        assert_eq!(stats.tail_reclaimable_slots, 0);
        assert_eq!(stats.middle_reusable_slots, 0);
        assert_eq!(stats.read_index_file_bytes, 0);
        assert_eq!(stats.value_segment_file_bytes, 0);
        assert_eq!(b.offset_of(g3).unwrap(), 0);

        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g3, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 3);
        assert!(!b.has_blob(g1).unwrap());
        assert!(!b.has_blob(g2).unwrap());
    }

    #[test]
    fn vacuum_relocates_read_accelerators_with_blob_slot() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g1: BlobGuid = [0x81; 16];
        let g2: BlobGuid = [0x82; 16];
        let g3: BlobGuid = [0x83; 16];

        b.write_blob(g1, &buf_with(1)).unwrap();
        b.write_blob(g2, &buf_with(2)).unwrap();
        b.write_blob(g3, &buf_with(3)).unwrap();
        b.flush().unwrap();
        b.publish_read_index(g3, &[0xAB; 512], &[0xCD; 512])
            .unwrap();
        b.delete_blob(g2).unwrap();
        b.flush().unwrap();

        let vacuum = b.vacuum().unwrap();
        assert_eq!(vacuum.slots_relocated, 1);
        assert_eq!(b.offset_of(g3).unwrap(), u64::from(PAGE_SIZE));

        let mut idx = [0u8; 512];
        assert!(b.read_index_range(g3, 0, &mut idx).unwrap());
        assert_eq!(idx, [0xAB; 512]);
        let mut val = [0u8; 512];
        assert!(b.read_value_segment_range(g3, 0, &mut val).unwrap());
        assert_eq!(val, [0xCD; 512]);

        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g1, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);
        b.read_blob(g3, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 3);
    }

    #[test]
    fn reusable_slots_reconstruct_sparse_manifest_as_ranges() {
        let mut slots = ReusableSlots::reconstruct(1_000_000, &[0, 999_999]).unwrap();

        assert_eq!(slots.single_count(), 0);
        assert_eq!(slots.range_count(), 1);
        assert_eq!(slots.pop(), Some(1));
        assert_eq!(slots.pop(), Some(2));
    }

    #[test]
    fn reusable_slots_trim_only_contiguous_tail() {
        let mut slots = ReusableSlots::reconstruct(10, &[0, 2, 5]).unwrap();
        let mut next_slot = 10;

        assert_eq!(slots.tail_len(next_slot), 4);
        assert_eq!(slots.trim_trailing(&mut next_slot), 4);
        assert_eq!(next_slot, 6);
        assert_eq!(slots.len(), 3, "holes below the new tail remain reusable");

        slots.append_slots(&mut vec![5]);
        assert_eq!(slots.tail_len(next_slot), 3);
        assert_eq!(slots.trim_trailing(&mut next_slot), 3);
        assert_eq!(next_slot, 3);
        assert_eq!(slots.pop(), Some(1));
    }

    #[test]
    fn batch_write_preserves_duplicate_guid_order() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g1: BlobGuid = [0xB1; 16];
        let g2: BlobGuid = [0xB2; 16];
        let one = buf_with(1);
        let two = buf_with(2);
        let three = buf_with(3);

        b.write_blobs(&[(g1, &one), (g1, &two), (g2, &three)])
            .unwrap();
        b.flush().unwrap();

        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g1, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2);
        b.read_blob(g2, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 3);
    }

    #[test]
    fn manifest_delta_log_replays_without_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let g: BlobGuid = [0xC1; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g, &buf_with(9)).unwrap();
            b.flush().unwrap();
            assert!(dir.path().join(MANIFEST_LOG_FILENAME).exists());
            assert!(!dir.path().join(MANIFEST_FILENAME).exists());
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 9);
    }

    #[test]
    fn manifest_delta_log_ignores_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let g: BlobGuid = [0xC2; 16];
        let g2: BlobGuid = [0xC5; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g, &buf_with(10)).unwrap();
            b.flush().unwrap();
        }
        {
            let mut log = OpenOptions::new()
                .append(true)
                .open(dir.path().join(MANIFEST_LOG_FILENAME))
                .unwrap();
            log.write_all(&MANIFEST_LOG_MAGIC[..3]).unwrap();
            log.sync_data().unwrap();
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 10);
        b.write_blob(g2, &buf_with(11)).unwrap();
        b.flush().unwrap();
        drop(b);

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 10);
        b.read_blob(g2, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 11);
    }

    #[test]
    fn manifest_snapshot_plus_old_log_replay_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let g1: BlobGuid = [0xC3; 16];
        let g2: BlobGuid = [0xC4; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g1, &buf_with(1)).unwrap();
            b.flush().unwrap();
            b.delete_blob(g1).unwrap();
            b.flush().unwrap();
            b.write_blob(g2, &buf_with(2)).unwrap();
            b.flush().unwrap();

            // Simulate the crash-safe middle of log compaction:
            // the new snapshot is durable, but the old log still
            // exists. Replaying that old log over the snapshot
            // must be idempotent and end at the same map.
            b.manifest
                .read()
                .unwrap()
                .persist_snapshot(dir.path())
                .unwrap();
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(b.offset_of(g2).unwrap(), 0);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g2, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2);
        assert!(b.read_blob(g1, &mut dst).is_err());
    }

    #[test]
    fn manifest_delta_log_compacts_to_snapshot_when_large() {
        let dir = tempfile::tempdir().unwrap();
        let g1: BlobGuid = [0xC6; 16];
        let g2: BlobGuid = [0xC7; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g1, &buf_with(1)).unwrap();
            b.flush().unwrap();
            b.manifest.write().unwrap().log_bytes = MANIFEST_LOG_MIN_COMPACT_BYTES;

            b.write_blob(g2, &buf_with(2)).unwrap();
            b.flush().unwrap();
            assert!(dir.path().join(MANIFEST_FILENAME).exists());
            assert_eq!(
                std::fs::metadata(dir.path().join(MANIFEST_LOG_FILENAME))
                    .unwrap()
                    .len(),
                0,
            );
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g1, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);
        b.read_blob(g2, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2);
    }

    #[test]
    fn delete_then_read_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x99; 16];
        b.write_blob(g, &buf_with(5)).unwrap();
        b.delete_blob(g).unwrap();
        let mut dst = AlignedBlobBuf::zeroed();
        assert!(b.read_blob(g, &mut dst).is_err());
    }

    #[test]
    fn read_index_round_trips_and_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x9A; 16];
        b.write_blob(g, &buf_with(1)).unwrap();
        b.publish_read_index(g, b"index-bytes", b"value-bytes")
            .unwrap();
        let mut dst = vec![0; b"index-bytes".len()];
        assert!(b.read_index_range(g, 0, &mut dst).unwrap());
        assert_eq!(dst, b"index-bytes");
        let mut value = vec![0; b"value-bytes".len()];
        assert!(b.read_value_segment_range(g, 0, &mut value).unwrap());
        assert_eq!(value, b"value-bytes");
        b.delete_read_index(g).unwrap();
        assert!(b.read_index_range(g, 0, &mut dst).unwrap());
        assert_ne!(dst, b"index-bytes");
        assert!(b.read_value_segment_range(g, 0, &mut value).unwrap());
        assert_ne!(value, b"value-bytes");
    }

    #[test]
    fn read_index_publish_overwrites_packed_slot() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x9C; 16];
        b.write_blob(g, &buf_with(1)).unwrap();
        b.publish_read_index(g, b"old", b"old-value").unwrap();
        let mut dst = vec![0; 3];
        assert!(b.read_index_range(g, 0, &mut dst).unwrap());
        assert_eq!(dst, b"old");
        let mut value = vec![0; 9];
        assert!(b.read_value_segment_range(g, 0, &mut value).unwrap());
        assert_eq!(value, b"old-value");
        b.publish_read_index(g, b"new", b"new-value").unwrap();
        assert!(b.read_index_range(g, 0, &mut dst).unwrap());
        assert_eq!(dst, b"new");
        assert!(b.read_value_segment_range(g, 0, &mut value).unwrap());
        assert_eq!(value, b"new-value");
    }

    #[test]
    fn blob_write_removes_stale_read_index() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x9B; 16];
        b.write_blob(g, &buf_with(1)).unwrap();
        b.publish_read_index(g, b"stale", b"stale-value").unwrap();
        b.write_blob(g, &buf_with(7)).unwrap();
        let mut dst = vec![0; b"stale".len()];
        assert!(b.read_index_range(g, 0, &mut dst).unwrap());
        assert_ne!(dst, b"stale");
        let mut value = vec![0; b"stale-value".len()];
        assert!(b.read_value_segment_range(g, 0, &mut value).unwrap());
        assert_ne!(value, b"stale-value");
    }

    #[test]
    fn manifest_round_trip_preserves_all_slots() {
        let dir = tempfile::tempdir().unwrap();
        let guids: Vec<BlobGuid> = (0..16).map(|i| [i as u8; 16]).collect();
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            for (i, g) in guids.iter().enumerate() {
                b.write_blob(*g, &buf_with(i as u8)).unwrap();
            }
            b.flush().unwrap();
        }
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let mut listed = b.list_blobs().unwrap();
        listed.sort();
        let mut expected = guids.clone();
        expected.sort();
        assert_eq!(listed, expected);
        for (i, g) in guids.iter().enumerate() {
            let mut dst = AlignedBlobBuf::zeroed();
            b.read_blob(*g, &mut dst).unwrap();
            assert_eq!(dst.as_slice()[100], i as u8);
        }
    }
}

#[cfg(test)]
mod range_read_test {
    use super::*;
    use crate::store::blob_store::{AlignedBlobBuf, BlobStore};
    use crate::{Tree, TreeConfig};

    // Page-granular reads (the indexed-read I/O optimization) must reconstruct
    // every real blob byte-for-byte vs the whole-frame read — on both the
    // O_DIRECT (Linux) and F_NOCACHE (macOS) paths.
    #[test]
    fn page_reads_reconstruct_each_blob() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut cfg = TreeConfig::new(dir.path());
            cfg.durability = crate::Durability::Wal { sync: false };
            let tree = Tree::open(cfg).unwrap();
            for i in 0..50_000u32 {
                let key = format!("bucket-{:02}/obj-{i:08}", i % 16);
                tree.put(key.as_bytes(), &[(i & 0xff) as u8; 40]).unwrap();
            }
            tree.checkpoint().unwrap();
        }
        let store = FileBlobStore::open(dir.path()).unwrap();
        let guids = store.list_blobs().unwrap();
        assert!(guids.len() > 1, "expected spillover into multiple blobs");

        let frame_pages = (PAGE_SIZE / 4096) as usize;
        let mut whole = AlignedBlobBuf::zeroed();
        let mut paged = AlignedBlobBuf::zeroed();
        for g in &guids {
            store.read_blob(*g, &mut whole).unwrap();
            for p in 0..frame_pages {
                let off = (p * 4096) as u64;
                let dst = &mut paged.as_mut_slice()[p * 4096..(p + 1) * 4096];
                store.read_blob_range(*g, off, dst).unwrap();
            }
            assert_eq!(
                whole.as_slice(),
                paged.as_slice(),
                "page reads must reconstruct blob {:02x?}",
                &g[..4]
            );
            // A multi-page ranged read matches the same window of the frame.
            let mut window = AlignedBlobBuf::zeroed();
            store
                .read_blob_range(*g, 4096 * 5, &mut window.as_mut_slice()[..4096 * 3])
                .unwrap();
            assert_eq!(
                &window.as_slice()[..4096 * 3],
                &whole.as_slice()[4096 * 5..4096 * 8]
            );
        }
    }
}
