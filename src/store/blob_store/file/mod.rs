//! `FileBlobStore` — file-backed durable blob store.
//!
//! Available on every Unix platform. The Linux build opens the
//! packed data file with `O_DIRECT` so the kernel does not cache
//! pages (the buffer manager *is* the cache); other Unixes drop the
//! flag (macOS additionally sets `F_NOCACHE` via `fcntl` for an
//! equivalent effect).
//!
//! Layout on disk:
//!
//! ```text
//!   <data_dir>/
//!     blobs.dat      — single packed file, blob N lives at byte
//!                      offset N * PAGE_SIZE
//!     manifest.bin   — small file mapping BlobGuid → slot number
//!                      plus `next_slot`; rewritten only when the
//!                      manifest delta log is compacted
//!     manifest.log   — append-only set/delete deltas replayed on
//!                      open; free slots are rebuilt from holes
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
//! - **O_DIRECT / F_NOCACHE** bypasses the page cache: ours *is*
//!   the cache. The buffer manager owns dirty pages and flushes
//!   through the store; the kernel must not silently cache
//!   anything. The packed data file is preallocated in coarse
//!   chunks (`posix_fallocate` on Linux, `F_PREALLOCATE` on
//!   macOS) so checkpoint bursts do not repeatedly pay file-growth
//!   allocation latency.
//! - **4 KB-aligned I/O** (every offset is a multiple of `PAGE_SIZE`
//!   = 512 KB, every buffer is [`AlignedBlobBuf`] = 4 KB aligned) so
//!   `O_DIRECT` accepts every submission without `EINVAL`.
//! - **Manifest** holds the GUID → slot mapping. Checkpoint rounds
//!   append small set/delete deltas to `manifest.log` and fsync it
//!   instead of rewriting the whole map. When the log grows well
//!   past the snapshot size it compacts into `manifest.bin` via
//!   tmp+rename and truncates the log.
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
use std::sync::{Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::api::errors::{Error, Result};
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
/// A v1 file is refused on load — the on-disk format is pre-1.0 and
/// not migrated.
const MANIFEST_VERSION: u16 = 2;
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
    /// guid → slot index (offset on disk = slot * u64::from(PAGE_SIZE)).
    slots: HashMap<BlobGuid, u64>,
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
        let manifest_path = data_dir.join(MANIFEST_FILENAME);
        let manifest_log_path = data_dir.join(MANIFEST_LOG_FILENAME);

        let custom_flags = {
            #[cfg(target_os = "linux")]
            {
                libc::O_DIRECT | libc::O_CLOEXEC
            }
            #[cfg(not(target_os = "linux"))]
            {
                libc::O_CLOEXEC
            }
        };
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(custom_flags)
            .open(&data_path)?;

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
            manifest: RwLock::new(manifest),
            manifest_dirty: AtomicBool::new(false),
            data_write_epoch: AtomicU64::new(0),
            data_sync_epoch: AtomicU64::new(0),
            data_io_lock: Mutex::new(()),
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
        let manifest_path = data_dir.join(MANIFEST_FILENAME);
        let manifest_log_path = data_dir.join(MANIFEST_LOG_FILENAME);

        let custom_flags = {
            #[cfg(target_os = "linux")]
            {
                libc::O_DIRECT | libc::O_CLOEXEC
            }
            #[cfg(not(target_os = "linux"))]
            {
                libc::O_CLOEXEC
            }
        };
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(custom_flags)
            .open(&data_path)?;

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
            manifest: RwLock::new(manifest),
            manifest_dirty: AtomicBool::new(false),
            data_write_epoch: AtomicU64::new(0),
            data_sync_epoch: AtomicU64::new(0),
            data_io_lock: Mutex::new(()),
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
        self.manifest.read().unwrap().slots.len()
    }

    /// True if the manifest is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.manifest.read().unwrap().slots.is_empty()
    }

    fn offset_of(&self, guid: BlobGuid) -> Result<u64> {
        let m = self.manifest.read().unwrap();
        let slot = m.slots.get(&guid).copied().ok_or_else(|| {
            Error::BlobStoreIo(io::Error::new(
                io::ErrorKind::NotFound,
                format!("blob {:02x?} not in manifest", &guid[..4]),
            ))
        })?;
        Ok(slot * u64::from(PAGE_SIZE))
    }

    fn assign_slot(&self, guid: BlobGuid) -> u64 {
        let mut m = self.manifest.write().unwrap();
        if let Some(&s) = m.slots.get(&guid) {
            return s;
        }
        let s = m.allocate_slot();
        m.slots.insert(guid, s);
        m.pending_log.push(ManifestDelta::Set { guid, slot: s });
        self.manifest_dirty.store(true, Ordering::Release);
        s
    }

    fn assign_slots(&self, guids: impl IntoIterator<Item = BlobGuid>) -> Vec<u64> {
        let mut m = self.manifest.write().unwrap();
        let mut out = Vec::new();
        let mut dirty = false;
        for guid in guids {
            if let Some(&s) = m.slots.get(&guid) {
                out.push(s);
                continue;
            }
            let s = m.allocate_slot();
            m.slots.insert(guid, s);
            m.pending_log.push(ManifestDelta::Set { guid, slot: s });
            dirty = true;
            out.push(s);
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
    ) -> Result<Vec<(u64, &'a AlignedBlobBuf)>> {
        if writes.is_empty() {
            return Ok(Vec::new());
        }
        let slots = self.assign_slots(writes.iter().map(|(guid, _)| *guid));
        if let Some(required_slots) = slots.iter().map(|slot| slot.saturating_add(1)).max() {
            self.ensure_data_capacity(required_slots)?;
        }
        let mut io = Vec::with_capacity(writes.len());
        for ((_, src), slot) in writes.iter().zip(slots) {
            io.push((slot * u64::from(PAGE_SIZE), *src));
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
    fn pwrite_many_at(&self, writes: &[(u64, &AlignedBlobBuf)]) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        ring.pwrite_many_at(writes)?;
        Ok(())
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn pwrite_many_and_sync_at(&self, writes: &[(u64, &AlignedBlobBuf)]) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        ring.pwrite_many_and_sync_at(writes)?;
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
    fn pwrite_many_at(&self, writes: &[(u64, &AlignedBlobBuf)]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }

        let mut ordered: Vec<_> = writes
            .iter()
            .enumerate()
            .map(|(order, (offset, src))| OrderedWrite {
                offset: *offset,
                src: src.as_slice(),
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
        let offset = self.offset_of(guid)?;
        self.pread_at(offset, dst)?;
        Ok(())
    }

    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
        let slot = self.assign_slot(guid);
        let offset = slot * u64::from(PAGE_SIZE);
        self.ensure_data_capacity(slot.saturating_add(1))?;
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
            let epoch = self.mark_data_write_started();
            self.pwrite_many_and_sync_at(&io)?;
            self.mark_data_synced(epoch);
            Ok(())
        }

        #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
        {
            let io = self.prepare_blob_writes(writes)?;
            self.mark_data_write_started();
            self.pwrite_many_at(&io)?;
            Ok(())
        }
    }

    fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        let mut m = self.manifest.write().unwrap();
        if let Some(slot) = m.slots.remove(&guid) {
            m.pending_free_slots.push(slot);
            m.pending_log.push(ManifestDelta::Delete { guid });
            self.manifest_dirty.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
        let m = self.manifest.read().unwrap();
        Ok(m.slots.keys().copied().collect())
    }

    fn flush(&self) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
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

    fn needs_flush(&self) -> bool {
        self.data_needs_sync().is_some() || self.manifest_dirty.load(Ordering::Acquire)
    }
}

impl Manifest {
    fn load_or_create(path: &Path, log_path: &Path) -> Result<Self> {
        let (mut slots, mut next_slot) = match File::open(path) {
            Ok(mut f) => Self::parse_snapshot(&mut f)?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => (HashMap::new(), 0),
            Err(e) => return Err(Error::BlobStoreIo(e)),
        };

        let replay = Self::replay_log(log_path, &mut slots, &mut next_slot)?;
        if replay.valid_bytes < replay.file_bytes {
            truncate_manifest_log(log_path, replay.valid_bytes)?;
        }
        let used_slots: Vec<_> = slots.values().copied().collect();
        let reusable_slots = ReusableSlots::reconstruct(next_slot, &used_slots)?;

        Ok(Self {
            slots,
            next_slot,
            reusable_slots,
            pending_free_slots: Vec::new(),
            path: path.to_path_buf(),
            log_path: log_path.to_path_buf(),
            log_bytes: replay.valid_bytes,
            pending_log: Vec::new(),
        })
    }

    fn parse_snapshot(f: &mut File) -> Result<(HashMap<BlobGuid, u64>, u64)> {
        // Header: magic 8 + version 2 + count 4 + reserved 2 + next_slot 8 = 24 B.
        let mut hdr = [0u8; 24];
        f.read_exact(&mut hdr)?;
        if hdr[..8] != MANIFEST_MAGIC {
            return Err(Error::node_corrupt("FileBlobStore::Manifest::magic"));
        }
        let version = u16::from_le_bytes([hdr[8], hdr[9]]);
        if version != MANIFEST_VERSION {
            return Err(Error::node_corrupt(
                "FileBlobStore::Manifest::version (v1 manifests are not migrated)",
            ));
        }
        let count = u32::from_le_bytes([hdr[10], hdr[11], hdr[12], hdr[13]]) as usize;
        // hdr[14..16] reserved (zero).
        let next_slot = u64::from_le_bytes(hdr[16..24].try_into().unwrap());

        let mut slots = HashMap::with_capacity(count);
        let mut used_slots = Vec::with_capacity(count);
        let mut entry = [0u8; 24];
        for _ in 0..count {
            f.read_exact(&mut entry)?;
            let mut g: BlobGuid = [0u8; 16];
            g.copy_from_slice(&entry[..16]);
            let s = u64::from_le_bytes(entry[16..24].try_into().unwrap());
            if slots.insert(g, s).is_some() {
                return Err(Error::node_corrupt(
                    "FileBlobStore::Manifest::duplicate guid",
                ));
            }
            used_slots.push(s);
        }
        ReusableSlots::reconstruct(next_slot, &used_slots)?;
        Ok((slots, next_slot))
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
        let snapshot_bytes = 24u64.saturating_add((self.slots.len() as u64).saturating_mul(24));
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
        let count = u32::try_from(self.slots.len()).map_err(|_| {
            Error::BlobStoreIo(io::Error::other("manifest slot count exceeds u32::MAX"))
        })?;
        hdr[10..14].copy_from_slice(&count.to_le_bytes());
        // Bytes 14..16 reserved (zero).
        f.write_all(&hdr)?;
        f.write_all(&self.next_slot.to_le_bytes())?;

        for (g, &s) in &self.slots {
            f.write_all(g)?;
            f.write_all(&s.to_le_bytes())?;
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
        slots: &mut HashMap<BlobGuid, u64>,
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
                    slots.insert(guid, slot);
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
                    slots.remove(&guid);
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
    fn reusable_slots_reconstruct_sparse_manifest_as_ranges() {
        let mut slots = ReusableSlots::reconstruct(1_000_000, &[0, 999_999]).unwrap();

        assert_eq!(slots.single_count(), 0);
        assert_eq!(slots.range_count(), 1);
        assert_eq!(slots.pop(), Some(1));
        assert_eq!(slots.pop(), Some(2));
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
