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
//!   tmp+rename and truncates the log. Blob replacement uses shadow
//!   paging: write and sync a fresh slot, publish its mapping, then
//!   make the old slot reusable only after the manifest delta is durable.
//! - **Read accelerators** are fixed-slot and rebuildable. `read.idx`
//!   and `value.seg` share the same manifest slot as `blobs.dat`.
//!   A shadow slot's stale accelerator header is cleared before its
//!   mapping is published; checkpoint publication writes value bytes
//!   first and the index header last. Slot reuse is the
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
//!
//! ## Path trust boundary
//!
//! Holt treats the configured path's ancestor directories as trusted
//! deployment input. Open rejects a symlink as the final `data_dir`
//! component and rejects symlinks for store entries; after open, all store
//! I/O stays relative to the held directory descriptor. Every authoritative
//! regular file must be owned by the effective user, mode 0600, and
//! single-link; its held descriptor retains an exclusive inode flock and
//! [`FileBlobStore::validate_object_set`] dynamically verifies name-to-fd
//! identity. Those checks deliberately cover cooperative processes and
//! operational mistakes involving hardlinks or renamed entries.
//!
//! Holt does not walk and independently pin every ancestor component. POSIX
//! also has no portable macOS/Linux operation that atomically validates a
//! name and renames an already-open fd, so an actively hostile same-UID
//! process with write access to the trusted store directory can still race
//! the final `fstatat`/`renameat` boundary. Post-operation checks fail-stop on
//! an observed race; directory permissions remain the enforcement boundary
//! against such an attacker.

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod uring;

use std::collections::HashMap;
use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
use std::os::unix::fs::FileExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::api::errors::{Error, Result};
use crate::api::stats::{StoreStats, VacuumStats};
use crate::layout::{BlobGuid, PAGE_SIZE};

#[cfg(all(target_os = "linux", feature = "io-uring"))]
use super::BlobBufPool;
use super::{AlignedBlobBuf, BlobStore, FileStoreObjectIdentity};

#[cfg(all(target_os = "linux", feature = "io-uring"))]
use self::uring::UringContext;

/// Filename of the packed blob data file inside `data_dir`.
const DATA_FILENAME: &str = "blobs.dat";
/// Advisory lock file inside `data_dir`, flock'd exclusively for
/// the lifetime of an open instance.
const LOCK_FILENAME: &str = "store.lock";
/// Logical WAL file opened by `Tree` / `DB` through this store's pinned
/// directory object.
const WAL_FILENAME: &str = "journal.wal";
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
/// Prefix used for unique, exclusively-created manifest staging files.
/// The historical fixed name is deliberately never truncated or reused.
const MANIFEST_TMP_FILENAME: &str = "manifest.bin.tmp";
static MANIFEST_TMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
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
const FILE_TYPE_MASK: u32 = 0o170_000;
const REGULAR_FILE_TYPE: u32 = 0o100_000;
#[cfg(test)]
const DIRECTORY_FILE_TYPE: u32 = 0o040_000;

/// NVMe-backed, O_DIRECT, single-packed-file blob store.
///
/// Construct via [`FileBlobStore::open`]. Thread-safe; the
/// underlying file handle is shared and `pread`/`pwrite` are
/// atomic at the syscall boundary.
#[derive(Debug)]
pub struct FileBlobStore {
    data_dir: PathBuf,
    /// Pinned directory object used for every file open, rename, and
    /// directory durability sync after startup. The lexical `data_dir` is
    /// retained only for diagnostics.
    directory: Arc<StoreDirectory>,
    object_identity: FileStoreObjectIdentity,
    /// Exclusive advisory lock on `data_dir`, held for the lifetime
    /// of this instance. Two live instances on one directory would
    /// each replay `manifest.log` into the same `next_slot`, assign
    /// the same slot to different blob GUIDs, and append conflicting
    /// set deltas — permanently corrupting the manifest. The kernel
    /// releases the lock when this handle closes, so a crashed
    /// holder never leaves a stale lock behind.
    store_lock: File,
    data_file: File,
    read_index_file: File,
    value_segment_file: File,
    /// Duplicate of the actual WAL descriptor handed to the Journal. Keeping
    /// it here preserves the per-entry flock and supports dynamic name/fd
    /// validation without exposing the descriptor.
    wal_guard: Mutex<Option<File>>,
    manifest: RwLock<Manifest>,
    /// Tracks GUID-to-slot updates that still need durable publication.
    /// Every blob rewrite uses a fresh shadow slot, so both inserts and
    /// replacements leave manifest work until [`Self::flush_locked`].
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
    /// Protects physical slot reuse and vacuum remapping. Readers hold the
    /// shared side from GUID-to-slot resolution through positional I/O.
    /// Reclamation drains those readers before an old slot enters the reusable
    /// pool; vacuum holds the exclusive side while relocating live slots.
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
    /// Slots superseded by a shadow rewrite or removed by `delete_blob`, but
    /// whose replacement/delete is not yet durable in the manifest. They
    /// become reusable only after `flush` persists the corresponding delta;
    /// reusing them earlier could overwrite a slot still referenced by the
    /// last durable manifest.
    pending_free_slots: Vec<u64>,
    /// Bytes currently in `manifest.log`, used to decide when a
    /// full snapshot compaction is worth paying for.
    log_bytes: u64,
    /// Ordered set/delete records not yet durable in
    /// `manifest.log`. The in-memory `slots` map already reflects
    /// them; this queue is the recovery contract.
    pending_log: Vec<ManifestDelta>,
    /// Held descriptor for the currently named snapshot, if one exists.
    snapshot_file: Option<File>,
    /// Held descriptor for the append-only manifest log, if one exists.
    log_file: Option<File>,
    /// Sticky fail-stop state after a publication race or externally linked
    /// retired snapshot is observed. No later compaction is allowed to
    /// accumulate additional retired descriptors.
    snapshot_poisoned: bool,
    /// At most one externally linked retired snapshot is retained until drop
    /// so a compliant opener remains excluded by its inode flock.
    poisoned_snapshot_guard: Option<File>,
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

#[derive(Debug)]
pub(crate) struct StoreDirectory {
    file: File,
}

impl StoreDirectory {
    fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;
        Self::open_existing(path)
    }

    fn open_existing(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        if !file.metadata()?.is_dir() {
            return Err(Error::BlobStoreIo(io::Error::new(
                io::ErrorKind::InvalidInput,
                "file-store path is not a directory",
            )));
        }
        // Lock the directory object as well as store.lock. This makes a
        // same-directory unlink/replacement of store.lock unable to admit a
        // second compliant opener; a rename of the directory itself yields a
        // distinct object identity and all I/O below remains pinned here.
        acquire_flock(&file, path, DIR_LOCK_ACQUIRE_TIMEOUT, "store directory")?;
        pause_after_directory_open();
        Ok(Self { file })
    }

    fn open_at(&self, name: &str, flags: libc::c_int, mode: libc::mode_t) -> Result<File> {
        let name = CString::new(name).map_err(|_| {
            Error::BlobStoreIo(io::Error::new(
                io::ErrorKind::InvalidInput,
                "file-store filename contains NUL",
            ))
        })?;
        let fd = loop {
            // SAFETY: `self.file` is a live directory descriptor, `name` is a
            // NUL-terminated relative filename, and successful ownership of
            // the returned descriptor is transferred exactly once to `File`.
            let fd = unsafe {
                libc::openat(
                    self.file.as_raw_fd(),
                    name.as_ptr(),
                    flags,
                    libc::c_uint::from(mode),
                )
            };
            if fd >= 0 {
                break fd;
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(Error::BlobStoreIo(error));
            }
        };
        // SAFETY: `openat` returned a new owned descriptor above.
        let file = unsafe { File::from_raw_fd(fd) };
        if !file.metadata()?.is_file() {
            return Err(Error::BlobStoreIo(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("file-store entry {name:?} is not a regular file"),
            )));
        }
        Ok(file)
    }

    fn open_existing_at(&self, name: &str, flags: libc::c_int) -> Result<Option<File>> {
        match self.open_at(name, flags | libc::O_CLOEXEC | libc::O_NOFOLLOW, 0) {
            Ok(file) => {
                self.secure_authority_file(name, &file, false)?;
                Ok(Some(file))
            }
            Err(Error::BlobStoreIo(error)) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn open_or_create_at(&self, name: &str, flags: libc::c_int) -> Result<File> {
        match self.open_at(
            name,
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_CREAT | libc::O_EXCL,
            0o600,
        ) {
            Ok(file) => {
                self.secure_authority_file(name, &file, true)?;
                Ok(file)
            }
            Err(Error::BlobStoreIo(error)) if error.kind() == io::ErrorKind::AlreadyExists => {
                self.open_existing_at(name, flags)?.ok_or_else(|| {
                    Error::BlobStoreIo(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("file-store entry {name} disappeared while opening"),
                    ))
                })
            }
            Err(error) => Err(error),
        }
    }

    fn create_new_at(&self, name: &str, flags: libc::c_int) -> Result<File> {
        let file = self.open_at(
            name,
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_CREAT | libc::O_EXCL,
            0o600,
        )?;
        self.secure_authority_file(name, &file, true)?;
        Ok(file)
    }

    fn entry_metadata(&self, name: &str) -> Result<Option<libc::stat>> {
        let name = CString::new(name).map_err(|_| {
            Error::BlobStoreIo(io::Error::new(
                io::ErrorKind::InvalidInput,
                "file-store filename contains NUL",
            ))
        })?;
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        loop {
            // SAFETY: the directory and relative C string are live; `stat`
            // points to writable storage initialized on success.
            let rc = unsafe {
                libc::fstatat(
                    self.file.as_raw_fd(),
                    name.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            };
            if rc == 0 {
                // SAFETY: successful `fstatat` initialized the structure.
                return Ok(Some(unsafe { stat.assume_init() }));
            }
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            if error.kind() == io::ErrorKind::NotFound {
                return Ok(None);
            }
            return Err(Error::BlobStoreIo(error));
        }
    }

    fn rename_at(&self, from: &str, to: &str) -> Result<()> {
        let from = CString::new(from).expect("static filename has no NUL");
        let to = CString::new(to).expect("static filename has no NUL");
        loop {
            // SAFETY: both names are relative C strings and both directory
            // descriptors remain live for the call.
            let rc = unsafe {
                libc::renameat(
                    self.file.as_raw_fd(),
                    from.as_ptr(),
                    self.file.as_raw_fd(),
                    to.as_ptr(),
                )
            };
            if rc == 0 {
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(Error::BlobStoreIo(error));
            }
        }
    }

    fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    fn secure_authority_file(&self, name: &str, file: &File, created: bool) -> Result<()> {
        let metadata = AuthorityFileMetadata::from_file(file)?;
        let expected_uid = unsafe { libc::geteuid() };
        if !metadata.has_safe_shape(expected_uid) {
            return Err(unsafe_authority_file_error(name));
        }

        // Exclude a migrated inode held by another compliant store before
        // changing even its permission bits. A hardlink that still has two
        // names was already rejected by the shape check above.
        acquire_flock(
            file,
            Path::new(name),
            Duration::ZERO,
            "authoritative store entry",
        )?;
        self.validate_authority_file_shape(name, file)?;

        let mut changed = created;
        if metadata.mode & 0o7777 != 0o600 {
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            file.sync_all()?;
            changed = true;
        }
        if created {
            file.sync_all()?;
        }
        if changed {
            self.sync()?;
        }

        self.validate_authority_file(name, file)?;
        Ok(())
    }

    fn validate_authority_file_shape(
        &self,
        name: &str,
        file: &File,
    ) -> Result<AuthorityFileIdentity> {
        let held = AuthorityFileMetadata::from_file(file)?;
        let expected_uid = unsafe { libc::geteuid() };
        if !held.has_safe_shape(expected_uid) {
            return Err(unsafe_authority_file_error(name));
        }
        let linked = self.entry_metadata(name)?.ok_or_else(|| {
            Error::BlobStoreIo(io::Error::new(
                io::ErrorKind::NotFound,
                format!("file-store entry {name} was unlinked while held"),
            ))
        })?;
        let linked = AuthorityFileMetadata::from_stat(linked)?;
        if !linked.has_safe_shape(expected_uid) || held.identity() != linked.identity() {
            return Err(Error::BlobStoreIo(io::Error::other(format!(
                "file-store entry {name} was replaced or hard-linked while held"
            ))));
        }
        Ok(held.identity())
    }

    fn validate_authority_file(&self, name: &str, file: &File) -> Result<AuthorityFileIdentity> {
        // Reasserting the same exclusive flock on the held open-file
        // description is idempotent and guarantees the guard is live when
        // this validation returns.
        acquire_flock(
            file,
            Path::new(name),
            Duration::ZERO,
            "authoritative store entry",
        )?;
        let held = AuthorityFileMetadata::from_file(file)?;
        let expected_uid = unsafe { libc::geteuid() };
        if !held.is_exact(expected_uid) {
            return Err(unsafe_authority_file_error(name));
        }
        let linked_identity = self.validate_authority_file_shape(name, file)?;
        let final_held = AuthorityFileMetadata::from_file(file)?;
        if !final_held.is_exact(expected_uid) || final_held.identity() != linked_identity {
            return Err(unsafe_authority_file_error(name));
        }
        Ok(final_held.identity())
    }

    pub(crate) fn open_wal(&self) -> Result<File> {
        let file = self.open_or_create_at(WAL_FILENAME, libc::O_RDWR | libc::O_APPEND)?;
        // Startup is the only creation point. Syncing the held directory is
        // cheap here and makes a newly-created WAL name durable without ever
        // reopening the configured pathname.
        self.sync()?;
        Ok(file)
    }
}

fn acquire_flock(file: &File, path: &Path, timeout: Duration, object: &str) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        // SAFETY: `file` owns a live descriptor for the duration of the call.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        match err.kind() {
            io::ErrorKind::Interrupted => {}
            io::ErrorKind::WouldBlock => {
                notify_lock_contention();
                if Instant::now() >= deadline {
                    return Err(Error::BlobStoreIo(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        format!(
                            "blob store {object} at {} is locked by another live instance \
                             (waited {timeout:?}); a second opener would corrupt \
                             the manifest",
                            path.display()
                        ),
                    )));
                }
                thread::sleep(DIR_LOCK_RETRY_INTERVAL);
            }
            _ => return Err(Error::BlobStoreIo(err)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AuthorityFileIdentity {
    device: u64,
    inode: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObjectSetSnapshot {
    root: FileStoreObjectIdentity,
    data: AuthorityFileIdentity,
    read_index: AuthorityFileIdentity,
    value_segment: AuthorityFileIdentity,
    manifest: ManifestFileIdentities,
    wal: Option<AuthorityFileIdentity>,
}

#[derive(Debug, Clone, Copy)]
struct AuthorityFileMetadata {
    mode: u32,
    uid: u32,
    links: u64,
    identity: AuthorityFileIdentity,
}

impl AuthorityFileMetadata {
    fn from_file(file: &File) -> Result<Self> {
        let metadata = file.metadata()?;
        Ok(Self {
            mode: metadata.mode(),
            uid: metadata.uid(),
            links: metadata.nlink(),
            identity: AuthorityFileIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            },
        })
    }

    fn from_stat(metadata: libc::stat) -> Result<Self> {
        Ok(Self {
            mode: stat_mode(&metadata),
            uid: metadata.st_uid,
            links: stat_links(&metadata),
            identity: AuthorityFileIdentity {
                device: stat_device(&metadata)?,
                inode: metadata.st_ino,
            },
        })
    }

    fn has_safe_shape(self, expected_uid: u32) -> bool {
        self.mode & FILE_TYPE_MASK == REGULAR_FILE_TYPE
            && self.uid == expected_uid
            && self.links == 1
    }

    fn is_exact(self, expected_uid: u32) -> bool {
        self.has_safe_shape(expected_uid) && self.mode & 0o7777 == 0o600
    }

    fn identity(self) -> AuthorityFileIdentity {
        self.identity
    }
}

#[cfg(target_os = "linux")]
fn stat_device(metadata: &libc::stat) -> Result<u64> {
    Ok(metadata.st_dev)
}

#[cfg(target_os = "macos")]
fn stat_device(metadata: &libc::stat) -> Result<u64> {
    u64::try_from(metadata.st_dev)
        .map_err(|_| Error::BlobStoreIo(io::Error::other("negative store-entry device id")))
}

#[cfg(target_os = "linux")]
fn stat_mode(metadata: &libc::stat) -> u32 {
    metadata.st_mode
}

#[cfg(target_os = "macos")]
fn stat_mode(metadata: &libc::stat) -> u32 {
    u32::from(metadata.st_mode)
}

#[cfg(target_os = "linux")]
#[allow(clippy::useless_conversion)] // `nlink_t` is u32/aarch64 and u64/x86_64.
fn stat_links(metadata: &libc::stat) -> u64 {
    u64::from(metadata.st_nlink)
}

#[cfg(target_os = "macos")]
fn stat_links(metadata: &libc::stat) -> u64 {
    u64::from(metadata.st_nlink)
}

fn unsafe_authority_file_error(name: &str) -> Error {
    Error::BlobStoreIo(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "file-store entry {name} must be a single-link, mode-0600 regular file owned by the effective user"
        ),
    ))
}

#[derive(Debug, Clone, Copy)]
struct LockFileMetadata {
    mode: u32,
    uid: u32,
    links: u64,
    size: u64,
}

impl LockFileMetadata {
    fn from_file(file: &File) -> Result<Self> {
        let metadata = file.metadata()?;
        Ok(Self {
            mode: metadata.mode(),
            uid: metadata.uid(),
            links: metadata.nlink(),
            size: metadata.len(),
        })
    }
}

fn validate_lock_file(file: &File) -> Result<()> {
    let metadata = LockFileMetadata::from_file(file)?;
    let expected_uid = unsafe { libc::geteuid() };
    validate_lock_metadata(metadata, expected_uid)
}

fn validate_lock_metadata(metadata: LockFileMetadata, expected_uid: u32) -> Result<()> {
    if metadata.mode & FILE_TYPE_MASK != REGULAR_FILE_TYPE
        || metadata.uid != expected_uid
        || metadata.links != 1
        || metadata.mode & 0o7777 != 0o600
        || metadata.size != 0
    {
        return Err(unsafe_lock_file_error());
    }
    Ok(())
}

fn unsafe_lock_file_error() -> Error {
    Error::BlobStoreIo(io::Error::new(
        io::ErrorKind::PermissionDenied,
        "store.lock must be a zero-byte, single-link, mode-0600 regular file owned by the effective user",
    ))
}

fn lock_metadata_can_be_hardened(metadata: LockFileMetadata, expected_uid: u32) -> bool {
    metadata.mode & FILE_TYPE_MASK == REGULAR_FILE_TYPE
        && metadata.uid == expected_uid
        && metadata.links == 1
        && metadata.size == 0
}

fn acquire_store_lock(directory: &StoreDirectory, path: &Path) -> Result<File> {
    let (lock, created) = match directory.open_at(
        LOCK_FILENAME,
        libc::O_RDWR | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_CREAT | libc::O_EXCL,
        0o600,
    ) {
        Ok(lock) => (lock, true),
        Err(Error::BlobStoreIo(error)) if error.kind() == io::ErrorKind::AlreadyExists => {
            let lock = directory.open_at(
                LOCK_FILENAME,
                libc::O_RDWR | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                0,
            )?;
            (lock, false)
        }
        Err(error) => return Err(error),
    };
    finish_store_lock_open(directory, path, &lock, created)?;
    Ok(lock)
}

fn acquire_existing_store_lock_unmodified(directory: &StoreDirectory, path: &Path) -> Result<File> {
    let lock = directory.open_at(
        LOCK_FILENAME,
        libc::O_RDWR | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        0,
    )?;
    acquire_flock(&lock, path, DIR_LOCK_ACQUIRE_TIMEOUT, LOCK_FILENAME)?;
    verify_named_lock(directory, &lock)?;
    Ok(lock)
}

fn finish_store_lock_open(
    directory: &StoreDirectory,
    path: &Path,
    lock: &File,
    created: bool,
) -> Result<()> {
    let expected_uid = unsafe { libc::geteuid() };
    let metadata = LockFileMetadata::from_file(lock)?;
    if !lock_metadata_can_be_hardened(metadata, expected_uid) {
        return Err(unsafe_lock_file_error());
    }
    acquire_flock(lock, path, DIR_LOCK_ACQUIRE_TIMEOUT, LOCK_FILENAME)?;
    verify_named_lock(directory, lock)?;
    let metadata = LockFileMetadata::from_file(lock)?;
    if !lock_metadata_can_be_hardened(metadata, expected_uid) {
        return Err(unsafe_lock_file_error());
    }

    let changed = created || metadata.mode & 0o7777 != 0o600;
    if metadata.mode & 0o7777 != 0o600 {
        // `openat`'s mode is filtered through umask, and older Holt versions
        // commonly left mode 0644. Tighten only after the safe-shape checks,
        // name/fd comparison, and inode flock all succeed.
        lock.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    if changed {
        lock.sync_all()?;
        directory.sync()?;
    }
    validate_lock_file(lock)?;
    acquire_flock(lock, path, Duration::ZERO, LOCK_FILENAME)?;
    validate_lock_file(lock)?;
    verify_named_lock(directory, lock)?;
    Ok(())
}

fn verify_named_lock(directory: &StoreDirectory, lock: &File) -> Result<()> {
    let held = lock.metadata()?;
    let linked = directory.entry_metadata(LOCK_FILENAME)?.ok_or_else(|| {
        Error::BlobStoreIo(io::Error::new(
            io::ErrorKind::NotFound,
            "store.lock was unlinked while opening",
        ))
    })?;
    let linked_device = stat_device(&linked)?;
    let linked_inode = linked.st_ino;
    if held.dev() != linked_device || held.ino() != linked_inode {
        return Err(Error::BlobStoreIo(io::Error::other(
            "store.lock was replaced while opening",
        )));
    }
    Ok(())
}

fn object_identity(directory: &StoreDirectory, lock: &File) -> Result<FileStoreObjectIdentity> {
    let directory = directory.file.metadata()?;
    let lock = lock.metadata()?;
    Ok(FileStoreObjectIdentity {
        directory_device: directory.dev(),
        directory_inode: directory.ino(),
        lock_device: lock.dev(),
        lock_inode: lock.ino(),
    })
}

fn open_store_root(
    data_dir: &Path,
    expected: Option<FileStoreObjectIdentity>,
) -> Result<(Arc<StoreDirectory>, File, FileStoreObjectIdentity)> {
    if let Some(expected) = expected {
        // Expected-aware reopen is deliberately non-creating and
        // non-repairing until both held kernel objects compare exactly.
        let directory = Arc::new(StoreDirectory::open_existing(data_dir)?);
        let store_lock = acquire_existing_store_lock_unmodified(&directory, data_dir)?;
        let actual = object_identity(&directory, &store_lock)?;
        if actual != expected {
            return Err(Error::FileStoreIdentityMismatch { expected, actual });
        }
        finish_store_lock_open(&directory, data_dir, &store_lock, false)?;
        Ok((directory, store_lock, actual))
    } else {
        let directory = Arc::new(StoreDirectory::open(data_dir)?);
        let store_lock = acquire_store_lock(&directory, data_dir)?;
        let actual = object_identity(&directory, &store_lock)?;
        Ok((directory, store_lock, actual))
    }
}

#[cfg(test)]
std::thread_local! {
    static LOCK_CONTENTION_NOTIFIER: std::cell::RefCell<Option<crossbeam_channel::Sender<()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn notify_lock_contention() {
    LOCK_CONTENTION_NOTIFIER.with(|slot| {
        if let Some(notifier) = slot.borrow().as_ref() {
            let _ = notifier.try_send(());
        }
    });
}

#[cfg(not(test))]
fn notify_lock_contention() {}

#[cfg(test)]
impl FileBlobStore {
    pub(crate) fn set_lock_contention_notifier_for_current_thread(
        notifier: crossbeam_channel::Sender<()>,
    ) {
        LOCK_CONTENTION_NOTIFIER.with(|slot| *slot.borrow_mut() = Some(notifier));
    }
}

#[cfg(test)]
struct OpenDirectoryBarrier {
    entered: std::sync::Barrier,
    release: std::sync::Barrier,
}

#[cfg(test)]
impl OpenDirectoryBarrier {
    fn new() -> Self {
        Self {
            entered: std::sync::Barrier::new(2),
            release: std::sync::Barrier::new(2),
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static OPEN_DIRECTORY_BARRIER: std::cell::RefCell<Option<Arc<OpenDirectoryBarrier>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn set_open_directory_barrier(barrier: Arc<OpenDirectoryBarrier>) {
    OPEN_DIRECTORY_BARRIER.with(|slot| *slot.borrow_mut() = Some(barrier));
}

#[cfg(test)]
fn pause_after_directory_open() {
    let barrier = OPEN_DIRECTORY_BARRIER.with(|slot| slot.borrow_mut().take());
    if let Some(barrier) = barrier {
        barrier.entered.wait();
        barrier.release.wait();
    }
}

#[cfg(not(test))]
fn pause_after_directory_open() {}

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
        Self::open_with_expected_object_identity(data_dir, None)
    }

    /// Open only if the held directory and `store.lock` match `expected`.
    ///
    /// With `Some`, Holt neither creates the directory/lock nor repairs their
    /// metadata until the two held `(device, inode)` pairs match exactly.
    /// Authoritative data, manifest, and WAL entries remain untouched on a
    /// mismatch. `None` permits first-time creation.
    pub fn open_with_expected_object_identity<P: Into<PathBuf>>(
        data_dir: P,
        expected: Option<FileStoreObjectIdentity>,
    ) -> Result<Self> {
        Self::open_with_registered_buffer_capacity(data_dir, REGISTERED_BUFFER_MAX_SLOTS, expected)
    }

    /// Open with a registered-buffer hot-pool hint derived from the
    /// caller's buffer-manager capacity. The actual pool is bounded
    /// by [`REGISTERED_BUFFER_MAX_SLOTS`] so large caches do not pin
    /// proportional memory at open/reopen time.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub(crate) fn open_with_buffer_pool_hint<P: Into<PathBuf>>(
        data_dir: P,
        buffer_pool_size: usize,
        expected: Option<FileStoreObjectIdentity>,
    ) -> Result<Self> {
        let slots = registered_buffer_slots(buffer_pool_size);
        Self::open_with_registered_buffer_capacity(data_dir, slots, expected)
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn open_with_registered_buffer_capacity<P: Into<PathBuf>>(
        data_dir: P,
        registered_buffer_slots: usize,
        expected: Option<FileStoreObjectIdentity>,
    ) -> Result<Self> {
        let data_dir = data_dir.into();
        let (directory, store_lock, object_identity) = open_store_root(&data_dir, expected)?;

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
        let data_file = directory.open_or_create_at(DATA_FILENAME, data_flags | libc::O_RDWR)?;
        let read_index_file =
            directory.open_or_create_at(READ_INDEX_FILENAME, index_flags | libc::O_RDWR)?;
        let value_segment_file =
            directory.open_or_create_at(VALUE_SEGMENT_FILENAME, index_flags | libc::O_RDWR)?;
        directory.sync()?;

        // macOS doesn't have O_DIRECT; F_NOCACHE on the fd is the
        // closest equivalent (tells the VFS not to populate the
        // unified buffer cache for this fd's I/O).
        #[cfg(target_os = "macos")]
        unsafe {
            let _ = libc::fcntl(data_file.as_raw_fd(), libc::F_NOCACHE, 1);
        }

        let manifest = Manifest::load_or_create(&directory)?;
        let wal_guard = directory.open_existing_at(WAL_FILENAME, libc::O_RDWR | libc::O_APPEND)?;
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
            directory,
            object_identity,
            store_lock,
            data_file,
            read_index_file,
            value_segment_file,
            wal_guard: Mutex::new(wal_guard),
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
        expected: Option<FileStoreObjectIdentity>,
    ) -> Result<Self> {
        let data_dir = data_dir.into();
        let (directory, store_lock, object_identity) = open_store_root(&data_dir, expected)?;

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
        let data_file = directory.open_or_create_at(DATA_FILENAME, data_flags | libc::O_RDWR)?;
        let read_index_file =
            directory.open_or_create_at(READ_INDEX_FILENAME, index_flags | libc::O_RDWR)?;
        let value_segment_file =
            directory.open_or_create_at(VALUE_SEGMENT_FILENAME, index_flags | libc::O_RDWR)?;
        directory.sync()?;

        #[cfg(target_os = "macos")]
        unsafe {
            let _ = libc::fcntl(data_file.as_raw_fd(), libc::F_NOCACHE, 1);
        }

        let manifest = Manifest::load_or_create(&directory)?;
        let wal_guard = directory.open_existing_at(WAL_FILENAME, libc::O_RDWR | libc::O_APPEND)?;
        let file_slots = slots_for_len(data_file.metadata()?.len());
        let preallocated_slots = file_slots.max(manifest.next_slot);

        Ok(Self {
            data_dir,
            directory,
            object_identity,
            store_lock,
            data_file,
            read_index_file,
            value_segment_file,
            wal_guard: Mutex::new(wal_guard),
            manifest: RwLock::new(manifest),
            manifest_dirty: AtomicBool::new(false),
            data_write_epoch: AtomicU64::new(0),
            data_sync_epoch: AtomicU64::new(0),
            data_io_lock: Mutex::new(()),
            slot_io_lock: RwLock::new(()),
            preallocated_slots: AtomicU64::new(preallocated_slots),
        })
    }

    /// Configured pathname used to open this store.
    ///
    /// The path may be renamed or replaced after open. It is retained for
    /// diagnostics only; all I/O uses the held directory object and callers
    /// that need fencing material should use [`Self::object_identity`].
    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Return the filesystem objects actually held by this store instance.
    #[must_use]
    pub fn object_identity(&self) -> FileStoreObjectIdentity {
        self.object_identity
    }

    pub(crate) fn open_wal_file(&self) -> Result<File> {
        let mut guard = self.wal_guard.lock().unwrap();
        if let Some(file) = guard.as_ref() {
            self.directory.validate_authority_file(WAL_FILENAME, file)?;
            return Ok(file.try_clone()?);
        }
        let file = self.directory.open_wal()?;
        self.directory
            .validate_authority_file(WAL_FILENAME, &file)?;
        *guard = Some(file.try_clone()?);
        Ok(file)
    }

    /// Dynamically validate the entire authoritative file-store object set.
    ///
    /// Each pass reasserts the directory, lock, and per-entry flocks and
    /// compares every held descriptor with its current name using `fstatat`.
    /// Two identical live passes are required, so this is a cut-point check,
    /// not a cached identity lookup. The returned root identity is suitable
    /// for the caller's fencing record only after this method succeeds.
    pub fn validate_object_set(&self) -> Result<FileStoreObjectIdentity> {
        let first = self.validate_object_set_once()?;
        let second = self.validate_object_set_once()?;
        if first != second {
            return Err(Error::BlobStoreIo(io::Error::other(
                "file-store object set changed during validation",
            )));
        }
        Ok(first.root)
    }

    fn validate_object_set_once(&self) -> Result<ObjectSetSnapshot> {
        acquire_flock(
            &self.directory.file,
            &self.data_dir,
            Duration::ZERO,
            "store directory",
        )?;
        validate_lock_file(&self.store_lock)?;
        acquire_flock(
            &self.store_lock,
            &self.data_dir,
            Duration::ZERO,
            LOCK_FILENAME,
        )?;
        validate_lock_file(&self.store_lock)?;
        verify_named_lock(&self.directory, &self.store_lock)?;
        validate_lock_file(&self.store_lock)?;
        let root = object_identity(&self.directory, &self.store_lock)?;
        if root != self.object_identity {
            return Err(Error::BlobStoreIo(io::Error::other(
                "held file-store root identity changed",
            )));
        }

        let data = self
            .directory
            .validate_authority_file(DATA_FILENAME, &self.data_file)?;
        let read_index = self
            .directory
            .validate_authority_file(READ_INDEX_FILENAME, &self.read_index_file)?;
        let value_segment = self
            .directory
            .validate_authority_file(VALUE_SEGMENT_FILENAME, &self.value_segment_file)?;
        let manifest = self
            .manifest
            .read()
            .unwrap()
            .validate_files(&self.directory)?;
        let wal_guard = self.wal_guard.lock().unwrap();
        let wal = if let Some(file) = wal_guard.as_ref() {
            Some(self.directory.validate_authority_file(WAL_FILENAME, file)?)
        } else {
            ensure_authority_entry_absent(&self.directory, WAL_FILENAME)?;
            None
        };

        Ok(ObjectSetSnapshot {
            root,
            data,
            read_index,
            value_segment,
            manifest,
            wal,
        })
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

    fn clear_read_accelerator_slots(&self, guid: BlobGuid) -> Result<()> {
        let Some(entry) = self.manifest.read().unwrap().entries.get(&guid).copied() else {
            return Ok(());
        };
        self.clear_read_accelerator_slot(entry.slot)
    }

    fn clear_read_accelerator_slot(&self, slot: u64) -> Result<()> {
        self.clear_read_index_slot(slot)?;
        self.clear_value_segment_slot(slot)
    }

    fn clear_read_index_slot(&self, slot: u64) -> Result<()> {
        let offset = slot.saturating_mul(u64::from(PAGE_SIZE));
        if offset >= file_len(&self.read_index_file) {
            return Ok(());
        }
        let zeros = [0u8; READ_INDEX_IO_ALIGN];
        self.write_read_index_aligned(offset, &zeros)
    }

    fn clear_value_segment_slot(&self, slot: u64) -> Result<()> {
        let offset = slot.saturating_mul(u64::from(PAGE_SIZE));
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

    fn reserve_write_entries(
        &self,
        guids: impl IntoIterator<Item = BlobGuid>,
    ) -> Vec<ReservedManifestEntry> {
        let mut m = self.manifest.write().unwrap();
        guids
            .into_iter()
            .map(|guid| ReservedManifestEntry {
                guid,
                slot: m.allocate_slot(),
            })
            .collect()
    }

    /// Roll back slots reserved by a write that failed before it could
    /// publish them.
    ///
    /// A reserved slot was never in any manifest — durable or in-memory —
    /// so no reader can resolve to it and returning it to the allocator
    /// needs no fence. `pending_free_slots` is the opposite case and must
    /// not be touched here: see [`Manifest::trim_trailing_reusable_slots`].
    fn release_reserved_slots(&self, slots: impl IntoIterator<Item = u64>) {
        let mut slots: Vec<_> = slots.into_iter().collect();
        if slots.is_empty() {
            return;
        }
        let mut m = self.manifest.write().unwrap();
        m.reusable_slots.append_slots(&mut slots);
        m.trim_trailing_reusable_slots();
    }

    fn publish_blob_writes(&self, writes: &[PreparedBlobWrite<'_>]) {
        if writes.is_empty() {
            return;
        }
        let mut m = self.manifest.write().unwrap();
        for write in writes {
            m.publish_write_entry(write.guid, write.slot);
        }
        self.manifest_dirty.store(true, Ordering::Release);
    }

    fn clear_reserved_read_accelerators(&self, writes: &[PreparedBlobWrite<'_>]) {
        for write in writes {
            let _ = self.clear_read_accelerator_slot(write.slot);
        }
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
        let entries = self.reserve_write_entries(writes.iter().map(|(guid, _)| *guid));
        if let Some(required_slots) = entries
            .iter()
            .map(|entry| entry.slot.saturating_add(1))
            .max()
        {
            if let Err(error) = self.ensure_data_capacity(required_slots) {
                self.release_reserved_slots(entries.iter().map(|entry| entry.slot));
                return Err(error);
            }
        }
        let mut io = Vec::with_capacity(writes.len());
        for ((guid, src), entry) in writes.iter().zip(entries) {
            debug_assert_eq!(*guid, entry.guid);
            io.push(PreparedBlobWrite {
                guid: entry.guid,
                slot: entry.slot,
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

        let mut publish_free_slots = false;
        if self.manifest_dirty.swap(false, Ordering::AcqRel) {
            let mut m = self.manifest.write().unwrap();
            if let Err(e) = m.persist_pending_deltas(&self.directory) {
                self.manifest_dirty.store(true, Ordering::Release);
                return Err(e);
            }
            m.pending_log.clear();
            publish_free_slots = !m.pending_free_slots.is_empty();
        }
        if publish_free_slots {
            // A reader may still be completing I/O through the previous
            // mapping. Drain it after the remap is durable and before the old
            // slot enters the allocator; new readers already resolve the new
            // mapping and therefore cannot prolong this fence.
            let _slot = self.enter_slot_write();
            self.manifest.write().unwrap().publish_pending_free_slots();
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
    fn file_store_object_identity(&self) -> Option<FileStoreObjectIdentity> {
        Some(self.object_identity)
    }

    fn validate_file_store_object_set(&self) -> Result<Option<FileStoreObjectIdentity>> {
        self.validate_object_set().map(Some)
    }

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
        let writes = [(guid, src)];
        let prepared = self.prepare_blob_writes(&writes)?;
        self.clear_reserved_read_accelerators(&prepared);
        self.mark_data_write_started();
        if let Err(error) = self.pwrite_at(prepared[0].offset, src) {
            self.release_reserved_slots(prepared.iter().map(|write| write.slot));
            return Err(error);
        }
        self.publish_blob_writes(&prepared);
        Ok(())
    }

    fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
        let prepared = self.prepare_blob_writes(writes)?;
        if prepared.is_empty() {
            return Ok(());
        }
        self.clear_reserved_read_accelerators(&prepared);
        self.mark_data_write_started();
        if let Err(error) = self.pwrite_many_at(&prepared) {
            self.release_reserved_slots(prepared.iter().map(|write| write.slot));
            return Err(error);
        }
        self.publish_blob_writes(&prepared);
        Ok(())
    }

    fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }
        let _io = self.data_io_lock.lock().unwrap();

        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        {
            let prepared = self.prepare_blob_writes(writes)?;
            self.clear_reserved_read_accelerators(&prepared);
            let epoch = self.mark_data_write_started();
            if let Err(error) = self.pwrite_many_and_sync_at(&prepared) {
                self.release_reserved_slots(prepared.iter().map(|write| write.slot));
                return Err(error);
            }
            self.mark_data_synced(epoch);
            self.publish_blob_writes(&prepared);
            Ok(())
        }

        #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
        {
            let prepared = self.prepare_blob_writes(writes)?;
            self.clear_reserved_read_accelerators(&prepared);
            self.mark_data_write_started();
            if let Err(error) = self.pwrite_many_at(&prepared) {
                self.release_reserved_slots(prepared.iter().map(|write| write.slot));
                return Err(error);
            }
            self.publish_blob_writes(&prepared);
            Ok(())
        }
    }

    fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        let _io = self.data_io_lock.lock().unwrap();
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
                m.persist_snapshot(&self.directory)?;
                m.truncate_log(&self.directory)?;
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
    guid: BlobGuid,
    slot: u64,
    offset: u64,
    src: &'a AlignedBlobBuf,
}

#[derive(Clone, Copy)]
struct ReservedManifestEntry {
    guid: BlobGuid,
    slot: u64,
}

#[derive(Debug, Clone, Copy)]
struct SlotMove {
    guid: BlobGuid,
    from_slot: u64,
    to_slot: u64,
}

impl Manifest {
    fn load_or_create(directory: &StoreDirectory) -> Result<Self> {
        let mut snapshot_file = directory.open_existing_at(MANIFEST_FILENAME, libc::O_RDWR)?;
        let (mut entries, mut next_slot) = match snapshot_file.as_mut() {
            Some(file) => {
                let parsed = Self::parse_snapshot(file)?;
                directory.validate_authority_file(MANIFEST_FILENAME, file)?;
                parsed
            }
            None => (HashMap::new(), 0),
        };

        let mut log_file =
            directory.open_existing_at(MANIFEST_LOG_FILENAME, libc::O_RDWR | libc::O_APPEND)?;
        let replay = match log_file.as_mut() {
            Some(file) => {
                let replay = Self::replay_log(file, &mut entries, &mut next_slot)?;
                directory.validate_authority_file(MANIFEST_LOG_FILENAME, file)?;
                if replay.valid_bytes < replay.file_bytes {
                    // The inode and directory entry have already been
                    // validated and exclusively flocked. Revalidate on both
                    // sides of the only recovery-time mutation.
                    directory.validate_authority_file(MANIFEST_LOG_FILENAME, file)?;
                    file.set_len(replay.valid_bytes)?;
                    file.sync_all()?;
                    directory.validate_authority_file(MANIFEST_LOG_FILENAME, file)?;
                }
                replay
            }
            None => ManifestLogReplay {
                file_bytes: 0,
                valid_bytes: 0,
            },
        };
        let used_slots: Vec<_> = entries.values().map(|entry| entry.slot).collect();
        let reusable_slots = ReusableSlots::reconstruct(next_slot, &used_slots)?;

        Ok(Self {
            entries,
            next_slot,
            reusable_slots,
            pending_free_slots: Vec::new(),
            log_bytes: replay.valid_bytes,
            pending_log: Vec::new(),
            snapshot_file,
            log_file,
            snapshot_poisoned: false,
            poisoned_snapshot_guard: None,
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

    fn publish_write_entry(&mut self, guid: BlobGuid, slot: u64) {
        if let Some(old) = self.entries.insert(guid, ManifestEntry { slot }) {
            debug_assert_ne!(old.slot, slot, "shadow write must use a fresh slot");
            self.pending_free_slots.push(old.slot);
        }
        self.pending_log.push(ManifestDelta::Set { guid, slot });
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

    /// Lower the high-water mark over slots that are *already* reusable.
    ///
    /// Unlike [`Self::trim_trailing_free_slots`] this does **not** drain
    /// `pending_free_slots`. Those slots are still referenced by the last
    /// durable manifest and may still be under an in-flight reader, so only
    /// `flush_locked`'s durability + reader-drain fences may publish them.
    /// Rollback paths, which run on I/O failure and pass neither fence, must
    /// use this variant.
    fn trim_trailing_reusable_slots(&mut self) -> u64 {
        self.reusable_slots.trim_trailing(&mut self.next_slot)
    }

    fn trim_trailing_free_slots(&mut self) -> u64 {
        self.publish_pending_free_slots();
        self.trim_trailing_reusable_slots()
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

    fn persist_pending_deltas(&mut self, directory: &StoreDirectory) -> Result<()> {
        if self.pending_log.is_empty() {
            return Ok(());
        }
        self.ensure_snapshot_healthy()?;

        let mut buf = Vec::with_capacity(self.pending_log.len() * 40);
        for delta in &self.pending_log {
            encode_manifest_delta(*delta, &mut buf)?;
        }
        if self.log_file.is_none() {
            self.log_file = Some(
                directory
                    .open_or_create_at(MANIFEST_LOG_FILENAME, libc::O_RDWR | libc::O_APPEND)?,
            );
        }
        let f = self.log_file.as_mut().expect("manifest log initialized");
        directory.validate_authority_file(MANIFEST_LOG_FILENAME, f)?;
        f.write_all(&buf)?;
        f.sync_data()?;
        directory.validate_authority_file(MANIFEST_LOG_FILENAME, f)?;

        self.log_bytes = self.log_bytes.saturating_add(buf.len() as u64);
        if self.should_compact_log() {
            self.persist_snapshot(directory)?;
            self.truncate_log(directory)?;
        }
        Ok(())
    }

    fn should_compact_log(&self) -> bool {
        let snapshot_bytes = 24u64.saturating_add((self.entries.len() as u64).saturating_mul(32));
        self.log_bytes >= MANIFEST_LOG_MIN_COMPACT_BYTES
            && self.log_bytes >= snapshot_bytes.saturating_mul(MANIFEST_LOG_COMPACT_RATIO)
    }

    fn persist_snapshot(&mut self, directory: &StoreDirectory) -> Result<()> {
        self.ensure_snapshot_healthy()?;
        if let Some(snapshot) = self.snapshot_file.as_ref() {
            directory.validate_authority_file(MANIFEST_FILENAME, snapshot)?;
        } else {
            ensure_authority_entry_absent(directory, MANIFEST_FILENAME)?;
        }

        let temp_name = format!(
            "{MANIFEST_TMP_FILENAME}.{}.{}",
            std::process::id(),
            MANIFEST_TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        let mut f = directory.create_new_at(&temp_name, libc::O_RDWR)?;

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
        if let Err(error) = directory.validate_authority_file(&temp_name, &f) {
            self.poison_snapshot(Some(f));
            return Err(error);
        }

        let old_snapshot = self.snapshot_file.take();
        if let Some(old) = old_snapshot.as_ref() {
            if let Err(error) = directory.validate_authority_file(MANIFEST_FILENAME, old) {
                self.snapshot_file = old_snapshot;
                self.poison_snapshot(None);
                return Err(error);
            }
        }
        if let Err(error) = directory.rename_at(&temp_name, MANIFEST_FILENAME) {
            self.snapshot_file = old_snapshot;
            self.poison_snapshot(Some(f));
            return Err(error);
        }
        self.snapshot_file = Some(f);
        let mut old_snapshot = old_snapshot;

        // Sync the parent directory so the rename itself is durable
        // (required by POSIX; ext4/xfs honour it). After rename, every error
        // is sticky: the new held fd remains installed and at most one old fd
        // with a surviving link is retained until drop.
        if let Err(error) = directory.sync() {
            let guard = old_snapshot
                .take()
                .and_then(|old| match AuthorityFileMetadata::from_file(&old) {
                    Ok(metadata) if metadata.links == 0 => None,
                    _ => Some(old),
                });
            self.poison_snapshot(guard);
            return Err(error);
        }

        let new_snapshot = self
            .snapshot_file
            .as_ref()
            .expect("renamed manifest snapshot remains held");
        if let Err(error) = directory.validate_authority_file(MANIFEST_FILENAME, new_snapshot) {
            let guard = old_snapshot
                .take()
                .and_then(|old| match AuthorityFileMetadata::from_file(&old) {
                    Ok(metadata) if metadata.links == 0 => None,
                    _ => Some(old),
                });
            self.poison_snapshot(guard);
            return Err(error);
        }
        if let Some(old) = old_snapshot {
            match AuthorityFileMetadata::from_file(&old) {
                Ok(metadata) if metadata.links == 0 => {}
                Ok(_) => {
                    self.poison_snapshot(Some(old));
                    return Err(Error::BlobStoreIo(io::Error::other(
                        "retired manifest snapshot remains externally hard-linked",
                    )));
                }
                Err(error) => {
                    self.poison_snapshot(Some(old));
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    fn truncate_log(&mut self, directory: &StoreDirectory) -> Result<()> {
        self.ensure_snapshot_healthy()?;
        if let Some(f) = self.log_file.as_ref() {
            directory.validate_authority_file(MANIFEST_LOG_FILENAME, f)?;
            f.set_len(0)?;
            f.sync_data()?;
            directory.validate_authority_file(MANIFEST_LOG_FILENAME, f)?;
        } else {
            ensure_authority_entry_absent(directory, MANIFEST_LOG_FILENAME)?;
        }
        self.log_bytes = 0;
        Ok(())
    }

    fn replay_log(
        f: &mut File,
        entries: &mut HashMap<BlobGuid, ManifestEntry>,
        next_slot: &mut u64,
    ) -> Result<ManifestLogReplay> {
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

    fn ensure_snapshot_healthy(&self) -> Result<()> {
        if self.snapshot_poisoned {
            return Err(Error::BlobStoreIo(io::Error::other(
                "manifest snapshot publication is poisoned",
            )));
        }
        Ok(())
    }

    fn poison_snapshot(&mut self, guard: Option<File>) {
        self.snapshot_poisoned = true;
        if self.poisoned_snapshot_guard.is_none() {
            self.poisoned_snapshot_guard = guard;
        }
    }

    fn validate_files(&self, directory: &StoreDirectory) -> Result<ManifestFileIdentities> {
        self.ensure_snapshot_healthy()?;
        let snapshot = if let Some(file) = self.snapshot_file.as_ref() {
            Some(directory.validate_authority_file(MANIFEST_FILENAME, file)?)
        } else {
            ensure_authority_entry_absent(directory, MANIFEST_FILENAME)?;
            None
        };
        let log = if let Some(file) = self.log_file.as_ref() {
            Some(directory.validate_authority_file(MANIFEST_LOG_FILENAME, file)?)
        } else {
            ensure_authority_entry_absent(directory, MANIFEST_LOG_FILENAME)?;
            None
        };
        Ok(ManifestFileIdentities { snapshot, log })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ManifestFileIdentities {
    snapshot: Option<AuthorityFileIdentity>,
    log: Option<AuthorityFileIdentity>,
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

fn ensure_authority_entry_absent(directory: &StoreDirectory, name: &str) -> Result<()> {
    if directory.entry_metadata(name)?.is_some() {
        return Err(Error::BlobStoreIo(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("unexpected file-store entry {name} appeared while held"),
        )));
    }
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

    #[derive(Debug, PartialEq, Eq)]
    struct TestFileState {
        bytes: Vec<u8>,
        len: u64,
        mode: u32,
        device: u64,
        inode: u64,
    }

    fn file_state(path: &Path) -> TestFileState {
        let metadata = std::fs::metadata(path).unwrap();
        TestFileState {
            bytes: std::fs::read(path).unwrap(),
            len: metadata.len(),
            mode: metadata.mode() & 0o7777,
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }

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
        let second_directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(dir.path())
            .unwrap();
        let second = acquire_flock(
            &second_directory,
            dir.path(),
            Duration::from_millis(50),
            "store directory",
        );
        match second {
            Err(Error::BlobStoreIo(e)) => {
                assert_eq!(e.kind(), io::ErrorKind::WouldBlock, "unexpected error: {e}");
            }
            Err(e) => panic!("unexpected error variant: {e}"),
            Ok(()) => panic!("second open acquired the lock while the store is live"),
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
    fn object_identity_comes_from_held_directory_and_lock_files() {
        let dir = tempfile::tempdir().unwrap();
        let Some(store) = try_open(dir.path()) else {
            return;
        };
        let identity = store.object_identity();
        let directory = File::open(dir.path()).unwrap().metadata().unwrap();
        let lock = File::open(dir.path().join(LOCK_FILENAME))
            .unwrap()
            .metadata()
            .unwrap();

        assert_eq!(identity.directory_device, directory.dev());
        assert_eq!(identity.directory_inode, directory.ino());
        assert_eq!(identity.lock_device, lock.dev());
        assert_eq!(identity.lock_inode, lock.ino());
        assert_eq!(
            BlobStore::file_store_object_identity(&store),
            Some(identity)
        );
    }

    #[test]
    fn authority_metadata_requires_exact_owner_mode_type_and_single_link() {
        let uid = unsafe { libc::geteuid() };
        let valid = AuthorityFileMetadata {
            mode: REGULAR_FILE_TYPE | 0o600,
            uid,
            links: 1,
            identity: AuthorityFileIdentity {
                device: 1,
                inode: 2,
            },
        };
        assert!(valid.is_exact(uid));

        let mut invalid = valid;
        invalid.mode = REGULAR_FILE_TYPE | 0o640;
        assert!(!invalid.is_exact(uid));
        invalid = valid;
        invalid.mode = DIRECTORY_FILE_TYPE | 0o600;
        assert!(!invalid.is_exact(uid));
        invalid = valid;
        invalid.uid = uid.wrapping_add(1);
        assert!(!invalid.is_exact(uid));
        invalid = valid;
        invalid.links = 2;
        assert!(!invalid.is_exact(uid));
    }

    #[test]
    fn hardlinked_data_is_rejected_without_mutating_source() {
        let source_dir = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let Some(source) = try_open(source_dir.path()) else {
            return;
        };
        source.write_blob([0xD1; 16], &buf_with(0x41)).unwrap();
        source.flush().unwrap();
        let source_path = source_dir.path().join(DATA_FILENAME);
        let target_path = target_dir.path().join(DATA_FILENAME);
        std::fs::hard_link(&source_path, &target_path).unwrap();
        let before = file_state(&source_path);

        assert!(FileBlobStore::open(target_dir.path()).is_err());
        assert_eq!(file_state(&source_path), before);
        let source_lock = std::fs::metadata(source_dir.path().join(LOCK_FILENAME)).unwrap();
        let target_lock = std::fs::metadata(target_dir.path().join(LOCK_FILENAME)).unwrap();
        assert_ne!(source_lock.ino(), target_lock.ino());
    }

    #[test]
    fn hardlinked_manifest_log_is_rejected_before_torn_tail_truncation() {
        let source_dir = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let Some(source) = try_open(source_dir.path()) else {
            return;
        };
        source.write_blob([0xD2; 16], &buf_with(0x42)).unwrap();
        source.flush().unwrap();
        let source_path = source_dir.path().join(MANIFEST_LOG_FILENAME);
        OpenOptions::new()
            .append(true)
            .open(&source_path)
            .unwrap()
            .write_all(b"torn-tail")
            .unwrap();
        std::fs::hard_link(&source_path, target_dir.path().join(MANIFEST_LOG_FILENAME)).unwrap();
        let before = file_state(&source_path);

        assert!(FileBlobStore::open(target_dir.path()).is_err());
        assert_eq!(file_state(&source_path), before);
    }

    #[test]
    fn hardlinked_manifest_snapshot_is_rejected_without_mutating_source() {
        let source_dir = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let Some(source) = try_open(source_dir.path()) else {
            return;
        };
        source.write_blob([0xD3; 16], &buf_with(0x43)).unwrap();
        source.flush().unwrap();
        source
            .manifest
            .write()
            .unwrap()
            .persist_snapshot(&source.directory)
            .unwrap();
        let source_path = source_dir.path().join(MANIFEST_FILENAME);
        std::fs::hard_link(&source_path, target_dir.path().join(MANIFEST_FILENAME)).unwrap();
        let before = file_state(&source_path);

        assert!(FileBlobStore::open(target_dir.path()).is_err());
        assert_eq!(file_state(&source_path), before);
    }

    #[test]
    fn hardlinked_wal_is_rejected_without_mutating_source() {
        let source_dir = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let Some(source) = try_open(source_dir.path()) else {
            return;
        };
        let mut wal = source.open_wal_file().unwrap();
        wal.write_all(b"sentinel-wal-bytes").unwrap();
        wal.sync_all().unwrap();
        let source_path = source_dir.path().join(WAL_FILENAME);
        std::fs::hard_link(&source_path, target_dir.path().join(WAL_FILENAME)).unwrap();
        let before = file_state(&source_path);

        assert!(FileBlobStore::open(target_dir.path()).is_err());
        assert_eq!(file_state(&source_path), before);
    }

    #[test]
    fn dynamic_object_set_validation_rejects_hardlink_and_entry_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let Some(store) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(
            store.validate_object_set().unwrap(),
            store.object_identity()
        );

        let alias = dir.path().join("blobs.alias");
        std::fs::hard_link(dir.path().join(DATA_FILENAME), &alias).unwrap();
        assert!(store.validate_object_set().is_err());
        std::fs::remove_file(alias).unwrap();
        assert!(store.validate_object_set().is_ok());

        let held = dir.path().join("blobs.held");
        std::fs::rename(dir.path().join(DATA_FILENAME), &held).unwrap();
        OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(dir.path().join(DATA_FILENAME))
            .unwrap();
        assert!(store.validate_object_set().is_err());
    }

    #[test]
    fn expected_identity_rejects_other_store_before_touching_it_then_reopens_original() {
        let a_dir = tempfile::tempdir().unwrap();
        let b_dir = tempfile::tempdir().unwrap();
        let a_identity = {
            let Some(store) = try_open(a_dir.path()) else {
                return;
            };
            store.object_identity()
        };
        {
            let Some(store) = try_open(b_dir.path()) else {
                return;
            };
            store.write_blob([0xD4; 16], &buf_with(0x44)).unwrap();
            store.flush().unwrap();
            store
                .manifest
                .write()
                .unwrap()
                .persist_snapshot(&store.directory)
                .unwrap();
            drop(store.open_wal_file().unwrap());
        }

        let authority_names = [
            LOCK_FILENAME,
            DATA_FILENAME,
            READ_INDEX_FILENAME,
            VALUE_SEGMENT_FILENAME,
            MANIFEST_FILENAME,
            MANIFEST_LOG_FILENAME,
            WAL_FILENAME,
        ];
        let mut before = Vec::new();
        for name in authority_names {
            let path = b_dir.path().join(name);
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
            before.push((path, file_state(&b_dir.path().join(name))));
        }

        let mut wrong =
            crate::TreeConfig::new(b_dir.path()).with_expected_file_store_identity(a_identity);
        wrong.checkpoint.enabled = false;
        match crate::DB::open(wrong).unwrap_err() {
            Error::FileStoreIdentityMismatch { expected, actual } => {
                assert_eq!(expected, a_identity);
                assert_ne!(actual, a_identity);
            }
            error => panic!("unexpected expected-identity error: {error}"),
        }
        for (path, state) in before {
            assert_eq!(
                file_state(&path),
                state,
                "expected mismatch touched {path:?}"
            );
        }

        let mut original =
            crate::TreeConfig::new(a_dir.path()).with_expected_file_store_identity(a_identity);
        original.checkpoint.enabled = false;
        let db = crate::DB::open(original).unwrap();
        assert_eq!(db.file_store_object_identity(), Some(a_identity));
        assert_eq!(
            db.validate_file_store_object_set().unwrap(),
            Some(a_identity)
        );
    }

    #[test]
    fn expected_identity_does_not_create_a_missing_store_root() {
        let source_dir = tempfile::tempdir().unwrap();
        let expected = {
            let Some(store) = try_open(source_dir.path()) else {
                return;
            };
            store.object_identity()
        };
        let parent = tempfile::tempdir().unwrap();
        let missing = parent.path().join("missing-store");
        assert!(
            FileBlobStore::open_with_expected_object_identity(&missing, Some(expected)).is_err()
        );
        assert!(!missing.exists());
    }

    #[test]
    fn per_entry_flock_blocks_second_directory_after_name_migration() {
        let source_dir = tempfile::tempdir().unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let Some(source) = try_open(source_dir.path()) else {
            return;
        };
        let Some(target_seed) = try_open(target_dir.path()) else {
            return;
        };
        drop(target_seed);

        std::fs::remove_file(target_dir.path().join(DATA_FILENAME)).unwrap();
        std::fs::hard_link(
            source_dir.path().join(DATA_FILENAME),
            target_dir.path().join(DATA_FILENAME),
        )
        .unwrap();
        std::fs::remove_file(source_dir.path().join(DATA_FILENAME)).unwrap();
        OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(source_dir.path().join(DATA_FILENAME))
            .unwrap();
        let migrated = target_dir.path().join(DATA_FILENAME);
        std::fs::set_permissions(&migrated, std::fs::Permissions::from_mode(0o640)).unwrap();
        let before = file_state(&migrated);

        // The migrated inode has nlink==1 in the other directory, but the
        // first store's held descriptor still owns its exclusive inode flock.
        assert!(FileBlobStore::open(target_dir.path()).is_err());
        assert_eq!(file_state(&migrated), before);
        assert!(source.validate_object_set().is_err());
    }

    #[test]
    fn fixed_manifest_temp_hardlink_is_never_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let external = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(external.path(), b"do-not-truncate").unwrap();
        std::fs::set_permissions(external.path(), std::fs::Permissions::from_mode(0o640)).unwrap();
        std::fs::hard_link(external.path(), dir.path().join(MANIFEST_TMP_FILENAME)).unwrap();
        let before = file_state(external.path());

        let Some(store) = try_open(dir.path()) else {
            return;
        };
        store
            .manifest
            .write()
            .unwrap()
            .persist_snapshot(&store.directory)
            .unwrap();

        assert_eq!(file_state(external.path()), before);
        assert_eq!(file_state(&dir.path().join(MANIFEST_TMP_FILENAME)), before);
    }

    #[test]
    fn directory_and_lock_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let target = parent.path().join("target");
        std::fs::create_dir(&target).unwrap();
        let directory_link = parent.path().join("directory-link");
        symlink(&target, &directory_link).unwrap();
        assert!(FileBlobStore::open(&directory_link).is_err());

        let lock_target = parent.path().join("lock-target");
        File::create(&lock_target).unwrap();
        symlink(&lock_target, target.join(LOCK_FILENAME)).unwrap();
        assert!(FileBlobStore::open(&target).is_err());
    }

    #[test]
    fn authority_symlink_is_rejected_and_safe_legacy_mode_is_hardened() {
        use std::os::unix::fs::symlink;

        let linked_dir = tempfile::tempdir().unwrap();
        let external = tempfile::NamedTempFile::new().unwrap();
        symlink(external.path(), linked_dir.path().join(DATA_FILENAME)).unwrap();
        assert!(FileBlobStore::open(linked_dir.path()).is_err());

        let legacy_dir = tempfile::tempdir().unwrap();
        let data = legacy_dir.path().join(DATA_FILENAME);
        OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o640)
            .open(&data)
            .unwrap();
        let Some(_store) = try_open(legacy_dir.path()) else {
            return;
        };
        assert_eq!(std::fs::metadata(data).unwrap().mode() & 0o7777, 0o600);
    }

    #[test]
    fn lock_metadata_rejects_mode_owner_links_and_size() {
        let uid = unsafe { libc::geteuid() };
        let valid = LockFileMetadata {
            mode: REGULAR_FILE_TYPE | 0o600,
            uid,
            links: 1,
            size: 0,
        };
        assert!(validate_lock_metadata(valid, uid).is_ok());

        let mut invalid = valid;
        invalid.mode = REGULAR_FILE_TYPE | 0o640;
        assert!(validate_lock_metadata(invalid, uid).is_err());
        invalid = valid;
        invalid.uid = uid.wrapping_add(1);
        assert!(validate_lock_metadata(invalid, uid).is_err());
        invalid = valid;
        invalid.links = 2;
        assert!(validate_lock_metadata(invalid, uid).is_err());
        invalid = valid;
        invalid.size = 1;
        assert!(validate_lock_metadata(invalid, uid).is_err());
    }

    #[test]
    fn legacy_owned_lock_mode_is_hardened_before_flock() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join(LOCK_FILENAME);
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o640)
            .open(&lock_path)
            .unwrap();

        let Some(_store) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(std::fs::metadata(lock_path).unwrap().mode() & 0o7777, 0o600);
    }

    #[test]
    fn unsafe_on_disk_lock_metadata_is_rejected_before_store_files_open() {
        for violation in ["links", "size"] {
            let dir = tempfile::tempdir().unwrap();
            let lock_path = dir.path().join(LOCK_FILENAME);
            let mut lock = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&lock_path)
                .unwrap();
            match violation {
                "links" => {
                    std::fs::hard_link(&lock_path, dir.path().join("lock-alias")).unwrap();
                }
                "size" => {
                    lock.write_all(b"x").unwrap();
                }
                _ => unreachable!(),
            }
            drop(lock);
            assert!(
                FileBlobStore::open(dir.path()).is_err(),
                "accepted {violation} violation"
            );
            assert!(!dir.path().join(DATA_FILENAME).exists());
        }
    }

    #[test]
    fn replacing_unlinked_lock_cannot_bypass_held_directory_lock() {
        let dir = tempfile::tempdir().unwrap();
        let Some(store) = try_open(dir.path()) else {
            return;
        };
        let original = store.object_identity();
        std::fs::remove_file(dir.path().join(LOCK_FILENAME)).unwrap();
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(dir.path().join(LOCK_FILENAME))
            .unwrap();
        let replacement = File::open(dir.path().join(LOCK_FILENAME))
            .unwrap()
            .metadata()
            .unwrap();
        assert_ne!(original.lock_inode, replacement.ino());
        let second_directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(dir.path())
            .unwrap();
        let second = acquire_flock(
            &second_directory,
            dir.path(),
            Duration::from_millis(50),
            "store directory",
        );
        assert!(matches!(
            second,
            Err(Error::BlobStoreIo(error)) if error.kind() == io::ErrorKind::WouldBlock
        ));

        drop(store);
        let Some(reopened) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(reopened.object_identity().lock_inode, replacement.ino());
    }

    #[test]
    fn db_open_stays_on_pinned_directory_across_path_swap() {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("store");
        let held_path = parent.path().join("held-store");
        std::fs::create_dir(&path).unwrap();
        let barrier = Arc::new(OpenDirectoryBarrier::new());
        let opener_barrier = Arc::clone(&barrier);
        let opener_path = path.clone();
        let opener = thread::spawn(move || {
            set_open_directory_barrier(opener_barrier);
            let mut cfg = crate::TreeConfig::new(opener_path);
            cfg.checkpoint.enabled = false;
            crate::DB::open(cfg)
        });

        barrier.entered.wait();
        std::fs::rename(&path, &held_path).unwrap();
        std::fs::create_dir(&path).unwrap();
        barrier.release.wait();
        let db = match opener.join().unwrap() {
            Ok(db) => db,
            Err(Error::BlobStoreIo(error)) if error.raw_os_error() == Some(libc::EINVAL) => {
                eprintln!("skipping: O_DIRECT not supported on this fs");
                return;
            }
            Err(error) => panic!("unexpected DB open error: {error}"),
        };

        let held = File::open(&held_path).unwrap().metadata().unwrap();
        let identity = db.file_store_object_identity().unwrap();
        assert_eq!(identity.directory_device, held.dev());
        assert_eq!(identity.directory_inode, held.ino());
        for filename in [
            LOCK_FILENAME,
            DATA_FILENAME,
            READ_INDEX_FILENAME,
            VALUE_SEGMENT_FILENAME,
            MANIFEST_LOG_FILENAME,
            WAL_FILENAME,
        ] {
            assert!(held_path.join(filename).exists(), "missing {filename}");
            assert!(
                !path.join(filename).exists(),
                "mixed {filename} into replacement"
            );
        }
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
    fn write_replaces_existing_through_shadow_slot() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x33; 16];
        b.write_blob(g, &buf_with(1)).unwrap();
        let first_slot = b.entry_of(g).unwrap().slot;
        b.write_blob(g, &buf_with(2)).unwrap();
        let second_slot = b.entry_of(g).unwrap().slot;
        assert_ne!(first_slot, second_slot);
        b.flush().unwrap();
        assert_eq!(b.len(), 1);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2);
    }

    #[test]
    fn manifest_publishes_fresh_slot_on_rewrite_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        let g: BlobGuid = [0x34; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g, &buf_with(1)).unwrap();
            assert_eq!(b.entry_of(g).unwrap().slot, 0);
            b.write_blob(g, &buf_with(2)).unwrap();
            assert_eq!(b.entry_of(g).unwrap().slot, 1, "rewrite uses a shadow slot");
            b.flush().unwrap();
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(b.entry_of(g).unwrap().slot, 1);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 2, "last write persists across reopen");
    }

    #[test]
    fn unflushed_shadow_rewrite_recovers_previous_frame() {
        let dir = tempfile::tempdir().unwrap();
        let g: BlobGuid = [0x36; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g, &buf_with(1)).unwrap();
            b.flush().unwrap();
            assert_eq!(b.entry_of(g).unwrap().slot, 0);

            b.write_blob(g, &buf_with(2)).unwrap();
            assert_eq!(b.entry_of(g).unwrap().slot, 1);
            // Simulate process loss after the new frame write but before the
            // GUID-to-slot manifest update reaches durable storage.
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(b.entry_of(g).unwrap().slot, 0);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);
    }

    #[test]
    fn corrupt_unpublished_shadow_cannot_replace_durable_frame() {
        let dir = tempfile::tempdir().unwrap();
        let g: BlobGuid = [0x3C; 16];
        {
            let Some(b) = try_open(dir.path()) else {
                return;
            };
            b.write_blob(g, &buf_with(1)).unwrap();
            b.flush().unwrap();

            b.write_blob(g, &buf_with(2)).unwrap();
            let shadow = b.entry_of(g).unwrap().slot;
            b.pwrite_at(shadow * u64::from(PAGE_SIZE), &AlignedBlobBuf::zeroed())
                .unwrap();
            b.sync_data_file().unwrap();
        }

        let Some(b) = try_open(dir.path()) else {
            return;
        };
        assert_eq!(b.entry_of(g).unwrap().slot, 0);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 1);
    }

    #[test]
    fn repeated_shadow_rewrites_reuse_two_slots() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let g: BlobGuid = [0x37; 16];
        b.write_blob(g, &buf_with(0)).unwrap();
        b.flush().unwrap();

        for value in 1..=64 {
            b.write_blob(g, &buf_with(value)).unwrap();
            b.flush().unwrap();
        }

        let stats = b.store_stats();
        assert_eq!(stats.live_slots, 1);
        assert_eq!(stats.next_slot, 2, "one live blob needs one shadow slot");
        assert_eq!(stats.reusable_slots, 1);
        let mut dst = AlignedBlobBuf::zeroed();
        b.read_blob(g, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[100], 64);
    }

    #[test]
    fn replaced_slot_is_reused_only_after_manifest_flush() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let first: BlobGuid = [0x38; 16];
        let second: BlobGuid = [0x39; 16];
        let third: BlobGuid = [0x3A; 16];

        b.write_blob(first, &buf_with(1)).unwrap();
        b.flush().unwrap();
        assert_eq!(b.entry_of(first).unwrap().slot, 0);

        b.write_blob(first, &buf_with(2)).unwrap();
        assert_eq!(b.entry_of(first).unwrap().slot, 1);
        b.write_blob(second, &buf_with(3)).unwrap();
        assert_eq!(
            b.entry_of(second).unwrap().slot,
            2,
            "the durable old slot must not be reused before the remap is durable",
        );

        b.flush().unwrap();
        b.write_blob(third, &buf_with(4)).unwrap();
        assert_eq!(
            b.entry_of(third).unwrap().slot,
            0,
            "the old slot becomes reusable after manifest fsync",
        );
    }

    /// The error-path sibling of
    /// [`replaced_slot_is_reused_only_after_manifest_flush`].
    ///
    /// Rolling back a failed write returns only the slots that write
    /// reserved. It must not also publish `pending_free_slots`: those are
    /// still referenced by the last durable manifest, and the rollback path
    /// passes neither the manifest-durability fence nor the reader drain.
    /// Publishing them there would hand a durably-referenced slot to the next
    /// write, so a crash before the next flush would resolve the old guid to
    /// another blob's bytes.
    ///
    /// This drives `release_reserved_slots` directly — the same call the
    /// `pwrite`/capacity error arms make — because the failure it rolls back
    /// is a genuine device error inside the store, below any injectable
    /// `BlobStore` boundary.
    #[test]
    fn rolling_back_reserved_slots_keeps_undurable_frees_pending() {
        let dir = tempfile::tempdir().unwrap();
        let Some(b) = try_open(dir.path()) else {
            return;
        };
        let live: BlobGuid = [0x3C; 16];
        let next: BlobGuid = [0x3D; 16];

        // Durable: live -> slot 0.
        b.write_blob(live, &buf_with(1)).unwrap();
        b.flush().unwrap();
        assert_eq!(b.entry_of(live).unwrap().slot, 0);

        // Shadow rewrite. Slot 0 is superseded in memory, but the durable
        // manifest still maps live -> 0, so it may not be reused yet.
        b.write_blob(live, &buf_with(2)).unwrap();
        assert_eq!(b.entry_of(live).unwrap().slot, 1);
        assert_eq!(b.store_stats().pending_free_slots, 1);

        // A following write reserves a fresh slot and then fails before it
        // can publish; only that reservation may go back to the allocator.
        let reserved = b.reserve_write_entries([next]);
        assert_eq!(reserved[0].slot, 2, "a rewrite must reserve a fresh slot");
        b.release_reserved_slots(reserved.iter().map(|entry| entry.slot));

        assert_eq!(
            b.store_stats().pending_free_slots,
            1,
            "rollback must not publish slots the durable manifest still references",
        );

        b.write_blob(next, &buf_with(3)).unwrap();
        assert_eq!(
            b.entry_of(next).unwrap().slot,
            2,
            "the rolled-back reservation is reusable, the superseded slot is not",
        );
    }

    #[test]
    fn manifest_flush_drains_old_slot_readers_before_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let Some(store) = try_open(dir.path()) else {
            return;
        };
        let store = std::sync::Arc::new(store);
        let guid: BlobGuid = [0x3B; 16];
        store.write_blob(guid, &buf_with(1)).unwrap();
        store.flush().unwrap();

        let (reader_entered_tx, reader_entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_reader_tx, release_reader_rx) = std::sync::mpsc::sync_channel(1);
        let reader_store = std::sync::Arc::clone(&store);
        let reader = std::thread::spawn(move || {
            let _slot = reader_store.enter_slot_read();
            assert_eq!(reader_store.entry_of(guid).unwrap().slot, 0);
            reader_entered_tx.send(()).unwrap();
            release_reader_rx.recv().unwrap();
        });
        reader_entered_rx.recv().unwrap();

        // Publishing the complete shadow frame does not wait for an old-slot
        // reader because the old slot is still protected from reuse.
        store.write_blob(guid, &buf_with(2)).unwrap();
        assert_eq!(store.entry_of(guid).unwrap().slot, 1);

        let (flush_done_tx, flush_done_rx) = std::sync::mpsc::sync_channel(1);
        let flush_store = std::sync::Arc::clone(&store);
        let flush = std::thread::spawn(move || {
            flush_store.flush().unwrap();
            flush_done_tx.send(()).unwrap();
        });
        assert!(
            flush_done_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "flush must not release an old slot while a reader still holds it",
        );

        release_reader_tx.send(()).unwrap();
        reader.join().unwrap();
        flush_done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        flush.join().unwrap();
        assert_eq!(store.store_stats().reusable_slots, 1);
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
                .write()
                .unwrap()
                .persist_snapshot(&b.directory)
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
