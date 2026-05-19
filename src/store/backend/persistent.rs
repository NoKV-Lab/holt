//! `PersistentBackend` — file-backed durable blob store.
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
//!                      (full rewrite via tmp+rename on flush)
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
//!   through the backend; the kernel must not silently cache
//!   anything.
//! - **4 KB-aligned I/O** (every offset is a multiple of `PAGE_SIZE`
//!   = 512 KB, every buffer is [`AlignedBlobBuf`] = 4 KB aligned) so
//!   `O_DIRECT` accepts every submission without `EINVAL`.
//! - **Manifest** holds the GUID → slot mapping. Crash-safe via
//!   atomic rename: writes go to `manifest.bin.tmp` then `rename(2)`.
//!
//! ## I/O backend
//!
//! Two code paths share the same `PersistentBackend` struct:
//!
//! - **`pread`/`pwrite`** (default): every Unix target, every build
//!   configuration. Uses `std::os::unix::fs::FileExt`.
//! - **`io_uring`** (`cfg(target_os = "linux")` + `feature =
//!   "io-uring"`): submits one SQE per read/write to a dedicated
//!   ring owned by the backend. Eliminates the per-syscall entry/
//!   exit cost on Linux.
//!
//! Both paths share the same on-disk layout and the same
//! `Backend::flush` semantics (`sync_data` + manifest persist).
//! Switching between them is an internal performance toggle; no
//! caller-visible behaviour changes.

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod uring;

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "macos")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
use std::sync::Mutex;

use crate::api::errors::{Error, Result};
use crate::layout::{BlobGuid, PAGE_SIZE};

use super::{AlignedBlobBuf, Backend};

#[cfg(all(target_os = "linux", feature = "io-uring"))]
use self::uring::UringContext;

/// Filename of the packed blob data file inside `data_dir`.
const DATA_FILENAME: &str = "blobs.dat";
/// Filename of the manifest inside `data_dir`.
const MANIFEST_FILENAME: &str = "manifest.bin";
/// Filename used as the rename staging target for the manifest.
const MANIFEST_TMP_FILENAME: &str = "manifest.bin.tmp";

/// Manifest file magic — recognised on load to refuse bogus files.
const MANIFEST_MAGIC: [u8; 8] = *b"ARTSNMNF";
/// Manifest format version. Bumped on any breaking change.
const MANIFEST_VERSION: u16 = 1;

/// NVMe-backed, O_DIRECT, single-packed-file blob store.
///
/// Construct via [`PersistentBackend::open`]. Thread-safe; the
/// underlying file handle is shared and `pread`/`pwrite` are
/// atomic at the syscall boundary.
#[derive(Debug)]
pub struct PersistentBackend {
    data_dir: PathBuf,
    data_file: File,
    manifest: RwLock<Manifest>,
    /// `io_uring` context — present iff Linux + `feature =
    /// "io-uring"`. Held behind a `Mutex` so concurrent callers
    /// serialise on the submission queue; with the v0.2 single I/O
    /// worker thread this lock is uncontended on the hot path.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    uring: Mutex<UringContext>,
}

#[derive(Debug)]
struct Manifest {
    /// guid → slot index (offset on disk = slot * u64::from(PAGE_SIZE)).
    slots: HashMap<BlobGuid, u64>,
    /// Next free slot to hand out. Monotonically increasing in
    /// Stage 1 (slot reuse comes in Stage 6 with the buffer
    /// manager's free-list reclamation).
    next_slot: u64,
    /// Path to the manifest file (for tmp+rename writes).
    path: PathBuf,
}

impl PersistentBackend {
    /// Open or create a persistent backend at `data_dir`.
    ///
    /// Creates the directory if missing. On Linux opens the packed
    /// data file with `O_DIRECT | O_CLOEXEC`; on other Unixes opens
    /// with `O_CLOEXEC` only (macOS additionally sets `F_NOCACHE`).
    /// Loads the manifest if present; otherwise starts empty.
    pub fn open<P: Into<PathBuf>>(data_dir: P) -> Result<Self> {
        let data_dir = data_dir.into();
        std::fs::create_dir_all(&data_dir)?;

        let data_path = data_dir.join(DATA_FILENAME);
        let manifest_path = data_dir.join(MANIFEST_FILENAME);

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

        let manifest = Manifest::load_or_create(&manifest_path)?;

        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let uring = Mutex::new(UringContext::new(&data_file)?);

        Ok(Self {
            data_dir,
            data_file,
            manifest: RwLock::new(manifest),
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            uring,
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
            Error::BackendIo(io::Error::new(
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
        let s = m.next_slot;
        m.next_slot += 1;
        m.slots.insert(guid, s);
        s
    }

    // ---------- I/O dispatch (uring vs pread/pwrite) ----------
    //
    // Two paired cfg-gated helpers per direction: the active one
    // compiles, the inactive one doesn't. Keeps `read_blob` /
    // `write_blob` clean of any conditional plumbing.

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn pread_at(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        ring.pread_at(offset, dst)?;
        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn pread_at(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.data_file.read_exact_at(dst, offset)?;
        Ok(())
    }

    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn pwrite_at(&self, offset: u64, src: &[u8]) -> Result<()> {
        let mut ring = self.uring.lock().unwrap();
        ring.pwrite_at(offset, src)?;
        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    fn pwrite_at(&self, offset: u64, src: &[u8]) -> Result<()> {
        self.data_file.write_all_at(src, offset)?;
        Ok(())
    }
}

impl Backend for PersistentBackend {
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
        let offset = self.offset_of(guid)?;
        self.pread_at(offset, dst.as_mut_slice())?;
        Ok(())
    }

    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
        let slot = self.assign_slot(guid);
        let offset = slot * u64::from(PAGE_SIZE);
        self.pwrite_at(offset, src.as_slice())?;
        Ok(())
    }

    fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        let mut m = self.manifest.write().unwrap();
        m.slots.remove(&guid);
        // Stage 6 will free the slot for reuse via a per-backend
        // free-list. Stage 1 just leaks the slot — the data on disk
        // becomes garbage until a future compaction overwrites it.
        Ok(())
    }

    fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
        let m = self.manifest.read().unwrap();
        Ok(m.slots.keys().copied().collect())
    }

    fn flush(&self) -> Result<()> {
        // Order matters: data must be on disk before the manifest
        // promotes any new slot. Otherwise a crash could leave the
        // manifest pointing at a slot whose data is still in NVMe's
        // write cache.
        self.data_file.sync_data()?;

        let m = self.manifest.read().unwrap();
        m.persist(&self.data_dir)?;
        Ok(())
    }
}

impl Manifest {
    fn load_or_create(path: &Path) -> Result<Self> {
        match File::open(path) {
            Ok(mut f) => Self::parse(&mut f, path.to_path_buf()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self {
                slots: HashMap::new(),
                next_slot: 0,
                path: path.to_path_buf(),
            }),
            Err(e) => Err(Error::BackendIo(e)),
        }
    }

    fn parse(f: &mut File, path: PathBuf) -> Result<Self> {
        // Header: magic 8 + version 2 + count 4 + reserved 2 + next_slot 8 = 24 B.
        let mut hdr = [0u8; 24];
        f.read_exact(&mut hdr)?;
        if hdr[..8] != MANIFEST_MAGIC {
            return Err(Error::NodeCorrupt {
                context: "PersistentBackend::Manifest::magic",
            });
        }
        let version = u16::from_le_bytes([hdr[8], hdr[9]]);
        if version != MANIFEST_VERSION {
            return Err(Error::NodeCorrupt {
                context: "PersistentBackend::Manifest::version",
            });
        }
        let count = u32::from_le_bytes([hdr[10], hdr[11], hdr[12], hdr[13]]) as usize;
        // hdr[14..16] reserved (zero).
        let next_slot = u64::from_le_bytes(hdr[16..24].try_into().unwrap());

        let mut slots = HashMap::with_capacity(count);
        let mut entry = [0u8; 24];
        for _ in 0..count {
            f.read_exact(&mut entry)?;
            let mut g: BlobGuid = [0u8; 16];
            g.copy_from_slice(&entry[..16]);
            let s = u64::from_le_bytes(entry[16..24].try_into().unwrap());
            slots.insert(g, s);
        }

        Ok(Self {
            slots,
            next_slot,
            path,
        })
    }

    fn persist(&self, data_dir: &Path) -> Result<()> {
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
            Error::BackendIo(io::Error::other("manifest slot count exceeds u32::MAX"))
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
        let dir = File::open(data_dir)?;
        dir.sync_all()?;
        Ok(())
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

    /// Skip every test in this module when O_DIRECT isn't supported
    /// by the filesystem we landed on (e.g. tmpfs on some kernels,
    /// or macOS-mounted-via-CI). Returns the open backend or `None`
    /// to skip cleanly.
    fn try_open(dir: &Path) -> Option<PersistentBackend> {
        match PersistentBackend::open(dir) {
            Ok(b) => Some(b),
            Err(Error::BackendIo(e)) if e.raw_os_error() == Some(libc::EINVAL) => {
                eprintln!("skipping: O_DIRECT not supported on this fs");
                None
            }
            Err(e) => panic!("unexpected open error: {e}"),
        }
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
