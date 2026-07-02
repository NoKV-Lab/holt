//! Blob-store layer.
//!
//! This layer is deliberately blob-granular: the ART and buffer
//! manager hand it complete 512 KB frames identified by `BlobGuid`.
//!
//! | BlobStore | Purpose |
//! |---|---|
//! | [`MemoryBlobStore`]     | Tests, ephemeral trees, in-memory KV |
//! | [`FileBlobStore`] | File-backed durable storage; `O_DIRECT` on Linux, `F_NOCACHE` on macOS |
//!
//! Both stores run on every supported platform — holt is **Unix-only**
//! (Linux + macOS); the crate refuses to compile on Windows.
//!
//! The trait surface ([`BlobStore`]) is blob-granular: read / write a
//! full `PAGE_SIZE` ([`crate::layout::PAGE_SIZE`]) frame, list, delete,
//! flush. Anything coarser (multi-blob atomicity, page caching,
//! eviction) lives above this layer in the buffer manager + WAL.
//!
//! All I/O flows through [`AlignedBlobBuf`] — a 4 KB-aligned
//! frame that is safe to hand directly to `O_DIRECT`. Linux
//! `io_uring` file stores can lease these frames from a registered
//! fixed-buffer pool, but that allocator stays below the store
//! boundary.

pub mod aligned;
#[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
mod buffer_pool;
pub mod file;
pub mod memory;

pub use aligned::AlignedBlobBuf;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub(crate) use buffer_pool::BlobBufPool;
pub use file::FileBlobStore;
pub use memory::MemoryBlobStore;

use crate::api::errors::Result;
use crate::api::stats::{StoreStats, VacuumStats};
use crate::layout::BlobGuid;

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexedBlobLookup {
    Unknown,
    NotFound,
    Found {
        value: Vec<u8>,
        seq: u64,
    },
    Crossing {
        child_guid: BlobGuid,
        child_depth: usize,
    },
}

/// A blob-granular storage interface.
///
/// All implementations are `Send + Sync` so the buffer manager can
/// drive concurrent I/O from multiple worker threads.
///
/// # Contract
/// - `read_blob` / `write_blob` always operate on a full
///   `PAGE_SIZE`-byte frame. Partial I/O is not supported.
/// - `write_blob` replaces the full frame visible to later
///   `read_blob` calls after it returns. The trait does not require
///   power-loss atomicity for a 512 KB frame; Holt's WAL/checkpoint
///   protocol is the recovery source of truth.
/// - `flush` blocks until **every** write that returned before the
///   call is durable on the underlying medium.
pub trait BlobStore: Send + Sync {
    /// Allocate a zero-filled blob buffer suitable for this store.
    ///
    /// The default is a heap-backed 4 KB-aligned frame. Linux
    /// `io_uring` file stores override this to lease from
    /// their registered fixed-buffer pool when available.
    fn alloc_blob_buf_zeroed(&self) -> AlignedBlobBuf {
        AlignedBlobBuf::zeroed()
    }

    /// Read blob `guid` into `dst`. `dst.len() == PAGE_SIZE`.
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()>;

    /// Read a batch of full-blob frames: `guids[i]` into `dsts[i]`
    /// (the slices must be equal length). Returns one `Result` per
    /// request — best-effort batch semantics, so a failed slot (e.g.
    /// a not-found guid) does **not** abort the others.
    ///
    /// The default loops over [`Self::read_blob`]. Stores that can
    /// issue the reads with more device parallelism than a serial
    /// loop override this: Linux `io_uring` submits one ring batch
    /// (deep read queue from a single SQ owner), and the `pread` file
    /// store fans the positional reads out across worker threads. The
    /// buffer manager's cold-scan read-ahead uses this to fetch
    /// upcoming child blobs at the device's natural queue depth
    /// instead of one serial round-trip each.
    fn read_blobs(&self, guids: &[BlobGuid], dsts: &mut [AlignedBlobBuf]) -> Vec<Result<()>> {
        debug_assert_eq!(guids.len(), dsts.len());
        guids
            .iter()
            .zip(dsts.iter_mut())
            .map(|(guid, dst)| self.read_blob(*guid, dst))
            .collect()
    }

    /// Read `dst.len()` bytes starting at `byte_offset` within blob `guid`.
    ///
    /// Enables page-granular fallback reads when a read-index hit
    /// points at a value that was too large to inline. For `O_DIRECT`
    /// backends the caller keeps `byte_offset`, `dst.len()`, and
    /// `dst`'s base 4 KB-aligned; the read-whole default below
    /// imposes no such requirement.
    ///
    /// The default reads the entire frame and copies the sub-range — correct
    /// for any store but with no I/O saving. `FileBlobStore` /
    /// `MemoryBlobStore` override it with a genuine ranged read.
    fn read_blob_range(&self, guid: BlobGuid, byte_offset: u64, dst: &mut [u8]) -> Result<()> {
        let mut full = self.alloc_blob_buf_zeroed();
        self.read_blob(guid, &mut full)?;
        let start = byte_offset as usize;
        dst.copy_from_slice(&full.as_slice()[start..start + dst.len()]);
        Ok(())
    }

    /// Read a byte range from the optional read index for `guid`.
    ///
    /// Read indexes are accelerators only: `None`, corrupt bytes, or a
    /// stale stamp must make callers fall back to authoritative blob
    /// reads. Memory/custom stores default to no read-index support.
    fn read_index_range(
        &self,
        _guid: BlobGuid,
        _byte_offset: u64,
        _dst: &mut [u8],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Read a byte range from the optional value segment for `guid`.
    ///
    /// Value segments are addressed by read-index entries. They are
    /// rebuildable accelerators only; absence or corruption must make
    /// callers fall back to the authoritative blob frame.
    fn read_value_segment_range(
        &self,
        _guid: BlobGuid,
        _byte_offset: u64,
        _dst: &mut [u8],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Publish rebuilt read-index bytes for `guid`.
    ///
    /// `index_bytes` is the routable directory/bucket index;
    /// `value_bytes` is an optional same-blob payload region used by
    /// large value segments. A failure here must not make committed blob
    /// data unrecoverable; callers may ignore the error after
    /// invalidating any cached index.
    fn publish_read_index(
        &self,
        _guid: BlobGuid,
        _index_bytes: &[u8],
        _value_bytes: &[u8],
    ) -> Result<()> {
        Ok(())
    }

    /// Delete the optional read index for `guid`.
    fn delete_read_index(&self, _guid: BlobGuid) -> Result<()> {
        Ok(())
    }

    /// Write `src` as blob `guid`. `src.len() == PAGE_SIZE`.
    ///
    /// Returns once the write has been *submitted* to the medium.
    /// Call [`BlobStore::flush`] to wait for it to be *durable*.
    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()>;

    /// Write a batch of full-blob images.
    ///
    /// The default implementation loops over [`Self::write_blob`].
    /// Stores with a cheaper native batch path should override
    /// this. The contract is conservative: if this returns `Err`,
    /// the caller must assume an arbitrary prefix may have reached
    /// the store and retry the whole batch later.
    fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        for (guid, src) in writes {
            self.write_blob(*guid, src)?;
        }
        Ok(())
    }

    /// Write a batch and, if the store can do it cheaply, make
    /// the data-file bytes durable before returning.
    ///
    /// This is deliberately narrower than [`Self::flush`]: callers
    /// must still call `flush` to persist metadata/manifest changes.
    /// The hook exists for Linux `io_uring`, where checkpoint
    /// write batches can keep data writes and `fdatasync` on the
    /// same ring turn, then let the later manifest flush skip the
    /// data sync if no newer writes raced in.
    fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
        self.write_blobs(writes)
    }

    /// Delete blob `guid`. No-op if it doesn't exist.
    fn delete_blob(&self, guid: BlobGuid) -> Result<()>;

    /// Enumerate every blob currently stored.
    fn list_blobs(&self) -> Result<Vec<BlobGuid>>;

    /// Wait until every previously-returned write is durable.
    fn flush(&self) -> Result<()>;

    /// Conservative hint for callers that want to skip a no-op
    /// flush. Stores should return `true` whenever a prior
    /// returned write, delete, or metadata update still needs
    /// [`Self::flush`] to make it durable.
    fn needs_flush(&self) -> bool {
        true
    }

    /// `true` iff `guid` exists. Default impl scans `list_blobs`.
    fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
        self.list_blobs().map(|v| v.contains(&guid))
    }

    /// Store-level space counters. Implementations that do not expose
    /// packed-file or read-index state may return zeros.
    fn store_stats(&self) -> StoreStats {
        StoreStats::default()
    }

    /// Reclaim physical store space that is already logically free.
    ///
    /// The default is a no-op for memory/custom stores. File-backed
    /// stores may relocate live high-water slots into lower reusable
    /// holes before trimming packed-file tails and, where supported,
    /// release physical blocks for reusable middle slots. GUID
    /// mappings remain authoritative and stable.
    fn vacuum(&self) -> Result<VacuumStats> {
        Ok(VacuumStats::default())
    }
}
