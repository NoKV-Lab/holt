//! Storage backend layer.
//!
//! Only two backends exist in artisan and only two ever will:
//!
//! | Backend | Purpose | Where it works |
//! |---|---|---|
//! | [`MemoryBackend`]     | Tests, ephemeral trees, in-memory KV | All platforms |
//! | [`PersistentBackend`] | File-backed durable storage; Linux fast path uses `O_DIRECT` + `io_uring` | All Unix |
//!
//! The trait surface ([`Backend`]) is blob-granular: read / write a
//! full `PAGE_SIZE` ([`crate::layout::PAGE_SIZE`]) frame, list, delete,
//! flush. Anything coarser (multi-blob transactions, page caching,
//! eviction) lives above this layer in the buffer manager + WAL.
//!
//! All I/O flows through [`AlignedBlobBuf`] — a 4 KB-aligned heap
//! buffer that is safe to hand directly to `O_DIRECT` and that
//! `io_uring` can register for zero-syscall submission.

pub mod aligned;
pub mod memory;

#[cfg(unix)]
pub mod persistent;

pub use aligned::{AlignedBlobBuf, BUF_ALIGN};
pub use memory::MemoryBackend;

#[cfg(unix)]
pub use persistent::PersistentBackend;

use crate::api::errors::Result;
use crate::layout::BlobGuid;

/// A blob-granular storage backend.
///
/// All implementations are `Send + Sync` so the buffer manager can
/// drive concurrent I/O from multiple worker threads.
///
/// # Contract
/// - `read_blob` / `write_blob` always operate on a full
///   `PAGE_SIZE`-byte frame. Partial I/O is not supported.
/// - `write_blob` is **atomic at the blob level**: either the entire
///   new image is visible to subsequent reads, or the old image is.
///   No torn writes (`PersistentBackend` achieves this via O_DIRECT
///   page-aligned writes on NVMe — physically atomic for ≤ 4 KB,
///   logically atomic for 512 KB via journal coordination).
/// - `flush` blocks until **every** write that returned before the
///   call is durable on the underlying medium.
pub trait Backend: Send + Sync {
    /// Read blob `guid` into `dst`. `dst.len() == PAGE_SIZE`.
    fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()>;

    /// Write `src` as blob `guid`. `src.len() == PAGE_SIZE`.
    ///
    /// Returns once the write has been *submitted* to the medium.
    /// Call [`Backend::flush`] to wait for it to be *durable*.
    fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()>;

    /// Delete blob `guid`. No-op if it doesn't exist.
    fn delete_blob(&self, guid: BlobGuid) -> Result<()>;

    /// Enumerate every blob currently stored.
    fn list_blobs(&self) -> Result<Vec<BlobGuid>>;

    /// Wait until every previously-returned write is durable.
    fn flush(&self) -> Result<()>;

    /// `true` iff `guid` exists. Default impl scans `list_blobs`.
    fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
        self.list_blobs().map(|v| v.contains(&guid))
    }
}
