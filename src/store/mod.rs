//! Storage layer.
//!
//! - [`BlobFrame`] — typed view over one 512 KB blob, with bump
//!   allocator + per-NodeType free list.
//! - [`blob_store`] — blob-addressed storage trait and bundled
//!   memory / file-backed stores.
//! - [`buffer_manager`] — cache residency, dirty tracking,
//!   deferred deletes, and per-blob latching.
//! - [`BufferManager`] — LRU-bounded cache wrapping any `BlobStore`;
//!   it also implements `BlobStore` so it remains transparent above
//!   the store layer.

mod blob_frame;
pub(crate) mod blob_store;
mod buffer_manager;
mod cold;

pub(crate) use blob_frame::{
    decode_child_off, encode_child_off, page_align_up, PAGE_4K, SPILLOVER_RESERVATION,
};
pub use blob_frame::{AllocError, BlobFrame, BlobFrameRef, FreeError};
pub(crate) use blob_store::ColdBlobLookup;
pub(crate) use buffer_manager::{
    BlobWriteGuard, BufferStats, DirtySnapshotEntry, WriteDeltaEntry, WriteThroughEntry,
    WriteThroughStatus, STRUCTURAL_SEQ,
};
pub use buffer_manager::{BufferManager, CachedBlob};
pub(crate) use cold::{ColdIndexAnswer, ColdIndexHit, ColdIndexStamp};
