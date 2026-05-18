//! Storage layer.
//!
//! - [`BlobFrame`] — typed view over one 512 KB blob, with bump
//!   allocator + per-NodeType free list.
//! - [`backend`] — pluggable storage backend trait
//!   (memory / persistent / future io_uring).
//! - [`BufferManager`] — LRU-bounded cache wrapping any `Backend`,
//!   itself implementing `Backend` so it's transparent.

pub mod backend;
mod blob_frame;
mod buffer_manager;

pub use blob_frame::{
    AllocError, AllocOutcome, BlobFrame, BlobFrameRef, ExtentAllocOutcome, FreeError,
};
pub use buffer_manager::{
    BlobReadGuard, BlobWriteGuard, BufferManager, CachedBlob, OptimisticGuard,
};
