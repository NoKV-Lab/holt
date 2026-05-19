//! ART walker — descent / insert / erase / scan / rename / compact.
//!
//! Stage 2a–2c: single-blob lookup + insert + erase land.
//! Stage 2d: multi-blob descent (BlobNode crossing + splitBlob).
//! Stage 6 phase 2a (current): read paths take [`BlobFrameRef`]
//! and run zero-copy against `BufferManager::pin`-ed buffers.
//!
//! [`simd`] hosts SIMD hot paths the walker calls into (Node16
//! byte search, longest-common-prefix).
//!
//! [`BlobFrameRef`]: crate::store::BlobFrameRef

pub mod compact;
pub mod iter;
pub mod simd;
pub mod walker;

pub use walker::{
    collect_blob_guids, compact_blob, erase, erase_multi, insert, insert_multi, lookup, lookup_at,
    lookup_multi, make_blob_from_node, BlobNodeCrossing, CompactStats, EraseOutcome, InsertOutcome,
    LookupResult, MakeBlobOutcome,
};
