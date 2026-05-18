//! ART walker — descent / insert / erase / scan / rename / compact.
//!
//! Stage 2a–2c (current): single-blob lookup + insert + erase land.
//! Stage 2d: multi-blob descent (BlobNode crossing + splitBlob).
//!
//! [`simd`] hosts SIMD hot paths the walker calls into (Node16
//! byte search, longest-common-prefix).

pub mod walker;
pub mod compact;
pub mod iter;
pub mod simd;

pub use walker::{erase, insert, lookup, EraseOutcome, InsertOutcome, LookupResult};
