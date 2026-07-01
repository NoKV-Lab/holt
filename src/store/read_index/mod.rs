//! Read-index accelerators.
//!
//! These structures are advisory only. The blob file remains the
//! source of truth; stale, missing, or corrupt indexed state must make the
//! caller fall back to the authoritative full-blob path.

mod index;
mod page_cache;

pub(crate) use index::{
    PrefixLiveness, ReadIndex, ReadIndexAnswer, ReadIndexCache, ReadIndexHit, ReadIndexStamp,
};
pub(crate) use page_cache::ReadPageCache;
