//! [`CompactReason`] — the on-disk tag carried in
//! [`crate::journal::TxnOp::Compact`] records.
//!
//! The compaction primitives themselves live next to their callers:
//!
//! - [`crate::engine::make_blob_from_node`] (deep-clone into a fresh
//!   blob) — in `walker/migrate.rs`, used by spillover.
//! - [`crate::engine::compact_blob`] (in-place rebuild reclaiming
//!   leaf-extent leaks and dropping orphans) — in `walker/migrate.rs`,
//!   wired into `insert_multi` / `insert_at_blob_node`'s OOM retry.
//! - `splitBlob` (out-of-space spillover) — in `walker/spillover.rs`.
//! - `mergeBlob` (inverse of split) — queued for v0.1.

/// Reason a compaction or split fired. Encoded into the WAL as the
/// `reason` body of [`crate::journal::TxnOp::Compact`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactReason {
    /// Too many tombstone leaves; rebuild dropping them.
    SplitTombstone,
    /// Bump-allocator wasted space exceeds threshold; rebuild
    /// compactly.
    SplitGapSpace,
    /// Alloc failed in current blob; spill a subtree.
    OutOfBlobFrame,
}
