//! Top-level error type.

use crate::layout::BlobGuid;
use crate::store::{AllocError, FreeError};

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Top-level error type covering the union of every failure mode.
///
/// Marked `#[non_exhaustive]` — new variants may be added in
/// minor releases without breaking SemVer. Callers should always
/// match with a `_` arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// BlobStore I/O failure.
    BlobStoreIo(std::io::Error),
    /// Bump allocator / slot table exhaustion / invalid alloc.
    Alloc(AllocError),
    /// Free-list misuse.
    Free(FreeError),
    /// Key longer than `u16::MAX` bytes.
    KeyTooLong {
        /// Caller-supplied length.
        len: usize,
    },
    /// Value longer than `u16::MAX` bytes.
    ValueTooLong {
        /// Caller-supplied length.
        len: usize,
    },
    /// A walker arm hit a code path that the engine doesn't yet
    /// implement (e.g. degenerate `Leaf` / `EmptyRoot` spillover
    /// or strict-prefix ART insert cases). The static string names
    /// the unimplemented case for diagnostics.
    NotYetImplemented(&'static str),
    /// An internal invariant the engine relies on was observed
    /// to be violated — typically a background thread closing a
    /// channel it shouldn't have closed, or a completion sender
    /// disappearing without producing a result. Distinct from
    /// [`Self::NotYetImplemented`] (genuine feature gap) and
    /// from [`Self::NodeCorrupt`] (on-disk / cache layout
    /// problem). The static string names the specific invariant
    /// for triage.
    Internal(&'static str),
    /// A blob's slot table or header is corrupt — recovery
    /// should bail out rather than silently misbehave.
    ///
    /// Construct via [`Error::node_corrupt`] and optionally
    /// enrich with [`Error::with_blob_guid`] / [`Error::with_slot`]
    /// when the surrounding code path knows the affected blob
    /// or slot. The buffer manager and walker entry points
    /// automatically attach blob context where they have it.
    NodeCorrupt {
        /// Static description of where the corruption was detected.
        context: &'static str,
        /// GUID of the blob that exposed the corruption, when
        /// the propagating code path knows it. `None` for low-
        /// level helpers that don't have blob context (e.g. raw
        /// `BlobFrame::wrap` on a stack buffer).
        blob_guid: Option<BlobGuid>,
        /// Slot inside `blob_guid` that exposed the corruption,
        /// when applicable. `None` for header / body-level
        /// problems.
        slot: Option<u16>,
    },
    /// WAL replay encountered a record whose sanity validation
    /// failed — record magic mismatch, CRC32 mismatch, unknown
    /// variant tag, truncated body, etc.
    ReplaySanityFailed {
        /// What went wrong (decoder-supplied static string).
        context: &'static str,
        /// Position in the journal file where the bad record
        /// starts. `0` when the codec is invoked on a raw
        /// in-memory buffer and the caller hasn't supplied an
        /// offset.
        record_offset: u64,
    },
    /// `Tree::rename` (or similar) called with a `src` that has no
    /// leaf in the tree.
    NotFound,
    /// `Tree::rename(.., force=false)` called with a `dst` that
    /// already has a leaf. Caller can retry with `force=true` to
    /// overwrite.
    DstExists,
}

impl Error {
    /// Construct a [`Error::NodeCorrupt`] with the given static
    /// context but no blob / slot metadata. Layers higher in the
    /// stack (buffer manager, walker entry points) typically
    /// enrich the error via [`Self::with_blob_guid`] /
    /// [`Self::with_slot`] before it surfaces to the caller.
    #[must_use]
    pub const fn node_corrupt(context: &'static str) -> Self {
        Self::NodeCorrupt {
            context,
            blob_guid: None,
            slot: None,
        }
    }

    /// Attach a `blob_guid` to a [`Self::NodeCorrupt`] error
    /// without overwriting one set by a deeper layer. No-op for
    /// other variants — safe to chain unconditionally on any
    /// `Error` value.
    #[must_use]
    pub fn with_blob_guid(mut self, guid: BlobGuid) -> Self {
        if let Self::NodeCorrupt { blob_guid, .. } = &mut self {
            if blob_guid.is_none() {
                *blob_guid = Some(guid);
            }
        }
        self
    }

    /// Attach a `slot` index to a [`Self::NodeCorrupt`] error
    /// without overwriting one set by a deeper layer.
    #[must_use]
    pub fn with_slot(mut self, slot_index: u16) -> Self {
        if let Self::NodeCorrupt { slot, .. } = &mut self {
            if slot.is_none() {
                *slot = Some(slot_index);
            }
        }
        self
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BlobStoreIo(e) => write!(f, "store I/O: {e}"),
            Self::Alloc(e) => write!(f, "alloc: {e}"),
            Self::Free(e) => write!(f, "free: {e}"),
            Self::KeyTooLong { len } => write!(f, "key too long ({len} bytes; max {})", u16::MAX),
            Self::ValueTooLong { len } => {
                write!(f, "value too long ({len} bytes; max {})", u16::MAX)
            }
            Self::NotYetImplemented(where_) => write!(f, "not yet implemented: {where_}"),
            Self::Internal(what) => write!(f, "internal invariant violated: {what}"),
            Self::NodeCorrupt {
                context,
                blob_guid,
                slot,
            } => {
                write!(f, "node corrupt at {context}")?;
                if let Some(g) = blob_guid {
                    // First 4 bytes is enough to disambiguate in
                    // logs without dumping the full 16-byte tag.
                    write!(f, " (blob={:02x?})", &g[..4])?;
                }
                if let Some(s) = slot {
                    write!(f, " (slot={s})")?;
                }
                Ok(())
            }
            Self::ReplaySanityFailed {
                context,
                record_offset,
            } => {
                write!(
                    f,
                    "WAL replay sanity-check failed at offset {record_offset}: {context}"
                )
            }
            Self::NotFound => write!(f, "key not found"),
            Self::DstExists => write!(
                f,
                "destination key already exists (use force=true to overwrite)"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BlobStoreIo(e) => Some(e),
            Self::Alloc(e) => Some(e),
            Self::Free(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::BlobStoreIo(e)
    }
}
impl From<AllocError> for Error {
    fn from(e: AllocError) -> Self {
        Self::Alloc(e)
    }
}
impl From<FreeError> for Error {
    fn from(e: FreeError) -> Self {
        Self::Free(e)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;
    use std::io;

    use super::*;

    #[test]
    fn node_corrupt_context_is_enriched_once() {
        let err = Error::node_corrupt("slot decode")
            .with_blob_guid([0xAB; 16])
            .with_blob_guid([0xCD; 16])
            .with_slot(7)
            .with_slot(9);

        let rendered = err.to_string();
        assert!(rendered.contains("node corrupt at slot decode"));
        assert!(rendered.contains("blob=[ab, ab, ab, ab]"));
        assert!(rendered.contains("slot=7"));
        assert!(!rendered.contains("cd"));
        assert!(!rendered.contains("slot=9"));
    }

    #[test]
    fn display_covers_public_error_variants() {
        let cases = [
            (
                Error::KeyTooLong { len: 70_000 }.to_string(),
                "key too long (70000 bytes; max 65535)",
            ),
            (
                Error::ValueTooLong { len: 70_001 }.to_string(),
                "value too long (70001 bytes; max 65535)",
            ),
            (
                Error::NotYetImplemented("strict prefix").to_string(),
                "not yet implemented: strict prefix",
            ),
            (
                Error::Internal("lost dirty image").to_string(),
                "internal invariant violated: lost dirty image",
            ),
            (
                Error::ReplaySanityFailed {
                    context: "bad CRC",
                    record_offset: 42,
                }
                .to_string(),
                "WAL replay sanity-check failed at offset 42: bad CRC",
            ),
            (Error::NotFound.to_string(), "key not found"),
            (
                Error::DstExists.to_string(),
                "destination key already exists (use force=true to overwrite)",
            ),
        ];

        for (actual, expected) in cases {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn error_sources_are_exposed_for_wrapped_errors_only() {
        let blob = Error::from(io::Error::other("disk"));
        assert!(blob.source().is_some());

        let alloc = Error::from(AllocError::OutOfSlots);
        assert!(alloc.source().is_some());
        assert_eq!(
            alloc.to_string(),
            format!("alloc: {}", AllocError::OutOfSlots)
        );

        let free = Error::from(FreeError::InvalidSlot(99));
        assert!(free.source().is_some());
        assert_eq!(free.to_string(), "free: free_node: invalid slot index 99");

        assert!(Error::NotFound.source().is_none());
        assert!(Error::DstExists.source().is_none());
    }
}
