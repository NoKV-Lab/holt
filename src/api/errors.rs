//! Top-level error type.

use crate::store::{AllocError, FreeError};

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Top-level error type covering the union of every failure mode.
#[derive(Debug)]
pub enum Error {
    /// Backend I/O failure.
    BackendIo(std::io::Error),
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
    /// A walker-arm hit a NodeType the v0.1 engine doesn't yet
    /// implement (Node48, Node256, Blob, etc.). Will go away as
    /// the engine fills out.
    NotYetImplemented(&'static str),
    /// A blob's slot table or header is corrupt — recovery
    /// should bail out rather than silently misbehave.
    NodeCorrupt {
        /// Where the corruption was detected.
        context: &'static str,
    },
    /// WAL replay encountered a TxnOp whose `sanity_info`
    /// validation failed — record magic mismatch, CRC32 mismatch,
    /// unknown variant tag, truncated body, etc.
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

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendIo(e) => write!(f, "backend I/O: {e}"),
            Self::Alloc(e) => write!(f, "alloc: {e}"),
            Self::Free(e) => write!(f, "free: {e}"),
            Self::KeyTooLong { len } => write!(f, "key too long ({len} bytes; max {})", u16::MAX),
            Self::ValueTooLong { len } => {
                write!(f, "value too long ({len} bytes; max {})", u16::MAX)
            }
            Self::NotYetImplemented(where_) => write!(f, "not yet implemented: {where_}"),
            Self::NodeCorrupt { context } => write!(f, "node corrupt at {context}"),
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
            Self::BackendIo(e) => Some(e),
            Self::Alloc(e) => Some(e),
            Self::Free(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::BackendIo(e)
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
