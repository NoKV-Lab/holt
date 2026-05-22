//! `TreeConfig` — the single argument to [`crate::Tree::open`].
//!
//! `TreeConfig` captures both **where** the tree lives ([`Storage`])
//! and how the engine internals are sized.
//!
//! The default — built via [`TreeConfig::new`] — is a file-backed
//! durable tree at the supplied directory. Override to memory mode with
//! [`TreeConfig::memory`] (or via [`crate::TreeBuilder::memory`]).

use std::path::PathBuf;

use crate::checkpoint::CheckpointConfig;

/// Commit acknowledgement boundary for file-backed WAL writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalCommit {
    /// Return after the WAL command is accepted by the journal
    /// worker queue. Highest foreground throughput, but an
    /// immediate process crash can lose queued records unless a
    /// checkpoint or later flush drains them first.
    Enqueue,
    /// Return after the journal worker has written the WAL bytes to
    /// the OS page cache. This matches the usual
    /// `WAL on, sync=false` benchmark/profile: process-crash
    /// recovery can replay the record, but power-loss durability is
    /// not forced per operation.
    Write,
    /// Return after the journal worker has called `sync_data`.
    /// This is the per-operation power-loss durability boundary;
    /// concurrent writers can share one fsync through group commit.
    Sync,
}

/// Where the tree's data lives.
///
/// `File` is the production target. `Memory` is for tests,
/// scratch use, and platforms without a usable file-backed store.
///
/// `#[non_exhaustive]` so adding new storage variants (e.g., a
/// future `RemoteObjectStore`) is a non-breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Storage {
    /// File-backed durable storage at `dir`. On Linux the
    /// [`crate::FileBlobStore`] opens the underlying file with
    /// `O_DIRECT` and (with the `io-uring` feature enabled) drives
    /// I/O through `io_uring`.
    File {
        /// Directory holding `blobs.dat`, `manifest.bin`,
        /// `manifest.log`, and `journal.wal`.
        dir: PathBuf,
    },
    /// In-memory only — volatile, drops on the last `Tree` handle.
    Memory,
}

/// Configuration passed to [`crate::Tree::open`].
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Where the tree's data lives.
    pub storage: Storage,
    /// How many 512 KB blob frames to keep pinned in the buffer
    /// pool. Default 64 (= 32 MB resident).
    pub buffer_pool_size: usize,
    /// Controls the WAL acknowledgement boundary on every
    /// file-backed mutation. The default [`WalCommit::Enqueue`]
    /// keeps foreground metadata writes on the async journal path.
    /// Benchmarks that want RocksDB-style `WAL on, sync=false`
    /// should set [`WalCommit::Write`] explicitly.
    pub wal_commit: WalCommit,
    /// **Memory-only** BM-commit toggle (no effect on
    /// file-backed trees — the WAL + `Tree::checkpoint` is the
    /// durability path there; see [`Self::wal_commit`]).
    ///
    /// For memory trees: `true` (the default) drains the BM
    /// dirty set into the backing `BlobStore` after every `put` /
    /// `delete` / `rename`, so custom stores supplied via
    /// [`crate::Tree::open_with_blob_store`] see state mirrored
    /// per op. `false` defers all writes to an explicit
    /// `Tree::checkpoint` call — useful in benches where the
    /// memcpy through `MemoryBlobStore` is uninteresting.
    pub memory_flush_on_write: bool,
    /// Background checkpointer policy. Default disabled —
    /// callers drive [`crate::Tree::checkpoint`] synchronously.
    /// Enable via [`CheckpointConfig::enabled`] or
    /// [`crate::TreeBuilder::checkpoint`].
    pub checkpoint: CheckpointConfig,
}

impl TreeConfig {
    /// File-backed durable tree rooted at `dir`. This is the **default**
    /// shape — `Tree::open(TreeConfig::new("/var/lib/myapp"))` is
    /// what production code typically writes.
    #[must_use]
    pub fn new<P: Into<PathBuf>>(dir: P) -> Self {
        Self {
            storage: Storage::File { dir: dir.into() },
            buffer_pool_size: 64,
            wal_commit: WalCommit::Enqueue,
            memory_flush_on_write: true,
            checkpoint: CheckpointConfig::default(),
        }
    }

    /// In-memory tree — volatile, for tests + scratch use.
    #[must_use]
    pub fn memory() -> Self {
        Self {
            storage: Storage::Memory,
            buffer_pool_size: 64,
            wal_commit: WalCommit::Enqueue,
            memory_flush_on_write: true,
            checkpoint: CheckpointConfig::default(),
        }
    }

    /// `true` iff [`Storage::Memory`].
    #[must_use]
    pub fn is_memory(&self) -> bool {
        matches!(self.storage, Storage::Memory)
    }

    /// Path of the WAL file for this configuration, if any.
    /// File-backed trees keep their log next to the data file at
    /// `<dir>/journal.wal`; memory trees have no WAL.
    #[must_use]
    pub fn wal_path(&self) -> Option<PathBuf> {
        match &self.storage {
            Storage::File { dir } => Some(dir.join("journal.wal")),
            Storage::Memory => None,
        }
    }
}
