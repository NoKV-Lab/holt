//! `TreeBuilder` — fluent constructor for [`Tree`].

use std::path::PathBuf;
use std::sync::Arc;

use super::config::{Durability, Storage, TreeConfig};
use super::tree::Tree;
use crate::api::errors::Result;
use crate::checkpoint::CheckpointConfig;
use crate::store::blob_store::BlobStore;
use crate::FileStoreObjectIdentity;

/// Fluent constructor for [`Tree`].
///
/// ```ignore
/// // Persistent (the default):
/// let tree = holt::TreeBuilder::new("/var/lib/myapp")
///     .buffer_pool_size(512) // 256 MiB total cache budget
///     .durability(holt::Durability::Wal { sync: true })
///     .open()?;
///
/// // In-memory (volatile, for tests / scratch):
/// let tree = holt::TreeBuilder::new("scratch")
///     .memory()
///     .open()?;
/// ```
#[derive(Debug, Clone)]
#[must_use = "TreeBuilder is consumed by `.open()` / `.open_with_blob_store()`; chained setters return a fresh builder you must use"]
pub struct TreeBuilder {
    cfg: TreeConfig,
}

impl TreeBuilder {
    /// Start a builder targeting `data_dir` in persistent mode
    /// (the default).
    pub fn new<P: Into<PathBuf>>(data_dir: P) -> Self {
        Self {
            cfg: TreeConfig::new(data_dir),
        }
    }

    /// Flip the builder to **in-memory** mode. The supplied
    /// `data_dir` becomes informational only.
    pub fn memory(mut self) -> Self {
        self.cfg.storage = Storage::Memory;
        self
    }

    /// Set cache budget, expressed in number of 512 KB blob frames.
    pub fn buffer_pool_size(mut self, n: usize) -> Self {
        self.cfg.buffer_pool_size = n;
        self
    }

    /// Set the durability policy (WAL vs materialized state machine).
    pub fn durability(mut self, durability: Durability) -> Self {
        self.cfg.durability = durability;
        self
    }

    /// Background checkpointer policy.
    ///
    /// Persistent trees enable it by default so the dirty set and WAL
    /// stay bounded. Pass a config with `enabled = false` to drive
    /// [`Tree::checkpoint`] synchronously instead.
    pub fn checkpoint(mut self, cfg: CheckpointConfig) -> Self {
        self.cfg.checkpoint = cfg;
        self
    }

    /// Require the existing file store to match `expected` before any
    /// authoritative data, manifest, or WAL file is opened.
    ///
    /// Existing authority files must already be exact mode 0600. Holt does
    /// not repair them while serving; stop all openers, confirm no old file
    /// descriptors remain, and correct legacy modes offline. After open, use
    /// [`Tree::validate_file_store_object_set`] before publishing or renewing
    /// a live fence.
    pub fn expected_file_store_identity(mut self, expected: FileStoreObjectIdentity) -> Self {
        self.cfg.expected_file_store_identity = Some(expected);
        self
    }

    /// Open with the configured storage mode.
    pub fn open(self) -> Result<Tree> {
        Tree::open(self.cfg)
    }

    /// Open with a caller-supplied [`BlobStore`] (overrides the
    /// builder's storage mode).
    pub fn open_with_blob_store(mut self, store: Arc<dyn BlobStore>) -> Result<Tree> {
        self.cfg.memory_flush_on_write = true;
        Tree::open_with_blob_store(self.cfg, store)
    }
}
