//! `TreeBuilder` — fluent constructor for [`Tree`].

use std::path::PathBuf;
use std::sync::Arc;

use super::config::{Durability, Storage, TreeConfig};
use super::tree::Tree;
use crate::api::errors::Result;
use crate::checkpoint::CheckpointConfig;
use crate::store::blob_store::BlobStore;

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
