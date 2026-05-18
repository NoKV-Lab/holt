//! Public `Tree` type — the main user-facing API.
//!
//! Stage 2c (current): `Tree::open`, `Tree::get`, `Tree::put`,
//! `Tree::delete`, `Tree::rename` are all wired against the walker.
//!
//! ## Internal key encoding
//!
//! Every user-supplied key is padded with a trailing `\0` byte
//! before reaching the walker. This is a standard ART trick to
//! resolve the "strict prefix" case where one key (e.g. `"abc"`)
//! is a prefix of another (e.g. `"abcdef"`): the terminator
//! guarantees the two keys diverge somewhere inside the radix
//! tree (at the `\0` vs `'d'` byte in this example).
//!
//! The trade-off: a user-supplied key MUST NOT end with `\0`.
//! Empty keys are fine — `b""` pads to `b"\0"` and round-trips
//! cleanly.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::config::{Storage, TreeConfig};
use super::errors::{Error, Result};
use crate::engine::{self, LookupResult};
use crate::layout::{BlobGuid, PAGE_SIZE};
use crate::store::backend::{AlignedBlobBuf, Backend, MemoryBackend};
use crate::store::BlobFrame;

#[cfg(unix)]
use crate::store::backend::PersistentBackend;

/// An `artisan` tree — your handle to one metadata store.
///
/// Clone the handle to share the same backing store: the backend
/// is held via `Arc` and writes serialise through a single
/// internal mutex (Stage 5 will swap the mutex for per-blob
/// `HybridLatch`).
#[derive(Clone)]
pub struct Tree {
    cfg: TreeConfig,
    backend: Arc<dyn Backend>,
    /// GUID of the blob holding the tree root. v0.1 uses a fixed
    /// sentinel; multi-blob support (Stage 2d) introduces a per-tree
    /// root manifest.
    root_guid: BlobGuid,
    /// Serialises mutations against the root blob. Stage 5
    /// (BufferManager + HybridLatch) makes this per-blob.
    write_lock: Arc<Mutex<()>>,
    /// Monotonically-increasing sequence stamped on every new
    /// leaf. Stage 5 ties this to the WAL record number.
    next_seq: Arc<AtomicU64>,
}

impl std::fmt::Debug for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tree")
            .field("storage", &self.cfg.storage)
            .field("root_guid", &self.root_guid)
            .finish_non_exhaustive()
    }
}

/// Fixed GUID of the root blob in v0.1. Multi-root trees (Stage 2d
/// onwards) will allocate per-tree root GUIDs from a manifest.
pub(crate) const ROOT_BLOB_GUID: BlobGuid = [0; 16];

/// Append the engine's internal terminator byte (`\0`) to a
/// user-supplied key. See the module docs.
#[inline]
fn pad_key(key: &[u8]) -> Vec<u8> {
    let mut padded = Vec::with_capacity(key.len() + 1);
    padded.extend_from_slice(key);
    padded.push(0u8);
    padded
}

impl Tree {
    /// Open a tree using the supplied configuration.
    ///
    /// `TreeConfig::new("/path")` opens a persistent tree at
    /// `"/path"` (the default). `TreeConfig::memory()` opens an
    /// in-memory tree.
    ///
    /// On non-Unix platforms, persistent mode is unavailable;
    /// passing a `Storage::Persistent` config there returns
    /// [`Error::NotYetImplemented`] — fall back to
    /// `TreeConfig::memory()` or supply your own [`Backend`] via
    /// [`Tree::open_with_backend`].
    pub fn open(cfg: TreeConfig) -> Result<Self> {
        let backend: Arc<dyn Backend> = match &cfg.storage {
            Storage::Memory => Arc::new(MemoryBackend::new()),
            Storage::Persistent { dir } => {
                #[cfg(unix)]
                {
                    Arc::new(PersistentBackend::open(dir)?)
                }
                #[cfg(not(unix))]
                {
                    let _ = dir;
                    return Err(Error::NotYetImplemented(
                        "PersistentBackend is Unix-only; use TreeConfig::memory() or supply a Backend via Tree::open_with_backend",
                    ));
                }
            }
        };
        Self::open_with_backend(cfg, backend)
    }

    /// Open a tree with a caller-supplied [`Backend`].
    ///
    /// Use this when you want to plug in something other than the
    /// built-in memory / persistent backends — e.g. a network-backed
    /// store, an instrumented wrapper, or a fault-injection harness.
    pub fn open_with_backend(cfg: TreeConfig, backend: Arc<dyn Backend>) -> Result<Self> {
        let root_guid = ROOT_BLOB_GUID;
        if !backend.has_blob(root_guid)? {
            let mut buf = AlignedBlobBuf::zeroed();
            BlobFrame::init(buf.as_mut_slice(), root_guid)?;
            backend.write_blob(root_guid, &buf)?;
            backend.flush()?;
        }
        Ok(Self {
            cfg,
            backend,
            root_guid,
            write_lock: Arc::new(Mutex::new(())),
            next_seq: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Look up `key`. Returns the value bytes, or `None` if no leaf
    /// matches.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let padded = pad_key(key);
        let mut buf = AlignedBlobBuf::zeroed();
        self.backend.read_blob(self.root_guid, &mut buf)?;
        let frame = BlobFrame::wrap(buf.as_mut_slice());
        let root_slot = frame.header().root_slot;
        match engine::lookup(&frame, root_slot, &padded)? {
            LookupResult::Found(v) => Ok(Some(v.to_vec())),
            LookupResult::NotFound => Ok(None),
        }
    }

    /// Insert or replace `(key, value)`. Returns the previous value
    /// if the key already existed.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<Option<Vec<u8>>> {
        let padded = pad_key(key);
        let _guard = self.write_lock.lock().unwrap();
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);

        let mut buf = AlignedBlobBuf::zeroed();
        self.backend.read_blob(self.root_guid, &mut buf)?;

        let outcome;
        {
            let mut frame = BlobFrame::wrap(buf.as_mut_slice());
            let root_slot = frame.header().root_slot;
            outcome = engine::walker::insert(&mut frame, root_slot, &padded, value, seq)?;
            frame.header_mut().root_slot = outcome.new_root_slot;
        }

        self.backend.write_blob(self.root_guid, &buf)?;
        Ok(outcome.previous)
    }

    /// Remove `key`. Returns the value that was stored at `key`, or
    /// `None` if no leaf matched.
    pub fn delete(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let padded = pad_key(key);
        let _guard = self.write_lock.lock().unwrap();

        let mut buf = AlignedBlobBuf::zeroed();
        self.backend.read_blob(self.root_guid, &mut buf)?;

        let outcome;
        {
            let mut frame = BlobFrame::wrap(buf.as_mut_slice());
            let root_slot = frame.header().root_slot;
            outcome = engine::walker::erase(&mut frame, root_slot, &padded)?;
            frame.header_mut().root_slot = outcome.new_root_slot;
        }

        // Even on NotFound we still rewrite — the frame is unchanged
        // (idempotent), so this is a 512 KB no-op write. Stage 6
        // BufferManager will skip the write when nothing was dirtied.
        self.backend.write_blob(self.root_guid, &buf)?;
        Ok(outcome.previous)
    }

    /// Move the value at `src` to `dst` in a single atomic step.
    ///
    /// - Returns [`Error::NotFound`] if `src` has no leaf.
    /// - Returns [`Error::DstExists`] if `dst` already has a leaf
    ///   **and** `force` is `false`.
    /// - When `force` is `true`, any existing leaf at `dst` is
    ///   overwritten.
    ///
    /// Atomic with respect to other writers (the internal
    /// `write_lock` is held for the whole erase-then-insert
    /// sequence) and atomic on disk (the underlying blob is
    /// rewritten exactly once at the end). Stage 5 will replace
    /// the lock with per-blob `HybridLatch` + a single
    /// `RenameObjectTxnOp` so observers can't see the intermediate
    /// "src gone, dst not yet present" state across blobs.
    pub fn rename(&self, src: &[u8], dst: &[u8], force: bool) -> Result<()> {
        let src_padded = pad_key(src);
        let dst_padded = pad_key(dst);

        let _guard = self.write_lock.lock().unwrap();
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);

        let mut buf = AlignedBlobBuf::zeroed();
        self.backend.read_blob(self.root_guid, &mut buf)?;

        {
            let mut frame = BlobFrame::wrap(buf.as_mut_slice());
            let root_slot = frame.header().root_slot;

            // Step 1: probe src to make sure it exists.
            //         Probe dst to honour the !force guard.
            let value: Vec<u8> = {
                match engine::lookup(&frame, root_slot, &src_padded)? {
                    LookupResult::Found(v) => v.to_vec(),
                    LookupResult::NotFound => return Err(Error::NotFound),
                }
            };

            // Same key? Treat as a no-op (no value-changing write,
            // but bump seq for caller-visible ordering).
            if src == dst {
                return Ok(());
            }

            if !force {
                let dst_exists = matches!(
                    engine::lookup(&frame, root_slot, &dst_padded)?,
                    LookupResult::Found(_),
                );
                if dst_exists {
                    return Err(Error::DstExists);
                }
            }

            // Step 2: erase src.
            let erase_out = engine::walker::erase(&mut frame, root_slot, &src_padded)?;
            frame.header_mut().root_slot = erase_out.new_root_slot;

            // Step 3: insert at dst — read root_slot fresh; erase
            //         may have collapsed/reseeded it.
            let new_root = frame.header().root_slot;
            let insert_out = engine::walker::insert(
                &mut frame,
                new_root,
                &dst_padded,
                &value,
                seq,
            )?;
            frame.header_mut().root_slot = insert_out.new_root_slot;
        }

        self.backend.write_blob(self.root_guid, &buf)?;
        Ok(())
    }

    /// Flush every previously-returned write through the backend.
    ///
    /// On the persistent backend this issues `fdatasync` on the
    /// underlying blobs file and rewrites the manifest. On the
    /// memory backend this is a no-op.
    pub fn checkpoint(&self) -> Result<()> {
        self.backend.flush()?;
        Ok(())
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &TreeConfig {
        &self.cfg
    }

    /// Total bytes a single blob frame consumes — useful for
    /// capacity sizing.
    #[must_use]
    pub const fn page_size() -> u32 {
        PAGE_SIZE
    }
}
