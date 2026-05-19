//! Public `Tree` type — the main user-facing API.
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
//! ## Concurrency model
//!
//! Tree owns an `Arc<BufferManager>`. The BM keeps each cached
//! blob behind a `HybridLatch` (LeanStore-style 3-mode latch)
//! wrapping an `UnsafeCell<AlignedBlobBuf>`:
//!
//! - **Reads** (`get`) walk every blob in **optimistic** mode —
//!   wait-free, no real lock taken. The walker snapshots the
//!   latch version, reads the buffer, then validates; on a torn
//!   read it restarts from the root. Readers never block writers
//!   and writers never block readers.
//! - **Writes** (`put` / `delete`) take **exclusive** mode on
//!   each blob they touch (always starting with the root). This
//!   serialises concurrent mutators on the same blob without any
//!   Tree-wide writer mutex; mutations on disjoint child blobs
//!   can proceed in parallel.
//! - **`rename`** is multi-step (lookup probe + erase + insert)
//!   and must be atomic across all three. It takes the
//!   `rename_lock` (a `Mutex<()>` scoped to rename only) to
//!   prevent racing renames from interleaving.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::config::{Storage, TreeConfig};
use super::errors::{Error, Result};
use crate::engine;
use crate::journal::reader::replay;
use crate::journal::txn_op::TxnOp;
use crate::journal::writer::WalWriter;
use crate::layout::{BlobGuid, PAGE_SIZE};
use crate::store::backend::{AlignedBlobBuf, Backend, MemoryBackend};
use crate::store::{BlobFrame, BlobFrameRef, BufferManager};

#[cfg(unix)]
use crate::store::backend::PersistentBackend;

/// An `holt` tree — your handle to one metadata store.
///
/// Clone the handle to share the same backing store: the
/// `BufferManager` is held via `Arc`. Reads run lock-free against
/// the writer mutex; writers serialise through `write_lock`.
#[derive(Clone)]
pub struct Tree {
    cfg: TreeConfig,
    backend: Arc<BufferManager>,
    /// GUID of the blob holding the tree root. v0.1 uses a fixed
    /// sentinel; multi-tenant trees (post-v0.1) will allocate
    /// per-tree root GUIDs from a manifest.
    root_guid: BlobGuid,
    /// Serialises **only `rename`** — the multi-step
    /// `lookup_multi(src)` + `erase_multi(src)` + `insert_multi(dst)`
    /// must appear atomic to other writers. `put` / `delete` /
    /// `get` never take this lock; they coordinate via the
    /// per-blob `HybridLatch` inside the BM.
    rename_lock: Arc<Mutex<()>>,
    /// Monotonically-increasing sequence stamped on every record.
    /// On open the tree replays the WAL and resumes at
    /// `highest_seq + 1`.
    next_seq: Arc<AtomicU64>,
    /// WAL handle — `Some` for persistent trees, `None` for
    /// memory trees (logging an in-memory engine has no point).
    wal: Option<Arc<Mutex<WalWriter>>>,
}

impl std::fmt::Debug for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tree")
            .field("storage", &self.cfg.storage)
            .field("root_guid", &self.root_guid)
            .finish_non_exhaustive()
    }
}

/// Fixed GUID of the root blob in v0.1. Multi-root trees
/// (post-v0.1) will allocate per-tree root GUIDs from a manifest.
pub(crate) const ROOT_BLOB_GUID: BlobGuid = [0; 16];

/// Per-blob counters captured by [`Tree::stats`].
///
/// Each field mirrors a [`BlobHeader`](crate::layout::BlobHeader)
/// counter and is read in a single shared-guard pass over the blob.
#[derive(Debug, Clone, Copy)]
pub struct BlobStats {
    /// GUID identifying this blob within the tree.
    pub guid: BlobGuid,
    /// Bytes currently consumed in the blob's data area (bump
    /// cursor, monotonically advancing).
    pub space_used: u32,
    /// Bytes lost to bump-allocator waste (extents freed without
    /// recycling). Compaction reclaims this to zero.
    pub gap_space: u32,
    /// High-water slot count — slot indices `[1, num_slots)` have
    /// ever held a node body in this blob.
    pub num_slots: u16,
    /// Count of cross-blob crossings (`BlobNode` slots) currently
    /// installed in this blob.
    pub num_ext_blobs: u16,
    /// Number of times this blob has been rebuilt by
    /// [`crate::engine::compact_blob`]. Bumped at the end of every
    /// successful compaction.
    pub compact_times: u32,
    /// Count of leaves in this blob currently in tombstone state
    /// (soft-deleted, awaiting reclaim by compaction).
    pub tombstone_leaf_cnt: u32,
}

/// Tree-wide aggregate counters from [`Tree::stats`].
///
/// `blobs` carries the per-blob breakdown in BFS order from the
/// root; the totals are pre-summed for the common "how big is the
/// tree?" question. All bytes / counts are sums over `blobs`.
#[derive(Debug, Clone)]
pub struct TreeStats {
    /// Number of distinct blobs reachable from the tree root.
    pub blob_count: u32,
    /// Sum of `space_used` over every blob.
    pub total_space_used: u64,
    /// Sum of `gap_space` over every blob.
    pub total_gap_space: u64,
    /// Sum of `num_slots` over every blob.
    pub total_slots: u64,
    /// Sum of `compact_times` over every blob (lifetime
    /// compactions across the whole tree).
    pub total_compactions: u64,
    /// Sum of `tombstone_leaf_cnt` over every blob.
    pub total_tombstones: u64,
    /// Per-blob breakdown in BFS order from the root.
    pub blobs: Vec<BlobStats>,
}

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
        // The auto-managed backend earns automatic WAL coverage.
        Self::open_inner(cfg, backend, /*attach_wal=*/ true)
    }

    /// Open a tree with a caller-supplied [`Backend`].
    ///
    /// **No WAL is attached.** The caller's backend has its own
    /// notion of durability (or is intentionally volatile —
    /// e.g. a `MemoryBackend` standing in for a real one in a
    /// test); holt stays out of that decision. If you want a
    /// WAL'd persistent tree, use [`Tree::open`] with a
    /// `Storage::Persistent` config.
    ///
    /// The supplied backend is **transparently wrapped** with a
    /// [`BufferManager`] of `cfg.buffer_pool_size` blobs.
    /// `BufferManager` owns the in-memory blob cache; the walker
    /// pins blobs from it for both reads and writes — no separate
    /// root buffer in `Tree`.
    ///
    /// If the backend doesn't yet contain a root blob, initialises
    /// an empty one and writes it through, flushing before
    /// returning.
    pub fn open_with_backend(cfg: TreeConfig, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::open_inner(cfg, backend, /*attach_wal=*/ false)
    }

    fn open_inner(cfg: TreeConfig, backend: Arc<dyn Backend>, attach_wal: bool) -> Result<Self> {
        let bm: Arc<BufferManager> = Arc::new(BufferManager::new(backend, cfg.buffer_pool_size));
        let root_guid = ROOT_BLOB_GUID;
        if !bm.has_blob(root_guid)? {
            // Seed an empty root blob and write it through.
            let mut scratch = AlignedBlobBuf::zeroed();
            BlobFrame::init(scratch.as_mut_slice(), root_guid)?;
            bm.write_blob(root_guid, &scratch)?;
            bm.flush()?;
        }

        // Persistent trees keep a WAL alongside the data file.
        // Replay every durable record onto the BM-cached blob
        // image before exposing the tree to callers: the on-disk
        // blob lags the WAL between the last `Tree::checkpoint`
        // and now, so the WAL is the source of truth for any op
        // committed via `flush_on_write = true`.
        let (wal, next_seq) = if attach_wal {
            match cfg.wal_path() {
                None => (None, 1u64),
                Some(path) => {
                    let next_seq = if path.exists() {
                        replay_wal(&path, &bm, root_guid)?
                    } else {
                        1
                    };
                    let writer = WalWriter::open_or_create(&path, /*tree_id=*/ 0)?;
                    (Some(Arc::new(Mutex::new(writer))), next_seq)
                }
            }
        } else {
            (None, 1u64)
        };

        Ok(Self {
            cfg,
            backend: bm,
            root_guid,
            rename_lock: Arc::new(Mutex::new(())),
            next_seq: Arc::new(AtomicU64::new(next_seq)),
            wal,
        })
    }

    /// Look up `key`. Returns the value bytes, or `None` if no leaf
    /// matches.
    ///
    /// **Zero-copy and lock-free against the writer lock**: pins
    /// each blob via the [`BufferManager`] and walks the cached
    /// buffer under a shared `RwLock` read guard. N readers on
    /// different blobs progress in parallel; readers on the same
    /// blob also progress in parallel via the read-half.
    ///
    /// Transparently follows `BlobNode` crossings — the lookup may
    /// span multiple blobs when the tree has been split by
    /// spillover.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let padded = pad_key(key);
        engine::lookup_multi(&self.backend, self.root_guid, &padded)
    }

    /// Insert or replace `(key, value)`. Returns the previous value
    /// if the key already existed.
    ///
    /// Walks across [`BlobNode`] crossings. When any blob hits
    /// `AllocError::OutOfSpace`, the walker automatically migrates
    /// a subtree out via `splitBlob` and retries — so trees may
    /// grow well past the 512 KB single-blob limit without caller
    /// involvement.
    ///
    /// Mutates the BM-pinned root buffer in place under an
    /// exclusive write guard; the durable write to the inner
    /// backend happens when `flush_on_write` is `true` (the
    /// default) via [`BufferManager::commit`]. Newly-created child
    /// blobs are **always** written through the backend at the
    /// moment of spillover, so crash-recovery never finds a
    /// dangling `BlobNode` pointing at nothing.
    ///
    /// [`BlobNode`]: crate::layout::BlobNode
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<Option<Vec<u8>>> {
        let padded = pad_key(key);
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        // Concurrent writers are serialised by the per-blob
        // `HybridLatch` (root blob always taken exclusive); no
        // Tree-wide writer mutex needed.
        let outcome = engine::insert_multi(&self.backend, self.root_guid, &padded, value, seq)?;

        // Durability model: the WAL flush is the **per-op
        // boundary**. The BM-cached blob image stays in memory
        // until `Tree::checkpoint`. A crash recovers by replaying
        // every record past the last checkpoint onto the blob
        // image that was durable at checkpoint time.
        //
        // The previous "commit BM per op" design double-counted
        // durability and made replay non-idempotent for `rename`
        // (the WAL re-inserted the source key on top of an
        // already-renamed blob).
        if let Some(wal) = &self.wal {
            let mut w = wal.lock().unwrap();
            // Fast-path: append the Insert record directly from
            // borrowed refs — skips the `TxnOp::Insert` enum's
            // three `Vec` clones (key, value, prev_value).
            w.append_insert(seq, 0, key, value, outcome.previous.as_deref())?;
            if self.cfg.wal_sync_on_commit {
                w.flush()?;
            }
        } else if self.cfg.flush_on_write {
            // No WAL (memory mode, or backend supplied by user).
            // `flush_on_write` still pushes the BM root through
            // its `Backend`'s write-through path so callers that
            // want per-op durability against a custom backend
            // keep getting it.
            self.backend.commit(self.root_guid)?;
        }
        Ok(outcome.previous)
    }

    /// Remove `key`. Returns the value that was stored at `key`, or
    /// `None` if no leaf matched.
    ///
    /// Walks across [`BlobNode`] crossings. When a child blob
    /// becomes empty as a result of the erase, its parent's
    /// `BlobNode` is freed and the orphaned child blob is dropped
    /// from cache + the inner backend — no GC pass needed.
    ///
    /// [`BlobNode`]: crate::layout::BlobNode
    pub fn delete(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let padded = pad_key(key);
        // No Tree-wide lock — per-blob `HybridLatch` exclusive on
        // the root serialises concurrent `delete` / `put` calls.
        let outcome = engine::erase_multi(&self.backend, self.root_guid, &padded)?;

        if let Some(wal) = &self.wal {
            if let Some(prev) = &outcome.previous {
                let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
                let mut w = wal.lock().unwrap();
                // Fast-path: skips the `TxnOp::Erase` enum's
                // two `Vec` clones (key, value).
                w.append_erase(seq, 0, key, prev)?;
                if self.cfg.wal_sync_on_commit {
                    w.flush()?;
                }
            }
            // No-op delete (key wasn't there) is not logged.
        } else if self.cfg.flush_on_write {
            self.backend.commit(self.root_guid)?;
        }
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
    /// Atomic with respect to other renames (`rename_lock` is held
    /// for the whole sequence). Concurrent `put`/`delete` on
    /// disjoint subtrees are not blocked. Stage 5 (WAL) will swap
    /// the multi-step path for a dedicated `RenameTxnOp` so the
    /// child-blob writes between erase and insert commit as one
    /// journal record.
    pub fn rename(&self, src: &[u8], dst: &[u8], force: bool) -> Result<()> {
        let src_padded = pad_key(src);
        let dst_padded = pad_key(dst);

        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let _r = self.rename_lock.lock().unwrap();

        // Probe src across all blobs — zero-copy via BM pin.
        let Some(value) = engine::lookup_multi(&self.backend, self.root_guid, &src_padded)? else {
            return Err(Error::NotFound);
        };

        // Same key? No-op (seq is already bumped).
        if src == dst {
            return Ok(());
        }

        // Probe dst across all blobs unless overwrite is allowed.
        if !force && engine::lookup_multi(&self.backend, self.root_guid, &dst_padded)?.is_some() {
            return Err(Error::DstExists);
        }

        // erase(src) + insert(dst, value). Both walk through
        // `BlobNode` crossings and commit any touched child blobs.
        engine::erase_multi(&self.backend, self.root_guid, &src_padded)?;
        engine::insert_multi(&self.backend, self.root_guid, &dst_padded, &value, seq)?;

        if let Some(wal) = &self.wal {
            let mut w = wal.lock().unwrap();
            // Fast-path: skips the `TxnOp::RenameObject` enum's
            // two `Vec` clones (src_key, dst_key).
            w.append_rename_object(seq, 0, src, dst, force)?;
            if self.cfg.wal_sync_on_commit {
                w.flush()?;
            }
        } else if self.cfg.flush_on_write {
            self.backend.commit(self.root_guid)?;
        }
        Ok(())
    }

    /// Make every previously-applied mutation durable and trim
    /// the WAL.
    ///
    /// Sequence:
    /// 1. Flush every buffered WAL record (`sync_data` on the log).
    /// 2. Write the BM-cached root blob through to the inner
    ///    backend.
    /// 3. `flush` the backend (`fdatasync` on persistent; no-op on
    ///    memory).
    /// 4. Truncate the WAL — its records are now redundant with
    ///    the freshly-durable blob image, so the next replay
    ///    starts from an empty log.
    ///
    /// `flush_on_write = false` callers rely on this to make
    /// batched writes survive a crash.
    pub fn checkpoint(&self) -> Result<()> {
        if let Some(wal) = &self.wal {
            wal.lock().unwrap().flush()?;
        }
        self.backend.commit(self.root_guid)?;
        self.backend.flush()?;
        if let Some(wal) = &self.wal {
            wal.lock().unwrap().truncate()?;
        }
        Ok(())
    }

    /// Snapshot per-blob and aggregate counters for every blob
    /// reachable from the root.
    ///
    /// Each blob is pinned + read under a single shared guard, so
    /// stats never block ongoing reads and only contend with writers
    /// on a blob-by-blob basis. Returned counters are a consistent
    /// snapshot of each individual blob but the aggregate is **not**
    /// linearised across blobs — a concurrent writer mid-traversal
    /// can shift one blob's counters before another's are read.
    /// Acceptable for observability; use [`Tree::checkpoint`] first
    /// if you need a quiescent snapshot.
    pub fn stats(&self) -> Result<TreeStats> {
        let guids = engine::collect_blob_guids(&self.backend, self.root_guid)?;
        let mut blobs: Vec<BlobStats> = Vec::with_capacity(guids.len());
        let mut total_space_used: u64 = 0;
        let mut total_gap_space: u64 = 0;
        let mut total_slots: u64 = 0;
        let mut total_compactions: u64 = 0;
        let mut total_tombstones: u64 = 0;
        for guid in &guids {
            let pin = self.backend.pin(*guid)?;
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let h = frame.header();
            let s = BlobStats {
                guid: *guid,
                space_used: h.space_used,
                gap_space: h.gap_space,
                num_slots: h.num_slots,
                num_ext_blobs: h.num_ext_blobs,
                compact_times: h.compact_times,
                tombstone_leaf_cnt: h.tombstone_leaf_cnt,
            };
            total_space_used += u64::from(s.space_used);
            total_gap_space += u64::from(s.gap_space);
            total_slots += u64::from(s.num_slots);
            total_compactions += u64::from(s.compact_times);
            total_tombstones += u64::from(s.tombstone_leaf_cnt);
            blobs.push(s);
        }
        Ok(TreeStats {
            blob_count: blobs.len() as u32,
            total_space_used,
            total_gap_space,
            total_slots,
            total_compactions,
            total_tombstones,
            blobs,
        })
    }

    /// Compact every blob reachable from the root in place.
    ///
    /// Per-blob: takes an exclusive guard, runs
    /// [`crate::engine::compact_blob`] (rebuild dropping orphans /
    /// tombstones and reclaiming bump-area waste), and commits the
    /// rebuilt buffer through the BM. `compact_times` is bumped by
    /// one on each blob; `tombstone_leaf_cnt` resets to zero (the
    /// leaves it counted are dropped from the new image — wired in
    /// the tombstone work).
    ///
    /// Does **not** fsync the backend or touch the WAL — compaction
    /// is logically idempotent (the post-compact tree is observationally
    /// identical to the pre-compact one), so a crash mid-compact just
    /// means the next run re-does the work. Call [`Tree::checkpoint`]
    /// after if you want the rebuilt blobs durable on disk.
    pub fn compact(&self) -> Result<()> {
        let guids = engine::collect_blob_guids(&self.backend, self.root_guid)?;
        for guid in guids {
            let pin = self.backend.pin(guid)?;
            {
                let mut guard = pin.write();
                engine::compact_blob(&mut guard)?;
            }
            drop(pin);
            self.backend.commit(guid)?;
        }
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

/// Replay `path` onto the BM-cached blobs and return the
/// `next_seq` the tree should resume from.
///
/// Each record's logical mutation is re-applied through the
/// engine. Structural ops (`Split` / `Merge` / `Compact`) are
/// already reflected in the blob image on disk, so they're
/// no-ops during replay; `MemMarker` is the explicit
/// post-replay reconciliation marker and is ignored.
///
/// `RenameObject` is rebuilt as the same erase + insert it ran
/// originally. `Rename` (cross-tree) doesn't apply to the
/// single-tree v0.1 surface and is rejected. `NewTree` / `RmTree`
/// are also future-multi-tenant ops, ignored here.
fn replay_wal(path: &std::path::Path, bm: &Arc<BufferManager>, root_guid: BlobGuid) -> Result<u64> {
    let mut highest = 0u64;
    let _ = replay(path, |op, seq, _off| {
        match op {
            TxnOp::Insert { key, value, .. } => {
                let padded = pad_key(key);
                engine::insert_multi(bm, root_guid, &padded, value, seq)?;
            }
            TxnOp::Erase { key, .. } => {
                let padded = pad_key(key);
                engine::erase_multi(bm, root_guid, &padded)?;
            }
            TxnOp::RenameObject {
                src_key,
                dst_key,
                force,
                ..
            } => {
                let src_padded = pad_key(src_key);
                let dst_padded = pad_key(dst_key);
                if engine::lookup_multi(bm, root_guid, &src_padded)?.is_none() {
                    // Already reconciled in a prior replay pass —
                    // skip.
                    return Ok(());
                }
                if !force && engine::lookup_multi(bm, root_guid, &dst_padded)?.is_some() {
                    return Ok(());
                }
                let value = engine::lookup_multi(bm, root_guid, &src_padded)?.unwrap_or_default();
                engine::erase_multi(bm, root_guid, &src_padded)?;
                engine::insert_multi(bm, root_guid, &dst_padded, &value, seq)?;
            }
            // Structural / multi-tenant / marker variants don't
            // affect logical state at v0.1's single-tree surface.
            TxnOp::Split { .. }
            | TxnOp::Merge { .. }
            | TxnOp::Compact { .. }
            | TxnOp::Rename { .. }
            | TxnOp::NewTree { .. }
            | TxnOp::RmTree { .. }
            | TxnOp::MemMarker { .. } => {}
        }
        highest = highest.max(seq);
        Ok(())
    })?;
    // After commit, the blob image is durable; we still want the
    // next allocated seq to be strictly greater than anything
    // ever seen in the log.
    Ok(highest + 1)
}
