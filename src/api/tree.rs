//! Public `Tree` type — the main user-facing API.
//!
//! ## Internal key encoding
//!
//! The walker treats every user-supplied point key as if it had a
//! trailing `\0` byte. This is a standard ART trick to resolve the
//! "strict prefix" case where one key (e.g. `"abc"`) is a prefix
//! of another (e.g. `"abcdef"`): the terminator guarantees the two
//! keys diverge somewhere inside the radix tree (at the `\0` vs
//! `'d'` byte in this example). The terminator is virtual during
//! descent and is materialised only when a new leaf key is written.
//!

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::config::{Storage, TreeConfig};
use super::errors::{Error, Result};
use super::stats::{BlobStats, CheckpointerStats, JournalStats, RouteCacheStats, TreeStats};
use super::view::View;
use crate::concurrency::{CommitGate, MaintenanceGate};
use crate::engine;
use crate::engine::{KeyRangeBuilder, KeyRangeEntry, RangeBuilder};
use crate::journal::codec::{
    encode_erase_record, encode_insert_record, encode_rename_object_record,
    encoded_erase_record_len, encoded_insert_record_len, encoded_rename_object_record_len,
    BatchEncoder, RECORD_FOOTER_SIZE, RECORD_HEADER_SIZE,
};
use crate::journal::group_commit::Journal;
use crate::journal::reader::replay;
use crate::journal::wal_op::WalOp;
use crate::layout::{BlobGuid, DATA_AREA_START, PAGE_SIZE, ROOT_BLOB_GUID};
use crate::store::blob_store::{BlobStore, FileBlobStore, MemoryBlobStore};
use crate::store::{BlobFrame, BlobFrameRef, BufferManager, CachedBlob, WriteThroughEntry};

use super::atomic::{AtomicBatch, BatchOp, Record, RecordVersion};

const ONLINE_COMPACT_BLOB_BUDGET: usize = 256;
const ONLINE_MERGE_PARENT_BUDGET: usize = 256;
const SHAPE_UNDERFILLED_CHILD_FILL_PER_MILLE: u32 = 350;
const SHAPE_OVERFULL_CHILD_FILL_PER_MILLE: u32 = 850;

type BatchOverlay = HashMap<Vec<u8>, Option<Record>>;

struct BlobStatsAggregate {
    blobs: Vec<BlobStats>,
    total_space_used: u64,
    total_gap_space: u64,
    total_slots: u64,
    total_compactions: u64,
    total_tombstones: u64,
    total_blob_edges: u64,
    leaf_blob_count: u32,
    max_blob_depth: u32,
    total_blob_depth: u64,
    max_blob_fill_per_mille: u32,
    underfilled_child_blobs: u32,
    overfull_child_blobs: u32,
}

fn blob_fill_per_mille(space_used: u32, blob_data_capacity: u64) -> u32 {
    if blob_data_capacity == 0 {
        return 0;
    }
    let data_used = u64::from(space_used).saturating_sub(DATA_AREA_START as u64);
    ((data_used.saturating_mul(1000)) / blob_data_capacity) as u32
}

/// An `holt` tree — your handle to one metadata store.
///
/// Clone the handle to share the same backing store: the
/// internal `BufferManager` is held via `Arc`.
///
/// ## Concurrency
///
/// - **Point reads** (`get`) take the shared maintenance gate,
///   then run against `HybridLatch::read_optimistic` — they capture
///   each blob's latch version, read the bytes, then `validate()`.
///   Restarts from the root on a torn read. Never blocks foreground
///   writers and never block each other.
/// - **Range reads** (`range`, `scan`, `range_keys`,
///   `scan_keys`) use a versioned cursor. Each cursor frame
///   records the blob content version it was built from; if an
///   interleaved writer changes a frame, the iterator discards its
///   stack and performs a marker-aware seek from the last emitted
///   key / delimiter lower bound.
/// - **Writes** (`put`, `delete`) hold the per-blob `HybridLatch`
///   exclusively for the blobs they touch. Persistent trees enter
///   the writer-shared `commit_gate` while publishing dirty state
///   and the journal record. If `TreeConfig::wal_sync` is set,
///   writes wait for the journal worker after leaving the gate.
/// - **Maintenance** (`compact`, background merge) takes short
///   exclusive windows on `maintenance_gate` while folding/deleting
///   cross-blob edges. Blob-local compaction runs on the shared
///   side under per-blob latches. Foreground reads and writers
///   enter the shared side around tree traversal, so ordinary
///   operations stay mutually concurrent.
/// - **`rename`** holds `rename_lock` (a `Mutex<()>` scoped to
///   rename only) so its multi-step
///   `lookup → erase → insert` appears atomic to other writers.
///   `put` / `delete` / `get` never take `rename_lock`.
#[derive(Clone)]
pub struct Tree {
    cfg: TreeConfig,
    store: Arc<BufferManager>,
    /// GUID of the blob holding the tree root. Currently a fixed
    /// sentinel; a multi-tenant manifest could allocate per-tree
    /// root GUIDs in the future.
    root_guid: BlobGuid,
    /// Cached pin on the root blob — held for the life of this
    /// `Tree` handle so every `get` / `put` / `delete` / `rename`
    /// skips the `BufferManager`'s `Mutex<HashMap>` lookup on
    /// the root hop. Cross-blob descents still pin children
    /// through the BM as normal.
    root_pin: Arc<CachedBlob>,
    /// Serialises **only `rename`** — the multi-step
    /// `lookup_multi_with(src)` + `erase_multi(src)` + `insert_multi(dst)`
    /// must appear atomic to other writers. `put` / `delete` /
    /// `get` never take this lock; they coordinate via the
    /// per-blob `HybridLatch` inside the BM.
    rename_lock: Arc<Mutex<()>>,
    /// Parent-validated route cache for path-shaped large trees.
    /// Entries cache prefix anchors at BlobNode crossings and are
    /// validated against the parent blob's latch version before
    /// use.
    route_cache: Arc<engine::RouteCache>,
    /// Tree-wide structural-maintenance gate.
    ///
    /// Foreground read and mutation paths enter the shared side
    /// while they may cross `BlobNode` boundaries. `compact()` and
    /// the background merge pass enter the exclusive side only
    /// around folding a child blob back into its parent and queuing
    /// the child for delete.
    /// Point reads participate too: per-blob optimistic validation
    /// handles in-place rewrites, while the maintenance gate keeps
    /// a merge pass from deleting a child blob after a reader has
    /// observed the parent `BlobNode` but before it pins the child.
    maintenance_gate: Arc<MaintenanceGate>,
    /// Monotonically-increasing sequence stamped on every record.
    /// On open the tree replays the WAL and resumes at
    /// `highest_seq + 1`.
    next_seq: Arc<AtomicU64>,
    /// Writer-shared / checkpoint-exclusive publish barrier for
    /// persistent mode. Foreground writers can mutate disjoint
    /// blobs concurrently, but checkpoint waits until every
    /// admitted writer has published its dirty state and journal
    /// record before cloning bytes for store write-through.
    commit_gate: Arc<CommitGate>,
    /// Group-commit WAL worker — `Some` for persistent trees,
    /// `None` for memory trees.
    journal: Option<Arc<Journal>>,
    /// Direct-mapped cache for short hot key-only prefix scans.
    /// Conservatively invalidated by `next_seq`, so every write
    /// makes old entries miss.
    prefix_list_cache: Arc<engine::PrefixListCache>,
    /// Background checkpointer handle. `Some` iff
    /// `cfg.checkpoint.enabled`. Shared via `Arc` so the thread
    /// shuts down on the **last** `Tree` clone's drop, not the
    /// first. Exposed to `Tree::stats` for counter readout.
    checkpointer: Option<Arc<crate::checkpoint::Checkpointer>>,
}

impl std::fmt::Debug for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tree")
            .field("storage", &self.cfg.storage)
            .field("root_guid", &self.root_guid)
            .finish_non_exhaustive()
    }
}

fn encoded_batch_record_len(ops: &[BatchOp]) -> usize {
    let body_prefix_len = 8 + 4; // tree_id + inner_count
    let mut len = RECORD_HEADER_SIZE + body_prefix_len + RECORD_FOOTER_SIZE;
    let mut i = 0usize;
    while i < ops.len() {
        if let Some((key, value)) = batch_insert_parts(&ops[i]) {
            let run = same_shape_insert_run_len(ops, i);
            if run > 1 {
                len += 1 + 8 + 4 + 4 + 4 + run * (key.len() + value.len());
            } else {
                len += 1 + 8 + 4 + key.len() + 4 + value.len();
            }
            i += run;
            continue;
        }
        len += match &ops[i] {
            BatchOp::Delete { key } | BatchOp::DeleteIfVersion { key, .. } => 1 + 8 + 4 + key.len(),
            BatchOp::Rename { src, dst, .. } => 1 + 8 + 4 + src.len() + 4 + dst.len() + 1,
            BatchOp::AssertVersion { .. } | BatchOp::AssertPrefixEmpty { .. } => 0,
            BatchOp::Put { .. } | BatchOp::PutIfAbsent { .. } | BatchOp::CompareAndPut { .. } => {
                unreachable!("insert-like ops handled above")
            }
        };
        i += 1;
    }
    len
}

fn batch_insert_parts(op: &BatchOp) -> Option<(&[u8], &[u8])> {
    match op {
        BatchOp::Put { key, value }
        | BatchOp::PutIfAbsent { key, value }
        | BatchOp::CompareAndPut { key, value, .. } => Some((key, value)),
        BatchOp::Delete { .. }
        | BatchOp::DeleteIfVersion { .. }
        | BatchOp::AssertVersion { .. }
        | BatchOp::AssertPrefixEmpty { .. }
        | BatchOp::Rename { .. } => None,
    }
}

fn same_shape_insert_run_len(ops: &[BatchOp], start: usize) -> usize {
    let Some((first_key, first_value)) = batch_insert_parts(&ops[start]) else {
        return 0;
    };
    let mut end = start + 1;
    while end < ops.len() {
        match batch_insert_parts(&ops[end]) {
            Some((key, value))
                if key.len() == first_key.len() && value.len() == first_value.len() =>
            {
                end += 1;
            }
            _ => break,
        }
    }
    end - start
}

impl Tree {
    /// Open a tree using the supplied configuration.
    ///
    /// `TreeConfig::new("/path")` opens a file-backed tree at
    /// `"/path"` (the default). `TreeConfig::memory()` opens an
    /// in-memory tree.
    ///
    /// holt is Unix-only — the file store uses `O_DIRECT`
    /// on Linux and `F_NOCACHE` on macOS. Building the crate on
    /// Windows fails at compile time (see the platform stance in
    /// `ROADMAP.md`).
    pub fn open(cfg: TreeConfig) -> Result<Self> {
        let store: Arc<dyn BlobStore> = match &cfg.storage {
            Storage::Memory => Arc::new(MemoryBlobStore::new()),
            Storage::File { dir } => {
                #[cfg(all(target_os = "linux", feature = "io-uring"))]
                {
                    Arc::new(FileBlobStore::open_with_buffer_pool_hint(
                        dir,
                        cfg.buffer_pool_size,
                    )?)
                }
                #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
                {
                    Arc::new(FileBlobStore::open(dir)?)
                }
            }
        };
        // The auto-managed store earns automatic WAL coverage.
        Self::open_inner(cfg, store, /*attach_wal=*/ true)
    }

    /// Open a tree with a caller-supplied [`BlobStore`].
    ///
    /// **No WAL is attached.** The caller's store has its own
    /// notion of durability (or is intentionally volatile —
    /// e.g. a `MemoryBlobStore` standing in for a real one in a
    /// test); holt stays out of that decision. If you want a
    /// WAL'd file-backed tree, use [`Tree::open`] with a
    /// `Storage::File` config.
    ///
    /// The supplied store is **transparently wrapped** with a
    /// `BufferManager` of `cfg.buffer_pool_size` blobs.
    /// `BufferManager` owns the in-memory blob cache; the walker
    /// pins blobs from it for both reads and writes — no separate
    /// root buffer in `Tree`.
    ///
    /// If the store doesn't yet contain a root blob, initialises
    /// an empty one and writes it through, flushing before
    /// returning.
    pub fn open_with_blob_store(cfg: TreeConfig, store: Arc<dyn BlobStore>) -> Result<Self> {
        Self::open_inner(cfg, store, /*attach_wal=*/ false)
    }

    fn open_inner(cfg: TreeConfig, store: Arc<dyn BlobStore>, attach_wal: bool) -> Result<Self> {
        let bm: Arc<BufferManager> = Arc::new(BufferManager::new(store, cfg.buffer_pool_size));
        let root_guid = ROOT_BLOB_GUID;
        if !bm.has_blob(root_guid)? {
            // Seed an empty root blob and write it through.
            let mut scratch = bm.alloc_blob_buf_zeroed();
            BlobFrame::init(scratch.as_mut_slice(), root_guid)?;
            bm.write_blob(root_guid, &scratch)?;
            bm.flush()?;
        }

        // File-backed trees keep a WAL alongside the data file.
        // Replay every durable record onto the BM-cached blob
        // image before exposing the tree to callers: the on-disk
        // blob lags the WAL between the last `Tree::checkpoint`
        // and now, so the WAL is the source of truth for any op
        // committed via `memory_flush_on_write = true`.
        let (journal, next_seq) = if attach_wal {
            match cfg.wal_path() {
                None => (None, 1u64),
                Some(path) => {
                    let next_seq = if path.exists() {
                        replay_wal(&path, &bm, root_guid)?
                    } else {
                        1
                    };
                    let journal = Journal::open_or_create(&path, /*tree_id=*/ 0)?;
                    (Some(Arc::new(journal)), next_seq)
                }
            }
        } else {
            (None, 1u64)
        };

        // Hot-path: pin the root blob once for the lifetime of
        // this `Tree` handle so every subsequent `get` / `put` /
        // `delete` / `rename` skips the BufferManager's per-pin
        // `Mutex<HashMap>` lookup on the root hop. `Tree::clone`
        // shares the pin via `Arc::clone`.
        let root_pin = bm.pin(root_guid)?;

        // Shared structural gate for foreground writers, manual
        // compact, and the background merge pass.
        let maintenance_gate = Arc::new(MaintenanceGate::new());
        let commit_gate = Arc::new(CommitGate::new());

        // Spawn the background checkpointer if opted-in.
        // `Checkpointer::spawn` returns `None` for disabled
        // configs, so the `Option` chain stays clean.
        let checkpointer = crate::checkpoint::Checkpointer::spawn(
            Arc::clone(&bm),
            journal.clone(),
            Arc::clone(&maintenance_gate),
            Arc::clone(&commit_gate),
            cfg.checkpoint.clone(),
        )
        .map(Arc::new);

        Ok(Self {
            cfg,
            store: bm,
            root_guid,
            root_pin,
            rename_lock: Arc::new(Mutex::new(())),
            route_cache: Arc::new(engine::RouteCache::new()),
            maintenance_gate,
            next_seq: Arc::new(AtomicU64::new(next_seq)),
            commit_gate,
            journal,
            prefix_list_cache: Arc::new(engine::PrefixListCache::new()),
            checkpointer,
        })
    }

    /// Look up `key`. Returns the value bytes, or `None` if no leaf
    /// matches.
    ///
    /// Pays one allocation + memcpy per hit; on a miss returns
    /// `Ok(None)` with no allocation. The walker itself reads
    /// cached blobs optimistically and restarts from the root when
    /// a concurrent writer invalidates its snapshot.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let _maintenance = self.maintenance_gate.enter_shared();
        self.lookup_record_unlocked(key)
            .map(|record| record.map(|record| record.value))
    }

    /// Look up `key` and return both value bytes and the current
    /// conditional-write version token.
    ///
    /// This is the preferred read before a compare-and-set update:
    /// it avoids the two-lookup `get()` + `get_version()` pattern.
    pub fn get_record(&self, key: &[u8]) -> Result<Option<Record>> {
        let _maintenance = self.maintenance_gate.enter_shared();
        self.lookup_record_unlocked(key)
    }

    /// Return the current version token for `key`.
    ///
    /// The token is the leaf sequence attached to the live record
    /// and is intended only for conditional writes
    /// ([`Self::compare_and_put`] / [`Self::delete_if_version`]).
    /// It is not an MVCC timestamp and cannot be used to read old
    /// values.
    pub fn get_version(&self, key: &[u8]) -> Result<Option<RecordVersion>> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let search = engine::SearchKey::user(key);
        engine::lookup_multi_with(
            &self.store,
            &self.root_pin,
            Some(&self.route_cache),
            search,
            |hit| RecordVersion::new(hit.seq),
        )
    }

    fn lookup_record_unlocked(&self, key: &[u8]) -> Result<Option<Record>> {
        let search = engine::SearchKey::user(key);
        engine::lookup_multi_with(
            &self.store,
            &self.root_pin,
            Some(&self.route_cache),
            search,
            |hit| Record {
                value: hit.value.to_vec(),
                version: RecordVersion::new(hit.seq),
            },
        )
    }

    /// Insert or replace `(key, value)`. Returns `Ok(())`.
    ///
    /// Blind hot path: the walker does **not** read or clone the
    /// existing leaf's value on a same-key update.
    ///
    /// Walks across `BlobNode` crossings. When any blob hits
    /// `AllocError::OutOfSpace`, the walker automatically migrates
    /// a subtree out via `splitBlob` and retries — so trees may
    /// grow well past the 512 KB single-blob limit without caller
    /// involvement.
    ///
    /// Mutates the BM-pinned root buffer in place under an
    /// exclusive write guard. Cross-blob mutations stage their
    /// changes via `mark_dirty` / `install_new_blob`; the durable
    /// write to the inner store happens when the WAL record
    /// covering this op is on disk — driven either by the
    /// background checkpoint round or by [`Tree::checkpoint`].
    /// Per-op `memory_flush_on_write` mode drains the dirty set
    /// inline after the WAL append.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.put_inner_conditional(key, value, engine::InsertCondition::Always)
            .map(|_| ())
    }

    /// Insert `(key, value)` only when `key` has no live record.
    ///
    /// Returns `Ok(true)` when the value was inserted and `Ok(false)`
    /// when a live value already existed. The existence check and
    /// insert happen under the target blob's exclusive latch.
    pub fn put_if_absent(&self, key: &[u8], value: &[u8]) -> Result<bool> {
        self.put_inner_conditional(key, value, engine::InsertCondition::IfAbsent)
            .map(|outcome| outcome.mutated)
    }

    /// Replace `(key, value)` only when the live record currently
    /// carries `expected_version`.
    ///
    /// Returns `Ok(false)` if the key is missing, tombstoned, or
    /// has been updated since the caller obtained the version.
    pub fn compare_and_put(
        &self,
        key: &[u8],
        expected_version: RecordVersion,
        value: &[u8],
    ) -> Result<bool> {
        self.put_inner_conditional(
            key,
            value,
            engine::InsertCondition::IfVersion(expected_version.as_u64()),
        )
        .map(|outcome| outcome.mutated)
    }

    fn put_inner_conditional(
        &self,
        key: &[u8],
        value: &[u8],
        condition: engine::InsertCondition,
    ) -> Result<engine::InsertOutcome> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let search = engine::SearchKey::user(key);
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

        let (outcome, journal_ack) = if let Some(journal) = &self.journal {
            let _commit = self.commit_gate.enter_writer();
            let outcome = engine::insert_multi_conditional(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                search,
                value,
                seq,
                condition,
            )?;
            if outcome.mutated {
                if outcome.root_dirty {
                    self.store
                        .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
                }
                let mut record =
                    journal.record_buffer(encoded_insert_record_len(key.len(), value.len()));
                encode_insert_record(&mut record, seq, 0, key, value);
                let ack = journal.submit(record, self.cfg.wal_sync)?;
                (outcome, ack)
            } else {
                (outcome, None)
            }
        } else {
            // No WAL — no journal/checkpoint publish boundary to
            // race with. `memory_flush_on_write` (if set)
            // flushes dirty + pending-delete sets inline before
            // returning.
            let outcome = engine::insert_multi_conditional(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                search,
                value,
                seq,
                condition,
            )?;
            if outcome.root_dirty {
                self.store
                    .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
            }
            if outcome.mutated && self.cfg.memory_flush_on_write {
                self.flush_dirty_inline()?;
                self.flush_pending_deletes_inline()?;
            }
            (outcome, None)
        };
        if let Some(ack) = journal_ack {
            ack.wait()?;
        }
        Ok(outcome)
    }

    /// Remove `key`. Returns `Ok(true)` if a leaf was removed,
    /// `Ok(false)` if no leaf matched.
    ///
    /// Blind hot path: the walker does **not** read or clone the
    /// existing leaf's value before tombstoning it.
    ///
    /// Walks across `BlobNode` crossings. Child-local mutations
    /// are staged through the BM dirty set; any conservative
    /// fallback that unlinks a child blob queues the manifest
    /// delete through the same W2D-safe pending-delete protocol.
    pub fn delete(&self, key: &[u8]) -> Result<bool> {
        self.delete_inner_conditional(key, engine::EraseCondition::Always)
            .map(|outcome| outcome.mutated)
    }

    /// Remove `key` only when the live record currently carries
    /// `expected_version`.
    ///
    /// Returns `Ok(false)` if the key is missing, already
    /// tombstoned, or has been updated since the caller obtained
    /// the version.
    pub fn delete_if_version(&self, key: &[u8], expected_version: RecordVersion) -> Result<bool> {
        self.delete_inner_conditional(
            key,
            engine::EraseCondition::IfVersion(expected_version.as_u64()),
        )
        .map(|outcome| outcome.mutated)
    }

    fn delete_inner_conditional(
        &self,
        key: &[u8],
        condition: engine::EraseCondition,
    ) -> Result<engine::EraseOutcome> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let search = engine::SearchKey::user(key);
        // Pre-allocate the seq before the walker descends so any
        // child blob the walker touches can `mark_dirty(child, seq)`
        // — invariant W2D (see `BufferManager` module docs) demands
        // a single seq for the whole op across all blobs it dirties.
        // A no-op delete (key absent) still burns the seq; that's
        // fine — `next_seq` is monotonic and the unused seq doesn't
        // appear in any WAL record or dirty entry.
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

        let (outcome, journal_ack) = if let Some(journal) = &self.journal {
            let _commit = self.commit_gate.enter_writer();
            let outcome = engine::erase_multi_conditional(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                search,
                seq,
                condition,
            )?;
            if outcome.mutated {
                // Only mark the root if the root blob changed.
                // Cross-blob erases mark their child blob inside
                // the walker; absent-key no-ops mark nothing.
                if outcome.root_dirty {
                    self.store
                        .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
                }
                let mut record = journal.record_buffer(encoded_erase_record_len(key.len()));
                encode_erase_record(&mut record, seq, 0, key);
                let ack = journal.submit(record, self.cfg.wal_sync)?;
                (outcome, ack)
            } else {
                (outcome, None)
            }
            // No-op delete (key wasn't there) is not logged.
        } else {
            let outcome = engine::erase_multi_conditional(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                search,
                seq,
                condition,
            )?;
            if outcome.mutated && outcome.root_dirty {
                self.store
                    .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
            }
            if outcome.mutated && self.cfg.memory_flush_on_write {
                // Flush every blob the walker touched (root + any
                // children) — no WAL means this is the sole
                // durability path. snapshot_dirty drains all
                // entries; we commit each through the store.
                self.flush_dirty_inline()?;
                // Plus drain any deferred deletes queued by a
                // conservative parent-unlink path.
                self.flush_pending_deletes_inline()?;
            }
            (outcome, None)
        };
        if let Some(ack) = journal_ack {
            ack.wait()?;
        }
        Ok(outcome)
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
    /// for the whole sequence). Concurrent `put` / `delete` on
    /// disjoint subtrees are not blocked. The op emits a single
    /// `RenameObject` WAL record so its erase + insert phases
    /// recover atomically on replay.
    pub fn rename(&self, src: &[u8], dst: &[u8], force: bool) -> Result<()> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let src_search = engine::SearchKey::user(src);
        let dst_search = engine::SearchKey::user(dst);

        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let _r = self.rename_lock.lock().unwrap();

        // Probe src across all blobs — zero-copy via BM pin.
        let Some(value) = engine::lookup_multi_with(
            &self.store,
            &self.root_pin,
            Some(&self.route_cache),
            src_search,
            |hit| hit.value.to_vec(),
        )?
        else {
            return Err(Error::NotFound);
        };

        // Same key? No-op (seq is already bumped).
        if src == dst {
            return Ok(());
        }

        // Probe dst across all blobs unless overwrite is allowed.
        if !force
            && engine::lookup_multi_with(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                dst_search,
                |_| (),
            )?
            .is_some()
        {
            return Err(Error::DstExists);
        }

        // W2D-strict protocol: walker + mark_dirty + journal
        // submission all happen under `commit_gate`. Sharing one
        // `seq` across both erase + insert phases keeps the rename
        // atomic from the dirty-tracking perspective — failing
        // halfway leaves a coherent partial-dirty set rather than
        // two separately-staged ops.
        let journal_ack = if let Some(journal) = &self.journal {
            let _commit = self.commit_gate.enter_writer();
            let erase_out = engine::erase_multi(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                src_search,
                seq,
            )?;
            let insert_out = engine::insert_multi(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                dst_search,
                &value,
                seq,
            )?;
            if erase_out.root_dirty || insert_out.root_dirty {
                self.store
                    .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
            }
            let mut record =
                journal.record_buffer(encoded_rename_object_record_len(src.len(), dst.len()));
            encode_rename_object_record(&mut record, seq, 0, src, dst, force);
            journal.submit(record, self.cfg.wal_sync)?
        } else {
            let erase_out = engine::erase_multi(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                src_search,
                seq,
            )?;
            let insert_out = engine::insert_multi(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                dst_search,
                &value,
                seq,
            )?;
            if erase_out.root_dirty || insert_out.root_dirty {
                self.store
                    .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
            }
            if self.cfg.memory_flush_on_write {
                // Walker may have dirtied child blobs across the
                // erase + insert sequence — drain the full set.
                // The erase half can also queue SubtreeGone deletes.
                self.flush_dirty_inline()?;
                self.flush_pending_deletes_inline()?;
            }
            None
        };
        if let Some(ack) = journal_ack {
            ack.wait()?;
        }
        Ok(())
    }

    /// Apply a batch of mutations under a single WAL record.
    ///
    /// The closure builds a [`AtomicBatch`] by calling its `put` /
    /// conditional write / `delete` / `rename` methods; on return,
    /// holt first validates every logical precondition, then applies
    /// the batch while holding the tree-wide mutation gate and emits
    /// **one** WAL record (`WalOp::Batch`) covering the sequence.
    ///
    /// ## Atomicity contract
    ///
    /// - **Logical atomicity**: yes. Missing rename sources,
    ///   destination collisions, and failed conditional guards are
    ///   detected before any walker mutation. A failing rename
    ///   returns `Err`; a failed conditional guard returns
    ///   `Ok(false)`. Neither publishes partial user mutations.
    /// - **Runtime visibility**: readers and writers are blocked
    ///   while the batch applies, so no concurrent operation can
    ///   observe an intermediate batch state.
    /// - **Crash atomicity**: yes. The single WAL record is the
    ///   recovery commit point; replay sees the whole batch or none.
    ///
    /// Returns `Ok(true)` when the batch committed, `Ok(false)` when
    /// a conditional guard failed, and `Err` for hard errors such as
    /// a missing rename source or store/journal failure.
    ///
    /// ## Example
    ///
    /// ```no_run
    /// # use holt::{Tree, TreeConfig};
    /// # let tree = Tree::open(TreeConfig::memory()).unwrap();
    /// tree.atomic(|batch| {
    ///     batch.put(b"a", b"1");
    ///     batch.put(b"b", b"2");
    ///     batch.delete(b"c");
    /// })
    /// .unwrap();
    /// ```
    pub fn atomic<F>(&self, build: F) -> Result<bool>
    where
        F: FnOnce(&mut AtomicBatch),
    {
        let mut batch = AtomicBatch::default();
        build(&mut batch);
        if batch.pending.is_empty() {
            return Ok(true);
        }
        self.apply_batch(batch.pending)
    }

    fn apply_batch(&self, pending: Vec<BatchOp>) -> Result<bool> {
        let _maintenance = self.maintenance_gate.enter_exclusive();
        let count = pending.iter().filter(|op| op.emits_wal()).count() as u64;
        // Reserve a contiguous seq range so each inner op's seq is
        // `base + mutating_index` and replay can derive it without
        // storing per-inner seqs in the body. Non-mutating prefix
        // assertions are not encoded in WAL and do not consume seqs.
        // Failed guard preflights may burn the range without
        // emitting a WAL record; `next_seq` is monotonic, not
        // gap-free.
        let base_seq = self.next_seq.fetch_add(count, Ordering::Relaxed);
        if !self.preflight_batch(&pending, base_seq)? {
            return Ok(false);
        }
        if count == 0 {
            return Ok(true);
        }

        // W2D-strict protocol: all inner ops' walker mutations +
        // `mark_dirty` calls, plus the single envelope WAL submit,
        // happen under `commit_gate` — see `Tree::put_inner_conditional`.
        if let Some(journal) = &self.journal {
            let ack = {
                let _commit = self.commit_gate.enter_writer();
                let mut record = journal.record_buffer(encoded_batch_record_len(&pending));
                let mut enc = BatchEncoder::begin(&mut record, base_seq, 0);
                self.apply_batch_walker_inline(&pending, base_seq, Some(&mut enc))?;
                let _n = enc.finish();
                journal.submit(record, self.cfg.wal_sync)?
            };
            if let Some(ack) = ack {
                ack.wait()?;
            }
        } else {
            self.apply_batch_walker_inline(&pending, base_seq, None)?;
            if self.cfg.memory_flush_on_write {
                // Every inner op may have dirtied root + cross-blob
                // children — drain the whole set rather than just
                // the root. Some fallback paths may also have
                // queued deferred deletes.
                self.flush_dirty_inline()?;
                self.flush_pending_deletes_inline()?;
            }
        }
        Ok(true)
    }

    fn preflight_batch(&self, pending: &[BatchOp], base_seq: u64) -> Result<bool> {
        if Self::batch_is_guard_free(pending) {
            Self::preflight_guard_free_batch(pending)?;
            return Ok(true);
        }

        let mut overlay = BatchOverlay::new();
        let mut seq_offset = 0u64;

        for op in pending {
            let seq = if op.emits_wal() {
                let seq = base_seq + seq_offset;
                seq_offset += 1;
                seq
            } else {
                base_seq + seq_offset
            };
            if !self.preflight_batch_op(&mut overlay, op, seq)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn preflight_batch_op(
        &self,
        overlay: &mut BatchOverlay,
        op: &BatchOp,
        seq: u64,
    ) -> Result<bool> {
        match op {
            BatchOp::Put { key, value } => {
                Self::validate_insert_shape(key, value)?;
                Self::overlay_put(overlay, key, value, seq);
            }
            BatchOp::PutIfAbsent { key, value } => {
                Self::validate_insert_shape(key, value)?;
                if self.projected_record(overlay, key)?.is_some() {
                    return Ok(false);
                }
                Self::overlay_put(overlay, key, value, seq);
            }
            BatchOp::CompareAndPut {
                key,
                expected,
                value,
            } => {
                Self::validate_insert_shape(key, value)?;
                match self.projected_record(overlay, key)? {
                    Some(record) if record.version == *expected => {
                        Self::overlay_put(overlay, key, value, seq);
                    }
                    _ => return Ok(false),
                }
            }
            BatchOp::Delete { key } => {
                overlay.insert(key.clone(), None);
            }
            BatchOp::DeleteIfVersion { key, expected } => {
                match self.projected_record(overlay, key)? {
                    Some(record) if record.version == *expected => {
                        overlay.insert(key.clone(), None);
                    }
                    _ => return Ok(false),
                }
            }
            BatchOp::AssertVersion { key, expected } => {
                match self.projected_record(overlay, key)? {
                    Some(record) if record.version == *expected => {}
                    _ => return Ok(false),
                }
            }
            BatchOp::AssertPrefixEmpty { prefix } => {
                if !self.projected_prefix_empty(overlay, prefix)? {
                    return Ok(false);
                }
            }
            BatchOp::Rename { src, dst, force } => {
                self.preflight_rename_op(overlay, src, dst, *force, seq)?;
            }
        }
        Ok(true)
    }

    fn preflight_rename_op(
        &self,
        overlay: &mut BatchOverlay,
        src: &[u8],
        dst: &[u8],
        force: bool,
        seq: u64,
    ) -> Result<()> {
        let Some(src_record) = self.projected_record(overlay, src)? else {
            return Err(Error::NotFound);
        };
        if src == dst {
            return Ok(());
        }
        if !force && self.projected_record(overlay, dst)?.is_some() {
            return Err(Error::DstExists);
        }
        Self::validate_insert_shape(dst, &src_record.value)?;
        overlay.insert(src.to_vec(), None);
        overlay.insert(
            dst.to_vec(),
            Some(Record {
                value: src_record.value,
                version: RecordVersion::new(seq),
            }),
        );
        Ok(())
    }

    fn overlay_put(overlay: &mut BatchOverlay, key: &[u8], value: &[u8], seq: u64) {
        overlay.insert(
            key.to_vec(),
            Some(Record {
                value: value.to_vec(),
                version: RecordVersion::new(seq),
            }),
        );
    }

    fn batch_is_guard_free(pending: &[BatchOp]) -> bool {
        pending
            .iter()
            .all(|op| matches!(op, BatchOp::Put { .. } | BatchOp::Delete { .. }))
    }

    fn preflight_guard_free_batch(pending: &[BatchOp]) -> Result<()> {
        for op in pending {
            if let BatchOp::Put { key, value } = op {
                Self::validate_insert_shape(key, value)?;
            }
        }
        Ok(())
    }

    fn projected_record(&self, overlay: &BatchOverlay, key: &[u8]) -> Result<Option<Record>> {
        match overlay.get(key) {
            Some(record) => Ok(record.clone()),
            None => self.lookup_record_unlocked(key),
        }
    }

    fn projected_prefix_empty(&self, overlay: &BatchOverlay, prefix: &[u8]) -> Result<bool> {
        if overlay
            .iter()
            .any(|(key, record)| record.is_some() && key.starts_with(prefix))
        {
            return Ok(false);
        }

        let mut iter = self.scan_keys(prefix).into_iter();
        while let Some(entry) = iter.next_unlocked().transpose()? {
            match entry {
                KeyRangeEntry::Key { key, .. } => match overlay.get(&key) {
                    Some(None) => {}
                    Some(Some(_)) | None => return Ok(false),
                },
                KeyRangeEntry::CommonPrefix(_) => return Ok(false),
            }
        }
        Ok(true)
    }

    fn validate_insert_shape(key: &[u8], value: &[u8]) -> Result<()> {
        let key_len = key.len().saturating_add(1);
        if key_len > u16::MAX as usize {
            return Err(Error::KeyTooLong { len: key_len });
        }
        if value.len() > u16::MAX as usize {
            return Err(Error::ValueTooLong { len: value.len() });
        }
        Ok(())
    }

    /// Walker-mutation + optional WAL-encode loop, shared between
    /// the WAL-on and WAL-off branches of [`Self::apply_batch`].
    ///
    /// When `enc` is `Some`, each successful walker mutation is
    /// followed by a `push_*` call on the encoder; when `None`, the
    /// walker mutations run alone (memory-only mode). Pulling the
    /// loop out keeps the two batch paths from drifting and makes
    /// the per-variant arms readable without macro tricks.
    #[allow(clippy::too_many_lines)] // one explicit match keeps batch apply order auditable
    fn apply_batch_walker_inline(
        &self,
        pending: &[BatchOp],
        base_seq: u64,
        mut enc: Option<&mut crate::journal::codec::BatchEncoder<'_>>,
    ) -> Result<()> {
        let mut seq_offset = 0u64;
        let mut i = 0usize;
        while i < pending.len() {
            if batch_insert_parts(&pending[i]).is_some() {
                let run_len = same_shape_insert_run_len(pending, i);
                let first_seq = base_seq + seq_offset;
                self.apply_batch_insert_run_walker(&pending[i..i + run_len], first_seq)?;
                seq_offset += run_len as u64;
                if let Some(enc) = enc.as_deref_mut() {
                    let (key, value) = batch_insert_parts(&pending[i])
                        .expect("insert run begins with insert-like op");
                    enc.push_insert_run(
                        0,
                        run_len,
                        key.len(),
                        value.len(),
                        pending[i..i + run_len]
                            .iter()
                            .map(|op| batch_insert_parts(op).expect("same-shape insert run")),
                    );
                }
                i += run_len;
                continue;
            }

            let op = &pending[i];
            let seq = if op.emits_wal() {
                let seq = base_seq + seq_offset;
                seq_offset += 1;
                seq
            } else {
                base_seq + seq_offset
            };
            match op {
                BatchOp::Put { .. }
                | BatchOp::PutIfAbsent { .. }
                | BatchOp::CompareAndPut { .. } => {
                    unreachable!("insert-like ops are handled by the run path");
                }
                BatchOp::Delete { key } => {
                    let search = engine::SearchKey::user(key);
                    let outcome = engine::erase_multi(
                        &self.store,
                        &self.root_pin,
                        Some(&self.route_cache),
                        search,
                        seq,
                    )?;
                    if outcome.mutated && outcome.root_dirty {
                        self.store
                            .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
                    }
                    // Batch replay derives per-inner seq from the
                    // inner index, so even no-op deletes are encoded
                    // to keep later record versions stable across
                    // crash/replay.
                    if let Some(enc) = enc.as_deref_mut() {
                        enc.push_erase(0, key);
                    }
                }
                BatchOp::DeleteIfVersion { key, expected } => {
                    let search = engine::SearchKey::user(key);
                    let outcome = engine::erase_multi_conditional(
                        &self.store,
                        &self.root_pin,
                        Some(&self.route_cache),
                        search,
                        seq,
                        engine::EraseCondition::IfVersion(expected.as_u64()),
                    )?;
                    if !outcome.mutated {
                        return Err(Error::Internal(
                            "atomic preflight missed delete_if_version guard",
                        ));
                    }
                    if outcome.root_dirty {
                        self.store
                            .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
                    }
                    if let Some(enc) = enc.as_deref_mut() {
                        enc.push_erase(0, key);
                    }
                }
                BatchOp::AssertVersion { .. } | BatchOp::AssertPrefixEmpty { .. } => {}
                BatchOp::Rename { src, dst, force } => {
                    self.apply_batch_rename_walker(src, dst, *force, seq)?;
                    if let Some(enc) = enc.as_deref_mut() {
                        enc.push_rename_object(0, src, dst, *force);
                    }
                }
            }
            i += 1;
        }
        Ok(())
    }

    fn apply_batch_insert_run_walker(&self, ops: &[BatchOp], first_seq: u64) -> Result<()> {
        let mut items = Vec::with_capacity(ops.len());
        for (idx, op) in ops.iter().enumerate() {
            let seq = first_seq + idx as u64;
            let (key, value, condition) = match op {
                BatchOp::Put { key, value } => (key, value, engine::InsertCondition::Always),
                BatchOp::PutIfAbsent { key, value } => {
                    (key, value, engine::InsertCondition::IfAbsent)
                }
                BatchOp::CompareAndPut {
                    key,
                    expected,
                    value,
                } => (
                    key,
                    value,
                    engine::InsertCondition::IfVersion(expected.as_u64()),
                ),
                BatchOp::Delete { .. }
                | BatchOp::DeleteIfVersion { .. }
                | BatchOp::AssertVersion { .. }
                | BatchOp::AssertPrefixEmpty { .. }
                | BatchOp::Rename { .. } => unreachable!("not an insert-like batch op"),
            };
            items.push(engine::InsertBatchItem::new(
                engine::SearchKey::user(key),
                value,
                seq,
                condition,
            ));
        }

        let mut applied = 0usize;
        while applied < items.len() {
            let outcome = engine::insert_multi_batch_conditional(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                &items[applied..],
            )?;
            if outcome.applied == 0 {
                return Err(Error::Internal("insert batch walker made no progress"));
            }
            if outcome.root_dirty {
                self.store.mark_dirty_cached(
                    self.root_guid,
                    items[applied].seq,
                    self.root_pin.as_ref(),
                );
            }
            applied += outcome.applied;
        }
        Ok(())
    }

    fn apply_batch_rename_walker(
        &self,
        src: &[u8],
        dst: &[u8],
        force: bool,
        seq: u64,
    ) -> Result<()> {
        let src_search = engine::SearchKey::user(src);
        let dst_search = engine::SearchKey::user(dst);
        let Some(value) = engine::lookup_multi_with(
            &self.store,
            &self.root_pin,
            Some(&self.route_cache),
            src_search,
            |hit| hit.value.to_vec(),
        )?
        else {
            return Err(Error::NotFound);
        };
        if src == dst {
            return Ok(());
        }
        if !force
            && engine::lookup_multi_with(
                &self.store,
                &self.root_pin,
                Some(&self.route_cache),
                dst_search,
                |_| (),
            )?
            .is_some()
        {
            return Err(Error::DstExists);
        }

        let erase_out = engine::erase_multi(
            &self.store,
            &self.root_pin,
            Some(&self.route_cache),
            src_search,
            seq,
        )?;
        let insert_out = engine::insert_multi(
            &self.store,
            &self.root_pin,
            Some(&self.route_cache),
            dst_search,
            &value,
            seq,
        )?;
        if erase_out.root_dirty || insert_out.root_dirty {
            self.store
                .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
        }
        Ok(())
    }

    /// Open a stateful range iterator anchored at this tree.
    ///
    /// Returns a [`RangeBuilder`] for chaining `prefix`,
    /// `start_after`, and `delimiter`. Call
    /// [`RangeBuilder::into_iter`] (or `for entry in builder`) to
    /// start emitting [`crate::RangeEntry`] items in lex key order.
    ///
    /// Restart-on-conflict cursor semantics: the iterator stores
    /// blob content versions in its path frames. If a concurrent
    /// writer invalidates the path through split / merge / compact
    /// / normal mutation, the next step seeks directly from the
    /// last emitted key or delimiter rollup boundary instead of
    /// continuing through stale `(blob_guid, slot)` state.
    ///
    /// This is not an MVCC snapshot: a long scan can observe keys
    /// committed after iterator creation if they sort after the
    /// current cursor. It is, however, monotonic with respect to
    /// already-emitted keys and rollups.
    pub fn range(&self) -> RangeBuilder {
        RangeBuilder::new(
            Arc::clone(&self.store),
            Arc::clone(&self.root_pin),
            self.root_guid,
            Arc::clone(&self.maintenance_gate),
        )
    }

    /// Shorthand for `tree.range().prefix(p)` — the
    /// common-90%-of-queries case.
    ///
    /// Returns a [`RangeBuilder`] already anchored to `prefix`;
    /// chain additional filters (`start_after`, `delimiter`)
    /// before iterating.
    pub fn scan(&self, prefix: &[u8]) -> RangeBuilder {
        self.range().prefix(prefix)
    }

    /// Open a key-only range iterator anchored at this tree.
    ///
    /// This has the same ordering, `prefix`, `start_after`,
    /// `delimiter`, and restart-on-conflict semantics as
    /// [`Self::range`], but [`KeyRangeEntry::Key`] does not carry
    /// value bytes. Use it for metadata listing paths that only
    /// need names and compare-and-set versions.
    pub fn range_keys(&self) -> KeyRangeBuilder {
        KeyRangeBuilder::new(self.range()).with_prefix_list_cache(
            Arc::clone(&self.prefix_list_cache),
            Arc::clone(&self.next_seq),
        )
    }

    /// Shorthand for `tree.range_keys().prefix(p)`.
    ///
    /// This is the fast path for prefix/delimiter scans where
    /// values are not needed for every emitted key.
    pub fn scan_keys(&self, prefix: &[u8]) -> KeyRangeBuilder {
        self.range_keys().prefix(prefix)
    }

    /// Run a read-only transaction over a prefix snapshot.
    ///
    /// The tree captures the blob frames reachable for `prefix`
    /// under a short exclusive maintenance gate, releases the live
    /// tree, then invokes `read` with an immutable [`View`]. Writes
    /// committed after the capture are invisible to all reads made
    /// through that view. The implementation copies blob frames, not
    /// decoded entries, so point lookup and range/list operations
    /// keep using the ART walker.
    ///
    /// A view is scoped: reads outside `prefix` return
    /// [`Error::OutsideViewScope`]. Use `prefix = b""` only when a
    /// whole-tree snapshot is intentional.
    pub fn view<F, R>(&self, prefix: &[u8], read: F) -> Result<R>
    where
        F: FnOnce(&View) -> Result<R>,
    {
        let view = self.capture_view(prefix)?;
        read(&view)
    }

    fn capture_view(&self, prefix: &[u8]) -> Result<View> {
        let scope = prefix.to_vec();
        let (snapshot_store, blob_count) = {
            let _maintenance = self.maintenance_gate.enter_exclusive();
            let topology =
                engine::collect_prefix_blob_topology_silent(&self.store, self.root_guid, prefix)?;
            let snapshot_store = Arc::new(MemoryBlobStore::new());
            for entry in &topology {
                let bytes = self.store.snapshot_blob_image(entry.guid)?;
                snapshot_store.write_blob(entry.guid, &bytes)?;
            }
            (snapshot_store, topology.len())
        };

        let snapshot_bm = Arc::new(BufferManager::new(snapshot_store, blob_count.max(1)));
        let root_pin = snapshot_bm.pin(self.root_guid)?;
        Ok(View::new(scope, snapshot_bm, self.root_guid, root_pin))
    }

    /// Return `true` if no live key starts with `prefix`.
    ///
    /// This is a point-in-time read helper. Concurrent writers may
    /// make the prefix non-empty immediately after it returns; use
    /// [`AtomicBatch::assert_prefix_empty`] inside [`Self::atomic`] when
    /// the emptiness check must be atomic with subsequent writes.
    pub fn is_prefix_empty(&self, prefix: &[u8]) -> Result<bool> {
        let _maintenance = self.maintenance_gate.enter_shared();
        self.projected_prefix_empty(&std::collections::HashMap::new(), prefix)
    }

    /// Drain the BM dirty map and synchronously push entries to
    /// the inner store via batched write-through (CAS-on-seq).
    ///
    /// Used by:
    /// - The no-WAL `memory_flush_on_write` path, where every op must
    ///   reach store before returning (no checkpointer to defer
    ///   to).
    /// - `Tree::checkpoint`, where the user explicitly asks for
    ///   a full-tree durability barrier.
    ///
    /// `snapshot_dirty` atomically drains the map; concurrent
    /// `mark_dirty` calls land in the fresh empty map and stay
    /// tracked for the next round. Write-through with `expected_seq`
    /// matches checkpoint: retire only the entry captured by this
    /// snapshot, leaving any racing newer seq for a later flush.
    fn flush_dirty_inline(&self) -> Result<()> {
        let snap = self.store.snapshot_dirty();
        let mut failed: std::collections::HashMap<BlobGuid, u64> = std::collections::HashMap::new();
        let mut first_err: Option<Error> = None;
        let mut entries = Vec::with_capacity(snap.len());
        for (guid, expected_seq) in snap {
            // `snapshot_bytes` clones the cached image under a
            // brief shared read guard so we hand owned bytes to
            // write-through. `None` means the blob was evicted
            // between snapshot_dirty and snapshot_bytes — invariant
            // I1 regressed, so restore the entry and fail loudly.
            if let Some(bytes) = self.store.snapshot_bytes(guid) {
                entries.push(WriteThroughEntry {
                    guid,
                    bytes,
                    expected_seq,
                });
            } else {
                failed.insert(guid, expected_seq);
                first_err.get_or_insert(Error::Internal(
                    "flush_dirty_inline: dirty entry lost cache image — invariant I1 violated",
                ));
            }
        }
        if !entries.is_empty() {
            let expected: Vec<_> = entries
                .iter()
                .map(|entry| (entry.guid, entry.expected_seq))
                .collect();
            if let Err(e) = self.store.write_through_batch(&entries) {
                for (guid, expected_seq) in expected {
                    failed.insert(guid, expected_seq);
                }
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if !failed.is_empty() {
            self.store.restore_dirty(failed);
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    }

    /// Drain the BM pending-delete queue and apply each
    /// `store.delete_blob` synchronously.
    ///
    /// Companion to [`Self::flush_dirty_inline`] for the deferred
    /// delete protocol — `erase` ops that emptied a child blob
    /// stage the delete here so the manifest mutation can't reach
    /// disk before the WAL record covering the erase is durable
    /// (invariant W2D).
    ///
    /// Must run **after** `flush_dirty_inline` (any new bytes in
    /// dirty land first) and **before** the trailing
    /// `store.flush` (which persists the manifest deletion).
    /// Restoration is automatic on individual failures — the
    /// remaining entries stay queued for the next attempt.
    fn flush_pending_deletes_inline(&self) -> Result<()> {
        let pending = self.store.snapshot_pending_deletes();
        let mut failed: std::collections::HashMap<BlobGuid, u64> = std::collections::HashMap::new();
        let mut first_err: Option<Error> = None;
        for (guid, seq) in pending {
            if let Err(e) = self.store.execute_pending_delete(guid) {
                failed.insert(guid, seq);
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if !failed.is_empty() {
            self.store.restore_pending_deletes(failed);
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    }

    /// Make every previously-applied mutation durable and trim
    /// the WAL.
    ///
    /// Mirrors the background checkpoint round's protocol so a
    /// manual checkpoint is just as concurrency-safe against
    /// in-flight writers as the background path. Phases are kept
    /// strictly ordered around the W2D invariant; every error
    /// path restores any snapshot it drained so the next round
    /// retries.
    ///
    /// 1. **Snapshot + journal flush** (under `commit_gate`):
    ///    drain BM dirty + pending-delete sets, force the journal
    ///    durable, and clone each snapshotted blob's bytes before
    ///    releasing the lock. Journal flush failure → restore both
    ///    snapshots, return.
    /// 2. **Per-blob write-through** with CAS-on-seq. The CAS
    ///    retires the dirty entry only if no racing writer bumped
    ///    it; failures stay in `dirty` for the next round.
    ///    If the snapshot had neither dirty blobs nor pending
    ///    deletes and the store reports no outstanding flush
    ///    work, skip the store Sync path entirely.
    /// 3. **Pre-delete sync** — `store.flush` (`sync_data` on
    ///    the data file + persist the manifest) so step 2's
    ///    writes hit stable storage *before* any manifest delete
    ///    runs. Sync failure → restore pending, return.
    /// 4. **Abort-on-dirty-failure gate**. If any write-through
    ///    at step 2 failed, the round must NOT apply pending
    ///    deletes: a parent that didn't flush might still
    ///    reference a child that's about to be removed from the
    ///    manifest, leaving the on-disk parent pointing into a
    ///    deleted slot. Restore pending and return the dirty
    ///    error. The next round will retry the parent write and
    ///    only then process its child's deletion.
    /// 5. **Apply pending deletes** (manifest mutation
    ///    in-memory). Each `execute_pending_delete` is idempotent
    ///    against a missing entry; failures are restored.
    /// 6. **Post-delete sync** — re-`store.flush` iff any delete
    ///    actually applied. Failure → restore the
    ///    already-applied entries so the truncate gate stays
    ///    closed and the next round retries the sync (the manifest
    ///    delete is idempotent on the second pass).
    /// 7. **Conditional WAL truncate** — only if
    ///    `dirty_count == 0` AND `pending_delete_count == 0`
    ///    *now*. A racing writer or a restored failure must keep
    ///    the WAL alive until a future flush.
    ///
    /// `memory_flush_on_write = false` callers rely on this to make
    /// batched writes survive a crash.
    #[allow(clippy::too_many_lines)]
    pub fn checkpoint(&self) -> Result<()> {
        use std::collections::HashMap;

        let _maintenance = self.maintenance_gate.enter_shared();

        // Phase 1: snapshot dirty/pending, clone bytes, and record
        // the WAL watermark under `commit_gate`. This closes the
        // subtle W2D hole where a foreground writer mutates a blob
        // after the dirty snapshot but before `snapshot_bytes`: the
        // checkpoint could otherwise write bytes whose WAL record was
        // not part of the checkpoint snapshot. The WAL flush itself
        // happens after releasing the gate so fsync does not block
        // foreground writers.
        let (snap_dirty, snap_pending, snap_bytes, wal_up_to) = if let Some(journal) = &self.journal
        {
            let _commit = self.commit_gate.enter_checkpoint();
            let snap_dirty = self.store.snapshot_dirty();
            let snap_pending = self.store.snapshot_pending_deletes();
            let wal_up_to = journal.wal_work();
            let mut snap_bytes = Vec::with_capacity(snap_dirty.len());
            for (guid, expected_seq) in &snap_dirty {
                let Some(bytes) = self.store.snapshot_bytes(*guid) else {
                    self.store.restore_pending_deletes(snap_pending);
                    self.store.restore_dirty(snap_dirty);
                    return Err(Error::Internal(
                        "checkpoint: dirty entry lost cache image — invariant I1 violated",
                    ));
                };
                snap_bytes.push((*guid, *expected_seq, bytes));
            }
            (snap_dirty, snap_pending, snap_bytes, Some(wal_up_to))
        } else {
            let snap_dirty = self.store.snapshot_dirty();
            let snap_pending = self.store.snapshot_pending_deletes();
            let mut snap_bytes = Vec::with_capacity(snap_dirty.len());
            for (guid, expected_seq) in &snap_dirty {
                let Some(bytes) = self.store.snapshot_bytes(*guid) else {
                    self.store.restore_pending_deletes(snap_pending);
                    self.store.restore_dirty(snap_dirty);
                    return Err(Error::Internal(
                        "checkpoint: dirty entry lost cache image — invariant I1 violated",
                    ));
                };
                snap_bytes.push((*guid, *expected_seq, bytes));
            }
            (snap_dirty, snap_pending, snap_bytes, None)
        };

        if let (Some(journal), Some(up_to)) = (&self.journal, wal_up_to) {
            if let Err(e) = journal.flush_up_to(up_to) {
                self.store.restore_pending_deletes(snap_pending);
                self.store.restore_dirty(snap_dirty);
                return Err(e);
            }
        }

        // Phase 2: batch write-through with CAS-on-seq.
        //
        // A drained dirty entry **must** have a cache image —
        // invariant I1 (dirty ⟺ cache newer than store). If
        // `snapshot_bytes` returns `None`, the BM's eviction
        // policy regressed and dropped a dirty cache image; that
        // would otherwise be a silent data-loss path (the next
        // checkpoint sees `dirty == 0` and truncates the WAL).
        // Restore both snapshots and bail loud.
        let mut dirty_failed: HashMap<BlobGuid, u64> = HashMap::new();
        let mut first_dirty_err: Option<Error> = None;
        let entries: Vec<_> = snap_bytes
            .into_iter()
            .map(|(guid, expected_seq, bytes)| WriteThroughEntry {
                guid,
                bytes,
                expected_seq,
            })
            .collect();
        if !entries.is_empty() {
            let expected: Vec<_> = entries
                .iter()
                .map(|entry| (entry.guid, entry.expected_seq))
                .collect();
            if let Err(e) = self.store.write_through_batch(&entries) {
                // BlobStore batch failure may have landed any prefix;
                // retry the whole snapshot next round.
                for (guid, expected_seq) in expected {
                    dirty_failed.insert(guid, expected_seq);
                }
                if first_dirty_err.is_none() {
                    first_dirty_err = Some(e);
                }
            }
        }
        let had_dirty_failure = !dirty_failed.is_empty();
        if had_dirty_failure {
            self.store.restore_dirty(dirty_failed);
        }

        if entries.is_empty() && snap_pending.is_empty() && !self.store.needs_flush() {
            if let Some(journal) = &self.journal {
                if journal.needs_checkpoint() {
                    let _commit = self.commit_gate.enter_checkpoint();
                    if self.store.dirty_count() == 0 && self.store.pending_delete_count() == 0 {
                        journal.truncate()?;
                    }
                }
            }
            return Ok(());
        }

        // Phase 3: pre-delete sync. Even when some writes failed at
        // phase 2, the successful ones already retired their dirty
        // entries via the write-through CAS — we must still fsync
        // so those bytes are stable on disk. On sync failure,
        // pending deletes haven't been applied yet, so restore them
        // and bail.
        if let Err(e) = self.store.flush() {
            self.store.restore_pending_deletes(snap_pending);
            return Err(e);
        }

        // Phase 4: abort-on-dirty-failure gate. A failed parent
        // write-through must NOT propagate to a manifest delete of
        // its dependent child — that would orphan the parent's
        // BlobNode pointer (parent on-disk still has the child
        // pointer; manifest no longer has the child entry; WAL
        // replay's walker descent would fail to read the deleted
        // child). Restore the entire pending snapshot and surface
        // the dirty error.
        if had_dirty_failure {
            self.store.restore_pending_deletes(snap_pending);
            return Err(first_dirty_err.expect("had_dirty_failure ⇒ first_dirty_err set"));
        }

        // Phase 5: apply pending deletes (manifest mutation).
        let mut pending_failed: HashMap<BlobGuid, u64> = HashMap::new();
        let mut first_pending_err: Option<Error> = None;
        for (guid, seq) in &snap_pending {
            if let Err(e) = self.store.execute_pending_delete(*guid) {
                pending_failed.insert(*guid, *seq);
                if first_pending_err.is_none() {
                    first_pending_err = Some(e);
                }
            }
        }
        if !pending_failed.is_empty() {
            self.store.restore_pending_deletes(pending_failed.clone());
        }

        // Phase 6: post-delete sync iff any delete actually applied.
        // On sync failure, restore the already-applied entries to
        // pending — the manifest mutation we did at phase 5 is
        // stuck in-memory until the next sync succeeds, so the
        // truncate gate at phase 7 must stay closed. Re-executing
        // `execute_pending_delete` on the restored entries is a
        // no-op (HashMap::remove on a missing key).
        let applied_deletes = snap_pending.len() - pending_failed.len();
        if applied_deletes > 0 {
            if let Err(e) = self.store.flush() {
                let restore_applied: HashMap<BlobGuid, u64> = snap_pending
                    .iter()
                    .filter(|(g, _)| !pending_failed.contains_key(*g))
                    .map(|(g, s)| (*g, *s))
                    .collect();
                self.store.restore_pending_deletes(restore_applied);
                return Err(e);
            }
        }

        if let Some(e) = first_pending_err {
            return Err(e);
        }

        // 6. Conditional truncate. A writer that landed a
        //    mark_dirty between our snapshot and here has its
        //    entry still in `dirty` (write-through CAS won't
        //    retire newer-seq entries, and snapshot only drained
        //    what we observed at step 1); leave the WAL alone so
        //    that entry's WAL record stays recoverable. Same
        //    logic for pending_delete_count.
        if let Some(journal) = &self.journal {
            if journal.needs_checkpoint() {
                let _commit = self.commit_gate.enter_checkpoint();
                if self.store.dirty_count() == 0 && self.store.pending_delete_count() == 0 {
                    journal.truncate()?;
                }
            }
        }
        Ok(())
    }

    /// Snapshot per-blob and aggregate counters for every blob
    /// reachable from the root.
    ///
    /// Each blob is pinned + read under a single shared guard, so
    /// stats never block ongoing reads and only contend with writers
    /// on a blob-by-blob basis. The maintenance read gate prevents a
    /// concurrent merge/compact pass from deleting a child while the
    /// tree shape is being enumerated. Returned counters are a
    /// consistent snapshot of each individual blob but the aggregate
    /// is **not** linearised across foreground writers — a concurrent
    /// writer mid-traversal can shift one blob's counters before
    /// another's are read. Acceptable for observability; pause writes
    /// externally if you need a quiescent snapshot.
    pub fn stats(&self) -> Result<TreeStats> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let aggregate = self.collect_blob_stats_silent()?;
        let bm_dirty_count = self.store.dirty_count();
        let bm_pending_delete_count = self.store.pending_delete_count();
        let bm_cache_hits = self.store.cache_hits();
        let bm_cache_misses = self.store.cache_misses();
        let bm_optimistic_restarts = self.store.optimistic_restarts();
        let bm_range_restarts = self.store.range_restarts();
        let bm_walker_ops = self.store.walker_ops();
        let bm_walker_blob_hops = self.store.walker_blob_hops();
        let bm_max_blob_hops = self.store.max_blob_hops();
        let bm_max_cross_blob_depth = self.store.max_cross_blob_depth();
        let bm_spillovers = self.store.spillover_count();
        let bm_merges = self.store.merge_count();
        let bm_route_resident_count = self.store.route_resident_count();
        let bm_route_resident_demotions = self.store.route_resident_demotions();
        let route = self.route_cache.stats();
        let route_cache = RouteCacheStats {
            entries: route.entries,
            hits: route.hits,
            misses: route.misses,
            learns: route.learns,
            evictions: route.evictions,
            invalidations: route.invalidations,
        };
        let journal = self.journal.as_ref().map(|j| {
            let s = j.stats();
            JournalStats {
                appends: s.appends,
                batches: s.batches,
                syncs: s.syncs,
            }
        });
        let checkpointer = self.checkpointer.as_ref().map(|ck| CheckpointerStats {
            rounds_attempted: ck.rounds_attempted(),
            rounds_succeeded: ck.rounds_succeeded(),
            blobs_flushed: ck.blobs_flushed(),
            merges_total: ck.merges_total(),
            truncates: ck.truncates(),
            evictions: ck.evictions(),
        });
        Ok(TreeStats {
            blob_count: aggregate.blobs.len() as u32,
            total_space_used: aggregate.total_space_used,
            total_gap_space: aggregate.total_gap_space,
            total_slots: aggregate.total_slots,
            total_compactions: aggregate.total_compactions,
            total_tombstones: aggregate.total_tombstones,
            total_blob_edges: aggregate.total_blob_edges,
            leaf_blob_count: aggregate.leaf_blob_count,
            max_blob_depth: aggregate.max_blob_depth,
            total_blob_depth: aggregate.total_blob_depth,
            max_blob_fill_per_mille: aggregate.max_blob_fill_per_mille,
            underfilled_child_blobs: aggregate.underfilled_child_blobs,
            overfull_child_blobs: aggregate.overfull_child_blobs,
            blobs: aggregate.blobs,
            bm_dirty_count,
            bm_pending_delete_count,
            bm_cache_hits,
            bm_cache_misses,
            bm_optimistic_restarts,
            bm_range_restarts,
            bm_walker_ops,
            bm_walker_blob_hops,
            bm_max_blob_hops,
            bm_max_cross_blob_depth,
            bm_spillovers,
            bm_merges,
            bm_route_resident_count,
            bm_route_resident_demotions,
            route_cache,
            journal,
            checkpointer,
        })
    }

    fn collect_blob_stats_silent(&self) -> Result<BlobStatsAggregate> {
        // `Tree::stats` is an introspection path — used by users
        // checking on the tree, and (via `holt::metrics`) by
        // Prometheus scrapes that read `bm_cache_hits`,
        // `bm_cache_misses`, `bm_optimistic_restarts`, etc. If
        // the walk went through `BufferManager::pin`, each scrape
        // would inflate cache counters and refresh eviction
        // recency. Use silent pins for both the topology pass and
        // the per-blob header reads.
        let topology = engine::collect_blob_topology_silent(&self.store, self.root_guid)?;
        let blob_data_capacity = (PAGE_SIZE - DATA_AREA_START) as u64;
        let mut aggregate = BlobStatsAggregate {
            blobs: Vec::with_capacity(topology.len()),
            total_space_used: 0,
            total_gap_space: 0,
            total_slots: 0,
            total_compactions: 0,
            total_tombstones: 0,
            total_blob_edges: 0,
            leaf_blob_count: 0,
            max_blob_depth: 0,
            total_blob_depth: 0,
            max_blob_fill_per_mille: 0,
            underfilled_child_blobs: 0,
            overfull_child_blobs: 0,
        };

        for entry in &topology {
            let pin = self.store.pin_silent(entry.guid)?;
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let h = frame.header();
            let stats = BlobStats {
                guid: entry.guid,
                space_used: h.space_used,
                gap_space: h.gap_space,
                num_slots: h.num_slots,
                num_ext_blobs: h.num_ext_blobs,
                compact_times: h.compact_times,
                tombstone_leaf_cnt: h.tombstone_leaf_cnt,
            };
            Self::accumulate_blob_stats(&mut aggregate, stats, entry.depth, blob_data_capacity);
        }
        Ok(aggregate)
    }

    fn accumulate_blob_stats(
        aggregate: &mut BlobStatsAggregate,
        stats: BlobStats,
        depth: u32,
        blob_data_capacity: u64,
    ) {
        aggregate.total_space_used += u64::from(stats.space_used);
        aggregate.total_gap_space += u64::from(stats.gap_space);
        aggregate.total_slots += u64::from(stats.num_slots);
        aggregate.total_compactions += u64::from(stats.compact_times);
        aggregate.total_tombstones += u64::from(stats.tombstone_leaf_cnt);
        aggregate.total_blob_edges += u64::from(stats.num_ext_blobs);
        if stats.num_ext_blobs == 0 {
            aggregate.leaf_blob_count += 1;
        }
        aggregate.max_blob_depth = aggregate.max_blob_depth.max(depth);
        aggregate.total_blob_depth += u64::from(depth);
        let fill_per_mille = blob_fill_per_mille(stats.space_used, blob_data_capacity);
        aggregate.max_blob_fill_per_mille = aggregate.max_blob_fill_per_mille.max(fill_per_mille);
        if depth != 0 {
            if fill_per_mille < SHAPE_UNDERFILLED_CHILD_FILL_PER_MILLE {
                aggregate.underfilled_child_blobs += 1;
            } else if fill_per_mille > SHAPE_OVERFULL_CHILD_FILL_PER_MILLE {
                aggregate.overfull_child_blobs += 1;
            }
        }
        aggregate.blobs.push(stats);
    }

    /// Run one online maintenance pass.
    ///
    /// ## Concurrency
    ///
    /// Safe to run while point reads and foreground writers are
    /// active. The pass is candidate-driven: deletes and leaf-slot
    /// churn enqueue blob-local compaction candidates; spillovers
    /// enqueue parent-merge candidates. A cold call with no queued
    /// candidates seeds the queues once by scanning reachable
    /// blobs. After that, each call processes at most
    /// `ONLINE_COMPACT_BLOB_BUDGET` compact candidates and
    /// `ONLINE_MERGE_PARENT_BUDGET` merge candidates, so online
    /// maintenance is bounded rather than a whole-tree sweep.
    ///
    /// Blob-local compaction holds only the shared maintenance side
    /// plus the candidate blob's latch; clean stale candidates are
    /// skipped after a shared-latch header check. Merge still uses the
    /// exclusive maintenance side, but only around the one parent
    /// being folded/deleted. Range iterators detect those rewrites
    /// through their versioned cursor frames and restart from the
    /// last emitted lower bound.
    ///
    /// Both phases stage their changes via `mark_dirty` /
    /// `mark_for_delete` on the internal `BufferManager`
    /// rather than writing through to store inline. This keeps
    /// compact compatible with invariant **W2D**: a naive
    /// `bm.commit(*guid)` per touched blob would push the cache
    /// image (including any user mutations whose WAL records
    /// aren't yet durable) straight to store, and a crash
    /// before those WAL records flushed would leave the store
    /// at a post-mutation state with no journal to reconcile
    /// against — silent data loss after a WAL replay rebuilds
    /// the cache to the pre-mutation state.
    ///
    /// Does **not** fsync the store or touch the WAL — call
    /// [`Tree::checkpoint`] after if you want the rebuilt blobs
    /// durable on disk. Compaction is logically idempotent (the
    /// post-compact tree is observationally identical to the
    /// pre-compact one), so a crash mid-compact just means the
    /// next run re-does the work; the W2D protocol keeps the
    /// store image consistent throughout.
    ///
    /// This is intentionally incremental. Re-invoke `compact` until
    /// [`Tree::stats`] shows the tombstone / merge backlog has
    /// settled if you want to force a tree all the way down after a
    /// heavy churn phase.
    pub fn compact(&self) -> Result<()> {
        if self.store.compaction_candidate_count() == 0 && self.store.merge_candidate_count() == 0 {
            self.seed_maintenance_candidates()?;
        }

        let compact_guids = self
            .store
            .pop_compaction_candidates(ONLINE_COMPACT_BLOB_BUDGET);
        let mut compacted_any = false;
        for guid in compact_guids {
            compacted_any |= self.compact_candidate_blob(guid)?;
        }

        if compacted_any && self.store.merge_candidate_count() == 0 {
            self.seed_maintenance_candidates()?;
        }

        let merge_guids = self.store.pop_merge_candidates(ONLINE_MERGE_PARENT_BUDGET);
        for guid in merge_guids {
            self.merge_candidate_parent(guid)?;
        }
        Ok(())
    }

    fn seed_maintenance_candidates(&self) -> Result<()> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let guids = engine::collect_blob_guids(&self.store, self.root_guid)?;
        for guid in guids {
            let pin = self.store.pin(guid)?;
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let header = frame.header();
            if engine::blob_needs_compaction(frame) {
                self.store.note_compaction_candidate(guid);
            }
            if header.num_ext_blobs != 0 {
                self.store.note_merge_candidate(guid);
            }
        }
        Ok(())
    }

    fn compact_candidate_blob(&self, guid: BlobGuid) -> Result<bool> {
        use crate::store::STRUCTURAL_SEQ;

        let _maintenance = self.maintenance_gate.enter_shared();
        if !self.store.has_blob(guid)? {
            return Ok(false);
        }
        let pin = self.store.pin(guid)?;
        let needs_compaction = {
            let guard = pin.read();
            engine::blob_needs_compaction(BlobFrameRef::wrap(guard.as_slice()))
        };
        if !needs_compaction {
            return Ok(false);
        }

        let _commit = self
            .journal
            .as_ref()
            .map(|_| self.commit_gate.enter_writer());
        let compacted = {
            let mut guard = pin.write();
            let still_needs_compaction = {
                let frame = guard.frame();
                engine::blob_needs_compaction(frame.as_ref())
            };
            if still_needs_compaction {
                engine::compact_blob(&mut guard)?;
            }
            still_needs_compaction
        };
        if compacted {
            // Keep the pin alive until after dirty publication so
            // eviction cannot drop the rebuilt cache image before a
            // checkpoint snapshots it.
            self.store.mark_dirty(guid, STRUCTURAL_SEQ);
        }
        drop(pin);
        Ok(compacted)
    }

    fn merge_candidate_parent(&self, guid: BlobGuid) -> Result<()> {
        use crate::store::STRUCTURAL_SEQ;

        let _maintenance = self.maintenance_gate.enter_exclusive();
        if !self.store.has_blob(guid)? {
            return Ok(());
        }
        let _commit = self
            .journal
            .as_ref()
            .map(|_| self.commit_gate.enter_writer());
        let pin = self.store.pin(guid)?;
        let (merged, has_children) = {
            let mut guard = pin.write();
            let mut frame = guard.frame();
            let merged = engine::try_merge_children(&self.store, &mut frame, STRUCTURAL_SEQ)?;
            (merged, frame.header().num_ext_blobs != 0)
        };
        if merged.merged > 0 {
            self.store.mark_dirty(guid, STRUCTURAL_SEQ);
            self.store.note_merges(u64::from(merged.merged));
            if has_children {
                self.store.note_merge_candidate(guid);
            }
        }
        drop(pin);
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
/// engine. Blob-shape changes (`splitBlob`, `mergeBlob`,
/// `compactBlob`) are deliberately not independent WAL records:
/// they are derived from replaying logical operations or from
/// checkpointed blob images. A standalone structural record would
/// need full physical context to be recoverable, so the codec
/// rejects those old draft tags instead of treating them as
/// successful no-ops.
///
/// `RenameObject` is rebuilt as the same erase + insert it ran
/// originally.
///
/// ## Dirty tracking on replay
///
/// Walker calls (`insert_multi` / `erase_multi`) mutate the
/// BM-cached root + any cross-blob children. The walker marks
/// touched child blobs dirty itself; the root's `mark_dirty` is
/// the **caller's** responsibility when the returned outcome says
/// `root_dirty`. Replay must honour that contract. Without this,
/// a `Tree::open` → `Tree::checkpoint` immediately after replay
/// could find an empty dirty set, write nothing to store, then
/// truncate the WAL — silently losing every replayed record.
fn replay_wal(path: &std::path::Path, bm: &Arc<BufferManager>, root_guid: BlobGuid) -> Result<u64> {
    // Pin the root once for the entire replay loop; saves a
    // BM-Mutex per op (replays can be thousands of ops on a
    // dirty WAL).
    let root_pin = bm.pin(root_guid)?;
    let (_header, stats) = replay(path, |op, seq, _off| {
        // `root_dirty` tracks whether this op actually mutated
        // the BM-cached root image. No-op replays (e.g. an erase
        // for a key already absent because a prior replay pass
        // reconciled it) leave the cache byte-identical to
        // store — skipping `mark_dirty` for those is a small
        // win and matches `Tree::delete`'s same-shape branch.
        let root_dirty = match op {
            WalOp::Insert { key, value } => {
                let search = engine::SearchKey::user(key);
                engine::insert_multi(bm, &root_pin, None, search, value, seq)?.root_dirty
            }
            WalOp::Erase { key } => {
                let search = engine::SearchKey::user(key);
                engine::erase_multi(bm, &root_pin, None, search, seq)?.root_dirty
            }
            WalOp::RenameObject {
                src_key,
                dst_key,
                force,
                ..
            } => {
                let src_search = engine::SearchKey::user(src_key);
                let dst_search = engine::SearchKey::user(dst_key);
                // Existence probes pass a `|_| ()` closure so the
                // walker doesn't even allocate / copy the value.
                if engine::lookup_multi_with(bm, &root_pin, None, src_search, |_| ())?.is_none() {
                    // Already reconciled in a prior replay pass —
                    // skip. `highest` was bumped above so the
                    // post-replay `next_seq` still advances past
                    // this record's seq.
                    return Ok(());
                }
                if !force
                    && engine::lookup_multi_with(bm, &root_pin, None, dst_search, |_| ())?.is_some()
                {
                    return Ok(());
                }
                let value = engine::lookup_multi_with(bm, &root_pin, None, src_search, |hit| {
                    hit.value.to_vec()
                })?
                .unwrap_or_default();
                let erase_out = engine::erase_multi(bm, &root_pin, None, src_search, seq)?;
                let insert_out =
                    engine::insert_multi(bm, &root_pin, None, dst_search, &value, seq)?;
                erase_out.root_dirty || insert_out.root_dirty
            }
            // `Batch` is unpacked into per-inner callbacks inside
            // `journal::reader::replay_bytes`, so it never reaches
            // this match — defensive arm only.
            WalOp::Batch { ops: _ } => false,
        };
        if root_dirty {
            // Honour the walker's caller-side `mark_dirty(root,
            // seq)` contract — see the module doc above.
            bm.mark_dirty(root_guid, seq);
        }
        Ok(())
    })?;
    debug_assert!(
        stats.records_seen > 0 || stats.highest_seq.is_none(),
        "an empty WAL replay cannot report a highest seq",
    );
    debug_assert!(
        stats.torn_tail_at.is_none() || stats.records_seen > 0 || stats.highest_seq.is_none(),
        "a torn tail without complete records must not report a highest seq",
    );
    // After commit, the blob image is durable; we still want the
    // next allocated seq to be strictly greater than anything
    // ever seen in the log.
    Ok(stats.highest_seq.unwrap_or(0) + 1)
}

#[cfg(test)]
mod tests {
    use crate::TreeBuilder;
    use std::sync::mpsc::sync_channel;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn strict_prefix_point_keys_round_trip_through_public_api() {
        let tree = TreeBuilder::new("ignored").memory().open().unwrap();
        tree.put(b"abc", b"short").unwrap();
        tree.put(b"abcdef", b"long").unwrap();

        assert_eq!(tree.get(b"abc").unwrap().as_deref(), Some(&b"short"[..]));
        assert_eq!(tree.get(b"abcdef").unwrap().as_deref(), Some(&b"long"[..]));
        assert!(tree.delete(b"abc").unwrap());
        assert_eq!(tree.get(b"abc").unwrap(), None);
        assert_eq!(tree.get(b"abcdef").unwrap().as_deref(), Some(&b"long"[..]));
    }

    #[test]
    fn compact_waits_for_maintenance_read_guard() {
        let tree = TreeBuilder::new("ignored")
            .memory()
            .buffer_pool_size(16)
            .open()
            .unwrap();
        let big = vec![0xCDu8; 4 * 1024];
        for i in 0..256u32 {
            tree.put(format!("k{i:08}").as_bytes(), &big).unwrap();
        }
        for i in 0..248u32 {
            tree.delete(format!("k{i:08}").as_bytes()).unwrap();
        }
        assert!(
            tree.stats().unwrap().blob_count > 1,
            "test precondition: compact must have a BlobNode merge phase"
        );

        let read_guard = tree.maintenance_gate.enter_shared();
        let worker_tree = tree.clone();
        let (started_tx, started_rx) = sync_channel(0);
        let (done_tx, done_rx) = sync_channel(0);
        let handle = thread::spawn(move || {
            started_tx.send(()).unwrap();
            worker_tree.compact().unwrap();
            done_tx.send(()).unwrap();
        });

        started_rx.recv().unwrap();
        assert!(
            done_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "compact must wait behind active shared maintenance readers"
        );

        drop(read_guard);
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        handle.join().unwrap();
    }
}
