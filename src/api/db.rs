//! Multi-tree database handle.
//!
//! `DB` owns one buffer manager, one WAL, one checkpoint frontier,
//! and any number of named ART roots. A named tree is still a normal
//! [`crate::Tree`] handle; the difference is that all trees opened
//! from the same `DB` share durability and maintenance gates, so a
//! DB-level atomic batch can commit mutations across trees in one
//! WAL record.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::atomic::{BatchOp, RecordVersion};
use super::checkpoint::{self, CheckpointImage};
use super::config::TreeConfig;
use super::errors::{AtomicErrorKind, Error, Result};
use super::journal::{
    JournalAnchor, JournalEnvelope, JournalEnvelopePage, JournalState, VolatileJournal,
};
use super::snapshot::Snapshot;
use super::stats::{CheckpointerStats, DBStats, JournalStats, OpenStats, VacuumStats};
use super::tree::{
    ensure_durable_root_blob, ensure_root_blob_for_open, replay_wal, Tree, TreeRuntime,
};
use super::view::View;
use crate::concurrency::{CommitGate, Gate};
use crate::engine::RangeEntry;
use crate::journal::codec::{encode_attached_batch_record, BatchEncoder};
use crate::journal::Journal;
use crate::layout::BlobGuid;
use crate::store::blob_store::BlobStore;
use crate::store::BufferManager;

const DB_ROOT_TAG: u8 = 0xDB;
const DB_CATALOG_TREE_ID: u64 = 0x686f_6c74_6462_0001;
const FIRST_USER_TREE_ID: u64 = 1;
const CATALOG_NEXT_TREE_ID_KEY: &[u8] = b"\0next-tree-id";
const CATALOG_VALUE_MAGIC: &[u8; 8] = b"holtdb02";
const CATALOG_NEXT_ID_MAGIC: &[u8; 8] = b"holtnx02";
const CATALOG_STATE_LIVE: u8 = 1;
const CATALOG_STATE_DROPPING: u8 = 2;
const CATALOG_VALUE_LEN: usize = 17;
const CATALOG_NEXT_ID_LEN: usize = 16;
const AUTO_GC_BATCH_SIZE: usize = 256;

fn atomic_definitely_not_applied(source: Error) -> Error {
    Error::Atomic {
        kind: AtomicErrorKind::DefinitelyNotApplied,
        source: Box::new(source),
    }
}

fn atomic_outcome_unknown(source: Error) -> Error {
    Error::Atomic {
        kind: AtomicErrorKind::OutcomeUnknown,
        source: Box::new(source),
    }
}

#[cfg(test)]
struct OpenTreeCatalogBarrier {
    entered: std::sync::Barrier,
    release: std::sync::Barrier,
}

#[cfg(test)]
impl OpenTreeCatalogBarrier {
    fn new() -> Self {
        Self {
            entered: std::sync::Barrier::new(2),
            release: std::sync::Barrier::new(2),
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static OPEN_TREE_CATALOG_BARRIER: std::cell::RefCell<Option<Arc<OpenTreeCatalogBarrier>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn set_open_tree_catalog_barrier_for_current_thread(barrier: Arc<OpenTreeCatalogBarrier>) {
    OPEN_TREE_CATALOG_BARRIER.with(|slot| *slot.borrow_mut() = Some(barrier));
}

#[cfg(test)]
fn pause_open_tree_after_catalog_lookup() {
    let barrier = OPEN_TREE_CATALOG_BARRIER.with(|slot| slot.borrow_mut().take());
    if let Some(barrier) = barrier {
        barrier.entered.wait();
        barrier.release.wait();
    }
}

#[cfg(test)]
struct ExportFirstEntryBarrier {
    entered: std::sync::Barrier,
    release: std::sync::Barrier,
}

#[cfg(test)]
impl ExportFirstEntryBarrier {
    fn new() -> Self {
        Self {
            entered: std::sync::Barrier::new(2),
            release: std::sync::Barrier::new(2),
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static EXPORT_FIRST_ENTRY_BARRIER: std::cell::RefCell<Option<Arc<ExportFirstEntryBarrier>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn set_export_first_entry_barrier_for_current_thread(barrier: Arc<ExportFirstEntryBarrier>) {
    EXPORT_FIRST_ENTRY_BARRIER.with(|slot| *slot.borrow_mut() = Some(barrier));
}

#[cfg(test)]
fn pause_export_after_first_entry() {
    let barrier = EXPORT_FIRST_ENTRY_BARRIER.with(|slot| slot.borrow_mut().take());
    if let Some(barrier) = barrier {
        barrier.entered.wait();
        barrier.release.wait();
    }
}

#[derive(Clone)]
struct OpenTree {
    root_guid: BlobGuid,
    runtime: TreeRuntime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatalogState {
    Live,
    Dropping,
}

#[derive(Clone, Copy, Debug)]
struct CatalogEntry {
    tree_id: u64,
    state: CatalogState,
}

/// A storage instance containing multiple named [`Tree`] roots.
///
/// Use `Tree` directly when one ART namespace is enough. Use `DB`
/// when a system needs independent logical indexes that still share
/// one WAL and one checkpoint boundary, for example `default`,
/// `lock`, and `write` trees in an MVCC metadata layer.
#[derive(Clone)]
pub struct DB {
    cfg: TreeConfig,
    store: Arc<BufferManager>,
    maintenance_gate: Arc<Gate>,
    next_seq: Arc<AtomicU64>,
    commit_gate: Arc<CommitGate>,
    journal: Option<Arc<Journal>>,
    volatile_journal: Option<Arc<VolatileJournal>>,
    checkpointer: Option<Arc<crate::checkpoint::Checkpointer>>,
    open_stats: OpenStats,
    trees: Arc<Mutex<HashMap<u64, OpenTree>>>,
    catalog_cache: Arc<Mutex<HashMap<String, CatalogEntry>>>,
}

impl std::fmt::Debug for DB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DB")
            .field("storage", &self.cfg.storage)
            .finish_non_exhaustive()
    }
}

impl DB {
    /// Open a multi-tree database using the supplied configuration.
    pub fn open(mut cfg: TreeConfig) -> Result<Self> {
        // The background merge queue is keyed only by blob GUID. In a
        // multi-tree DB, a queued parent may become unreachable from all
        // live roots while still sharing children with a live tree or a
        // snapshot. DB-wide merge therefore runs through `DB::compact`,
        // which walks from live roots; the background checkpointer only
        // drains dirty bytes and pending deletes.
        cfg.checkpoint.auto_merge = false;

        let bm = Tree::open_buffer_manager(&cfg)?;
        Self::open_with_buffer_manager(cfg, bm)
    }

    #[cfg(test)]
    fn open_with_blob_store(mut cfg: TreeConfig, store: Arc<dyn BlobStore>) -> Result<Self> {
        cfg.checkpoint.auto_merge = false;
        let bm = Arc::new(BufferManager::new(store, cfg.buffer_pool_size));
        Self::open_with_buffer_manager(cfg, bm)
    }

    fn open_with_buffer_manager(cfg: TreeConfig, bm: Arc<BufferManager>) -> Result<Self> {
        let mut open_stats = OpenStats::default();

        let (journal, next_seq) = match cfg.wal_path() {
            Some(path) => {
                let next_seq = if path.exists() {
                    let start = std::time::Instant::now();
                    let (next_seq, replay_stats) =
                        replay_wal(&path, &bm, !cfg.is_read_only(), |tree_id| {
                            Ok(root_guid_for_tree_id(tree_id))
                        })?;
                    if cfg.is_read_only() {
                        crate::journal::reader::validate_attached_journal(&path)?;
                    }
                    open_stats.wal_replay_micros = start.elapsed().as_micros() as u64;
                    open_stats.wal_replay_records = replay_stats.records_seen;
                    open_stats.wal_torn_tail = replay_stats.torn_tail_at.is_some();
                    if let Ok(meta) = std::fs::metadata(&path) {
                        open_stats.wal_replay_bytes = meta.len();
                    }
                    next_seq
                } else {
                    1
                };
                let journal = if cfg.is_read_only() {
                    None
                } else {
                    Some(Arc::new(Journal::open_or_create(&path, 0)?))
                };
                (journal, next_seq)
            }
            _ => (None, 1),
        };

        let volatile_journal =
            (cfg.is_memory() && !cfg.is_read_only()).then(|| Arc::new(VolatileJournal::new()));
        let maintenance_gate = Arc::new(Gate::new());
        let commit_gate = Arc::new(CommitGate::new());
        let mut db = Self {
            cfg,
            store: bm,
            maintenance_gate,
            next_seq: Arc::new(AtomicU64::new(next_seq)),
            commit_gate,
            journal,
            volatile_journal,
            checkpointer: None,
            open_stats,
            trees: Arc::new(Mutex::new(HashMap::new())),
            catalog_cache: Arc::new(Mutex::new(HashMap::new())),
        };
        // Replay restores logical parents but the BufferManager epoch starts
        // at one in every process. Recover the maximum epoch across every
        // frame reachable from the catalog before any snapshot or background
        // delta flush can serve.
        let epoch_recovery_start = std::time::Instant::now();
        db.restore_epoch_high_water()?;
        db.open_stats.epoch_recovery_micros = epoch_recovery_start
            .elapsed()
            .as_micros()
            .min(u128::from(u64::MAX)) as u64;
        db.restore_dropping_runtime_fences()?;
        db.checkpointer = if db.cfg.is_read_only() {
            None
        } else {
            crate::checkpoint::Checkpointer::spawn_with_volatile(
                Arc::clone(&db.store),
                db.journal.clone(),
                db.volatile_journal.clone(),
                Arc::clone(&db.maintenance_gate),
                Arc::clone(&db.commit_gate),
                db.cfg.checkpoint.clone(),
            )
            .map(Arc::new)
        };
        Ok(db)
    }

    fn ensure_writable(&self) -> Result<()> {
        if self.cfg.is_read_only() {
            Err(Error::ReadOnly)
        } else {
            Ok(())
        }
    }

    fn ensure_ordinary_writes_allowed(&self) -> Result<()> {
        if let Some(journal) = &self.journal {
            journal.ensure_ordinary_writes_allowed()?;
        }
        if let Some(journal) = &self.volatile_journal {
            journal.ensure_ordinary_writes_allowed()?;
        }
        Ok(())
    }

    /// Create a named tree inside this DB.
    ///
    /// Creation is recorded in the internal catalog before the
    /// handle is returned. Re-creating an existing name returns
    /// [`Error::TreeExists`].
    pub fn create_tree(&self, name: &str) -> Result<Tree> {
        self.ensure_writable()?;
        let name_bytes = validate_tree_name(name)?;
        let _maintenance = self.maintenance_gate.enter_exclusive();
        self.ensure_ordinary_writes_allowed()?;
        if self.catalog_entry(name_bytes)?.is_some() {
            return Err(Error::TreeExists {
                name: name.to_owned(),
            });
        }
        let tree_id = self.allocate_tree_id()?;
        let root_guid = root_guid_for_tree_id(tree_id);

        // Publish the deterministic empty root durably before making the
        // catalog entry Live. If catalog publication later fails, the root is
        // merely unreachable GC debt; the inverse ordering can leave a Live
        // catalog entry pointing at a missing root and poison DB reopen.
        ensure_durable_root_blob(&self.store, root_guid)?;

        self.apply_system_batch_unlocked(
            DB_CATALOG_TREE_ID,
            vec![
                BatchOp::PutIfAbsent {
                    key: name_bytes.to_vec(),
                    value: encode_catalog_value(tree_id, CatalogState::Live).to_vec(),
                },
                BatchOp::Put {
                    key: CATALOG_NEXT_TREE_ID_KEY.to_vec(),
                    value: encode_next_tree_id(next_allocated_tree_id(tree_id)?).to_vec(),
                },
            ],
        )?;
        self.catalog_cache.lock().unwrap().insert(
            name.to_owned(),
            CatalogEntry {
                tree_id,
                state: CatalogState::Live,
            },
        );
        let open = self.open_tree_state(tree_id)?;
        self.tree_from_state(tree_id, open)
    }

    fn allocate_tree_id(&self) -> Result<u64> {
        let tree_id = self.catalog_next_tree_id()?;
        if tree_id == 0 || tree_id == DB_CATALOG_TREE_ID {
            return Err(Error::node_corrupt("db catalog next tree id"));
        }
        Ok(tree_id)
    }

    fn catalog_next_tree_id(&self) -> Result<u64> {
        let catalog = self.catalog_tree()?;
        catalog
            .get(CATALOG_NEXT_TREE_ID_KEY)?
            .map(|value| decode_next_tree_id(&value))
            .transpose()
            .map(|id| id.unwrap_or(FIRST_USER_TREE_ID))
    }

    /// Open an existing named tree inside this DB.
    ///
    /// Use [`Self::open_or_create_tree`] when lazy creation is the
    /// desired behavior.
    pub fn open_tree(&self, name: &str) -> Result<Tree> {
        let name_bytes = validate_tree_name(name)?;
        // Serialize the catalog Live decision with drop_tree's exclusive
        // maintenance transition through runtime lookup and Tree construction.
        // Once this shared guard releases, a queued drop can mark the exact
        // runtime returned below as dropped before it returns to its caller.
        let _maintenance = self.maintenance_gate.enter_shared();
        let tree_id = self
            .catalog_lookup_live(name_bytes)?
            .ok_or_else(|| Error::TreeNotFound {
                name: name.to_owned(),
            })?;
        #[cfg(test)]
        pause_open_tree_after_catalog_lookup();
        let open = self.open_tree_state(tree_id)?;
        self.tree_from_state(tree_id, open)
    }

    /// Open a named tree, creating it when the catalog has no entry.
    pub fn open_or_create_tree(&self, name: &str) -> Result<Tree> {
        match self.open_tree(name) {
            Ok(tree) => Ok(tree),
            Err(Error::TreeNotFound { .. }) => match self.create_tree(name) {
                Ok(tree) => Ok(tree),
                Err(Error::TreeExists { .. }) => self.open_tree(name),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        }
    }

    /// Return every named tree recorded in the durable catalog.
    pub fn list_trees(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for (key, entry) in self.catalog_entries()? {
            if entry.state == CatalogState::Live {
                let name =
                    String::from_utf8(key).map_err(|_| Error::node_corrupt("db catalog key"))?;
                names.push(name);
            }
        }
        Ok(names)
    }

    /// Mark a named tree `Dropping` in the durable catalog.
    ///
    /// The catalog tombstone is hidden from [`Self::list_trees`] and
    /// from [`Self::open_tree`]. Existing handles are fenced before
    /// this call returns. A later [`Self::checkpoint`] or [`Self::gc`]
    /// reclaims the unreachable closure and removes the catalog tombstone
    /// after old handles and iterators release the exact live-root pin.
    /// Snapshot roots are protected independently and do not by themselves
    /// retain a `Dropping` family's live root.
    pub fn drop_tree(&self, name: &str) -> Result<()> {
        self.ensure_writable()?;
        let name_bytes = validate_tree_name(name)?;
        let _maintenance = self.maintenance_gate.enter_exclusive();
        self.ensure_ordinary_writes_allowed()?;
        let entry = match self.catalog_entry(name_bytes)? {
            Some(entry) if entry.state == CatalogState::Live => entry,
            Some(_) | None => {
                return Err(Error::TreeNotFound {
                    name: name.to_owned(),
                });
            }
        };
        // Publish every acknowledged deferred write before the tree becomes
        // Dropping. Later cleanup is reachability-based and must not depend
        // on a writer path that `mark_runtime_dropped` has fenced.
        {
            let _commit = self.commit_gate.enter_writer();
            self.store.flush_write_deltas_for_tree(entry.tree_id)?;
        }
        self.apply_system_batch_unlocked(
            DB_CATALOG_TREE_ID,
            vec![BatchOp::Put {
                key: name_bytes.to_vec(),
                value: encode_catalog_value(entry.tree_id, CatalogState::Dropping).to_vec(),
            }],
        )?;
        self.catalog_cache.lock().unwrap().insert(
            name.to_owned(),
            CatalogEntry {
                tree_id: entry.tree_id,
                state: CatalogState::Dropping,
            },
        );
        self.mark_runtime_dropped(entry.tree_id);
        Ok(())
    }

    /// Apply mutations across named trees under one WAL record.
    ///
    /// The closure buffers operations in a [`DBAtomicBatch`]. Holt
    /// validates all guards for every touched tree before applying
    /// any mutation; if a guard fails, the method returns `Ok(false)`
    /// and emits no WAL record.
    ///
    /// Every error is returned as [`Error::Atomic`].
    /// [`AtomicErrorKind::DefinitelyNotApplied`] means none of this batch's
    /// mutations ran, so callers may retry after normal guard revalidation.
    /// [`AtomicErrorKind::OutcomeUnknown`] means Holt entered the apply path;
    /// callers must reconcile state instead of retrying blindly. The
    /// classification applies only to this batch: Holt may flush writes
    /// acknowledged by earlier calls before it starts the batch.
    pub fn atomic<F>(&self, build: F) -> Result<bool>
    where
        F: FnOnce(&mut DBAtomicBatch),
    {
        self.ensure_writable()
            .map_err(atomic_definitely_not_applied)?;
        let mut batch = DBAtomicBatch::default();
        build(&mut batch);
        if batch.pending.is_empty() {
            return Ok(true);
        }
        self.apply_atomic(batch.pending)
    }

    /// Initialize the application recovery stream at one durable anchor.
    ///
    /// A file-backed DB stores the anchor in its WAL header. A memory DB keeps
    /// the stream only for that DB's process lifetime. In both profiles the DB
    /// must be writable and cleanly checkpointed before first initialization.
    /// An exact retry with the same anchor succeeds; a different anchor fails
    /// closed. A file I/O error can leave initialization incomplete. In that
    /// state Holt rejects ordinary writes, attached writes, state reads, and
    /// scans until an exact retry durably repairs the anchor mirror.
    pub fn initialize_journal_stream(&self, genesis: JournalAnchor) -> Result<()> {
        self.ensure_writable()?;
        let _maintenance = self.maintenance_gate.enter_exclusive();
        let _commit = self.commit_gate.enter_checkpoint();

        if let Some(journal) = &self.journal {
            if journal.state().is_some() {
                return journal.initialize_anchor(genesis);
            }
        } else if let Some(journal) = &self.volatile_journal {
            if journal.state().is_some() {
                return journal.initialize(genesis);
            }
        } else {
            return Err(Error::JournalStreamUnavailable {
                reason: "file-backed WAL or memory storage is required",
            });
        }

        if self
            .catalog_entries_unlocked()?
            .iter()
            .any(|(_, entry)| entry.state == CatalogState::Dropping)
        {
            return Err(Error::JournalStreamUnavailable {
                reason: "dropping trees must be finalized before stream initialization",
            });
        }

        if self.store.dirty_count() != 0
            || self.store.flushing_count() != 0
            || self.store.pending_delete_count() != 0
            || self.store.orphan_staging_count() != 0
            || self.store.write_delta_count() != 0
            || self.store.needs_flush()
            || self
                .journal
                .as_ref()
                .is_some_and(|journal| journal.needs_checkpoint())
        {
            return Err(Error::JournalStreamUnavailable {
                reason: "database must be cleanly checkpointed before stream initialization",
            });
        }
        if let Some(journal) = &self.journal {
            journal.initialize_anchor(genesis)
        } else {
            self.volatile_journal
                .as_ref()
                .expect("memory journal checked above")
                .initialize(genesis)
        }
    }

    /// Return the checkpoint floor and current attached-stream tail.
    pub fn journal_state(&self) -> Result<JournalState> {
        if self.cfg.is_read_only() {
            let path = self.cfg.wal_path().ok_or(Error::JournalStreamUnavailable {
                reason: "file-backed WAL or memory storage is required",
            })?;
            return crate::journal::reader::attached_journal_state(&path);
        }
        let _commit = self.commit_gate.enter_checkpoint();
        let state = if let Some(journal) = &self.journal {
            journal.state()
        } else if let Some(journal) = &self.volatile_journal {
            journal.state()
        } else {
            None
        };
        state.ok_or(Error::JournalStreamUnavailable {
            reason: "stream has not been initialized",
        })
    }

    /// Read a bounded page of retained recovery envelopes after `cursor`.
    ///
    /// A cursor before the local checkpoint floor returns
    /// [`Error::JournalPositionExpired`]. This local API does not imply remote
    /// or shared-log retention. `row_limit` must be non-zero.
    /// `payload_byte_limit` is a soft boundary over payload bytes: the first
    /// retained envelope is returned even when its payload exceeds the limit,
    /// including a zero-byte limit.
    pub fn journal_envelopes_after(
        &self,
        cursor: JournalAnchor,
        row_limit: usize,
        payload_byte_limit: usize,
    ) -> Result<JournalEnvelopePage> {
        if self.cfg.is_read_only() {
            let path = self.cfg.wal_path().ok_or(Error::JournalStreamUnavailable {
                reason: "file-backed WAL or memory storage is required",
            })?;
            return crate::journal::reader::scan_attached_envelope_page(
                &path,
                cursor,
                row_limit,
                payload_byte_limit,
            );
        }
        // Stabilize both the committed ring prefix and checkpoint truncation
        // for the complete scan. Journal then flushes that prefix before it
        // opens the WAL file.
        let _commit = self.commit_gate.enter_checkpoint();
        if let Some(journal) = &self.journal {
            journal.envelopes_after(cursor, row_limit, payload_byte_limit)
        } else if let Some(journal) = &self.volatile_journal {
            journal.envelopes_after(cursor, row_limit, payload_byte_limit)
        } else {
            Err(Error::JournalStreamUnavailable {
                reason: "file-backed WAL or memory storage is required",
            })
        }
    }

    /// Apply one guarded DB batch and attach canonical application recovery
    /// bytes to the same WAL record.
    ///
    /// Holt validates the caller's previous anchor before any mutation. A
    /// failed batch guard returns `Ok(false)` without emitting an envelope.
    /// Every error uses the same definitely-not-applied versus outcome-unknown
    /// boundary as [`Self::atomic`].
    pub fn atomic_with_journal_envelope<F>(
        &self,
        envelope: JournalEnvelope,
        build: F,
    ) -> Result<bool>
    where
        F: FnOnce(&mut DBAtomicBatch),
    {
        self.ensure_writable()
            .map_err(atomic_definitely_not_applied)?;
        let mut batch = DBAtomicBatch::default();
        build(&mut batch);
        if batch.pending.is_empty() {
            return Err(atomic_definitely_not_applied(Error::Internal(
                "attached journal batch must mutate database state",
            )));
        }
        self.apply_atomic_with_journal_envelope(batch.pending, envelope)
    }

    /// Run a read-only transaction over explicit tree/prefix scopes.
    ///
    /// Holt captures every listed scope while holding each touched
    /// tree's exclusive mutation gate, releases the live DB, then
    /// invokes `read` with an immutable [`DBView`]. Writes committed
    /// after the capture are invisible to every captured tree view.
    /// Cloned tree views and their range builders or owned cursors may escape
    /// the callback and retain their process-local snapshot epoch leases until
    /// the final derived handle is dropped.
    ///
    /// Scopes are explicit so callers choose exactly which catalog
    /// trees participate in the consistent read view.
    pub fn view<F, R>(&self, scopes: &[(&str, &[u8])], read: F) -> Result<R>
    where
        F: FnOnce(&DBView) -> Result<R>,
    {
        let view = {
            let _maintenance = self.maintenance_gate.enter_shared();
            let mut scoped = Vec::with_capacity(scopes.len());
            for (name, prefix) in scopes {
                let name_bytes = validate_tree_name(name)?;
                let tree_id =
                    self.catalog_lookup_live(name_bytes)?
                        .ok_or_else(|| Error::TreeNotFound {
                            name: (*name).to_owned(),
                        })?;
                let open = self.open_tree_state(tree_id)?;
                let tree = self.tree_from_state(tree_id, open)?;
                scoped.push((tree_id, (*name).to_owned(), *prefix, tree));
            }
            let mut gates = scoped
                .iter()
                .map(|(tree_id, _, _, tree)| (*tree_id, tree.mutation_gate()))
                .collect::<Vec<_>>();
            gates.sort_by_key(|(tree_id, _)| *tree_id);
            gates.dedup_by_key(|(tree_id, _)| *tree_id);
            let _tree_guards = gates
                .iter()
                .map(|(_, gate)| gate.enter_exclusive())
                .collect::<Vec<_>>();
            let _commit = self.commit_gate.enter_writer();
            for (tree_id, _, _, _) in &scoped {
                self.store.flush_write_deltas_for_tree(*tree_id)?;
            }
            let mut trees = HashMap::with_capacity(scoped.len());
            for (_, name, prefix, tree) in scoped {
                trees.insert(name, tree.snapshot_unlocked(prefix)?);
            }
            DBView { trees }
        };
        read(&view)
    }

    /// Reclaim persisted frames not reachable from the catalog, each live
    /// tree's root, or a live snapshot root — the DB-wide analog of
    /// [`crate::Tree::gc`].
    ///
    /// GC freezes every tree, checkpoints the exact parent images it will
    /// walk, and only then deletes unreachable blobs. This prevents a
    /// durable old parent from losing a child that only a newer in-memory
    /// parent stopped referencing. Returns the count reclaimed and is
    /// idempotent. Concurrent create/drop operations serialize behind the
    /// same DB maintenance fence.
    pub fn gc(&self) -> Result<usize> {
        self.ensure_writable()?;
        self.restore_dropping_runtime_fences()?;
        let (freed, cleanup_complete) = self.gc_reachability_pass(usize::MAX, true)?;
        if cleanup_complete && self.finalize_dropped_trees()? {
            Tree::checkpoint_shared_store(
                &self.store,
                self.journal.as_ref(),
                &self.maintenance_gate,
                &self.commit_gate,
            )?;
        }
        Ok(freed)
    }

    /// Reclaim logical garbage and physical space from free store slots.
    ///
    /// This is the DB-wide analog of [`Tree::vacuum`](crate::Tree::vacuum):
    /// it collects reachability across the catalog, every live named tree,
    /// and live snapshots, checkpoints the shared store, then asks the
    /// file backend to relocate live high-water slots into lower reusable
    /// holes, truncate durably free packed-file tails, and, where supported,
    /// hole-punch remaining reusable middle slots. GUID/key visibility is
    /// unchanged.
    pub fn vacuum(&self) -> Result<VacuumStats> {
        self.ensure_writable()?;
        let unreachable = self.gc()?;
        let mut stats = self.store.vacuum_storage()?;
        stats.unreachable_blobs = unreachable;
        Ok(stats)
    }

    /// Export a consistent point-in-time image of every live family.
    ///
    /// Each family is captured with a copy-on-write snapshot taken under a
    /// brief all-families freeze, so the image is a single consistent
    /// instant; serialization then runs *outside* the freeze while live
    /// applies continue (forking the frames the snapshots reference).
    pub fn export_checkpoint(&self) -> Result<CheckpointImage> {
        // The DB maintenance fence covers catalog enumeration, deferred-write
        // publication, and every O(1) snapshot capture. A concurrent create
        // therefore lands wholly before or after this exported generation;
        // it cannot be omitted after its data became visible.
        let snaps: Vec<(Vec<u8>, Snapshot)> = {
            let _maintenance = self.maintenance_gate.enter_exclusive();
            {
                let _commit = self.commit_gate.enter_writer();
                self.store.flush_write_deltas_for_tree(DB_CATALOG_TREE_ID)?;
            }
            let mut families: Vec<(Vec<u8>, u64, Tree)> = Vec::new();
            for (name, entry) in self.catalog_entries_unlocked()? {
                if entry.state == CatalogState::Live {
                    let open = self.open_tree_state(entry.tree_id)?;
                    families.push((
                        name,
                        entry.tree_id,
                        self.tree_from_state(entry.tree_id, open)?,
                    ));
                }
            }
            {
                let _commit = self.commit_gate.enter_writer();
                for (_, tree_id, _) in &families {
                    self.store.flush_write_deltas_for_tree(*tree_id)?;
                }
            }

            let mut snaps = Vec::with_capacity(families.len());
            for (name, _, tree) in &families {
                snaps.push((name.clone(), tree.snapshot_unlocked_unfenced(b"")?));
            }
            snaps
        };

        // Serialize after releasing the freeze — applies resume here.
        let mut buf = checkpoint::begin(snaps.len() as u32);
        for (name, snap) in &snaps {
            let mut block = Vec::new();
            for entry in snap.range() {
                if let RangeEntry::Key { key, value, .. } = entry? {
                    checkpoint::put_kv(&mut block, &key, &value);
                    #[cfg(test)]
                    pause_export_after_first_entry();
                }
            }
            checkpoint::put_family(&mut buf, name, &block);
        }
        Ok(CheckpointImage::from_raw(buf))
    }

    /// Install a checkpoint produced by [`Self::export_checkpoint`] into
    /// this fresh DB.
    ///
    /// Intended for a fresh / wiped DB: every family is recreated and
    /// repopulated. On error the partially-installed DB must be discarded
    /// and the install retried — do not serve from a half-installed DB.
    /// Holt does not yet provide online replacement of a live DB image.
    pub fn install_checkpoint(&self, image: &CheckpointImage) -> Result<()> {
        self.ensure_writable()?;
        self.ensure_ordinary_writes_allowed()?;
        let decoded = checkpoint::decode(image.as_bytes())?;
        for (name, kv) in &decoded.families {
            let name = std::str::from_utf8(name)
                .map_err(|_| Error::node_corrupt("checkpoint image: non-utf8 family name"))?;
            let tree = self.create_tree(name)?;
            for (key, value) in kv {
                tree.put(key, value)?;
            }
        }
        Ok(())
    }

    /// Force one DB-wide checkpoint round.
    ///
    /// This flushes the shared BufferManager, applies pending deletes, and
    /// truncates the shared WAL when it is safe. It then performs one bounded
    /// exact-orphan / Dropping-tree cleanup pass and durably finalizes any
    /// catalog removals completed by that pass. It is not tied to any one
    /// named tree.
    pub fn checkpoint(&self) -> Result<()> {
        self.ensure_writable()?;
        self.restore_dropping_runtime_fences()?;
        Tree::checkpoint_shared_store(
            &self.store,
            self.journal.as_ref(),
            &self.maintenance_gate,
            &self.commit_gate,
        )?;
        let (_, cleanup_complete) = self.gc_reachability_pass(AUTO_GC_BATCH_SIZE, false)?;
        if cleanup_complete
            && self.store.pending_delete_count() == 0
            && self.finalize_dropped_trees()?
        {
            Tree::checkpoint_shared_store(
                &self.store,
                self.journal.as_ref(),
                &self.maintenance_gate,
                &self.commit_gate,
            )?;
        }
        if let Some(journal) = &self.volatile_journal {
            // Bind the volatile floor to an exact clean in-memory store image.
            // The earlier DB maintenance passes release their gates between
            // phases, so a writer could otherwise land after the last flush
            // but before this floor advance.
            let _maintenance = self.maintenance_gate.enter_exclusive();
            Tree::checkpoint_shared_store_with_maintenance_held(
                &self.store,
                None,
                &self.commit_gate,
            )?;
            let _commit = self.commit_gate.enter_checkpoint();
            journal.checkpoint();
        }
        Ok(())
    }

    /// Run one online maintenance pass for the catalog and every
    /// named tree.
    pub fn compact(&self) -> Result<()> {
        self.ensure_writable()?;
        self.catalog_tree()?.compact()?;
        for name in self.list_trees()? {
            self.open_tree(&name)?.compact()?;
        }
        Ok(())
    }

    /// Snapshot shared DB resource counters.
    ///
    /// Shape counters remain available from each [`Tree::stats`]
    /// because blob topology is root-specific. `DBStats` reports
    /// the shared WAL, checkpoint, and BufferManager counters.
    pub fn stats(&self) -> DBStats {
        let journal = self.journal.as_ref().map(|j| {
            let s = j.stats();
            JournalStats {
                appends: s.appends,
                batches: s.batches,
                syncs: s.syncs,
                queued_work: s.queued_work,
                written_work: s.written_work,
                flushed_work: s.flushed_work,
                checkpointed_work: s.checkpointed_work,
                pending_work: s.pending_work,
                checkpoint_debt: s.checkpoint_debt,
            }
        });
        let checkpointer = self.checkpointer.as_ref().map(|ck| CheckpointerStats {
            rounds_attempted: ck.rounds_attempted(),
            rounds_succeeded: ck.rounds_succeeded(),
            rounds_failed: ck.rounds_failed(),
            blobs_flushed: ck.blobs_flushed(),
            merges_total: ck.merges_total(),
            truncates: ck.truncates(),
            evictions: ck.evictions(),
            last_dirty_count: ck.last_dirty_count(),
            last_pending_delete_count: ck.last_pending_delete_count(),
            last_round_micros: ck.last_round_micros(),
        });
        let bm = self.store.stats();
        DBStats {
            open_tree_count: self
                .trees
                .lock()
                .unwrap()
                .iter()
                .filter(|(tree_id, open)| {
                    **tree_id != DB_CATALOG_TREE_ID && !open.runtime.is_dropped()
                })
                .count(),
            bm_dirty_count: bm.dirty_count,
            bm_pending_delete_count: bm.pending_delete_count,
            bm_gc_orphan_backlog_count: bm.gc_orphan_backlog_count,
            bm_gc_reclaimed_count: bm.gc_reclaimed_count,
            bm_gc_last_full_sweep_deferred_count: bm.gc_last_full_sweep_deferred_count,
            bm_write_delta_count: bm.write_delta_count,
            bm_read_index_token_count: bm.read_index_token_count,
            bm_read_index_cache_entries: bm.read_index_cache_entries,
            bm_read_index_cache_bytes: bm.read_index_cache_bytes,
            bm_read_index_cache_budget_bytes: bm.read_index_cache_budget_bytes,
            bm_read_page_cache_entries: bm.read_page_cache_entries,
            bm_read_page_cache_bytes: bm.read_page_cache_bytes,
            bm_read_page_cache_ghost_entries: bm.read_page_cache_ghost_entries,
            bm_read_page_cache_budget_bytes: bm.read_page_cache_budget_bytes,
            bm_cache_hits: bm.cache_hits,
            bm_cache_misses: bm.cache_misses,
            bm_full_blob_reads: bm.full_blob_reads,
            bm_full_blob_read_bytes: bm.full_blob_read_bytes,
            bm_point_full_blob_reads: bm.point_full_blob_reads,
            bm_scan_full_blob_reads: bm.scan_full_blob_reads,
            bm_silent_full_blob_reads: bm.silent_full_blob_reads,
            bm_read_page_hits: bm.read_page_hits,
            bm_read_page_misses: bm.read_page_misses,
            bm_read_index_cache_hits: bm.read_index_cache_hits,
            bm_read_index_cache_misses: bm.read_index_cache_misses,
            bm_read_index_loads: bm.read_index_loads,
            bm_read_index_dir_read_bytes: bm.read_index_dir_read_bytes,
            bm_read_index_bucket_reads: bm.read_index_bucket_reads,
            bm_read_index_bucket_read_bytes: bm.read_index_bucket_read_bytes,
            bm_read_index_inline_hits: bm.read_index_inline_hits,
            bm_read_index_value_hits: bm.read_index_value_hits,
            bm_read_index_value_read_bytes: bm.read_index_value_read_bytes,
            bm_read_index_offset_hits: bm.read_index_offset_hits,
            bm_read_index_negative_hits: bm.read_index_negative_hits,
            bm_read_index_crossing_hits: bm.read_index_crossing_hits,
            bm_read_index_unknowns: bm.read_index_unknowns,
            bm_optimistic_restarts: bm.optimistic_restarts,
            bm_range_restarts: bm.range_restarts,
            bm_walker_ops: bm.walker_ops,
            bm_walker_blob_hops: bm.walker_blob_hops,
            bm_max_blob_hops: bm.max_blob_hops,
            bm_max_cross_blob_depth: bm.max_cross_blob_depth,
            bm_spillovers: bm.spillovers,
            bm_merges: bm.merges,
            bm_route_resident_count: bm.route_resident_count,
            bm_route_resident_demotions: bm.route_resident_demotions,
            bm_cache_evictions: bm.cache_evictions,
            bm_eviction_skips_protected: bm.eviction_skips_protected,
            bm_eviction_skips_route_resident: bm.eviction_skips_route_resident,
            bm_admission_protects: bm.admission_protects,
            store: bm.store,
            open: self.open_stats,
            journal,
            checkpointer,
        }
    }

    fn catalog_tree(&self) -> Result<Tree> {
        let open = self.open_tree_state(DB_CATALOG_TREE_ID)?;
        self.tree_from_state(DB_CATALOG_TREE_ID, open)
    }

    fn restore_epoch_high_water(&self) -> Result<()> {
        let catalog_root = root_guid_for_tree_id(DB_CATALOG_TREE_ID);
        if !self.store.has_blob(catalog_root)? {
            if self.cfg.is_read_only() {
                return ensure_root_blob_for_open(&self.store, catalog_root, false);
            }
            let durable = self.store.store_blob_guids()?;
            let truly_fresh = durable.is_empty()
                && self.store.cached_count() == 0
                && self.store.dirty_count() == 0;
            if !truly_fresh {
                return Err(Error::node_corrupt(
                    "db catalog root missing from an existing store",
                ));
            }
            ensure_durable_root_blob(&self.store, catalog_root)?;
        }
        let mut roots = vec![catalog_root];
        for (_, entry) in self.catalog_entries()? {
            if entry.state == CatalogState::Dropping {
                // Bounded drop GC may durably remove descendants before the
                // root itself. A crash in that intermediate state leaves an
                // intentionally incomplete Dropping closure, which must not
                // make DB open fail before cleanup can resume. No process-
                // local handle or snapshot lease survives reopen, and this
                // family can never become live again, so its epoch cannot
                // constrain future COW decisions for surviving families.
                continue;
            }
            let root_guid = root_guid_for_tree_id(entry.tree_id);
            if !self.store.has_blob(root_guid)? {
                return Err(Error::node_corrupt(
                    "db catalog references a missing live tree root",
                ));
            }
            roots.push(root_guid);
        }

        // Root high-water is the fast durable summary, but a DB epoch is
        // shared across families: a high-epoch frame can be reachable from a
        // different root than the snapshot that advanced the epoch. Walk the
        // complete live closure so reopen is correct even after the family
        // that originally advanced the epoch was dropped in an older session.
        let mut frames = HashSet::new();
        for root in roots {
            frames.insert(root);
            frames.extend(crate::engine::collect_blob_guids(&self.store, root)?);
        }
        let mut high_water = 1u64;
        for guid in frames {
            let pin = self.store.pin(guid)?;
            let frame = pin.read();
            let root_high_water = crate::layout::frame_epoch_high_water(frame.as_slice());
            let created_epoch = crate::layout::frame_created_epoch(frame.as_slice());
            if root_high_water == u64::MAX || created_epoch == u64::MAX {
                return Err(Error::node_corrupt("snapshot epoch exhausted"));
            }
            high_water = high_water.max(root_high_water).max(created_epoch);
        }
        self.store.set_current_epoch(high_water);
        Ok(())
    }

    fn catalog_lookup_live(&self, name: &[u8]) -> Result<Option<u64>> {
        Ok(self
            .catalog_entry(name)?
            .and_then(|entry| (entry.state == CatalogState::Live).then_some(entry.tree_id)))
    }

    fn catalog_entry(&self, name: &[u8]) -> Result<Option<CatalogEntry>> {
        let name = std::str::from_utf8(name).map_err(|_| Error::node_corrupt("db catalog key"))?;
        if let Some(entry) = self.catalog_cache.lock().unwrap().get(name).copied() {
            return Ok(Some(entry));
        }
        let name_bytes = name.as_bytes();
        let catalog = self.catalog_tree()?;
        let entry = catalog
            .get(name_bytes)?
            .map(|value| decode_catalog_value(name_bytes, &value))
            .transpose()?;
        if let Some(entry) = entry {
            self.catalog_cache
                .lock()
                .unwrap()
                .insert(name.to_owned(), entry);
        }
        Ok(entry)
    }

    fn catalog_entries(&self) -> Result<Vec<(Vec<u8>, CatalogEntry)>> {
        let catalog = self.catalog_tree()?;
        let mut entries = Vec::new();
        for item in catalog.range() {
            if let RangeEntry::Key { key, value, .. } = item? {
                if key == CATALOG_NEXT_TREE_ID_KEY {
                    continue;
                }
                let entry = decode_catalog_value(&key, &value)?;
                let name = String::from_utf8(key.clone())
                    .map_err(|_| Error::node_corrupt("db catalog key"))?;
                self.catalog_cache.lock().unwrap().insert(name, entry);
                entries.push((key, entry));
            }
        }
        Ok(entries)
    }

    fn catalog_entries_unlocked(&self) -> Result<Vec<(Vec<u8>, CatalogEntry)>> {
        let catalog = self.catalog_tree()?;
        let mut cursor = catalog.range_unlocked();
        let mut entries = Vec::new();
        while let Some(item) = cursor.next_unlocked() {
            if let RangeEntry::Key { key, value, .. } = item? {
                if key == CATALOG_NEXT_TREE_ID_KEY {
                    continue;
                }
                let entry = decode_catalog_value(&key, &value)?;
                let name = String::from_utf8(key.clone())
                    .map_err(|_| Error::node_corrupt("db catalog key"))?;
                self.catalog_cache.lock().unwrap().insert(name, entry);
                entries.push((key, entry));
            }
        }
        Ok(entries)
    }

    fn restore_dropping_runtime_fences(&self) -> Result<()> {
        for (_, entry) in self.catalog_entries()? {
            if entry.state == CatalogState::Dropping {
                self.mark_runtime_dropped(entry.tree_id);
            }
        }
        Ok(())
    }

    fn finalize_dropped_trees(&self) -> Result<bool> {
        let _maintenance = self.maintenance_gate.enter_exclusive();
        {
            let _commit = self.commit_gate.enter_writer();
            self.store.flush_write_deltas_for_tree(DB_CATALOG_TREE_ID)?;
        }
        let mut ops = Vec::new();
        let mut finalized_tree_ids = Vec::new();
        let mut finalized_names = Vec::new();
        for (name, entry) in self.catalog_entries_unlocked()? {
            if entry.state == CatalogState::Dropping
                && !self
                    .store
                    .store_has_blob(root_guid_for_tree_id(entry.tree_id))?
            {
                let name_str = String::from_utf8(name.clone())
                    .map_err(|_| Error::node_corrupt("db catalog key"))?;
                ops.push(BatchOp::Delete { key: name });
                finalized_tree_ids.push(entry.tree_id);
                finalized_names.push(name_str);
            }
        }
        if ops.is_empty() {
            return Ok(false);
        }
        self.apply_system_batch_unlocked(DB_CATALOG_TREE_ID, ops)?;
        let mut cache = self.catalog_cache.lock().unwrap();
        for name in finalized_names {
            cache.remove(&name);
        }
        drop(cache);
        let mut trees = self.trees.lock().unwrap();
        for tree_id in finalized_tree_ids {
            trees.remove(&tree_id);
        }
        Ok(true)
    }

    /// Freeze the catalog/tree set, make that exact topology durable, and
    /// reclaim at most `limit` blobs outside its pinned-root closure.
    /// `Dropping` roots with an exact live pin are treated as canonical for
    /// this pass. Snapshot roots are walked independently, so a snapshot in
    /// one family never forces traversal of an unrelated Dropping closure.
    fn gc_reachability_pass(&self, limit: usize, force_full_scan: bool) -> Result<(usize, bool)> {
        let _maintenance = self.maintenance_gate.enter_exclusive();
        {
            let _commit = self.commit_gate.enter_writer();
            self.store.flush_write_deltas_for_tree(DB_CATALOG_TREE_ID)?;
        }
        let entries = self.catalog_entries_unlocked()?;
        for (_, entry) in &entries {
            if entry.state == CatalogState::Dropping {
                self.mark_runtime_dropped(entry.tree_id);
            }
        }
        {
            let _commit = self.commit_gate.enter_writer();
            for (_, entry) in &entries {
                self.store.flush_write_deltas_for_tree(entry.tree_id)?;
            }
        }
        Tree::checkpoint_shared_store_with_maintenance_held(
            &self.store,
            self.journal.as_ref(),
            &self.commit_gate,
        )?;

        let mut reachable = HashSet::new();
        let catalog_root = root_guid_for_tree_id(DB_CATALOG_TREE_ID);
        reachable.insert(catalog_root);
        reachable.extend(self.collect_tree_guids(DB_CATALOG_TREE_ID)?);
        let snapshot_roots = self.store.snapshot_roots_pinned()?;
        for (_, entry) in &entries {
            let root = root_guid_for_tree_id(entry.tree_id);
            let retain = entry.state == CatalogState::Live
                || (entry.state == CatalogState::Dropping && self.store.blob_is_pinned(root));
            if retain {
                reachable.insert(root);
                reachable.extend(self.collect_tree_guids(entry.tree_id)?);
            }
        }
        let canonical_reachable = reachable.clone();
        for snapshot_root in snapshot_roots {
            let root = snapshot_root.guid();
            reachable.insert(root);
            reachable.extend(crate::engine::collect_blob_guids(&self.store, root)?);
        }
        if force_full_scan
            || entries
                .iter()
                .any(|(_, entry)| entry.state == CatalogState::Dropping)
        {
            let outcome = self.store.gc_sweep_unreachable_with_canonical_bounded(
                &reachable,
                &canonical_reachable,
                limit,
            )?;
            Ok((outcome.freed, outcome.complete))
        } else {
            self.store
                .reclaim_retired_orphans_bounded(limit)
                .map(|freed| (freed, true))
        }
    }

    fn collect_tree_guids(&self, tree_id: u64) -> Result<Vec<BlobGuid>> {
        let root_guid = root_guid_for_tree_id(tree_id);
        if !self.store.has_blob(root_guid)? {
            return Ok(Vec::new());
        }
        crate::engine::collect_blob_guids(&self.store, root_guid)
    }

    fn mark_runtime_dropped(&self, tree_id: u64) {
        if let Some(open) = self.trees.lock().unwrap().get(&tree_id) {
            open.runtime.mark_dropped();
        }
    }

    fn open_tree_state(&self, tree_id: u64) -> Result<OpenTree> {
        let mut trees = self.trees.lock().unwrap();
        if let Some(open) = trees.get(&tree_id) {
            if !open.runtime.is_dropped() {
                return Ok(open.clone());
            }
            return Err(Error::TreeDropped);
        }
        let root_guid = root_guid_for_tree_id(tree_id);
        ensure_root_blob_for_open(&self.store, root_guid, !self.cfg.is_read_only())?;
        let open = OpenTree {
            root_guid,
            runtime: TreeRuntime::new(),
        };
        trees.insert(tree_id, open.clone());
        Ok(open)
    }

    fn tree_from_state(&self, tree_id: u64, open: OpenTree) -> Result<Tree> {
        Tree::from_shared(
            self.cfg.clone(),
            open.root_guid,
            tree_id,
            Arc::clone(&self.store),
            open.runtime,
            Arc::clone(&self.maintenance_gate),
            Arc::clone(&self.next_seq),
            Arc::clone(&self.commit_gate),
            self.journal.clone(),
            self.volatile_journal.clone(),
            self.checkpointer.clone(),
            self.open_stats,
        )
    }

    fn apply_atomic(&self, pending: Vec<DBBatchOp>) -> Result<bool> {
        let _maintenance = self.maintenance_gate.enter_shared();
        self.ensure_ordinary_writes_allowed()
            .map_err(atomic_definitely_not_applied)?;
        let groups = self
            .group_batch_ops(pending)
            .map_err(atomic_definitely_not_applied)?;
        let count = count_wal_ops(&groups);
        if count != 0 {
            if let Some(journal) = &self.journal {
                journal
                    .preflight_submit(encoded_db_batch_record_len(&groups))
                    .map_err(atomic_definitely_not_applied)?;
            }
        }
        let mut gates = groups
            .iter()
            .map(|group| (group.tree_id, group.tree.mutation_gate()))
            .collect::<Vec<_>>();
        gates.sort_by_key(|(tree_id, _)| *tree_id);
        gates.dedup_by_key(|(tree_id, _)| *tree_id);
        let _tree_guards = gates
            .iter()
            .map(|(_, gate)| gate.enter_batch())
            .collect::<Vec<_>>();
        {
            let _commit = self.commit_gate.enter_writer();
            for group in &groups {
                self.store
                    .flush_write_deltas_for_tree(group.tree_id)
                    .map_err(atomic_definitely_not_applied)?;
            }
        }
        let base_seq = self.next_seq.fetch_add(count, Ordering::Relaxed);
        if !Self::preflight_batch_groups(&groups, base_seq)
            .map_err(atomic_definitely_not_applied)?
        {
            return Ok(false);
        }
        if count == 0 {
            return Ok(true);
        }

        let apply = if let Some(journal) = &self.journal {
            self.apply_batch_groups_with_journal(&groups, base_seq, journal)
        } else {
            self.apply_batch_groups_in_memory(&groups, base_seq)
        };
        apply.map_err(atomic_outcome_unknown)?;
        Ok(true)
    }

    fn apply_atomic_with_journal_envelope(
        &self,
        pending: Vec<DBBatchOp>,
        envelope: JournalEnvelope,
    ) -> Result<bool> {
        let _maintenance = self.maintenance_gate.enter_shared();
        if self.journal.is_none() && self.volatile_journal.is_none() {
            return Err(atomic_definitely_not_applied(
                Error::JournalStreamUnavailable {
                    reason: "file-backed WAL or memory storage is required",
                },
            ));
        }
        let groups = self
            .group_batch_ops(pending)
            .map_err(atomic_definitely_not_applied)?;
        let count = count_wal_ops(&groups);
        if count == 0 {
            return Err(atomic_definitely_not_applied(Error::Internal(
                "attached journal batch must contain a mutation",
            )));
        }
        let record_len = encoded_db_batch_record_len(&groups)
            .checked_add(encoded_journal_envelope_len(&envelope))
            .ok_or_else(|| {
                atomic_definitely_not_applied(Error::Internal(
                    "attached journal record length overflow",
                ))
            })?;
        let mut gates = groups
            .iter()
            .map(|group| (group.tree_id, group.tree.mutation_gate()))
            .collect::<Vec<_>>();
        gates.sort_by_key(|(tree_id, _)| *tree_id);
        gates.dedup_by_key(|(tree_id, _)| *tree_id);
        let _tree_guards = gates
            .iter()
            .map(|(_, gate)| gate.enter_batch())
            .collect::<Vec<_>>();
        {
            let _commit = self.commit_gate.enter_writer();
            for group in &groups {
                self.store
                    .flush_write_deltas_for_tree(group.tree_id)
                    .map_err(atomic_definitely_not_applied)?;
            }
        }
        let base_seq = self.next_seq.fetch_add(count, Ordering::Relaxed);
        if !Self::preflight_batch_groups(&groups, base_seq)
            .map_err(atomic_definitely_not_applied)?
        {
            return Ok(false);
        }

        if let Some(journal) = &self.journal {
            // Lock order is commit gate -> attached-tail gate. Checkpoint takes
            // the commit gate exclusively before it snapshots the tail, so the
            // reverse order would deadlock a writer against WAL truncation.
            let ack = {
                let _commit = self.commit_gate.enter_writer();
                let append = journal
                    .begin_attached(envelope.previous(), record_len)
                    .map_err(atomic_definitely_not_applied)?;
                self.apply_batch_groups_with_attached_journal(
                    &groups, base_seq, append, envelope, record_len,
                )
                .map_err(atomic_outcome_unknown)?
            };
            if let Some(ack) = ack {
                ack.wait().map_err(atomic_outcome_unknown)?;
            }
        } else {
            let journal = self.volatile_journal.as_ref().ok_or_else(|| {
                atomic_definitely_not_applied(Error::JournalStreamUnavailable {
                    reason: "memory stream is unavailable",
                })
            })?;
            {
                let _commit = self.commit_gate.enter_writer();
                let append = journal
                    .begin_attached(envelope.previous(), record_len)
                    .map_err(atomic_definitely_not_applied)?;
                let mut group_base = base_seq;
                for group in &groups {
                    group
                        .tree
                        .apply_batch_walker_inline(&group.ops, group_base, None)
                        .map_err(atomic_outcome_unknown)?;
                    group_base += count_group_wal_ops(group);
                }
                append.submit(envelope).map_err(atomic_outcome_unknown)?;
            }
            if self.cfg.memory_flush_on_write {
                if let Some(group) = groups.first() {
                    group.tree.flush_inline().map_err(atomic_outcome_unknown)?;
                }
            }
        }
        Ok(true)
    }

    fn group_batch_ops(&self, pending: Vec<DBBatchOp>) -> Result<Vec<DBBatchGroup>> {
        let mut groups: Vec<DBBatchGroup> = Vec::new();
        let mut group_by_name: HashMap<String, usize> =
            HashMap::with_capacity(pending.len().min(16));
        for item in pending {
            let DBBatchOp { tree_name, op } = item;
            if let Some(&group_idx) = group_by_name.get(tree_name.as_str()) {
                groups[group_idx].ops.push(op);
                continue;
            }

            let name_bytes = validate_tree_name(&tree_name)?;
            let tree_id =
                self.catalog_lookup_live(name_bytes)?
                    .ok_or_else(|| Error::TreeNotFound {
                        name: tree_name.clone(),
                    })?;
            let open = self.open_tree_state(tree_id)?;
            let group_idx = groups.len();
            group_by_name.insert(tree_name, group_idx);
            groups.push(DBBatchGroup {
                tree_id,
                tree: self.tree_from_state(tree_id, open)?,
                ops: vec![op],
            });
        }
        Ok(groups)
    }

    fn preflight_batch_groups(groups: &[DBBatchGroup], base_seq: u64) -> Result<bool> {
        let mut group_base = base_seq;
        for group in groups {
            if !group.tree.preflight_batch(&group.ops, group_base)? {
                return Ok(false);
            }
            group_base += count_group_wal_ops(group);
        }
        Ok(true)
    }

    fn apply_batch_groups_with_journal(
        &self,
        groups: &[DBBatchGroup],
        base_seq: u64,
        journal: &Arc<Journal>,
    ) -> Result<()> {
        let ack = {
            let _commit = self.commit_gate.enter_writer();
            let mut record = journal.record_buffer(encoded_db_batch_record_len(groups));
            let mut enc = BatchEncoder::begin(&mut record, base_seq, 0);
            let mut group_base = base_seq;
            for group in groups {
                group
                    .tree
                    .apply_batch_walker_inline(&group.ops, group_base, Some(&mut enc))?;
                group_base += count_group_wal_ops(group);
            }
            let _n = enc.finish();
            journal.submit(record, self.cfg.durability.wal_sync())?
        };
        if let Some(ack) = ack {
            ack.wait()?;
        }
        Ok(())
    }

    fn apply_batch_groups_with_attached_journal(
        &self,
        groups: &[DBBatchGroup],
        base_seq: u64,
        append: crate::journal::JournalAppendGuard<'_>,
        envelope: JournalEnvelope,
        record_len: usize,
    ) -> Result<Option<crate::journal::JournalAck>> {
        let record = encode_attached_batch_record(
            append.record_buffer(record_len),
            base_seq,
            0,
            envelope,
            |encoder| {
                let mut group_base = base_seq;
                for group in groups {
                    group
                        .tree
                        .apply_batch_walker_inline(&group.ops, group_base, Some(encoder))?;
                    group_base += count_group_wal_ops(group);
                }
                Ok(())
            },
        )?;
        append.submit(record, self.cfg.durability.wal_sync())
    }

    fn apply_batch_groups_in_memory(&self, groups: &[DBBatchGroup], base_seq: u64) -> Result<()> {
        let commit = (self.store.fork_barrier() != 0).then(|| self.commit_gate.enter_writer());
        let mut group_base = base_seq;
        for group in groups {
            group
                .tree
                .apply_batch_walker_inline(&group.ops, group_base, None)?;
            group_base += count_group_wal_ops(group);
        }
        drop(commit);
        if self.cfg.memory_flush_on_write {
            if let Some(group) = groups.first() {
                group.tree.flush_inline()?;
            }
        }
        Ok(())
    }

    fn apply_system_batch_unlocked(&self, tree_id: u64, ops: Vec<BatchOp>) -> Result<u64> {
        self.ensure_ordinary_writes_allowed()?;
        let open = {
            let mut trees = self.trees.lock().unwrap();
            if let Some(open) = trees.get(&tree_id) {
                open.clone()
            } else {
                let root_guid = root_guid_for_tree_id(tree_id);
                ensure_durable_root_blob(&self.store, root_guid)?;
                let open = OpenTree {
                    root_guid,
                    runtime: TreeRuntime::new(),
                };
                trees.insert(tree_id, open.clone());
                open
            }
        };
        let groups = vec![DBBatchGroup {
            tree_id,
            tree: self.tree_from_state(tree_id, open)?,
            ops,
        }];
        let count = count_wal_ops(&groups);
        if count != 0 {
            if let Some(journal) = &self.journal {
                journal.preflight_submit(encoded_db_batch_record_len(&groups))?;
            }
        }
        let base_seq = self.next_seq.fetch_add(count, Ordering::Relaxed);
        if !Self::preflight_batch_groups(&groups, base_seq)? {
            return Err(Error::Internal("system DB batch preflight failed"));
        }
        if let Some(journal) = &self.journal {
            self.apply_batch_groups_with_journal(&groups, base_seq, journal)?;
        } else {
            self.apply_batch_groups_in_memory(&groups, base_seq)?;
        }
        Ok(base_seq)
    }
}

/// Immutable read transaction over one or more named tree scopes.
///
/// Created by [`DB::view`]. Each captured tree is exposed as a
/// normal [`View`], so point lookup and range/list APIs stay the
/// same as single-tree snapshots. Each view owns a copied root, initially
/// shares descendants with its live tree, and retains a process-local epoch
/// lease through any cloned view, range builder, or owned cursor.
pub struct DBView {
    trees: HashMap<String, Snapshot>,
}

impl DBView {
    /// Return the captured view for `name`, if the caller listed it
    /// in [`DB::view`]'s scope array.
    #[must_use]
    pub fn tree(&self, name: &str) -> Option<&View> {
        self.trees.get(name).map(Snapshot::view)
    }

    /// Number of captured named tree views.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// `true` if no tree scopes were captured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }
}

struct DBBatchGroup {
    tree_id: u64,
    tree: Tree,
    ops: Vec<BatchOp>,
}

#[derive(Debug)]
struct DBBatchOp {
    tree_name: String,
    op: BatchOp,
}

/// Builder for [`DB::atomic`].
#[derive(Debug, Default)]
pub struct DBAtomicBatch {
    pending: Vec<DBBatchOp>,
}

impl DBAtomicBatch {
    /// Buffer a put in `tree`.
    pub fn put(&mut self, tree: &str, key: &[u8], value: &[u8]) {
        self.push(
            tree,
            BatchOp::Put {
                key: key.to_vec(),
                value: value.to_vec(),
            },
        );
    }

    /// Buffer a create-only put in `tree`.
    pub fn put_if_absent(&mut self, tree: &str, key: &[u8], value: &[u8]) {
        self.push(
            tree,
            BatchOp::PutIfAbsent {
                key: key.to_vec(),
                value: value.to_vec(),
            },
        );
    }

    /// Buffer a version-guarded update in `tree`.
    pub fn compare_and_put(
        &mut self,
        tree: &str,
        key: &[u8],
        expected: RecordVersion,
        value: &[u8],
    ) {
        self.push(
            tree,
            BatchOp::CompareAndPut {
                key: key.to_vec(),
                expected,
                value: value.to_vec(),
            },
        );
    }

    /// Buffer a delete in `tree`.
    pub fn delete(&mut self, tree: &str, key: &[u8]) {
        self.push(tree, BatchOp::Delete { key: key.to_vec() });
    }

    /// Buffer a version-guarded delete in `tree`.
    pub fn delete_if_version(&mut self, tree: &str, key: &[u8], expected: RecordVersion) {
        self.push(
            tree,
            BatchOp::DeleteIfVersion {
                key: key.to_vec(),
                expected,
            },
        );
    }

    /// Require that `key` has `expected` in `tree`.
    pub fn assert_version(&mut self, tree: &str, key: &[u8], expected: RecordVersion) {
        self.push(
            tree,
            BatchOp::AssertVersion {
                key: key.to_vec(),
                expected,
            },
        );
    }

    /// Require that `key` is absent in `tree`.
    pub fn assert_absent(&mut self, tree: &str, key: &[u8]) {
        self.push(tree, BatchOp::AssertAbsent { key: key.to_vec() });
    }

    /// Require that no live key starts with `prefix` in `tree`.
    pub fn assert_prefix_empty(&mut self, tree: &str, prefix: &[u8]) {
        self.push(
            tree,
            BatchOp::AssertPrefixEmpty {
                prefix: prefix.to_vec(),
            },
        );
    }

    /// Buffer a rename inside one named tree.
    pub fn rename(&mut self, tree: &str, src: &[u8], dst: &[u8], force: bool) {
        self.push(
            tree,
            BatchOp::Rename {
                src: src.to_vec(),
                dst: dst.to_vec(),
                force,
            },
        );
    }

    /// Number of buffered operations.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// `true` when no operations have been buffered.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn push(&mut self, tree: &str, op: BatchOp) {
        self.pending.push(DBBatchOp {
            tree_name: tree.to_owned(),
            op,
        });
    }
}

fn encoded_db_batch_record_len(groups: &[DBBatchGroup]) -> usize {
    let mut len = crate::journal::codec::RECORD_HEADER_SIZE + 8 + 4;
    for group in groups {
        for op in &group.ops {
            len += match op {
                BatchOp::Put { key, value }
                | BatchOp::PutIfAbsent { key, value }
                | BatchOp::CompareAndPut { key, value, .. } => {
                    1 + 8 + 4 + key.len() + 4 + value.len()
                }
                BatchOp::Delete { key } | BatchOp::DeleteIfVersion { key, .. } => {
                    1 + 8 + 4 + key.len()
                }
                BatchOp::Rename { src, dst, .. } => 1 + 8 + 4 + src.len() + 4 + dst.len() + 1,
                BatchOp::AssertVersion { .. }
                | BatchOp::AssertAbsent { .. }
                | BatchOp::AssertPrefixEmpty { .. } => 0,
            };
        }
    }
    len + crate::journal::codec::RECORD_FOOTER_SIZE
}

fn encoded_journal_envelope_len(envelope: &JournalEnvelope) -> usize {
    1 + 8
        + super::journal::JOURNAL_DIGEST_BYTES
        + 8
        + super::journal::JOURNAL_DIGEST_BYTES
        + 4
        + envelope.payload().len()
}

fn count_wal_ops(groups: &[DBBatchGroup]) -> u64 {
    groups.iter().map(count_group_wal_ops).sum::<u64>()
}

fn count_group_wal_ops(group: &DBBatchGroup) -> u64 {
    group.ops.iter().filter(|op| op.emits_wal()).count() as u64
}

fn root_guid_for_tree_id(tree_id: u64) -> BlobGuid {
    let mut guid = [0u8; 16];
    guid[0..8].copy_from_slice(&tree_id.to_le_bytes());
    guid[8..15].copy_from_slice(b"holt-db");
    guid[15] = DB_ROOT_TAG;
    guid
}

fn validate_tree_name(name: &str) -> Result<&[u8]> {
    if name.is_empty() {
        return Err(Error::InvalidTreeName { reason: "empty" });
    }
    if name.as_bytes().first() == Some(&0) {
        return Err(Error::InvalidTreeName {
            reason: "reserved prefix",
        });
    }
    Ok(name.as_bytes())
}

fn encode_catalog_value(tree_id: u64, state: CatalogState) -> [u8; CATALOG_VALUE_LEN] {
    let mut out = [0u8; CATALOG_VALUE_LEN];
    out[..CATALOG_VALUE_MAGIC.len()].copy_from_slice(CATALOG_VALUE_MAGIC);
    out[CATALOG_VALUE_MAGIC.len()] = match state {
        CatalogState::Live => CATALOG_STATE_LIVE,
        CatalogState::Dropping => CATALOG_STATE_DROPPING,
    };
    out[CATALOG_VALUE_MAGIC.len() + 1..].copy_from_slice(&tree_id.to_le_bytes());
    out
}

fn decode_catalog_value(_name: &[u8], value: &[u8]) -> Result<CatalogEntry> {
    if value.len() != CATALOG_VALUE_LEN
        || &value[..CATALOG_VALUE_MAGIC.len()] != CATALOG_VALUE_MAGIC
    {
        return Err(Error::node_corrupt("db catalog value"));
    }
    let state = match value[CATALOG_VALUE_MAGIC.len()] {
        CATALOG_STATE_LIVE => CatalogState::Live,
        CATALOG_STATE_DROPPING => CatalogState::Dropping,
        _ => return Err(Error::node_corrupt("db catalog state")),
    };
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&value[CATALOG_VALUE_MAGIC.len() + 1..]);
    let tree_id = u64::from_le_bytes(raw);
    if tree_id == 0 || tree_id == DB_CATALOG_TREE_ID {
        return Err(Error::node_corrupt("db catalog tree id"));
    }
    Ok(CatalogEntry { tree_id, state })
}

fn encode_next_tree_id(tree_id: u64) -> [u8; CATALOG_NEXT_ID_LEN] {
    let mut out = [0u8; CATALOG_NEXT_ID_LEN];
    out[..CATALOG_NEXT_ID_MAGIC.len()].copy_from_slice(CATALOG_NEXT_ID_MAGIC);
    out[CATALOG_NEXT_ID_MAGIC.len()..].copy_from_slice(&tree_id.to_le_bytes());
    out
}

fn decode_next_tree_id(value: &[u8]) -> Result<u64> {
    if value.len() != CATALOG_NEXT_ID_LEN
        || &value[..CATALOG_NEXT_ID_MAGIC.len()] != CATALOG_NEXT_ID_MAGIC
    {
        return Err(Error::node_corrupt("db catalog next tree id"));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&value[CATALOG_NEXT_ID_MAGIC.len()..]);
    let tree_id = u64::from_le_bytes(raw);
    if tree_id == 0 || tree_id == DB_CATALOG_TREE_ID {
        return Err(Error::node_corrupt("db catalog next tree id"));
    }
    Ok(tree_id)
}

fn next_allocated_tree_id(tree_id: u64) -> Result<u64> {
    let mut next = tree_id
        .checked_add(1)
        .ok_or(Error::Internal("DB tree id space exhausted"))?;
    if next == DB_CATALOG_TREE_ID {
        next = next
            .checked_add(1)
            .ok_or(Error::Internal("DB tree id space exhausted"))?;
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::blob_store::{AlignedBlobBuf, MemoryBlobStore};
    use std::collections::{BTreeMap, HashSet};
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::sync::{mpsc, Condvar};
    use std::thread;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    struct FailIoStore {
        inner: MemoryBlobStore,
        fail_read_guids: Mutex<HashSet<BlobGuid>>,
        last_failed_read: Mutex<Option<BlobGuid>>,
        fail_next_flush: AtomicBool,
    }

    impl FailIoStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                fail_read_guids: Mutex::new(HashSet::new()),
                last_failed_read: Mutex::new(None),
                fail_next_flush: AtomicBool::new(false),
            }
        }

        fn arm_read_failures(&self, guids: impl IntoIterator<Item = BlobGuid>) {
            self.fail_read_guids.lock().unwrap().extend(guids);
            *self.last_failed_read.lock().unwrap() = None;
        }

        fn clear_read_failures(&self) {
            self.fail_read_guids.lock().unwrap().clear();
        }

        fn take_last_failed_read(&self) -> Option<BlobGuid> {
            self.last_failed_read.lock().unwrap().take()
        }
    }

    impl BlobStore for FailIoStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            if self.fail_read_guids.lock().unwrap().contains(&guid) {
                *self.last_failed_read.lock().unwrap() = Some(guid);
                return Err(Error::BlobStoreIo(std::io::Error::other(
                    "injected blob read failure",
                )));
            }
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.inner.delete_blob(guid)
        }

        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            self.inner.list_blobs()
        }

        fn flush(&self) -> Result<()> {
            if self.fail_next_flush.swap(false, AtomicOrdering::AcqRel) {
                return Err(Error::BlobStoreIo(std::io::Error::other(
                    "injected root flush failure",
                )));
            }
            self.inner.flush()
        }

        fn needs_flush(&self) -> bool {
            self.inner.needs_flush()
        }

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    #[derive(Default)]
    struct BlockingCheckpointState {
        armed: bool,
        entered: bool,
        released: bool,
    }

    struct BlockingCheckpointStore {
        inner: MemoryBlobStore,
        state: Mutex<BlockingCheckpointState>,
        wake: Condvar,
    }

    impl BlockingCheckpointStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                state: Mutex::new(BlockingCheckpointState::default()),
                wake: Condvar::new(),
            }
        }

        fn arm(&self) {
            let mut state = self.state.lock().unwrap();
            state.armed = true;
            state.entered = false;
            state.released = false;
        }

        fn wait_until_blocked(&self, timeout: Duration) -> bool {
            let state = self.state.lock().unwrap();
            let (state, _) = self
                .wake
                .wait_timeout_while(state, timeout, |state| !state.entered)
                .unwrap();
            state.entered
        }

        fn release(&self) {
            let mut state = self.state.lock().unwrap();
            state.released = true;
            self.wake.notify_all();
        }

        fn block_if_armed(&self) {
            let mut state = self.state.lock().unwrap();
            if !state.armed {
                return;
            }
            state.armed = false;
            state.entered = true;
            self.wake.notify_all();
            while !state.released {
                state = self.wake.wait(state).unwrap();
            }
        }
    }

    impl BlobStore for BlockingCheckpointStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.block_if_armed();
            self.inner.write_blob(guid, src)
        }

        fn write_blobs(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            self.block_if_armed();
            self.inner.write_blobs(writes)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.inner.delete_blob(guid)
        }

        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            self.inner.list_blobs()
        }

        fn flush(&self) -> Result<()> {
            self.inner.flush()
        }

        fn needs_flush(&self) -> bool {
            self.inner.needs_flush()
        }

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    fn background_memory_config() -> TreeConfig {
        let mut cfg = TreeConfig::memory();
        cfg.memory_flush_on_write = false;
        cfg.checkpoint.enabled = true;
        cfg.checkpoint.idle_interval = Duration::from_secs(10);
        cfg.checkpoint.dirty_blob_threshold = 1;
        cfg.checkpoint.auto_vacuum = false;
        cfg
    }

    fn journal_anchor(sequence: u64, byte: u8) -> JournalAnchor {
        JournalAnchor::new(
            sequence,
            [byte; super::super::journal::JOURNAL_DIGEST_BYTES],
        )
    }

    fn assert_oversized_first_envelope_advances(db: &DB) {
        db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0xc0);
        let first = journal_anchor(1, 0xc1);
        let second = journal_anchor(2, 0xc2);
        let first_envelope = JournalEnvelope::new(genesis, first, vec![0x41; 64]).unwrap();
        let second_envelope = JournalEnvelope::new(first, second, vec![0x42; 32]).unwrap();
        db.initialize_journal_stream(genesis).unwrap();
        assert!(db
            .atomic_with_journal_envelope(first_envelope.clone(), |batch| {
                batch.put("data", b"first", b"value");
            })
            .unwrap());
        assert!(db
            .atomic_with_journal_envelope(second_envelope.clone(), |batch| {
                batch.put("data", b"second", b"value");
            })
            .unwrap());

        for payload_byte_limit in [0, 1] {
            let first_page = db
                .journal_envelopes_after(genesis, 8, payload_byte_limit)
                .unwrap();
            assert_eq!(
                first_page.envelopes(),
                std::slice::from_ref(&first_envelope)
            );
            assert_eq!(first_page.next(), first);
            assert!(first_page.has_more());

            let second_page = db
                .journal_envelopes_after(first, 8, payload_byte_limit)
                .unwrap();
            assert_eq!(
                second_page.envelopes(),
                std::slice::from_ref(&second_envelope)
            );
            assert_eq!(second_page.next(), second);
            assert!(!second_page.has_more());
        }

        assert!(matches!(
            db.journal_envelopes_after(genesis, 0, 1),
            Err(Error::InvalidJournalScanLimit { .. })
        ));
    }

    #[test]
    fn file_journal_byte_limit_is_soft_for_first_envelope() {
        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        cfg.durability = crate::Durability::Wal { sync: true };
        let db = DB::open(cfg).unwrap();

        assert_oversized_first_envelope_advances(&db);
    }

    #[test]
    fn volatile_journal_byte_limit_is_soft_for_first_envelope() {
        let db = DB::open(TreeConfig::memory()).unwrap();

        assert_oversized_first_envelope_advances(&db);
    }

    #[test]
    fn background_checkpoint_advances_volatile_floor_after_clean_image() {
        let db = DB::open(background_memory_config()).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0xc0);
        let first = journal_anchor(1, 0xc1);
        db.initialize_journal_stream(genesis).unwrap();
        assert!(db
            .atomic_with_journal_envelope(
                JournalEnvelope::new(genesis, first, b"first".to_vec()).unwrap(),
                |batch| batch.put("data", b"key", b"value"),
            )
            .unwrap());
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, first)
        );

        db.checkpointer.as_ref().unwrap().wake();
        let deadline = Instant::now() + Duration::from_secs(2);
        while db.journal_state().unwrap().checkpoint() != first {
            assert!(
                Instant::now() < deadline,
                "background checkpoint never advanced the volatile floor"
            );
            thread::sleep(Duration::from_millis(2));
        }

        assert_eq!(data.get(b"key").unwrap().as_deref(), Some(&b"value"[..]));
        assert!(matches!(
            db.journal_envelopes_after(genesis, 8, 4096),
            Err(Error::JournalPositionExpired {
                requested: 0,
                checkpoint: 1,
            })
        ));
    }

    #[test]
    fn concurrent_attached_write_does_not_outrun_volatile_checkpoint_image() {
        let store = Arc::new(BlockingCheckpointStore::new());
        let db = DB::open_with_blob_store(
            background_memory_config(),
            Arc::clone(&store) as Arc<dyn BlobStore>,
        )
        .unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0xd0);
        let first = journal_anchor(1, 0xd1);
        let second = journal_anchor(2, 0xd2);
        db.initialize_journal_stream(genesis).unwrap();

        store.arm();
        assert!(db
            .atomic_with_journal_envelope(
                JournalEnvelope::new(genesis, first, b"first".to_vec()).unwrap(),
                |batch| batch.put("data", b"first", b"one"),
            )
            .unwrap());
        db.checkpointer.as_ref().unwrap().wake();
        assert!(
            store.wait_until_blocked(Duration::from_secs(2)),
            "background checkpoint never reached the blocked store write"
        );
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, first),
            "floor advanced before the captured image reached the store"
        );

        let writer_db = db.clone();
        let (writer_tx, writer_rx) = mpsc::channel();
        let writer = thread::spawn(move || {
            writer_tx
                .send(writer_db.atomic_with_journal_envelope(
                    JournalEnvelope::new(first, second, b"second".to_vec()).unwrap(),
                    |batch| batch.put("data", b"second", b"two"),
                ))
                .unwrap();
        });
        writer_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("attached writer blocked behind checkpoint I/O")
            .unwrap();
        writer.join().unwrap();
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, second),
            "floor advanced across a concurrent dirty write"
        );

        store.release();
        db.checkpointer.as_ref().unwrap().wake();
        let deadline = Instant::now() + Duration::from_secs(2);
        while db.journal_state().unwrap().checkpoint() != second {
            assert!(
                Instant::now() < deadline,
                "background checkpoint never caught up to the concurrent tail"
            );
            thread::sleep(Duration::from_millis(2));
        }

        assert_eq!(data.get(b"first").unwrap().as_deref(), Some(&b"one"[..]));
        assert_eq!(data.get(b"second").unwrap().as_deref(), Some(&b"two"[..]));
        assert!(matches!(
            db.journal_envelopes_after(genesis, 8, 4096),
            Err(Error::JournalPositionExpired {
                requested: 0,
                checkpoint: 2,
            })
        ));
    }

    #[test]
    fn attached_journal_batch_advances_once_and_survives_checkpoint_reopen() {
        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        cfg.durability = crate::Durability::Wal { sync: true };

        let db = DB::open(cfg.clone()).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0x10);
        let first = journal_anchor(1, 0x11);
        db.initialize_journal_stream(genesis).unwrap();
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, genesis)
        );

        let first_envelope =
            JournalEnvelope::new(genesis, first, b"first canonical command".to_vec()).unwrap();
        assert!(db
            .atomic_with_journal_envelope(first_envelope.clone(), |batch| {
                batch.put_if_absent("data", b"key", b"value");
            })
            .unwrap());
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, first)
        );
        let page = db.journal_envelopes_after(genesis, 8, 4096).unwrap();
        assert_eq!(page.envelopes(), &[first_envelope]);
        assert_eq!(page.next(), first);
        assert!(!page.has_more());

        let second = journal_anchor(2, 0x12);
        let rejected = JournalEnvelope::new(first, second, b"guarded command".to_vec()).unwrap();
        assert!(!db
            .atomic_with_journal_envelope(rejected.clone(), |batch| {
                batch.put_if_absent("data", b"key", b"replacement");
            })
            .unwrap());
        assert_eq!(db.journal_state().unwrap().tail(), first);
        assert!(db
            .journal_envelopes_after(first, 8, 4096)
            .unwrap()
            .envelopes()
            .is_empty());

        db.checkpoint().unwrap();
        assert_eq!(db.journal_state().unwrap(), JournalState::new(first, first));
        assert!(matches!(
            db.journal_envelopes_after(genesis, 8, 4096),
            Err(Error::JournalPositionExpired {
                requested: 0,
                checkpoint: 1,
            })
        ));
        drop(data);
        drop(db);

        let reopened = DB::open(cfg).unwrap();
        assert_eq!(
            reopened.journal_state().unwrap(),
            JournalState::new(first, first)
        );
        assert_eq!(
            reopened.open_tree("data").unwrap().get(b"key").unwrap(),
            Some(b"value".to_vec())
        );
        assert!(reopened
            .atomic_with_journal_envelope(rejected, |batch| {
                batch.put("data", b"next", b"value-2");
            })
            .unwrap());
        assert_eq!(reopened.journal_state().unwrap().tail(), second);
    }

    #[test]
    fn attached_journal_rejects_missing_or_stale_stream_before_mutation() {
        let memory = DB::open(TreeConfig::memory()).unwrap();
        memory.create_tree("data").unwrap();
        let genesis = journal_anchor(0, 0x20);
        let first = journal_anchor(1, 0x21);
        let envelope = JournalEnvelope::new(genesis, first, b"command".to_vec()).unwrap();
        assert!(matches!(
            memory.atomic_with_journal_envelope(envelope.clone(), |batch| {
                batch.put("data", b"key", b"value");
            }),
            Err(Error::Atomic {
                kind: AtomicErrorKind::DefinitelyNotApplied,
                source,
            }) if matches!(source.as_ref(), Error::JournalStreamUnavailable { .. })
        ));
        assert!(memory
            .open_tree("data")
            .unwrap()
            .get(b"key")
            .unwrap()
            .is_none());

        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        let db = DB::open(cfg).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();
        db.initialize_journal_stream(genesis).unwrap();
        db.initialize_journal_stream(genesis).unwrap();
        assert!(matches!(
            db.initialize_journal_stream(journal_anchor(0, 0x22)),
            Err(Error::JournalAnchorMismatch { .. })
        ));

        let wrong_previous = journal_anchor(0, 0x23);
        let stale = JournalEnvelope::new(wrong_previous, first, b"stale".to_vec()).unwrap();
        assert!(matches!(
            db.atomic_with_journal_envelope(stale, |batch| {
                batch.put("data", b"key", b"value");
            }),
            Err(Error::Atomic {
                kind: AtomicErrorKind::DefinitelyNotApplied,
                source,
            }) if matches!(source.as_ref(), Error::JournalAnchorMismatch { .. })
        ));
        assert!(data.get(b"key").unwrap().is_none());
        assert_eq!(db.journal_state().unwrap().tail(), genesis);

        assert!(db
            .atomic_with_journal_envelope(envelope, |batch| {
                batch.put("data", b"key", b"value");
            })
            .unwrap());
        assert_eq!(data.get(b"key").unwrap(), Some(b"value".to_vec()));
    }

    #[test]
    fn initialized_journal_rejects_ordinary_writes_before_mutation() {
        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        let db = DB::open(cfg).unwrap();
        let data = db.create_tree("data").unwrap();
        data.put(b"existing", b"old").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0x70);
        db.initialize_journal_stream(genesis).unwrap();

        assert!(matches!(
            data.put(b"existing", b"new"),
            Err(Error::JournalStreamUnavailable { .. })
        ));
        assert_eq!(data.get(b"existing").unwrap(), Some(b"old".to_vec()));
        assert!(data.get(b"new-key").unwrap().is_none());
        assert!(matches!(
            data.put_many_if_absent(&[(b"bulk".as_slice(), b"value".as_slice())]),
            Err(Error::JournalStreamUnavailable { .. })
        ));
        assert!(data.get(b"bulk").unwrap().is_none());
        assert!(matches!(
            db.create_tree("late"),
            Err(Error::JournalStreamUnavailable { .. })
        ));
        assert!(matches!(
            db.drop_tree("data"),
            Err(Error::JournalStreamUnavailable { .. })
        ));

        assert!(matches!(
            db.atomic(|batch| batch.put("data", b"new-key", b"new-value")),
            Err(Error::Atomic {
                kind: AtomicErrorKind::DefinitelyNotApplied,
                source,
            }) if matches!(source.as_ref(), Error::JournalStreamUnavailable { .. })
        ));
        assert!(data.get(b"new-key").unwrap().is_none());
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, genesis)
        );

        let first = journal_anchor(1, 0x71);
        let envelope = JournalEnvelope::new(genesis, first, b"attached".to_vec()).unwrap();
        assert!(db
            .atomic_with_journal_envelope(envelope, |batch| {
                batch.put("data", b"new-key", b"new-value");
            })
            .unwrap());
        assert_eq!(data.get(b"new-key").unwrap(), Some(b"new-value".to_vec()));
        assert_eq!(db.journal_state().unwrap().tail(), first);
    }

    #[test]
    fn interrupted_stream_initialization_stays_fenced_until_exact_retry() {
        for successful_anchor_writes in [0, 1] {
            let dir = tempdir().unwrap();
            let mut cfg = TreeConfig::new(dir.path());
            cfg.checkpoint.enabled = false;
            cfg.durability = super::super::config::Durability::Wal { sync: true };

            let db = DB::open(cfg.clone()).unwrap();
            let data = db.create_tree("data").unwrap();
            db.checkpoint().unwrap();
            let genesis = journal_anchor(0, 0x7a);
            let first = journal_anchor(1, 0x7b);

            crate::journal::writer::WalWriter::fail_anchor_generation_after_for_test(
                successful_anchor_writes,
            );
            assert!(db.initialize_journal_stream(genesis).is_err());

            assert!(matches!(
                data.put(b"tree-outside-chain", b"visible"),
                Err(Error::JournalStreamUnavailable { .. })
            ));
            assert!(data.get(b"tree-outside-chain").unwrap().is_none());
            assert!(matches!(
                db.atomic(|batch| batch.put("data", b"db-outside-chain", b"visible")),
                Err(Error::Atomic {
                    kind: AtomicErrorKind::DefinitelyNotApplied,
                    source,
                }) if matches!(source.as_ref(), Error::JournalStreamUnavailable { .. })
            ));
            assert!(data.get(b"db-outside-chain").unwrap().is_none());
            assert!(matches!(
                db.journal_state(),
                Err(Error::JournalStreamUnavailable { .. })
            ));
            assert!(matches!(
                db.journal_envelopes_after(genesis, 8, 4096),
                Err(Error::JournalStreamUnavailable { .. })
            ));

            let envelope = JournalEnvelope::new(genesis, first, b"attached".to_vec()).unwrap();
            assert!(matches!(
                db.atomic_with_journal_envelope(envelope.clone(), |batch| {
                    batch.put("data", b"attached", b"premature");
                }),
                Err(Error::Atomic {
                    kind: AtomicErrorKind::DefinitelyNotApplied,
                    source,
                }) if matches!(source.as_ref(), Error::JournalStreamUnavailable { .. })
            ));
            assert!(data.get(b"attached").unwrap().is_none());

            assert!(matches!(
                db.initialize_journal_stream(journal_anchor(0, 0x7c)),
                Err(Error::JournalAnchorMismatch { .. })
            ));

            // A checkpoint cannot clear an incomplete initialization fence.
            db.checkpoint().unwrap();
            assert!(matches!(
                data.put(b"after-checkpoint", b"blocked"),
                Err(Error::JournalStreamUnavailable { .. })
            ));

            // An exact retry repairs the missing generation and is the only
            // transition that makes the attached stream ready in-process.
            db.initialize_journal_stream(genesis).unwrap();
            assert_eq!(
                db.journal_state().unwrap(),
                JournalState::new(genesis, genesis)
            );
            assert!(db
                .atomic_with_journal_envelope(envelope, |batch| {
                    batch.put("data", b"attached", b"ready");
                })
                .unwrap());
            drop(data);
            drop(db);

            let reopened = DB::open(cfg).unwrap();
            assert_eq!(reopened.journal_state().unwrap().checkpoint(), genesis);
            let reopened_data = reopened.open_tree("data").unwrap();
            assert!(reopened_data.get(b"tree-outside-chain").unwrap().is_none());
            assert!(reopened_data.get(b"db-outside-chain").unwrap().is_none());
            assert_eq!(
                reopened_data.get(b"attached").unwrap(),
                Some(b"ready".to_vec())
            );
        }
    }

    #[test]
    fn interrupted_stream_reopen_uses_only_durable_anchor_slots() {
        for successful_anchor_writes in [0, 1] {
            let dir = tempdir().unwrap();
            let mut cfg = TreeConfig::new(dir.path());
            cfg.checkpoint.enabled = false;
            let db = DB::open(cfg.clone()).unwrap();
            db.create_tree("data").unwrap();
            db.checkpoint().unwrap();
            let genesis = journal_anchor(0, 0x7d);

            crate::journal::writer::WalWriter::fail_anchor_generation_after_for_test(
                successful_anchor_writes,
            );
            assert!(db.initialize_journal_stream(genesis).is_err());
            drop(db);

            let reopened = DB::open(cfg).unwrap();
            if successful_anchor_writes == 0 {
                assert!(matches!(
                    reopened.journal_state(),
                    Err(Error::JournalStreamUnavailable { .. })
                ));
                assert!(reopened
                    .atomic(|batch| batch.put("data", b"ordinary", b"allowed"))
                    .unwrap());
            } else {
                assert_eq!(
                    reopened.journal_state().unwrap(),
                    JournalState::new(genesis, genesis)
                );
                assert!(matches!(
                    reopened.atomic(|batch| batch.put("data", b"ordinary", b"blocked")),
                    Err(Error::Atomic {
                        kind: AtomicErrorKind::DefinitelyNotApplied,
                        source,
                    }) if matches!(source.as_ref(), Error::JournalStreamUnavailable { .. })
                ));
            }
        }
    }

    #[test]
    fn reopen_rejects_an_ordinary_record_after_an_initialized_anchor() {
        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        cfg.durability = super::super::config::Durability::Wal { sync: true };
        let wal_path = cfg.wal_path().unwrap();

        let db = DB::open(cfg.clone()).unwrap();
        db.create_tree("data").unwrap();
        db.checkpoint().unwrap();
        db.initialize_journal_stream(journal_anchor(0, 0x7e))
            .unwrap();
        drop(db);

        let mut writer = crate::journal::writer::WalWriter::open_existing(&wal_path).unwrap();
        writer
            .append(
                &crate::journal::wal_op::WalOp::Insert {
                    tree_id: FIRST_USER_TREE_ID,
                    key: b"outside-chain".to_vec(),
                    value: b"invalid".to_vec(),
                },
                100,
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);
        let before = std::fs::read(&wal_path).unwrap();

        for reopen_cfg in [cfg.clone(), cfg.read_only()] {
            assert!(matches!(
                DB::open(reopen_cfg),
                Err(Error::ReplaySanityFailed {
                    context: "initialized attached WAL contains an ordinary record",
                    ..
                })
            ));
            assert_eq!(std::fs::read(&wal_path).unwrap(), before);
        }
    }

    #[test]
    fn memory_db_uses_an_explicit_volatile_attached_stream() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0x80);
        let first = journal_anchor(1, 0x81);
        let second = journal_anchor(2, 0x82);
        let third = journal_anchor(3, 0x83);
        db.initialize_journal_stream(genesis).unwrap();
        assert!(db
            .atomic_with_journal_envelope(
                JournalEnvelope::new(genesis, first, b"volatile command".to_vec()).unwrap(),
                |batch| batch.put("data", b"key", b"value"),
            )
            .unwrap());
        assert_eq!(data.get(b"key").unwrap(), Some(b"value".to_vec()));
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, first)
        );
        assert_eq!(
            db.journal_envelopes_after(genesis, 8, 4096)
                .unwrap()
                .envelopes()
                .len(),
            1
        );
        for (previous, current, key) in [
            (first, second, b"second".as_slice()),
            (second, third, b"third".as_slice()),
        ] {
            assert!(db
                .atomic_with_journal_envelope(
                    JournalEnvelope::new(previous, current, key.to_vec()).unwrap(),
                    |batch| batch.put("data", key, b"value"),
                )
                .unwrap());
        }
        let first_page = db.journal_envelopes_after(genesis, 1, 4096).unwrap();
        assert_eq!(first_page.envelopes().len(), 1);
        assert_eq!(first_page.next(), first);
        assert!(first_page.has_more());
        let second_page = db.journal_envelopes_after(first, 8, 4096).unwrap();
        assert_eq!(second_page.envelopes().len(), 2);
        assert_eq!(second_page.next(), third);
        assert!(!second_page.has_more());
        assert!(matches!(
            db.journal_envelopes_after(journal_anchor(1, 0xff), 8, 4096),
            Err(Error::JournalAnchorMismatch {
                requested: 1,
                expected: 1,
            })
        ));
        assert!(matches!(
            data.put(b"ordinary", b"rejected"),
            Err(Error::JournalStreamUnavailable { .. })
        ));

        data.checkpoint().unwrap();
        assert_eq!(db.journal_state().unwrap(), JournalState::new(third, third));
        assert!(matches!(
            db.journal_envelopes_after(genesis, 8, 4096),
            Err(Error::JournalPositionExpired {
                requested: 0,
                checkpoint: 3,
            })
        ));
        assert_eq!(data.get(b"third").unwrap(), Some(b"value".to_vec()));
    }

    #[test]
    fn stream_initialization_fences_a_tree_write_waiting_outside_maintenance() {
        use crate::api::tree::{
            set_ordinary_write_gate_barrier_for_current_thread, OrdinaryWriteGateBarrier,
        };

        let db = DB::open(TreeConfig::memory()).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let barrier = Arc::new(OrdinaryWriteGateBarrier::new());
        let writer_barrier = Arc::clone(&barrier);
        let writer = thread::spawn(move || {
            set_ordinary_write_gate_barrier_for_current_thread(writer_barrier);
            data.put(b"racing", b"value")
        });

        barrier.entered.wait();
        let genesis = journal_anchor(0, 0x82);
        db.initialize_journal_stream(genesis).unwrap();
        barrier.release.wait();

        assert!(matches!(
            writer.join().unwrap(),
            Err(Error::JournalStreamUnavailable { .. })
        ));
        assert!(db
            .open_tree("data")
            .unwrap()
            .get(b"racing")
            .unwrap()
            .is_none());
        assert_eq!(
            db.journal_state().unwrap(),
            JournalState::new(genesis, genesis)
        );
    }

    #[test]
    fn concurrent_envelopes_with_same_previous_have_one_winner() {
        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        let db = DB::open(cfg).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0x90);
        db.initialize_journal_stream(genesis).unwrap();
        let start = Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for (byte, key) in [(0x91, b"one".as_slice()), (0x92, b"two".as_slice())] {
            let db = db.clone();
            let start = Arc::clone(&start);
            let key = key.to_vec();
            workers.push(thread::spawn(move || {
                let current = journal_anchor(1, byte);
                let envelope = JournalEnvelope::new(genesis, current, vec![byte; 8]).unwrap();
                start.wait();
                (
                    key.clone(),
                    current,
                    db.atomic_with_journal_envelope(envelope, |batch| {
                        batch.put("data", &key, b"value");
                    }),
                )
            }));
        }
        start.wait();

        let results = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            results
                .iter()
                .filter(|(_, _, result)| result.as_ref().is_ok_and(|value| *value))
                .count(),
            1
        );
        assert_eq!(
            results
                .iter()
                .filter(|(_, _, result)| matches!(
                    result,
                    Err(Error::Atomic {
                        kind: AtomicErrorKind::DefinitelyNotApplied,
                        source,
                    }) if matches!(source.as_ref(), Error::JournalAnchorMismatch { .. })
                ))
                .count(),
            1
        );
        let page = db.journal_envelopes_after(genesis, 8, 4096).unwrap();
        assert_eq!(page.envelopes().len(), 1);
        let winner = page.envelopes()[0].current();
        assert_eq!(db.journal_state().unwrap().tail(), winner);
        for (key, current, _) in results {
            assert_eq!(data.get(&key).unwrap().is_some(), current == winner);
        }
    }

    #[test]
    fn scan_flushes_an_async_attached_record_before_reading() {
        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        cfg.durability = crate::Durability::Wal { sync: false };
        let db = DB::open(cfg).unwrap();
        db.create_tree("data").unwrap();
        db.checkpoint().unwrap();

        let genesis = journal_anchor(0, 0xa0);
        let first = journal_anchor(1, 0xa1);
        db.initialize_journal_stream(genesis).unwrap();
        let envelope = JournalEnvelope::new(genesis, first, b"async".to_vec()).unwrap();
        assert!(db
            .atomic_with_journal_envelope(envelope.clone(), |batch| {
                batch.put("data", b"key", b"value");
            })
            .unwrap());

        assert_eq!(
            db.journal_envelopes_after(genesis, 8, 4096)
                .unwrap()
                .envelopes(),
            &[envelope]
        );
    }

    #[test]
    fn checkpoint_and_attached_writes_complete_without_lock_inversion() {
        let dir = tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.checkpoint.enabled = false;
        let db = DB::open(cfg).unwrap();
        db.create_tree("data").unwrap();
        db.checkpoint().unwrap();
        let genesis = journal_anchor(0, 0xb0);
        db.initialize_journal_stream(genesis).unwrap();

        let (done_tx, done_rx) = mpsc::channel();
        let writer_db = db.clone();
        let writer_tx = done_tx.clone();
        thread::spawn(move || {
            let mut previous = genesis;
            for sequence in 1..=12 {
                let current = journal_anchor(sequence, 0xb0 + sequence as u8);
                let envelope =
                    JournalEnvelope::new(previous, current, vec![sequence as u8]).unwrap();
                writer_db
                    .atomic_with_journal_envelope(envelope, |batch| {
                        batch.put("data", &sequence.to_le_bytes(), b"value");
                    })
                    .unwrap();
                previous = current;
                thread::yield_now();
            }
            writer_tx.send(()).unwrap();
        });
        let checkpoint_db = db.clone();
        thread::spawn(move || {
            for _ in 0..12 {
                checkpoint_db.checkpoint().unwrap();
                thread::yield_now();
            }
            done_tx.send(()).unwrap();
        });

        done_rx.recv_timeout(Duration::from_secs(20)).unwrap();
        done_rx.recv_timeout(Duration::from_secs(20)).unwrap();
        assert_eq!(db.journal_state().unwrap().tail().sequence(), 12);
    }

    #[test]
    fn volatile_named_tree_checkpoint_and_attached_writes_do_not_deadlock() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();
        let genesis = journal_anchor(0, 0xc0);
        db.initialize_journal_stream(genesis).unwrap();

        let writer_db = db.clone();
        let writer = thread::spawn(move || {
            let mut previous = genesis;
            for sequence in 1..=12 {
                let current = journal_anchor(sequence, 0xc0 + sequence as u8);
                writer_db
                    .atomic_with_journal_envelope(
                        JournalEnvelope::new(previous, current, vec![sequence as u8]).unwrap(),
                        |batch| batch.put("data", &sequence.to_le_bytes(), b"value"),
                    )
                    .unwrap();
                previous = current;
                thread::yield_now();
            }
        });
        let checkpointer = thread::spawn(move || {
            for _ in 0..12 {
                data.checkpoint().unwrap();
                thread::yield_now();
            }
        });

        writer.join().unwrap();
        checkpointer.join().unwrap();
        db.checkpoint().unwrap();
        assert_eq!(db.journal_state().unwrap().tail().sequence(), 12);
        assert_eq!(db.journal_state().unwrap().checkpoint().sequence(), 12);
    }

    #[test]
    fn oversized_attached_record_is_definitely_not_applied() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let data = db.create_tree("data").unwrap();
        db.checkpoint().unwrap();
        let genesis = journal_anchor(0, 0xc0);
        let first = journal_anchor(1, 0xc1);
        db.initialize_journal_stream(genesis).unwrap();
        let payload = vec![0xc1; super::super::journal::MAX_JOURNAL_RECORD_BYTES];
        let envelope = JournalEnvelope::new(genesis, first, payload).unwrap();

        assert!(matches!(
            db.atomic_with_journal_envelope(envelope, |batch| {
                batch.put("data", b"key", b"value");
            }),
            Err(Error::Atomic {
                kind: AtomicErrorKind::DefinitelyNotApplied,
                source,
            }) if matches!(source.as_ref(), Error::WalRecordTooLarge { .. })
        ));
        assert!(data.get(b"key").unwrap().is_none());
        assert_eq!(db.journal_state().unwrap().tail(), genesis);
    }

    #[test]
    fn create_tree_durably_publishes_root_before_live_catalog() {
        let store = Arc::new(FailIoStore::new());
        let mut cfg = TreeConfig::memory();
        cfg.checkpoint.enabled = false;
        cfg.memory_flush_on_write = true;
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let db = DB::open_with_blob_store(cfg.clone(), store_dyn).unwrap();

        store.fail_next_flush.store(true, AtomicOrdering::Release);
        assert!(db.create_tree("empty").is_err());
        assert!(db.catalog_lookup_live(b"empty").unwrap().is_none());
        assert!(matches!(
            db.open_tree("empty"),
            Err(Error::TreeNotFound { .. })
        ));
        assert!(store
            .inner
            .has_blob(root_guid_for_tree_id(FIRST_USER_TREE_ID))
            .unwrap());
        drop(db);

        // Reopen while the failed attempt's deterministic root is orphaned.
        // The catalog stays healthy and retry flushes that root before Live.
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let reopened = DB::open_with_blob_store(cfg.clone(), store_dyn).unwrap();
        assert!(matches!(
            reopened.open_tree("empty"),
            Err(Error::TreeNotFound { .. })
        ));
        drop(reopened.create_tree("empty").unwrap());
        drop(reopened);

        // An empty tree has no user WAL record; its root+catalog ordering
        // alone must be sufficient for a clean subsequent reopen.
        let store_dyn: Arc<dyn BlobStore> = store;
        let final_db = DB::open_with_blob_store(cfg, store_dyn).unwrap();
        let empty = final_db.open_tree("empty").unwrap();
        assert!(empty.get(b"missing").unwrap().is_none());
    }

    #[test]
    fn atomic_catalog_error_is_not_applied_and_can_be_retried() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let present = db.create_tree("present").unwrap();

        let error = db
            .atomic(|batch| {
                batch.put("present", b"key", b"present-value");
                batch.put("missing", b"key", b"missing-value");
            })
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Atomic {
                kind: AtomicErrorKind::DefinitelyNotApplied,
                source,
            } if matches!(source.as_ref(), Error::TreeNotFound { .. })
        ));
        assert!(present.get(b"key").unwrap().is_none());

        let missing = db.create_tree("missing").unwrap();
        assert!(db
            .atomic(|batch| {
                batch.put("present", b"key", b"present-value");
                batch.put("missing", b"key", b"missing-value");
            })
            .unwrap());
        assert_eq!(
            present.get(b"key").unwrap().as_deref(),
            Some(&b"present-value"[..])
        );
        assert_eq!(
            missing.get(b"key").unwrap().as_deref(),
            Some(&b"missing-value"[..])
        );
    }

    #[test]
    fn atomic_preflight_error_is_not_applied_and_can_be_retried() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let tree = db.create_tree("objects").unwrap();

        let error = db
            .atomic(|batch| {
                batch.put("objects", b"side", b"value");
                batch.rename("objects", b"source", b"destination", false);
            })
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Atomic {
                kind: AtomicErrorKind::DefinitelyNotApplied,
                source,
            } if matches!(source.as_ref(), Error::NotFound)
        ));
        assert!(tree.get(b"side").unwrap().is_none());
        assert!(tree.get(b"destination").unwrap().is_none());

        tree.put(b"source", b"payload").unwrap();
        assert!(db
            .atomic(|batch| {
                batch.put("objects", b"side", b"value");
                batch.rename("objects", b"source", b"destination", false);
            })
            .unwrap());
        assert_eq!(tree.get(b"side").unwrap().as_deref(), Some(&b"value"[..]));
        assert_eq!(
            tree.get(b"destination").unwrap().as_deref(),
            Some(&b"payload"[..])
        );
        assert!(tree.get(b"source").unwrap().is_none());
    }

    #[test]
    fn atomic_delta_flush_read_error_is_not_applied_and_can_be_retried() {
        let store = Arc::new(FailIoStore::new());
        let mut cfg = TreeConfig::memory();
        cfg.buffer_pool_size = 32;
        cfg.checkpoint.enabled = false;
        cfg.memory_flush_on_write = false;
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let db = DB::open_with_blob_store(cfg, store_dyn).unwrap();
        let tree = db.create_tree("objects").unwrap();
        let old = vec![0x3A; 1024];
        let keys = (0..1200u32)
            .map(|i| format!("deferred/{i:08}").into_bytes())
            .collect::<Vec<_>>();
        for key in &keys {
            tree.put(key, &old).unwrap();
        }
        db.checkpoint().unwrap();

        let tree_id = db
            .catalog_lookup_live(b"objects")
            .unwrap()
            .expect("created tree id");
        let root_guid = root_guid_for_tree_id(tree_id);
        let child_guids = crate::engine::collect_blob_topology_silent(&db.store, root_guid)
            .unwrap()
            .into_iter()
            .filter_map(|entry| (entry.depth > 0).then_some(entry.guid))
            .collect::<Vec<_>>();
        assert!(!child_guids.is_empty(), "test requires a spilled tree");
        // DB grouping pins the already-resident root. Keeping only its
        // children cold makes the injected read exclusive to delta flush.
        for &guid in &child_guids {
            assert!(
                db.store.evict_from_cache_for_test(guid),
                "test child remained pinned"
            );
        }

        store.arm_read_failures(child_guids.iter().copied());
        let target = keys
            .iter()
            .find_map(|key| match tree.get(key) {
                Err(Error::BlobStoreIo(_)) => Some(key.clone()),
                Ok(Some(_)) => None,
                other => panic!("unexpected target probe result: {other:?}"),
            })
            .expect("no key traversed a cold child");
        let failed_guid = store
            .take_last_failed_read()
            .expect("failed probe did not record its child");
        store.clear_read_failures();
        assert!(child_guids.contains(&failed_guid));
        let seq = db.next_seq.fetch_add(1, Ordering::Relaxed);
        db.store
            .stage_write_delta_put(tree_id, root_guid, &target, b"deferred-value", seq, false);

        store.arm_read_failures([failed_guid]);
        let error = db
            .atomic(|batch| batch.put("objects", b"atomic-current", b"current-value"))
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Atomic {
                kind: AtomicErrorKind::DefinitelyNotApplied,
                source,
            } if matches!(source.as_ref(), Error::BlobStoreIo(_))
        ));
        assert_eq!(store.take_last_failed_read(), Some(failed_guid));
        store.clear_read_failures();
        assert!(tree.get(b"atomic-current").unwrap().is_none());
        assert!(db.store.lookup_write_delta(tree_id, &target).is_some());

        assert!(db
            .atomic(|batch| batch.put("objects", b"atomic-current", b"current-value"))
            .unwrap());
        assert!(db.store.lookup_write_delta(tree_id, &target).is_none());
        assert_eq!(
            tree.get(&target).unwrap().as_deref(),
            Some(&b"deferred-value"[..])
        );
        assert_eq!(
            tree.get(b"atomic-current").unwrap().as_deref(),
            Some(&b"current-value"[..])
        );
    }

    #[test]
    fn atomic_post_apply_failure_has_unknown_outcome() {
        let store = Arc::new(FailIoStore::new());
        let mut cfg = TreeConfig::memory();
        cfg.checkpoint.enabled = false;
        cfg.memory_flush_on_write = true;
        let store_dyn: Arc<dyn BlobStore> = store.clone();
        let db = DB::open_with_blob_store(cfg, store_dyn).unwrap();
        let tree = db.create_tree("objects").unwrap();

        store.fail_next_flush.store(true, AtomicOrdering::Release);
        let error = db
            .atomic(|batch| batch.put("objects", b"key", b"value"))
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Atomic {
                kind: AtomicErrorKind::OutcomeUnknown,
                source,
            } if matches!(source.as_ref(), Error::BlobStoreIo(_))
        ));
        assert_eq!(tree.get(b"key").unwrap().as_deref(), Some(&b"value"[..]));
    }

    #[test]
    fn db_reopens_max_minus_one_as_exhausted_and_rejects_max() {
        let dir = tempdir().unwrap();
        let cfg = || {
            let mut cfg = TreeConfig::new(dir.path());
            cfg.checkpoint.enabled = false;
            cfg.buffer_pool_size = 16;
            cfg
        };

        let tree_id;
        {
            let db = DB::open(cfg()).unwrap();
            let tree = db.create_tree("objects").unwrap();
            tree.put(b"key", b"before").unwrap();
            tree_id = db.catalog_lookup_live(b"objects").unwrap().unwrap();
            db.store.set_current_epoch(u64::MAX - 2);

            let snapshot = tree.snapshot(b"").unwrap();
            assert_eq!(snapshot.epoch(), u64::MAX - 2);
            assert_eq!(db.store.current_epoch(), u64::MAX - 1);
            drop(snapshot);
            db.checkpoint().unwrap();
        }

        {
            let db = DB::open(cfg()).unwrap();
            assert_eq!(db.store.current_epoch(), u64::MAX - 1);
            let tree = db.open_tree("objects").unwrap();
            let error = tree.snapshot(b"").unwrap_err();
            assert!(matches!(error, Error::SnapshotEpochExhausted));
            assert_eq!(tree.get(b"key").unwrap().as_deref(), Some(&b"before"[..]));
            tree.put(b"key", b"after").unwrap();

            let root_guid = root_guid_for_tree_id(tree_id);
            let root = db.store.pin(root_guid).unwrap();
            {
                let mut frame = root.write();
                crate::layout::set_frame_epoch_high_water(frame.as_mut_slice(), u64::MAX);
            }
            db.store
                .mark_dirty_cached(root_guid, crate::store::STRUCTURAL_SEQ, root.as_ref());
            drop(root);
            db.checkpoint().unwrap();
        }

        let error = DB::open(cfg()).unwrap_err();
        assert!(matches!(
            error,
            Error::NodeCorrupt {
                context: "snapshot epoch exhausted",
                ..
            }
        ));
    }

    #[test]
    fn open_tree_cannot_resurrect_runtime_after_concurrent_drop() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let existing = db.create_tree("objects").unwrap();
        let tree_id = db.catalog_lookup_live(b"objects").unwrap().unwrap();
        drop(existing);
        db.trees.lock().unwrap().remove(&tree_id);

        let barrier = Arc::new(OpenTreeCatalogBarrier::new());
        let opener_db = db.clone();
        let opener_barrier = Arc::clone(&barrier);
        let opener = thread::spawn(move || {
            set_open_tree_catalog_barrier_for_current_thread(opener_barrier);
            opener_db.open_tree("objects")
        });
        barrier.entered.wait();

        let drop_db = db.clone();
        let (drop_done_tx, drop_done_rx) = mpsc::sync_channel(1);
        let dropper = thread::spawn(move || {
            let result = drop_db.drop_tree("objects");
            drop_done_tx.send(()).unwrap();
            result
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while !db.maintenance_gate.writer_pending_for_test() {
            assert!(
                Instant::now() < deadline,
                "drop_tree never queued behind open_tree's shared fence",
            );
            thread::yield_now();
        }
        assert!(
            drop_done_rx
                .recv_timeout(Duration::from_millis(30))
                .is_err(),
            "drop_tree bypassed the in-flight open_tree construction",
        );

        barrier.release.wait();
        let opened = opener.join().unwrap().unwrap();
        dropper.join().unwrap().unwrap();
        assert!(matches!(
            opened.put(b"hidden", b"write"),
            Err(Error::TreeDropped)
        ));
        assert!(matches!(
            db.open_tree("objects"),
            Err(Error::TreeNotFound { .. })
        ));
        assert!(db
            .trees
            .lock()
            .unwrap()
            .get(&tree_id)
            .unwrap()
            .runtime
            .is_dropped());
    }

    #[test]
    fn export_checkpoint_restarts_after_concurrent_compaction() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let tree = db.create_tree("objects").unwrap();
        let mut expected = BTreeMap::new();
        for index in 0..600u32 {
            let key = format!("key/{index:06}").into_bytes();
            let value = vec![(index % 251) as u8; 512];
            tree.put(&key, &value).unwrap();
            expected.insert(key, value);
        }
        for index in (0..600u32).step_by(4) {
            let key = format!("key/{index:06}").into_bytes();
            assert!(tree.delete(&key).unwrap());
            expected.remove(&key);
        }

        let barrier = Arc::new(ExportFirstEntryBarrier::new());
        let export_db = db.clone();
        let export_barrier = Arc::clone(&barrier);
        let exporter = thread::spawn(move || {
            set_export_first_entry_barrier_for_current_thread(export_barrier);
            export_db.export_checkpoint()
        });
        barrier.entered.wait();

        let compactions_before = tree.stats().unwrap().total_compactions;
        for _ in 0..4 {
            tree.compact().unwrap();
        }
        let compactions_after = tree.stats().unwrap().total_compactions;
        assert!(
            compactions_after > compactions_before,
            "test must rewrite at least one shared snapshot frame",
        );
        barrier.release.wait();

        let image = exporter.join().unwrap().unwrap();
        let decoded = checkpoint::decode(image.as_bytes()).unwrap();
        let (_, records) = decoded
            .families
            .iter()
            .find(|(name, _)| *name == b"objects")
            .unwrap();
        let exported = records
            .iter()
            .map(|(key, value)| (key.to_vec(), value.to_vec()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(exported, expected);
    }

    #[test]
    fn export_checkpoint_captures_family_created_before_its_fence() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let blocker = db.maintenance_gate.enter_shared();

        let create_db = db.clone();
        let (create_done_tx, create_done_rx) = mpsc::channel();
        let creator = thread::spawn(move || {
            let result = create_db.create_tree("created-before-export");
            create_done_tx.send(()).unwrap();
            result
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        while !db.maintenance_gate.writer_pending_for_test() {
            assert!(
                Instant::now() < deadline,
                "create_tree never queued on the maintenance fence"
            );
            thread::yield_now();
        }

        let export_db = db.clone();
        let (export_started_tx, export_started_rx) = mpsc::channel();
        let exporter = thread::spawn(move || {
            export_started_tx.send(()).unwrap();
            export_db.export_checkpoint()
        });
        export_started_rx.recv().unwrap();
        assert!(
            create_done_rx
                .recv_timeout(Duration::from_millis(30))
                .is_err(),
            "create_tree bypassed an existing maintenance reader"
        );

        // The queued creator owns WRITE_BIT first. Export cannot enumerate
        // until that catalog mutation is complete, so the family must be in
        // the exported generation (its later user writes may validly fall on
        // either side of the boundary).
        drop(blocker);
        drop(creator.join().unwrap().unwrap());
        let image = exporter.join().unwrap().unwrap();
        let decoded = checkpoint::decode(image.as_bytes()).unwrap();
        assert!(
            decoded
                .families
                .iter()
                .any(|(name, _)| *name == b"created-before-export"),
            "export omitted a family whose catalog commit preceded its fence"
        );
    }

    #[test]
    fn db_view_waits_for_checkpoint_delta_flush_before_registering_barrier() {
        let db = DB::open(TreeConfig::memory()).unwrap();
        let tree = db.create_tree("objects").unwrap();
        tree.put(b"key", b"old").unwrap();
        let tree_id = db
            .catalog_lookup_live(b"objects")
            .unwrap()
            .expect("created tree id");
        let root_guid = root_guid_for_tree_id(tree_id);
        let seq = db.next_seq.fetch_add(1, Ordering::Relaxed);
        db.store
            .stage_write_delta_put(tree_id, root_guid, b"key", b"checkpoint-value", seq, false);

        let root_pin = db.store.pin(root_guid).unwrap();
        let root_guard = root_pin.write();
        let checkpoint_db = db.clone();
        let (checkpoint_tx, checkpoint_rx) = mpsc::channel();
        let checkpoint = thread::spawn(move || {
            checkpoint_tx.send(checkpoint_db.checkpoint()).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while !db.commit_gate.checkpoint_pending_for_test() {
            assert!(
                Instant::now() < deadline,
                "checkpoint never acquired commit-exclusive"
            );
            thread::yield_now();
        }

        let view_db = db.clone();
        let (view_tx, view_rx) = mpsc::channel();
        let view_worker = thread::spawn(move || {
            let result = view_db.view(&[("objects", b"".as_slice())], |view| {
                view.tree("objects")
                    .expect("captured objects tree")
                    .get(b"key")
            });
            view_tx.send(result).unwrap();
        });
        assert!(
            view_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "DB view must not register between delta shared-check and write",
        );

        drop(root_guard);
        drop(root_pin);
        checkpoint_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        checkpoint.join().unwrap();
        let value = view_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        view_worker.join().unwrap();
        assert_eq!(value.as_deref(), Some(&b"checkpoint-value"[..]));
    }

    #[test]
    fn reopen_skips_partially_reclaimed_dropping_closure() {
        let dir = tempdir().unwrap();
        let cfg = || {
            let mut cfg = TreeConfig::new(dir.path());
            cfg.checkpoint.enabled = false;
            cfg.buffer_pool_size = 32;
            cfg
        };

        let tree_id;
        {
            let db = DB::open(cfg()).unwrap();
            let doomed = db.create_tree("doomed").unwrap();
            tree_id = db
                .catalog_lookup_live(b"doomed")
                .unwrap()
                .expect("created doomed tree id");
            let value = vec![0x5A; 1024];
            for i in 0..1200u32 {
                doomed
                    .put(format!("drop/{i:08}").as_bytes(), &value)
                    .unwrap();
            }
            db.checkpoint().unwrap();

            let root = root_guid_for_tree_id(tree_id);
            let closure = crate::engine::collect_blob_guids(&db.store, root).unwrap();
            assert!(
                closure.len() > 1,
                "partial-drop recovery test requires a non-root child",
            );
            let child = closure[1];

            db.drop_tree("doomed").unwrap();
            drop(doomed);
            // Persist the Dropping catalog state without invoking DB's drop
            // sweep, then deterministically reclaim exactly one child while
            // retaining the still-referencing root. This is the durable
            // intermediate state a bounded sweep can leave at crash time.
            Tree::checkpoint_shared_store(
                &db.store,
                db.journal.as_ref(),
                &db.maintenance_gate,
                &db.commit_gate,
            )
            .unwrap();
            let mut retain: HashSet<_> = db.store.list_blobs().unwrap().into_iter().collect();
            assert!(retain.remove(&child));
            let outcome = db.store.gc_sweep_unreachable_bounded(&retain, 1).unwrap();
            assert_eq!(outcome.freed, 1);
            assert!(db.store.store_has_blob(root).unwrap());
            assert!(!db.store.store_has_blob(child).unwrap());
            assert!(
                crate::engine::collect_blob_guids(&db.store, root).is_err(),
                "test did not create a partially reclaimed Dropping closure",
            );
        }

        // Open must ignore the intentionally incomplete Dropping family's
        // epoch metadata so the normal recovery sweep can finish it.
        {
            let db = DB::open(cfg()).unwrap();
            assert!(db.list_trees().unwrap().is_empty());

            // An unrelated live snapshot must not make the incomplete
            // Dropping closure canonical again. Snapshot roots are protected
            // by their own pinned closure.
            let unrelated = db.create_tree("unrelated").unwrap();
            unrelated.put(b"live", b"value").unwrap();
            let unrelated_snapshot = unrelated.snapshot(b"").unwrap();
            db.gc().unwrap();
            assert!(
                !db.store
                    .store_has_blob(root_guid_for_tree_id(tree_id))
                    .unwrap(),
                "an unrelated family snapshot must not retain the Dropping root",
            );
            assert!(
                db.catalog_entry(b"doomed").unwrap().is_none(),
                "the completed Dropping family must leave the catalog while the unrelated snapshot is live",
            );
            assert!(matches!(
                db.open_tree("doomed"),
                Err(Error::TreeNotFound { .. })
            ));
            assert_eq!(
                unrelated_snapshot.get(b"live").unwrap().as_deref(),
                Some(&b"value"[..]),
            );
            drop(unrelated_snapshot);
            db.drop_tree("unrelated").unwrap();
            drop(unrelated);
            db.gc().unwrap();
        }

        let db = DB::open(cfg()).unwrap();
        assert!(db.list_trees().unwrap().is_empty());
        assert!(!db
            .store
            .store_has_blob(root_guid_for_tree_id(tree_id))
            .unwrap());
    }
}
