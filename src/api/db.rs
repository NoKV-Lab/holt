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
use super::errors::{Error, Result};
use super::snapshot::Snapshot;
use super::stats::{CheckpointerStats, DBStats, JournalStats, OpenStats};
use super::tree::{ensure_root_blob, replay_wal, Tree, TreeRuntime};
use super::view::View;
use crate::concurrency::{CommitGate, Gate};
use crate::engine::RangeEntry;
use crate::journal::codec::BatchEncoder;
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
        let mut open_stats = OpenStats::default();

        let (journal, next_seq) = match cfg.wal_path() {
            Some(path) => {
                let next_seq = if path.exists() {
                    let start = std::time::Instant::now();
                    let (next_seq, replay_stats) =
                        replay_wal(&path, &bm, |tree_id| Ok(root_guid_for_tree_id(tree_id)))?;
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
                let journal = Journal::open_or_create(&path, 0)?;
                (Some(Arc::new(journal)), next_seq)
            }
            _ => (None, 1),
        };

        let maintenance_gate = Arc::new(Gate::new());
        let commit_gate = Arc::new(CommitGate::new());
        let checkpointer = crate::checkpoint::Checkpointer::spawn(
            Arc::clone(&bm),
            journal.clone(),
            Arc::clone(&maintenance_gate),
            Arc::clone(&commit_gate),
            cfg.checkpoint.clone(),
        )
        .map(Arc::new);

        let db = Self {
            cfg,
            store: bm,
            maintenance_gate,
            next_seq: Arc::new(AtomicU64::new(next_seq)),
            commit_gate,
            journal,
            checkpointer,
            open_stats,
            trees: Arc::new(Mutex::new(HashMap::new())),
            catalog_cache: Arc::new(Mutex::new(HashMap::new())),
        };
        db.stage_dropped_trees()?;
        Ok(db)
    }

    /// Create a named tree inside this DB.
    ///
    /// Creation is recorded in the internal catalog before the
    /// handle is returned. Re-creating an existing name returns
    /// [`Error::TreeExists`].
    pub fn create_tree(&self, name: &str) -> Result<Tree> {
        let name_bytes = validate_tree_name(name)?;
        let _maintenance = self.maintenance_gate.enter_exclusive();
        if self.catalog_entry(name_bytes)?.is_some() {
            return Err(Error::TreeExists {
                name: name.to_owned(),
            });
        }
        let tree_id = self.allocate_tree_id()?;

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
        let tree_id = self
            .catalog_lookup_live(name_bytes)?
            .ok_or_else(|| Error::TreeNotFound {
                name: name.to_owned(),
            })?;
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

    /// Drop a named tree from the catalog and stage its blobs for
    /// checkpoint-time deletion.
    ///
    /// The catalog tombstone is hidden from [`Self::list_trees`] and
    /// from [`Self::open_tree`]. Existing handles are fenced before
    /// this call returns. Physical cleanup completes in a later
    /// [`Self::checkpoint`] after old handles/iterators have dropped
    /// their cached root pins.
    pub fn drop_tree(&self, name: &str) -> Result<()> {
        let name_bytes = validate_tree_name(name)?;
        let _maintenance = self.maintenance_gate.enter_exclusive();
        let entry = match self.catalog_entry(name_bytes)? {
            Some(entry) if entry.state == CatalogState::Live => entry,
            Some(_) | None => {
                return Err(Error::TreeNotFound {
                    name: name.to_owned(),
                });
            }
        };
        let guids = self.collect_tree_guids(entry.tree_id)?;
        let seq = self.apply_system_batch_unlocked(
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
        self.stage_tree_delete_guids(&guids, seq);
        Ok(())
    }

    /// Apply mutations across named trees under one WAL record.
    ///
    /// The closure buffers operations in a [`DBAtomicBatch`]. Holt
    /// validates all guards for every touched tree before applying
    /// any mutation; if a guard fails, the method returns `Ok(false)`
    /// and emits no WAL record.
    pub fn atomic<F>(&self, build: F) -> Result<bool>
    where
        F: FnOnce(&mut DBAtomicBatch),
    {
        let mut batch = DBAtomicBatch::default();
        build(&mut batch);
        if batch.pending.is_empty() {
            return Ok(true);
        }
        self.apply_atomic(batch.pending)
    }

    /// Run a read-only transaction over explicit tree/prefix scopes.
    ///
    /// Holt captures every listed scope while holding each touched
    /// tree's exclusive mutation gate, releases the live DB, then
    /// invokes `read` with an immutable [`DBView`]. Writes committed
    /// after the capture are invisible to every captured tree view.
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
            let mut trees = HashMap::with_capacity(scoped.len());
            for (_, name, prefix, tree) in scoped {
                trees.insert(name, tree.snapshot_unlocked(prefix)?);
            }
            DBView { trees }
        };
        read(&view)
    }

    /// Reclaim copy-on-write frames left unreachable by a crash that
    /// happened while a snapshot was live — the DB-wide analog of
    /// [`crate::Tree::gc`]. Marks every frame reachable from the catalog,
    /// each live tree's root, and each live snapshot root, then frees the
    /// rest from the shared buffer manager. Returns the count reclaimed.
    /// Idempotent. Callers must not create or drop trees concurrently.
    pub fn gc(&self) -> Result<usize> {
        // Read the live tree set gate-free (`catalog_entries` runs a range
        // scan that manages its own shared maintenance gate — holding the
        // gate here would deadlock against it). Then freeze every tree's
        // writers via their mutation gates, taken in tree-id order to
        // match `DB::view`/`apply_atomic` and avoid deadlock. Callers must
        // not create or drop trees concurrently with gc.
        let mut scoped: Vec<(u64, Tree)> = Vec::new();
        for (_, entry) in self.catalog_entries()? {
            if entry.state == CatalogState::Live {
                let open = self.open_tree_state(entry.tree_id)?;
                scoped.push((entry.tree_id, self.tree_from_state(entry.tree_id, open)?));
            }
        }
        let mut gates: Vec<(u64, Arc<Gate>)> = scoped
            .iter()
            .map(|(id, t)| (*id, t.mutation_gate()))
            .collect();
        gates.sort_by_key(|(id, _)| *id);
        gates.dedup_by_key(|(id, _)| *id);
        let _guards: Vec<_> = gates
            .iter()
            .map(|(_, gate)| gate.enter_exclusive())
            .collect();

        let mut reachable: HashSet<BlobGuid> = HashSet::new();
        reachable.insert(root_guid_for_tree_id(DB_CATALOG_TREE_ID));
        reachable.extend(self.collect_tree_guids(DB_CATALOG_TREE_ID)?);
        for (tree_id, _) in &scoped {
            reachable.insert(root_guid_for_tree_id(*tree_id));
            reachable.extend(self.collect_tree_guids(*tree_id)?);
        }
        for snap_root in self.store.snapshot_roots() {
            reachable.insert(snap_root);
            reachable.extend(crate::engine::collect_blob_guids(&self.store, snap_root)?);
        }
        self.store.gc_sweep_unreachable(&reachable)
    }

    /// Export a consistent point-in-time image of every live family.
    ///
    /// Each family is captured with a copy-on-write snapshot taken under a
    /// brief all-families freeze, so the image is a single consistent
    /// instant; serialization then runs *outside* the freeze while live
    /// applies continue (forking the frames the snapshots reference).
    pub fn export_checkpoint(&self) -> Result<CheckpointImage> {
        // Enumerate live families gate-free (the catalog range scan manages
        // its own maintenance gate; holding one here would deadlock it).
        let mut families: Vec<(Vec<u8>, u64, Tree)> = Vec::new();
        for (name, entry) in self.catalog_entries()? {
            if entry.state == CatalogState::Live {
                let open = self.open_tree_state(entry.tree_id)?;
                families.push((
                    name,
                    entry.tree_id,
                    self.tree_from_state(entry.tree_id, open)?,
                ));
            }
        }

        // Freeze every family's writers (tree-id order, matching
        // DB::view/apply_atomic) and snapshot each — O(1) per snapshot.
        let snaps: Vec<(Vec<u8>, Snapshot)> = {
            let mut gates: Vec<(u64, Arc<Gate>)> = families
                .iter()
                .map(|(_, id, t)| (*id, t.mutation_gate()))
                .collect();
            gates.sort_by_key(|(id, _)| *id);
            gates.dedup_by_key(|(id, _)| *id);
            let _guards: Vec<_> = gates
                .iter()
                .map(|(_, gate)| gate.enter_exclusive())
                .collect();

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
    /// This flushes the shared BufferManager, applies pending
    /// deletes, and truncates the shared WAL when it is safe. It is
    /// not tied to any one named tree.
    pub fn checkpoint(&self) -> Result<()> {
        self.stage_dropped_trees()?;
        Tree::checkpoint_shared_parts(
            &self.store,
            self.journal.as_ref(),
            &self.maintenance_gate,
            &self.commit_gate,
        )?;
        if self.store.pending_delete_count() == 0 && self.finalize_dropped_trees()? {
            Tree::checkpoint_shared_parts(
                &self.store,
                self.journal.as_ref(),
                &self.maintenance_gate,
                &self.commit_gate,
            )?;
        }
        Ok(())
    }

    /// Run one online maintenance pass for the catalog and every
    /// named tree.
    pub fn compact(&self) -> Result<()> {
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
            bm_dirty_count: self.store.dirty_count(),
            bm_pending_delete_count: self.store.pending_delete_count(),
            bm_cache_hits: self.store.cache_hits(),
            bm_cache_misses: self.store.cache_misses(),
            bm_full_blob_reads: self.store.full_blob_reads(),
            bm_full_blob_read_bytes: self.store.full_blob_read_bytes(),
            bm_point_full_blob_reads: self.store.point_full_blob_reads(),
            bm_scan_full_blob_reads: self.store.scan_full_blob_reads(),
            bm_silent_full_blob_reads: self.store.silent_full_blob_reads(),
            bm_optimistic_restarts: self.store.optimistic_restarts(),
            bm_range_restarts: self.store.range_restarts(),
            bm_walker_ops: self.store.walker_ops(),
            bm_walker_blob_hops: self.store.walker_blob_hops(),
            bm_max_blob_hops: self.store.max_blob_hops(),
            bm_max_cross_blob_depth: self.store.max_cross_blob_depth(),
            bm_spillovers: self.store.spillover_count(),
            bm_merges: self.store.merge_count(),
            bm_route_resident_count: self.store.route_resident_count(),
            bm_route_resident_demotions: self.store.route_resident_demotions(),
            bm_cache_evictions: self.store.cache_evictions(),
            bm_eviction_skips_protected: self.store.eviction_skips_protected(),
            bm_eviction_skips_route_resident: self.store.eviction_skips_route_resident(),
            bm_admission_protects: self.store.admission_protects(),
            open: self.open_stats,
            journal,
            checkpointer,
        }
    }

    fn catalog_tree(&self) -> Result<Tree> {
        let open = self.open_tree_state(DB_CATALOG_TREE_ID)?;
        self.tree_from_state(DB_CATALOG_TREE_ID, open)
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

    fn stage_dropped_trees(&self) -> Result<()> {
        for (_, entry) in self.catalog_entries()? {
            if entry.state == CatalogState::Dropping {
                let _maintenance = self.maintenance_gate.enter_exclusive();
                self.mark_runtime_dropped(entry.tree_id);
                let guids = self.collect_tree_guids(entry.tree_id)?;
                self.stage_tree_delete_guids(&guids, self.next_seq.load(Ordering::Acquire));
            }
        }
        Ok(())
    }

    fn finalize_dropped_trees(&self) -> Result<bool> {
        let mut ops = Vec::new();
        let mut finalized_tree_ids = Vec::new();
        let mut finalized_names = Vec::new();
        for (name, entry) in self.catalog_entries()? {
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
        let _maintenance = self.maintenance_gate.enter_exclusive();
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

    fn collect_tree_guids(&self, tree_id: u64) -> Result<Vec<BlobGuid>> {
        let root_guid = root_guid_for_tree_id(tree_id);
        if !self.store.has_blob(root_guid)? {
            return Ok(Vec::new());
        }
        crate::engine::collect_blob_guids(&self.store, root_guid)
    }

    fn stage_tree_delete_guids(&self, guids: &[BlobGuid], seq: u64) {
        for guid in guids {
            self.store.mark_for_delete(*guid, seq);
        }
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
        ensure_root_blob(&self.store, root_guid)?;
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
            self.checkpointer.clone(),
            self.open_stats,
        )
    }

    fn apply_atomic(&self, pending: Vec<DBBatchOp>) -> Result<bool> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let groups = self.group_batch_ops(pending)?;
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
        let count = count_wal_ops(&groups);
        let base_seq = self.next_seq.fetch_add(count, Ordering::Relaxed);
        if !Self::preflight_batch_groups(&groups, base_seq)? {
            return Ok(false);
        }
        if count == 0 {
            return Ok(true);
        }

        if let Some(journal) = &self.journal {
            self.apply_batch_groups_with_journal(&groups, base_seq, journal)?;
        } else {
            self.apply_batch_groups_in_memory(&groups, base_seq)?;
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

    fn apply_batch_groups_in_memory(&self, groups: &[DBBatchGroup], base_seq: u64) -> Result<()> {
        let mut group_base = base_seq;
        for group in groups {
            group
                .tree
                .apply_batch_walker_inline(&group.ops, group_base, None)?;
            group_base += count_group_wal_ops(group);
        }
        if self.cfg.memory_flush_on_write {
            if let Some(group) = groups.first() {
                group.tree.flush_dirty_inline()?;
                group.tree.flush_pending_deletes_inline()?;
            }
        }
        Ok(())
    }

    fn apply_system_batch_unlocked(&self, tree_id: u64, ops: Vec<BatchOp>) -> Result<u64> {
        let open = {
            let mut trees = self.trees.lock().unwrap();
            if let Some(open) = trees.get(&tree_id) {
                open.clone()
            } else {
                let root_guid = root_guid_for_tree_id(tree_id);
                ensure_root_blob(&self.store, root_guid)?;
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
/// same as single-tree snapshots.
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
                BatchOp::AssertVersion { .. } | BatchOp::AssertPrefixEmpty { .. } => 0,
            };
        }
    }
    len + crate::journal::codec::RECORD_FOOTER_SIZE
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
