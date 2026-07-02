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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use super::config::{Storage, TreeConfig};
use super::errors::{Error, Result};
use super::snapshot::Snapshot;
use super::stats::OpenStats;
use super::stats::{
    BlobStats, CheckpointerStats, JournalStats, RouteCacheStats, TreeStats, VacuumStats,
};
use super::view::View;
use crate::concurrency::{CommitGate, EndpointLocks, Gate};
use crate::engine;
use crate::engine::{
    KeyRangeBuilder, KeyRangeEntry, KeyRangeEntryRef, KeyScanOutcome, PrefixCount, RangeBuilder,
};
use crate::journal::codec::{
    encode_erase_record, encode_insert_record, encode_rename_object_record,
    encoded_erase_record_len, encoded_insert_record_len, encoded_rename_object_record_len,
    BatchEncoder, RECORD_FOOTER_SIZE, RECORD_HEADER_SIZE,
};
use crate::journal::reader::replay;
use crate::journal::wal_op::WalOp;
use crate::journal::Journal;
use crate::layout::{BlobGuid, DATA_AREA_START, PAGE_SIZE, ROOT_BLOB_GUID};
use crate::store::blob_store::{AlignedBlobBuf, BlobStore, FileBlobStore, MemoryBlobStore};
use crate::store::{
    BlobFrame, BlobFrameRef, BufferManager, BufferStats, CachedBlob, DirtySnapshotEntry,
    WriteDeltaEntry, WriteThroughEntry,
};

use super::atomic::{AtomicBatch, BatchOp, Record, RecordVersion};

const ONLINE_COMPACT_BLOB_BUDGET: usize = 256;
const ONLINE_MERGE_PARENT_BUDGET: usize = 256;
const SHAPE_UNDERFILLED_CHILD_FILL_PER_MILLE: u32 = 350;
const SHAPE_OVERFULL_CHILD_FILL_PER_MILLE: u32 = 850;
const WRITE_DELTA_FLUSH_THRESHOLD: usize = 262_144;

type BatchOverlay = HashMap<Vec<u8>, Option<Record>>;
type CheckpointMap = HashMap<BlobGuid, u64>;
type CheckpointBytes = Vec<(BlobGuid, u64, u64, AlignedBlobBuf)>;

/// Per-key result of [`Tree::put_many_if_absent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutOutcome {
    /// The key was absent and is now created.
    Created,
    /// A live record already existed; nothing was written.
    AlreadyExists,
}

#[derive(Clone)]
pub(crate) struct TreeRuntime {
    endpoint_locks: Arc<EndpointLocks>,
    route_cache: Arc<engine::RouteCache>,
    prefix_list_cache: Arc<engine::PrefixListCache>,
    mutation_gate: Arc<Gate>,
    dropped: Arc<AtomicBool>,
}

impl TreeRuntime {
    pub(crate) fn new() -> Self {
        Self {
            endpoint_locks: Arc::new(EndpointLocks::new()),
            route_cache: Arc::new(engine::RouteCache::new()),
            prefix_list_cache: Arc::new(engine::PrefixListCache::new()),
            mutation_gate: Arc::new(Gate::new()),
            dropped: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn mark_dropped(&self) {
        self.dropped.store(true, Ordering::Release);
    }

    pub(crate) fn is_dropped(&self) -> bool {
        self.dropped.load(Ordering::Acquire)
    }
}

struct DirtyWriteOutcome {
    wrote_any: bool,
    failed: CheckpointMap,
    first_err: Option<Error>,
}

struct PendingDeleteOutcome {
    failed: CheckpointMap,
    first_err: Option<Error>,
}

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

pub(crate) fn count_scan_limit(limit: usize) -> usize {
    if limit == 0 {
        usize::MAX
    } else {
        limit.saturating_add(1)
    }
}

pub(crate) fn prefix_count_from_seen(
    seen: u64,
    limit: usize,
    outcome: KeyScanOutcome,
) -> PrefixCount {
    if limit == 0 {
        PrefixCount {
            count: seen,
            exact: true,
            stats: outcome.stats,
            cache_hit: outcome.cache_hit,
        }
    } else {
        let limit_u64 = limit as u64;
        PrefixCount {
            count: seen.min(limit_u64),
            exact: seen <= limit_u64,
            stats: outcome.stats,
            cache_hit: outcome.cache_hit,
        }
    }
}

/// An `holt` tree — your handle to one metadata store.
///
/// Clone the handle to share the same backing store: the
/// internal `BufferManager` is held via `Arc`.
///
/// ## Concurrency
///
/// - **Point reads** (`get`) run against
///   `HybridLatch::read_optimistic` — they capture each blob's
///   latch version, read the bytes, then `validate()`. Cross-blob
///   hops revalidate the parent `BlobNode` edge under a short
///   shared blob latch before pinning the child, so reads do not
///   enter the tree-wide maintenance gate. Restarts from the root
///   on a torn read. Never blocks foreground writers and never
///   block each other.
/// - **Range reads** (`range`, `scan`, `range_keys`,
///   `scan_keys`) use a versioned cursor. Each cursor frame
///   records the blob content version it was built from; if an
///   interleaved writer changes a frame, the iterator discards its
///   stack and performs a marker-aware seek from the last emitted
///   key / delimiter lower bound.
/// - **Writes** (`put`, `delete`) enter the shared side of
///   `maintenance_gate`, lock the key's endpoint shard, then hold the
///   per-blob `HybridLatch` exclusively for the blobs they touch.
///   Persistent trees enter the writer-shared `commit_gate` while
///   publishing dirty state and the journal record. If
///   `TreeConfig::wal_sync` is set, writes wait for the journal worker
///   after leaving both gates.
/// - **Maintenance** (`compact`, background merge) takes short
///   exclusive windows on `maintenance_gate` while folding/deleting
///   cross-blob edges. Blob-local compaction runs on the shared
///   side under per-blob latches. Point reads rely on parent/child
///   blob latches instead of this tree-wide gate; range scans and
///   foreground writers enter the shared side, while `atomic` and
///   scoped `view` capture enter the exclusive side.
/// - **`rename`** locks the two endpoint shards for `src` and
///   `dst` in canonical order after entering the shared maintenance
///   side. `put` / `delete` lock their single endpoint shard, so a
///   rename cannot interleave with writes touching either endpoint.
///   `get` never takes these endpoint locks.
#[derive(Clone)]
pub struct Tree {
    cfg: TreeConfig,
    store: Arc<BufferManager>,
    /// Logical tree id carried by WAL records.
    tree_id: u64,
    /// GUID of the blob holding this tree's root.
    root_guid: BlobGuid,
    /// Cached pin on the root blob — held for the life of this
    /// `Tree` handle so every `get` / `put` / `delete` / `rename`
    /// skips the `BufferManager`'s `Mutex<HashMap>` lookup on
    /// the root hop. Cross-blob descents still pin children
    /// through the BM as normal.
    root_pin: Arc<CachedBlob>,
    /// Serializes writes that touch the same logical endpoint shard.
    ///
    /// Single-key writes lock one shard; rename locks the source and
    /// destination shards in canonical order. Disjoint endpoints stay
    /// concurrent and still coordinate through per-blob latches.
    endpoint_locks: Arc<EndpointLocks>,
    /// Parent-validated route cache for path-shaped large trees.
    /// Entries cache prefix anchors at BlobNode crossings and are
    /// validated against the parent blob's latch version before
    /// use.
    route_cache: Arc<engine::RouteCache>,
    /// Tree-wide structural-maintenance gate.
    ///
    /// Range and foreground write paths enter the shared side while
    /// they may cross `BlobNode` boundaries. `atomic()` and
    /// `view()` enter the exclusive side to make predicate/apply and
    /// topology/copy phases linear. Point reads rely only on
    /// parent/child blob latches.
    maintenance_gate: Arc<Gate>,
    /// Monotonically-increasing sequence stamped on every record.
    /// On open the tree replays the WAL and resumes at
    /// `highest_seq + 1`.
    next_seq: Arc<AtomicU64>,
    /// Writer-shared / checkpoint-exclusive publish barrier for
    /// persistent mode. Foreground writers can mutate disjoint
    /// blobs concurrently, but checkpoint waits until every
    /// admitted writer has published its dirty state and journal
    /// record before the checkpoint captures versioned store bytes.
    commit_gate: Arc<CommitGate>,
    /// Group-commit WAL worker — `Some` for persistent trees,
    /// `None` for memory trees.
    journal: Option<Arc<Journal>>,
    /// Direct-mapped cache for short hot key-only prefix scans.
    /// Conservatively invalidated by `next_seq`, so every write
    /// makes old entries miss.
    prefix_list_cache: Arc<engine::PrefixListCache>,
    mutation_gate: Arc<Gate>,
    /// Shared liveness flag for DB-managed named trees. `DB::drop_tree`
    /// flips it so existing handles can no longer publish writes.
    dropped: Arc<AtomicBool>,
    /// Background checkpointer handle. `Some` iff
    /// `cfg.checkpoint.enabled`. Shared via `Arc` so the thread
    /// shuts down on the **last** `Tree` clone's drop, not the
    /// first. Exposed to `Tree::stats` for counter readout.
    checkpointer: Option<Arc<crate::checkpoint::Checkpointer>>,
    /// Reopen-time recovery telemetry captured once at open.
    open_stats: OpenStats,
}

impl std::fmt::Debug for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tree")
            .field("storage", &self.cfg.storage)
            .field("tree_id", &self.tree_id)
            .field("root_guid", &self.root_guid)
            .finish_non_exhaustive()
    }
}

fn encoded_batch_record_len(ops: &[BatchOp]) -> usize {
    let body_prefix_len = 8 + 4; // tree_id + inner_count
    let mut len = RECORD_HEADER_SIZE + body_prefix_len + RECORD_FOOTER_SIZE;
    let mut i = 0usize;
    while i < ops.len() {
        if let Some(plan) = insert_run_plan(ops, i) {
            len += plan.encoded_len(&ops[i..i + plan.len]);
            i += plan.len;
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

const INSERT_PREFIX_RUN_MIN_PREFIX: usize = 12;
const INSERT_PREFIX_RUN_MIN_COUNT: usize = 2;

#[derive(Clone, Copy)]
enum InsertRunEncoding {
    Fixed { key_len: usize, value_len: usize },
    Prefix { prefix_len: usize },
}

#[derive(Clone, Copy)]
struct InsertRunPlan {
    len: usize,
    encoding: InsertRunEncoding,
}

impl InsertRunPlan {
    fn encoded_len(self, ops: &[BatchOp]) -> usize {
        match self.encoding {
            InsertRunEncoding::Fixed { key_len, value_len } => {
                if self.len == 1 {
                    1 + 8 + 4 + key_len + 4 + value_len
                } else {
                    1 + 8 + 4 + 4 + 4 + self.len * (key_len + value_len)
                }
            }
            InsertRunEncoding::Prefix { prefix_len } => {
                let mut len = 1 + 8 + 4 + 4 + prefix_len;
                for op in ops {
                    let (key, value) = batch_insert_parts(op).expect("prefix run insert op");
                    len += 4 + (key.len() - prefix_len) + 4 + value.len();
                }
                len
            }
        }
    }
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

fn batch_insert_condition(op: &BatchOp) -> Option<engine::InsertCondition> {
    match op {
        BatchOp::Put { .. } => Some(engine::InsertCondition::Always),
        BatchOp::PutIfAbsent { .. } => Some(engine::InsertCondition::IfAbsent),
        BatchOp::CompareAndPut { expected, .. } => {
            Some(engine::InsertCondition::IfVersion(expected.as_u64()))
        }
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

fn insert_run_plan(ops: &[BatchOp], start: usize) -> Option<InsertRunPlan> {
    let (first_key, first_value) = batch_insert_parts(&ops[start])?;
    let fixed_len = same_shape_insert_run_len(ops, start);
    let fixed = InsertRunPlan {
        len: fixed_len,
        encoding: InsertRunEncoding::Fixed {
            key_len: first_key.len(),
            value_len: first_value.len(),
        },
    };

    let Some(prefix) = same_prefix_insert_run(ops, start) else {
        return Some(fixed);
    };
    let prefix_plan = InsertRunPlan {
        len: prefix.len,
        encoding: InsertRunEncoding::Prefix {
            prefix_len: prefix.prefix_len,
        },
    };

    let individual_len = ops[start..start + prefix.len]
        .iter()
        .map(|op| batch_insert_parts(op).expect("prefix run insert op"))
        .map(|(key, value)| 1 + 8 + 4 + key.len() + 4 + value.len())
        .sum::<usize>();
    if prefix_plan.encoded_len(&ops[start..start + prefix.len]) < individual_len {
        Some(prefix_plan)
    } else {
        Some(fixed)
    }
}

struct PrefixInsertRun {
    len: usize,
    prefix_len: usize,
}

fn same_prefix_insert_run(ops: &[BatchOp], start: usize) -> Option<PrefixInsertRun> {
    let (first_key, _) = batch_insert_parts(&ops[start])?;
    let mut prefix_len = first_key.len();
    let mut len = 1usize;
    let mut end = start + 1;
    while end < ops.len() {
        let Some((key, _)) = batch_insert_parts(&ops[end]) else {
            break;
        };
        prefix_len = common_prefix_len(&first_key[..prefix_len], key);
        if prefix_len < INSERT_PREFIX_RUN_MIN_PREFIX {
            break;
        }
        len += 1;
        end += 1;
    }
    (len >= INSERT_PREFIX_RUN_MIN_COUNT).then_some(PrefixInsertRun { len, prefix_len })
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    let len = left.len().min(right.len());
    let mut i = 0usize;
    while i < len && left[i] == right[i] {
        i += 1;
    }
    i
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
        let bm = Self::open_buffer_manager(&cfg)?;
        Self::open_inner(cfg, bm, /*attach_journal=*/ true)
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
    /// `BufferManager` whose generic/custom-store cache is capped at
    /// `cfg.buffer_pool_size` resident blob frames.
    /// `BufferManager` owns the in-memory blob cache; the walker
    /// pins blobs from it for both reads and writes — no separate
    /// root buffer in `Tree`.
    ///
    /// If the store doesn't yet contain a root blob, initialises
    /// an empty one and writes it through, flushing before
    /// returning.
    pub fn open_with_blob_store(cfg: TreeConfig, store: Arc<dyn BlobStore>) -> Result<Self> {
        let bm = Arc::new(BufferManager::new(store, cfg.buffer_pool_size));
        Self::open_inner(cfg, bm, /*attach_journal=*/ false)
    }

    pub(crate) fn open_buffer_manager(cfg: &TreeConfig) -> Result<Arc<BufferManager>> {
        let bm = match &cfg.storage {
            Storage::Memory => {
                let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
                Arc::new(BufferManager::new(store, cfg.buffer_pool_size))
            }
            Storage::File { dir } => {
                #[cfg(all(target_os = "linux", feature = "io-uring"))]
                {
                    let store = Arc::new(FileBlobStore::open_with_buffer_pool_hint(
                        dir,
                        cfg.buffer_pool_size,
                    )?);
                    let store_dyn: Arc<dyn BlobStore> = store.clone();
                    let alloc_store = Arc::clone(&store);
                    Arc::new(BufferManager::new_file(
                        store_dyn,
                        cfg.buffer_pool_size,
                        move || {
                            // SAFETY: BufferManager initializes every
                            // returned buffer before reading it.
                            unsafe { alloc_store.alloc_blob_buf_uninit() }
                        },
                    ))
                }
                #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
                {
                    let store: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(dir)?);
                    Arc::new(BufferManager::new_file(store, cfg.buffer_pool_size, || {
                        // SAFETY: BufferManager initializes every
                        // returned buffer before reading it.
                        unsafe { AlignedBlobBuf::uninit() }
                    }))
                }
            }
        };
        Ok(bm)
    }

    fn open_inner(cfg: TreeConfig, bm: Arc<BufferManager>, attach_journal: bool) -> Result<Self> {
        let root_guid = ROOT_BLOB_GUID;
        ensure_root_blob(&bm, root_guid)?;

        let mut open_stats = OpenStats::default();
        // Restore the CoW epoch above every persisted frame's
        // `created_epoch` (the high-water is stamped on the live root
        // at each snapshot) so snapshots taken after reopen are correct.
        {
            let root = bm.pin(root_guid)?;
            let high_water = crate::layout::frame_epoch_high_water(root.read().as_slice());
            bm.set_current_epoch(high_water);
        }
        // File-backed WAL trees replay every durable record onto the
        // BM-cached blob image: the on-disk blob lags the WAL between
        // the last `Tree::checkpoint` and now.
        let (journal, next_seq) = if attach_journal {
            match cfg.wal_path() {
                None => (None, 1u64),
                Some(path) => {
                    let next_seq = if path.exists() {
                        let start = std::time::Instant::now();
                        let (next_seq, replay_stats) = replay_wal(&path, &bm, |tree_id| {
                            if tree_id == 0 {
                                Ok(root_guid)
                            } else {
                                Err(Error::ReplaySanityFailed {
                                    context: "WAL record tree_id does not belong to this Tree",
                                    record_offset: 0,
                                })
                            }
                        })?;
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
                    let journal = Journal::open_or_create(&path, /*tree_id=*/ 0)?;
                    (Some(Arc::new(journal)), next_seq)
                }
            }
        } else {
            (None, 1u64)
        };

        // Shared structural gate for foreground writers, manual
        // compact, and the background merge pass.
        let maintenance_gate = Arc::new(Gate::new());
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

        Self::from_shared(
            cfg,
            root_guid,
            0,
            bm,
            TreeRuntime::new(),
            maintenance_gate,
            Arc::new(AtomicU64::new(next_seq)),
            commit_gate,
            journal,
            checkpointer,
            open_stats,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_shared(
        cfg: TreeConfig,
        root_guid: BlobGuid,
        tree_id: u64,
        bm: Arc<BufferManager>,
        runtime: TreeRuntime,
        maintenance_gate: Arc<Gate>,
        next_seq: Arc<AtomicU64>,
        commit_gate: Arc<CommitGate>,
        journal: Option<Arc<Journal>>,
        checkpointer: Option<Arc<crate::checkpoint::Checkpointer>>,
        open_stats: OpenStats,
    ) -> Result<Self> {
        let root_pin = bm.pin(root_guid)?;
        Ok(Self {
            cfg,
            store: bm,
            tree_id,
            root_guid,
            root_pin,
            endpoint_locks: runtime.endpoint_locks,
            route_cache: runtime.route_cache,
            maintenance_gate,
            next_seq,
            commit_gate,
            journal,
            prefix_list_cache: runtime.prefix_list_cache,
            mutation_gate: runtime.mutation_gate,
            dropped: runtime.dropped,
            checkpointer,
            open_stats,
        })
    }

    pub(crate) fn mutation_gate(&self) -> Arc<Gate> {
        Arc::clone(&self.mutation_gate)
    }

    fn ensure_live(&self) -> Result<()> {
        if self.dropped.load(Ordering::Acquire) {
            Err(Error::TreeDropped)
        } else {
            Ok(())
        }
    }

    /// Look up `key`. Returns the value bytes, or `None` if no leaf
    /// matches.
    ///
    /// Pays one allocation + memcpy per hit; on a miss returns
    /// `Ok(None)` with no allocation. The walker itself reads
    /// cached blobs optimistically and restarts from the root when
    /// a concurrent writer invalidates its snapshot.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.ensure_live()?;
        self.lookup_record_unlocked(key)
            .map(|record| record.map(|record| record.value))
    }

    /// Look up `key` and return both value bytes and the current
    /// conditional-write version token.
    ///
    /// This is the preferred read before a compare-and-set update:
    /// it avoids the two-lookup `get()` + `get_version()` pattern.
    pub fn get_record(&self, key: &[u8]) -> Result<Option<Record>> {
        self.ensure_live()?;
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
        self.ensure_live()?;
        if let Some(delta) = self.store.lookup_write_delta(self.tree_id, key) {
            return Ok(match delta {
                WriteDeltaEntry::Put { seq, .. } => Some(RecordVersion::new(seq)),
                WriteDeltaEntry::Delete { .. } => None,
            });
        }
        let search = engine::SearchKey::user(key);
        engine::lookup_multi_with(&self.store, &self.root_pin, None, search, |hit| {
            RecordVersion::new(hit.seq)
        })
    }

    fn lookup_record_unlocked(&self, key: &[u8]) -> Result<Option<Record>> {
        if let Some(delta) = self.store.lookup_write_delta(self.tree_id, key) {
            return Ok(match delta {
                WriteDeltaEntry::Put { value, seq, .. } => Some(Record {
                    value,
                    version: RecordVersion::new(seq),
                }),
                WriteDeltaEntry::Delete { .. } => None,
            });
        }
        let search = engine::SearchKey::user(key);
        engine::lookup_multi_with(&self.store, &self.root_pin, None, search, |hit| Record {
            value: hit.value.to_vec(),
            version: RecordVersion::new(hit.seq),
        })
    }

    /// Insert or replace `(key, value)`. Returns `Ok(())`.
    ///
    /// Deferred persistent hot path: the writer checks only whether
    /// the key-set changes, then appends WAL and stages the value in
    /// the write delta. It does not read or clone the previous value
    /// on a same-key update.
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
        if self.can_stage_deferred_write() {
            self.put_deferred(key, value)
        } else {
            self.put_inner_conditional(key, value, engine::InsertCondition::Always)
                .map(|_| ())
        }
    }

    /// Insert `(key, value)` only when `key` has no live record.
    ///
    /// Returns `Ok(true)` when the value was inserted and `Ok(false)`
    /// when a live value already existed. The existence check and
    /// insert happen under the target blob's exclusive latch.
    pub fn put_if_absent(&self, key: &[u8], value: &[u8]) -> Result<bool> {
        self.flush_write_delta_for_tree()?;
        self.put_inner_conditional(key, value, engine::InsertCondition::IfAbsent)
            .map(|outcome| outcome.mutated)
    }

    /// Insert every entry whose key is currently absent, as one atomic
    /// batch, reporting per key whether it was [`PutOutcome::Created`] or
    /// already present.
    ///
    /// Same per-key semantics as [`Self::put_if_absent`], but the
    /// genuinely-new keys commit under a single WAL record (crash-atomic)
    /// and a same-parent run lands in one frame under one latch — the
    /// "create N entries under one directory" metadata path. Duplicate
    /// keys within `entries` create once; later copies report
    /// `AlreadyExists`.
    pub fn put_many_if_absent(&self, entries: &[(&[u8], &[u8])]) -> Result<Vec<PutOutcome>> {
        self.flush_write_delta_for_tree()?;
        let _maintenance = self.maintenance_gate.enter_shared();
        self.ensure_live()?;
        let _mutation = self.mutation_gate.enter_exclusive();

        // Preflight existence (under the mutation gate, so it stays
        // consistent with the apply): present in the tree OR seen earlier
        // in this batch ⇒ `AlreadyExists`; otherwise queue the insert.
        let mut results = Vec::with_capacity(entries.len());
        let mut new_ops: Vec<BatchOp> = Vec::new();
        let mut creating: HashSet<&[u8]> = HashSet::new();
        for &(key, value) in entries {
            let fresh = creating.insert(key) && self.get_version(key)?.is_none();
            if fresh {
                results.push(PutOutcome::Created);
                new_ops.push(BatchOp::PutIfAbsent {
                    key: key.to_vec(),
                    value: value.to_vec(),
                });
            } else {
                results.push(PutOutcome::AlreadyExists);
            }
        }

        if !new_ops.is_empty() {
            let base_seq = self
                .next_seq
                .fetch_add(new_ops.len() as u64, Ordering::Relaxed);
            self.commit_batch(&new_ops, base_seq)?;
        }
        Ok(results)
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
        self.flush_write_delta_for_tree()?;
        self.put_inner_conditional(
            key,
            value,
            engine::InsertCondition::IfVersion(expected_version.as_u64()),
        )
        .map(|outcome| outcome.mutated)
    }

    fn can_stage_deferred_write(&self) -> bool {
        self.journal.is_some() && !self.cfg.is_memory() && !self.cfg.memory_flush_on_write
    }

    pub(crate) fn flush_write_delta_for_tree(&self) -> Result<()> {
        if self.store.write_delta_count_for_tree(self.tree_id) == 0 {
            return Ok(());
        }
        let _maintenance = self.maintenance_gate.enter_shared();
        self.ensure_live()?;
        let _tree_mutation = self.mutation_gate.enter_batch();
        let _commit = self.commit_gate.enter_writer();
        self.store.flush_write_deltas_for_tree(self.tree_id)
    }

    fn put_deferred(&self, key: &[u8], value: &[u8]) -> Result<()> {
        Self::validate_insert_shape(key, value)?;
        let ack = {
            let _mutation = self.maintenance_gate.enter_shared();
            self.ensure_live()?;
            let _tree_mutation = self.mutation_gate.enter_shared();
            let _endpoint = self.endpoint_locks.lock_key(key);
            let creates_key = self.get_version(key)?.is_none();
            let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
            let journal = self
                .journal
                .as_ref()
                .expect("can_stage_deferred_write requires journal");
            let _commit = self.commit_gate.enter_writer();
            let mut record =
                journal.record_buffer(encoded_insert_record_len(key.len(), value.len()));
            encode_insert_record(&mut record, seq, self.tree_id, key, value);
            let ack = journal.submit(record, self.cfg.durability.wal_sync())?;
            self.store.stage_write_delta_put(
                self.tree_id,
                self.root_guid,
                key,
                value,
                seq,
                creates_key,
            );
            ack
        };
        if let Some(ack) = ack {
            ack.wait()?;
        }
        self.flush_write_delta_debt_if_needed()?;
        Ok(())
    }

    fn put_inner_conditional(
        &self,
        key: &[u8],
        value: &[u8],
        condition: engine::InsertCondition,
    ) -> Result<engine::InsertOutcome> {
        let search = engine::SearchKey::user(key);

        let (outcome, journal_ack) = {
            let _mutation = self.maintenance_gate.enter_shared();
            self.ensure_live()?;
            let _tree_mutation = self.mutation_gate.enter_shared();
            let _endpoint = self.endpoint_locks.lock_key(key);
            let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

            if let Some(journal) = &self.journal {
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
                    encode_insert_record(&mut record, seq, self.tree_id, key, value);
                    let ack = journal.submit(record, self.cfg.durability.wal_sync())?;
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
            }
        };
        if let Some(ack) = journal_ack {
            ack.wait()?;
        }
        Ok(outcome)
    }

    /// Remove `key`. Returns `Ok(true)` if a leaf was removed,
    /// `Ok(false)` if no leaf matched.
    ///
    /// Deferred persistent hot path: the writer checks only the live
    /// version before appending WAL and staging a tombstone. It does
    /// not read or clone the previous value.
    ///
    /// Walks across `BlobNode` crossings. Child-local mutations
    /// are staged through the BM dirty set; any conservative
    /// fallback that unlinks a child blob queues the manifest
    /// delete through the same W2D-safe pending-delete protocol.
    pub fn delete(&self, key: &[u8]) -> Result<bool> {
        if self.can_stage_deferred_write() {
            self.delete_deferred(key)
        } else {
            self.delete_inner_conditional(key, engine::EraseCondition::Always)
                .map(|outcome| outcome.mutated)
        }
    }

    /// Remove `key` only when the live record currently carries
    /// `expected_version`.
    ///
    /// Returns `Ok(false)` if the key is missing, already
    /// tombstoned, or has been updated since the caller obtained
    /// the version.
    pub fn delete_if_version(&self, key: &[u8], expected_version: RecordVersion) -> Result<bool> {
        self.flush_write_delta_for_tree()?;
        self.delete_inner_conditional(
            key,
            engine::EraseCondition::IfVersion(expected_version.as_u64()),
        )
        .map(|outcome| outcome.mutated)
    }

    fn delete_deferred(&self, key: &[u8]) -> Result<bool> {
        let ack = {
            let _mutation = self.maintenance_gate.enter_shared();
            self.ensure_live()?;
            let _tree_mutation = self.mutation_gate.enter_shared();
            let _endpoint = self.endpoint_locks.lock_key(key);
            if self.get_version(key)?.is_none() {
                return Ok(false);
            }
            let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
            let journal = self
                .journal
                .as_ref()
                .expect("can_stage_deferred_write requires journal");
            let _commit = self.commit_gate.enter_writer();
            let mut record = journal.record_buffer(encoded_erase_record_len(key.len()));
            encode_erase_record(&mut record, seq, self.tree_id, key);
            let ack = journal.submit(record, self.cfg.durability.wal_sync())?;
            self.store
                .stage_write_delta_delete(self.tree_id, self.root_guid, key, seq);
            ack
        };
        if let Some(ack) = ack {
            ack.wait()?;
        }
        self.flush_write_delta_debt_if_needed()?;
        Ok(true)
    }

    fn flush_write_delta_debt_if_needed(&self) -> Result<()> {
        if self.store.write_delta_count_for_tree(self.tree_id) >= WRITE_DELTA_FLUSH_THRESHOLD {
            self.flush_write_delta_for_tree()?;
        }
        Ok(())
    }

    fn delete_inner_conditional(
        &self,
        key: &[u8],
        condition: engine::EraseCondition,
    ) -> Result<engine::EraseOutcome> {
        let search = engine::SearchKey::user(key);
        // Pre-allocate the seq before the walker descends so any
        // child blob the walker touches can `mark_dirty(child, seq)`
        // — invariant W2D (see `BufferManager` module docs) demands
        // a single seq for the whole op across all blobs it dirties.
        // A no-op delete (key absent) still burns the seq; that's
        // fine — `next_seq` is monotonic and the unused seq doesn't
        // appear in any WAL record or dirty entry.

        let (outcome, journal_ack) = {
            let _mutation = self.maintenance_gate.enter_shared();
            self.ensure_live()?;
            let _tree_mutation = self.mutation_gate.enter_shared();
            let _endpoint = self.endpoint_locks.lock_key(key);
            let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

            if let Some(journal) = &self.journal {
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
                    encode_erase_record(&mut record, seq, self.tree_id, key);
                    let ack = journal.submit(record, self.cfg.durability.wal_sync())?;
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
            }
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
    /// Atomic with respect to writes touching either endpoint shard;
    /// unrelated endpoints can proceed in parallel. The op emits a
    /// single `RenameObject` WAL record so its erase + insert phases
    /// recover atomically on replay.
    pub fn rename(&self, src: &[u8], dst: &[u8], force: bool) -> Result<()> {
        self.flush_write_delta_for_tree()?;
        let src_search = engine::SearchKey::user(src);
        let dst_search = engine::SearchKey::user(dst);

        let journal_ack = {
            let _mutation = self.maintenance_gate.enter_shared();
            self.ensure_live()?;
            let _tree_mutation = self.mutation_gate.enter_shared();
            let _endpoints = self.endpoint_locks.lock_pair(src, dst);
            let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);

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
            // atomic from the dirty-tracking perspective.
            if let Some(journal) = &self.journal {
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
                encode_rename_object_record(&mut record, seq, self.tree_id, src, dst, force);
                journal.submit(record, self.cfg.durability.wal_sync())?
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
            }
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
    /// the batch while holding the exclusive maintenance gate and emits
    /// **one** WAL record (`WalOp::Batch`) covering the sequence.
    ///
    /// ## Atomicity contract
    ///
    /// - **Logical atomicity**: yes. Missing rename sources,
    ///   destination collisions, and failed conditional guards are
    ///   detected before any walker mutation. A failing rename
    ///   returns `Err`; a failed conditional guard returns
    ///   `Ok(false)`. Neither publishes partial user mutations.
    /// - **Runtime visibility**: foreground writes, range scans, and view
    ///   capture are blocked while the batch applies, so they cannot
    ///   observe an intermediate batch state. Point reads stay optimistic
    ///   and wait-free; each individual key read linearizes either before
    ///   or after the corresponding leaf mutation.
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

    pub(crate) fn apply_batch(&self, pending: Vec<BatchOp>) -> Result<bool> {
        self.flush_write_delta_for_tree()?;
        let _maintenance = self.maintenance_gate.enter_shared();
        self.ensure_live()?;
        let _tree_mutation = self.mutation_gate.enter_batch();
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
        if count != 0 {
            self.commit_batch(&pending, base_seq)?;
        }
        Ok(true)
    }

    /// Commit a pre-validated batch under one WAL record. The caller
    /// holds the maintenance + mutation gates and has reserved the
    /// `base_seq` range; every inner op applies via the run walker.
    fn commit_batch(&self, pending: &[BatchOp], base_seq: u64) -> Result<()> {
        // W2D-strict protocol: all inner ops' walker mutations +
        // `mark_dirty` calls, plus the single envelope WAL submit, happen
        // under `commit_gate` — see `Tree::put_inner_conditional`.
        if let Some(journal) = &self.journal {
            let ack = {
                let _commit = self.commit_gate.enter_writer();
                let mut record = journal.record_buffer(encoded_batch_record_len(pending));
                let mut enc = BatchEncoder::begin(&mut record, base_seq, self.tree_id);
                self.apply_batch_walker_inline(pending, base_seq, Some(&mut enc))?;
                let _n = enc.finish();
                journal.submit(record, self.cfg.durability.wal_sync())?
            };
            if let Some(ack) = ack {
                ack.wait()?;
            }
        } else {
            self.apply_batch_walker_inline(pending, base_seq, None)?;
            if self.cfg.memory_flush_on_write {
                self.flush_dirty_inline()?;
                self.flush_pending_deletes_inline()?;
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_batch(&self, pending: &[BatchOp], base_seq: u64) -> Result<bool> {
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
        if overlay.is_empty() {
            return self.base_prefix_empty(prefix);
        }

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

    fn base_prefix_empty(&self, prefix: &[u8]) -> Result<bool> {
        let outcome = self.scan_keys(prefix).prefix_exists_unlocked()?;
        Ok(outcome.stats.returned + outcome.stats.rollup == 0)
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
    pub(crate) fn apply_batch_walker_inline(
        &self,
        pending: &[BatchOp],
        base_seq: u64,
        mut enc: Option<&mut crate::journal::codec::BatchEncoder<'_>>,
    ) -> Result<()> {
        let mut seq_offset = 0u64;
        let mut i = 0usize;
        while i < pending.len() {
            if let Some(plan) = insert_run_plan(pending, i) {
                let run_len = plan.len;
                let first_seq = base_seq + seq_offset;
                self.apply_batch_insert_run_walker(&pending[i..i + run_len], first_seq)?;
                seq_offset += run_len as u64;
                if let Some(enc) = enc.as_deref_mut() {
                    match plan.encoding {
                        InsertRunEncoding::Fixed { key_len, value_len } => {
                            enc.push_insert_run(
                                self.tree_id,
                                run_len,
                                key_len,
                                value_len,
                                pending[i..i + run_len]
                                    .iter()
                                    .map(|op| batch_insert_parts(op).expect("insert run op")),
                            );
                        }
                        InsertRunEncoding::Prefix { prefix_len } => {
                            let (key, _) = batch_insert_parts(&pending[i])
                                .expect("prefix run begins with insert-like op");
                            enc.push_insert_prefix_run(
                                self.tree_id,
                                &key[..prefix_len],
                                run_len,
                                pending[i..i + run_len].iter().map(|op| {
                                    batch_insert_parts(op).expect("prefix insert run op")
                                }),
                            );
                        }
                    }
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
                        enc.push_erase(self.tree_id, key);
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
                        enc.push_erase(self.tree_id, key);
                    }
                }
                BatchOp::AssertVersion { .. } | BatchOp::AssertPrefixEmpty { .. } => {}
                BatchOp::Rename { src, dst, force } => {
                    self.apply_batch_rename_walker(src, dst, *force, seq)?;
                    if let Some(enc) = enc.as_deref_mut() {
                        enc.push_rename_object(self.tree_id, src, dst, *force);
                    }
                }
            }
            i += 1;
        }
        Ok(())
    }

    fn apply_batch_insert_run_walker(&self, ops: &[BatchOp], first_seq: u64) -> Result<()> {
        if ops.len() == 1 {
            return self.apply_single_insert_batch_op(&ops[0], first_seq);
        }

        let mut items = Vec::with_capacity(ops.len());
        for (idx, op) in ops.iter().enumerate() {
            let seq = first_seq + idx as u64;
            let (key, value) = batch_insert_parts(op).expect("not an insert-like batch op");
            let condition = batch_insert_condition(op).expect("not an insert-like batch op");
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

    fn apply_single_insert_batch_op(&self, op: &BatchOp, seq: u64) -> Result<()> {
        let (key, value) = batch_insert_parts(op).expect("not an insert-like batch op");
        let condition = batch_insert_condition(op).expect("not an insert-like batch op");
        let outcome = engine::insert_multi_conditional(
            &self.store,
            &self.root_pin,
            Some(&self.route_cache),
            engine::SearchKey::user(key),
            value,
            seq,
            condition,
        )?;
        if !outcome.mutated {
            return Err(Error::Internal("atomic preflight missed insert guard"));
        }
        if outcome.root_dirty {
            self.store
                .mark_dirty_cached(self.root_guid, seq, self.root_pin.as_ref());
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
        let builder = self.range_builder_base();
        match self.flush_write_delta_for_tree() {
            Ok(()) => builder,
            Err(e) => builder.with_preflight_error(e),
        }
    }

    fn range_builder_base(&self) -> RangeBuilder {
        RangeBuilder::new(
            Arc::clone(&self.store),
            Arc::clone(&self.root_pin),
            self.root_guid,
            Arc::clone(&self.maintenance_gate),
        )
        .with_mutation_gate(Arc::clone(&self.mutation_gate))
        .with_write_delta_flush(self.tree_id, Arc::clone(&self.commit_gate))
        .with_liveness(Arc::clone(&self.dropped))
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
        KeyRangeBuilder::new(self.range_builder_base()).with_prefix_list_cache(
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

    /// Count live keys under `prefix`, optionally capped by `limit`.
    ///
    /// `limit == 0` means exact / unbounded. For non-zero limits, the
    /// implementation walks at most one entry past the limit so callers can
    /// distinguish "exactly N" from "N or more" without materialising a full
    /// giant directory.
    pub fn prefix_count(&self, prefix: &[u8], limit: usize) -> Result<PrefixCount> {
        let scan_limit = count_scan_limit(limit);
        let mut seen = 0u64;
        let outcome = self
            .scan_keys(prefix)
            .visit_with_outcome(scan_limit, |entry| {
                if let KeyRangeEntryRef::Key { .. } = entry {
                    seen = seen.saturating_add(1);
                }
                Ok(())
            })?;
        Ok(prefix_count_from_seen(seen, limit, outcome))
    }

    /// Run a read-only transaction over a prefix snapshot.
    ///
    /// Internally a [`Self::snapshot`] held for the duration of `read`: a
    /// copy-on-write capture (O(1) — only the root frame is copied; later
    /// live writes fork the frames this view references). Writes committed
    /// after the capture are invisible to all reads made through the view,
    /// and point lookup / range / list keep using the ART walker.
    ///
    /// A view is scoped: reads outside `prefix` return
    /// [`Error::OutsideViewScope`]. Use `prefix = b""` only when a
    /// whole-tree snapshot is intentional.
    pub fn view<F, R>(&self, prefix: &[u8], read: F) -> Result<R>
    where
        F: FnOnce(&View) -> Result<R>,
    {
        let snap = self.snapshot(prefix)?;
        read(snap.view())
    }

    /// Capture a stable copy-on-write [`Snapshot`] of the subtree under
    /// `prefix`.
    ///
    /// A snapshot copies only the root frame up front and shares the rest
    /// with the live tree; subsequent writes fork (copy-on-write) only
    /// the frames the snapshot still references. Creation is O(one frame
    /// copy), reads have 1× amplification, and the per-write overhead is
    /// zero whenever no snapshot is live. [`Self::view`] is the same
    /// mechanism exposed as a scoped closure; this returns an owned handle.
    ///
    /// Reads outside `prefix` return [`Error::OutsideViewScope`]; use
    /// `prefix = b""` for a whole-tree snapshot. The snapshot stays valid
    /// until its handle is dropped (or [`Snapshot::retire`] is called).
    pub fn snapshot(&self, prefix: &[u8]) -> Result<Snapshot> {
        self.flush_write_delta_for_tree()?;
        let _maintenance = self.maintenance_gate.enter_shared();
        self.ensure_live()?;
        // Freeze live writers so the root frame is byte-stable for the
        // copy, and so no writer is mid-overwriting a soon-to-be-shared
        // frame at the instant the fork barrier rises.
        let _freeze = self.mutation_gate.enter_exclusive();
        Ok(self.snapshot_unlocked(prefix))
    }

    /// Reclaim copy-on-write frames left unreachable by a crash that
    /// happened while a snapshot was live: the in-memory orphan list is
    /// lost on restart, so those forked-away frames would otherwise leak
    /// in the store forever. Sweeps every persisted frame not reachable
    /// from the live root or a live snapshot root, returning the count
    /// freed. Idempotent.
    ///
    /// Only supported on standalone trees — trees opened through a `DB`
    /// share one buffer manager, so a safe sweep needs a DB-wide pass and
    /// this returns [`Error::GcRequiresStandaloneTree`].
    pub fn gc(&self) -> Result<usize> {
        if self.tree_id != 0 {
            return Err(Error::GcRequiresStandaloneTree);
        }
        self.flush_write_delta_for_tree()?;
        let _maintenance = self.maintenance_gate.enter_shared();
        self.ensure_live()?;
        // Freeze writers so the reachable set is stable across the walk.
        let _freeze = self.mutation_gate.enter_exclusive();

        let mut reachable: HashSet<BlobGuid> = HashSet::new();
        reachable.insert(self.root_guid);
        reachable.extend(engine::collect_blob_guids(&self.store, self.root_guid)?);
        for snap_root in self.store.snapshot_roots() {
            reachable.insert(snap_root);
            reachable.extend(engine::collect_blob_guids(&self.store, snap_root)?);
        }
        self.store.gc_sweep_unreachable(&reachable)
    }

    /// Reclaim logical garbage and physical space from free store slots.
    ///
    /// [`Self::gc`] only makes unreachable blob frames reusable. `vacuum`
    /// follows that with a checkpoint and store-level physical cleanup:
    /// live high-water slots may be relocated into lower reusable holes,
    /// packed-file tails that are durably free are truncated, and on
    /// Linux any remaining reusable middle slots are hole-punched. This
    /// can change physical slot addresses inside the file backend, but
    /// GUID/key visibility is unchanged.
    ///
    /// Only supported on standalone trees. Named trees inside a
    /// [`DB`](crate::DB) share one store; use [`DB::vacuum`](crate::DB::vacuum)
    /// there so every live tree participates in the reachable set.
    pub fn vacuum(&self) -> Result<VacuumStats> {
        let unreachable = self.gc()?;
        self.checkpoint()?;
        let mut stats = self.store.vacuum_storage()?;
        stats.unreachable_blobs = unreachable;
        Ok(stats)
    }

    /// [`Self::snapshot`] without taking this tree's maintenance/mutation
    /// gates — the caller must already hold the mutation gate exclusively.
    /// Used by [`crate::DB::view`] to capture several trees atomically
    /// under a single coordinated freeze.
    pub(crate) fn snapshot_unlocked(&self, prefix: &[u8]) -> Snapshot {
        self.snapshot_unlocked_with_scan_fence(prefix, true)
    }

    pub(crate) fn snapshot_unlocked_unfenced(&self, prefix: &[u8]) -> Snapshot {
        self.snapshot_unlocked_with_scan_fence(prefix, false)
    }

    fn snapshot_unlocked_with_scan_fence(
        &self,
        prefix: &[u8],
        fence_live_writers: bool,
    ) -> Snapshot {
        use crate::store::STRUCTURAL_SEQ;

        let snap_root = engine::fresh_blob_guid();
        let root_pin = self.store.install_snapshot_root(snap_root, &self.root_pin);
        let epoch = self.store.register_snapshot(snap_root);

        // Persist the bumped epoch on the live root so a reopened tree
        // restores `current_epoch` above every frame's `created_epoch`.
        {
            let mut root = self.root_pin.write();
            crate::layout::set_frame_epoch_high_water(
                root.as_mut_slice(),
                self.store.current_epoch(),
            );
        }
        self.store
            .mark_dirty_cached(self.root_guid, STRUCTURAL_SEQ, self.root_pin.as_ref());

        let view = View::new(
            prefix.to_vec(),
            Arc::clone(&self.store),
            snap_root,
            root_pin,
            fence_live_writers.then(|| {
                (
                    Arc::clone(&self.maintenance_gate),
                    Arc::clone(&self.mutation_gate),
                )
            }),
        );
        Snapshot::new(view, Arc::clone(&self.store), epoch)
    }

    /// Return `true` if no live key starts with `prefix`.
    ///
    /// This is a point-in-time read helper. Concurrent writers may
    /// make the prefix non-empty immediately after it returns; use
    /// [`AtomicBatch::assert_prefix_empty`] inside [`Self::atomic`] when
    /// the emptiness check must be atomic with subsequent writes.
    pub fn is_prefix_empty(&self, prefix: &[u8]) -> Result<bool> {
        let outcome = self.scan_keys(prefix).prefix_exists()?;
        Ok(outcome.stats.returned + outcome.stats.rollup == 0)
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
    pub(crate) fn flush_dirty_inline(&self) -> Result<()> {
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
                    content_version: None,
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
    pub(crate) fn flush_pending_deletes_inline(&self) -> Result<()> {
        let pending = self.store.snapshot_pending_deletes();
        let mut failed: std::collections::HashMap<BlobGuid, u64> = std::collections::HashMap::new();
        let mut first_err: Option<Error> = None;
        for (guid, seq) in pending {
            match self.store.execute_pending_delete(guid, seq) {
                Ok(true) => {}
                Ok(false) => {
                    failed.insert(guid, seq);
                }
                Err(e) => {
                    failed.insert(guid, seq);
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
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
    /// 1. **Snapshot intent**: drain BM dirty + pending-delete sets
    ///    under `commit_gate` and capture each dirty blob's content
    ///    version. WAL flush failure → restore both snapshots,
    ///    return.
    /// 2. **Clone bytes outside `commit_gate`**: if any blob changed
    ///    after intent capture, restore the whole snapshot and retry
    ///    so manual checkpoint remains a durability barrier.
    /// 3. **Per-blob write-through** with CAS-on-seq. The CAS
    ///    retires the dirty entry only if no racing writer bumped
    ///    it; failures stay in `dirty` for the next round.
    ///    If the snapshot had neither dirty blobs nor pending
    ///    deletes and the store reports no outstanding flush
    ///    work, skip the store Sync path entirely.
    /// 4. **Pre-delete sync** — `store.flush` (`sync_data` on
    ///    the data file + persist the manifest) so step 3's
    ///    writes hit stable storage *before* any manifest delete
    ///    runs. Sync failure → restore pending, return.
    /// 5. **Abort-on-dirty-failure gate**. If any write-through
    ///    at step 3 failed, the round must NOT apply pending
    ///    deletes: a parent that didn't flush might still
    ///    reference a child that's about to be removed from the
    ///    manifest, leaving the on-disk parent pointing into a
    ///    deleted slot. Restore pending and return the dirty
    ///    error. The next round will retry the parent write and
    ///    only then process its child's deletion.
    /// 6. **Apply pending deletes**. User WAL deletes mutate the
    ///    manifest in memory; structural deletes only retire their
    ///    fence and leave orphan cleanup to a future reachability
    ///    GC. Each `execute_pending_delete` is idempotent against a
    ///    missing entry; failures are restored.
    /// 7. **Post-delete sync** — re-`store.flush` iff any manifest
    ///    delete actually applied. Failure → restore the
    ///    already-applied entries so the truncate gate stays
    ///    closed and the next round retries the sync.
    /// 8. **Conditional WAL truncate** — only if
    ///    `dirty_count == 0`, `flushing_count == 0`,
    ///    `pending_delete_count == 0`, and the store reports no
    ///    deferred flush work *now*. A racing writer, restored
    ///    failure, or deferred data/manifest sync must keep the WAL
    ///    alive until a future flush.
    ///
    /// `memory_flush_on_write = false` callers rely on this to make
    /// batched writes survive a crash.
    pub fn checkpoint(&self) -> Result<()> {
        Self::checkpoint_shared_parts(
            &self.store,
            self.journal.as_ref(),
            &self.maintenance_gate,
            &self.commit_gate,
        )
    }

    pub(crate) fn checkpoint_shared_parts(
        store: &Arc<BufferManager>,
        journal: Option<&Arc<Journal>>,
        maintenance_gate: &Arc<Gate>,
        commit_gate: &Arc<CommitGate>,
    ) -> Result<()> {
        let _maintenance = maintenance_gate.enter_shared();

        loop {
            let (snap_dirty, snap_pending, versioned_snap, wal_up_to) =
                Self::capture_checkpoint_intent_shared(store, journal, commit_gate)?;

            if let (Some(journal), Some(up_to)) = (journal, wal_up_to) {
                if let Err(e) = journal.flush_up_to(up_to) {
                    store.restore_pending_deletes(snap_pending);
                    store.restore_dirty(snap_dirty);
                    return Err(e);
                }
            }

            let snap_bytes = match Self::clone_checkpoint_bytes_shared(store, &versioned_snap) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => {
                    store.restore_pending_deletes(snap_pending);
                    store.restore_dirty(snap_dirty);
                    continue;
                }
                Err(e) => {
                    store.restore_pending_deletes(snap_pending);
                    store.restore_dirty(snap_dirty);
                    return Err(e);
                }
            };

            return Self::finish_checkpoint_snapshot_shared(
                store,
                journal,
                commit_gate,
                snap_pending,
                snap_bytes,
            );
        }
    }

    fn clone_checkpoint_bytes_shared(
        store: &Arc<BufferManager>,
        versioned_snap: &[DirtySnapshotEntry],
    ) -> Result<Option<CheckpointBytes>> {
        let mut snap_bytes = Vec::with_capacity(versioned_snap.len());
        for entry in versioned_snap {
            match store.snapshot_bytes_if_version(entry.guid, entry.content_version)? {
                Some(bytes) => {
                    snap_bytes.push((entry.guid, entry.expected_seq, entry.content_version, bytes));
                }
                None => return Ok(None),
            }
        }
        Ok(Some(snap_bytes))
    }

    fn capture_checkpoint_intent_shared(
        store: &Arc<BufferManager>,
        journal: Option<&Arc<Journal>>,
        commit_gate: &Arc<CommitGate>,
    ) -> Result<(
        CheckpointMap,
        CheckpointMap,
        Vec<DirtySnapshotEntry>,
        Option<u64>,
    )> {
        if let Some(journal) = journal {
            let wal_up_to = {
                let _commit = commit_gate.enter_checkpoint();
                journal.queued_work()
            };
            store.flush_write_deltas()?;
            let _commit = commit_gate.enter_checkpoint();
            let snap_dirty = store.snapshot_dirty();
            let snap_pending = store.snapshot_pending_deletes();
            match store.snapshot_dirty_versions(&snap_dirty) {
                Ok(versioned_snap) => {
                    Ok((snap_dirty, snap_pending, versioned_snap, Some(wal_up_to)))
                }
                Err(e) => {
                    store.restore_pending_deletes(snap_pending);
                    store.restore_dirty(snap_dirty);
                    Err(e)
                }
            }
        } else {
            store.flush_write_deltas()?;
            let snap_dirty = store.snapshot_dirty();
            let snap_pending = store.snapshot_pending_deletes();
            match store.snapshot_dirty_versions(&snap_dirty) {
                Ok(versioned_snap) => Ok((snap_dirty, snap_pending, versioned_snap, None)),
                Err(e) => {
                    store.restore_pending_deletes(snap_pending);
                    store.restore_dirty(snap_dirty);
                    Err(e)
                }
            }
        }
    }

    fn finish_checkpoint_snapshot_shared(
        store: &Arc<BufferManager>,
        journal: Option<&Arc<Journal>>,
        commit_gate: &Arc<CommitGate>,
        snap_pending: CheckpointMap,
        snap_bytes: CheckpointBytes,
    ) -> Result<()> {
        let DirtyWriteOutcome {
            wrote_any,
            failed: dirty_failed,
            first_err: first_dirty_err,
        } = Self::write_checkpoint_bytes_shared(store, snap_bytes);
        let had_dirty_failure = !dirty_failed.is_empty();
        if had_dirty_failure {
            store.restore_dirty(dirty_failed);
        }

        if !wrote_any && snap_pending.is_empty() && !store.needs_flush() {
            Self::maybe_truncate_journal_shared(store, journal, commit_gate)?;
            return Ok(());
        }

        // Successful write-throughs already retired their dirty
        // entries; sync them before any manifest delete can land.
        if let Err(e) = store.flush() {
            store.restore_pending_deletes(snap_pending);
            return Err(e);
        }

        if had_dirty_failure {
            store.restore_pending_deletes(snap_pending);
            return Err(first_dirty_err.expect("had_dirty_failure ⇒ first_dirty_err set"));
        }

        let PendingDeleteOutcome {
            failed: pending_failed,
            first_err: first_pending_err,
        } = Self::apply_pending_deletes_shared(store, &snap_pending);
        if !pending_failed.is_empty() {
            store.restore_pending_deletes(pending_failed.clone());
        }
        Self::sync_applied_deletes_shared(store, &snap_pending, &pending_failed)?;

        if let Some(e) = first_pending_err {
            return Err(e);
        }

        Self::maybe_truncate_journal_shared(store, journal, commit_gate)
    }

    fn write_checkpoint_bytes_shared(
        store: &Arc<BufferManager>,
        snap_bytes: CheckpointBytes,
    ) -> DirtyWriteOutcome {
        let entries: Vec<_> = snap_bytes
            .into_iter()
            .map(
                |(guid, expected_seq, content_version, bytes)| WriteThroughEntry {
                    guid,
                    bytes,
                    expected_seq,
                    content_version: Some(content_version),
                },
            )
            .collect();
        let mut failed = CheckpointMap::new();
        let mut first_err = None;
        if !entries.is_empty() {
            let expected: Vec<_> = entries
                .iter()
                .map(|entry| (entry.guid, entry.expected_seq))
                .collect();
            if let Err(e) = store.write_through_batch(&entries) {
                // BlobStore batch failure may have landed any prefix;
                // retry the whole snapshot next round.
                for (guid, expected_seq) in expected {
                    failed.insert(guid, expected_seq);
                }
                first_err = Some(e);
            }
        }
        DirtyWriteOutcome {
            wrote_any: !entries.is_empty(),
            failed,
            first_err,
        }
    }

    fn apply_pending_deletes_shared(
        store: &Arc<BufferManager>,
        snap_pending: &CheckpointMap,
    ) -> PendingDeleteOutcome {
        let mut failed = CheckpointMap::new();
        let mut first_err = None;
        for (guid, seq) in snap_pending {
            match store.execute_pending_delete(*guid, *seq) {
                Ok(true) => {}
                Ok(false) => {
                    failed.insert(*guid, *seq);
                }
                Err(e) => {
                    failed.insert(*guid, *seq);
                    first_err.get_or_insert(e);
                }
            }
        }
        PendingDeleteOutcome { failed, first_err }
    }

    fn sync_applied_deletes_shared(
        store: &Arc<BufferManager>,
        snap_pending: &CheckpointMap,
        pending_failed: &CheckpointMap,
    ) -> Result<()> {
        use crate::store::STRUCTURAL_SEQ;

        let applied_manifest_deletes = snap_pending
            .iter()
            .filter(|(guid, seq)| **seq != STRUCTURAL_SEQ && !pending_failed.contains_key(*guid))
            .count();
        if applied_manifest_deletes > 0 {
            if let Err(e) = store.flush() {
                let restore_applied: CheckpointMap = snap_pending
                    .iter()
                    .filter(|(g, seq)| **seq != STRUCTURAL_SEQ && !pending_failed.contains_key(*g))
                    .map(|(g, s)| (*g, *s))
                    .collect();
                store.restore_pending_deletes(restore_applied);
                return Err(e);
            }
        }
        Ok(())
    }

    fn maybe_truncate_journal_shared(
        store: &Arc<BufferManager>,
        journal: Option<&Arc<Journal>>,
        commit_gate: &Arc<CommitGate>,
    ) -> Result<()> {
        if let Some(journal) = journal {
            if journal.needs_checkpoint() {
                let _commit = commit_gate.enter_checkpoint();
                if store.dirty_count() == 0
                    && store.flushing_count() == 0
                    && store.pending_delete_count() == 0
                    && store.write_delta_count() == 0
                    && !store.needs_flush()
                {
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
        self.flush_write_delta_for_tree()?;
        let _maintenance = self.maintenance_gate.enter_shared();
        let aggregate = self.collect_blob_stats_silent()?;
        let bm: BufferStats = self.store.stats();
        let route = self.route_cache.stats();
        let route_cache = RouteCacheStats {
            entries: route.entries,
            hits: route.hits,
            misses: route.misses,
            learns: route.learns,
            evictions: route.evictions,
            invalidations: route.invalidations,
        };
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
            bm_dirty_count: bm.dirty_count,
            bm_pending_delete_count: bm.pending_delete_count,
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
            route_cache,
            open: self.open_stats,
            journal: self.journal_stats(),
            checkpointer: self.checkpointer_stats(),
        })
    }

    fn journal_stats(&self) -> Option<JournalStats> {
        self.journal.as_ref().map(|j| {
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
        })
    }

    fn checkpointer_stats(&self) -> Option<CheckpointerStats> {
        self.checkpointer.as_ref().map(|ck| CheckpointerStats {
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
        self.ensure_live()?;
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

    /// Whether a maintenance pass should rebuild the blob `guid` whose
    /// current cached image is `frame`. Two independent reasons:
    ///
    /// - **churn** — [`engine::blob_needs_compaction`]: tombstones or
    ///   dead-byte waste to reclaim. Unconditional; compaction already
    ///   serialises under the mutation gate.
    /// - **routing** — the blob is not yet routed but
    ///   [`engine::blob_would_route`] (a write-cold, bulk-loaded blob
    ///   that never accrued churn), AND it has settled write-cold
    ///   (`!has_unflushed_blob`). Without this, a write-once dataset
    ///   stays legacy forever and every cold point read pins a full
    ///   512 KB frame. The write-cold gate is a pure efficiency
    ///   heuristic — it avoids routing a blob mid-write that the next
    ///   structural mutation would just de-route; correctness never
    ///   depends on it (a stale route is de-routed by `alloc_node`, and
    ///   a cold routed read falls back to a full pin on any mismatch).
    fn blob_wants_compaction(&self, guid: BlobGuid, frame: BlobFrameRef<'_>) -> bool {
        engine::blob_needs_compaction(frame)
            || (engine::blob_would_route(frame) && !self.store.has_unflushed_blob(guid))
    }

    fn seed_maintenance_candidates(&self) -> Result<()> {
        let _maintenance = self.maintenance_gate.enter_shared();
        let _tree_mutation = self.mutation_gate.enter_shared();
        let guids = engine::collect_blob_guids(&self.store, self.root_guid)?;
        for guid in guids {
            let pin = self.store.pin(guid)?;
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let header = frame.header();
            if self.blob_wants_compaction(guid, frame) {
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
        let _tree_mutation = self.mutation_gate.enter_exclusive();
        if !self.store.has_blob(guid)? {
            return Ok(false);
        }
        let pin = self.store.pin(guid)?;
        let needs_compaction = {
            let guard = pin.read();
            self.blob_wants_compaction(guid, BlobFrameRef::wrap(guard.as_slice()))
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
                self.blob_wants_compaction(guid, frame.as_ref())
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
        let _tree_mutation = self.mutation_gate.enter_exclusive();
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
pub(crate) fn ensure_root_blob(bm: &Arc<BufferManager>, root_guid: BlobGuid) -> Result<()> {
    if !bm.has_blob(root_guid)? {
        let mut scratch = bm.alloc_blob_buf_zeroed();
        BlobFrame::init(scratch.as_mut_slice(), root_guid)?;
        bm.write_blob(root_guid, &scratch)?;
        bm.flush()?;
    }
    Ok(())
}

pub(crate) fn replay_wal<F>(
    path: &std::path::Path,
    bm: &Arc<BufferManager>,
    mut root_for_tree_id: F,
) -> Result<(u64, crate::journal::reader::ReplayStats)>
where
    F: FnMut(u64) -> Result<BlobGuid>,
{
    let mut root_pins: HashMap<u64, (BlobGuid, Arc<CachedBlob>)> = HashMap::new();
    let (_header, stats) = replay(path, |op, seq, _off| {
        let tree_id = op.tree_id().unwrap_or(0);
        let (root_guid, root_pin) = match root_pins.entry(tree_id) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let (guid, pin) = entry.get();
                (*guid, Arc::clone(pin))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let guid = root_for_tree_id(tree_id)?;
                ensure_root_blob(bm, guid)?;
                let pin = bm.pin(guid)?;
                let (guid, pin) = entry.insert((guid, pin));
                (*guid, Arc::clone(pin))
            }
        };
        // `root_dirty` tracks whether this op actually mutated
        // the BM-cached root image. No-op replays (e.g. an erase
        // for a key already absent because a prior replay pass
        // reconciled it) leave the cache byte-identical to
        // store — skipping `mark_dirty` for those is a small
        // win and matches `Tree::delete`'s same-shape branch.
        let root_dirty = match op {
            WalOp::Insert { key, value, .. } => {
                let search = engine::SearchKey::user(key);
                engine::insert_multi(bm, &root_pin, None, search, value, seq)?.root_dirty
            }
            WalOp::Erase { key, .. } => {
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
    // Physically drop a torn tail record before the writer reopens with
    // O_APPEND. Otherwise the next session appends a good record *after*
    // the torn bytes, turning them into a MID-log torn record; a later
    // replay's torn-tail `break` stops there and silently discards every
    // acked record written after it. The torn record was never acked (the
    // crash hit mid-write, before its fdatasync), so truncating it to the
    // last complete record loses nothing durable — standard WAL recovery.
    if let Some(off) = stats.torn_tail_at {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)?
            .set_len(off)?;
    }
    // After commit, the blob image is durable; we still want the
    // next allocated seq to be strictly greater than anything
    // ever seen in the log.
    Ok((stats.highest_seq.unwrap_or(0) + 1, stats))
}

#[cfg(test)]
mod tests {
    use crate::concurrency::CommitGate;
    use crate::journal::Journal;
    use crate::layout::BlobGuid;
    use crate::store::blob_store::{AlignedBlobBuf, BlobStore, MemoryBlobStore};
    use crate::store::BufferManager;
    use crate::TreeBuilder;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::sync_channel;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    struct FlushPendingStore {
        inner: MemoryBlobStore,
        pending: AtomicBool,
    }

    impl FlushPendingStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                pending: AtomicBool::new(true),
            }
        }

        fn release_flush(&self) {
            self.pending.store(false, Ordering::Release);
        }
    }

    impl BlobStore for FlushPendingStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> crate::Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> crate::Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn delete_blob(&self, guid: BlobGuid) -> crate::Result<()> {
            self.inner.delete_blob(guid)
        }

        fn list_blobs(&self) -> crate::Result<Vec<BlobGuid>> {
            self.inner.list_blobs()
        }

        fn flush(&self) -> crate::Result<()> {
            self.inner.flush()
        }

        fn needs_flush(&self) -> bool {
            self.pending.load(Ordering::Acquire)
        }
    }

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
    fn point_get_does_not_enter_maintenance_gate() {
        let tree = TreeBuilder::new("ignored").memory().open().unwrap();
        tree.put(b"hot/key", b"value").unwrap();

        let exclusive = tree.maintenance_gate.enter_exclusive();
        let worker_tree = tree.clone();
        let (done_tx, done_rx) = sync_channel(0);
        let handle = thread::spawn(move || {
            let got = worker_tree.get(b"hot/key").unwrap();
            done_tx.send(got).unwrap();
        });

        let got = done_rx.recv_timeout(Duration::from_secs(1));
        drop(exclusive);
        handle.join().unwrap();
        assert_eq!(got.unwrap().as_deref(), Some(&b"value"[..]));
    }

    #[test]
    fn single_key_writes_wait_behind_maintenance_exclusive() {
        let tree = TreeBuilder::new("ignored").memory().open().unwrap();
        tree.put(b"src", b"old").unwrap();

        let exclusive = tree.maintenance_gate.enter_exclusive();
        let worker_tree = tree.clone();
        let (done_tx, done_rx) = sync_channel(0);
        let handle = thread::spawn(move || {
            worker_tree.put(b"k1", b"v1").unwrap();
            assert!(worker_tree.delete(b"src").unwrap());
            worker_tree.put(b"rename-src", b"v2").unwrap();
            worker_tree
                .rename(b"rename-src", b"rename-dst", false)
                .unwrap();
            let k1 = worker_tree.get(b"k1").unwrap();
            let renamed = worker_tree.get(b"rename-dst").unwrap();
            done_tx.send((k1, renamed)).unwrap();
        });

        let got = done_rx.recv_timeout(Duration::from_secs(1));
        assert!(
            got.is_err(),
            "single-key writers must wait behind an exclusive mutation gate"
        );
        drop(exclusive);
        let got = done_rx.recv_timeout(Duration::from_secs(1));
        handle.join().unwrap();
        let (k1, renamed) = got.unwrap();
        assert_eq!(k1.as_deref(), Some(&b"v1"[..]));
        assert_eq!(renamed.as_deref(), Some(&b"v2"[..]));
    }

    #[test]
    fn single_key_writes_take_endpoint_shard() {
        let tree = TreeBuilder::new("ignored").memory().open().unwrap();
        let endpoint = tree.endpoint_locks.lock_key(b"same/key");
        let worker_tree = tree.clone();
        let (done_tx, done_rx) = sync_channel(0);
        let handle = thread::spawn(move || {
            worker_tree.put(b"same/key", b"value").unwrap();
            done_tx.send(()).unwrap();
        });

        assert!(
            done_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "put must wait behind the key endpoint shard"
        );
        drop(endpoint);
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        handle.join().unwrap();
        assert_eq!(
            tree.get(b"same/key").unwrap().as_deref(),
            Some(&b"value"[..])
        );
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

    #[test]
    fn synchronous_checkpoint_truncate_waits_for_store_flush() {
        let store = Arc::new(FlushPendingStore::new());
        let bm = Arc::new(BufferManager::new(store.clone(), 8));
        let journal_dir = tempfile::tempdir().unwrap();
        let journal =
            Arc::new(Journal::open_or_create(&journal_dir.path().join("wal.log"), 0).unwrap());
        journal.submit(vec![1, 2, 3, 4], false).unwrap();
        assert!(journal.needs_checkpoint());

        let commit_gate = Arc::new(CommitGate::new());
        super::Tree::maybe_truncate_journal_shared(&bm, Some(&journal), &commit_gate).unwrap();
        assert!(
            journal.needs_checkpoint(),
            "WAL must stay live while the store still owes a flush"
        );

        store.release_flush();
        super::Tree::maybe_truncate_journal_shared(&bm, Some(&journal), &commit_gate).unwrap();
        assert!(
            !journal.needs_checkpoint(),
            "WAL should truncate once BM state is clean and store is durable"
        );
    }
}
