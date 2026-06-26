//! Stateful range iterator — walk leaves in lex key order across
//! blobs with marker-aware lower-bound seek, `prefix` filtering,
//! and S3-style `delimiter` rollup.
//!
//! ## Concurrency
//!
//! The cursor is restart-on-conflict. Each stack frame records the
//! blob content version observed while the frame was pushed. Before
//! using a frame — and again before emitting a leaf or
//! `CommonPrefix` — the iterator verifies those versions. If an
//! interleaved writer split, merged, compacted, or otherwise rewrote
//! any blob on the cursor path, the stack is discarded and the walk
//! seeks from the last emitted lower bound. Callers never see an
//! unsafe "invalid iterator" state.
//!
//! This is not MVCC: a long iterator may observe keys committed
//! after it was created if they sort after the current cursor. The
//! guarantee is that iteration never knowingly continues through a
//! stale `(blob_guid, slot)` path and does not re-emit keys or
//! rollups it has already returned.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::api::atomic::RecordVersion;
use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::concurrency::Gate;
use crate::layout::{BlobGuid, BlobNode, Leaf, NodeType, BLOB_MAX_INLINE, PREFIX_MAX_INLINE};
use crate::store::{decode_child_off, BlobFrameRef, BufferManager, CachedBlob};

use smallvec::SmallVec;

use super::cast;
use super::readers::{
    child_offset, leaf_any_key, leaf_extent, leaf_key_extent, ntype_of, read_node16, read_node256,
    read_node4, read_node48, read_prefix, resolve_typed,
};
use crate::engine::simd;

type KeyBuf = SmallVec<[u8; 64]>;

/// An entry yielded by [`RangeIter`].
///
/// `#[non_exhaustive]` so adding new emission types (e.g., a
/// future tombstone-marker variant) is a non-breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RangeEntry {
    /// A leaf — user key + value + live record version (engine
    /// terminator already stripped).
    Key {
        /// User-supplied key bytes (terminator byte stripped).
        key: Vec<u8>,
        /// Value bytes.
        value: Vec<u8>,
        /// Current compare-and-set token for this live leaf.
        version: RecordVersion,
    },
    /// S3-style rollup — a common prefix collapsed because the
    /// caller set a [`RangeBuilder::delimiter`] and the iterator
    /// crossed it within a leaf key. The byte string includes the
    /// delimiter byte (`b"img/subfolder/"` for `prefix=b"img/"`
    /// and `delimiter=b'/'`).
    CommonPrefix(Vec<u8>),
}

/// An entry yielded by [`KeyRangeIter`].
///
/// This is the key-only companion to [`RangeEntry`]. It uses the
/// same cursor, prefix, marker, delimiter, and restart semantics as
/// [`RangeIter`], but it does not materialise value bytes for leaf
/// entries.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum KeyRangeEntry {
    /// A leaf — user key + live record version (engine terminator
    /// already stripped).
    Key {
        /// User-supplied key bytes (terminator byte stripped).
        key: Vec<u8>,
        /// Current compare-and-set token for this live leaf.
        version: RecordVersion,
    },
    /// S3-style rollup — a common prefix collapsed because the
    /// caller set a [`KeyRangeBuilder::delimiter`] and the iterator
    /// crossed it within a leaf key. The byte string includes the
    /// delimiter byte.
    CommonPrefix(Vec<u8>),
}

/// Borrowed key-only range entry passed to
/// [`KeyRangeBuilder::visit`].
///
/// The byte slices are valid only for the duration of the callback.
/// They point into the range cursor's reusable scratch buffer, not
/// into the underlying blob frame.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum KeyRangeEntryRef<'a> {
    /// A leaf under the requested key range.
    Key {
        /// User-supplied key bytes (terminator byte stripped).
        key: &'a [u8],
        /// Current compare-and-set token for this live leaf.
        version: RecordVersion,
    },
    /// Delimiter rollup prefix including the delimiter byte.
    CommonPrefix(&'a [u8]),
}

/// Per-scan work accounting for a single range cursor.
///
/// A scan can step over far more of the tree than it returns —
/// tombstoned leaves linger until compaction, delimiter rollups collapse
/// whole subtrees into one emission, and a concurrent writer can force
/// the cursor to restart. `ScanStats` exposes that gap so callers can
/// spot pathological listings (a `readdir` that walks 10k tombstones to
/// return 10 live names) and decide when to compact.
///
/// Read it from a [`RangeIter`] / [`KeyRangeIter`] after draining, or as
/// the return of [`KeyRangeBuilder::visit`]. Counters accumulate over the
/// scan's lifetime: `visited` measures work, the rest measure outcomes.
/// A cache-served prefix list reports `visited == 0` — it never walked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanStats {
    /// Stored entries the cursor examined: every leaf it read (including
    /// tombstoned or out-of-range ones skipped without emission) plus
    /// every delimiter fold. Always `>= returned + rollup`; the surplus
    /// is wasted work (a `readdir` that examined 10k tombstones to return
    /// 10 names reports `visited = 10_010`, `returned = 10`).
    pub visited: u64,
    /// Live leaves emitted (`Key` entries).
    pub returned: u64,
    /// Delimiter rollups emitted (`CommonPrefix` entries).
    pub rollup: u64,
    /// Cursor restarts forced by a concurrent writer rewriting a blob on
    /// the descent path — a read/write contention signal.
    pub restarts: u64,
}

/// Result of a borrowed key-only range visit.
///
/// This is the non-breaking companion to [`ScanStats`]: `ScanStats`
/// keeps its stable public field set, while `KeyScanOutcome` adds
/// execution metadata that callers use to tell whether a hot directory
/// listing came from the prefix-list cache or walked the ART.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyScanOutcome {
    /// Per-scan work accounting.
    pub stats: ScanStats,
    /// `true` when the result was served entirely from the prefix-list cache.
    pub cache_hit: bool,
}

/// Bounded count of live keys under a prefix.
///
/// `limit == 0` means unbounded / exact. For non-zero limits, `count`
/// is capped at `limit`; `exact == false` means at least one more key
/// exists beyond the reported count. `stats` and `cache_hit` describe
/// the underlying key-only scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCount {
    /// Number of matching live keys, capped by the requested limit.
    pub count: u64,
    /// `true` when `count` is the exact number of matching keys.
    pub exact: bool,
    /// Per-scan work accounting for the count operation.
    pub stats: ScanStats,
    /// `true` when the count was served entirely from the prefix-list cache.
    pub cache_hit: bool,
}

const PREFIX_LIST_CACHE_SLOTS: usize = 256;
const PREFIX_LIST_CACHE_MAX_LIMIT: usize = 256;

/// Direct-mapped cache for short hot prefix/delimiter scans.
#[derive(Debug)]
pub(crate) struct PrefixListCache {
    slots: Vec<Mutex<Option<PrefixListCacheEntry>>>,
}

#[derive(Debug)]
struct PrefixListCacheEntry {
    hash: u64,
    epoch: u64,
    delimiter: Option<u8>,
    limit: usize,
    prefix: Vec<u8>,
    start_after: Option<Vec<u8>>,
    entries: Arc<[CachedKeyRangeEntry]>,
}

#[derive(Debug, Clone)]
enum CachedKeyRangeEntry {
    Key {
        key: Vec<u8>,
        version: RecordVersion,
    },
    CommonPrefix(Vec<u8>),
}

impl CachedKeyRangeEntry {
    fn from_ref(entry: KeyRangeEntryRef<'_>) -> Self {
        match entry {
            KeyRangeEntryRef::Key { key, version } => Self::Key {
                key: key.to_vec(),
                version,
            },
            KeyRangeEntryRef::CommonPrefix(prefix) => Self::CommonPrefix(prefix.to_vec()),
        }
    }

    fn as_ref(&self) -> KeyRangeEntryRef<'_> {
        match self {
            Self::Key { key, version } => KeyRangeEntryRef::Key {
                key,
                version: *version,
            },
            Self::CommonPrefix(prefix) => KeyRangeEntryRef::CommonPrefix(prefix),
        }
    }
}

impl PrefixListCache {
    pub(crate) fn new() -> Self {
        Self {
            slots: (0..PREFIX_LIST_CACHE_SLOTS)
                .map(|_| Mutex::new(None))
                .collect(),
        }
    }

    fn visit<F>(
        &self,
        epoch: u64,
        prefix: &[u8],
        start_after: Option<&[u8]>,
        delimiter: Option<u8>,
        limit: usize,
        mut visitor: F,
    ) -> Result<Option<usize>>
    where
        F: FnMut(KeyRangeEntryRef<'_>) -> Result<()>,
    {
        if !Self::cacheable_limit(limit) {
            return Ok(None);
        }
        let hash = cache_hash(prefix, start_after, delimiter, limit);
        let entries = {
            let guard = self.slots[slot_index(hash)].lock().unwrap();
            let Some(entry) = guard.as_ref() else {
                return Ok(None);
            };
            if entry.hash != hash
                || entry.epoch != epoch
                || entry.delimiter != delimiter
                || entry.limit != limit
                || entry.prefix != prefix
                || entry.start_after.as_deref() != start_after
            {
                return Ok(None);
            }
            Arc::clone(&entry.entries)
        };
        for cached in entries.iter() {
            visitor(cached.as_ref())?;
        }
        Ok(Some(entries.len()))
    }

    fn store(
        &self,
        epoch: u64,
        prefix: &[u8],
        start_after: Option<&[u8]>,
        delimiter: Option<u8>,
        limit: usize,
        entries: Vec<CachedKeyRangeEntry>,
    ) {
        if !Self::cacheable_limit(limit) {
            return;
        }
        let hash = cache_hash(prefix, start_after, delimiter, limit);
        let mut guard = self.slots[slot_index(hash)].lock().unwrap();
        *guard = Some(PrefixListCacheEntry {
            hash,
            epoch,
            delimiter,
            limit,
            prefix: prefix.to_vec(),
            start_after: start_after.map(<[u8]>::to_vec),
            entries: entries.into(),
        });
    }

    fn cacheable_limit(limit: usize) -> bool {
        limit != 0 && limit <= PREFIX_LIST_CACHE_MAX_LIMIT
    }
}

fn cache_hash(
    prefix: &[u8],
    start_after: Option<&[u8]>,
    delimiter: Option<u8>,
    limit: usize,
) -> u64 {
    let mut h = DefaultHasher::new();
    prefix.hash(&mut h);
    start_after.hash(&mut h);
    delimiter.hash(&mut h);
    limit.hash(&mut h);
    h.finish()
}

fn slot_index(hash: u64) -> usize {
    (hash as usize) & (PREFIX_LIST_CACHE_SLOTS - 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangeProjection {
    Records,
    KeysOnly,
    KeyRefs,
}

enum ProjectedRangeEntry {
    Record(RangeEntry),
    Key(KeyRangeEntry),
}

impl ProjectedRangeEntry {
    fn into_record(self) -> RangeEntry {
        match self {
            Self::Record(entry) => entry,
            Self::Key(_) => unreachable!("key-only entry emitted from record range iterator"),
        }
    }

    fn into_key(self) -> KeyRangeEntry {
        match self {
            Self::Key(entry) => entry,
            Self::Record(_) => unreachable!("record entry emitted from key-only range iterator"),
        }
    }
}

/// Builder produced by [`crate::Tree::range`].
///
/// The builder is consumed by [`RangeBuilder::into_iter`] into a
/// [`RangeIter`] yielding [`RangeEntry`] items in lex order.
#[must_use = "RangeBuilder is lazy — call `.into_iter()` or use it in a `for` loop"]
pub struct RangeBuilder {
    bm: Arc<BufferManager>,
    root_pin: Arc<CachedBlob>,
    root_guid: BlobGuid,
    maintenance_gate: Arc<Gate>,
    mutation_gate: Option<Arc<Gate>>,
    dropped: Option<Arc<AtomicBool>>,
    preflight_error: Option<Error>,
    snapshot_cursor: bool,
    prefix: KeyBuf,
    start_after: Option<KeyBuf>,
    delimiter: Option<u8>,
}

impl RangeBuilder {
    /// Construct a builder anchored at `root_guid` of the BM-backed
    /// tree. Internal — user surface is [`crate::Tree::range`] /
    /// [`crate::Tree::scan`]; both signature dependencies
    /// (`BufferManager`, `BlobGuid`) live in crate-private modules.
    pub(crate) fn new(
        bm: Arc<BufferManager>,
        root_pin: Arc<CachedBlob>,
        root_guid: BlobGuid,
        maintenance_gate: Arc<Gate>,
    ) -> Self {
        Self {
            bm,
            root_pin,
            root_guid,
            maintenance_gate,
            mutation_gate: None,
            dropped: None,
            preflight_error: None,
            snapshot_cursor: false,
            prefix: KeyBuf::new(),
            start_after: None,
            delimiter: None,
        }
    }

    pub(crate) fn with_liveness(mut self, dropped: Arc<AtomicBool>) -> Self {
        self.dropped = Some(dropped);
        self
    }

    pub(crate) fn with_mutation_gate(mut self, mutation_gate: Arc<Gate>) -> Self {
        self.mutation_gate = Some(mutation_gate);
        self
    }

    pub(crate) fn with_preflight_error(mut self, error: Error) -> Self {
        self.preflight_error = Some(error);
        self
    }

    pub(crate) fn snapshot_cursor(mut self) -> Self {
        self.snapshot_cursor = true;
        self
    }

    fn ensure_live(&self) -> Result<()> {
        if self
            .dropped
            .as_ref()
            .is_some_and(|dropped| dropped.load(Ordering::Acquire))
        {
            Err(Error::TreeDropped)
        } else {
            Ok(())
        }
    }

    /// Restrict the scan to keys starting with `prefix`. Default:
    /// empty (the whole tree).
    pub fn prefix(mut self, prefix: &[u8]) -> Self {
        self.prefix.clear();
        self.prefix.extend_from_slice(prefix);
        self
    }

    /// Strict-greater-than lower bound. Default: none (start at
    /// the first matching leaf).
    pub fn start_after(mut self, key: &[u8]) -> Self {
        self.start_after = Some(KeyBuf::from_slice(key));
        self
    }

    /// S3-style delimiter byte. When set, leaves whose key (past
    /// `prefix`) contains the delimiter are folded into a single
    /// [`RangeEntry::CommonPrefix`] emission per distinct common
    /// prefix. Default: no delimiter (every leaf yielded as
    /// [`RangeEntry::Key`]).
    pub fn delimiter(mut self, byte: u8) -> Self {
        self.delimiter = Some(byte);
        self
    }
}

impl IntoIterator for RangeBuilder {
    type Item = Result<RangeEntry>;
    type IntoIter = RangeIter;

    fn into_iter(self) -> RangeIter {
        self.into_iter_with_projection(RangeProjection::Records)
    }
}

impl RangeBuilder {
    fn into_iter_with_projection(self, projection: RangeProjection) -> RangeIter {
        RangeIter {
            bm: self.bm,
            root_pin: self.root_pin,
            root_guid: self.root_guid,
            maintenance_gate: self.maintenance_gate,
            mutation_gate: self.mutation_gate,
            dropped: self.dropped,
            preflight_error: self.preflight_error,
            snapshot_cursor: self.snapshot_cursor,
            stack: Vec::with_capacity(8),
            curr_key: Vec::with_capacity(self.prefix.len().saturating_add(64)),
            emit_buf: Vec::with_capacity(self.prefix.len().saturating_add(64)),
            cursor_floor: 0,
            prefix: self.prefix.to_vec(),
            lower_bound: self
                .start_after
                .map(|bound| LowerBound::exclusive(bound.to_vec())),
            delimiter: self.delimiter,
            projection,
            initialized: false,
            terminated: false,
            stats: ScanStats::default(),
            prefetch: HashMap::new(),
        }
    }
}

/// Builder produced by [`crate::Tree::range_keys`].
///
/// It mirrors [`RangeBuilder`] but yields [`KeyRangeEntry`] items
/// and deliberately skips value materialisation.
#[must_use = "KeyRangeBuilder is lazy — call `.into_iter()` or use it in a `for` loop"]
pub struct KeyRangeBuilder {
    inner: RangeBuilder,
    prefix_list_cache: Option<Arc<PrefixListCache>>,
    epoch: Option<Arc<AtomicU64>>,
}

impl KeyRangeBuilder {
    /// Wrap a record range builder with key-only projection.
    pub(crate) fn new(inner: RangeBuilder) -> Self {
        Self {
            inner,
            prefix_list_cache: None,
            epoch: None,
        }
    }

    pub(crate) fn with_prefix_list_cache(
        mut self,
        cache: Arc<PrefixListCache>,
        epoch: Arc<AtomicU64>,
    ) -> Self {
        self.prefix_list_cache = Some(cache);
        self.epoch = Some(epoch);
        self
    }

    /// Restrict the scan to keys starting with `prefix`. Default:
    /// empty (the whole tree).
    pub fn prefix(mut self, prefix: &[u8]) -> Self {
        self.inner = self.inner.prefix(prefix);
        self
    }

    /// Strict-greater-than lower bound. Default: none (start at
    /// the first matching leaf).
    pub fn start_after(mut self, key: &[u8]) -> Self {
        self.inner = self.inner.start_after(key);
        self
    }

    /// S3-style delimiter byte. When set, leaves whose key (past
    /// `prefix`) contains the delimiter are folded into a single
    /// [`KeyRangeEntry::CommonPrefix`] emission per distinct
    /// common prefix.
    pub fn delimiter(mut self, byte: u8) -> Self {
        self.inner = self.inner.delimiter(byte);
        self
    }

    /// Visit key-only range entries with borrowed key bytes.
    ///
    /// This has the same ordering, prefix, start-after,
    /// delimiter-rollup, and restart semantics as [`KeyRangeIter`],
    /// but avoids allocating one `Vec<u8>` per emitted entry. The
    /// slices passed to `visitor` are valid only for the duration
    /// of that callback.
    pub fn visit<F>(self, limit: usize, visitor: F) -> Result<ScanStats>
    where
        F: FnMut(KeyRangeEntryRef<'_>) -> Result<()>,
    {
        self.visit_with_outcome(limit, visitor)
            .map(|outcome| outcome.stats)
    }

    /// Visit key-only range entries and return scan/cache outcome metadata.
    ///
    /// Prefer this over [`Self::visit`] when the caller wants to distinguish
    /// a real ART walk from a prefix-list cache hit.
    pub fn visit_with_outcome<F>(self, limit: usize, mut visitor: F) -> Result<KeyScanOutcome>
    where
        F: FnMut(KeyRangeEntryRef<'_>) -> Result<()>,
    {
        if limit == 0 {
            return Ok(KeyScanOutcome::default());
        }

        let mut builder = self;
        builder.inner.ensure_live()?;
        let prefix = builder.inner.prefix.as_slice();
        let start_after = builder.inner.start_after.as_deref();
        let delimiter = builder.inner.delimiter;

        if builder.inner.preflight_error.is_none() {
            if let (Some(cache), Some(epoch)) = (&builder.prefix_list_cache, &builder.epoch) {
                let current_epoch = epoch.load(Ordering::Acquire);
                // Count the served entries by kind in a wrapper so the cache
                // stays unaware of stats. On a hit `visited`/`restarts` stay
                // zero — nothing was walked.
                let mut stats = ScanStats::default();
                let hit = {
                    let mut counting = |entry: KeyRangeEntryRef<'_>| {
                        match entry {
                            KeyRangeEntryRef::Key { .. } => stats.returned += 1,
                            KeyRangeEntryRef::CommonPrefix(_) => stats.rollup += 1,
                        }
                        visitor(entry)
                    };
                    cache.visit(
                        current_epoch,
                        prefix,
                        start_after,
                        delimiter,
                        limit,
                        &mut counting,
                    )?
                };
                if hit.is_some() {
                    return Ok(KeyScanOutcome {
                        stats,
                        cache_hit: true,
                    });
                }
            }
        }

        let mut cached =
            if builder.prefix_list_cache.is_some() && PrefixListCache::cacheable_limit(limit) {
                Some(Vec::with_capacity(limit))
            } else {
                None
            };
        let cache_prefix = cached.as_ref().map(|_| builder.inner.prefix.clone());
        let cache_start_after = cached
            .as_ref()
            .and_then(|_| builder.inner.start_after.clone());
        let epoch_before = builder.epoch.as_ref().map(|e| e.load(Ordering::Acquire));
        let maintenance_gate = Arc::clone(&builder.inner.maintenance_gate);
        let mutation_gate = builder.inner.mutation_gate.clone();
        let _maintenance = maintenance_gate.enter_shared();
        let _tree_mutation = mutation_gate.as_ref().map(|gate| gate.enter_shared());
        let mut iter = KeyRangeIter {
            inner: builder
                .inner
                .into_iter_with_projection(RangeProjection::KeyRefs),
        };
        iter.visit_key_entries_unlocked(limit, |entry| {
            if let Some(cached) = cached.as_mut() {
                cached.push(CachedKeyRangeEntry::from_ref(entry));
            }
            visitor(entry)
        })?;
        if let (Some(cache), Some(epoch), Some(epoch_before), Some(cached)) = (
            builder.prefix_list_cache.take(),
            builder.epoch.take(),
            epoch_before,
            cached,
        ) {
            let epoch_after = epoch.load(Ordering::Acquire);
            if epoch_before == epoch_after {
                cache.store(
                    epoch_after,
                    cache_prefix
                        .as_deref()
                        .expect("cached entries require a prefix clone"),
                    cache_start_after.as_deref(),
                    delimiter,
                    limit,
                    cached,
                );
            }
        }
        Ok(KeyScanOutcome {
            stats: iter.stats(),
            cache_hit: false,
        })
    }
}

impl IntoIterator for KeyRangeBuilder {
    type Item = Result<KeyRangeEntry>;
    type IntoIter = KeyRangeIter;

    fn into_iter(self) -> KeyRangeIter {
        KeyRangeIter {
            inner: self
                .inner
                .into_iter_with_projection(RangeProjection::KeysOnly),
        }
    }
}

/// Active key-only iteration state — see
/// [`KeyRangeBuilder::into_iter`].
pub struct KeyRangeIter {
    inner: RangeIter,
}

impl Iterator for KeyRangeIter {
    type Item = Result<KeyRangeEntry>;

    fn next(&mut self) -> Option<Result<KeyRangeEntry>> {
        self.inner
            .next_projected_maybe_guarded(true)
            .map(|entry| entry.map(ProjectedRangeEntry::into_key))
    }
}

impl KeyRangeIter {
    /// Per-scan work accounting — see [`ScanStats`].
    pub fn stats(&self) -> ScanStats {
        self.inner.stats()
    }

    /// Advance without entering `maintenance_gate`.
    /// Caller must already hold the tree's maintenance guard.
    pub(crate) fn next_unlocked(&mut self) -> Option<Result<KeyRangeEntry>> {
        self.inner
            .next_projected_maybe_guarded(false)
            .map(|entry| entry.map(ProjectedRangeEntry::into_key))
    }

    /// Visit key-only entries without entering `maintenance_gate`.
    ///
    /// Caller must hold the tree's shared maintenance guard for the
    /// whole call. Entries borrow from the cursor's scratch buffer and
    /// are invalid after the callback returns.
    pub(crate) fn visit_key_entries_unlocked<F>(&mut self, limit: usize, visit: F) -> Result<usize>
    where
        F: FnMut(KeyRangeEntryRef<'_>) -> Result<()>,
    {
        self.inner.projection = RangeProjection::KeyRefs;
        self.inner.visit_key_entries_unlocked(limit, visit)
    }
}

/// Active iteration state — see [`RangeBuilder::into_iter`].
pub struct RangeIter {
    bm: Arc<BufferManager>,
    root_pin: Arc<CachedBlob>,
    root_guid: BlobGuid,
    maintenance_gate: Arc<Gate>,
    mutation_gate: Option<Arc<Gate>>,
    dropped: Option<Arc<AtomicBool>>,
    preflight_error: Option<Error>,
    snapshot_cursor: bool,
    /// Descent stack. Empty = no init done (if `!initialized`) or
    /// exhausted (if `terminated`).
    stack: Vec<Frame>,
    /// Bytes contributed to the current path by every live frame.
    /// On pop, the bytes the frame pushed are truncated.
    curr_key: Vec<u8>,
    /// Reusable output buffer for callback-based key listing.
    emit_buf: Vec<u8>,
    /// Depth of the root lower-bound cursor. The iterator stops
    /// once the cursor has exhausted the rooted search path.
    cursor_floor: usize,
    /// Prefix filter (raw user bytes; no engine terminator).
    prefix: Vec<u8>,
    /// Current restart lower bound. Starts as
    /// `RangeBuilder::start_after`; advances after every emitted
    /// key or delimiter rollup so a stale cursor can restart from a
    /// monotonic position.
    lower_bound: Option<LowerBound>,
    /// Delimiter byte applied to bytes past `prefix`.
    delimiter: Option<u8>,
    projection: RangeProjection,
    initialized: bool,
    terminated: bool,
    stats: ScanStats,
    /// Cold-scan I/O read-ahead window: child blobs pinned ahead of the
    /// descent reaching them, so a run of cross-blob children reads at
    /// device queue depth instead of one serial round-trip each. Filled
    /// from the (cached) parent when entering a run of `Blob` crossings
    /// and the window has drained; consumed (removed) at the pin point;
    /// cleared on restart. Holding the `Arc` keeps each blob resident
    /// until consumed. Correctness-neutral — a pre-pinned blob is the
    /// same blob, re-validated by the per-frame `version` check on use.
    prefetch: HashMap<BlobGuid, Arc<CachedBlob>>,
}

struct Frame {
    /// Pin keeps the blob in BM cache for the frame's lifetime.
    pin: Arc<CachedBlob>,
    blob_guid: BlobGuid,
    /// Byte offset of this frame's node body (v4 offset addressing).
    off: u32,
    ntype: NodeType,
    /// Blob content version captured while this frame was pushed.
    /// Any mismatch means a writer has rewritten this blob and the
    /// path must be rebuilt from the restart lower bound.
    version: u64,
    /// Cursor inside this frame. Semantics depend on `ntype`:
    /// - `Prefix` / `Blob`: `0` = "descend child", `1` = "done".
    /// - `Node4` / `Node16`: index into the sorted keys array.
    /// - `Node48` / `Node256`: next byte (0..=256, where 256 means
    ///   "no more children").
    /// - `Leaf`: `0` = "emit leaf", `1` = "done".
    /// - `EmptyRoot` / `Invalid`: always `0`, immediately popped.
    next: u16,
    /// Bytes this frame contributed to `curr_key` (branch byte for
    /// inner nodes, prefix bytes for `Prefix` / `Blob`). Truncated
    /// off `curr_key` when the frame is popped.
    pushed_bytes: u16,
}

#[derive(Clone, Copy)]
struct InlinePrefix {
    bytes: [u8; PREFIX_MAX_INLINE],
    len: u16,
}

impl InlinePrefix {
    #[inline]
    fn from_slice(src: &[u8]) -> Self {
        debug_assert!(src.len() <= PREFIX_MAX_INLINE);
        let mut bytes = [0; PREFIX_MAX_INLINE];
        bytes[..src.len()].copy_from_slice(src);
        Self {
            bytes,
            len: src.len() as u16,
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

fn project_range_leaf(
    frame: BlobFrameRef<'_>,
    off: u32,
    prefix: &[u8],
    lower_bound: Option<&LowerBound>,
    delimiter: Option<u8>,
    projection: RangeProjection,
    emit_buf: &mut Vec<u8>,
) -> Result<LeafAction> {
    let (_ntype, body) = resolve_typed(frame, off)?;
    // A leaf is one contiguous, self-describing node:
    // `[16B header][key][value]`. Decode the 16-byte header, then read
    // key (and, for record projections, value) from the same body.
    let leaf = *cast::<Leaf>(&body[..std::mem::size_of::<Leaf>()]);
    let (stored_key, record_value): (&[u8], Option<&[u8]>) = match projection {
        RangeProjection::Records => {
            let (k, v) = leaf_extent(frame, off)?;
            (k, Some(v))
        }
        RangeProjection::KeysOnly | RangeProjection::KeyRefs => {
            (leaf_key_extent(frame, off)?, None)
        }
    };
    let (tombstone, seq) = (leaf.tombstone, leaf.seq);
    if tombstone != 0 {
        return Ok(LeafAction::Skip);
    }
    let user_key = if stored_key.last() == Some(&0) {
        &stored_key[..stored_key.len() - 1]
    } else {
        stored_key
    };
    match prefix_filter_relation(user_key, prefix) {
        PrefixFilterRelation::Match => {}
        PrefixFilterRelation::Before => return Ok(LeafAction::Skip),
        PrefixFilterRelation::Past => return Ok(LeafAction::Done),
    }
    if let Some(bound) = lower_bound {
        if !bound.allows(user_key) {
            return Ok(LeafAction::Skip);
        }
    }
    if let Some(d) = delimiter {
        let rest = &user_key[prefix.len()..];
        if let Some(idx) = simd::find_byte(rest, d, 0) {
            if matches!(projection, RangeProjection::KeyRefs) {
                emit_buf.clear();
                emit_buf.extend_from_slice(&user_key[..=prefix.len() + idx]);
                return Ok(LeafAction::KeyRefCommonPrefix);
            }
            let common: Vec<u8> = user_key[..=prefix.len() + idx].to_vec();
            return Ok(LeafAction::CommonPrefix(common));
        }
    }
    if matches!(projection, RangeProjection::KeyRefs) {
        emit_buf.clear();
        emit_buf.extend_from_slice(user_key);
        return Ok(LeafAction::KeyRef {
            version: RecordVersion::new(seq),
        });
    }
    let key = user_key.to_vec();
    let version = RecordVersion::new(seq);
    Ok(LeafAction::Key {
        key,
        value: record_value.map(<[u8]>::to_vec),
        version,
    })
}

fn key_at_or_past_prefix_successor(key: &[u8], prefix: &[u8]) -> bool {
    let Some(pos) = prefix.iter().rposition(|&b| b != u8::MAX) else {
        return false;
    };
    let successor_len = pos + 1;
    let limit = key.len().min(successor_len);
    for i in 0..limit {
        let successor_byte = if i == pos { prefix[i] + 1 } else { prefix[i] };
        if key[i] != successor_byte {
            return key[i] > successor_byte;
        }
    }
    key.len() >= successor_len
}

fn concat_starts_with(left: &[u8], right: &[u8], prefix: &[u8]) -> bool {
    if left.len().saturating_add(right.len()) < prefix.len() {
        return false;
    }
    let mut i = 0usize;
    while i < prefix.len() {
        if concat_byte(left, right, i) != prefix[i] {
            return false;
        }
        i += 1;
    }
    true
}

fn concat_byte(left: &[u8], right: &[u8], idx: usize) -> u8 {
    if idx < left.len() {
        left[idx]
    } else {
        right[idx - left.len()]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LowerBound {
    key: Vec<u8>,
    inclusive: bool,
}

impl LowerBound {
    fn exclusive(key: Vec<u8>) -> Self {
        Self {
            key,
            inclusive: false,
        }
    }

    #[inline]
    fn key(&self) -> &[u8] {
        &self.key
    }

    #[inline]
    fn allows(&self, key: &[u8]) -> bool {
        if self.inclusive {
            key >= self.key.as_slice()
        } else {
            key > self.key.as_slice()
        }
    }

    fn set_exclusive(&mut self, key: &[u8]) {
        self.key.clear();
        self.key.extend_from_slice(key);
        self.inclusive = false;
    }

    fn set_inclusive_successor(&mut self, key: &[u8]) -> bool {
        self.key.clear();
        self.key.extend_from_slice(key);
        for i in (0..self.key.len()).rev() {
            if self.key[i] != u8::MAX {
                self.key[i] += 1;
                self.key.truncate(i + 1);
                self.inclusive = true;
                return true;
            }
        }
        false
    }
}

enum InitResult {
    Ready,
    Empty,
    Restart,
}

enum RangeAdvance {
    Entry(ProjectedRangeEntry),
    KeyRef(KeyRefKind),
    Done,
    Restart,
}

enum LeafAction {
    Skip,
    Done,
    Key {
        key: Vec<u8>,
        value: Option<Vec<u8>>,
        version: RecordVersion,
    },
    CommonPrefix(Vec<u8>),
    KeyRef {
        version: RecordVersion,
    },
    KeyRefCommonPrefix,
}

#[derive(Clone, Copy)]
enum KeyRefKind {
    Key { version: RecordVersion },
    CommonPrefix,
}

#[derive(Clone, Copy)]
enum SeekStart {
    None,
    Empty,
    Prefix,
    LowerBound,
}

enum SeekRelation {
    Seeking,
    Min,
    Skip,
}

enum SegmentRelation {
    Continue,
    Min,
    Skip,
}

enum PrefixFilterRelation {
    Match,
    Before,
    Past,
}

impl Iterator for RangeIter {
    type Item = Result<RangeEntry>;

    fn next(&mut self) -> Option<Result<RangeEntry>> {
        self.next_projected_maybe_guarded(true)
            .map(|entry| entry.map(ProjectedRangeEntry::into_record))
    }
}

impl RangeIter {
    /// Work accounting accumulated by this cursor so far. Read it after
    /// draining — or partway through — to see how much of the tree the
    /// scan stepped over versus what it returned. See [`ScanStats`].
    pub fn stats(&self) -> ScanStats {
        self.stats
    }

    fn next_projected_maybe_guarded(
        &mut self,
        enter_gate: bool,
    ) -> Option<Result<ProjectedRangeEntry>> {
        loop {
            if self.terminated {
                return None;
            }
            let step = if enter_gate {
                let maintenance_gate = Arc::clone(&self.maintenance_gate);
                let mutation_gate = self.mutation_gate.clone();
                let _maintenance = maintenance_gate.enter_shared();
                let _tree_mutation = mutation_gate.as_ref().map(|gate| gate.enter_shared());
                self.ensure_live().and_then(|()| self.next_step())
            } else {
                self.ensure_live().and_then(|()| self.next_step())
            };
            match step {
                Ok(RangeAdvance::Done) => {
                    self.terminated = true;
                    return None;
                }
                Ok(RangeAdvance::Restart) => {
                    self.restart_cursor();
                }
                Ok(RangeAdvance::Entry(entry)) => return Some(Ok(entry)),
                Ok(RangeAdvance::KeyRef(_)) => {
                    unreachable!("borrowed key entry emitted from public range iterator")
                }
                Err(e) => {
                    self.terminated = true;
                    return Some(Err(e));
                }
            }
        }
    }

    fn next_step(&mut self) -> Result<RangeAdvance> {
        if let Some(error) = self.preflight_error.take() {
            return Err(error);
        }
        if !self.initialized {
            match self.init_descent()? {
                InitResult::Ready => {
                    self.initialized = true;
                }
                InitResult::Empty => return Ok(RangeAdvance::Done),
                InitResult::Restart => return Ok(RangeAdvance::Restart),
            }
        }
        self.advance_to_next_entry()
    }

    fn ensure_live(&self) -> Result<()> {
        if self
            .dropped
            .as_ref()
            .is_some_and(|dropped| dropped.load(Ordering::Acquire))
        {
            Err(Error::TreeDropped)
        } else {
            Ok(())
        }
    }

    fn visit_key_entries_unlocked<F>(&mut self, limit: usize, mut visit: F) -> Result<usize>
    where
        F: FnMut(KeyRangeEntryRef<'_>) -> Result<()>,
    {
        let mut emitted = 0usize;
        while emitted < limit {
            let step = loop {
                if self.terminated {
                    return Ok(emitted);
                }
                match self.next_step()? {
                    RangeAdvance::Restart => self.restart_cursor(),
                    other => break other,
                }
            };
            match step {
                RangeAdvance::Done => {
                    self.terminated = true;
                    return Ok(emitted);
                }
                RangeAdvance::KeyRef(KeyRefKind::Key { version }) => {
                    visit(KeyRangeEntryRef::Key {
                        key: &self.emit_buf,
                        version,
                    })?;
                    emitted += 1;
                }
                RangeAdvance::KeyRef(KeyRefKind::CommonPrefix) => {
                    visit(KeyRangeEntryRef::CommonPrefix(&self.emit_buf))?;
                    emitted += 1;
                }
                RangeAdvance::Entry(_) => {
                    unreachable!("owned entry emitted from borrowed key projection")
                }
                RangeAdvance::Restart => unreachable!("restart handled by inner loop"),
            }
        }
        Ok(emitted)
    }

    fn init_descent(&mut self) -> Result<InitResult> {
        let seek_start = self.effective_seek_start();
        if matches!(seek_start, SeekStart::Empty) {
            return Ok(InitResult::Empty);
        }

        // Seed the stack with the root blob's root slot.
        let root_pin = Arc::clone(&self.root_pin);
        let (root_off, root_ntype, root_version) = {
            let guard = root_pin.read();
            let version = root_pin.content_version();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let off = decode_child_off(frame.header().root_slot);
            (off, ntype_of(frame, off)?, version)
        };
        self.stack.push(Frame {
            pin: root_pin,
            blob_guid: self.root_guid,
            off: root_off,
            ntype: root_ntype,
            version: root_version,
            next: 0,
            pushed_bytes: 0,
        });

        // Full-tree lower-bound cursor. Prefix filtering happens at
        // the leaf boundary and stops at the first key beyond the
        // prefix range, so restarts can jump straight to the last
        // emitted marker instead of re-walking the prefix subtree.
        self.cursor_floor = self.stack.len();
        match seek_start {
            SeekStart::None => Ok(InitResult::Ready),
            SeekStart::Empty => unreachable!("handled before root pin"),
            SeekStart::Prefix | SeekStart::LowerBound => self.seek_to_lower_bound(seek_start),
        }
    }

    fn effective_seek_start(&self) -> SeekStart {
        let Some(bound) = self.lower_bound.as_ref() else {
            if self.prefix.is_empty() {
                return SeekStart::None;
            }
            return SeekStart::Prefix;
        };
        let bound_key = bound.key();
        if self.prefix.is_empty() {
            return SeekStart::LowerBound;
        }
        if bound_key < self.prefix.as_slice() {
            return SeekStart::Prefix;
        }
        if key_at_or_past_prefix_successor(bound_key, &self.prefix) {
            return SeekStart::Empty;
        }
        SeekStart::LowerBound
    }

    fn seek_target(&self, source: SeekStart) -> &[u8] {
        match source {
            SeekStart::Prefix => self.prefix.as_slice(),
            SeekStart::LowerBound => self
                .lower_bound
                .as_ref()
                .expect("lower-bound seek source has a lower bound")
                .key(),
            SeekStart::None | SeekStart::Empty => {
                unreachable!("non-key seek source has no target bytes")
            }
        }
    }

    fn set_lower_bound_exclusive(&mut self, key: &[u8]) {
        match self.lower_bound.as_mut() {
            Some(bound) => bound.set_exclusive(key),
            None => self.lower_bound = Some(LowerBound::exclusive(key.to_vec())),
        }
    }

    fn set_lower_bound_to_emit_exclusive(&mut self) {
        match self.lower_bound.as_mut() {
            Some(bound) => bound.set_exclusive(&self.emit_buf),
            None => self.lower_bound = Some(LowerBound::exclusive(self.emit_buf.clone())),
        }
    }

    fn set_lower_bound_to_emit_successor(&mut self) -> bool {
        if let Some(bound) = self.lower_bound.as_mut() {
            bound.set_inclusive_successor(&self.emit_buf)
        } else {
            let mut bound = LowerBound::exclusive(Vec::new());
            let ok = bound.set_inclusive_successor(&self.emit_buf);
            if ok {
                self.lower_bound = Some(bound);
            }
            ok
        }
    }

    fn set_lower_bound_successor(&mut self, key: &[u8]) -> bool {
        if let Some(bound) = self.lower_bound.as_mut() {
            bound.set_inclusive_successor(key)
        } else {
            let mut bound = LowerBound::exclusive(Vec::new());
            let ok = bound.set_inclusive_successor(key);
            if ok {
                self.lower_bound = Some(bound);
            }
            ok
        }
    }

    fn common_prefix_advance_from_emit(&mut self) -> RangeAdvance {
        // A segment rollup folds a whole subtree at an inner node without
        // descending to its leaves — count it as one examined unit so the
        // `visited >= returned + rollup` invariant holds across both the
        // leaf-time and descent-time rollup paths.
        self.stats.visited += 1;
        self.stats.rollup += 1;
        match self.projection {
            RangeProjection::Records => RangeAdvance::Entry(ProjectedRangeEntry::Record(
                RangeEntry::CommonPrefix(self.emit_buf.clone()),
            )),
            RangeProjection::KeysOnly => RangeAdvance::Entry(ProjectedRangeEntry::Key(
                KeyRangeEntry::CommonPrefix(self.emit_buf.clone()),
            )),
            RangeProjection::KeyRefs => RangeAdvance::KeyRef(KeyRefKind::CommonPrefix),
        }
    }

    fn segment_has_rollup_candidate(&self, segment: &[u8]) -> bool {
        self.segment_rollup_len(segment).is_some()
    }

    fn prepare_segment_rollup(&mut self, segment: &[u8]) -> bool {
        let Some(common_len) = self.segment_rollup_len(segment) else {
            return false;
        };
        self.emit_buf.clear();
        if common_len <= self.curr_key.len() {
            self.emit_buf
                .extend_from_slice(&self.curr_key[..common_len]);
        } else {
            self.emit_buf.extend_from_slice(&self.curr_key);
            self.emit_buf
                .extend_from_slice(&segment[..common_len - self.curr_key.len()]);
        }
        self.lower_bound
            .as_ref()
            .is_none_or(|bound| bound.allows(&self.emit_buf))
    }

    fn prepare_branch_rollup(&mut self, byte: u8) -> bool {
        self.prepare_segment_rollup(std::slice::from_ref(&byte))
    }

    fn segment_rollup_len(&self, segment: &[u8]) -> Option<usize> {
        let delimiter = self.delimiter?;
        let total_len = self.curr_key.len().checked_add(segment.len())?;
        if total_len <= self.prefix.len() {
            return None;
        }
        if !concat_starts_with(&self.curr_key, segment, &self.prefix) {
            return None;
        }

        if self.prefix.len() < self.curr_key.len() {
            if let Some(pos) = simd::find_byte(&self.curr_key, delimiter, self.prefix.len()) {
                return Some(pos + 1);
            }
        }

        let start_in_segment = self.prefix.len().saturating_sub(self.curr_key.len());
        simd::find_byte(segment, delimiter, start_in_segment)
            .map(|pos| self.curr_key.len() + pos + 1)
    }

    #[allow(clippy::too_many_lines)] // one cursor-state machine over every ART node kind
    fn seek_to_lower_bound(&mut self, source: SeekStart) -> Result<InitResult> {
        loop {
            if self.stack.len() < self.cursor_floor {
                self.stack.clear();
                return Ok(InitResult::Empty);
            }
            let Some(top) = self.stack.last() else {
                return Ok(InitResult::Empty);
            };
            let top_ntype = top.ntype;
            match top_ntype {
                NodeType::Leaf => {
                    let idx = self.stack.len() - 1;
                    if self.stack[idx].next == 0 {
                        let is_candidate = {
                            let top = &self.stack[idx];
                            let guard = top.pin.read();
                            if !self.frame_content_still_valid(top) {
                                return Ok(InitResult::Restart);
                            }
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let stored_key = leaf_any_key(frame, top.off)?;
                            let user_key: &[u8] = if stored_key.last() == Some(&0) {
                                &stored_key[..stored_key.len() - 1]
                            } else {
                                stored_key
                            };
                            user_key >= self.seek_target(source)
                        };
                        if is_candidate {
                            return Ok(InitResult::Ready);
                        }
                    }
                    self.pop_frame();
                }
                NodeType::EmptyRoot | NodeType::Invalid => {
                    self.pop_frame();
                }
                NodeType::Prefix => {
                    let idx = self.stack.len() - 1;
                    if self.stack[idx].next != 0 {
                        self.pop_frame();
                        continue;
                    }
                    let top_pin = self.stack[idx].pin.clone();
                    let top_blob_guid = self.stack[idx].blob_guid;
                    let (child_off, child_ntype, p_bytes, version) = {
                        let top = &self.stack[idx];
                        let guard = top_pin.read();
                        let version = top_pin.content_version();
                        if !self.frame_version_still_valid(top, version) {
                            return Ok(InitResult::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let p = read_prefix(frame, top.off)?;
                        let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
                        let child_off = child_offset(p.child as u16);
                        (
                            child_off,
                            ntype_of(frame, child_off)?,
                            InlinePrefix::from_slice(&p.bytes[..plen]),
                            version,
                        )
                    };
                    let relation = {
                        let target = self.seek_target(source);
                        segment_seek_relation(&self.curr_key, p_bytes.as_slice(), target)
                    };
                    match relation {
                        SegmentRelation::Skip => {
                            self.stack[idx].next = 1;
                            self.pop_frame();
                        }
                        SegmentRelation::Continue | SegmentRelation::Min => {
                            if self.segment_has_rollup_candidate(p_bytes.as_slice()) {
                                return Ok(InitResult::Ready);
                            }
                            self.stack[idx].next = 1;
                            self.push_within_blob(
                                top_pin,
                                top_blob_guid,
                                child_off,
                                child_ntype,
                                version,
                                p_bytes.as_slice(),
                            );
                        }
                    }
                }
                NodeType::Blob => {
                    let idx = self.stack.len() - 1;
                    if self.stack[idx].next != 0 {
                        self.pop_frame();
                        continue;
                    }
                    let (child_guid, p_bytes) = {
                        let top = &self.stack[idx];
                        let guard = top.pin.read();
                        let version = top.pin.content_version();
                        if !self.frame_version_still_valid(top, version) {
                            return Ok(InitResult::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let body = frame
                            .body_at_offset(top.off)
                            .ok_or(Error::node_corrupt("range::seek: BlobNode body resolution"))?;
                        let b = cast::<BlobNode>(body);
                        let plen = (b.prefix_len as usize).min(BLOB_MAX_INLINE);
                        (
                            b.child_blob_guid,
                            InlinePrefix::from_slice(&b.bytes[..plen]),
                        )
                    };
                    let relation = {
                        let target = self.seek_target(source);
                        segment_seek_relation(&self.curr_key, p_bytes.as_slice(), target)
                    };
                    match relation {
                        SegmentRelation::Skip => {
                            self.stack[idx].next = 1;
                            self.pop_frame();
                        }
                        SegmentRelation::Continue | SegmentRelation::Min => {
                            if self.segment_has_rollup_candidate(p_bytes.as_slice()) {
                                return Ok(InitResult::Ready);
                            }
                            self.stack[idx].next = 1;
                            if !self.push_in_other_blob(child_guid, p_bytes.as_slice())? {
                                self.pop_frame();
                            }
                        }
                    }
                }
                NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
                    let idx = self.stack.len() - 1;
                    let (relation, min_byte) = {
                        let target = self.seek_target(source);
                        let relation = path_seek_relation(&self.curr_key, target);
                        let min_byte = match relation {
                            SeekRelation::Seeking => Some(target[self.curr_key.len()]),
                            SeekRelation::Skip | SeekRelation::Min => None,
                        };
                        (relation, min_byte)
                    };
                    if matches!(relation, SeekRelation::Skip) {
                        self.pop_frame();
                        continue;
                    }
                    let top_pin = self.stack[idx].pin.clone();
                    let top_blob_guid = self.stack[idx].blob_guid;
                    let result = {
                        let top = &self.stack[idx];
                        let guard = top_pin.read();
                        let version = top_pin.content_version();
                        if !self.frame_version_still_valid(top, version) {
                            return Ok(InitResult::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let result =
                            next_inner_child_ge(frame, top.off, top_ntype, top.next, min_byte)?;
                        match result {
                            Some((byte, child_off, next_cursor)) => Some((
                                byte,
                                child_off,
                                ntype_of(frame, child_off)?,
                                next_cursor,
                                version,
                            )),
                            None => None,
                        }
                    };
                    match result {
                        None => self.pop_frame(),
                        Some((byte, child_off, child_ntype, next_cursor, version)) => {
                            if self.segment_has_rollup_candidate(std::slice::from_ref(&byte)) {
                                return Ok(InitResult::Ready);
                            }
                            self.stack[idx].next = next_cursor;
                            self.curr_key.push(byte);
                            self.stack.push(Frame {
                                pin: top_pin,
                                blob_guid: top_blob_guid,
                                off: child_off,
                                ntype: child_ntype,
                                version,
                                next: 0,
                                pushed_bytes: 1,
                            });
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)] // single match over six NodeType variants — splitting hides the loop shape
    fn advance_to_next_entry(&mut self) -> Result<RangeAdvance> {
        loop {
            // Cursor stop: dropping below the rooted cursor means
            // the walk is exhausted.
            if self.stack.len() < self.cursor_floor {
                return Ok(RangeAdvance::Done);
            }
            let Some(top) = self.stack.last() else {
                return Ok(RangeAdvance::Done);
            };
            let top_ntype = top.ntype;
            if !matches!(top_ntype, NodeType::EmptyRoot | NodeType::Invalid)
                && self.prepare_segment_rollup(&[])
            {
                let idx = self.stack.len() - 1;
                let no_tombstones = {
                    let top = &self.stack[idx];
                    let guard = top.pin.read();
                    let version = top.pin.content_version();
                    if !self.frame_version_still_valid(top, version) {
                        return Ok(RangeAdvance::Restart);
                    }
                    BlobFrameRef::wrap(guard.as_slice())
                        .header()
                        .tombstone_leaf_cnt
                        == 0
                };
                if no_tombstones {
                    if !self.path_is_still_valid() {
                        return Ok(RangeAdvance::Restart);
                    }
                    let common_len = self.emit_buf.len();
                    while self.curr_key.len() >= common_len && self.stack.len() > self.cursor_floor
                    {
                        self.pop_frame();
                    }
                    if !self.set_lower_bound_to_emit_successor() {
                        self.terminated = true;
                    }
                    if !self.terminated {
                        self.reseek_after_emit();
                    }
                    return Ok(self.common_prefix_advance_from_emit());
                }
            }
            match top_ntype {
                NodeType::Leaf => {
                    let idx = self.stack.len() - 1;
                    if self.stack[idx].next == 0 {
                        self.stack[idx].next = 1;
                        let kv = {
                            let top = &self.stack[idx];
                            let guard = top.pin.read();
                            if !self.frame_content_still_valid(top) {
                                return Ok(RangeAdvance::Restart);
                            }
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            // Soft-deleted leaves stay physically in
                            // the slot table (and their key/value
                            // extent bytes stay allocated) until
                            // `compact_blob` rebuilds the blob; range
                            // iteration must skip them so a leaf
                            // that was erased between snapshot and
                            // iteration isn't emitted.
                            project_range_leaf(
                                frame,
                                top.off,
                                &self.prefix,
                                self.lower_bound.as_ref(),
                                self.delimiter,
                                self.projection,
                                &mut self.emit_buf,
                            )?
                        };
                        self.stats.visited += 1;
                        match kv {
                            LeafAction::Skip => {}
                            LeafAction::Done => return Ok(RangeAdvance::Done),
                            LeafAction::Key {
                                key,
                                value,
                                version,
                            } => {
                                if !self.path_is_still_valid() {
                                    return Ok(RangeAdvance::Restart);
                                }
                                self.set_lower_bound_exclusive(&key);
                                self.stats.returned += 1;
                                let entry = match self.projection {
                                    RangeProjection::Records => {
                                        ProjectedRangeEntry::Record(RangeEntry::Key {
                                            key,
                                            value: value.expect("record projection carries value"),
                                            version,
                                        })
                                    }
                                    RangeProjection::KeysOnly => {
                                        ProjectedRangeEntry::Key(KeyRangeEntry::Key {
                                            key,
                                            version,
                                        })
                                    }
                                    RangeProjection::KeyRefs => {
                                        unreachable!(
                                            "borrowed key projection uses LeafAction::KeyRef"
                                        )
                                    }
                                };
                                return Ok(RangeAdvance::Entry(entry));
                            }
                            LeafAction::CommonPrefix(common) => {
                                if !self.path_is_still_valid() {
                                    return Ok(RangeAdvance::Restart);
                                }
                                // Fast-forward past `common`'s subtree.
                                // Ascend the descent stack while
                                // `curr_key` still extends into the
                                // rolled-up region; each pop trims its
                                // `pushed_bytes`. The top frame's cursor
                                // is already positioned past the byte
                                // that led into `common` (descend always
                                // advances the parent cursor before
                                // pushing a child), so the natural
                                // advance loop on the next `next()` call
                                // picks the next sibling and skips the
                                // whole subtree — `O(leaves_under_rollup)`
                                // dedup-scans collapse to `O(stack_pops)`.
                                let common_len = common.len();
                                while self.curr_key.len() >= common_len
                                    && self.stack.len() > self.cursor_floor
                                {
                                    self.pop_frame();
                                }
                                if !self.set_lower_bound_successor(&common) {
                                    self.terminated = true;
                                }
                                if !self.terminated {
                                    self.reseek_after_emit();
                                }
                                self.stats.rollup += 1;
                                let entry = match self.projection {
                                    RangeProjection::Records => ProjectedRangeEntry::Record(
                                        RangeEntry::CommonPrefix(common),
                                    ),
                                    RangeProjection::KeysOnly => ProjectedRangeEntry::Key(
                                        KeyRangeEntry::CommonPrefix(common),
                                    ),
                                    RangeProjection::KeyRefs => unreachable!(
                                        "borrowed key projection uses LeafAction::KeyRefCommonPrefix"
                                    ),
                                };
                                return Ok(RangeAdvance::Entry(entry));
                            }
                            LeafAction::KeyRef { version } => {
                                if !self.path_is_still_valid() {
                                    return Ok(RangeAdvance::Restart);
                                }
                                self.set_lower_bound_to_emit_exclusive();
                                self.stats.returned += 1;
                                return Ok(RangeAdvance::KeyRef(KeyRefKind::Key { version }));
                            }
                            LeafAction::KeyRefCommonPrefix => {
                                if !self.path_is_still_valid() {
                                    return Ok(RangeAdvance::Restart);
                                }
                                let common_len = self.emit_buf.len();
                                while self.curr_key.len() >= common_len
                                    && self.stack.len() > self.cursor_floor
                                {
                                    self.pop_frame();
                                }
                                if !self.set_lower_bound_to_emit_successor() {
                                    self.terminated = true;
                                }
                                if !self.terminated {
                                    self.reseek_after_emit();
                                }
                                self.stats.rollup += 1;
                                return Ok(RangeAdvance::KeyRef(KeyRefKind::CommonPrefix));
                            }
                        }
                        // Tombstoned — fall through to pop_frame and
                        // resume scanning.
                    }
                    self.pop_frame();
                }
                NodeType::EmptyRoot | NodeType::Invalid => {
                    self.pop_frame();
                }
                NodeType::Prefix => {
                    let idx = self.stack.len() - 1;
                    if self.stack[idx].next == 0 {
                        let top_pin = self.stack[idx].pin.clone();
                        let top_blob_guid = self.stack[idx].blob_guid;
                        let (child_off, child_ntype, p_bytes, version, no_tombstones) = {
                            let top = &self.stack[idx];
                            let guard = top_pin.read();
                            let version = top_pin.content_version();
                            if !self.frame_version_still_valid(top, version) {
                                return Ok(RangeAdvance::Restart);
                            }
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let p = read_prefix(frame, top.off)?;
                            let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
                            let child_off = child_offset(p.child as u16);
                            (
                                child_off,
                                ntype_of(frame, child_off)?,
                                InlinePrefix::from_slice(&p.bytes[..plen]),
                                version,
                                frame.header().tombstone_leaf_cnt == 0,
                            )
                        };
                        self.stack[idx].next = 1;
                        if no_tombstones
                            && !matches!(child_ntype, NodeType::Blob | NodeType::EmptyRoot)
                            && self.prepare_segment_rollup(p_bytes.as_slice())
                        {
                            if !self.path_is_still_valid() {
                                return Ok(RangeAdvance::Restart);
                            }
                            let needs_reseek = self.curr_key.len() < self.emit_buf.len();
                            if !self.set_lower_bound_to_emit_successor() {
                                self.terminated = true;
                            }
                            if needs_reseek && !self.terminated {
                                self.reseek_after_emit();
                            }
                            let entry = self.common_prefix_advance_from_emit();
                            return Ok(entry);
                        }
                        self.push_within_blob(
                            top_pin,
                            top_blob_guid,
                            child_off,
                            child_ntype,
                            version,
                            p_bytes.as_slice(),
                        );
                    } else {
                        self.pop_frame();
                    }
                }
                NodeType::Blob => {
                    let idx = self.stack.len() - 1;
                    if self.stack[idx].next == 0 {
                        let (child_guid, p_bytes) = {
                            let top = &self.stack[idx];
                            let guard = top.pin.read();
                            let version = top.pin.content_version();
                            if !self.frame_version_still_valid(top, version) {
                                return Ok(RangeAdvance::Restart);
                            }
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let body = frame.body_at_offset(top.off).ok_or(Error::node_corrupt(
                                "range::advance: BlobNode body resolution",
                            ))?;
                            let b = cast::<BlobNode>(body);
                            let plen = (b.prefix_len as usize).min(BLOB_MAX_INLINE);
                            (
                                b.child_blob_guid,
                                InlinePrefix::from_slice(&b.bytes[..plen]),
                            )
                        };
                        self.stack[idx].next = 1;
                        let Some(child_pin) = self.pin_scan_child(child_guid)? else {
                            self.pop_frame();
                            continue;
                        };
                        child_pin.prefetch_header();
                        let child_can_rollup = {
                            let guard = child_pin.read();
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let root_off = decode_child_off(frame.header().root_slot);
                            frame.header().tombstone_leaf_cnt == 0
                                && !matches!(
                                    ntype_of(frame, root_off)?,
                                    NodeType::EmptyRoot | NodeType::Invalid
                                )
                        };
                        if child_can_rollup && self.prepare_segment_rollup(p_bytes.as_slice()) {
                            if !self.path_is_still_valid() {
                                return Ok(RangeAdvance::Restart);
                            }
                            let needs_reseek = self.curr_key.len() < self.emit_buf.len();
                            if !self.set_lower_bound_to_emit_successor() {
                                self.terminated = true;
                            }
                            if needs_reseek && !self.terminated {
                                self.reseek_after_emit();
                            }
                            let entry = self.common_prefix_advance_from_emit();
                            return Ok(entry);
                        }
                        self.push_pinned_other_blob(child_pin, child_guid, p_bytes.as_slice())?;
                    } else {
                        self.pop_frame();
                    }
                }
                NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
                    let idx = self.stack.len() - 1;
                    let top_pin = self.stack[idx].pin.clone();
                    let top_blob_guid = self.stack[idx].blob_guid;
                    let result = {
                        let top = &self.stack[idx];
                        let guard = top_pin.read();
                        let version = top_pin.content_version();
                        if !self.frame_version_still_valid(top, version) {
                            return Ok(RangeAdvance::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let no_tombstones = frame.header().tombstone_leaf_cnt == 0;
                        let result = next_inner_child_from(frame, top.off, top_ntype, top.next)?;
                        match result {
                            Some((byte, child_off, next_cursor)) => {
                                // Scan-ahead prefetch: the child we
                                // descend into now emits a whole
                                // subtree of leaves before the iterator
                                // reaches the next sibling, so warm
                                // that sibling's body now — its load
                                // overlaps the entire current-subtree
                                // traversal. Best-effort: a peek error
                                // just skips the hint.
                                if let Some((_, sib_off, _)) =
                                    next_inner_child_from(frame, top.off, top_ntype, next_cursor)
                                        .ok()
                                        .flatten()
                                {
                                    frame.prefetch_at(sib_off);
                                }
                                let child_ntype = ntype_of(frame, child_off)?;
                                // Cold-scan I/O read-ahead: when entering a run
                                // of cross-blob children and the window has
                                // drained, collect the upcoming child-blob guids
                                // from this (cached) parent so they can be pinned
                                // concurrently (device queue depth) below.
                                let prefetch_guids = if !self.snapshot_cursor
                                    && child_ntype == NodeType::Blob
                                    && self.prefetch.is_empty()
                                {
                                    collect_blob_child_window(
                                        frame,
                                        top.off,
                                        top_ntype,
                                        child_off,
                                        next_cursor,
                                    )
                                } else {
                                    Vec::new()
                                };
                                Some((
                                    byte,
                                    child_off,
                                    child_ntype,
                                    next_cursor,
                                    version,
                                    no_tombstones,
                                    prefetch_guids,
                                ))
                            }
                            None => None,
                        }
                    };
                    match result {
                        None => self.pop_frame(),
                        Some((
                            byte,
                            child_off,
                            child_ntype,
                            next_cursor,
                            version,
                            no_tombstones,
                            prefetch_guids,
                        )) => {
                            if no_tombstones
                                && !matches!(child_ntype, NodeType::Blob | NodeType::EmptyRoot)
                                && self.prepare_branch_rollup(byte)
                            {
                                if !self.path_is_still_valid() {
                                    return Ok(RangeAdvance::Restart);
                                }
                                if !self.set_lower_bound_to_emit_successor() {
                                    self.terminated = true;
                                }
                                if !self.terminated {
                                    self.reseek_after_emit();
                                }
                                return Ok(self.common_prefix_advance_from_emit());
                            }
                            // Pin the read-ahead window concurrently (QD =
                            // window size); each held pin is consumed at the
                            // pin point. Best-effort — a failed prefetch just
                            // falls back to a normal pin when reached.
                            if !prefetch_guids.is_empty() {
                                let pins = self.bm.pin_scan_many(&prefetch_guids);
                                for (guid, pin) in prefetch_guids.iter().zip(pins) {
                                    if let Some(pin) = pin {
                                        self.prefetch.insert(*guid, pin);
                                    }
                                }
                            }
                            self.stack[idx].next = next_cursor;
                            self.curr_key.push(byte);
                            self.stack.push(Frame {
                                pin: top_pin,
                                blob_guid: top_blob_guid,
                                off: child_off,
                                ntype: child_ntype,
                                version,
                                next: 0,
                                pushed_bytes: 1,
                            });
                        }
                    }
                }
            }
        }
    }

    fn push_within_blob(
        &mut self,
        pin: Arc<CachedBlob>,
        blob_guid: BlobGuid,
        child_off: u32,
        child_ntype: NodeType,
        version: u64,
        prefix_bytes: &[u8],
    ) {
        self.curr_key.extend_from_slice(prefix_bytes);
        self.stack.push(Frame {
            pin,
            blob_guid,
            off: child_off,
            ntype: child_ntype,
            version,
            next: 0,
            pushed_bytes: prefix_bytes.len() as u16,
        });
    }

    fn push_in_other_blob(&mut self, child_guid: BlobGuid, prefix_bytes: &[u8]) -> Result<bool> {
        let Some(child_pin) = self.pin_scan_child(child_guid)? else {
            return Ok(false);
        };
        child_pin.prefetch_header();
        self.push_pinned_other_blob(child_pin, child_guid, prefix_bytes)?;
        Ok(true)
    }

    fn pin_scan_child(&mut self, child_guid: BlobGuid) -> Result<Option<Arc<CachedBlob>>> {
        // Consume a read-ahead pin if one was warmed for this child.
        if let Some(pin) = self.prefetch.remove(&child_guid) {
            return Ok(Some(pin));
        }
        match self.bm.pin_scan(child_guid) {
            Ok(pin) => Ok(Some(pin)),
            Err(e) if is_blob_store_not_found(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn push_pinned_other_blob(
        &mut self,
        child_pin: Arc<CachedBlob>,
        child_guid: BlobGuid,
        prefix_bytes: &[u8],
    ) -> Result<()> {
        let (child_off, child_ntype, child_version) = {
            let guard = child_pin.read();
            let version = child_pin.content_version();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let child_off = decode_child_off(frame.header().root_slot);
            (child_off, ntype_of(frame, child_off)?, version)
        };
        self.curr_key.extend_from_slice(prefix_bytes);
        self.stack.push(Frame {
            pin: child_pin,
            blob_guid: child_guid,
            off: child_off,
            ntype: child_ntype,
            version: child_version,
            next: 0,
            pushed_bytes: prefix_bytes.len() as u16,
        });
        Ok(())
    }

    fn path_is_still_valid(&self) -> bool {
        if self.snapshot_cursor {
            return true;
        }
        self.stack
            .iter()
            .all(|frame| frame.pin.validate_content_version(frame.version))
    }

    fn frame_content_still_valid(&self, frame: &Frame) -> bool {
        self.snapshot_cursor || frame.pin.content_version() == frame.version
    }

    fn frame_version_still_valid(&self, frame: &Frame, current_version: u64) -> bool {
        self.snapshot_cursor || current_version == frame.version
    }

    fn restart_cursor(&mut self) {
        self.bm.note_range_restart();
        self.stats.restarts += 1;
        self.stack.clear();
        self.curr_key.clear();
        self.cursor_floor = 0;
        self.initialized = false;
        // Drop any read-ahead pins from the abandoned path so the new
        // path refills its own window (and we don't hold stale pins).
        self.prefetch.clear();
    }

    fn reseek_after_emit(&mut self) {
        self.stack.clear();
        self.curr_key.clear();
        self.cursor_floor = 0;
        self.initialized = false;
        self.prefetch.clear();
    }

    fn pop_frame(&mut self) {
        let Some(f) = self.stack.pop() else { return };
        let new_len = self.curr_key.len().saturating_sub(f.pushed_bytes as usize);
        self.curr_key.truncate(new_len);
    }
}

fn path_seek_relation(path: &[u8], target: &[u8]) -> SeekRelation {
    let limit = path.len().min(target.len());
    let common = simd::longest_common_prefix(path, target);
    if common == path.len() && path.len() < target.len() {
        SeekRelation::Seeking
    } else if common == limit {
        if path.len() >= target.len() {
            SeekRelation::Min
        } else {
            SeekRelation::Skip
        }
    } else if path[common] >= target[common] {
        SeekRelation::Min
    } else {
        SeekRelation::Skip
    }
}

fn prefix_filter_relation(key: &[u8], prefix: &[u8]) -> PrefixFilterRelation {
    if prefix.is_empty() {
        return PrefixFilterRelation::Match;
    }
    let limit = key.len().min(prefix.len());
    let common = simd::longest_common_prefix(key, prefix);
    if common == prefix.len() {
        PrefixFilterRelation::Match
    } else if common == limit || key[common] < prefix[common] {
        PrefixFilterRelation::Before
    } else {
        PrefixFilterRelation::Past
    }
}

fn segment_seek_relation(path: &[u8], segment: &[u8], target: &[u8]) -> SegmentRelation {
    match path_seek_relation(path, target) {
        SeekRelation::Skip => SegmentRelation::Skip,
        SeekRelation::Min => SegmentRelation::Min,
        SeekRelation::Seeking => {
            let remaining = &target[path.len()..];
            let limit = segment.len().min(remaining.len());
            let common = simd::longest_common_prefix(segment, remaining);
            if common < limit {
                return match segment[common].cmp(&remaining[common]) {
                    std::cmp::Ordering::Less => SegmentRelation::Skip,
                    std::cmp::Ordering::Equal => unreachable!("lcp stopped on equal byte"),
                    std::cmp::Ordering::Greater => SegmentRelation::Min,
                };
            }
            if segment.len() < remaining.len() {
                SegmentRelation::Continue
            } else {
                SegmentRelation::Min
            }
        }
    }
}

/// Given the inner node at byte offset `off` and a cursor `from`,
/// return the next `(byte, child_off, cursor_after)` whose branch byte
/// is at least `min_byte` when present. `None` means "the minimum
/// child". `child_off` is the child body's absolute byte offset.
fn next_inner_child_ge(
    frame: BlobFrameRef<'_>,
    off: u32,
    ntype: NodeType,
    from: u16,
    min_byte: Option<u8>,
) -> Result<Option<(u8, u32, u16)>> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame, off)?;
            let count = (n.count as usize).min(4);
            let start = (from as usize).min(count);
            let min = min_byte.unwrap_or(0);
            for i in start..count {
                if n.keys[i] >= min {
                    return Ok(Some((
                        n.keys[i],
                        child_offset(n.children[i]),
                        (i + 1) as u16,
                    )));
                }
            }
            Ok(None)
        }
        NodeType::Node16 => {
            let n = read_node16(frame, off)?;
            let count = (n.count as usize).min(16);
            let start = (from as usize).min(count);
            let min = min_byte.unwrap_or(0);
            for i in start..count {
                if n.keys[i] >= min {
                    return Ok(Some((
                        n.keys[i],
                        child_offset(n.children[i]),
                        (i + 1) as u16,
                    )));
                }
            }
            Ok(None)
        }
        NodeType::Node48 => {
            let n = read_node48(frame, off)?;
            let min = min_byte.map_or(0, usize::from);
            let start = (from as usize).max(min).min(256);
            let Some(b) = simd::find_next_nonzero_byte(&n.index, start) else {
                return Ok(None);
            };
            let idx = n.index[b];
            let ci = idx as usize - 1;
            if ci >= 48 {
                return Err(Error::node_corrupt(
                    "range::next_inner_child_ge: Node48 index out of range",
                ));
            }
            Ok(Some((
                b as u8,
                child_offset(n.children[ci]),
                (b + 1) as u16,
            )))
        }
        NodeType::Node256 => {
            let n = read_node256(frame, off)?;
            let min = min_byte.map_or(0, usize::from);
            let start = (from as usize).max(min).min(256);
            let Some(b) = simd::find_next_nonzero_u16(&n.children, start) else {
                return Ok(None);
            };
            Ok(Some((b as u8, child_offset(n.children[b]), (b + 1) as u16)))
        }
        _ => Err(Error::node_corrupt(
            "range::next_inner_child_ge: not an inner node",
        )),
    }
}

/// Read the target `BlobGuid` of the `Blob` crossing at `off`, or
/// `None` if the node there is not a `Blob` (or is undecodable). Reads
/// only from the already-resident `frame` — no I/O.
fn blob_child_guid(frame: BlobFrameRef<'_>, off: u32) -> Option<BlobGuid> {
    if ntype_of(frame, off).ok()? != NodeType::Blob {
        return None;
    }
    let body = frame.body_at_offset(off)?;
    Some(cast::<BlobNode>(body).child_blob_guid)
}

/// Collect up to a bounded window of consecutive child-blob guids for a
/// cold-scan read-ahead: the current `Blob` child at `first_off` plus
/// following siblings of the parent inner node, stopping at the first
/// non-`Blob` sibling (the run of cross-blob children has ended). Pure
/// reads from the cached `frame`; the caller pins the returned guids
/// concurrently so their cold reads pipeline.
fn collect_blob_child_window(
    frame: BlobFrameRef<'_>,
    parent_off: u32,
    parent_ntype: NodeType,
    first_off: u32,
    mut cursor: u16,
) -> Vec<BlobGuid> {
    const DEPTH: usize = 8;
    let mut guids = Vec::with_capacity(DEPTH);
    if let Some(g) = blob_child_guid(frame, first_off) {
        guids.push(g);
    }
    while guids.len() < DEPTH {
        match next_inner_child_from(frame, parent_off, parent_ntype, cursor) {
            Ok(Some((_, sib_off, next))) => match blob_child_guid(frame, sib_off) {
                Some(g) => {
                    guids.push(g);
                    cursor = next;
                }
                None => break,
            },
            _ => break,
        }
    }
    guids
}

/// Given the inner node at byte offset `off` and a cursor `from`,
/// return the next `(byte, child_off, cursor_after)` if any. For
/// `Node4` / `Node16`, `from` is a key index; for `Node48` /
/// `Node256`, it's the next byte to consider (0..=256, where 256 means
/// "no more"). `child_off` is the child body's absolute byte offset.
fn next_inner_child_from(
    frame: BlobFrameRef<'_>,
    off: u32,
    ntype: NodeType,
    from: u16,
) -> Result<Option<(u8, u32, u16)>> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame, off)?;
            let count = (n.count as usize).min(4);
            let i = from as usize;
            if i >= count {
                return Ok(None);
            }
            Ok(Some((
                n.keys[i],
                child_offset(n.children[i]),
                (i + 1) as u16,
            )))
        }
        NodeType::Node16 => {
            let n = read_node16(frame, off)?;
            let count = (n.count as usize).min(16);
            let i = from as usize;
            if i >= count {
                return Ok(None);
            }
            Ok(Some((
                n.keys[i],
                child_offset(n.children[i]),
                (i + 1) as u16,
            )))
        }
        NodeType::Node48 => {
            let n = read_node48(frame, off)?;
            let start = (from as usize).min(256);
            // SIMD-scan the 256-byte index for the next non-zero
            // entry; saves ≈40 ns vs the scalar 256-iter loop on a
            // sparse Node48.
            let Some(b) = simd::find_next_nonzero_byte(&n.index, start) else {
                return Ok(None);
            };
            let idx = n.index[b];
            let ci = idx as usize - 1;
            if ci >= 48 {
                return Err(Error::node_corrupt(
                    "range::next_inner_child: Node48 index out of range",
                ));
            }
            Ok(Some((
                b as u8,
                child_offset(n.children[ci]),
                (b + 1) as u16,
            )))
        }
        NodeType::Node256 => {
            let n = read_node256(frame, off)?;
            let start = (from as usize).min(256);
            // SIMD-scan the 256-`u16` children array for the next
            // populated child.
            let Some(b) = simd::find_next_nonzero_u16(&n.children, start) else {
                return Ok(None);
            };
            Ok(Some((b as u8, child_offset(n.children[b]), (b + 1) as u16)))
        }
        _ => Err(Error::node_corrupt(
            "range::next_inner_child: not an inner node",
        )),
    }
}
