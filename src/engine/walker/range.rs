//! Stateful range iterator — walk leaves in lex key order across
//! blobs with marker-aware lower-bound seek, `prefix` filtering,
//! and S3-style `delimiter` rollup.
//!
//! Modelled on the upstream `fa_iter` shape extracted from the
//! original binary's log strings: `path` (parent-chain stack of
//! `(blob_guid, slot)`), `curr_key` (materialised current path
//! bytes), `marker` (exclusive lower bound), `delimiter` (single
//! byte that collapses sub-subtrees into a single `CommonPrefix`
//! emit), `start_index_in_node` (resume cursor inside `Node4/16/48/256`
//! to avoid re-scanning all children). Forward-only — no `findPrev`.
//!
//! ## Concurrency
//!
//! The cursor is restart-on-conflict. Each stack frame records the
//! blob content version observed while the frame was pushed. Before
//! using a frame — and again before emitting a leaf or
//! `CommonPrefix` — the iterator verifies those versions. If an
//! interleaved writer split, merged, compacted, or otherwise rewrote
//! any blob on the cursor path, the stack is discarded and the walk
//! seeks from the last emitted lower bound. This mirrors the
//! upstream `fa_iter` invalidation/restart shape without exposing an
//! unsafe "invalid iterator" state to callers.
//!
//! This is not MVCC: a long iterator may observe keys committed
//! after it was created if they sort after the current cursor. The
//! guarantee is that iteration never knowingly continues through a
//! stale `(blob_guid, slot)` path and does not re-emit keys or
//! rollups it has already returned.

use std::sync::Arc;

use crate::api::atomic::RecordVersion;
use crate::api::errors::{Error, Result};
use crate::concurrency::MaintenanceGate;
use crate::layout::{BlobGuid, BlobNode, Leaf, NodeType, BLOB_MAX_INLINE, PREFIX_MAX_INLINE};
use crate::store::{BlobFrameRef, BufferManager, CachedBlob};

use super::cast;
use super::readers::{
    leaf_extent, ntype_of, read_leaf_key_ref, read_node16, read_node256, read_node4, read_node48,
    read_prefix,
};
use crate::engine::simd;

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

/// Builder produced by [`crate::Tree::range`].
///
/// The builder is consumed by [`RangeBuilder::into_iter`] into a
/// [`RangeIter`] yielding [`RangeEntry`] items in lex order.
#[must_use = "RangeBuilder is lazy — call `.into_iter()` or use it in a `for` loop"]
pub struct RangeBuilder {
    bm: Arc<BufferManager>,
    root_guid: BlobGuid,
    maintenance_gate: Arc<MaintenanceGate>,
    prefix: Vec<u8>,
    start_after: Option<Vec<u8>>,
    delimiter: Option<u8>,
}

impl RangeBuilder {
    /// Construct a builder anchored at `root_guid` of the BM-backed
    /// tree. Internal — user surface is [`crate::Tree::range`] /
    /// [`crate::Tree::scan_prefix`]; both signature dependencies
    /// (`BufferManager`, `BlobGuid`) live in crate-private modules.
    pub(crate) fn new(
        bm: Arc<BufferManager>,
        root_guid: BlobGuid,
        maintenance_gate: Arc<MaintenanceGate>,
    ) -> Self {
        Self {
            bm,
            root_guid,
            maintenance_gate,
            prefix: Vec::new(),
            start_after: None,
            delimiter: None,
        }
    }

    /// Restrict the scan to keys starting with `prefix`. Default:
    /// empty (the whole tree).
    pub fn prefix(mut self, prefix: &[u8]) -> Self {
        self.prefix = prefix.to_vec();
        self
    }

    /// Strict-greater-than lower bound. Default: none (start at
    /// the first matching leaf).
    pub fn start_after(mut self, key: &[u8]) -> Self {
        self.start_after = Some(key.to_vec());
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
        RangeIter {
            bm: self.bm,
            root_guid: self.root_guid,
            maintenance_gate: self.maintenance_gate,
            stack: Vec::with_capacity(32),
            curr_key: Vec::with_capacity(self.prefix.len().saturating_add(64)),
            cursor_floor: 0,
            prefix: self.prefix,
            lower_bound: self.start_after.map(LowerBound::Exclusive),
            delimiter: self.delimiter,
            last_common_prefix: None,
            initialized: false,
            terminated: false,
        }
    }
}

/// Active iteration state — see [`RangeBuilder::into_iter`].
pub struct RangeIter {
    bm: Arc<BufferManager>,
    root_guid: BlobGuid,
    maintenance_gate: Arc<MaintenanceGate>,
    /// Descent stack. Empty = no init done (if `!initialized`) or
    /// exhausted (if `terminated`).
    stack: Vec<Frame>,
    /// Bytes contributed to the current path by every live frame.
    /// On pop, the bytes the frame pushed are truncated.
    curr_key: Vec<u8>,
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
    /// Most recent `CommonPrefix` emission, used to dedup further
    /// leaves under the same rollup.
    last_common_prefix: Option<Vec<u8>>,
    initialized: bool,
    terminated: bool,
}

struct Frame {
    /// Pin keeps the blob in BM cache for the frame's lifetime.
    pin: Arc<CachedBlob>,
    blob_guid: BlobGuid,
    slot: u16,
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
    slot: u16,
    prefix: &[u8],
    lower_bound: Option<&LowerBound>,
    delimiter: Option<u8>,
    last_common_prefix: Option<&[u8]>,
) -> Result<LeafAction> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("project_range_leaf: body"))?;
    let leaf = *cast::<Leaf>(body);
    if leaf.tombstone != 0 {
        return Ok(LeafAction::Skip);
    }

    let (stored_key, value) = leaf_extent(frame, &leaf)?;
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
            let common: Vec<u8> = user_key[..=prefix.len() + idx].to_vec();
            if last_common_prefix == Some(common.as_slice()) {
                return Ok(LeafAction::Skip);
            }
            return Ok(LeafAction::CommonPrefix(common));
        }
    }
    Ok(LeafAction::Key {
        key: user_key.to_vec(),
        value: value.to_vec(),
        version: RecordVersion::new(leaf.seq),
    })
}

fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut out = prefix.to_vec();
    for i in (0..out.len()).rev() {
        if out[i] != u8::MAX {
            out[i] += 1;
            out.truncate(i + 1);
            return Some(out);
        }
    }
    None
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LowerBound {
    Exclusive(Vec<u8>),
    Inclusive(Vec<u8>),
}

impl LowerBound {
    #[inline]
    fn key(&self) -> &[u8] {
        match self {
            Self::Exclusive(bound) | Self::Inclusive(bound) => bound,
        }
    }

    #[inline]
    fn allows(&self, key: &[u8]) -> bool {
        match self {
            Self::Exclusive(bound) => key > bound.as_slice(),
            Self::Inclusive(bound) => key >= bound.as_slice(),
        }
    }
}

enum InitResult {
    Ready,
    Empty,
    Restart,
}

enum RangeAdvance {
    Entry(RangeEntry),
    Done,
    Restart,
}

enum LeafAction {
    Skip,
    Done,
    Key {
        key: Vec<u8>,
        value: Vec<u8>,
        version: RecordVersion,
    },
    CommonPrefix(Vec<u8>),
}

enum SeekStart {
    None,
    Empty,
    Key(Vec<u8>),
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
        self.next_maybe_guarded(true)
    }
}

impl RangeIter {
    /// Advance without entering `maintenance_gate`.
    /// Caller must already hold the tree's maintenance guard.
    pub(crate) fn next_unlocked(&mut self) -> Option<Result<RangeEntry>> {
        self.next_maybe_guarded(false)
    }

    fn next_maybe_guarded(&mut self, enter_gate: bool) -> Option<Result<RangeEntry>> {
        loop {
            if self.terminated {
                return None;
            }
            let step = if enter_gate {
                let maintenance_gate = Arc::clone(&self.maintenance_gate);
                let _maintenance = maintenance_gate.enter_shared();
                self.next_step()
            } else {
                self.next_step()
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
                Err(e) => {
                    self.terminated = true;
                    return Some(Err(e));
                }
            }
        }
    }

    fn next_step(&mut self) -> Result<RangeAdvance> {
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

    fn init_descent(&mut self) -> Result<InitResult> {
        let seek_start = self.effective_seek_start();
        if matches!(seek_start, SeekStart::Empty) {
            return Ok(InitResult::Empty);
        }

        // Seed the stack with the root blob's root slot.
        let root_pin = self.bm.pin(self.root_guid)?;
        let (root_slot, root_ntype, root_version) = {
            let guard = root_pin.read();
            let version = root_pin.content_version();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let slot = frame.header().root_slot;
            (slot, ntype_of(frame, slot)?, version)
        };
        self.stack.push(Frame {
            pin: root_pin,
            blob_guid: self.root_guid,
            slot: root_slot,
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
            SeekStart::Key(target) => self.seek_to_lower_bound(&target),
        }
    }

    fn effective_seek_start(&self) -> SeekStart {
        let Some(bound) = self.lower_bound.as_ref() else {
            if self.prefix.is_empty() {
                return SeekStart::None;
            }
            return SeekStart::Key(self.prefix.clone());
        };
        let bound_key = bound.key();
        if self.prefix.is_empty() {
            return SeekStart::Key(bound_key.to_vec());
        }
        if bound_key < self.prefix.as_slice() {
            return SeekStart::Key(self.prefix.clone());
        }
        if let Some(upper) = prefix_successor(&self.prefix) {
            if bound_key >= upper.as_slice() {
                return SeekStart::Empty;
            }
        }
        SeekStart::Key(bound_key.to_vec())
    }

    #[allow(clippy::too_many_lines)] // one cursor-state machine over every ART node kind
    fn seek_to_lower_bound(&mut self, target: &[u8]) -> Result<InitResult> {
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
                            if top.pin.content_version() != top.version {
                                return Ok(InitResult::Restart);
                            }
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let (stored_key, _leaf) = read_leaf_key_ref(frame, top.slot)?;
                            let user_key: &[u8] = if stored_key.last() == Some(&0) {
                                &stored_key[..stored_key.len() - 1]
                            } else {
                                stored_key
                            };
                            user_key >= target
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
                    let (child_slot, child_ntype, p_bytes, version) = {
                        let top = &self.stack[idx];
                        let guard = top_pin.read();
                        let version = top_pin.content_version();
                        if version != top.version {
                            return Ok(InitResult::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let p = read_prefix(frame, top.slot)?;
                        let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
                        let child_slot = p.child as u16;
                        (
                            child_slot,
                            ntype_of(frame, child_slot)?,
                            InlinePrefix::from_slice(&p.bytes[..plen]),
                            version,
                        )
                    };
                    match segment_seek_relation(&self.curr_key, p_bytes.as_slice(), target) {
                        SegmentRelation::Skip => {
                            self.stack[idx].next = 1;
                            self.pop_frame();
                        }
                        SegmentRelation::Continue | SegmentRelation::Min => {
                            self.stack[idx].next = 1;
                            self.push_within_blob(
                                top_pin,
                                top_blob_guid,
                                child_slot,
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
                        if top.pin.content_version() != top.version {
                            return Ok(InitResult::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let body = frame
                            .body_of_slot(top.slot)
                            .ok_or(Error::node_corrupt("range::seek: BlobNode body resolution"))?;
                        let b = cast::<BlobNode>(body);
                        let plen = (b.prefix_len as usize).min(BLOB_MAX_INLINE);
                        (
                            b.child_blob_guid,
                            InlinePrefix::from_slice(&b.bytes[..plen]),
                        )
                    };
                    match segment_seek_relation(&self.curr_key, p_bytes.as_slice(), target) {
                        SegmentRelation::Skip => {
                            self.stack[idx].next = 1;
                            self.pop_frame();
                        }
                        SegmentRelation::Continue | SegmentRelation::Min => {
                            self.stack[idx].next = 1;
                            self.push_in_other_blob(child_guid, p_bytes.as_slice())?;
                        }
                    }
                }
                NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
                    let idx = self.stack.len() - 1;
                    let relation = path_seek_relation(&self.curr_key, target);
                    let min_byte = match relation {
                        SeekRelation::Skip => {
                            self.pop_frame();
                            continue;
                        }
                        SeekRelation::Min => None,
                        SeekRelation::Seeking => Some(target[self.curr_key.len()]),
                    };
                    let top_pin = self.stack[idx].pin.clone();
                    let top_blob_guid = self.stack[idx].blob_guid;
                    let result = {
                        let top = &self.stack[idx];
                        let guard = top_pin.read();
                        let version = top_pin.content_version();
                        if version != top.version {
                            return Ok(InitResult::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let result =
                            next_inner_child_ge(frame, top.slot, top_ntype, top.next, min_byte)?;
                        match result {
                            Some((byte, child_slot, next_cursor)) => Some((
                                byte,
                                child_slot,
                                ntype_of(frame, child_slot)?,
                                next_cursor,
                                version,
                            )),
                            None => None,
                        }
                    };
                    match result {
                        None => self.pop_frame(),
                        Some((byte, child_slot, child_ntype, next_cursor, version)) => {
                            self.stack[idx].next = next_cursor;
                            self.curr_key.push(byte);
                            self.stack.push(Frame {
                                pin: top_pin,
                                blob_guid: top_blob_guid,
                                slot: child_slot,
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
            match top_ntype {
                NodeType::Leaf => {
                    let idx = self.stack.len() - 1;
                    if self.stack[idx].next == 0 {
                        self.stack[idx].next = 1;
                        let kv = {
                            let top = &self.stack[idx];
                            let guard = top.pin.read();
                            if top.pin.content_version() != top.version {
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
                                top.slot,
                                &self.prefix,
                                self.lower_bound.as_ref(),
                                self.delimiter,
                                self.last_common_prefix.as_deref(),
                            )?
                        };
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
                                self.lower_bound = Some(LowerBound::Exclusive(key.clone()));
                                self.last_common_prefix = None;
                                return Ok(RangeAdvance::Entry(RangeEntry::Key {
                                    key,
                                    value,
                                    version,
                                }));
                            }
                            LeafAction::CommonPrefix(common) => {
                                if !self.path_is_still_valid() {
                                    return Ok(RangeAdvance::Restart);
                                }
                                self.last_common_prefix = Some(common.clone());
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
                                while self.curr_key.len() > common_len
                                    && self.stack.len() > self.cursor_floor
                                {
                                    self.pop_frame();
                                }
                                if let Some(next) = prefix_successor(&common) {
                                    self.lower_bound = Some(LowerBound::Inclusive(next));
                                } else {
                                    self.terminated = true;
                                }
                                return Ok(RangeAdvance::Entry(RangeEntry::CommonPrefix(common)));
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
                        let (child_slot, child_ntype, p_bytes, version) = {
                            let top = &self.stack[idx];
                            let guard = top_pin.read();
                            let version = top_pin.content_version();
                            if version != top.version {
                                return Ok(RangeAdvance::Restart);
                            }
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let p = read_prefix(frame, top.slot)?;
                            let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
                            let child_slot = p.child as u16;
                            (
                                child_slot,
                                ntype_of(frame, child_slot)?,
                                InlinePrefix::from_slice(&p.bytes[..plen]),
                                version,
                            )
                        };
                        self.stack[idx].next = 1;
                        self.push_within_blob(
                            top_pin,
                            top_blob_guid,
                            child_slot,
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
                            if top.pin.content_version() != top.version {
                                return Ok(RangeAdvance::Restart);
                            }
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let body = frame.body_of_slot(top.slot).ok_or(Error::node_corrupt(
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
                        self.push_in_other_blob(child_guid, p_bytes.as_slice())?;
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
                        if version != top.version {
                            return Ok(RangeAdvance::Restart);
                        }
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let result = next_inner_child_from(frame, top.slot, top_ntype, top.next)?;
                        match result {
                            Some((byte, child_slot, next_cursor)) => Some((
                                byte,
                                child_slot,
                                ntype_of(frame, child_slot)?,
                                next_cursor,
                                version,
                            )),
                            None => None,
                        }
                    };
                    match result {
                        None => self.pop_frame(),
                        Some((byte, child_slot, child_ntype, next_cursor, version)) => {
                            self.stack[idx].next = next_cursor;
                            self.curr_key.push(byte);
                            self.stack.push(Frame {
                                pin: top_pin,
                                blob_guid: top_blob_guid,
                                slot: child_slot,
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
        child_slot: u16,
        child_ntype: NodeType,
        version: u64,
        prefix_bytes: &[u8],
    ) {
        self.curr_key.extend_from_slice(prefix_bytes);
        self.stack.push(Frame {
            pin,
            blob_guid,
            slot: child_slot,
            ntype: child_ntype,
            version,
            next: 0,
            pushed_bytes: prefix_bytes.len() as u16,
        });
    }

    fn push_in_other_blob(&mut self, child_guid: BlobGuid, prefix_bytes: &[u8]) -> Result<()> {
        let child_pin = self.bm.pin(child_guid)?;
        let (child_slot, child_ntype, child_version) = {
            let guard = child_pin.read();
            let version = child_pin.content_version();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let child_slot = frame.header().root_slot;
            (child_slot, ntype_of(frame, child_slot)?, version)
        };
        self.curr_key.extend_from_slice(prefix_bytes);
        self.stack.push(Frame {
            pin: child_pin,
            blob_guid: child_guid,
            slot: child_slot,
            ntype: child_ntype,
            version: child_version,
            next: 0,
            pushed_bytes: prefix_bytes.len() as u16,
        });
        Ok(())
    }

    fn path_is_still_valid(&self) -> bool {
        self.stack
            .iter()
            .all(|frame| frame.pin.validate_content_version(frame.version))
    }

    fn restart_cursor(&mut self) {
        self.bm.note_range_restart();
        self.stack.clear();
        self.curr_key.clear();
        self.cursor_floor = 0;
        self.last_common_prefix = None;
        self.initialized = false;
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

/// Given the inner node at `slot` and a cursor `from`, return the
/// next `(byte, child_slot, cursor_after)` whose branch byte is at
/// least `min_byte` when present. `None` means "the minimum child".
fn next_inner_child_ge(
    frame: BlobFrameRef<'_>,
    slot: u16,
    ntype: NodeType,
    from: u16,
    min_byte: Option<u8>,
) -> Result<Option<(u8, u16, u16)>> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame, slot)?;
            let count = (n.count as usize).min(4);
            let start = (from as usize).min(count);
            let min = min_byte.unwrap_or(0);
            for i in start..count {
                if n.keys[i] >= min {
                    return Ok(Some((n.keys[i], n.children[i] as u16, (i + 1) as u16)));
                }
            }
            Ok(None)
        }
        NodeType::Node16 => {
            let n = read_node16(frame, slot)?;
            let count = (n.count as usize).min(16);
            let start = (from as usize).min(count);
            let min = min_byte.unwrap_or(0);
            for i in start..count {
                if n.keys[i] >= min {
                    return Ok(Some((n.keys[i], n.children[i] as u16, (i + 1) as u16)));
                }
            }
            Ok(None)
        }
        NodeType::Node48 => {
            let n = read_node48(frame, slot)?;
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
            Ok(Some((b as u8, n.children[ci] as u16, (b + 1) as u16)))
        }
        NodeType::Node256 => {
            let n = read_node256(frame, slot)?;
            let min = min_byte.map_or(0, usize::from);
            let start = (from as usize).max(min).min(256);
            let Some(b) = simd::find_next_nonzero_u32(&n.children, start) else {
                return Ok(None);
            };
            let s = n.children[b];
            Ok(Some((b as u8, s as u16, (b + 1) as u16)))
        }
        _ => Err(Error::node_corrupt(
            "range::next_inner_child_ge: not an inner node",
        )),
    }
}

/// Given the inner node at `slot` and a cursor `from`, return the
/// next `(byte, child_slot, cursor_after)` if any. For `Node4` /
/// `Node16`, `from` is a key index; for `Node48` / `Node256`, it's
/// the next byte to consider (0..=256, where 256 means "no more").
fn next_inner_child_from(
    frame: BlobFrameRef<'_>,
    slot: u16,
    ntype: NodeType,
    from: u16,
) -> Result<Option<(u8, u16, u16)>> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame, slot)?;
            let count = (n.count as usize).min(4);
            let i = from as usize;
            if i >= count {
                return Ok(None);
            }
            Ok(Some((n.keys[i], n.children[i] as u16, (i + 1) as u16)))
        }
        NodeType::Node16 => {
            let n = read_node16(frame, slot)?;
            let count = (n.count as usize).min(16);
            let i = from as usize;
            if i >= count {
                return Ok(None);
            }
            Ok(Some((n.keys[i], n.children[i] as u16, (i + 1) as u16)))
        }
        NodeType::Node48 => {
            let n = read_node48(frame, slot)?;
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
            Ok(Some((b as u8, n.children[ci] as u16, (b + 1) as u16)))
        }
        NodeType::Node256 => {
            let n = read_node256(frame, slot)?;
            let start = (from as usize).min(256);
            // SIMD-scan the 256-`u32` children array for the next
            // populated slot index.
            let Some(b) = simd::find_next_nonzero_u32(&n.children, start) else {
                return Ok(None);
            };
            let s = n.children[b];
            Ok(Some((b as u8, s as u16, (b + 1) as u16)))
        }
        _ => Err(Error::node_corrupt(
            "range::next_inner_child: not an inner node",
        )),
    }
}
