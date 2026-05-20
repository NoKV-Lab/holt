//! Stateful range iterator — walk leaves in lex key order across
//! blobs with `prefix` anchoring, `start_after` skip, and S3-style
//! `delimiter` rollup.
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
//! Best-effort snapshot semantics: each `next()` re-acquires a
//! shared read guard on the topmost frame's blob for the duration
//! of one step, then drops it. Writers can interleave between
//! steps; the iterator does NOT block writers across the whole
//! traversal. A concurrent split that relocates a leaf the
//! iterator was about to visit may cause the leaf to be skipped or
//! visited twice (the path stack uses raw `(guid, slot)` pairs —
//! see the upstream `invalid iterator(#1)` warning for the same
//! failure mode). For workloads that need strong iteration
//! semantics, hold a Tree-wide write barrier externally.

use std::sync::Arc;

use crate::api::errors::{Error, Result};
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
    /// A leaf — user key + value (engine terminator already stripped).
    Key {
        /// User-supplied key bytes (terminator byte stripped).
        key: Vec<u8>,
        /// Value bytes.
        value: Vec<u8>,
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
    prefix: Vec<u8>,
    start_after: Option<Vec<u8>>,
    delimiter: Option<u8>,
}

impl RangeBuilder {
    /// Construct a builder anchored at `root_guid` of the BM-backed
    /// tree. Internal — user surface is [`crate::Tree::range`] /
    /// [`crate::Tree::scan_prefix`]; both signature dependencies
    /// (`BufferManager`, `BlobGuid`) live in crate-private modules.
    pub(crate) fn new(bm: Arc<BufferManager>, root_guid: BlobGuid) -> Self {
        Self {
            bm,
            root_guid,
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
            stack: Vec::new(),
            curr_key: Vec::new(),
            anchor_depth: 0,
            prefix: self.prefix,
            start_after: self.start_after,
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
    /// Descent stack. Empty = no init done (if `!initialized`) or
    /// exhausted (if `terminated`).
    stack: Vec<Frame>,
    /// Bytes contributed to the current path by every live frame.
    /// On pop, the bytes the frame pushed are truncated.
    curr_key: Vec<u8>,
    /// Depth at which the prefix anchor sits. The iterator stops
    /// as soon as the stack shrinks below this depth (= we'd
    /// otherwise visit siblings outside the prefix subtree).
    anchor_depth: usize,
    /// Anchor filter (raw user bytes; no engine terminator).
    prefix: Vec<u8>,
    /// `start_after` filter (raw user bytes; strict-greater compare).
    start_after: Option<Vec<u8>>,
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

fn read_range_leaf_kv(frame: BlobFrameRef<'_>, slot: u16) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
    let body = frame
        .body_of_slot(slot)
        .ok_or(Error::node_corrupt("read_range_leaf_kv: body"))?;
    let leaf = *cast::<Leaf>(body);
    if leaf.tombstone != 0 {
        return Ok(None);
    }

    let (stored_key, value) = leaf_extent(frame, &leaf)?;
    let user_key = if stored_key.last() == Some(&0) {
        &stored_key[..stored_key.len() - 1]
    } else {
        stored_key
    };
    Ok(Some((user_key.to_vec(), value.to_vec())))
}

impl Iterator for RangeIter {
    type Item = Result<RangeEntry>;

    fn next(&mut self) -> Option<Result<RangeEntry>> {
        if self.terminated {
            return None;
        }
        if !self.initialized {
            if let Err(e) = self.init_descent() {
                self.terminated = true;
                return Some(Err(e));
            }
            self.initialized = true;
            if self.stack.is_empty() {
                self.terminated = true;
                return None;
            }
        }
        loop {
            match self.advance_to_next_leaf() {
                Ok(None) => {
                    self.terminated = true;
                    return None;
                }
                Err(e) => {
                    self.terminated = true;
                    return Some(Err(e));
                }
                Ok(Some((user_key, value))) => {
                    // Defensive: the anchor descent guarantees this,
                    // but a concurrent rename could in principle
                    // surface a key outside our prefix; drop it.
                    if !user_key.starts_with(&self.prefix) {
                        continue;
                    }
                    if let Some(after) = &self.start_after {
                        if user_key.as_slice() <= after.as_slice() {
                            continue;
                        }
                    }
                    if let Some(d) = self.delimiter {
                        let rest = &user_key[self.prefix.len()..];
                        if let Some(idx) = rest.iter().position(|b| *b == d) {
                            let common: Vec<u8> = user_key[..=self.prefix.len() + idx].to_vec();
                            if self.last_common_prefix.as_deref() == Some(common.as_slice()) {
                                // Defensive dedup: fast-forward below
                                // skips most repeats, but a `Prefix`
                                // node whose bytes span the delimiter
                                // can over-pop the ascent and re-enter
                                // the same rollup. The dedup catches
                                // those.
                                continue;
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
                                && self.stack.len() > self.anchor_depth
                            {
                                self.pop_frame();
                            }
                            return Some(Ok(RangeEntry::CommonPrefix(common)));
                        }
                        // No delimiter in `rest` — emit as key and
                        // reset the rollup dedup tracker.
                        self.last_common_prefix = None;
                    }
                    return Some(Ok(RangeEntry::Key {
                        key: user_key,
                        value,
                    }));
                }
            }
        }
    }
}

impl RangeIter {
    fn init_descent(&mut self) -> Result<()> {
        // Seed the stack with the root blob's root slot.
        let root_pin = self.bm.pin(self.root_guid)?;
        let (root_slot, root_ntype) = {
            let guard = root_pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let slot = frame.header().root_slot;
            (slot, ntype_of(frame, slot)?)
        };
        self.stack.push(Frame {
            pin: root_pin,
            blob_guid: self.root_guid,
            slot: root_slot,
            ntype: root_ntype,
            next: 0,
            pushed_bytes: 0,
        });

        // Walk the prefix one byte / segment at a time.
        while self.curr_key.len() < self.prefix.len() {
            let top_ntype = self.stack.last().expect("stack non-empty").ntype;
            match top_ntype {
                NodeType::Prefix => {
                    if !self.descend_prefix_for_anchor()? {
                        self.stack.clear();
                        return Ok(());
                    }
                }
                NodeType::Blob => {
                    if !self.descend_blob_for_anchor()? {
                        self.stack.clear();
                        return Ok(());
                    }
                }
                NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
                    let need = self.prefix[self.curr_key.len()];
                    if !self.descend_inner_for_anchor(need)? {
                        self.stack.clear();
                        return Ok(());
                    }
                }
                NodeType::Leaf => {
                    // Leaf reached mid-prefix: check if its full
                    // key satisfies the user prefix; if yes, anchor
                    // here, else empty result.
                    let matches_prefix = {
                        let top = self.stack.last().unwrap();
                        let guard = top.pin.read();
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        let (stored_key, _leaf) = read_leaf_key_ref(frame, top.slot)?;
                        let user_key: &[u8] = if stored_key.last() == Some(&0) {
                            &stored_key[..stored_key.len() - 1]
                        } else {
                            stored_key
                        };
                        user_key.starts_with(&self.prefix)
                    };
                    if !matches_prefix {
                        self.stack.clear();
                        return Ok(());
                    }
                    break;
                }
                NodeType::EmptyRoot | NodeType::Invalid => {
                    self.stack.clear();
                    return Ok(());
                }
            }
        }
        self.anchor_depth = self.stack.len();
        Ok(())
    }

    /// Descend a `Prefix` node during the anchor walk. Returns
    /// `true` if descent succeeded (anchor walk continues), `false`
    /// if the stored prefix bytes mismatch the user prefix
    /// (iterator is empty).
    fn descend_prefix_for_anchor(&mut self) -> Result<bool> {
        let (child_slot, p_bytes) = {
            let top = self.stack.last().expect("stack non-empty");
            let guard = top.pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let p = read_prefix(frame, top.slot)?;
            let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
            (p.child as u16, InlinePrefix::from_slice(&p.bytes[..plen]))
        };
        let p_bytes = p_bytes.as_slice();
        let remaining = &self.prefix[self.curr_key.len()..];
        let cmp_len = p_bytes.len().min(remaining.len());
        if p_bytes[..cmp_len] != remaining[..cmp_len] {
            return Ok(false);
        }
        let (top_pin, top_blob_guid) = {
            let top = self.stack.last().expect("stack non-empty");
            (top.pin.clone(), top.blob_guid)
        };
        self.stack.last_mut().unwrap().next = 1;
        self.curr_key.extend_from_slice(p_bytes);
        let child_ntype = {
            let guard = top_pin.read();
            ntype_of(BlobFrameRef::wrap(guard.as_slice()), child_slot)?
        };
        let pushed = p_bytes.len() as u16;
        self.stack.push(Frame {
            pin: top_pin,
            blob_guid: top_blob_guid,
            slot: child_slot,
            ntype: child_ntype,
            next: 0,
            pushed_bytes: pushed,
        });
        Ok(true)
    }

    fn descend_blob_for_anchor(&mut self) -> Result<bool> {
        let (child_guid, child_slot, p_bytes) = {
            let top = self.stack.last().expect("stack non-empty");
            let guard = top.pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let body = frame.body_of_slot(top.slot).ok_or(Error::node_corrupt(
                "range::descend_blob_for_anchor: body resolution",
            ))?;
            let b = cast::<BlobNode>(body);
            let plen = (b.prefix_len as usize).min(BLOB_MAX_INLINE);
            (
                b.child_blob_guid,
                b.child_entry_ptr as u16,
                InlinePrefix::from_slice(&b.bytes[..plen]),
            )
        };
        let p_bytes = p_bytes.as_slice();
        let remaining = &self.prefix[self.curr_key.len()..];
        let cmp_len = p_bytes.len().min(remaining.len());
        if !p_bytes.is_empty() && p_bytes[..cmp_len] != remaining[..cmp_len] {
            return Ok(false);
        }
        self.stack.last_mut().unwrap().next = 1;
        self.curr_key.extend_from_slice(p_bytes);
        let child_pin = self.bm.pin(child_guid)?;
        let child_ntype = {
            let guard = child_pin.read();
            ntype_of(BlobFrameRef::wrap(guard.as_slice()), child_slot)?
        };
        let pushed = p_bytes.len() as u16;
        self.stack.push(Frame {
            pin: child_pin,
            blob_guid: child_guid,
            slot: child_slot,
            ntype: child_ntype,
            next: 0,
            pushed_bytes: pushed,
        });
        Ok(true)
    }

    fn descend_inner_for_anchor(&mut self, need: u8) -> Result<bool> {
        // Find the child for `need`; if absent, anchor walk fails.
        let (top_pin, top_blob_guid, top_slot, top_ntype, child_slot, cursor_after) = {
            let top = self.stack.last().expect("stack non-empty");
            let guard = top.pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let Some((slot, cursor)) =
                find_inner_child_and_cursor(frame, top.slot, top.ntype, need)?
            else {
                return Ok(false);
            };
            (
                top.pin.clone(),
                top.blob_guid,
                top.slot,
                top.ntype,
                slot,
                cursor,
            )
        };
        let _ = (top_slot, top_ntype); // for borrow-scope clarity
        self.stack.last_mut().unwrap().next = cursor_after;
        self.curr_key.push(need);
        let child_ntype = {
            let guard = top_pin.read();
            ntype_of(BlobFrameRef::wrap(guard.as_slice()), child_slot)?
        };
        self.stack.push(Frame {
            pin: top_pin,
            blob_guid: top_blob_guid,
            slot: child_slot,
            ntype: child_ntype,
            next: 0,
            pushed_bytes: 1,
        });
        Ok(true)
    }

    #[allow(clippy::too_many_lines)] // single match over six NodeType variants — splitting hides the loop shape
    fn advance_to_next_leaf(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        loop {
            // Anchor stop: dropping below the anchor depth means we
            // would walk siblings outside the prefix subtree.
            if self.stack.len() < self.anchor_depth {
                return Ok(None);
            }
            let Some(top) = self.stack.last_mut() else {
                return Ok(None);
            };
            let top_ntype = top.ntype;
            match top_ntype {
                NodeType::Leaf => {
                    if top.next == 0 {
                        top.next = 1;
                        let kv = {
                            let guard = top.pin.read();
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            // Soft-deleted leaves stay physically in
                            // the slot table (and their key/value
                            // extent bytes stay allocated) until
                            // `compact_blob` rebuilds the blob; range
                            // iteration must skip them so a leaf
                            // that was erased between snapshot and
                            // iteration isn't emitted.
                            read_range_leaf_kv(frame, top.slot)?
                        };
                        if let Some((key, value)) = kv {
                            return Ok(Some((key, value)));
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
                    if top.next == 0 {
                        top.next = 1;
                        let (top_pin, top_blob_guid) = (top.pin.clone(), top.blob_guid);
                        let (child_slot, p_bytes) = {
                            let guard = top_pin.read();
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let p = read_prefix(frame, top.slot)?;
                            let plen = (p.prefix_len as usize).min(PREFIX_MAX_INLINE);
                            (p.child as u16, InlinePrefix::from_slice(&p.bytes[..plen]))
                        };
                        self.push_within_blob(
                            top_pin,
                            top_blob_guid,
                            child_slot,
                            p_bytes.as_slice(),
                        )?;
                    } else {
                        self.pop_frame();
                    }
                }
                NodeType::Blob => {
                    if top.next == 0 {
                        top.next = 1;
                        let (child_guid, child_slot, p_bytes) = {
                            let guard = top.pin.read();
                            let frame = BlobFrameRef::wrap(guard.as_slice());
                            let body = frame.body_of_slot(top.slot).ok_or(Error::node_corrupt(
                                "range::advance: BlobNode body resolution",
                            ))?;
                            let b = cast::<BlobNode>(body);
                            let plen = (b.prefix_len as usize).min(BLOB_MAX_INLINE);
                            (
                                b.child_blob_guid,
                                b.child_entry_ptr as u16,
                                InlinePrefix::from_slice(&b.bytes[..plen]),
                            )
                        };
                        self.push_in_other_blob(child_guid, child_slot, p_bytes.as_slice())?;
                    } else {
                        self.pop_frame();
                    }
                }
                NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
                    let (top_pin, top_blob_guid, top_slot, cursor) =
                        (top.pin.clone(), top.blob_guid, top.slot, top.next);
                    let result = {
                        let guard = top_pin.read();
                        let frame = BlobFrameRef::wrap(guard.as_slice());
                        next_inner_child_from(frame, top_slot, top_ntype, cursor)?
                    };
                    match result {
                        None => self.pop_frame(),
                        Some((byte, child_slot, next_cursor)) => {
                            self.stack.last_mut().unwrap().next = next_cursor;
                            let child_ntype = {
                                let guard = top_pin.read();
                                ntype_of(BlobFrameRef::wrap(guard.as_slice()), child_slot)?
                            };
                            self.curr_key.push(byte);
                            self.stack.push(Frame {
                                pin: top_pin,
                                blob_guid: top_blob_guid,
                                slot: child_slot,
                                ntype: child_ntype,
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
        prefix_bytes: &[u8],
    ) -> Result<()> {
        let child_ntype = {
            let guard = pin.read();
            ntype_of(BlobFrameRef::wrap(guard.as_slice()), child_slot)?
        };
        self.curr_key.extend_from_slice(prefix_bytes);
        self.stack.push(Frame {
            pin,
            blob_guid,
            slot: child_slot,
            ntype: child_ntype,
            next: 0,
            pushed_bytes: prefix_bytes.len() as u16,
        });
        Ok(())
    }

    fn push_in_other_blob(
        &mut self,
        child_guid: BlobGuid,
        child_slot: u16,
        prefix_bytes: &[u8],
    ) -> Result<()> {
        let child_pin = self.bm.pin(child_guid)?;
        let child_ntype = {
            let guard = child_pin.read();
            ntype_of(BlobFrameRef::wrap(guard.as_slice()), child_slot)?
        };
        self.curr_key.extend_from_slice(prefix_bytes);
        self.stack.push(Frame {
            pin: child_pin,
            blob_guid: child_guid,
            slot: child_slot,
            ntype: child_ntype,
            next: 0,
            pushed_bytes: prefix_bytes.len() as u16,
        });
        Ok(())
    }

    fn pop_frame(&mut self) {
        let Some(f) = self.stack.pop() else { return };
        let new_len = self.curr_key.len().saturating_sub(f.pushed_bytes as usize);
        self.curr_key.truncate(new_len);
    }
}

/// Look up `byte` in the inner node at `slot`. Returns
/// `Some((child_slot, cursor_after_byte))` if found — `cursor_after_byte`
/// is the cursor value that, plugged back into `next_inner_child_from`,
/// would skip past this child.
fn find_inner_child_and_cursor(
    frame: BlobFrameRef<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
) -> Result<Option<(u16, u16)>> {
    match ntype {
        NodeType::Node4 => {
            let n = read_node4(frame, slot)?;
            let count = (n.count as usize).min(4);
            for i in 0..count {
                if n.keys[i] == byte {
                    return Ok(Some((n.children[i] as u16, (i + 1) as u16)));
                }
                if n.keys[i] > byte {
                    return Ok(None);
                }
            }
            Ok(None)
        }
        NodeType::Node16 => {
            let n = read_node16(frame, slot)?;
            // SIMD lookup over the 16-key array; Node16 sorts its
            // keys so a single positive hit replaces the
            // scalar-loop + ordered-early-out idiom.
            match simd::node16_find_byte(&n.keys, n.count, byte) {
                Some(i) => Ok(Some((n.children[i as usize] as u16, u16::from(i) + 1))),
                None => Ok(None),
            }
        }
        NodeType::Node48 => {
            let n = read_node48(frame, slot)?;
            let idx = n.index[byte as usize];
            if idx == 0 {
                return Ok(None);
            }
            let ci = idx as usize - 1;
            if ci >= 48 {
                return Err(Error::node_corrupt(
                    "range::find_inner_child: Node48 index out of range",
                ));
            }
            Ok(Some((n.children[ci] as u16, u16::from(byte) + 1)))
        }
        NodeType::Node256 => {
            let n = read_node256(frame, slot)?;
            let s = n.children[byte as usize];
            if s == 0 {
                return Ok(None);
            }
            Ok(Some((s as u16, u16::from(byte) + 1)))
        }
        _ => Err(Error::node_corrupt(
            "range::find_inner_child: not an inner node",
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
