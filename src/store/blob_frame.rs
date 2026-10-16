//! `BlobFrame` — typed view over one 512 KB blob.
//!
//! Provides:
//!   - `init`: format a fresh buffer (writes header, seeds the
//!     empty_root sentinel at slot 1)
//!   - `alloc_node(ntype)`: try free-list first, else bump
//!   - `free_node(slot)`: push onto per-NodeType LIFO
//!   - `alloc_leaf(total_aligned)`: variable-size bump for a single
//!     self-describing leaf node (`[16B header][key][value]`)
//!   - `body_of_slot(slot)`: resolve a slot to its body slice (for a
//!     `Leaf` the size is read back from the header — Option B,
//!     self-describing)
//!
//! All operations enforce the on-disk invariants: 8-byte body
//! alignment, MAX_SLOTS cap, and slot-entry bit-packing.

use crate::layout::{
    leaf_body_size, size_of_node, BlobGuid, BlobHeader, Leaf, NodeType, SlotEntry, SlotEntryRaw,
    DATA_AREA_START, HEADER_SIZE, MAX_SLOTS, PAGE_SIZE,
};
use std::mem::size_of;

/// Bytes the bump allocator reserves for spillover's
/// emergency `BlobNode` install. Walker `alloc_node` (non-Blob) +
/// `alloc_leaf` refuse to consume the last `SPILLOVER_RESERVATION`
/// bytes; `alloc_node(NodeType::Blob)` is exempt and may consume
/// them. Without this, a 99 %-full blob has no room left for the
/// spillover code path to install its own placeholder BlobNode.
///
/// Equals one `BlobNode` body — the only structure spillover ever
/// emits.
pub const SPILLOVER_RESERVATION: u32 = 128;

/// `DATA_AREA_START / 8` — the biased origin all child offsets are
/// measured from. Every node body lives at `byte_offset >=
/// DATA_AREA_START`, so its `byte_offset / 8` is `>= DATA_BASE_DIV8`.
const DATA_BASE_DIV8: u32 = DATA_AREA_START / 8;

/// Largest biased child value that can occur — `(PAGE_SIZE -
/// DATA_AREA_START)/8 + 1`. Asserted `< u16::MAX` so the existing
/// `[u16; N]` child arrays hold any in-blob offset with headroom.
const MAX_CHILD_BIAS: u32 = (PAGE_SIZE - DATA_AREA_START) / 8 + 1;
const _: () = assert!(MAX_CHILD_BIAS < u16::MAX as u32);

/// 4 KB OS page granularity — distinct from [`PAGE_SIZE`], which is the
/// 512 KB blob frame. A indexed read fetches leaves a page at a time, so
/// the routing-aware compactor (`compact_blob`) page-aligns the leaf
/// region to this boundary.
pub const PAGE_4K: u32 = 0x1000;
// DATA_AREA_START is itself a 4 KB multiple, so routing_off needs no
// rounding — only the leaf region start does.
const _: () = assert!(DATA_AREA_START % PAGE_4K == 0);

/// Round `off` up to the next [`PAGE_4K`] boundary.
///
/// Used by the routing compactor to place `leaf_region_start` on a page
/// boundary. `off` is always a small in-frame offset (well under
/// `PAGE_SIZE`), so `off + (PAGE_4K - 1)` cannot overflow.
#[inline]
#[must_use]
pub const fn page_align_up(off: u32) -> u32 {
    (off + (PAGE_4K - 1)) & !(PAGE_4K - 1)
}

/// Bytes of a frame image that carry state: the header and slot table plus
/// the data area up to `space_used`, rounded up to whole 4 KB pages.
///
/// Every live node sits below `space_used`, a routed blob's leaf region
/// included, so the file store persists only this prefix of a slot.
#[must_use]
pub fn persisted_extent(frame: &[u8]) -> usize {
    let used = BlobFrameRef::wrap(frame)
        .header()
        .space_used
        .clamp(DATA_AREA_START, PAGE_SIZE);
    page_align_up(used) as usize
}

/// Zero a frame past its persisted extent.
///
/// A slot is reused across blobs and only its persisted extent is
/// rewritten, so a read must not carry another blob's bytes past it.
pub fn clear_unpersisted_tail(frame: &mut [u8]) {
    let extent = persisted_extent(frame);
    frame[extent..].fill(0);
}

/// Encode a node body's absolute `byte_offset` into the biased `u16`
/// stored in a child field (`children[N]`, `Prefix.child`,
/// `header.root`).
///
/// Encoding is `(byte_offset / 8) - (DATA_AREA_START / 8) + 1`. The
/// `+1` bias reserves the encoded value `0` as the universal
/// "no child / null" sentinel (Node48 `index -> children`, Node256
/// direct slots, 1-based-ness). A real body sits at `byte_offset >=
/// DATA_AREA_START`, so its encoded value is always `>= 1` — never
/// collides with the null sentinel even for a body at exactly
/// `DATA_AREA_START` (which only ever happens for the init EmptyRoot,
/// never a child target).
#[inline]
#[must_use]
pub fn encode_child_off(byte_offset: u32) -> u16 {
    debug_assert_eq!(byte_offset % 8, 0, "child body must be 8-byte aligned");
    debug_assert!(
        byte_offset >= DATA_AREA_START,
        "child body offset below data area: {byte_offset:#x}"
    );
    debug_assert!(byte_offset < PAGE_SIZE, "child body offset past page");
    let biased = byte_offset / 8 - DATA_BASE_DIV8 + 1;
    debug_assert!(
        biased >= 1,
        "encoded child offset must never be the 0 sentinel"
    );
    debug_assert!(biased <= MAX_CHILD_BIAS);
    biased as u16
}

/// Decode a biased child `u16` back to the absolute `byte_offset`.
/// Inverse of [`encode_child_off`]. The caller must have already
/// rejected the `0` null sentinel.
#[inline]
#[must_use]
pub fn decode_child_off(encoded: u16) -> u32 {
    debug_assert_ne!(encoded, 0, "decode_child_off on the 0 null sentinel");
    (u32::from(encoded) - 1 + DATA_BASE_DIV8) * 8
}

/// Read a node's `NodeType` from its body at absolute `off`.
///
/// Every node body carries its `node_type` discriminant at `+1`
/// (every inner node, `Prefix`, `BlobNode`, and — post-reorder — the
/// `Leaf` header; the `EmptyRoot` sentinel's `+1` is stamped in
/// [`BlobFrame::init`]). This is the offset-addressed replacement for
/// the slot-table `node_type` lookup. Returns `None` if `off + 2`
/// runs past the buffer or the byte doesn't decode to a `NodeType`.
#[inline]
#[must_use]
pub fn ntype_at_offset(buf: &[u8], off: usize) -> Option<NodeType> {
    let tag_off = off.checked_add(1)?;
    if tag_off >= buf.len() {
        return None;
    }
    NodeType::from_raw(buf[tag_off])
}

/// Free-list size class for `ntype` — the index into
/// [`BlobHeader::free_list_head`](crate::layout::BlobHeader). Every
/// type maps to its own `ntype - 1` slot. (Leaf nodes are now
/// variable-size and self-describing; a freed leaf slot is reclaimed
/// by compaction exactly as its bytes were when leaves used separate
/// extents, so it parks on its own class-0 list.)
const fn free_list_class(ntype: NodeType) -> usize {
    ntype.as_u8() as usize - 1
}

/// Resolve the true on-disk body length of a `Leaf` slot whose header
/// begins at byte `off` in `buf`.
///
/// A leaf is a single, contiguous, self-describing node
/// (`[16B header][key][value]`). Its slot-table entry records only the
/// header offset; `size_of_node(NodeType::Leaf)` is just the 16-byte
/// header. To size the whole node we read `key_len`/`value_len` from
/// the header and return `leaf_body_size(key_len, value_len)` =
/// `align8(16 + key_len + value_len)` — this is the ONE shared helper
/// used by all three `body_of_slot` implementations (Option B,
/// self-describing). Returns `None` if the 16-byte header or the sized
/// body would run past the buffer.
fn leaf_body_len_at(buf: &[u8], off: usize) -> Option<usize> {
    let hdr_end = off.checked_add(size_of::<Leaf>())?;
    if hdr_end > buf.len() {
        return None;
    }
    // SAFETY-equivalent: read the two length fields directly from the
    // header bytes (value_len @ +2, key_len @ +4) without forming a
    // misaligned reference. Little-endian, matching `#[repr(C)]`.
    let value_len = u32::from(u16::from_le_bytes([buf[off + 2], buf[off + 3]]));
    let key_len = u32::from(u16::from_le_bytes([buf[off + 4], buf[off + 5]]));
    let total = leaf_body_size(key_len, value_len) as usize;
    if off.checked_add(total)? > buf.len() {
        return None;
    }
    Some(total)
}

/// Errors from `alloc_node` / `alloc_leaf`.
#[derive(Debug, PartialEq, Eq)]
pub enum AllocError {
    /// `num_slots` reached `MAX_SLOTS` — no more slot indices.
    OutOfSlots,
    /// Data area can't fit the requested size.
    OutOfSpace {
        /// Bytes the caller requested (8-byte aligned for leaves).
        need: u32,
        /// Bytes actually free in the data area.
        avail: u32,
    },
    /// `alloc_node(NodeType::Invalid)` or `alloc_leaf(0)`.
    InvalidRequest,
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfSlots => write!(f, "blob slot table exhausted ({MAX_SLOTS} slots)"),
            Self::OutOfSpace { need, avail } => {
                write!(
                    f,
                    "blob data area exhausted (need {need} bytes, {avail} available)"
                )
            }
            Self::InvalidRequest => write!(f, "invalid allocation request"),
        }
    }
}
impl std::error::Error for AllocError {}

/// Errors from `free_node`.
#[derive(Debug, PartialEq, Eq)]
pub enum FreeError {
    /// Slot index 0 or > `num_slots`.
    InvalidSlot(u16),
    /// Slot entry's tag doesn't decode to a valid NodeType.
    TypeMismatch {
        /// Which slot was attempted.
        slot: u16,
        /// The 15-bit tag we found instead of a NodeType.
        tag: u16,
    },
}

impl std::fmt::Display for FreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSlot(s) => write!(f, "free_node: invalid slot index {s}"),
            Self::TypeMismatch { slot, tag } => {
                write!(f, "free_node: slot {slot} has invalid type tag {tag}")
            }
        }
    }
}
impl std::error::Error for FreeError {}

/// Outcome of a node alloc — just the 1-based slot index.
/// Callers write into the body via `body_of_slot_mut(slot)`.
#[derive(Debug, Clone, Copy)]
pub struct AllocOutcome {
    /// 1-based slot index of the allocated body.
    pub slot: u16,
}

/// Read-only typed view over a `PAGE_SIZE`-byte buffer.
///
/// `BlobFrameRef` is `Copy` — pass it by value. Walker `lookup` /
/// `descend` paths take `BlobFrameRef` so they work against a
/// `&[u8]` borrowed from a `RwLock` read-guard (i.e., directly
/// against the [`crate::BufferManager`]'s cached buffer with no
/// per-op `memcpy`).
///
/// For mutating walker paths see [`BlobFrame`], which holds
/// `&mut [u8]` instead.
#[derive(Clone, Copy)]
pub struct BlobFrameRef<'a> {
    buf: &'a [u8],
}

impl<'a> BlobFrameRef<'a> {
    /// Wrap an existing `PAGE_SIZE`-byte buffer.
    #[must_use]
    pub fn wrap(buf: &'a [u8]) -> Self {
        assert_eq!(
            buf.len(),
            PAGE_SIZE as usize,
            "BlobFrameRef requires PAGE_SIZE buffer"
        );
        Self { buf }
    }

    /// Const reference to the header.
    #[must_use]
    pub fn header(&self) -> &'a BlobHeader {
        // SAFETY: buffer is PAGE_SIZE bytes, starts with a
        // properly-aligned BlobHeader (4096 B; alignment satisfied
        // by AlignedBlobBuf's 4 KB-aligned heap allocation).
        unsafe { &*self.buf.as_ptr().cast::<BlobHeader>() }
    }

    /// Resolve a node's `NodeType` directly from its body at absolute
    /// `byte_offset` — the offset-addressed replacement for
    /// `slot_entry(slot).node_type()`. See [`ntype_at_offset`].
    #[inline]
    #[must_use]
    pub fn ntype_at(&self, byte_offset: u32) -> Option<NodeType> {
        ntype_at_offset(self.buf, byte_offset as usize)
    }

    /// Resolve a node body at absolute `byte_offset` to a const slice.
    ///
    /// The offset-addressed twin of [`Self::body_of_slot`]: it reads the
    /// `NodeType` from `body[off + 1]` (no slot lookup) and sizes the
    /// body — a `Leaf` via its self-describing header, everything else
    /// via `size_of_node`. Returns `None` on an out-of-range offset or
    /// an undecodable / `Invalid` type tag.
    #[must_use]
    pub fn body_at_offset(&self, byte_offset: u32) -> Option<&'a [u8]> {
        let off = byte_offset as usize;
        let ntype = ntype_at_offset(self.buf, off)?;
        if ntype == NodeType::Invalid {
            return None;
        }
        let size = if ntype == NodeType::Leaf {
            leaf_body_len_at(self.buf, off)?
        } else {
            size_of_node(ntype) as usize
        };
        let end = off.checked_add(size)?;
        if end > self.buf.len() {
            return None;
        }
        Some(&self.buf[off..end])
    }

    /// Best-effort prefetch of the node body at absolute `byte_offset`.
    /// Offset-addressed twin of [`Self::prefetch_node`]; never faults.
    #[inline]
    pub fn prefetch_at(&self, byte_offset: u32) {
        let off = byte_offset as usize;
        if off < self.buf.len() {
            // SAFETY: `off < buf.len()`, so the pointer is in-bounds;
            // a prefetch hint reads nothing and cannot fault.
            crate::engine::prefetch_read_data(unsafe { self.buf.as_ptr().add(off) });
        }
    }
}

/// Typed view over a 524288-byte buffer formatted as a BlobFrame.
///
/// `BlobFrame` does not own the buffer; the caller (BufferManager
/// or a test) provides `&mut [u8]` of length `PAGE_SIZE`.
///
/// For read-only access (e.g. walker `lookup` against a
/// `RwLock`-guarded `BufferManager` blob), use [`BlobFrameRef`]
/// instead — it wraps `&[u8]` and is `Copy`.
///
pub struct BlobFrame<'a> {
    /// Backing buffer (must be exactly `PAGE_SIZE` bytes).
    buf: &'a mut [u8],
}

impl<'a> BlobFrame<'a> {
    /// Wrap an existing buffer that's already been formatted.
    ///
    /// Caller asserts that `buf` was previously initialized via
    /// `init` (or loaded from disk in the same format). No
    /// validation is done.
    pub fn wrap(buf: &'a mut [u8]) -> Self {
        assert_eq!(
            buf.len(),
            PAGE_SIZE as usize,
            "BlobFrame requires PAGE_SIZE buffer"
        );
        Self { buf }
    }

    /// Cheap conversion to a read-only [`BlobFrameRef`]. Useful
    /// for forwarding into walker `lookup` / `descend` paths
    /// (which take `BlobFrameRef` so they also work against
    /// `RwLock`-guarded `BufferManager` slices).
    #[must_use]
    pub fn as_ref(&self) -> BlobFrameRef<'_> {
        BlobFrameRef { buf: self.buf }
    }

    /// Copy the complete frame image into another frame-sized buffer.
    ///
    /// Structural walkers use this to prepare a fallible rewrite off to
    /// the side, then publish it with one infallible replacement only after
    /// every allocation and edge update has succeeded.
    pub(crate) fn copy_bytes_to(&self, dst: &mut [u8]) {
        assert_eq!(dst.len(), PAGE_SIZE as usize);
        dst.copy_from_slice(self.buf);
    }

    /// Replace this complete frame with a previously prepared frame image.
    pub(crate) fn replace_bytes_from(&mut self, src: &[u8]) {
        assert_eq!(src.len(), PAGE_SIZE as usize);
        self.buf.copy_from_slice(src);
    }

    /// Initialize a fresh blob from a zeroed buffer.
    ///
    /// Writes the header, sets `space_used` to `DATA_AREA_START`,
    /// allocates the empty-tree root sentinel (8-byte all-zero
    /// node at slot 1), and stores its slot index in `root_slot`.
    pub fn init(buf: &'a mut [u8], guid: BlobGuid) -> Result<Self, AllocError> {
        assert_eq!(buf.len(), PAGE_SIZE as usize);
        // Zero the buffer so all our atomic-by-construction
        // invariants hold (header counters = 0, slot table empty,
        // data area zeroed).
        for b in buf.iter_mut() {
            *b = 0;
        }
        let mut frame = Self { buf };
        // Seed the header.
        {
            let h = frame.header_mut();
            h.blob_guid = guid;
            h.space_used = DATA_AREA_START;
        }
        // Allocate the empty-tree root sentinel.
        let out = frame.alloc_node(NodeType::EmptyRoot)?;
        debug_assert_eq!(out.slot, 1);
        // The 8-byte body is all-zero from the memset above; stamp its
        // self-describing `node_type @ +1` byte so an offset-addressed
        // reader resolves it as `EmptyRoot` (the only node whose body
        // isn't otherwise written with a `node_type` byte). Then record
        // its encoded body offset as the tree root.
        let root_off = frame
            .offset_of_slot(out.slot)
            .expect("freshly allocated sentinel has a slot offset");
        if let Some(body) = frame.bytes_at_mut(root_off, 8) {
            body[1] = NodeType::EmptyRoot.as_u8();
        }
        frame.header_mut().root_slot = encode_child_off(root_off);
        Ok(frame)
    }

    /// Const reference to the header.
    #[must_use]
    pub fn header(&self) -> &BlobHeader {
        // SAFETY: BlobFrame's buffer is PAGE_SIZE bytes and starts
        // with a properly-aligned BlobHeader (header is 4096 B
        // with natural u64 alignment requirements satisfied by
        // PAGE_SIZE-aligned allocations; callers must provide an
        // 8-byte aligned buffer).
        unsafe { &*self.buf.as_ptr().cast::<BlobHeader>() }
    }

    /// Mutable reference to the header.
    #[must_use]
    pub fn header_mut(&mut self) -> &mut BlobHeader {
        // SAFETY: see `header`.
        unsafe { &mut *self.buf.as_mut_ptr().cast::<BlobHeader>() }
    }

    /// Read the slot table entry for a 1-based slot index.
    ///
    /// Returns `None` if `slot` is 0 or beyond `num_slots`.
    #[must_use]
    pub fn slot_entry(&self, slot: u16) -> Option<SlotEntry> {
        let h = self.header();
        if slot == 0 || slot > h.num_slots {
            return None;
        }
        let off = HEADER_SIZE as usize + (slot as usize - 1) * 4;
        let raw_bytes = &self.buf[off..off + 4];
        let raw = u32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
        Some(SlotEntryRaw(raw).decode())
    }

    /// Write a slot table entry. `slot` must be in 1..=num_slots.
    fn write_slot_entry(&mut self, slot: u16, entry: SlotEntry) {
        debug_assert!(slot >= 1);
        debug_assert!(slot <= self.header().num_slots || slot == self.header().num_slots + 1);
        let off = HEADER_SIZE as usize + (slot as usize - 1) * 4;
        let raw = entry.raw().0;
        self.buf[off..off + 4].copy_from_slice(&raw.to_le_bytes());
    }

    /// Resolve a slot to a const view of its body bytes.
    ///
    /// For a `Leaf` the body is variable-size and self-describing:
    /// its length is read back from the 16-byte header via
    /// [`leaf_body_len_at`]. Every other type uses its fixed
    /// `size_of_node`. v4 addresses nodes by offset
    /// ([`Self::body_at_offset`]); this slot reader survives for tests.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub fn body_of_slot(&self, slot: u16) -> Option<&[u8]> {
        let e = self.slot_entry(slot)?;
        let ntype = e.node_type()?;
        if ntype == NodeType::Invalid {
            return None;
        }
        let off = e.byte_offset() as usize;
        let size = if ntype == NodeType::Leaf {
            leaf_body_len_at(self.buf, off)?
        } else {
            size_of_node(ntype) as usize
        };
        if off + size > self.buf.len() {
            return None;
        }
        Some(&self.buf[off..off + size])
    }

    /// Resolve a node's `NodeType` directly from its body at absolute
    /// `byte_offset` — offset-addressed twin of
    /// `slot_entry(slot).node_type()`.
    #[inline]
    #[must_use]
    pub fn ntype_at(&self, byte_offset: u32) -> Option<NodeType> {
        ntype_at_offset(self.buf, byte_offset as usize)
    }

    /// Resolve a node body at absolute `byte_offset` to a const slice.
    /// Offset-addressed twin of [`Self::body_of_slot`].
    #[must_use]
    pub fn body_at_offset(&self, byte_offset: u32) -> Option<&[u8]> {
        let off = byte_offset as usize;
        let ntype = ntype_at_offset(self.buf, off)?;
        if ntype == NodeType::Invalid {
            return None;
        }
        let size = if ntype == NodeType::Leaf {
            leaf_body_len_at(self.buf, off)?
        } else {
            size_of_node(ntype) as usize
        };
        let end = off.checked_add(size)?;
        if end > self.buf.len() {
            return None;
        }
        Some(&self.buf[off..end])
    }

    /// Mutable node body at absolute `byte_offset`. Offset-addressed
    /// twin of [`Self::body_of_slot_mut`].
    pub fn body_at_offset_mut(&mut self, byte_offset: u32) -> Option<&mut [u8]> {
        let off = byte_offset as usize;
        let ntype = ntype_at_offset(self.buf, off)?;
        if ntype == NodeType::Invalid {
            return None;
        }
        let size = if ntype == NodeType::Leaf {
            leaf_body_len_at(self.buf, off)?
        } else {
            size_of_node(ntype) as usize
        };
        let end = off.checked_add(size)?;
        if end > self.buf.len() {
            return None;
        }
        Some(&mut self.buf[off..end])
    }

    /// Byte offset of the body registered in slot `slot`, if any.
    /// Used by the allocator-facing call sites that still allocate a
    /// slot for bookkeeping but address the body by offset.
    #[must_use]
    pub fn offset_of_slot(&self, slot: u16) -> Option<u32> {
        self.slot_entry(slot).map(SlotEntry::byte_offset)
    }

    /// Record that the node body at `byte_offset` has been abandoned
    /// (made unreachable by a structural op). Bumps `header.dead_bytes`
    /// by the body's size so [`crate::engine::walker::blob_needs_compaction`]
    /// can trigger a rebuild before the blob bloats. Best-effort: an
    /// unresolvable offset is silently ignored (the bytes are still
    /// reclaimed at the next compaction regardless of the counter).
    pub fn note_abandoned(&mut self, byte_offset: u32) {
        let size = {
            let off = byte_offset as usize;
            match ntype_at_offset(self.buf, off) {
                Some(NodeType::Leaf) => leaf_body_len_at(self.buf, off).unwrap_or(0) as u32,
                Some(nt) if nt != NodeType::Invalid => size_of_node(nt),
                _ => 0,
            }
        };
        let h = self.header_mut();
        h.dead_bytes = h.dead_bytes.saturating_add(size);
    }

    /// Allocate a node of the given NodeType.
    ///
    /// Tries the per-NodeType free list first; for size-128
    /// NodeTypes (`Prefix` ↔ `Blob`) also tries the sibling
    /// 128-byte free list before falling back to bump-allocation
    /// from `space_used`. Returns `(slot, body_offset, size)` —
    /// caller writes the body via `body_of_slot_mut(slot)`.
    ///
    /// **Why the cross-type fallback?** Spillover allocates a
    /// `BlobNode` (128 B) at the exact moment the blob is full
    /// — bump-allocation can't satisfy it. But the same spillover
    /// just freed a victim subtree, which typically contains
    /// several `Prefix` nodes (also 128 B). Letting `alloc(Blob)`
    /// reuse those slot bodies makes spillover succeed without
    /// extending the bump cursor.
    ///
    pub fn alloc_node(&mut self, ntype: NodeType) -> Result<AllocOutcome, AllocError> {
        let outcome = self.alloc_node_inner(ntype)?;
        // A new internal node allocated into a *routed* frame lands at the
        // leaf-arena high-water (>= leaf_region_start), where a indexed reader
        // classifies offsets as leaves — silently breaking the routing
        // invariant. Invalidate the routing layout (the next compaction
        // re-routes); indexed reads fall back to the legacy whole-frame path
        // until then. (In-place leaf allocs de-route too — see
        // `alloc_leaf` — but for a different reason: they rewrite a parent
        // child pointer in the routing region, so any cached indexed-read
        // navigation page is stale.) During the routing build itself
        // routing_len is still 0, so this is a no-op.
        if self.header().routing_len != 0 {
            self.header_mut().routing_len = 0;
        }
        if ntype == NodeType::Blob {
            self.header_mut().num_ext_blobs = self.header().num_ext_blobs.saturating_add(1);
        }
        Ok(outcome)
    }

    fn alloc_node_inner(&mut self, ntype: NodeType) -> Result<AllocOutcome, AllocError> {
        if ntype == NodeType::Invalid {
            return Err(AllocError::InvalidRequest);
        }
        let size = size_of_node(ntype);
        let ntype_idx = free_list_class(ntype);

        // Try same-type free list first.
        let free_head = self.header().free_list_head[ntype_idx];
        if free_head != 0 {
            let e = self
                .slot_entry(free_head)
                .ok_or(AllocError::InvalidRequest)?;
            let next_free = e.next_free();
            self.header_mut().free_list_head[ntype_idx] = next_free;

            let off = e.byte_offset();
            self.write_slot_entry(free_head, SlotEntry::live(ntype, off));
            self.header_mut().gap_space = self.header().gap_space.wrapping_add(size);
            return Ok(AllocOutcome { slot: free_head });
        }

        // Same-size cross-type fallback for the 128-byte pair
        // (`Prefix` and `Blob`). The slot body has the right size;
        // we just re-tag the slot entry with the requested ntype.
        let sibling_idx_opt = match ntype {
            NodeType::Blob => Some((NodeType::Prefix.as_u8() - 1) as usize),
            NodeType::Prefix => Some((NodeType::Blob.as_u8() - 1) as usize),
            _ => None,
        };
        if let Some(sibling_idx) = sibling_idx_opt {
            let sibling_head = self.header().free_list_head[sibling_idx];
            if sibling_head != 0 {
                let e = self
                    .slot_entry(sibling_head)
                    .ok_or(AllocError::InvalidRequest)?;
                let next_free = e.next_free();
                self.header_mut().free_list_head[sibling_idx] = next_free;
                let off = e.byte_offset();
                self.write_slot_entry(sibling_head, SlotEntry::live(ntype, off));
                self.header_mut().gap_space = self.header().gap_space.wrapping_add(size);
                return Ok(AllocOutcome { slot: sibling_head });
            }
        }

        // Bump-allocate. The last `SPILLOVER_RESERVATION` bytes of
        // the data area are off-limits to every NodeType except
        // `Blob` — they exist precisely so spillover can install
        // its own BlobNode in a 99 %-full blob.
        let h = self.header();
        if h.num_slots >= MAX_SLOTS as u16 {
            return Err(AllocError::OutOfSlots);
        }
        let raw_avail = PAGE_SIZE.saturating_sub(h.space_used);
        let avail = if ntype == NodeType::Blob {
            raw_avail
        } else {
            raw_avail.saturating_sub(SPILLOVER_RESERVATION)
        };
        if avail < size {
            return Err(AllocError::OutOfSpace { need: size, avail });
        }
        let body_off = h.space_used;
        debug_assert!(body_off >= DATA_AREA_START);
        debug_assert!(body_off + size <= PAGE_SIZE);

        let new_slot = h.num_slots + 1; // 1-based
                                        // Write the slot table entry BEFORE bumping num_slots so
                                        // the slot is visible at slot[new_slot] when we then bump
                                        // num_slots in the header.
                                        // (The `write_slot_entry` debug_assert checks
                                        // `slot <= num_slots + 1` for exactly this case.)
        self.write_slot_entry(new_slot, SlotEntry::live(ntype, body_off));
        let h = self.header_mut();
        h.num_slots += 1;
        h.space_used += size;
        h.gap_space = h.gap_space.wrapping_add(size);

        Ok(AllocOutcome { slot: new_slot })
    }

    /// Push a slot onto its NodeType's free list. The body bytes
    /// remain in place (they'll be overwritten on the next alloc
    /// that reuses this slot).
    ///
    /// **v4 abandon-on-free**: the walker no longer calls `free_node` —
    /// structural ops (node grow/shrink/collapse, leaf realloc, prefix
    /// split, EmptyRoot replacement) abandon the old node instead
    /// (leaving it unreachable, reclaimed at the next compaction) and
    /// bump `header.dead_bytes`. The per-NodeType free lists are
    /// therefore no longer populated on the hot path; `alloc_node`'s
    /// free-list-reuse branches are dormant (the bump path always
    /// fires). This method is retained for tests / diagnostics and any
    /// future explicit-reclaim path.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn free_node(&mut self, slot: u16) -> Result<(), FreeError> {
        let e = self.slot_entry(slot).ok_or(FreeError::InvalidSlot(slot))?;
        let ntype = e.node_type().ok_or(FreeError::TypeMismatch {
            slot,
            tag: e.ntype_or_next_free,
        })?;
        self.free_node_inner(slot)?;
        if ntype == NodeType::Blob {
            self.header_mut().num_ext_blobs = self.header().num_ext_blobs.saturating_sub(1);
        }
        Ok(())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn free_node_inner(&mut self, slot: u16) -> Result<(), FreeError> {
        let e = self.slot_entry(slot).ok_or(FreeError::InvalidSlot(slot))?;
        let ntype = e.node_type().ok_or(FreeError::TypeMismatch {
            slot,
            tag: e.ntype_or_next_free,
        })?;
        if ntype == NodeType::Invalid {
            return Err(FreeError::TypeMismatch {
                slot,
                tag: e.ntype_or_next_free,
            });
        }
        let ntype_idx = free_list_class(ntype);
        let old_head = self.header().free_list_head[ntype_idx];
        let off = e.byte_offset();
        self.write_slot_entry(slot, SlotEntry::freed(old_head, off));
        self.header_mut().free_list_head[ntype_idx] = slot;
        Ok(())
    }

    /// Bump-allocate a single variable-size leaf node and register it
    /// in the slot table as `NodeType::Leaf`.
    ///
    /// `total_aligned` is the full, 8-byte-aligned body size of the
    /// self-describing leaf (`[16B header][key][value]`), as computed
    /// by [`crate::layout::leaf_body_size`]. Mirrors the bump branch
    /// of [`Self::alloc_node`] but with a runtime size.
    ///
    /// CRITICAL: like the old extent allocator, this honours the
    /// [`SPILLOVER_RESERVATION`] — leaves are the dominant non-Blob
    /// bump-area consumer, and a 99 %-full data area is exactly where
    /// spillover needs to install its emergency `BlobNode`. Only
    /// `alloc_node(NodeType::Blob)` may consume the last
    /// `SPILLOVER_RESERVATION` bytes. `gap_space` is bumped by the
    /// allocated size, consistent with `alloc_node`.
    pub fn alloc_leaf(&mut self, total_aligned: u32) -> Result<AllocOutcome, AllocError> {
        if total_aligned == 0 {
            return Err(AllocError::InvalidRequest);
        }
        debug_assert_eq!(total_aligned & 7, 0, "alloc_leaf size must be 8-aligned");
        let h = self.header();
        if h.num_slots >= MAX_SLOTS as u16 {
            return Err(AllocError::OutOfSlots);
        }
        // Leaves are non-Blob: refuse the last SPILLOVER_RESERVATION
        // bytes of the data area (copied from the old `alloc_extent`
        // avail computation).
        let avail = PAGE_SIZE
            .saturating_sub(h.space_used)
            .saturating_sub(SPILLOVER_RESERVATION);
        if avail < total_aligned {
            return Err(AllocError::OutOfSpace {
                need: total_aligned,
                avail,
            });
        }
        let body_off = h.space_used;
        debug_assert!(body_off >= DATA_AREA_START);
        debug_assert!(body_off + total_aligned <= PAGE_SIZE);

        let new_slot = h.num_slots + 1; // 1-based
        self.write_slot_entry(new_slot, SlotEntry::live(NodeType::Leaf, body_off));
        let h = self.header_mut();
        h.num_slots += 1;
        h.space_used += total_aligned;
        h.gap_space = h.gap_space.wrapping_add(total_aligned);
        // De-route on an in-place leaf alloc into a routed blob. A new (or
        // value-grown-relocated) leaf keeps the routing *invariant* intact
        // — leaves are correctly classified above `leaf_region_start` — but
        // linking it rewrites the parent node's child pointer, which lives
        // in the routing region, and a value-grow also repoints the parent
        // at the relocated leaf. That makes any indexed-read navigation page
        // captured from the old routing region stale. De-routing
        // (`routing_len = 0`) takes the blob off the cold routed path until
        // the next compaction re-routes it; indexed reads fall back to the
        // authoritative full pin meanwhile. The routing build uses
        // `alloc_leaf_at` (not this path) and runs with `routing_len == 0`,
        // so it is unaffected.
        if h.routing_len != 0 {
            h.routing_len = 0;
        }

        Ok(AllocOutcome { slot: new_slot })
    }

    /// Place a leaf body at the caller-supplied `body_off` and register
    /// its slot, **without** advancing `header.space_used`.
    ///
    /// This is the leaf-arena allocator for the routing-aware compaction
    /// build (`compact_blob`): internal nodes keep bumping `space_used`
    /// (the routing arena, growing up from `DATA_AREA_START`) while
    /// leaves are placed at a separate page-aligned cursor the caller
    /// owns. The caller stamps `space_used` to the final leaf-cursor
    /// high-water at finalize, so subsequent in-place `alloc_leaf`
    /// appends land above the leaf arena.
    ///
    /// Mirrors [`Self::alloc_leaf`]'s bump branch except the body offset
    /// is supplied and `space_used` is left untouched. The free-list
    /// reuse branch is intentionally omitted: a freshly `init`-ed
    /// compaction destination has empty free lists, and reusing a
    /// routing-arena slot for a leaf would break arena segregation.
    pub fn alloc_leaf_at(
        &mut self,
        body_off: u32,
        total_aligned: u32,
    ) -> Result<AllocOutcome, AllocError> {
        if total_aligned == 0 {
            return Err(AllocError::InvalidRequest);
        }
        debug_assert_eq!(total_aligned & 7, 0, "alloc_leaf_at size must be 8-aligned");
        debug_assert!(
            body_off >= DATA_AREA_START,
            "alloc_leaf_at body_off below data area: {body_off:#x}"
        );
        debug_assert_eq!(
            body_off & 7,
            0,
            "alloc_leaf_at body_off must be 8-aligned: {body_off:#x}"
        );
        let h = self.header();
        if h.num_slots >= MAX_SLOTS as u16 {
            return Err(AllocError::OutOfSlots);
        }
        // The leaf arena lives just below PAGE_SIZE; reserve the
        // spillover tail exactly as `alloc_leaf` does (leaves are
        // non-Blob). Phrased to avoid `body_off + total_aligned`
        // overflow.
        let ceiling = PAGE_SIZE - SPILLOVER_RESERVATION;
        if body_off > ceiling || total_aligned > ceiling - body_off {
            return Err(AllocError::OutOfSpace {
                need: total_aligned,
                avail: ceiling.saturating_sub(body_off),
            });
        }
        let new_slot = h.num_slots + 1; // 1-based
        self.write_slot_entry(new_slot, SlotEntry::live(NodeType::Leaf, body_off));
        let h = self.header_mut();
        h.num_slots += 1;
        h.gap_space = h.gap_space.wrapping_add(total_aligned);
        // NB: `space_used` intentionally NOT advanced — the caller owns
        // the leaf cursor and stamps the final high-water at finalize.
        Ok(AllocOutcome { slot: new_slot })
    }

    /// Mutable raw byte view at an arbitrary offset.
    ///
    /// Used by the leaf write path to populate a freshly-allocated
    /// leaf body (`[16B header][key][value]`) *before* its header is
    /// written. While the header's `key_len`/`value_len` are still
    /// zero, `body_of_slot` would size the leaf as the bare 16-byte
    /// header, so the writer must address the allocated region by its
    /// byte offset instead. Returns `None` if `offset + len` would run
    /// past `PAGE_SIZE`.
    pub fn bytes_at_mut(&mut self, offset: u32, len: u32) -> Option<&mut [u8]> {
        let o = offset as usize;
        let l = len as usize;
        let end = o.checked_add(l)?;
        if end > self.buf.len() {
            return None;
        }
        Some(&mut self.buf[o..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> Vec<u8> {
        vec![0u8; PAGE_SIZE as usize]
    }

    #[test]
    fn persisted_extent_covers_the_used_data_area_in_whole_pages() {
        let mut buf = fresh();
        // A zeroed header still persists the header and slot table.
        assert_eq!(persisted_extent(&buf), DATA_AREA_START as usize);
        for (space_used, extent) in [
            (DATA_AREA_START, DATA_AREA_START),
            (DATA_AREA_START + 1, DATA_AREA_START + PAGE_4K),
            (DATA_AREA_START + PAGE_4K, DATA_AREA_START + PAGE_4K),
            (PAGE_SIZE - 1, PAGE_SIZE),
            (PAGE_SIZE, PAGE_SIZE),
            (u32::MAX, PAGE_SIZE),
        ] {
            BlobFrame::wrap(&mut buf).header_mut().space_used = space_used;
            assert_eq!(persisted_extent(&buf), extent as usize, "{space_used:#x}");
        }

        BlobFrame::wrap(&mut buf).header_mut().space_used = DATA_AREA_START + 1;
        buf[DATA_AREA_START as usize..].fill(0x5a);
        clear_unpersisted_tail(&mut buf);
        let extent = (DATA_AREA_START + PAGE_4K) as usize;
        assert!(buf[DATA_AREA_START as usize..extent]
            .iter()
            .all(|&byte| byte == 0x5a));
        assert!(buf[extent..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn init_seeds_empty_root_sentinel() {
        let mut buf = fresh();
        let frame = BlobFrame::init(&mut buf, [0xAB; 16]).unwrap();
        let h = frame.header();
        assert_eq!(h.blob_guid, [0xAB; 16]);
        assert_eq!(h.num_slots, 1);
        // `root_slot` now stores the encoded root offset; the sentinel
        // sits at DATA_AREA_START, which encodes to 1.
        assert_eq!(h.root_slot, encode_child_off(DATA_AREA_START));
        assert_eq!(h.root_slot, 1);
        // After allocating the 8-byte sentinel, space_used is
        // DATA_AREA_START + 8.
        assert_eq!(h.space_used, DATA_AREA_START + 8);

        // The sentinel is self-describing via its `node_type @ +1`
        // byte and resolves to EmptyRoot by offset (v4 addressing) and
        // by slot (bookkeeping).
        let root_off = decode_child_off(h.root_slot);
        assert_eq!(frame.ntype_at(root_off), Some(NodeType::EmptyRoot));
        let e = frame.slot_entry(1).unwrap();
        assert_eq!(e.node_type(), Some(NodeType::EmptyRoot));
        let body = frame.body_of_slot(1).unwrap();
        assert_eq!(body.len(), 8);
        // Only the node_type byte at +1 is set; the rest stay zero.
        assert_eq!(body[1], NodeType::EmptyRoot.as_u8());
        assert!(body.iter().enumerate().all(|(i, b)| i == 1 || *b == 0));
    }

    #[test]
    fn alloc_node_bumps_space_used() {
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        let space_before = frame.header().space_used;
        let out = frame.alloc_node(NodeType::Node4).unwrap();
        assert_eq!(out.slot, 2);
        assert_eq!(frame.header().space_used, space_before + 16);
        assert_eq!(frame.header().num_slots, 2);

        // The slot entry decodes back to Node4 with the right offset.
        let e = frame.slot_entry(2).unwrap();
        assert_eq!(e.node_type(), Some(NodeType::Node4));
        assert_eq!(e.byte_offset(), space_before);
    }

    #[test]
    fn free_then_realloc_reuses_slot() {
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        // Leaves are variable-size, self-describing nodes allocated
        // via `alloc_leaf`; freed leaf slots park on their own
        // class-0 free list and are reclaimed by compaction. A freed
        // slot index is still reused LIFO by `alloc_node(Leaf)`.
        let a = frame.alloc_leaf(24).unwrap();
        let b = frame.alloc_leaf(24).unwrap();
        let c = frame.alloc_leaf(24).unwrap();
        assert_eq!(a.slot, 2);
        assert_eq!(b.slot, 3);
        assert_eq!(c.slot, 4);

        frame.free_node(b.slot).unwrap();
        // Re-alloc — should pop slot 3 (LIFO) off the Leaf free list.
        let r = frame.alloc_node(NodeType::Leaf).unwrap();
        assert_eq!(r.slot, 3);
    }

    #[test]
    fn num_ext_blobs_tracks_live_blob_nodes() {
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        assert_eq!(frame.header().num_ext_blobs, 0);

        let blob = frame.alloc_node(NodeType::Blob).unwrap();
        assert_eq!(frame.header().num_ext_blobs, 1);
        frame.free_node(blob.slot).unwrap();
        assert_eq!(frame.header().num_ext_blobs, 0);

        let prefix = frame.alloc_node(NodeType::Prefix).unwrap();
        assert_eq!(frame.header().num_ext_blobs, 0);
        frame.free_node(prefix.slot).unwrap();
        assert_eq!(frame.header().num_ext_blobs, 0);

        let blob_from_prefix_slot = frame.alloc_node(NodeType::Blob).unwrap();
        assert_eq!(frame.header().num_ext_blobs, 1);
        frame.free_node(blob_from_prefix_slot.slot).unwrap();
        assert_eq!(frame.header().num_ext_blobs, 0);
    }

    #[test]
    fn alloc_leaf_registers_slot_and_is_self_describing() {
        use crate::layout::{leaf_body_size, Leaf};
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        let space_before = frame.header().space_used;
        let slots_before = frame.header().num_slots;

        // A leaf with a 5-byte key + 7-byte value: total = align8(28) = 32.
        let total = leaf_body_size(5, 7);
        assert_eq!(total, 32);
        let out = frame.alloc_leaf(total).unwrap();
        assert_eq!(out.slot, slots_before + 1);
        assert_eq!(frame.header().num_slots, slots_before + 1);
        assert_eq!(frame.header().space_used, space_before + total);

        let e = frame.slot_entry(out.slot).unwrap();
        assert_eq!(e.node_type(), Some(NodeType::Leaf));
        assert_eq!(e.byte_offset(), space_before);
        let body_off = e.byte_offset();

        // Before the header is written it is all-zero, so the leaf is
        // self-described as the bare 16-byte header.
        assert_eq!(frame.body_of_slot(out.slot).unwrap().len(), 16);

        // Write the header via the raw byte offset, then confirm
        // body_of_slot now resolves the full variable size.
        let leaf = Leaf::live(5, 7, 99, 0xAB);
        {
            let body = frame.bytes_at_mut(body_off, total).unwrap();
            assert_eq!(body.len(), total as usize);
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    std::ptr::from_ref::<Leaf>(&leaf).cast::<u8>(),
                    std::mem::size_of::<Leaf>(),
                )
            };
            body[..16].copy_from_slice(bytes);
        }
        assert_eq!(frame.body_of_slot(out.slot).unwrap().len(), total as usize);
    }

    #[test]
    fn out_of_space_at_data_area_boundary() {
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        // Drain the data area with leaves. alloc_leaf respects the
        // SPILLOVER_RESERVATION, so the last 128 bytes of the data
        // area are off-limits.
        let remaining = PAGE_SIZE - frame.header().space_used - SPILLOVER_RESERVATION;
        frame.alloc_leaf(remaining - 8).unwrap();
        frame.alloc_leaf(8).unwrap();
        assert!(matches!(
            frame.alloc_leaf(8),
            Err(AllocError::OutOfSpace { .. })
        ));

        // The last 128 bytes are still reachable via
        // `alloc_node(NodeType::Blob)` — that's the spillover
        // emergency path. We don't need to verify the size here;
        // SIZE_BY_TYPE[NodeType::Blob] is compile-time-asserted to
        // 128 in `layout::mod.rs`.
        let _bn = frame.alloc_node(NodeType::Blob).unwrap();
    }
}
