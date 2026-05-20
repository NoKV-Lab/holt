//! `BlobFrame` — typed view over one 512 KB blob.
//!
//! Provides:
//!   - `init`: format a fresh buffer (writes header, seeds the
//!     empty_root sentinel at slot 1)
//!   - `alloc_node(ntype)`: try free-list first, else bump
//!   - `free_node(slot)`: push onto per-NodeType LIFO
//!   - `alloc_extent(size)`: raw bump for Leaf key/value bytes
//!   - `body_of_slot(slot)`: resolve a slot to its body slice
//!
//! All operations enforce the on-disk invariants: 8-byte body
//! alignment, MAX_SLOTS cap, and slot-entry bit-packing.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::layout::{
    size_of_node, BlobGuid, BlobHeader, NodeType, SlotEntry, SlotEntryRaw, DATA_AREA_START,
    HEADER_SIZE, MAX_SLOTS, PAGE_SIZE,
};

/// Bytes the bump allocator reserves for spillover's
/// emergency `BlobNode` install. Walker `alloc_node` (non-Blob) +
/// `alloc_extent` refuse to consume the last `SPILLOVER_RESERVATION`
/// bytes; `alloc_node(NodeType::Blob)` is exempt and may consume
/// them. Without this, a 99 %-full blob has no room left for the
/// spillover code path to install its own placeholder BlobNode.
///
/// Equals one `BlobNode` body — the only structure spillover ever
/// emits.
pub const SPILLOVER_RESERVATION: u32 = 128;

/// Errors from `alloc_node` / `alloc_extent`.
#[derive(Debug, PartialEq, Eq)]
pub enum AllocError {
    /// `num_slots` reached `MAX_SLOTS` — no more slot indices.
    OutOfSlots,
    /// Data area can't fit the requested size.
    OutOfSpace {
        /// Bytes the caller requested (8-byte aligned for extents).
        need: u32,
        /// Bytes actually free in the data area.
        avail: u32,
    },
    /// `alloc_node(NodeType::Invalid)` or `alloc_extent(0)`.
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

/// Outcome of a raw-extent alloc (for Leaf key/value bytes).
#[derive(Debug, Clone, Copy)]
pub struct ExtentAllocOutcome {
    /// Byte offset of the extent within the blob.
    pub byte_offset: u32,
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

    /// Read a 1-based slot table entry. `None` if out of range.
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

    /// Resolve a slot to a const view of its body bytes.
    #[must_use]
    pub fn body_of_slot(&self, slot: u16) -> Option<&'a [u8]> {
        let e = self.slot_entry(slot)?;
        let ntype = e.node_type()?;
        if ntype == NodeType::Invalid {
            return None;
        }
        let off = e.byte_offset() as usize;
        let size = size_of_node(ntype) as usize;
        if off + size > self.buf.len() {
            return None;
        }
        Some(&self.buf[off..off + size])
    }

    /// Raw byte view at an arbitrary offset (Leaf extents).
    #[must_use]
    pub fn bytes_at(&self, offset: u32, len: u32) -> Option<&'a [u8]> {
        let o = offset as usize;
        let l = len as usize;
        let end = o.checked_add(l)?;
        if end > self.buf.len() {
            return None;
        }
        Some(&self.buf[o..end])
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
/// # Slot version tracking
///
/// Two constructors:
///
/// - [`Self::wrap`] — no version tracking. Used by init paths
///   (a fresh blob being constructed before it's installed in
///   the cache) and tests.
/// - [`Self::wrap_versioned`] — wires up the owning
///   [`crate::store::CachedBlob`]'s per-slot version counters; every
///   slot mutation (alloc / free / body rewrite via
///   `write_struct_to_slot`) bumps the corresponding counter
///   with `Release` ordering. The walker's write paths construct
///   this variant via [`crate::store::BlobWriteGuard::frame`] so observers
///   reading the slot after the latch release can detect the
///   mutation. Foundation for cross-blob lock-coupling (queued
///   Phases 2-3 of the per-node-latch milestone).
pub struct BlobFrame<'a> {
    /// Backing buffer (must be exactly `PAGE_SIZE` bytes).
    buf: &'a mut [u8],
    /// Per-slot version counters from the owning [`crate::store::CachedBlob`],
    /// or `None` for init / local-buf paths that don't yet have
    /// a cache entry. When `Some`, every slot mutation routed
    /// through this frame bumps the corresponding counter.
    slot_versions: Option<&'a [AtomicU64]>,
}

impl<'a> BlobFrame<'a> {
    /// Wrap an existing buffer that's already been formatted —
    /// **without** per-slot version tracking.
    ///
    /// Use for init paths (fresh blob being constructed before
    /// it's installed in the cache) and for tests / ad-hoc
    /// callers that don't need observers to detect mutations
    /// across latch releases.
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
        Self {
            buf,
            slot_versions: None,
        }
    }

    /// Wrap an existing buffer **with** per-slot version tracking.
    ///
    /// `slot_versions` is the owning [`crate::store::CachedBlob`]'s
    /// per-slot counter array — typically obtained via
    /// [`crate::store::BlobWriteGuard::frame`] which pulls the slice off
    /// the cached blob and pairs it with the write guard's mutable
    /// buffer in one call.
    ///
    /// Every subsequent slot mutation through this frame bumps the
    /// corresponding counter with `Release` ordering. The bump
    /// happens **after** the mutation completes, so any observer
    /// that performs an `Acquire` load of the counter and then
    /// reads the slot's body either sees both the old body + old
    /// counter or both the new body + new counter — the bumps act
    /// as the read-after-validate fence for optimistic
    /// concurrency.
    pub fn wrap_versioned(buf: &'a mut [u8], slot_versions: &'a [AtomicU64]) -> Self {
        assert_eq!(
            buf.len(),
            PAGE_SIZE as usize,
            "BlobFrame requires PAGE_SIZE buffer"
        );
        Self {
            buf,
            slot_versions: Some(slot_versions),
        }
    }

    /// Bump the per-slot version counter for the given 1-based
    /// slot index. No-op if this frame was constructed via
    /// [`Self::wrap`] (no version tracking) or if the slot index
    /// is out of range.
    ///
    /// Uses `Release` ordering so the bump is ordered after any
    /// prior writes to the slot's body, matching the `Acquire`
    /// load on the observer side.
    ///
    /// Called from the centralised slot-write paths (this module's
    /// `alloc_node` / `free_node`; the walker's
    /// `writers::write_struct_to_slot`); usually the caller
    /// doesn't invoke this directly.
    pub(crate) fn bump_slot_version(&self, slot: u16) {
        let Some(svs) = self.slot_versions else {
            return;
        };
        if slot == 0 {
            return;
        }
        if let Some(av) = svs.get((slot as usize) - 1) {
            av.fetch_add(1, Ordering::Release);
        }
    }

    /// Cheap conversion to a read-only [`BlobFrameRef`]. Useful
    /// for forwarding into walker `lookup` / `descend` paths
    /// (which take `BlobFrameRef` so they also work against
    /// `RwLock`-guarded `BufferManager` slices).
    #[must_use]
    pub fn as_ref(&self) -> BlobFrameRef<'_> {
        BlobFrameRef { buf: self.buf }
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
        let mut frame = Self {
            buf,
            slot_versions: None,
        };
        // Seed the header.
        {
            let h = frame.header_mut();
            h.blob_guid = guid;
            h.space_used = DATA_AREA_START;
        }
        // Allocate the empty-tree root sentinel.
        let out = frame.alloc_node(NodeType::EmptyRoot)?;
        debug_assert_eq!(out.slot, 1);
        // Body is already zero (we memset the whole buffer above);
        // record it as the tree's root.
        frame.header_mut().root_slot = out.slot;
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
    #[must_use]
    pub fn body_of_slot(&self, slot: u16) -> Option<&[u8]> {
        let e = self.slot_entry(slot)?;
        let ntype = e.node_type()?;
        if ntype == NodeType::Invalid {
            return None;
        }
        let off = e.byte_offset() as usize;
        let size = size_of_node(ntype) as usize;
        if off + size > self.buf.len() {
            return None;
        }
        Some(&self.buf[off..off + size])
    }

    /// Mutable view of a slot's body bytes.
    pub fn body_of_slot_mut(&mut self, slot: u16) -> Option<&mut [u8]> {
        let e = self.slot_entry(slot)?;
        let ntype = e.node_type()?;
        if ntype == NodeType::Invalid {
            return None;
        }
        let off = e.byte_offset() as usize;
        let size = size_of_node(ntype) as usize;
        if off + size > self.buf.len() {
            return None;
        }
        Some(&mut self.buf[off..off + size])
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
    /// On success, the allocated slot's per-slot version counter
    /// is bumped (see [`Self::wrap_versioned`]). The bump signals
    /// "this slot's identity changed" — important for observers
    /// who captured a version before the latch was acquired.
    pub fn alloc_node(&mut self, ntype: NodeType) -> Result<AllocOutcome, AllocError> {
        let outcome = self.alloc_node_inner(ntype)?;
        self.bump_slot_version(outcome.slot);
        Ok(outcome)
    }

    fn alloc_node_inner(&mut self, ntype: NodeType) -> Result<AllocOutcome, AllocError> {
        if ntype == NodeType::Invalid {
            return Err(AllocError::InvalidRequest);
        }
        let size = size_of_node(ntype);
        let ntype_idx = (ntype.as_u8() - 1) as usize;

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
    /// On success, the freed slot's per-slot version counter is
    /// bumped (see [`Self::wrap_versioned`]) — observers who
    /// captured a version before the latch was acquired will see
    /// the mismatch on re-validate.
    pub fn free_node(&mut self, slot: u16) -> Result<(), FreeError> {
        self.free_node_inner(slot)?;
        self.bump_slot_version(slot);
        Ok(())
    }

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
        let ntype_idx = (ntype.as_u8() - 1) as usize;
        let old_head = self.header().free_list_head[ntype_idx];
        let off = e.byte_offset();
        self.write_slot_entry(slot, SlotEntry::freed(old_head, off));
        self.header_mut().free_list_head[ntype_idx] = slot;
        Ok(())
    }

    /// Bump-allocate a raw byte extent (NOT a node — does not
    /// touch the slot table). Used for Leaf key/value bytes.
    /// Increments `space_used` by the 8-byte-aligned size.
    pub fn alloc_extent(&mut self, size: u32) -> Result<ExtentAllocOutcome, AllocError> {
        if size == 0 {
            return Err(AllocError::InvalidRequest);
        }
        let aligned = (size + 7) & !7;
        let h = self.header();
        // Honour the spillover reservation (see [`SPILLOVER_RESERVATION`]):
        // leaf extents are the dominant bump-area consumer, and a 99 %-
        // full extent area is exactly where spillover needs to install
        // its emergency BlobNode.
        let avail = PAGE_SIZE
            .saturating_sub(h.space_used)
            .saturating_sub(SPILLOVER_RESERVATION);
        if avail < aligned {
            return Err(AllocError::OutOfSpace {
                need: aligned,
                avail,
            });
        }
        let off = h.space_used;
        self.header_mut().space_used += aligned;
        Ok(ExtentAllocOutcome { byte_offset: off })
    }

    /// Raw byte view at an arbitrary offset.
    ///
    /// Used to read Leaf extents (key/value bytes) which live in
    /// the data area but are not registered in the slot table.
    /// Returns `None` if `offset + len` would run past `PAGE_SIZE`.
    #[must_use]
    pub fn bytes_at(&self, offset: u32, len: u32) -> Option<&[u8]> {
        let o = offset as usize;
        let l = len as usize;
        let end = o.checked_add(l)?;
        if end > self.buf.len() {
            return None;
        }
        Some(&self.buf[o..end])
    }

    /// Mutable view at an arbitrary offset.
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
    fn init_seeds_empty_root_sentinel() {
        let mut buf = fresh();
        let frame = BlobFrame::init(&mut buf, [0xAB; 16]).unwrap();
        let h = frame.header();
        assert_eq!(h.blob_guid, [0xAB; 16]);
        assert_eq!(h.num_slots, 1);
        assert_eq!(h.root_slot, 1);
        // After allocating the 8-byte sentinel, space_used is
        // DATA_AREA_START + 8.
        assert_eq!(h.space_used, DATA_AREA_START + 8);

        // The root slot is tagged empty_root and its body is all zero.
        let e = frame.slot_entry(1).unwrap();
        assert_eq!(e.node_type(), Some(NodeType::EmptyRoot));
        let body = frame.body_of_slot(1).unwrap();
        assert_eq!(body.len(), 8);
        assert!(body.iter().all(|b| *b == 0));
    }

    #[test]
    fn alloc_node_bumps_space_used() {
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        let space_before = frame.header().space_used;
        let out = frame.alloc_node(NodeType::Node4).unwrap();
        assert_eq!(out.slot, 2);
        assert_eq!(frame.header().space_used, space_before + 24);
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
        let a = frame.alloc_node(NodeType::Leaf).unwrap();
        let b = frame.alloc_node(NodeType::Leaf).unwrap();
        let c = frame.alloc_node(NodeType::Leaf).unwrap();
        assert_eq!(a.slot, 2);
        assert_eq!(b.slot, 3);
        assert_eq!(c.slot, 4);

        frame.free_node(b.slot).unwrap();
        // Re-alloc — should pop slot 3 (LIFO).
        let r = frame.alloc_node(NodeType::Leaf).unwrap();
        assert_eq!(r.slot, 3);
    }

    #[test]
    fn alloc_extent_does_not_touch_slot_table() {
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        let space_before = frame.header().space_used;
        let slots_before = frame.header().num_slots;
        let e = frame.alloc_extent(13).unwrap();
        // 13 → padded to 16.
        assert_eq!(e.byte_offset, space_before);
        assert_eq!(frame.header().space_used, space_before + 16);
        // Slot table untouched.
        assert_eq!(frame.header().num_slots, slots_before);
    }

    #[test]
    fn out_of_space_at_data_area_boundary() {
        let mut buf = fresh();
        let mut frame = BlobFrame::init(&mut buf, [0; 16]).unwrap();
        // Drain the data area with one giant extent. alloc_extent
        // respects the SPILLOVER_RESERVATION, so the last 128 bytes
        // of the data area are off-limits.
        let remaining = PAGE_SIZE - frame.header().space_used - SPILLOVER_RESERVATION;
        frame.alloc_extent(remaining - 8).unwrap();
        frame.alloc_extent(8).unwrap();
        assert!(matches!(
            frame.alloc_extent(8),
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
