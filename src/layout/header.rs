//! `BlobHeader` — the 4096-byte fixed header at the start of every
//! 512 KB blob frame.
//!
//! Field offsets are pinned at compile time via
//! `const _: () = assert!(...)` blocks. If a field is ever moved,
//! the build fails — preventing accidental on-disk format drift
//! across releases.

use std::mem::{offset_of, size_of};

/// Total bytes per blob frame. The whole engine assumes 524288.
pub const PAGE_SIZE: u32 = 0x80000;

/// Header reserves the first 4096 bytes. Slot table starts at +0x1000.
pub const HEADER_SIZE: u32 = 0x1000;

/// Hard cap on slots in one blob.
pub const MAX_SLOTS: u32 = 0x2800;

/// Slot table size = MAX_SLOTS × `sizeof(u32)` = 40 KB.
pub const SLOT_TABLE_SIZE: u32 = MAX_SLOTS * 4;

/// Data area starts after the header + slot table.
pub const DATA_AREA_START: u32 = HEADER_SIZE + SLOT_TABLE_SIZE;

/// Bytes available for node bodies + key/value extents.
pub const DATA_AREA_CAPACITY: u32 = PAGE_SIZE - DATA_AREA_START;

const _: () = assert!(DATA_AREA_START == 0xB000);

/// 128-bit blob identifier (stored as 16 bytes).
pub type BlobGuid = [u8; 16];

/// On-disk header for a 512 KB blob frame.
///
/// 4096 bytes fixed. Field positions are chosen for natural
/// alignment + room to grow without breaking compatibility:
/// counter fields are clustered near the front, the per-NodeType
/// free list lives at +0x70, the blob GUID at +0xa0, and the
/// remainder is reserved for future metadata.
///
/// Padding bytes (`_pad_NN`) are reserved space; future versions
/// may name them without moving any existing field's offset.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlobHeader {
    _pad_0: [u8; 0x50],
    /// Reserved counter slot.
    pub field_50: u16,
    /// Reserved counter slot.
    pub field_52: u16,
    /// Slot-table high-water mark — new slots allocated at this index.
    pub num_slots: u16,
    /// Slot index pointing at the tree root. `0` is never valid;
    /// initial value is 1 (the empty-tree sentinel allocated on init).
    pub root_slot: u16,
    /// Absolute byte offset of the next free byte in the data area.
    /// Starts at `DATA_AREA_START` (= 0xB000) after init.
    pub space_used: u32,
    /// Count of external-blob references currently held.
    pub num_ext_blobs: u16,
    /// Reserved counter slot.
    pub field_5e: u16,
    _pad_60: [u8; 0x08],
    /// Cumulative count of size-table bytes ever allocated for
    /// nodes (used to drive compaction triggers).
    pub gap_space: u32,
    _pad_6c: u32,
    /// Per-NodeType free-list head. Index 0 = ntype 1 (Leaf),
    /// index 1 = ntype 2 (Prefix), …, index 7 = ntype 8 (EmptyRoot).
    /// Value `0` means the list is empty.
    pub free_list_head: [u16; 8],
    _pad_80: [u8; 0x20],
    /// 128-bit blob identifier.
    pub blob_guid: BlobGuid,
    _pad_b0: [u8; (HEADER_SIZE as usize) - 0xb0],
}

// Pin every field offset at compile time. Drift breaks the build.
const _: () = assert!(size_of::<BlobHeader>() == HEADER_SIZE as usize);
const _: () = assert!(offset_of!(BlobHeader, field_50) == 0x50);
const _: () = assert!(offset_of!(BlobHeader, num_slots) == 0x54);
const _: () = assert!(offset_of!(BlobHeader, root_slot) == 0x56);
const _: () = assert!(offset_of!(BlobHeader, space_used) == 0x58);
const _: () = assert!(offset_of!(BlobHeader, num_ext_blobs) == 0x5c);
const _: () = assert!(offset_of!(BlobHeader, gap_space) == 0x68);
const _: () = assert!(offset_of!(BlobHeader, free_list_head) == 0x70);
const _: () = assert!(offset_of!(BlobHeader, blob_guid) == 0xa0);

impl BlobHeader {
    /// All-zeros header; callers fill in `blob_guid` and the bump
    /// cursor before use.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            _pad_0: [0; 0x50],
            field_50: 0,
            field_52: 0,
            num_slots: 0,
            root_slot: 0,
            space_used: 0,
            num_ext_blobs: 0,
            field_5e: 0,
            _pad_60: [0; 0x08],
            gap_space: 0,
            _pad_6c: 0,
            free_list_head: [0; 8],
            _pad_80: [0; 0x20],
            blob_guid: [0; 16],
            _pad_b0: [0; (HEADER_SIZE as usize) - 0xb0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_and_offsets() {
        assert_eq!(size_of::<BlobHeader>(), 4096);
        assert_eq!(offset_of!(BlobHeader, num_slots), 0x54);
        assert_eq!(offset_of!(BlobHeader, root_slot), 0x56);
        assert_eq!(offset_of!(BlobHeader, space_used), 0x58);
        assert_eq!(offset_of!(BlobHeader, gap_space), 0x68);
        assert_eq!(offset_of!(BlobHeader, free_list_head), 0x70);
        assert_eq!(offset_of!(BlobHeader, blob_guid), 0xa0);
    }

    #[test]
    fn constants_consistent() {
        assert_eq!(PAGE_SIZE, 524_288);
        assert_eq!(HEADER_SIZE, 4096);
        assert_eq!(MAX_SLOTS, 10_240);
        assert_eq!(DATA_AREA_START, 0xB000);
        assert_eq!(DATA_AREA_CAPACITY + DATA_AREA_START, PAGE_SIZE);
    }
}
