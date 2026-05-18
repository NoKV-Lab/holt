//! `Leaf` body (16 bytes) + key/value extent helper.
//!
//! Layout (`#[repr(C)]`):
//!
//! - `value_size: u16 @ +0`
//! - `tombstone:  u8  @ +2`
//! - `_pad:       u8  @ +3`
//! - `key_offset: u32 @ +4` — byte offset within the blob to a
//!   separately bump-allocated extent holding
//!   `(u16 key_len, key bytes, value bytes)`.
//! - `seq:        u64 @ +8`
//!
//! The 16-byte body is allocated as a node (registered in the
//! slot table); the extent is allocated separately via
//! `BlobFrame::alloc_extent` and is not registered in the slot
//! table.

use std::mem::{offset_of, size_of};

/// 16-byte Leaf body. The key/value bytes themselves live in a
/// separate bump-allocated extent in the same blob, addressed by
/// `key_offset`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leaf {
    /// Size in bytes of the value portion of the extent.
    pub value_size: u16,
    /// 0 = live leaf, 1 = tombstone (soft-deleted; pending
    /// reclaim via compactBlob).
    pub tombstone: u8,
    _pad: u8,
    /// Byte offset within the blob to the key/value extent. The
    /// extent layout is `u16 key_len ++ key_bytes ++ value_bytes`,
    /// 8-byte-aligned tail-padded.
    pub key_offset: u32,
    /// Monotonically-increasing seq counter — bumped on every
    /// write that touches this slot. Reused for MVCC snapshot
    /// isolation and journal-replay ordering.
    pub seq: u64,
}

const _: () = assert!(size_of::<Leaf>() == 16);
const _: () = assert!(offset_of!(Leaf, value_size) == 0);
const _: () = assert!(offset_of!(Leaf, tombstone) == 2);
const _: () = assert!(offset_of!(Leaf, key_offset) == 4);
const _: () = assert!(offset_of!(Leaf, seq) == 8);

impl Leaf {
    /// Construct a live (non-tombstone) leaf.
    #[must_use]
    pub const fn live(key_offset: u32, value_size: u16, seq: u64) -> Self {
        Self {
            value_size,
            tombstone: 0,
            _pad: 0,
            key_offset,
            seq,
        }
    }
}

/// Compute the 8-byte-aligned extent size needed for
/// `(u16 key_len + key.len() + value.len())`.
#[must_use]
pub const fn leaf_extent_size(key_len: u32, value_len: u32) -> u32 {
    let raw = 2 + key_len + value_len;
    (raw + 7) & !7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_size_alignment() {
        assert_eq!(leaf_extent_size(0, 0), 8);
        assert_eq!(leaf_extent_size(3, 3), 8); // 2+3+3=8
        assert_eq!(leaf_extent_size(4, 4), 16); // 2+4+4=10 → 16
        assert_eq!(leaf_extent_size(10, 4), 16); // 2+10+4=16
        assert_eq!(leaf_extent_size(10, 5), 24); // 2+10+5=17 → 24
        assert_eq!(leaf_extent_size(100, 200), (2 + 100 + 200 + 7) & !7);
    }
}
