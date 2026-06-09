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
    /// One-byte fingerprint of the full key (a non-zero hash). A
    /// point lookup compares it before touching the key/value
    /// extent, so ~255/256 of non-matching leaves are rejected
    /// without the second (extent) cache miss. `0` means "no
    /// fingerprint" (an older-format leaf) — the reader then always
    /// falls back to the full extent compare. Never a false
    /// negative: a mismatch only fires when the keys truly differ.
    pub key_fp: u8,
    /// Byte offset within the blob to the key/value extent. The
    /// extent layout is `u16 key_len ++ key_bytes ++ value_bytes`,
    /// 8-byte-aligned tail-padded.
    pub key_offset: u32,
    /// Monotonic record sequence, bumped on every write that
    /// touches this slot. Used for CAS tokens and WAL replay.
    pub seq: u64,
}

const _: () = assert!(size_of::<Leaf>() == 16);
const _: () = assert!(offset_of!(Leaf, value_size) == 0);
const _: () = assert!(offset_of!(Leaf, tombstone) == 2);
const _: () = assert!(offset_of!(Leaf, key_fp) == 3);
const _: () = assert!(offset_of!(Leaf, key_offset) == 4);
const _: () = assert!(offset_of!(Leaf, seq) == 8);

impl Leaf {
    /// Construct a live (non-tombstone) leaf. `key_fp` is the
    /// one-byte key fingerprint (non-zero) the lookup uses to skip
    /// the extent read on a mismatch; pass `0` to disable it.
    #[must_use]
    pub const fn live(key_offset: u32, value_size: u16, seq: u64, key_fp: u8) -> Self {
        Self {
            value_size,
            tombstone: 0,
            key_fp,
            key_offset,
            seq,
        }
    }
}

/// Maximum inline key+value bytes a [`LeafInline`] node holds.
/// Records with `key.len() + value.len() <= LEAF_INLINE_CAP` are
/// stored inline (no separate extent); larger records use [`Leaf`]
/// + a bump-allocated extent.
pub const LEAF_INLINE_CAP: usize = 44;

/// 56-byte leaf with the key+value bytes inlined into the node body
/// (no separate extent). Same size as `Node16`, so it shares
/// `Node16`'s free-list size class. Eliminates the extent
/// allocation and the second cache miss that the extent-based
/// [`Leaf`] pays on small records — the common shape for object /
/// filesystem metadata (existence markers, ref counts, version
/// tokens, short dentries, small inode fields).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LeafInline {
    /// Monotonic record sequence (CAS token / WAL replay), mirrors
    /// [`Leaf::seq`].
    pub seq: u64,
    /// 0 = live, 1 = tombstone (soft-deleted; pending compaction).
    pub tombstone: u8,
    /// Length of the inline key (`<= LEAF_INLINE_CAP`).
    pub key_len: u8,
    /// Length of the inline value (`key_len + value_len <= CAP`).
    pub value_len: u8,
    _pad: u8,
    /// Inline bytes: `key ++ value`, only the first
    /// `key_len + value_len` are valid.
    pub bytes: [u8; LEAF_INLINE_CAP],
}

const _: () = assert!(size_of::<LeafInline>() == 56);
const _: () = assert!(offset_of!(LeafInline, seq) == 0);
const _: () = assert!(offset_of!(LeafInline, tombstone) == 8);
const _: () = assert!(offset_of!(LeafInline, key_len) == 9);
const _: () = assert!(offset_of!(LeafInline, value_len) == 10);
const _: () = assert!(offset_of!(LeafInline, bytes) == 12);

/// `true` iff a `(key, value)` pair fits in a [`LeafInline`] body.
#[must_use]
pub const fn fits_leaf_inline(key_len: usize, value_len: usize) -> bool {
    key_len <= u8::MAX as usize
        && value_len <= u8::MAX as usize
        && key_len + value_len <= LEAF_INLINE_CAP
}

impl LeafInline {
    /// Build a live inline leaf. Caller must have checked
    /// [`fits_leaf_inline`].
    #[must_use]
    pub fn live(key: &[u8], value: &[u8], seq: u64) -> Self {
        debug_assert!(fits_leaf_inline(key.len(), value.len()));
        let mut bytes = [0u8; LEAF_INLINE_CAP];
        bytes[..key.len()].copy_from_slice(key);
        bytes[key.len()..key.len() + value.len()].copy_from_slice(value);
        Self {
            seq,
            tombstone: 0,
            key_len: key.len() as u8,
            value_len: value.len() as u8,
            _pad: 0,
            bytes,
        }
    }

    /// Borrow the inline key bytes.
    #[must_use]
    pub fn key(&self) -> &[u8] {
        let k = (self.key_len as usize).min(LEAF_INLINE_CAP);
        &self.bytes[..k]
    }

    /// Borrow the inline value bytes.
    #[must_use]
    pub fn value(&self) -> &[u8] {
        let k = (self.key_len as usize).min(LEAF_INLINE_CAP);
        let v = (self.value_len as usize).min(LEAF_INLINE_CAP - k);
        &self.bytes[k..k + v]
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
