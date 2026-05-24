//! Extern struct layouts for on-disk types.
//!
//! Every type in this module is a `#[repr(C)]` extern struct with
//! a compile-time size assertion pinning its byte layout. If a
//! field is ever moved, the assertion fails at compile time —
//! protecting against accidental layout drift across releases.

mod blob_node;
mod header;
mod leaf;
mod node;
mod nodes;
mod prefix;
mod slot;

pub use blob_node::{BlobNode, BLOB_MAX_INLINE};
pub(crate) use header::ROOT_BLOB_GUID;
pub use header::{BlobGuid, BlobHeader, DATA_AREA_START, HEADER_SIZE, MAX_SLOTS, PAGE_SIZE};
pub use leaf::{leaf_extent_size, Leaf};
pub use node::{size_of_node, NodeType, SIZE_BY_TYPE};
pub use nodes::{Node16, Node256, Node4, Node48};
pub use prefix::{Prefix, PREFIX_MAX_INLINE};
pub use slot::{SlotEntry, SlotEntryRaw};

/// Sanity: ensure all per-NodeType bodies match the size-table
/// constants. If any drift, the compiler refuses to build.
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<Leaf>() == SIZE_BY_TYPE[0] as usize);
    assert!(size_of::<Prefix>() == SIZE_BY_TYPE[1] as usize);
    assert!(size_of::<BlobNode>() == SIZE_BY_TYPE[2] as usize);
    assert!(size_of::<Node4>() == SIZE_BY_TYPE[3] as usize);
    assert!(size_of::<Node16>() == SIZE_BY_TYPE[4] as usize);
    assert!(size_of::<Node48>() == SIZE_BY_TYPE[5] as usize);
    assert!(size_of::<Node256>() == SIZE_BY_TYPE[6] as usize);
    // SIZE_BY_TYPE[7] is the empty-tree sentinel (8 B all-zero,
    // no struct counterpart — it's just a zero u64).
    assert!(SIZE_BY_TYPE[7] == 8);
};

#[cfg(test)]
mod tests {
    //! On-disk layout invariants pinned at runtime.
    //!
    //! These sizes + offsets are also pinned at compile time via
    //! the `const _: () = assert!(...)` block above. The runtime
    //! tests give the same guarantees as a smoke-test layer
    //! during local iteration.

    use super::*;
    use std::mem::size_of;

    #[test]
    fn on_disk_constants() {
        assert_eq!(PAGE_SIZE, 524_288);
        assert_eq!(HEADER_SIZE, 4096);
        assert_eq!(MAX_SLOTS, 10_240);
        assert_eq!(DATA_AREA_START, 0xB000);
        assert_eq!(PREFIX_MAX_INLINE, 112);
        assert_eq!(BLOB_MAX_INLINE, 104);
    }

    #[test]
    fn blob_header_is_exactly_4096_bytes() {
        assert_eq!(size_of::<BlobHeader>(), 4096);
    }

    #[test]
    fn per_node_sizes() {
        assert_eq!(size_of::<Leaf>(), 16);
        assert_eq!(size_of::<Prefix>(), 128);
        assert_eq!(size_of::<BlobNode>(), 128);
        assert_eq!(size_of::<Node4>(), 24);
        assert_eq!(size_of::<Node16>(), 88);
        assert_eq!(size_of::<Node48>(), 456);
        assert_eq!(size_of::<Node256>(), 1032);
    }

    #[test]
    fn size_of_node_matches_per_type_struct() {
        assert_eq!(size_of_node(NodeType::Leaf) as usize, size_of::<Leaf>());
        assert_eq!(size_of_node(NodeType::Prefix) as usize, size_of::<Prefix>());
        assert_eq!(size_of_node(NodeType::Blob) as usize, size_of::<BlobNode>());
        assert_eq!(size_of_node(NodeType::Node4) as usize, size_of::<Node4>());
        assert_eq!(size_of_node(NodeType::Node16) as usize, size_of::<Node16>());
        assert_eq!(size_of_node(NodeType::Node48) as usize, size_of::<Node48>());
        assert_eq!(
            size_of_node(NodeType::Node256) as usize,
            size_of::<Node256>()
        );
        assert_eq!(size_of_node(NodeType::EmptyRoot), 8);
    }

    #[test]
    fn leaf_extent_size_is_always_aligned_to_8() {
        // The bump allocator's invariant: all extent allocations
        // are 8-byte aligned (so subsequent body allocs stay
        // aligned).
        for key_len in 0..32 {
            for value_len in 0..32 {
                let s = leaf_extent_size(key_len, value_len);
                assert_eq!(s % 8, 0, "leaf_extent_size({key_len}, {value_len}) = {s}");
                // And it's the smallest 8-aligned size ≥ 2+key+value.
                let need = 2 + key_len + value_len;
                assert!(s >= need);
                assert!(s < need + 8);
            }
        }
    }
}
