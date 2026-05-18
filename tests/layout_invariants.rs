//! Integration tests pinning the on-disk layout invariants.
//!
//! These sizes + offsets are also pinned at compile time via
//! `const _: () = assert!(...)` blocks in `src/layout/`. These
//! tests double-check from the public API surface so any
//! external observer can verify the contract.

use artisan::layout::{
    leaf_extent_size, size_of_node, BlobHeader, BlobNode, Leaf, Node16, Node256, Node4, Node48,
    NodeType, Prefix, BLOB_MAX_INLINE, DATA_AREA_START, HEADER_SIZE, MAX_SLOTS, PAGE_SIZE,
    PREFIX_MAX_INLINE,
};
use std::mem::size_of;

#[test]
fn on_disk_constants() {
    assert_eq!(PAGE_SIZE, 524288);
    assert_eq!(HEADER_SIZE, 4096);
    assert_eq!(MAX_SLOTS, 10240);
    assert_eq!(DATA_AREA_START, 0xB000);
    assert_eq!(PREFIX_MAX_INLINE, 112);
    assert_eq!(BLOB_MAX_INLINE, 96);
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
    // The bump allocator's invariant: all extent allocations are
    // 8-byte aligned (so subsequent body allocs stay aligned).
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
