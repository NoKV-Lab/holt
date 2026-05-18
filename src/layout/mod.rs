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
pub use header::{
    BlobGuid, BlobHeader, DATA_AREA_CAPACITY, DATA_AREA_START, HEADER_SIZE, MAX_SLOTS, PAGE_SIZE,
};
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
