//! `NodeType` enum + size table.

/// NodeType discriminant.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Sentinel — never appears in a valid tree. Reading a slot
    /// tagged `invalid` panics.
    Invalid = 0,
    /// Key-value leaf (16-byte body + bump-allocated extent for
    /// key/value bytes).
    Leaf = 1,
    /// Path-compressed prefix (128-byte fixed body; up to 112
    /// inline bytes).
    Prefix = 2,
    /// In-tree blob crossing (128-byte body carrying
    /// `child_blob_guid` + `child_entry_ptr`).
    Blob = 3,
    /// 1..4 children, parallel sorted `keys[4]` + `children[4]`.
    Node4 = 4,
    /// 5..16 children, sorted `keys[16]` for SIMD scan.
    Node16 = 5,
    /// 17..48 children, byte-indexed `index[256]` → `children[48]`.
    Node48 = 6,
    /// 49..256 children, direct `children[256]`.
    Node256 = 7,
    /// Empty-tree sentinel: 8 bytes all zero. Allocated once on
    /// `BlobFrame::init` and stored at `header.root_slot`.
    EmptyRoot = 8,
}

impl NodeType {
    /// Convert a raw byte (e.g. from a `SlotEntry`'s
    /// `ntype_or_next_free` field) into a `NodeType`. Returns
    /// `None` for values outside 0..=8.
    #[must_use]
    pub fn from_raw(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Invalid),
            1 => Some(Self::Leaf),
            2 => Some(Self::Prefix),
            3 => Some(Self::Blob),
            4 => Some(Self::Node4),
            5 => Some(Self::Node16),
            6 => Some(Self::Node48),
            7 => Some(Self::Node256),
            8 => Some(Self::EmptyRoot),
            _ => None,
        }
    }

    /// Underlying byte representation.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Per-NodeType allocation sizes (bytes), indexed by `ntype - 1`.
///
/// Sizes are chosen so the four ART-internal variants
/// (Node{4,16,48,256}) fit their children + index arrays exactly
/// with no slack. Leaf is a fixed 16-byte header pointing at a
/// separate key/value extent; Prefix and Blob are both 128 B so
/// their inline path-compressed bytes fit comfortably.
pub const SIZE_BY_TYPE: [u32; 8] = [
    16,   // Leaf
    128,  // Prefix
    128,  // Blob
    24,   // Node4
    88,   // Node16
    456,  // Node48
    1032, // Node256
    8,    // EmptyRoot
];

/// Bytes a single allocation of the given NodeType consumes.
///
/// Panics on `NodeType::Invalid` (which has no associated size).
#[must_use]
pub fn size_of_node(ntype: NodeType) -> u32 {
    assert!(ntype != NodeType::Invalid, "size_of_node(Invalid)");
    let idx = ntype as usize - 1;
    SIZE_BY_TYPE[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntype_round_trip_via_raw() {
        let all = [
            NodeType::Invalid,
            NodeType::Leaf,
            NodeType::Prefix,
            NodeType::Blob,
            NodeType::Node4,
            NodeType::Node16,
            NodeType::Node48,
            NodeType::Node256,
            NodeType::EmptyRoot,
        ];
        for t in all {
            assert_eq!(NodeType::from_raw(t.as_u8()), Some(t));
        }
        // Values 9 and above are not in the enum.
        assert_eq!(NodeType::from_raw(9), None);
        assert_eq!(NodeType::from_raw(255), None);
    }

    #[test]
    fn size_table_per_node_type() {
        assert_eq!(size_of_node(NodeType::Leaf), 16);
        assert_eq!(size_of_node(NodeType::Prefix), 128);
        assert_eq!(size_of_node(NodeType::Blob), 128);
        assert_eq!(size_of_node(NodeType::Node4), 24);
        assert_eq!(size_of_node(NodeType::Node16), 88);
        assert_eq!(size_of_node(NodeType::Node48), 456);
        assert_eq!(size_of_node(NodeType::Node256), 1032);
        assert_eq!(size_of_node(NodeType::EmptyRoot), 8);
    }
}
