//! Unit tests for the walker — single-blob inserts / lookups /
//! erases + cross-blob primitives (`make_blob_from_node`,
//! BlobNode descent) + `compact_blob`.

use super::cast;
use super::erase::erase;
use super::insert::insert;
use super::lookup::{lookup, lookup_at};
use super::migrate::{blob_needs_compaction, compact_blob, make_blob_from_node};
use super::readers::read_prefix;
use super::types::LookupResult;
use super::writers::write_struct_to_slot;
use super::SearchKey;
use crate::api::errors::Error;
use crate::layout::{BlobGuid, BlobNode, NodeType, PAGE_SIZE};
use crate::store::blob_store::AlignedBlobBuf;
use crate::store::BlobFrame;

fn fresh_blob() -> (Vec<u8>, BlobGuid) {
    let guid: BlobGuid = [0x11; 16];
    let mut buf = vec![0u8; PAGE_SIZE as usize];
    BlobFrame::init(&mut buf, guid).unwrap();
    (buf, guid)
}

fn put(frame: &mut BlobFrame<'_>, k: &[u8], v: &[u8], seq: u64) {
    let root = frame.header().root_slot;
    insert(frame, root, k, v, seq).unwrap();
}

fn get(frame: &BlobFrame<'_>, k: &[u8]) -> Option<Vec<u8>> {
    let root = frame.header().root_slot;
    match lookup(frame.as_ref(), root, k).unwrap() {
        LookupResult::Found(hit) => Some(hit.value.to_vec()),
        LookupResult::NotFound => None,
        LookupResult::Crossing(_) => {
            panic!("walker unit tests never construct a BlobNode")
        }
    }
}

/// Run filter-mode compaction on a Vec-backed test blob in place.
///
/// Erase only flips the leaf's `tombstone` byte and bumps the
/// blob's `tombstone_leaf_cnt`; the structural collapse — lone-child
/// `Prefix` wrap, `Node256→48→16→4` downshift, `EmptyRoot` reseat
/// — runs inside `compact_blob`. Tests that assert post-erase shape
/// drop their `BlobFrame` view, call this, and re-wrap.
fn compact_in_place(buf: &mut [u8]) {
    let mut ab = AlignedBlobBuf::zeroed();
    ab.as_mut_slice().copy_from_slice(buf);
    compact_blob(&mut ab).unwrap();
    buf.copy_from_slice(ab.as_slice());
}

fn replace_root_with_blob_node(buf: &mut AlignedBlobBuf, child_guid: BlobGuid) {
    let mut frame = BlobFrame::wrap(buf.as_mut_slice());
    let bn_out = frame.alloc_node(NodeType::Blob).unwrap();
    let bn = BlobNode::new(b"", child_guid);
    write_struct_to_slot(&mut frame, bn_out.slot, &bn).unwrap();
    frame.header_mut().root_slot = bn_out.slot;
}

#[test]
fn single_insert_then_lookup() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"hello", b"world", 1);
    assert_eq!(get(&frame, b"hello").as_deref(), Some(&b"world"[..]));
    assert_eq!(get(&frame, b"hellx"), None);
}

#[test]
fn update_same_key_replaces_value() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"k", b"v1", 1);
    let root = frame.header().root_slot;
    insert(&mut frame, root, b"k", b"v2", 2).unwrap();
    assert_eq!(get(&frame, b"k").as_deref(), Some(&b"v2"[..]));
}

fn ntype_at(frame: &BlobFrame<'_>, slot: u16) -> NodeType {
    frame.slot_entry(slot).unwrap().node_type().unwrap()
}

#[test]
fn small_record_inlines_and_round_trips() {
    use crate::layout::fits_leaf_inline;
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);

    // Single small key: the root points straight at the leaf, so the
    // root slot itself must be a LeafInline (key+value in the body).
    assert!(fits_leaf_inline(3, 5));
    put(&mut frame, b"abc", b"value", 1);
    let root = frame.header().root_slot;
    assert_eq!(ntype_at(&frame, root), NodeType::LeafInline);
    assert_eq!(get(&frame, b"abc").as_deref(), Some(&b"value"[..]));

    // Same-key small→small update rewrites the fixed body in place:
    // node stays a LeafInline and no new slot is consumed.
    let slots_before = frame.header().num_slots;
    let root = frame.header().root_slot;
    insert(&mut frame, root, b"abc", b"v2", 2).unwrap();
    let root = frame.header().root_slot;
    assert_eq!(ntype_at(&frame, root), NodeType::LeafInline);
    assert_eq!(frame.header().num_slots, slots_before, "in-place update");
    assert_eq!(get(&frame, b"abc").as_deref(), Some(&b"v2"[..]));
}

#[test]
fn leaf_inline_grows_to_extent_then_reuses_it() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"abc", b"tiny", 1);
    let root = frame.header().root_slot;
    assert_eq!(ntype_at(&frame, root), NodeType::LeafInline);

    // Value over the inline cap spills to a regular extent Leaf.
    let big = vec![0xAB; 200];
    let root = frame.header().root_slot;
    insert(&mut frame, root, b"abc", &big, 2).unwrap();
    let root = frame.header().root_slot;
    assert_eq!(ntype_at(&frame, root), NodeType::Leaf);
    assert_eq!(get(&frame, b"abc").as_deref(), Some(big.as_slice()));

    // Shrinking back below the inline cap deliberately keeps the
    // existing extent and reuses it in place — that's cheaper than
    // freeing the extent to re-inline, so the node stays an extent
    // Leaf while the value round-trips correctly.
    let root = frame.header().root_slot;
    insert(&mut frame, root, b"abc", b"sm", 3).unwrap();
    let root = frame.header().root_slot;
    assert_eq!(ntype_at(&frame, root), NodeType::Leaf);
    assert_eq!(get(&frame, b"abc").as_deref(), Some(&b"sm"[..]));
}

#[test]
fn leaf_inline_cap_boundary() {
    use crate::layout::{fits_leaf_inline, LEAF_INLINE_CAP};
    // Exactly at the cap inlines; one byte past does not. Length
    // fields are u8, so neither dimension may exceed 255.
    assert!(fits_leaf_inline(10, LEAF_INLINE_CAP - 10));
    assert!(!fits_leaf_inline(10, LEAF_INLINE_CAP - 10 + 1));
    assert!(!fits_leaf_inline(256, 0));
    assert!(!fits_leaf_inline(0, 256));

    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let key = vec![b'k'; 10];
    let val = vec![0x7u8; LEAF_INLINE_CAP - 10]; // fills the body exactly
    put(&mut frame, &key, &val, 1);
    let root = frame.header().root_slot;
    assert_eq!(ntype_at(&frame, root), NodeType::LeafInline);
    assert_eq!(get(&frame, &key).as_deref(), Some(val.as_slice()));
}

#[test]
fn leaf_inline_split_erase_and_compaction() {
    let (mut buf, _) = fresh_blob();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        // Two small keys sharing a prefix split the original inline
        // leaf into an inner node with two inline-leaf children.
        put(&mut frame, b"abc/01", b"x", 1);
        put(&mut frame, b"abc/02", b"y", 2);
        assert_eq!(get(&frame, b"abc/01").as_deref(), Some(&b"x"[..]));
        assert_eq!(get(&frame, b"abc/02").as_deref(), Some(&b"y"[..]));

        // Tombstone one; the survivor stays readable.
        let root = frame.header().root_slot;
        erase(&mut frame, root, b"abc/01").unwrap();
        assert_eq!(get(&frame, b"abc/01"), None);
        assert_eq!(get(&frame, b"abc/02").as_deref(), Some(&b"y"[..]));
    }
    // Compaction rebuilds the live tree via clone_leaf_inline.
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    assert_eq!(get(&frame, b"abc/02").as_deref(), Some(&b"y"[..]));
    assert_eq!(get(&frame, b"abc/01"), None);
}

#[test]
fn leaf_fingerprint_rejects_wrong_key_keeps_present_key() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);

    // Large value forces an extent Leaf (not LeafInline); the
    // fingerprint gate only guards the extent read path.
    let big = vec![0xCD; 100];
    put(&mut frame, b"hello", &big, 1);

    // Lazy expansion: looking up a different key descends to the one
    // leaf, whose fingerprint rejects the wrong key — without a false
    // negative on the present key.
    assert_eq!(get(&frame, b"hello").as_deref(), Some(big.as_slice()));
    assert_eq!(get(&frame, b"hellx"), None);
    assert_eq!(get(&frame, b"help"), None);

    // The written extent leaf carries a non-zero fingerprint.
    let root = frame.header().root_slot;
    assert_eq!(ntype_at(&frame, root), NodeType::Leaf);
    let body = frame.as_ref().body_of_slot(root).unwrap();
    let leaf = *cast::<crate::layout::Leaf>(body);
    assert_ne!(leaf.key_fp, 0, "written leaf must carry a fingerprint");

    // Keys sharing a long prefix: a missing sibling resolves to
    // NotFound; both present keys are still found.
    put(&mut frame, b"shared/prefix/aaaa", &big, 2);
    put(&mut frame, b"shared/prefix/bbbb", &big, 3);
    assert_eq!(
        get(&frame, b"shared/prefix/aaaa").as_deref(),
        Some(big.as_slice())
    );
    assert_eq!(
        get(&frame, b"shared/prefix/bbbb").as_deref(),
        Some(big.as_slice())
    );
    assert_eq!(get(&frame, b"shared/prefix/cccc"), None);
}

#[test]
fn two_keys_with_shared_prefix_creates_prefix_plus_node4() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"abc/01", b"v1", 1);
    put(&mut frame, b"abc/02", b"v2", 2);
    assert_eq!(get(&frame, b"abc/01").as_deref(), Some(&b"v1"[..]));
    assert_eq!(get(&frame, b"abc/02").as_deref(), Some(&b"v2"[..]));
    assert_eq!(get(&frame, b"abc/03"), None);
    let root_slot = frame.header().root_slot;
    let entry = frame.slot_entry(root_slot).unwrap();
    assert_eq!(entry.node_type(), Some(NodeType::Prefix));
}

#[test]
fn two_keys_no_shared_prefix_creates_naked_node4() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"a", b"va", 1);
    put(&mut frame, b"b", b"vb", 2);
    assert_eq!(get(&frame, b"a").as_deref(), Some(&b"va"[..]));
    assert_eq!(get(&frame, b"b").as_deref(), Some(&b"vb"[..]));
    let root_slot = frame.header().root_slot;
    let entry = frame.slot_entry(root_slot).unwrap();
    assert_eq!(entry.node_type(), Some(NodeType::Node4));
}

#[test]
fn grow_node4_to_node16() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    for i in 0..5u8 {
        let k = [b'k', b'0' + i];
        put(&mut frame, &k, &[b'v', b'0' + i], i as u64 + 1);
    }
    for i in 0..5u8 {
        let k = [b'k', b'0' + i];
        let v = [b'v', b'0' + i];
        assert_eq!(get(&frame, &k).as_deref(), Some(&v[..]));
    }
    let root_slot = frame.header().root_slot;
    let entry = frame.slot_entry(root_slot).unwrap();
    assert_eq!(entry.node_type(), Some(NodeType::Prefix));
    let p = read_prefix(frame.as_ref(), root_slot).unwrap();
    let inner_slot = p.child as u16;
    let ie = frame.slot_entry(inner_slot).unwrap();
    assert_eq!(ie.node_type(), Some(NodeType::Node16));
}

#[test]
fn grow_chain_node4_to_node16_to_node48() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    for i in 0..20u8 {
        let k = [b'p', i];
        put(&mut frame, &k, &[i], i as u64 + 1);
    }
    for i in 0..20u8 {
        let k = [b'p', i];
        assert_eq!(get(&frame, &k).as_deref(), Some(&[i][..]));
    }
    let root_slot = frame.header().root_slot;
    let p = read_prefix(frame.as_ref(), root_slot).unwrap();
    let inner_slot = p.child as u16;
    assert_eq!(
        frame.slot_entry(inner_slot).unwrap().node_type(),
        Some(NodeType::Node48)
    );
}

#[test]
fn grow_chain_through_node256() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    for i in 0..60u8 {
        let k = [b'q', i];
        put(&mut frame, &k, &[i, i ^ 0xFF], i as u64 + 1);
    }
    for i in 0..60u8 {
        let k = [b'q', i];
        let v = [i, i ^ 0xFF];
        assert_eq!(get(&frame, &k).as_deref(), Some(&v[..]));
    }
    let root_slot = frame.header().root_slot;
    let p = read_prefix(frame.as_ref(), root_slot).unwrap();
    let inner_slot = p.child as u16;
    assert_eq!(
        frame.slot_entry(inner_slot).unwrap().node_type(),
        Some(NodeType::Node256)
    );
}

#[test]
fn prefix_split_at_divergence() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"abcdef", b"v1", 1);
    put(&mut frame, b"abcXYZ", b"v2", 2);
    assert_eq!(get(&frame, b"abcdef").as_deref(), Some(&b"v1"[..]));
    assert_eq!(get(&frame, b"abcXYZ").as_deref(), Some(&b"v2"[..]));
    assert_eq!(get(&frame, b"abcdeg"), None);
}

#[test]
fn deep_prefix_chain_long_keys() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let mut k1 = vec![b'x'; 250];
    let mut k2 = k1.clone();
    k1.push(b'1');
    k2.push(b'2');
    put(&mut frame, &k1, b"v1", 1);
    put(&mut frame, &k2, b"v2", 2);
    assert_eq!(get(&frame, &k1).as_deref(), Some(&b"v1"[..]));
    assert_eq!(get(&frame, &k2).as_deref(), Some(&b"v2"[..]));
}

#[test]
fn strict_prefix_returns_not_yet_implemented() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"abc", b"v1", 1);
    let root = frame.header().root_slot;
    let r = insert(&mut frame, root, b"abcdef", b"v2", 2);
    assert!(matches!(r, Err(Error::NotYetImplemented(_))));
}

#[test]
fn many_inserts_all_readable() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for i in 0..200u32 {
        let k = format!("key/{i:04}/end").into_bytes();
        let v = format!("val#{i}").into_bytes();
        pairs.push((k, v));
    }
    for (i, (k, v)) in pairs.iter().enumerate() {
        put(&mut frame, k, v, i as u64 + 1);
    }
    for (k, v) in &pairs {
        assert_eq!(get(&frame, k).as_deref(), Some(&v[..]));
    }
}

// -------- erase --------

fn del(frame: &mut BlobFrame<'_>, k: &[u8]) -> bool {
    let root = frame.header().root_slot;
    let r = erase(frame, root, k).unwrap();
    r.mutated
}

#[test]
fn erase_only_leaf_marks_tombstone_and_compacts_empty() {
    let (mut buf, _) = fresh_blob();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"k", b"v", 1);
        assert!(del(&mut frame, b"k"));
        assert_eq!(get(&frame, b"k"), None);
        // Erase tombstones; the root is still a (tombstoned) leaf
        // until compact rebuilds.
        assert_eq!(frame.header().tombstone_leaf_cnt, 1);
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    let root_slot = frame.header().root_slot;
    let e = frame.slot_entry(root_slot).unwrap();
    assert_eq!(e.node_type(), Some(NodeType::EmptyRoot));
    assert_eq!(frame.header().tombstone_leaf_cnt, 0);
    assert_eq!(frame.header().compact_times, 1);
}

#[test]
fn erase_missing_key_is_noop() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"a", b"1", 1);
    assert!(!del(&mut frame, b"b"));
    assert_eq!(get(&frame, b"a").as_deref(), Some(&b"1"[..]));
}

#[test]
fn erase_one_of_two_node4_collapses_to_prefix_over_lone_leaf() {
    let (mut buf, _) = fresh_blob();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        put(&mut frame, b"a", b"1", 1);
        put(&mut frame, b"b", b"2", 2);
        del(&mut frame, b"a");
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    let root_slot = frame.header().root_slot;
    let e = frame.slot_entry(root_slot).unwrap();
    assert_eq!(e.node_type(), Some(NodeType::Prefix));
    assert_eq!(get(&frame, b"b").as_deref(), Some(&b"2"[..]));
    assert_eq!(get(&frame, b"a"), None);
}

#[test]
fn erase_collapses_node16_lone_child() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    for i in 0..5u8 {
        let k = [b'k', b'0' + i];
        put(&mut frame, &k, &[i], i as u64 + 1);
    }
    for i in 0..4u8 {
        let k = [b'k', b'0' + i];
        del(&mut frame, &k);
    }
    let k_last = [b'k', b'0' + 4];
    assert_eq!(get(&frame, &k_last).as_deref(), Some(&[4][..]));
    for i in 0..4u8 {
        let k = [b'k', b'0' + i];
        assert_eq!(get(&frame, &k), None);
    }
}

#[test]
fn erase_collapses_node48_lone_child() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    for i in 0..17u8 {
        let k = [b'p', i];
        put(&mut frame, &k, &[i], i as u64 + 1);
    }
    for i in 0..16u8 {
        let k = [b'p', i];
        del(&mut frame, &k);
    }
    let k_last = [b'p', 16];
    assert_eq!(get(&frame, &k_last).as_deref(), Some(&[16][..]));
}

#[test]
fn erase_collapses_node256_lone_child() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    for i in 0..60u8 {
        let k = [b'q', i];
        put(&mut frame, &k, &[i, i ^ 0xFF], i as u64 + 1);
    }
    for i in 0..59u8 {
        let k = [b'q', i];
        del(&mut frame, &k);
    }
    let k_last = [b'q', 59];
    let v_last = [59u8, 0x3B ^ 0xFFu8];
    assert_eq!(get(&frame, &k_last).as_deref(), Some(&v_last[..]));
}

/// Walk through the root's `Prefix` chain and return the slot of
/// the first node that isn't a Prefix — the test's "inner node"
/// of interest.
fn inner_node_slot(frame: &BlobFrame<'_>) -> u16 {
    let mut s = frame.header().root_slot;
    loop {
        let ntype = frame.slot_entry(s).unwrap().node_type().unwrap();
        if ntype != NodeType::Prefix {
            return s;
        }
        let p = read_prefix(frame.as_ref(), s).unwrap();
        s = p.child as u16;
    }
}

#[test]
fn shrink_node16_to_node4_at_three_remaining() {
    // 5 children → grows to Node16. Erase down to 3 live children
    // + compact → shrinks to Node4. (The `pack_inner_node` arm
    // picks Node4 for survivor counts in `2..=4`.)
    let (mut buf, _) = fresh_blob();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        for i in 0..5u8 {
            let k = [b'k', b'0' + i];
            put(&mut frame, &k, &[i], i as u64 + 1);
        }
        assert_eq!(
            frame
                .slot_entry(inner_node_slot(&frame))
                .unwrap()
                .node_type(),
            Some(NodeType::Node16),
        );
        // Erase two so 3 children remain live.
        del(&mut frame, b"k0");
        del(&mut frame, &[b'k', b'0' + 1]);
        // Inner node is still Node16 until compaction filters the
        // tombstones and rebuilds.
        assert_eq!(
            frame
                .slot_entry(inner_node_slot(&frame))
                .unwrap()
                .node_type(),
            Some(NodeType::Node16),
        );
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node4),
        "Node16 with 3 live children should compact to Node4",
    );
    for i in 2..5u8 {
        let k = [b'k', b'0' + i];
        assert_eq!(get(&frame, &k).as_deref(), Some(&[i][..]));
    }
}

#[test]
fn shrink_node48_to_node16_at_twelve_remaining() {
    // 20 children → grows through Node16 → Node48. Erase down to
    // 12 live children + compact → shrinks to Node16. The
    // `pack_inner_node` arm picks Node16 for survivor counts in
    // `5..=16`.
    let (mut buf, _) = fresh_blob();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        for i in 0..20u8 {
            let k = [b'p', i];
            put(&mut frame, &k, &[i], i as u64 + 1);
        }
        assert_eq!(
            frame
                .slot_entry(inner_node_slot(&frame))
                .unwrap()
                .node_type(),
            Some(NodeType::Node48),
        );
        // Erase 8 so 12 live children remain.
        for i in 0..8u8 {
            let k = [b'p', i];
            del(&mut frame, &k);
        }
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node16),
        "Node48 with 12 live children should compact to Node16",
    );
    for i in 8..20u8 {
        let k = [b'p', i];
        assert_eq!(get(&frame, &k).as_deref(), Some(&[i][..]));
    }
}

#[test]
fn shrink_node256_to_node48_at_thirty_seven_remaining() {
    // 60 children → grows through Node48 → Node256. Erase down to
    // 37 live children + compact → shrinks to Node48. The
    // `pack_inner_node` arm picks Node48 for survivor counts in
    // `17..=48`.
    let (mut buf, _) = fresh_blob();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        for i in 0..60u8 {
            let k = [b'q', i];
            put(&mut frame, &k, &[i, i ^ 0xFF], i as u64 + 1);
        }
        assert_eq!(
            frame
                .slot_entry(inner_node_slot(&frame))
                .unwrap()
                .node_type(),
            Some(NodeType::Node256),
        );
        // Erase 23 so 37 live children remain.
        for i in 0..23u8 {
            let k = [b'q', i];
            del(&mut frame, &k);
        }
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node48),
        "Node256 with 37 live children should compact to Node48",
    );
    for i in 23..60u8 {
        let k = [b'q', i];
        let v = [i, i ^ 0xFF];
        assert_eq!(get(&frame, &k).as_deref(), Some(&v[..]));
    }
}

#[test]
fn shrink_chain_node256_node48_node16_node4() {
    // One sustained churn: grow up through Node256, then erase +
    // compact past every `pack_inner_node` boundary in order.
    // Confirms the survivor-driven downshift composes end-to-end:
    // 60 → 37 → 12 → 3 live children pulls the inner node through
    // Node256 → Node48 → Node16 → Node4.
    let (mut buf, _) = fresh_blob();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        for i in 0..60u8 {
            let k = [b'q', i];
            put(&mut frame, &k, &[i], i as u64 + 1);
        }
        assert_eq!(
            frame
                .slot_entry(inner_node_slot(&frame))
                .unwrap()
                .node_type(),
            Some(NodeType::Node256),
        );

        // Erase 23 so 37 live children remain.
        for i in 0..23u8 {
            del(&mut frame, &[b'q', i]);
        }
    }
    compact_in_place(&mut buf);
    {
        let frame = BlobFrame::wrap(&mut buf);
        assert_eq!(
            frame
                .slot_entry(inner_node_slot(&frame))
                .unwrap()
                .node_type(),
            Some(NodeType::Node48),
        );
    }

    {
        let mut frame = BlobFrame::wrap(&mut buf);
        // Erase another 25 (total 48 erased) so 12 live remain.
        for i in 23..48u8 {
            del(&mut frame, &[b'q', i]);
        }
    }
    compact_in_place(&mut buf);
    {
        let frame = BlobFrame::wrap(&mut buf);
        assert_eq!(
            frame
                .slot_entry(inner_node_slot(&frame))
                .unwrap()
                .node_type(),
            Some(NodeType::Node16),
        );
    }

    {
        let mut frame = BlobFrame::wrap(&mut buf);
        // Erase another 9 (total 57 erased) so 3 live remain.
        for i in 48..57u8 {
            del(&mut frame, &[b'q', i]);
        }
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node4),
    );

    // The last three keys (57, 58, 59) still readable.
    for i in 57..60u8 {
        let k = [b'q', i];
        assert_eq!(get(&frame, &k).as_deref(), Some(&[i][..]));
    }
    // Three compactions ran; nothing tombstoned in the survivor.
    assert_eq!(frame.header().compact_times, 3);
    assert_eq!(frame.header().tombstone_leaf_cnt, 0);
}

#[test]
fn erase_all_returns_to_empty_root() {
    let (mut buf, _) = fresh_blob();
    let pairs = [
        (&b"alpha"[..], &b"A"[..]),
        (&b"beta"[..], &b"B"[..]),
        (&b"gamma"[..], &b"G"[..]),
    ];
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        for (i, (k, v)) in pairs.iter().enumerate() {
            put(&mut frame, k, v, i as u64 + 1);
        }
        for (k, _) in &pairs {
            assert!(del(&mut frame, k));
        }
        assert_eq!(frame.header().tombstone_leaf_cnt as usize, pairs.len());
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    let root_slot = frame.header().root_slot;
    assert_eq!(
        frame.slot_entry(root_slot).unwrap().node_type(),
        Some(NodeType::EmptyRoot)
    );
    assert_eq!(frame.header().tombstone_leaf_cnt, 0);
}

#[test]
fn erase_through_prefix_keeps_other_branches() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"img/01.jpg", b"a", 1);
    put(&mut frame, b"img/02.jpg", b"b", 2);
    put(&mut frame, b"img/03.jpg", b"c", 3);
    assert!(del(&mut frame, b"img/02.jpg"));
    assert_eq!(get(&frame, b"img/01.jpg").as_deref(), Some(&b"a"[..]));
    assert_eq!(get(&frame, b"img/02.jpg"), None);
    assert_eq!(get(&frame, b"img/03.jpg").as_deref(), Some(&b"c"[..]));
}

#[test]
fn insert_after_erase_reinstates_key() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"k", b"v1", 1);
    del(&mut frame, b"k");
    put(&mut frame, b"k", b"v2", 2);
    assert_eq!(get(&frame, b"k").as_deref(), Some(&b"v2"[..]));
}

#[test]
fn churn_100_keys_inserted_then_all_erased() {
    let (mut buf, _) = fresh_blob();
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..100u32)
        .map(|i| {
            (
                format!("k{i:04}").into_bytes(),
                format!("v{i}").into_bytes(),
            )
        })
        .collect();
    {
        let mut frame = BlobFrame::wrap(&mut buf);
        for (i, (k, v)) in pairs.iter().enumerate() {
            put(&mut frame, k, v, i as u64 + 1);
        }
        for (k, _) in &pairs {
            assert!(del(&mut frame, k));
        }
        for (k, _) in &pairs {
            assert_eq!(get(&frame, k), None);
        }
    }
    compact_in_place(&mut buf);
    let frame = BlobFrame::wrap(&mut buf);
    let root_slot = frame.header().root_slot;
    assert_eq!(
        frame.slot_entry(root_slot).unwrap().node_type(),
        Some(NodeType::EmptyRoot)
    );
}

// ============================================================
// Multi-blob lookup + `make_blob_from_node`
// ============================================================

fn install_blob_node(frame: &mut BlobFrame<'_>, slot: u16, prefix: &[u8], child_guid: BlobGuid) {
    let bn = crate::layout::BlobNode::new(prefix, child_guid);
    write_struct_to_slot(frame, slot, &bn).unwrap();
}

#[test]
fn lookup_blob_node_emits_crossing_on_match() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let out = frame.alloc_node(NodeType::Blob).unwrap();
    let child_guid: BlobGuid = [0xAA; 16];
    install_blob_node(&mut frame, out.slot, b"img/", child_guid);
    frame.header_mut().root_slot = out.slot;

    let r = lookup(frame.as_ref(), out.slot, b"img/01.jpg").unwrap();
    match r {
        LookupResult::Crossing(c) => {
            assert_eq!(c.child_guid, child_guid);
            assert_eq!(c.child_depth, 4);
        }
        other => panic!("expected Crossing, got {other:?}"),
    }
}

#[test]
fn lookup_blob_node_returns_not_found_when_prefix_diverges() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let out = frame.alloc_node(NodeType::Blob).unwrap();
    install_blob_node(&mut frame, out.slot, b"img/", [0xAA; 16]);
    frame.header_mut().root_slot = out.slot;

    let r = lookup(frame.as_ref(), out.slot, b"doc/page1.txt").unwrap();
    assert!(matches!(r, LookupResult::NotFound));
}

#[test]
fn lookup_at_continues_descent_from_supplied_depth() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"img/01.jpg", b"v1", 1);
    let root = frame.header().root_slot;

    let r0 = lookup(frame.as_ref(), root, b"img/01.jpg").unwrap();
    assert!(matches!(r0, LookupResult::Found(hit) if hit.value == b"v1"));

    let r1 = lookup_at(frame.as_ref(), root, SearchKey::exact(b"img/01.jpg"), 0).unwrap();
    assert!(matches!(r1, LookupResult::Found(hit) if hit.value == b"v1"));
}

#[test]
fn insert_splits_blob_node_inline_prefix_on_first_byte_divergence() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let out = frame.alloc_node(NodeType::Blob).unwrap();
    let child_guid: BlobGuid = [0xAA; 16];
    install_blob_node(&mut frame, out.slot, b"img/", child_guid);
    frame.header_mut().root_slot = out.slot;

    let root = frame.header().root_slot;
    insert(&mut frame, root, b"doc/page1.txt", b"meta", 7).unwrap();
    assert_eq!(get(&frame, b"doc/page1.txt").as_deref(), Some(&b"meta"[..]));

    let root = frame.header().root_slot;
    assert_eq!(
        frame.slot_entry(root).unwrap().node_type(),
        Some(NodeType::Node4)
    );
    let r = lookup(frame.as_ref(), root, b"img/01.jpg").unwrap();
    match r {
        LookupResult::Crossing(c) => {
            assert_eq!(c.child_guid, child_guid);
            assert_eq!(c.child_depth, 4);
        }
        other => panic!("expected Crossing, got {other:?}"),
    }
}

#[test]
fn insert_splits_blob_node_inline_prefix_after_shared_prefix() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let out = frame.alloc_node(NodeType::Blob).unwrap();
    let child_guid: BlobGuid = [0xBB; 16];
    install_blob_node(&mut frame, out.slot, b"bucket-a/", child_guid);
    frame.header_mut().root_slot = out.slot;

    let root = frame.header().root_slot;
    insert(&mut frame, root, b"bucket-b/file", b"meta", 8).unwrap();
    assert_eq!(get(&frame, b"bucket-b/file").as_deref(), Some(&b"meta"[..]));

    let root = frame.header().root_slot;
    assert_eq!(
        frame.slot_entry(root).unwrap().node_type(),
        Some(NodeType::Prefix)
    );
    let p = read_prefix(frame.as_ref(), root).unwrap();
    assert_eq!(&p.bytes[..p.prefix_len as usize], b"bucket-");

    let r = lookup(frame.as_ref(), root, b"bucket-a/file").unwrap();
    match r {
        LookupResult::Crossing(c) => {
            assert_eq!(c.child_guid, child_guid);
            assert_eq!(c.child_depth, b"bucket-a/".len());
        }
        other => panic!("expected Crossing, got {other:?}"),
    }
}

// ---- make_blob_from_node ----

fn read_value_from_new_blob(buf: &mut AlignedBlobBuf, key: &[u8]) -> Option<Vec<u8>> {
    let frame = BlobFrame::wrap(buf.as_mut_slice());
    let root = frame.header().root_slot;
    match lookup(frame.as_ref(), root, key).unwrap() {
        LookupResult::Found(hit) => Some(hit.value.to_vec()),
        _ => None,
    }
}

#[test]
fn make_blob_from_node_round_trips_single_leaf() {
    let (mut src_buf, _) = fresh_blob();
    let mut src_frame = BlobFrame::wrap(&mut src_buf);
    put(&mut src_frame, b"k", b"v", 1);
    let src_root = src_frame.header().root_slot;

    let new_guid: BlobGuid = [0xAA; 16];
    let mut outcome = make_blob_from_node(&src_frame, src_root, new_guid).unwrap();

    assert_eq!(
        read_value_from_new_blob(&mut outcome.buf, b"k").as_deref(),
        Some(&b"v"[..]),
    );

    let new_frame = BlobFrame::wrap(outcome.buf.as_mut_slice());
    assert_ne!(new_frame.header().root_slot, 0);
    assert_eq!(new_frame.header().blob_guid, new_guid);
}

#[test]
fn make_blob_from_node_round_trips_prefix_node4_two_leaves() {
    let (mut src_buf, _) = fresh_blob();
    let mut src_frame = BlobFrame::wrap(&mut src_buf);
    put(&mut src_frame, b"img/01.jpg", b"a", 1);
    put(&mut src_frame, b"img/02.jpg", b"b", 2);
    let src_root = src_frame.header().root_slot;

    let new_guid: BlobGuid = [0xCC; 16];
    let mut outcome = make_blob_from_node(&src_frame, src_root, new_guid).unwrap();
    assert_eq!(
        read_value_from_new_blob(&mut outcome.buf, b"img/01.jpg").as_deref(),
        Some(&b"a"[..]),
    );
    assert_eq!(
        read_value_from_new_blob(&mut outcome.buf, b"img/02.jpg").as_deref(),
        Some(&b"b"[..]),
    );
    assert_eq!(get(&src_frame, b"img/01.jpg").as_deref(), Some(&b"a"[..]));
    assert_eq!(get(&src_frame, b"img/02.jpg").as_deref(), Some(&b"b"[..]));
}

#[test]
fn make_blob_from_node_round_trips_after_node_growth_chain() {
    let (mut src_buf, _) = fresh_blob();
    let mut src_frame = BlobFrame::wrap(&mut src_buf);
    for i in 0..60u8 {
        put(&mut src_frame, &[b'q', i], &[i, i ^ 0xFF], i as u64 + 1);
    }
    let src_root = src_frame.header().root_slot;

    let mut outcome = make_blob_from_node(&src_frame, src_root, [0xEE; 16]).unwrap();
    for i in 0..60u8 {
        let key = [b'q', i];
        let expected = [i, i ^ 0xFF];
        assert_eq!(
            read_value_from_new_blob(&mut outcome.buf, &key).as_deref(),
            Some(&expected[..]),
        );
    }
}

#[test]
fn make_blob_from_node_preserves_existing_blob_node_crossings() {
    let (mut src_buf, _) = fresh_blob();
    let original_child_guid: BlobGuid = [0x77; 16];

    let bn_slot = {
        let mut src_frame = BlobFrame::wrap(&mut src_buf);
        let bn_out = src_frame.alloc_node(NodeType::Blob).unwrap();
        install_blob_node(&mut src_frame, bn_out.slot, b"data/", original_child_guid);
        src_frame.header_mut().root_slot = bn_out.slot;
        bn_out.slot
    };

    let src_frame = BlobFrame::wrap(&mut src_buf);
    assert_eq!(src_frame.header().num_ext_blobs, 1);
    let outcome = make_blob_from_node(&src_frame, bn_slot, [0x33; 16]).unwrap();

    let mut new_buf = outcome.buf;
    let new_frame = BlobFrame::wrap(new_buf.as_mut_slice());
    assert_eq!(new_frame.header().num_ext_blobs, 1);
    let new_root = new_frame.header().root_slot;
    let entry = new_frame.slot_entry(new_root).unwrap();
    assert_eq!(entry.node_type(), Some(NodeType::Blob));

    let body = new_frame.body_of_slot(new_root).unwrap();
    let bn = cast::<crate::layout::BlobNode>(body);
    assert_eq!(bn.child_blob_guid, original_child_guid);
    assert_eq!(bn.prefix_len, 5);
    assert_eq!(&bn.bytes[..5], b"data/");
}

#[test]
fn make_blob_from_node_then_lookup_yields_crossing_when_root_is_blob_node() {
    let (mut src_buf, _) = fresh_blob();
    let bn_slot = {
        let mut src_frame = BlobFrame::wrap(&mut src_buf);
        let bn_out = src_frame.alloc_node(NodeType::Blob).unwrap();
        install_blob_node(&mut src_frame, bn_out.slot, b"", [0x99; 16]);
        src_frame.header_mut().root_slot = bn_out.slot;
        bn_out.slot
    };
    let src_frame = BlobFrame::wrap(&mut src_buf);
    let mut outcome = make_blob_from_node(&src_frame, bn_slot, [0x44; 16]).unwrap();
    let new_frame = BlobFrame::wrap(outcome.buf.as_mut_slice());
    let r = lookup(
        new_frame.as_ref(),
        new_frame.header().root_slot,
        b"whatever",
    )
    .unwrap();
    match r {
        LookupResult::Crossing(c) => {
            assert_eq!(c.child_guid, [0x99; 16]);
        }
        other => panic!("expected Crossing, got {other:?}"),
    }
}

// ============================================================
// `compact_blob` — in-place blob reclaim
// ============================================================

fn aligned_from_vec(v: &[u8]) -> AlignedBlobBuf {
    let mut buf = AlignedBlobBuf::zeroed();
    buf.as_mut_slice().copy_from_slice(v);
    buf
}

#[test]
fn compact_blob_is_noop_on_empty_tree() {
    let (buf_vec, guid) = fresh_blob();
    let mut buf = aligned_from_vec(&buf_vec);
    let before = { BlobFrame::wrap(buf.as_mut_slice()).header().space_used };
    compact_blob(&mut buf).unwrap();
    let frame = BlobFrame::wrap(buf.as_mut_slice());
    let after = frame.header().space_used;
    assert!(
        after <= before + 32,
        "empty-tree compact grew unexpectedly: {before} -> {after}",
    );
    assert_eq!(frame.header().blob_guid, guid);
}

#[test]
fn blob_needs_compaction_tracks_tombstones() {
    let (buf_vec, _) = fresh_blob();
    let mut buf = aligned_from_vec(&buf_vec);

    {
        let mut frame = BlobFrame::wrap(buf.as_mut_slice());
        assert!(!blob_needs_compaction(frame.as_ref()));
        let root = frame.header().root_slot;
        insert(&mut frame, root, b"k", b"v", 1).unwrap();
        assert!(!blob_needs_compaction(frame.as_ref()));
        let root = frame.header().root_slot;
        erase(&mut frame, root, b"k").unwrap();
        assert!(blob_needs_compaction(frame.as_ref()));
    }

    compact_blob(&mut buf).unwrap();
    let frame = BlobFrame::wrap(buf.as_mut_slice());
    assert!(!blob_needs_compaction(frame.as_ref()));
}

#[test]
fn blob_needs_compaction_tracks_freed_node_slots() {
    let (buf_vec, _) = fresh_blob();
    let mut buf = aligned_from_vec(&buf_vec);
    let mut frame = BlobFrame::wrap(buf.as_mut_slice());

    let root = frame.header().root_slot;
    insert(&mut frame, root, b"k", b"v", 1).unwrap();
    assert!(!blob_needs_compaction(frame.as_ref()));

    let root = frame.header().root_slot;
    insert(&mut frame, root, b"k", &[0xAB; 128], 2).unwrap();

    assert_eq!(frame.header().tombstone_leaf_cnt, 0);
    assert!(
        blob_needs_compaction(frame.as_ref()),
        "alloc-fresh same-key updates leave the old leaf slot on a free list"
    );
}

#[test]
fn compact_blob_reclaims_extents_after_churn() {
    let (buf_vec, _) = fresh_blob();
    let mut buf = aligned_from_vec(&buf_vec);

    {
        let mut frame = BlobFrame::wrap(buf.as_mut_slice());
        for i in 0..200u32 {
            let k = format!("k{i:04}").into_bytes();
            let v = vec![0xAB; 120];
            let root = frame.header().root_slot;
            insert(&mut frame, root, &k, &v, i as u64 + 1).unwrap();
        }
        for i in 0..200u32 {
            if i % 2 == 0 {
                let k = format!("k{i:04}").into_bytes();
                let root = frame.header().root_slot;
                erase(&mut frame, root, &k).unwrap();
            }
        }
    }

    let bytes_before = { BlobFrame::wrap(buf.as_mut_slice()).header().space_used };
    compact_blob(&mut buf).unwrap();
    let bytes_after = { BlobFrame::wrap(buf.as_mut_slice()).header().space_used };
    assert!(
        bytes_before > bytes_after,
        "compact should reclaim something after 100 deletes: {bytes_before} -> {bytes_after}",
    );

    let frame = BlobFrame::wrap(buf.as_mut_slice());
    for i in 0..200u32 {
        let k = format!("k{i:04}").into_bytes();
        let v = vec![0xAB; 120];
        let root = frame.header().root_slot;
        let r = lookup(frame.as_ref(), root, &k).unwrap();
        if i % 2 == 0 {
            assert!(matches!(r, LookupResult::NotFound));
        } else {
            match r {
                LookupResult::Found(got) => assert_eq!(got.value, v.as_slice()),
                _ => panic!("survivor {k:?} missing after compact"),
            }
        }
    }
}

#[test]
fn compact_blob_preserves_guid_and_lets_inserts_continue() {
    let (buf_vec, guid) = fresh_blob();
    let mut buf = aligned_from_vec(&buf_vec);
    {
        let mut frame = BlobFrame::wrap(buf.as_mut_slice());
        for i in 0..100u32 {
            let k = format!("img/{i:04}.jpg").into_bytes();
            let v = vec![0xFE; 64];
            let root = frame.header().root_slot;
            insert(&mut frame, root, &k, &v, i as u64 + 1).unwrap();
        }
        for i in 0..50u32 {
            let k = format!("img/{i:04}.jpg").into_bytes();
            let root = frame.header().root_slot;
            erase(&mut frame, root, &k).unwrap();
        }
    }
    compact_blob(&mut buf).unwrap();

    let mut frame = BlobFrame::wrap(buf.as_mut_slice());
    assert_eq!(frame.header().blob_guid, guid);
    for i in 200..250u32 {
        let k = format!("img/{i:04}.jpg").into_bytes();
        let v = vec![0xFD; 64];
        let root = frame.header().root_slot;
        insert(&mut frame, root, &k, &v, i as u64 + 1).unwrap();
    }
    for i in 200..250u32 {
        let k = format!("img/{i:04}.jpg").into_bytes();
        let v = vec![0xFD; 64];
        let root = frame.header().root_slot;
        match lookup(frame.as_ref(), root, &k).unwrap() {
            LookupResult::Found(got) => assert_eq!(got.value, v.as_slice()),
            _ => panic!("post-compact insert {k:?} unreadable"),
        }
    }
}

/// Synthetic two-blob tree: build a normal tree, deep-clone its
/// subtree into a fresh child blob, then rewrite the root blob's
/// `header.root_slot` to point at a freshly-allocated `BlobNode`
/// referencing the child. Verifies `Tree::get` follows the
/// crossing even when it was built outside the spillover path and
/// the only child entry is the child blob's own `header.root_slot`.
///
/// Lives here (not in `tests/tree_smoke.rs`) because it needs
/// `engine::walker::*` internals (`make_blob_from_node`) that
/// stay `pub(crate)`. Organic cross-blob coverage is in the
/// integration suite's
/// `compact_merges_shrunk_child_blob_back_into_parent` and
/// friends; this test exists to keep BlobNode descent honest
/// against a synthetic shape.
#[test]
fn tree_get_and_put_follow_child_header_root_across_blob_node() {
    use crate::store::blob_store::{BlobStore, MemoryBlobStore};
    use crate::TreeBuilder;
    use std::sync::Arc;

    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    {
        let tree = TreeBuilder::new("ignored")
            .open_with_blob_store(store.clone())
            .unwrap();
        for i in 0..10u32 {
            let k = format!("k{i:02}").into_bytes();
            let v = format!("v{i}").into_bytes();
            tree.put(&k, &v).unwrap();
        }
    }

    let root_guid = [0u8; 16];
    let child_guid = [0xAA; 16];

    let mut root_buf = AlignedBlobBuf::zeroed();
    store.read_blob(root_guid, &mut root_buf).unwrap();

    let (saved_root_slot, mut child_outcome) = {
        let root_frame = BlobFrame::wrap(root_buf.as_mut_slice());
        let saved_root = root_frame.header().root_slot;
        let outcome = make_blob_from_node(&root_frame, saved_root, child_guid).unwrap();
        (saved_root, outcome)
    };

    let child_root_slot = {
        let child_frame = BlobFrame::wrap(child_outcome.buf.as_mut_slice());
        child_frame.header().root_slot
    };
    assert_ne!(
        child_root_slot, 1,
        "test needs child header root to differ from the default slot"
    );
    store.write_blob(child_guid, &child_outcome.buf).unwrap();

    replace_root_with_blob_node(&mut root_buf, child_guid);
    let _ = saved_root_slot;
    store.write_blob(root_guid, &root_buf).unwrap();

    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();
    tree.put(b"k00", b"updated").unwrap();
    assert!(tree.delete(b"k01").unwrap());
    for i in 0..10u32 {
        let k = format!("k{i:02}").into_bytes();
        let v = format!("v{i}").into_bytes();
        if i == 1 {
            assert!(tree.get(&k).unwrap().is_none());
            continue;
        }
        let expected: &[u8] = if i == 0 { b"updated" } else { &v };
        assert_eq!(
            tree.get(&k).unwrap().as_deref(),
            Some(expected),
            "post-crossing lookup failed for key {k:?}",
        );
    }
    let keys: Vec<Vec<u8>> = tree
        .range()
        .into_iter()
        .map(|entry| match entry.unwrap() {
            crate::RangeEntry::Key { key, .. } => key,
            other => panic!("unexpected range entry: {other:?}"),
        })
        .collect();
    assert_eq!(keys.len(), 9);
    assert_eq!(keys[0], b"k00");
    assert!(tree.get(b"k99").unwrap().is_none());
}

/// Same synthetic setup as the one-level test above, but with
/// two consecutive BlobNode crossings. This exercises the
/// recursive lock-coupled writer path, not just the root→child
/// fast path.
#[test]
fn tree_put_and_delete_follow_nested_child_header_roots() {
    use crate::store::blob_store::{BlobStore, MemoryBlobStore};
    use crate::TreeBuilder;
    use std::sync::Arc;

    let store: Arc<dyn BlobStore> = Arc::new(MemoryBlobStore::new());
    {
        let tree = TreeBuilder::new("ignored")
            .open_with_blob_store(store.clone())
            .unwrap();
        for i in 0..12u32 {
            let k = format!("k{i:02}").into_bytes();
            let v = format!("v{i}").into_bytes();
            tree.put(&k, &v).unwrap();
        }
    }

    let root_guid = [0u8; 16];
    let child_guid = [0xBB; 16];
    let grandchild_guid = [0xCC; 16];

    let mut root_buf = AlignedBlobBuf::zeroed();
    store.read_blob(root_guid, &mut root_buf).unwrap();

    let mut child_outcome = {
        let root_frame = BlobFrame::wrap(root_buf.as_mut_slice());
        make_blob_from_node(&root_frame, root_frame.header().root_slot, child_guid).unwrap()
    };
    let child_root_slot = {
        let child_frame = BlobFrame::wrap(child_outcome.buf.as_mut_slice());
        child_frame.header().root_slot
    };
    assert_ne!(
        child_root_slot, 1,
        "test needs child header root to differ from the default slot"
    );

    let mut child_buf = child_outcome.buf.clone();
    let mut grandchild_outcome = {
        let child_frame = BlobFrame::wrap(child_buf.as_mut_slice());
        make_blob_from_node(
            &child_frame,
            child_frame.header().root_slot,
            grandchild_guid,
        )
        .unwrap()
    };
    let grandchild_root_slot = {
        let grandchild_frame = BlobFrame::wrap(grandchild_outcome.buf.as_mut_slice());
        grandchild_frame.header().root_slot
    };
    assert_ne!(
        grandchild_root_slot, 1,
        "test needs grandchild header root to differ from the default slot"
    );

    store
        .write_blob(grandchild_guid, &grandchild_outcome.buf)
        .unwrap();
    replace_root_with_blob_node(&mut child_buf, grandchild_guid);
    store.write_blob(child_guid, &child_buf).unwrap();
    replace_root_with_blob_node(&mut root_buf, child_guid);
    store.write_blob(root_guid, &root_buf).unwrap();

    let tree = TreeBuilder::new("ignored")
        .open_with_blob_store(store.clone())
        .unwrap();
    tree.put(b"k00", b"updated").unwrap();
    tree.put(b"k99", b"new").unwrap();
    assert!(tree.delete(b"k01").unwrap());

    assert_eq!(tree.get(b"k00").unwrap().as_deref(), Some(&b"updated"[..]));
    assert!(tree.get(b"k01").unwrap().is_none());
    assert_eq!(tree.get(b"k02").unwrap().as_deref(), Some(&b"v2"[..]));
    assert_eq!(tree.get(b"k99").unwrap().as_deref(), Some(&b"new"[..]));

    let keys: Vec<Vec<u8>> = tree
        .range()
        .into_iter()
        .map(|entry| match entry.unwrap() {
            crate::RangeEntry::Key { key, .. } => key,
            other => panic!("unexpected range entry: {other:?}"),
        })
        .collect();
    assert_eq!(keys.len(), 12);
    assert_eq!(keys.first().map(Vec::as_slice), Some(&b"k00"[..]));
    assert_eq!(keys.last().map(Vec::as_slice), Some(&b"k99"[..]));
}
