//! Unit tests for the walker — single-blob inserts / lookups /
//! erases + cross-blob primitives (`make_blob_from_node`,
//! BlobNode descent) + `compact_blob`.

use super::cast;
use super::readers::read_prefix;
use super::writers::write_struct_to_slot;
use super::{compact_blob, erase, insert, lookup, lookup_at, make_blob_from_node, LookupResult};
use crate::api::errors::Error;
use crate::layout::{BlobGuid, NodeType, PAGE_SIZE};
use crate::store::backend::AlignedBlobBuf;
use crate::store::BlobFrame;

fn fresh_blob() -> (Vec<u8>, BlobGuid) {
    let guid: BlobGuid = [0x11; 16];
    let mut buf = vec![0u8; PAGE_SIZE as usize];
    BlobFrame::init(&mut buf, guid).unwrap();
    (buf, guid)
}

fn put(frame: &mut BlobFrame<'_>, k: &[u8], v: &[u8], seq: u64) {
    let root = frame.header().root_slot;
    let r = insert(frame, root, k, v, seq).unwrap();
    frame.header_mut().root_slot = r.new_root_slot;
}

fn get(frame: &BlobFrame<'_>, k: &[u8]) -> Option<Vec<u8>> {
    let root = frame.header().root_slot;
    match lookup(frame.as_ref(), root, k).unwrap() {
        LookupResult::Found(v) => Some(v.to_vec()),
        LookupResult::NotFound => None,
        LookupResult::Crossing(_) => {
            panic!("walker unit tests never construct a BlobNode")
        }
    }
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
fn update_same_key_returns_previous() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"k", b"v1", 1);
    let root = frame.header().root_slot;
    let r = insert(&mut frame, root, b"k", b"v2", 2).unwrap();
    frame.header_mut().root_slot = r.new_root_slot;
    assert_eq!(r.previous.as_deref(), Some(&b"v1"[..]));
    assert_eq!(get(&frame, b"k").as_deref(), Some(&b"v2"[..]));
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

fn del(frame: &mut BlobFrame<'_>, k: &[u8]) -> Option<Vec<u8>> {
    let root = frame.header().root_slot;
    let r = erase(frame, root, k).unwrap();
    frame.header_mut().root_slot = r.new_root_slot;
    r.previous
}

#[test]
fn erase_only_leaf_returns_value_and_empties_tree() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"k", b"v", 1);
    assert_eq!(del(&mut frame, b"k").as_deref(), Some(&b"v"[..]));
    assert_eq!(get(&frame, b"k"), None);
    let root_slot = frame.header().root_slot;
    let e = frame.slot_entry(root_slot).unwrap();
    assert_eq!(e.node_type(), Some(NodeType::EmptyRoot));
}

#[test]
fn erase_missing_key_is_noop_returns_none() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"a", b"1", 1);
    assert_eq!(del(&mut frame, b"b"), None);
    assert_eq!(get(&frame, b"a").as_deref(), Some(&b"1"[..]));
}

#[test]
fn erase_one_of_two_node4_collapses_to_prefix_over_lone_leaf() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"a", b"1", 1);
    put(&mut frame, b"b", b"2", 2);
    del(&mut frame, b"a");
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
    // 5 children → grows to Node16. Erase down to 3 children →
    // shrinks to Node4. (Threshold is `count <= 3`.)
    let (mut buf, _) = fresh_blob();
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
    // Erase two so 3 children remain.
    del(&mut frame, b"k0");
    del(&mut frame, &[b'k', b'0' + 1]);
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node4),
        "Node16 with 3 remaining children should shrink to Node4",
    );
    // All survivors readable.
    for i in 2..5u8 {
        let k = [b'k', b'0' + i];
        assert_eq!(get(&frame, &k).as_deref(), Some(&[i][..]));
    }
}

#[test]
fn shrink_node48_to_node16_at_twelve_remaining() {
    // 20 children → grows through Node16 → Node48. Erase down to
    // 12 children → shrinks to Node16. Threshold `count <= 12`.
    let (mut buf, _) = fresh_blob();
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
    // Erase 8 so 12 children remain (still above the Node16
    // shrink-to-Node4 threshold of 3).
    for i in 0..8u8 {
        let k = [b'p', i];
        del(&mut frame, &k);
    }
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node16),
        "Node48 with 12 remaining children should shrink to Node16",
    );
    for i in 8..20u8 {
        let k = [b'p', i];
        assert_eq!(get(&frame, &k).as_deref(), Some(&[i][..]));
    }
}

#[test]
fn shrink_node256_to_node48_at_thirty_seven_remaining() {
    // 60 children → grows through Node48 → Node256. Erase down to
    // 37 children → shrinks to Node48. Threshold `count <= 37`.
    let (mut buf, _) = fresh_blob();
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
    // Erase 23 so 37 children remain.
    for i in 0..23u8 {
        let k = [b'q', i];
        del(&mut frame, &k);
    }
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node48),
        "Node256 with 37 remaining children should shrink to Node48",
    );
    for i in 23..60u8 {
        let k = [b'q', i];
        let v = [i, i ^ 0xFF];
        assert_eq!(get(&frame, &k).as_deref(), Some(&v[..]));
    }
}

#[test]
fn shrink_chain_node256_node48_node16_node4() {
    // One sustained churn: grow up through Node256, then erase
    // back down past every shrink threshold in order. Confirms
    // the shrink rules compose end-to-end.
    let (mut buf, _) = fresh_blob();
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

    // Erase 23 → Node48.
    for i in 0..23u8 {
        del(&mut frame, &[b'q', i]);
    }
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node48),
    );

    // Erase another 25 (total 48) so 12 children remain → Node16.
    for i in 23..48u8 {
        del(&mut frame, &[b'q', i]);
    }
    assert_eq!(
        frame
            .slot_entry(inner_node_slot(&frame))
            .unwrap()
            .node_type(),
        Some(NodeType::Node16),
    );

    // Erase another 9 (total 57) so 3 children remain → Node4.
    for i in 48..57u8 {
        del(&mut frame, &[b'q', i]);
    }
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
}

#[test]
fn erase_all_returns_to_empty_root() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let pairs = [
        (&b"alpha"[..], &b"A"[..]),
        (&b"beta"[..], &b"B"[..]),
        (&b"gamma"[..], &b"G"[..]),
    ];
    for (i, (k, v)) in pairs.iter().enumerate() {
        put(&mut frame, k, v, i as u64 + 1);
    }
    for (k, v) in &pairs {
        assert_eq!(del(&mut frame, k).as_deref(), Some(*v));
    }
    let root_slot = frame.header().root_slot;
    assert_eq!(
        frame.slot_entry(root_slot).unwrap().node_type(),
        Some(NodeType::EmptyRoot)
    );
}

#[test]
fn erase_through_prefix_keeps_other_branches() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    put(&mut frame, b"img/01.jpg", b"a", 1);
    put(&mut frame, b"img/02.jpg", b"b", 2);
    put(&mut frame, b"img/03.jpg", b"c", 3);
    assert_eq!(del(&mut frame, b"img/02.jpg").as_deref(), Some(&b"b"[..]));
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
    let mut frame = BlobFrame::wrap(&mut buf);
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..100u32)
        .map(|i| {
            (
                format!("k{i:04}").into_bytes(),
                format!("v{i}").into_bytes(),
            )
        })
        .collect();
    for (i, (k, v)) in pairs.iter().enumerate() {
        put(&mut frame, k, v, i as u64 + 1);
    }
    for (k, v) in &pairs {
        assert_eq!(del(&mut frame, k).as_deref(), Some(&v[..]));
    }
    for (k, _) in &pairs {
        assert_eq!(get(&frame, k), None);
    }
    let root_slot = frame.header().root_slot;
    assert_eq!(
        frame.slot_entry(root_slot).unwrap().node_type(),
        Some(NodeType::EmptyRoot)
    );
}

// ============================================================
// Stage 2d phase A — multi-blob lookup + make_blob_from_node
// ============================================================

fn install_blob_node(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    prefix: &[u8],
    child_guid: BlobGuid,
    entry: u32,
) {
    let bn = crate::layout::BlobNode::new(prefix, child_guid, entry);
    write_struct_to_slot(frame, slot, &bn).unwrap();
}

#[test]
fn lookup_blob_node_emits_crossing_on_match() {
    let (mut buf, _) = fresh_blob();
    let mut frame = BlobFrame::wrap(&mut buf);
    let out = frame.alloc_node(NodeType::Blob).unwrap();
    let child_guid: BlobGuid = [0xAA; 16];
    install_blob_node(&mut frame, out.slot, b"img/", child_guid, 42);
    frame.header_mut().root_slot = out.slot;

    let r = lookup(frame.as_ref(), out.slot, b"img/01.jpg").unwrap();
    match r {
        LookupResult::Crossing(c) => {
            assert_eq!(c.child_guid, child_guid);
            assert_eq!(c.child_slot, 42);
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
    install_blob_node(&mut frame, out.slot, b"img/", [0xAA; 16], 1);
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
    assert!(matches!(r0, LookupResult::Found(v) if v == b"v1"));

    let r1 = lookup_at(frame.as_ref(), root, b"img/01.jpg", 0).unwrap();
    assert!(matches!(r1, LookupResult::Found(v) if v == b"v1"));
}

// ---- make_blob_from_node ----

fn read_value_from_new_blob(buf: &mut AlignedBlobBuf, key: &[u8]) -> Option<Vec<u8>> {
    let frame = BlobFrame::wrap(buf.as_mut_slice());
    let root = frame.header().root_slot;
    match lookup(frame.as_ref(), root, key).unwrap() {
        LookupResult::Found(v) => Some(v.to_vec()),
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
    assert_eq!(new_frame.header().root_slot, outcome.entry_slot);
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
        install_blob_node(
            &mut src_frame,
            bn_out.slot,
            b"data/",
            original_child_guid,
            7,
        );
        src_frame.header_mut().root_slot = bn_out.slot;
        bn_out.slot
    };

    let src_frame = BlobFrame::wrap(&mut src_buf);
    let outcome = make_blob_from_node(&src_frame, bn_slot, [0x33; 16]).unwrap();

    let mut new_buf = outcome.buf;
    let new_frame = BlobFrame::wrap(new_buf.as_mut_slice());
    let new_root = new_frame.header().root_slot;
    let entry = new_frame.slot_entry(new_root).unwrap();
    assert_eq!(entry.node_type(), Some(NodeType::Blob));

    let body = new_frame.body_of_slot(new_root).unwrap();
    let bn = cast::<crate::layout::BlobNode>(body);
    assert_eq!(bn.child_blob_guid, original_child_guid);
    assert_eq!(bn.child_entry_ptr, 7);
    assert_eq!(bn.prefix_len, 5);
    assert_eq!(&bn.bytes[..5], b"data/");
}

#[test]
fn make_blob_from_node_then_lookup_yields_crossing_when_root_is_blob_node() {
    let (mut src_buf, _) = fresh_blob();
    let bn_slot = {
        let mut src_frame = BlobFrame::wrap(&mut src_buf);
        let bn_out = src_frame.alloc_node(NodeType::Blob).unwrap();
        install_blob_node(&mut src_frame, bn_out.slot, b"", [0x99; 16], 11);
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
            assert_eq!(c.child_slot, 11);
        }
        other => panic!("expected Crossing, got {other:?}"),
    }
}

// ============================================================
// Stage 6 (reclaim) — compact_blob
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
    let stats = compact_blob(&mut buf).unwrap();
    assert!(
        stats.bytes_after <= before + 32,
        "empty-tree compact grew unexpectedly: {} -> {}",
        stats.bytes_before,
        stats.bytes_after,
    );
    let frame = BlobFrame::wrap(buf.as_mut_slice());
    assert_eq!(frame.header().blob_guid, guid);
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
            let out = insert(&mut frame, root, &k, &v, i as u64 + 1).unwrap();
            frame.header_mut().root_slot = out.new_root_slot;
        }
        for i in 0..200u32 {
            if i % 2 == 0 {
                let k = format!("k{i:04}").into_bytes();
                let root = frame.header().root_slot;
                let out = erase(&mut frame, root, &k).unwrap();
                frame.header_mut().root_slot = out.new_root_slot;
            }
        }
    }

    let stats = compact_blob(&mut buf).unwrap();
    assert!(
        stats.bytes_reclaimed > 0,
        "compact should reclaim something after 100 deletes: {stats:?}",
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
                LookupResult::Found(got) => assert_eq!(got, v),
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
            let out = insert(&mut frame, root, &k, &v, i as u64 + 1).unwrap();
            frame.header_mut().root_slot = out.new_root_slot;
        }
        for i in 0..50u32 {
            let k = format!("img/{i:04}.jpg").into_bytes();
            let root = frame.header().root_slot;
            let out = erase(&mut frame, root, &k).unwrap();
            frame.header_mut().root_slot = out.new_root_slot;
        }
    }
    compact_blob(&mut buf).unwrap();

    let mut frame = BlobFrame::wrap(buf.as_mut_slice());
    assert_eq!(frame.header().blob_guid, guid);
    for i in 200..250u32 {
        let k = format!("img/{i:04}.jpg").into_bytes();
        let v = vec![0xFD; 64];
        let root = frame.header().root_slot;
        let out = insert(&mut frame, root, &k, &v, i as u64 + 1).unwrap();
        frame.header_mut().root_slot = out.new_root_slot;
    }
    for i in 200..250u32 {
        let k = format!("img/{i:04}.jpg").into_bytes();
        let v = vec![0xFD; 64];
        let root = frame.header().root_slot;
        match lookup(frame.as_ref(), root, &k).unwrap() {
            LookupResult::Found(got) => assert_eq!(got, v),
            _ => panic!("post-compact insert {k:?} unreadable"),
        }
    }
}
