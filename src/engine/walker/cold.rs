use crate::api::errors::{Error, Result};
use crate::layout::{
    BlobGuid, BlobNode, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix, BLOB_MAX_INLINE,
};
use crate::store::{BlobFrameRef, ColdBlobLookup};
use std::mem::size_of;

use super::readers::{child_offset, resolve_typed};
use super::{cast, SearchKey};

#[derive(Debug, Default)]
pub(crate) struct ColdBlobSummary {
    pub(crate) leaves: Vec<ColdLeaf>,
    pub(crate) crossings: Vec<ColdCrossing>,
}

#[derive(Debug)]
pub(crate) struct ColdLeaf {
    pub(crate) key: Vec<u8>,
    pub(crate) value: Option<Vec<u8>>,
    pub(crate) seq: u64,
}

#[derive(Debug)]
pub(crate) struct ColdCrossing {
    pub(crate) prefix: Vec<u8>,
    pub(crate) child_guid: BlobGuid,
}

pub(crate) fn summarize_blob_for_cold_index(
    frame: BlobFrameRef<'_>,
    inline_value_limit: usize,
) -> Result<ColdBlobSummary> {
    let mut out = ColdBlobSummary::default();
    let mut prefix = Vec::new();
    let root_slot = frame.header().root_slot;
    if root_slot == 0 {
        return Err(Error::node_corrupt("cold index summary: empty root slot"));
    }
    let root = child_offset(root_slot);
    summarize_node(frame, root, inline_value_limit, &mut prefix, &mut out)?;
    Ok(out)
}

pub(crate) fn lookup_cold_summary(
    summary: &ColdBlobSummary,
    key: &[u8],
    depth: usize,
) -> ColdBlobLookup {
    for leaf in &summary.leaves {
        if leaf.key == key {
            return match &leaf.value {
                Some(value) => ColdBlobLookup::Found {
                    value: value.clone(),
                    seq: leaf.seq,
                },
                None => ColdBlobLookup::Unknown,
            };
        }
    }

    let search = SearchKey::user(key);
    let mut best: Option<&ColdCrossing> = None;
    for crossing in &summary.crossings {
        if search.range_eq(depth, &crossing.prefix) {
            match best {
                Some(existing) if existing.prefix.len() >= crossing.prefix.len() => {}
                _ => best = Some(crossing),
            }
        }
    }

    if let Some(crossing) = best {
        return ColdBlobLookup::Crossing {
            child_guid: crossing.child_guid,
            child_depth: depth + crossing.prefix.len(),
        };
    }

    ColdBlobLookup::NotFound
}

fn summarize_node(
    frame: BlobFrameRef<'_>,
    off: u32,
    inline_value_limit: usize,
    prefix: &mut Vec<u8>,
    out: &mut ColdBlobSummary,
) -> Result<()> {
    let (ntype, body) = resolve_typed(frame, off)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt("cold index summary: invalid node type")),
        NodeType::EmptyRoot => Ok(()),
        NodeType::Leaf => summarize_leaf(body, inline_value_limit, out),
        NodeType::Prefix => {
            let p = cast::<Prefix>(body);
            let plen = p.prefix_len as usize;
            if plen > p.bytes.len() {
                return Err(Error::node_corrupt(
                    "cold index summary: prefix length exceeds inline buffer",
                ));
            }
            let old_len = prefix.len();
            prefix.extend_from_slice(&p.bytes[..plen]);
            summarize_node(
                frame,
                child_offset(p.child as u16),
                inline_value_limit,
                prefix,
                out,
            )?;
            prefix.truncate(old_len);
            Ok(())
        }
        NodeType::Node4 => summarize_node4(frame, body, inline_value_limit, prefix, out),
        NodeType::Node16 => summarize_node16(frame, body, inline_value_limit, prefix, out),
        NodeType::Node48 => summarize_node48(frame, body, inline_value_limit, prefix, out),
        NodeType::Node256 => summarize_node256(frame, body, inline_value_limit, prefix, out),
        NodeType::Blob => {
            let b = cast::<BlobNode>(body);
            let plen = b.prefix_len as usize;
            if plen > BLOB_MAX_INLINE {
                return Err(Error::node_corrupt(
                    "cold index summary: blob prefix length exceeds inline buffer",
                ));
            }
            let old_len = prefix.len();
            prefix.extend_from_slice(&b.bytes[..plen]);
            out.crossings.push(ColdCrossing {
                prefix: prefix.clone(),
                child_guid: b.child_blob_guid,
            });
            prefix.truncate(old_len);
            Ok(())
        }
    }
}

fn summarize_node4(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    inline_value_limit: usize,
    prefix: &mut Vec<u8>,
    out: &mut ColdBlobSummary,
) -> Result<()> {
    let n = cast::<Node4>(body);
    let count = (n.count as usize).min(4);
    for i in 0..count {
        summarize_child(
            frame,
            n.keys[i],
            n.children[i],
            inline_value_limit,
            prefix,
            out,
        )?;
    }
    Ok(())
}

fn summarize_node16(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    inline_value_limit: usize,
    prefix: &mut Vec<u8>,
    out: &mut ColdBlobSummary,
) -> Result<()> {
    let n = cast::<Node16>(body);
    let count = (n.count as usize).min(16);
    for i in 0..count {
        summarize_child(
            frame,
            n.keys[i],
            n.children[i],
            inline_value_limit,
            prefix,
            out,
        )?;
    }
    Ok(())
}

fn summarize_node48(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    inline_value_limit: usize,
    prefix: &mut Vec<u8>,
    out: &mut ColdBlobSummary,
) -> Result<()> {
    let n = cast::<Node48>(body);
    for byte in 0..=u8::MAX {
        let idx = n.index[byte as usize];
        if idx == 0 {
            continue;
        }
        let child_idx = idx as usize - 1;
        if child_idx >= 48 {
            return Err(Error::node_corrupt(
                "cold index summary: node48 child index out of range",
            ));
        }
        summarize_child(
            frame,
            byte,
            n.children[child_idx],
            inline_value_limit,
            prefix,
            out,
        )?;
    }
    Ok(())
}

fn summarize_node256(
    frame: BlobFrameRef<'_>,
    body: &[u8],
    inline_value_limit: usize,
    prefix: &mut Vec<u8>,
    out: &mut ColdBlobSummary,
) -> Result<()> {
    let n = cast::<Node256>(body);
    for byte in 0..=u8::MAX {
        let child = n.children[byte as usize];
        if child == 0 {
            continue;
        }
        summarize_child(frame, byte, child, inline_value_limit, prefix, out)?;
    }
    Ok(())
}

fn summarize_child(
    frame: BlobFrameRef<'_>,
    byte: u8,
    child: u16,
    inline_value_limit: usize,
    prefix: &mut Vec<u8>,
    out: &mut ColdBlobSummary,
) -> Result<()> {
    prefix.push(byte);
    summarize_node(frame, child_offset(child), inline_value_limit, prefix, out)?;
    prefix.pop();
    Ok(())
}

fn summarize_leaf(body: &[u8], inline_value_limit: usize, out: &mut ColdBlobSummary) -> Result<()> {
    let leaf = *cast::<Leaf>(&body[..size_of::<Leaf>()]);
    if leaf.tombstone != 0 {
        return Ok(());
    }
    let key_len = leaf.key_len as usize;
    let value_len = leaf.value_len as usize;
    let key_end = size_of::<Leaf>() + key_len;
    let value_end = key_end + value_len;
    if value_end > body.len() {
        return Err(Error::node_corrupt(
            "cold index summary: leaf key/value out of range",
        ));
    }

    let mut key = body[size_of::<Leaf>()..key_end].to_vec();
    if key.last() == Some(&0) {
        key.pop();
    }
    let value = (value_len <= inline_value_limit).then(|| body[key_end..value_end].to_vec());
    out.leaves.push(ColdLeaf {
        key,
        value,
        seq: leaf.seq,
    });
    Ok(())
}
