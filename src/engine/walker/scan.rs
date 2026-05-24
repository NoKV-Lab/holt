//! Tree-wide traversal — enumerate every blob reachable from a
//! starting root, in BFS order, by scanning each blob's tree shape
//! for [`NodeType::Blob`] crossings.
//!
//! Used by [`crate::Tree::stats`] and [`crate::Tree::compact`]
//! to fan out across the whole on-disk tree without each caller
//! having to reimplement cross-blob descent.

use std::collections::{HashSet, VecDeque};

use crate::api::errors::{Error, Result};
use crate::layout::{
    BlobGuid, BlobNode, Node16, Node256, Node4, Node48, NodeType, Prefix, BLOB_MAX_INLINE,
    PREFIX_MAX_INLINE,
};
use crate::store::{BlobFrameRef, BufferManager};

use super::cast;
use super::readers::resolve_typed;

/// One reachable blob plus its cross-blob depth from the root.
///
/// Depth `0` is the root blob. Each [`NodeType::Blob`] crossing
/// increments the depth by one. This is a blob-graph metric, not
/// a per-node ART depth trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobTopologyEntry {
    /// GUID identifying the reachable blob.
    pub guid: BlobGuid,
    /// Number of BlobNode crossings from the root to this blob.
    pub depth: u32,
}

/// Return every blob GUID reachable from `root_guid` (including
/// `root_guid` itself), in BFS order.
///
/// Each blob is pinned + read under a shared guard exactly once; no
/// blob bytes are copied. The returned vector's first element is
/// always `root_guid`.
///
/// Uses `BufferManager::pin_scan`, which bumps cache hit/miss
/// counters without promoting blobs in the eviction policy. Callers
/// on the observability path (`Tree::stats`, metrics scrapes)
/// should use [`collect_blob_topology_silent`] instead to avoid
/// polluting the counters they're about to report.
pub fn collect_blob_guids(bm: &BufferManager, root_guid: BlobGuid) -> Result<Vec<BlobGuid>> {
    collect_blob_topology(bm, root_guid)
        .map(|entries| entries.into_iter().map(|entry| entry.guid).collect())
}

/// Return every reachable blob plus its blob-graph depth from
/// `root_guid`, in BFS order.
fn collect_blob_topology(
    bm: &BufferManager,
    root_guid: BlobGuid,
) -> Result<Vec<BlobTopologyEntry>> {
    collect_blob_topology_inner(bm, root_guid, /*silent=*/ false)
}

/// Same as [`collect_blob_topology`] but uses
/// `BufferManager::pin_silent`, so observability walks do not
/// perturb cache counters or eviction recency.
pub fn collect_blob_topology_silent(
    bm: &BufferManager,
    root_guid: BlobGuid,
) -> Result<Vec<BlobTopologyEntry>> {
    collect_blob_topology_inner(bm, root_guid, /*silent=*/ true)
}

/// Return the blobs needed to read `prefix`.
///
/// The root blob is always included. Once the walk enters `prefix`,
/// it switches to a topology-only scan instead of decoding leaves.
pub fn collect_prefix_blob_topology_silent(
    bm: &BufferManager,
    root_guid: BlobGuid,
    prefix: &[u8],
) -> Result<Vec<BlobTopologyEntry>> {
    if prefix.is_empty() {
        return collect_blob_topology_silent(bm, root_guid);
    }

    let mut all = vec![BlobTopologyEntry {
        guid: root_guid,
        depth: 0,
    }];
    let mut seen = HashSet::from([root_guid]);
    let mut queue = VecDeque::from([PrefixQueueEntry {
        guid: root_guid,
        depth: 0,
        mode: PrefixQueueMode::Match { path: Vec::new() },
    }]);

    while let Some(entry) = queue.pop_front() {
        let pin = bm.pin_silent(entry.guid)?;
        let mut children = Vec::new();
        {
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let root_slot = frame.header().root_slot;
            match entry.mode {
                PrefixQueueMode::All => collect_all_children(frame, root_slot, &mut children)?,
                PrefixQueueMode::Match { mut path } => {
                    scan_prefix_subtree(frame, root_slot, &mut path, prefix, &mut children)?;
                }
            }
        }

        for child in children {
            if seen.insert(child.guid) {
                let depth = entry.depth.saturating_add(1);
                all.push(BlobTopologyEntry {
                    guid: child.guid,
                    depth,
                });
                queue.push_back(PrefixQueueEntry {
                    guid: child.guid,
                    depth,
                    mode: child.mode,
                });
            }
        }
    }

    Ok(all)
}

fn collect_blob_topology_inner(
    bm: &BufferManager,
    root_guid: BlobGuid,
    silent: bool,
) -> Result<Vec<BlobTopologyEntry>> {
    let mut all = vec![BlobTopologyEntry {
        guid: root_guid,
        depth: 0,
    }];
    let mut queue: VecDeque<BlobTopologyEntry> = VecDeque::from([BlobTopologyEntry {
        guid: root_guid,
        depth: 0,
    }]);
    while let Some(entry) = queue.pop_front() {
        let pin = if silent {
            bm.pin_silent(entry.guid)?
        } else {
            bm.pin_scan(entry.guid)?
        };
        let mut found = Vec::new();
        {
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let root_slot = frame.header().root_slot;
            scan_subtree(frame, root_slot, &mut found)?;
        }
        for child_guid in found {
            let child = BlobTopologyEntry {
                guid: child_guid,
                depth: entry.depth.saturating_add(1),
            };
            all.push(child);
            queue.push_back(child);
        }
    }
    Ok(all)
}

#[derive(Debug)]
struct PrefixQueueEntry {
    guid: BlobGuid,
    depth: u32,
    mode: PrefixQueueMode,
}

#[derive(Debug)]
enum PrefixQueueMode {
    All,
    Match { path: Vec<u8> },
}

#[derive(Debug)]
struct PrefixChild {
    guid: BlobGuid,
    mode: PrefixQueueMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixRelation {
    All,
    Continue,
    Skip,
}

fn prefix_relation(path: &[u8], prefix: &[u8]) -> PrefixRelation {
    if path.starts_with(prefix) {
        PrefixRelation::All
    } else if prefix.starts_with(path) {
        PrefixRelation::Continue
    } else {
        PrefixRelation::Skip
    }
}

fn collect_all_children(
    frame: BlobFrameRef<'_>,
    slot: u16,
    out: &mut Vec<PrefixChild>,
) -> Result<()> {
    let mut guids = Vec::new();
    scan_subtree(frame, slot, &mut guids)?;
    out.extend(guids.into_iter().map(|guid| PrefixChild {
        guid,
        mode: PrefixQueueMode::All,
    }));
    Ok(())
}

fn collect_matching_child(
    child_guid: BlobGuid,
    path: &[u8],
    prefix: &[u8],
    out: &mut Vec<PrefixChild>,
) {
    match prefix_relation(path, prefix) {
        PrefixRelation::All => out.push(PrefixChild {
            guid: child_guid,
            mode: PrefixQueueMode::All,
        }),
        PrefixRelation::Continue => out.push(PrefixChild {
            guid: child_guid,
            mode: PrefixQueueMode::Match {
                path: path.to_vec(),
            },
        }),
        PrefixRelation::Skip => {}
    }
}

fn scan_prefix_subtree(
    frame: BlobFrameRef<'_>,
    slot: u16,
    path: &mut Vec<u8>,
    prefix: &[u8],
    out: &mut Vec<PrefixChild>,
) -> Result<()> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::scan::scan_prefix_subtree: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot | NodeType::Leaf => Ok(()),
        NodeType::Prefix => {
            let p = cast::<Prefix>(body);
            let plen = p.prefix_len as usize;
            if plen > PREFIX_MAX_INLINE {
                return Err(Error::node_corrupt(
                    "walker::scan::scan_prefix_subtree: Prefix prefix_len exceeds inline",
                ));
            }
            let len = path.len();
            path.extend_from_slice(&p.bytes[..plen]);
            match prefix_relation(path, prefix) {
                PrefixRelation::All => collect_all_children(frame, p.child as u16, out)?,
                PrefixRelation::Continue => {
                    scan_prefix_subtree(frame, p.child as u16, path, prefix, out)?;
                }
                PrefixRelation::Skip => {}
            }
            path.truncate(len);
            Ok(())
        }
        NodeType::Node4 => {
            let n = cast::<Node4>(body);
            let count = (n.count as usize).min(4);
            for i in 0..count {
                scan_prefix_branch(frame, n.keys[i], n.children[i] as u16, path, prefix, out)?;
            }
            Ok(())
        }
        NodeType::Node16 => {
            let n = cast::<Node16>(body);
            let count = (n.count as usize).min(16);
            for i in 0..count {
                scan_prefix_branch(frame, n.keys[i], n.children[i] as u16, path, prefix, out)?;
            }
            Ok(())
        }
        NodeType::Node48 => {
            let n = cast::<Node48>(body);
            for byte in 0..=u8::MAX {
                let idx = n.index[byte as usize];
                if idx == 0 {
                    continue;
                }
                let ci = idx as usize - 1;
                if ci >= 48 {
                    return Err(Error::node_corrupt(
                        "walker::scan::scan_prefix_subtree: Node48 index out of range",
                    ));
                }
                scan_prefix_branch(frame, byte, n.children[ci] as u16, path, prefix, out)?;
            }
            Ok(())
        }
        NodeType::Node256 => {
            let n = cast::<Node256>(body);
            for (byte, child) in n.children.iter().enumerate() {
                if *child != 0 {
                    scan_prefix_branch(frame, byte as u8, *child as u16, path, prefix, out)?;
                }
            }
            Ok(())
        }
        NodeType::Blob => {
            let b = cast::<BlobNode>(body);
            let plen = b.prefix_len as usize;
            if plen > BLOB_MAX_INLINE {
                return Err(Error::node_corrupt(
                    "walker::scan::scan_prefix_subtree: BlobNode prefix_len exceeds inline",
                ));
            }
            let len = path.len();
            path.extend_from_slice(&b.bytes[..plen]);
            collect_matching_child(b.child_blob_guid, path, prefix, out);
            path.truncate(len);
            Ok(())
        }
    }
}

fn scan_prefix_branch(
    frame: BlobFrameRef<'_>,
    byte: u8,
    child_slot: u16,
    path: &mut Vec<u8>,
    prefix: &[u8],
    out: &mut Vec<PrefixChild>,
) -> Result<()> {
    path.push(byte);
    match prefix_relation(path, prefix) {
        PrefixRelation::All => collect_all_children(frame, child_slot, out)?,
        PrefixRelation::Continue => scan_prefix_subtree(frame, child_slot, path, prefix, out)?,
        PrefixRelation::Skip => {}
    }
    path.pop();
    Ok(())
}

fn scan_subtree(frame: BlobFrameRef<'_>, slot: u16, out: &mut Vec<BlobGuid>) -> Result<()> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::node_corrupt(
            "walker::scan::scan_subtree: hit NodeType::Invalid",
        )),
        NodeType::EmptyRoot | NodeType::Leaf => Ok(()),
        NodeType::Prefix => {
            let p = cast::<Prefix>(body);
            scan_subtree(frame, p.child as u16, out)
        }
        NodeType::Node4 => {
            let n = cast::<Node4>(body);
            let count = (n.count as usize).min(4);
            for i in 0..count {
                scan_subtree(frame, n.children[i] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node16 => {
            let n = cast::<Node16>(body);
            let count = (n.count as usize).min(16);
            for i in 0..count {
                scan_subtree(frame, n.children[i] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node48 => {
            let n = cast::<Node48>(body);
            for i in 0..256usize {
                let idx = n.index[i];
                if idx == 0 {
                    continue;
                }
                let ci = idx as usize - 1;
                if ci >= 48 {
                    return Err(Error::node_corrupt(
                        "walker::scan::scan_subtree: Node48 index out of range",
                    ));
                }
                scan_subtree(frame, n.children[ci] as u16, out)?;
            }
            Ok(())
        }
        NodeType::Node256 => {
            let n = cast::<Node256>(body);
            for c in n.children {
                if c != 0 {
                    scan_subtree(frame, c as u16, out)?;
                }
            }
            Ok(())
        }
        NodeType::Blob => {
            let b = cast::<BlobNode>(body);
            let plen = b.prefix_len as usize;
            if plen > BLOB_MAX_INLINE {
                return Err(Error::node_corrupt(
                    "walker::scan::scan_subtree: BlobNode prefix_len exceeds inline",
                ));
            }
            out.push(b.child_blob_guid);
            Ok(())
        }
    }
}
