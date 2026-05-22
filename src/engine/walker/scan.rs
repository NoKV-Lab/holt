//! Tree-wide traversal — enumerate every blob reachable from a
//! starting root, in BFS order, by scanning each blob's tree shape
//! for [`NodeType::Blob`] crossings.
//!
//! Used by [`crate::api::Tree::stats`] and [`crate::api::Tree::compact`]
//! to fan out across the whole on-disk tree without each caller
//! having to reimplement cross-blob descent.

use crate::api::errors::{Error, Result};
use crate::layout::{
    BlobGuid, BlobNode, Node16, Node256, Node4, Node48, NodeType, Prefix, BLOB_MAX_INLINE,
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
/// Uses `BufferManager::pin`, which bumps cache hit/miss
/// counters and refreshes `last_touched`. Callers on the
/// observability path (`Tree::stats`, metrics scrapes) should
/// use [`collect_blob_topology_silent`] instead to avoid the
/// scrape polluting the counters it's about to report.
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

fn collect_blob_topology_inner(
    bm: &BufferManager,
    root_guid: BlobGuid,
    silent: bool,
) -> Result<Vec<BlobTopologyEntry>> {
    use std::collections::VecDeque;

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
            bm.pin(entry.guid)?
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
