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

/// Return every blob GUID reachable from `root_guid` (including
/// `root_guid` itself), in BFS order.
///
/// Each blob is pinned + read under a shared guard exactly once; no
/// blob bytes are copied. The returned vector's first element is
/// always `root_guid`.
pub fn collect_blob_guids(bm: &BufferManager, root_guid: BlobGuid) -> Result<Vec<BlobGuid>> {
    let mut all = vec![root_guid];
    let mut queue: Vec<BlobGuid> = vec![root_guid];
    while let Some(guid) = queue.pop() {
        let pin = bm.pin(guid)?;
        let mut found = Vec::new();
        {
            let guard = pin.read();
            let frame = BlobFrameRef::wrap(guard.as_slice());
            let root_slot = frame.header().root_slot;
            scan_subtree(frame, root_slot, &mut found)?;
        }
        for child_guid in found {
            all.push(child_guid);
            queue.push(child_guid);
        }
    }
    Ok(all)
}

fn scan_subtree(frame: BlobFrameRef<'_>, slot: u16, out: &mut Vec<BlobGuid>) -> Result<()> {
    let (ntype, body) = resolve_typed(frame, slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::NodeCorrupt {
            context: "walker::scan::scan_subtree: hit NodeType::Invalid",
        }),
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
                    return Err(Error::NodeCorrupt {
                        context: "walker::scan::scan_subtree: Node48 index out of range",
                    });
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
                return Err(Error::NodeCorrupt {
                    context: "walker::scan::scan_subtree: BlobNode prefix_len exceeds inline",
                });
            }
            out.push(b.child_blob_guid);
            Ok(())
        }
    }
}
