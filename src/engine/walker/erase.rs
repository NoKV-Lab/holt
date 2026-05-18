//! Erase path — `erase` / `erase_multi` + recursive `erase_at`
//! dispatch + per-NodeType arms + collapse-on-lone-child rewiring
//! + `erase_at_blob_node` cross-blob arm.

use crate::api::errors::{Error, Result};
use crate::layout::{BlobGuid, BlobNode, NodeType, BLOB_MAX_INLINE};
use crate::store::backend::Backend;
use crate::store::{BlobFrame, BufferManager};

use super::cast;
use super::readers::{
    ntype_of, read_leaf_kv, read_node16, read_node256, read_node4, read_node48, read_prefix,
};
use super::types::{EraseOutcome, EraseReturn, EraseSignal};
use super::writers::{
    finish_inner_with_sorted, inner_find_child, inner_update_child, set_prefix_child,
    shrink_node16_to_node4, shrink_node256_to_node48, shrink_node48_to_node16, write_prefix_chain,
    write_struct_to_slot, SHRINK_NODE16_TO_NODE4_AT, SHRINK_NODE256_TO_NODE48_AT,
    SHRINK_NODE48_TO_NODE16_AT,
};

// ---------- public entry points ----------

/// Single-blob erase. Surfaces [`Error::NotYetImplemented`] if the
/// descent reaches a [`NodeType::Blob`] crossing — Stage 2d
/// callers wanting cross-blob erase should use [`erase_multi`].
///
/// Returns the new root slot (caller updates `header.root_slot`)
/// and the prior value if the key was present. If `key` was not in
/// the tree, `previous` is `None` and `new_root_slot == root_slot`.
pub fn erase(frame: &mut BlobFrame<'_>, root_slot: u16, key: &[u8]) -> Result<EraseOutcome> {
    let r = erase_at(None, frame, root_slot, key, 0)?;
    let new_root = resolve_new_root_after_erase(frame, root_slot, &r.signal)?;
    Ok(EraseOutcome {
        new_root_slot: new_root,
        previous: r.previous,
    })
}

/// Multi-blob erase. Pins the root via the [`BufferManager`] and
/// walks across [`NodeType::Blob`] crossings. When a child blob
/// becomes empty (signal = `SubtreeGone`) the parent's `BlobNode`
/// is freed and the orphaned child blob is dropped from the BM
/// cache + the inner backend in the same step — no GC pass needed.
pub fn erase_multi(bm: &BufferManager, root_guid: BlobGuid, key: &[u8]) -> Result<EraseOutcome> {
    let root_pin = bm.pin(root_guid)?;
    // One continuous exclusive critical section — see
    // `insert_multi` for why per-phase guard drops would race.
    let mut guard = root_pin.write();

    let r = {
        let mut frame = BlobFrame::wrap(guard.as_mut_slice());
        let root_slot = frame.header().root_slot;
        erase_at(Some(bm), &mut frame, root_slot, key, 0)?
    };
    let new_root = {
        let mut frame = BlobFrame::wrap(guard.as_mut_slice());
        let root_slot = frame.header().root_slot;
        resolve_new_root_after_erase(&mut frame, root_slot, &r.signal)?
    };
    {
        let mut frame = BlobFrame::wrap(guard.as_mut_slice());
        frame.header_mut().root_slot = new_root;
    }
    Ok(EraseOutcome {
        new_root_slot: new_root,
        previous: r.previous,
    })
}

fn resolve_new_root_after_erase(
    frame: &mut BlobFrame<'_>,
    root_slot: u16,
    signal: &EraseSignal,
) -> Result<u16> {
    match signal {
        EraseSignal::Unchanged => Ok(root_slot),
        EraseSignal::Replaced(s) => Ok(*s),
        EraseSignal::SubtreeGone => {
            // The whole tree is empty — re-seed the EmptyRoot
            // sentinel so subsequent lookups return NotFound and
            // subsequent inserts replace the sentinel cleanly.
            let out = frame.alloc_node(NodeType::EmptyRoot)?;
            Ok(out.slot)
        }
    }
}

// ---------- recursive dispatch ----------

pub(super) fn erase_at(
    bm: Option<&BufferManager>,
    frame: &mut BlobFrame<'_>,
    slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    let ntype = ntype_of(frame.as_ref(), slot)?;
    match ntype {
        NodeType::Invalid => Err(Error::NodeCorrupt {
            context: "walker::erase_at: hit NodeType::Invalid",
        }),
        NodeType::EmptyRoot => Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        }),
        NodeType::Leaf => erase_at_leaf(frame, slot, key),
        NodeType::Prefix => erase_at_prefix(bm, frame, slot, key, depth),
        NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
            erase_at_inner(bm, frame, slot, ntype, key, depth)
        }
        NodeType::Blob => match bm {
            Some(b) => erase_at_blob_node(b, frame, slot, key, depth),
            None => Err(Error::NotYetImplemented(
                "walker::erase_at: BlobNode crossing requires BufferManager — use erase_multi",
            )),
        },
    }
}

fn erase_at_leaf(frame: &mut BlobFrame<'_>, leaf_slot: u16, key: &[u8]) -> Result<EraseReturn> {
    let (existing_key, existing_value) = read_leaf_kv(frame.as_ref(), leaf_slot)?;
    if existing_key != key {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    }
    frame.free_node(leaf_slot)?;
    Ok(EraseReturn {
        signal: EraseSignal::SubtreeGone,
        previous: Some(existing_value),
    })
}

fn erase_at_prefix(
    bm: Option<&BufferManager>,
    frame: &mut BlobFrame<'_>,
    pfx_slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    let p = read_prefix(frame.as_ref(), pfx_slot)?;
    let plen = p.prefix_len as usize;
    let prefix_bytes_copy: Vec<u8> = p.bytes[..plen].to_vec();
    let child_slot = p.child as u16;

    if depth + plen > key.len() || prefix_bytes_copy[..] != key[depth..depth + plen] {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    }

    let r = erase_at(bm, frame, child_slot, key, depth + plen)?;
    match r.signal {
        EraseSignal::Unchanged => Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: r.previous,
        }),
        EraseSignal::Replaced(new_child) => {
            set_prefix_child(frame, pfx_slot, u32::from(new_child))?;
            Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                previous: r.previous,
            })
        }
        EraseSignal::SubtreeGone => {
            frame.free_node(pfx_slot)?;
            Ok(EraseReturn {
                signal: EraseSignal::SubtreeGone,
                previous: r.previous,
            })
        }
    }
}

fn erase_at_inner(
    bm: Option<&BufferManager>,
    frame: &mut BlobFrame<'_>,
    inner_slot: u16,
    ntype: NodeType,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    if depth >= key.len() {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    }
    let byte = key[depth];
    let Some(child) = inner_find_child(frame, inner_slot, ntype, byte)? else {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    };

    let r = erase_at(bm, frame, child, key, depth + 1)?;
    match r.signal {
        EraseSignal::Unchanged => Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: r.previous,
        }),
        EraseSignal::Replaced(new_child) => {
            inner_update_child(frame, inner_slot, ntype, byte, u32::from(new_child))?;
            Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                previous: r.previous,
            })
        }
        EraseSignal::SubtreeGone => {
            let sig = inner_remove_child_and_collapse(frame, inner_slot, ntype, byte)?;
            Ok(EraseReturn {
                signal: sig,
                previous: r.previous,
            })
        }
    }
}

/// Remove `byte` from `slot`'s child set. After removal:
/// - `count == 0` → free the inner node, signal `SubtreeGone`.
/// - `count == 1` → free the inner node, wrap the lone child in a
///   `Prefix([surviving_byte])` so descendant depth indexing stays
///   valid, signal `Replaced(prefix_slot)`.
/// - `count` dropped to the shrink threshold for the current
///   `NodeType` → allocate the next-smaller variant
///   (`Node256→Node48`, `Node48→Node16`, `Node16→Node4`), copy the
///   remaining children across, free the old slot, signal
///   `Replaced(new_slot)`. Thresholds (12, 37, 3) leave hysteresis
///   so a single re-insert doesn't immediately grow back.
/// - otherwise → rewrite the body in place, signal `Unchanged`.
///
/// The `Prefix` wrap on lone-child collapse is load-bearing: an
/// inner-node child sits one byte deeper in the descent than its
/// parent, so dropping the inner node without re-inserting its
/// pointing-byte breaks every leaf below it.
#[allow(clippy::too_many_lines)] // intentional — one match over 4 NodeTypes
fn inner_remove_child_and_collapse(
    frame: &mut BlobFrame<'_>,
    slot: u16,
    ntype: NodeType,
    byte: u8,
) -> Result<EraseSignal> {
    match ntype {
        NodeType::Node4 => {
            let mut n = read_node4(frame.as_ref(), slot)?;
            let count = (n.count as usize).min(4);
            let mut idx = None;
            for i in 0..count {
                if n.keys[i] == byte {
                    idx = Some(i);
                    break;
                }
            }
            let i = idx.ok_or(Error::NodeCorrupt {
                context: "inner_remove_child_and_collapse: byte not present (Node4)",
            })?;
            for j in i..count - 1 {
                n.keys[j] = n.keys[j + 1];
                n.children[j] = n.children[j + 1];
            }
            n.keys[count - 1] = 0;
            n.children[count - 1] = 0;
            n.count -= 1;
            finish_inner_with_sorted(frame, slot, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node16 => {
            let mut n = read_node16(frame.as_ref(), slot)?;
            let count = (n.count as usize).min(16);
            let mut idx = None;
            for i in 0..count {
                if n.keys[i] == byte {
                    idx = Some(i);
                    break;
                }
            }
            let i = idx.ok_or(Error::NodeCorrupt {
                context: "inner_remove_child_and_collapse: byte not present (Node16)",
            })?;
            for j in i..count - 1 {
                n.keys[j] = n.keys[j + 1];
                n.children[j] = n.children[j + 1];
            }
            n.keys[count - 1] = 0;
            n.children[count - 1] = 0;
            n.count -= 1;

            // Try shrinking to Node4 before the count<=1 paths so
            // that the freed Node16 slot is the only old slot we
            // hand back to the free list (the Prefix-wrap below
            // already does that for count==1).
            if n.count >= 2 && n.count <= SHRINK_NODE16_TO_NODE4_AT {
                let shrunk = shrink_node16_to_node4(frame, slot, n)?;
                return Ok(EraseSignal::Replaced(shrunk));
            }
            finish_inner_with_sorted(frame, slot, n.count, &n, n.keys[0], n.children[0])
        }
        NodeType::Node48 => {
            let mut n = read_node48(frame.as_ref(), slot)?;
            let ci = n.index[byte as usize];
            if ci == 0 {
                return Err(Error::NodeCorrupt {
                    context: "inner_remove_child_and_collapse: byte not present (Node48)",
                });
            }
            n.children[(ci as usize) - 1] = 0;
            n.index[byte as usize] = 0;
            n.count -= 1;

            if n.count == 0 {
                frame.free_node(slot)?;
                return Ok(EraseSignal::SubtreeGone);
            }
            if n.count == 1 {
                let (surviving_byte, surviving_child) = {
                    let mut found = (0u8, 0u32);
                    for b in 0..256usize {
                        if n.index[b] != 0 {
                            found = (b as u8, n.children[(n.index[b] as usize) - 1]);
                            break;
                        }
                    }
                    found
                };
                frame.free_node(slot)?;
                let new_slot =
                    write_prefix_chain(frame, &[surviving_byte], surviving_child as u16)?;
                return Ok(EraseSignal::Replaced(new_slot));
            }
            if n.count <= SHRINK_NODE48_TO_NODE16_AT {
                let shrunk = shrink_node48_to_node16(frame, slot, n)?;
                return Ok(EraseSignal::Replaced(shrunk));
            }
            write_struct_to_slot(frame, slot, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        NodeType::Node256 => {
            let mut n = read_node256(frame.as_ref(), slot)?;
            if n.children[byte as usize] == 0 {
                return Err(Error::NodeCorrupt {
                    context: "inner_remove_child_and_collapse: byte not present (Node256)",
                });
            }
            n.children[byte as usize] = 0;
            n.count = n.count.saturating_sub(1);

            if n.count == 0 {
                frame.free_node(slot)?;
                return Ok(EraseSignal::SubtreeGone);
            }
            if n.count == 1 {
                let (surviving_byte, surviving_child) = {
                    let mut found = (0u8, 0u32);
                    for (i, c) in n.children.iter().enumerate() {
                        if *c != 0 {
                            found = (i as u8, *c);
                            break;
                        }
                    }
                    found
                };
                frame.free_node(slot)?;
                let new_slot =
                    write_prefix_chain(frame, &[surviving_byte], surviving_child as u16)?;
                return Ok(EraseSignal::Replaced(new_slot));
            }
            if n.count <= SHRINK_NODE256_TO_NODE48_AT {
                let shrunk = shrink_node256_to_node48(frame, slot, n)?;
                return Ok(EraseSignal::Replaced(shrunk));
            }
            write_struct_to_slot(frame, slot, &n)?;
            Ok(EraseSignal::Unchanged)
        }
        _ => Err(Error::NodeCorrupt {
            context: "inner_remove_child_and_collapse: not an inner node",
        }),
    }
}

// ---------- multi-blob arm ----------

/// Erase across a [`NodeType::Blob`] crossing.
///
/// Pins the child blob, runs the recursive erase in place, then
/// maps the child's [`EraseSignal`] back to the parent:
///
/// - `Unchanged`: commit the pinned buffer and return `Unchanged`
///   upward.
/// - `Replaced(new_entry)`: the child's entry slot changed (e.g.
///   collapse-to-lone-child). Update the child blob's
///   `header.root_slot`, patch the parent's
///   `BlobNode.child_entry_ptr`, commit the child, return
///   `Unchanged` upward.
/// - `SubtreeGone`: the child blob is now empty. Free the parent's
///   BlobNode slot, drop the orphaned child blob from cache + disk,
///   propagate `SubtreeGone` upward so the grandparent collapses
///   too.
fn erase_at_blob_node(
    bm: &BufferManager,
    parent_frame: &mut BlobFrame<'_>,
    bn_slot: u16,
    key: &[u8],
    depth: usize,
) -> Result<EraseReturn> {
    let bn = {
        let body = parent_frame
            .body_of_slot(bn_slot)
            .ok_or(Error::NodeCorrupt {
                context: "erase_at_blob_node: body resolution failed",
            })?;
        *cast::<BlobNode>(body)
    };
    let plen = bn.prefix_len as usize;
    if plen > BLOB_MAX_INLINE {
        return Err(Error::NodeCorrupt {
            context: "erase_at_blob_node: prefix_len exceeds inline buffer",
        });
    }

    if depth + plen > key.len() || key[depth..depth + plen] != bn.bytes[..plen] {
        return Ok(EraseReturn {
            signal: EraseSignal::Unchanged,
            previous: None,
        });
    }

    let child_guid = bn.child_blob_guid;
    let child_entry = bn.child_entry_ptr as u16;
    let child_depth = depth + plen;

    let child_pin = bm.pin(child_guid)?;

    let r = {
        let mut guard = child_pin.write();
        let mut cf = BlobFrame::wrap(guard.as_mut_slice());
        erase_at(Some(bm), &mut cf, child_entry, key, child_depth)?
    };

    match r.signal {
        EraseSignal::Unchanged => {
            drop(child_pin);
            bm.commit(child_guid)?;
            Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                previous: r.previous,
            })
        }
        EraseSignal::Replaced(new_entry) => {
            {
                let mut guard = child_pin.write();
                let mut cf = BlobFrame::wrap(guard.as_mut_slice());
                cf.header_mut().root_slot = new_entry;
            }
            let mut new_bn = bn;
            new_bn.child_entry_ptr = u32::from(new_entry);
            write_struct_to_slot(parent_frame, bn_slot, &new_bn)?;
            drop(child_pin);
            bm.commit(child_guid)?;
            Ok(EraseReturn {
                signal: EraseSignal::Unchanged,
                previous: r.previous,
            })
        }
        EraseSignal::SubtreeGone => {
            parent_frame.free_node(bn_slot)?;
            drop(child_pin);
            bm.delete_blob(child_guid)?;
            Ok(EraseReturn {
                signal: EraseSignal::SubtreeGone,
                previous: r.previous,
            })
        }
    }
}
