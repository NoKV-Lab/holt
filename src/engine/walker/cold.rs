//! Cold-read page-touch analysis tool (`#[ignore]`'d).
//!
//! Measures, for a real objstore blob layout, how many distinct 4 KB
//! pages a point-lookup descent actually touches (BlobHeader + path
//! nodes + leaf+value) — i.e. the floor for a page-granular cold read
//! vs today's whole-512 KB pin. Also reports the structure/value byte
//! split (sizes a "keep structure resident, page the values" redesign).
//!
//! The key→value+crossing summarizer that fed the (now-removed)
//! `cold.idx` sidecar used to live here too; it went with the sidecar.
//! The routing region (`compact_blob`) + the cold routed read
//! (`cold_read_routed`) are the production replacement.
//!
//!   cargo test -p holt --release cold_read_page_touch_ceiling -- --ignored --nocapture

#[cfg(test)]
mod page_ceiling {
    use crate::api::errors::Result;
    use crate::layout::{Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix};
    use crate::store::blob_store::{AlignedBlobBuf, BlobStore, FileBlobStore};
    use crate::store::BlobFrameRef;
    use crate::{Tree, TreeConfig};
    use std::collections::{BTreeMap, BTreeSet};
    use std::mem::size_of;

    use super::super::cast;
    use super::super::readers::{child_offset, resolve_typed};

    const PAGE: u32 = 4096;

    fn objkey(i: usize) -> Vec<u8> {
        format!(
            "bucket-{:02}/path-{:04}/sub/obj-{:08}",
            i % 32,
            (i / 64) % 4096,
            i
        )
        .into_bytes()
    }

    #[derive(Default)]
    struct Acc {
        hist: BTreeMap<usize, u64>,
        leaves: u64,
        crossings: u64,
        value_bytes: u64,
    }

    fn walk(frame: BlobFrameRef<'_>, off: u32, path: &mut Vec<u32>, acc: &mut Acc) -> Result<()> {
        path.push(off);
        let (ntype, body) = resolve_typed(frame, off)?;
        match ntype {
            NodeType::Leaf => {
                let leaf = *cast::<Leaf>(&body[..size_of::<Leaf>()]);
                if leaf.tombstone == 0 {
                    let total =
                        size_of::<Leaf>() as u32 + leaf.key_len as u32 + leaf.value_len as u32;
                    // Distinct 4 KB pages: header (root_slot lives there) + every
                    // node on the root→leaf path + the bytes the leaf+value span.
                    let mut pages: BTreeSet<u32> = path.iter().map(|o| o / PAGE).collect();
                    pages.insert(0);
                    let end = off + total;
                    let mut p = off / PAGE;
                    while p <= end.saturating_sub(1) / PAGE {
                        pages.insert(p);
                        p += 1;
                    }
                    *acc.hist.entry(pages.len()).or_default() += 1;
                    acc.leaves += 1;
                    acc.value_bytes += u64::from(leaf.value_len);
                }
            }
            NodeType::Prefix => {
                let pfx = cast::<Prefix>(body);
                walk(frame, child_offset(pfx.child as u16), path, acc)?;
            }
            NodeType::Node4 => {
                let n = cast::<Node4>(body);
                for i in 0..(n.count as usize).min(4) {
                    walk(frame, child_offset(n.children[i]), path, acc)?;
                }
            }
            NodeType::Node16 => {
                let n = cast::<Node16>(body);
                for i in 0..(n.count as usize).min(16) {
                    walk(frame, child_offset(n.children[i]), path, acc)?;
                }
            }
            NodeType::Node48 => {
                let n = cast::<Node48>(body);
                for b in 0..=u8::MAX {
                    let idx = n.index[b as usize];
                    if idx != 0 {
                        walk(frame, child_offset(n.children[idx as usize - 1]), path, acc)?;
                    }
                }
            }
            NodeType::Node256 => {
                let n = cast::<Node256>(body);
                for b in 0..=u8::MAX {
                    let c = n.children[b as usize];
                    if c != 0 {
                        walk(frame, child_offset(c), path, acc)?;
                    }
                }
            }
            NodeType::Blob => acc.crossings += 1,
            NodeType::EmptyRoot | NodeType::Invalid => {}
        }
        path.pop();
        Ok(())
    }

    #[test]
    #[ignore = "analysis tool; run explicitly with --ignored --nocapture"]
    #[allow(clippy::cast_precision_loss)]
    fn cold_read_page_touch_ceiling() {
        let dir = tempfile::tempdir().unwrap();
        let n_keys = 300_000usize;
        let value_len = 48usize;
        {
            let mut cfg = TreeConfig::new(dir.path());
            cfg.durability = crate::Durability::Wal { sync: false };
            let tree = Tree::open(cfg).unwrap();
            for i in 0..n_keys {
                tree.put(&objkey(i), &vec![(i & 0xff) as u8; value_len])
                    .unwrap();
            }
            tree.checkpoint().unwrap();
        }
        let store = FileBlobStore::open(dir.path()).unwrap();
        let guids = store.list_blobs().unwrap();
        let mut buf = AlignedBlobBuf::zeroed();
        let mut acc = Acc::default();
        let mut used_total = 0u64;
        for g in &guids {
            store.read_blob(*g, &mut buf).unwrap();
            let frame = BlobFrameRef::wrap(buf.as_slice());
            used_total += u64::from(frame.header().space_used);
            let root = child_offset(frame.header().root_slot);
            let _ = walk(frame, root, &mut Vec::new(), &mut acc);
        }

        let total = acc.leaves.max(1);
        let (mut cum, mut p50, mut p95, mut mean_num) = (0u64, 0usize, 0usize, 0u64);
        for (pages, cnt) in &acc.hist {
            mean_num += (*pages as u64) * cnt;
            cum += cnt;
            if p50 == 0 && cum * 2 >= total {
                p50 = *pages;
            }
            if p95 == 0 && cum * 100 >= total * 95 {
                p95 = *pages;
            }
        }
        eprintln!(
            "\n=== COLD-READ PAGE-TOUCH CEILING (objstore, {n_keys} keys, val={value_len}B) ==="
        );
        eprintln!(
            "blobs={} leaves={} crossings={}",
            guids.len(),
            acc.leaves,
            acc.crossings
        );
        eprintln!("distinct 4KB pages touched per point lookup:");
        for (pages, cnt) in &acc.hist {
            eprintln!(
                "  {:2} pages (~{:3} KB): {:>8} leaves  {:5.1}%",
                pages,
                pages * 4,
                cnt,
                *cnt as f64 * 100.0 / total as f64
            );
        }
        eprintln!(
            "mean={:.2} pages (~{:.1} KB)  p50={} (~{} KB)  p95={} (~{} KB)   vs 512 KB whole-blob pin",
            mean_num as f64 / total as f64,
            mean_num as f64 * 4.0 / total as f64,
            p50,
            p50 * 4,
            p95,
            p95 * 4
        );
        let val = acc.value_bytes;
        let used = used_total.max(1);
        eprintln!(
            "structure/value split: value={:.1} MB ({:.1}%)  structure={:.1} MB ({:.1}%)  of {:.1} MB live",
            val as f64 / 1e6,
            val as f64 * 100.0 / used as f64,
            (used - val) as f64 / 1e6,
            (used - val) as f64 * 100.0 / used as f64,
            used as f64 / 1e6
        );
        eprintln!(
            "→ 'keep structure resident, page values' would cost ~{:.0} MB RAM for this {:.0} MB dataset (vs whole-blob caching)\n",
            (used - val) as f64 / 1e6,
            used as f64 / 1e6
        );
    }
}
