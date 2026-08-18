use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::layout::BlobGuid;

#[derive(Clone)]
pub(crate) enum DeltaEntry {
    Put {
        value: Vec<u8>,
        seq: u64,
        creates_key: bool,
    },
    Delete {
        seq: u64,
    },
}

#[derive(Clone)]
pub(crate) struct DeltaOp {
    pub(crate) tree_id: u64,
    pub(crate) root_guid: BlobGuid,
    pub(crate) key: Vec<u8>,
    pub(crate) entry: DeltaEntry,
}

#[derive(Default)]
pub(crate) struct WriteDelta {
    inner: Mutex<DeltaMaps>,
    has_any: AtomicBool,
}

#[derive(Default)]
struct DeltaMaps {
    pending: HashMap<u64, HashMap<Vec<u8>, DeltaOp>>,
    flushing: HashMap<u64, HashMap<Vec<u8>, DeltaOp>>,
    pending_key_set: HashMap<u64, usize>,
    flushing_key_set: HashMap<u64, usize>,
}

impl WriteDelta {
    pub(crate) fn stage_put(
        &self,
        tree_id: u64,
        root_guid: BlobGuid,
        key: &[u8],
        value: &[u8],
        seq: u64,
        creates_key: bool,
    ) {
        let op = DeltaOp {
            tree_id,
            root_guid,
            key: key.to_vec(),
            entry: DeltaEntry::Put {
                value: value.to_vec(),
                seq,
                creates_key,
            },
        };
        let mut guard = self.inner.lock().unwrap();
        guard.insert_pending(op);
        self.has_any.store(true, Ordering::Release);
    }

    pub(crate) fn stage_delete(&self, tree_id: u64, root_guid: BlobGuid, key: &[u8], seq: u64) {
        let op = DeltaOp {
            tree_id,
            root_guid,
            key: key.to_vec(),
            entry: DeltaEntry::Delete { seq },
        };
        let mut guard = self.inner.lock().unwrap();
        guard.insert_pending(op);
        self.has_any.store(true, Ordering::Release);
    }

    pub(crate) fn get(&self, tree_id: u64, key: &[u8]) -> Option<DeltaEntry> {
        self.inner.lock().unwrap().get(tree_id, key).cloned()
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub(crate) fn has_any(&self) -> bool {
        self.has_any.load(Ordering::Acquire)
    }

    pub(crate) fn tree_len(&self, tree_id: u64) -> usize {
        self.inner.lock().unwrap().tree_len(tree_id)
    }

    pub(crate) fn tree_key_set_len(&self, tree_id: u64) -> usize {
        self.inner.lock().unwrap().tree_key_set_len(tree_id)
    }

    pub(crate) fn begin_flush_tree(&self, tree_id: u64) -> Vec<DeltaOp> {
        let mut guard = self.inner.lock().unwrap();
        let Some(tree) = guard.pending.remove(&tree_id) else {
            return Vec::new();
        };
        guard.pending_key_set.remove(&tree_id);
        let mut out: Vec<_> = tree.into_values().collect();
        sort_tree_ops(&mut out);
        guard.publish_flushing(&out);
        out
    }

    pub(crate) fn begin_flush_all(&self) -> Vec<DeltaOp> {
        let mut guard = self.inner.lock().unwrap();
        let mut out = Vec::new();
        for (_, tree) in guard.pending.drain() {
            out.extend(tree.into_values());
        }
        guard.pending_key_set.clear();
        sort_all_ops(&mut out);
        guard.publish_flushing(&out);
        out
    }

    pub(crate) fn finish_flush(&self, ops: &[DeltaOp]) {
        if ops.is_empty() {
            return;
        }
        let mut guard = self.inner.lock().unwrap();
        for op in ops {
            guard.remove_flushing(op);
        }
        if guard.len() == 0 {
            self.has_any.store(false, Ordering::Release);
        }
    }

    pub(crate) fn abort_flush(&self, ops: Vec<DeltaOp>) {
        if ops.is_empty() {
            return;
        }
        let mut guard = self.inner.lock().unwrap();
        for op in ops {
            guard.remove_flushing(&op);
            guard.insert_pending(op);
        }
        self.has_any.store(true, Ordering::Release);
    }
}

impl DeltaEntry {
    pub(crate) fn seq(&self) -> u64 {
        match self {
            Self::Put { seq, .. } | Self::Delete { seq } => *seq,
        }
    }

    fn changes_key_set(&self) -> bool {
        match self {
            Self::Put { creates_key, .. } => *creates_key,
            Self::Delete { .. } => true,
        }
    }
}

impl DeltaMaps {
    fn insert_pending(&mut self, op: DeltaOp) {
        let tree_id = op.tree_id;
        let changes_key_set = op.entry.changes_key_set();
        let old = self
            .pending
            .entry(tree_id)
            .or_default()
            .insert(op.key.clone(), op);
        if old.is_some_and(|old| old.entry.changes_key_set()) {
            decrement_tree_count(&mut self.pending_key_set, tree_id);
        }
        if changes_key_set {
            increment_tree_count(&mut self.pending_key_set, tree_id);
        }
    }

    fn publish_flushing(&mut self, ops: &[DeltaOp]) {
        for op in ops {
            let old = self
                .flushing
                .entry(op.tree_id)
                .or_default()
                .insert(op.key.clone(), op.clone());
            if old.is_some_and(|old| old.entry.changes_key_set()) {
                decrement_tree_count(&mut self.flushing_key_set, op.tree_id);
            }
            if op.entry.changes_key_set() {
                increment_tree_count(&mut self.flushing_key_set, op.tree_id);
            }
        }
    }

    fn remove_flushing(&mut self, op: &DeltaOp) {
        let Some(tree) = self.flushing.get_mut(&op.tree_id) else {
            return;
        };
        let remove = tree
            .get(&op.key)
            .is_some_and(|current| current.entry.seq() == op.entry.seq());
        if remove {
            if let Some(removed) = tree.remove(&op.key) {
                if removed.entry.changes_key_set() {
                    decrement_tree_count(&mut self.flushing_key_set, op.tree_id);
                }
            }
            if tree.is_empty() {
                self.flushing.remove(&op.tree_id);
            }
        }
    }

    fn get(&self, tree_id: u64, key: &[u8]) -> Option<&DeltaEntry> {
        let pending = self
            .pending
            .get(&tree_id)
            .and_then(|tree| tree.get(key))
            .map(|op| &op.entry);
        let flushing = self
            .flushing
            .get(&tree_id)
            .and_then(|tree| tree.get(key))
            .map(|op| &op.entry);

        match (pending, flushing) {
            (Some(a), Some(b)) if a.seq() >= b.seq() => Some(a),
            (Some(_) | None, Some(b)) => Some(b),
            (Some(a), None) => Some(a),
            (None, None) => None,
        }
    }

    fn len(&self) -> usize {
        map_len(&self.pending) + map_len(&self.flushing)
    }

    fn tree_len(&self, tree_id: u64) -> usize {
        self.pending.get(&tree_id).map_or(0, HashMap::len)
            + self.flushing.get(&tree_id).map_or(0, HashMap::len)
    }

    fn tree_key_set_len(&self, tree_id: u64) -> usize {
        self.pending_key_set.get(&tree_id).copied().unwrap_or(0)
            + self.flushing_key_set.get(&tree_id).copied().unwrap_or(0)
    }
}

fn map_len(map: &HashMap<u64, HashMap<Vec<u8>, DeltaOp>>) -> usize {
    map.values().map(HashMap::len).sum()
}

fn increment_tree_count(counts: &mut HashMap<u64, usize>, tree_id: u64) {
    *counts.entry(tree_id).or_insert(0) += 1;
}

fn decrement_tree_count(counts: &mut HashMap<u64, usize>, tree_id: u64) {
    let Some(count) = counts.get_mut(&tree_id) else {
        return;
    };
    *count = count.saturating_sub(1);
    if *count == 0 {
        counts.remove(&tree_id);
    }
}

fn sort_tree_ops(out: &mut [DeltaOp]) {
    out.sort_by(|a, b| {
        let a_seq = a.entry.seq();
        let b_seq = b.entry.seq();
        a.root_guid
            .cmp(&b.root_guid)
            .then_with(|| a.key.cmp(&b.key))
            .then_with(|| a_seq.cmp(&b_seq))
    });
}

fn sort_all_ops(out: &mut [DeltaOp]) {
    out.sort_by(|a, b| {
        let a_seq = a.entry.seq();
        let b_seq = b.entry.seq();
        a.tree_id
            .cmp(&b.tree_id)
            .then_with(|| a.root_guid.cmp(&b.root_guid))
            .then_with(|| a.key.cmp(&b.key))
            .then_with(|| a_seq.cmp(&b_seq))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(tree_id: u64, root_guid: BlobGuid, key: &[u8], seq: u64) -> DeltaOp {
        DeltaOp {
            tree_id,
            root_guid,
            key: key.to_vec(),
            entry: DeltaEntry::Put {
                value: b"v".to_vec(),
                seq,
                creates_key: false,
            },
        }
    }

    #[test]
    fn begin_flush_all_groups_by_tree_root_and_key() {
        let delta = WriteDelta::default();
        delta.stage_put(2, [2; 16], b"z", b"v", 10, false);
        delta.stage_put(1, [9; 16], b"b", b"v", 11, false);
        delta.stage_put(1, [1; 16], b"c", b"v", 12, false);
        delta.stage_put(1, [1; 16], b"a", b"v", 13, false);
        delta.stage_put(2, [1; 16], b"a", b"v", 14, false);

        let keys: Vec<_> = delta
            .begin_flush_all()
            .into_iter()
            .map(|op| (op.tree_id, op.root_guid[0], op.key, op.entry.seq()))
            .collect();

        assert_eq!(
            keys,
            vec![
                (1, 1, b"a".to_vec(), 13),
                (1, 1, b"c".to_vec(), 12),
                (1, 9, b"b".to_vec(), 11),
                (2, 1, b"a".to_vec(), 14),
                (2, 2, b"z".to_vec(), 10),
            ]
        );
    }

    #[test]
    fn begin_flush_tree_groups_by_root_and_coalesces_latest_key() {
        let delta = WriteDelta::default();
        delta.stage_put(7, [9; 16], b"b", b"old", 1, false);
        delta.stage_put(7, [9; 16], b"b", b"new", 4, false);
        delta.stage_put(7, [1; 16], b"z", b"v", 3, false);
        delta.stage_put(7, [1; 16], b"a", b"v", 2, false);

        let ops = delta.begin_flush_tree(7);
        let keys: Vec<_> = ops
            .iter()
            .map(|op| (op.root_guid[0], op.key.as_slice(), op.entry.seq()))
            .collect();

        assert_eq!(
            keys,
            vec![
                (1, b"a".as_slice(), 2),
                (1, b"z".as_slice(), 3),
                (9, b"b".as_slice(), 4)
            ]
        );
        match &ops[2].entry {
            DeltaEntry::Put { value, .. } => assert_eq!(value, b"new"),
            DeltaEntry::Delete { .. } => panic!("latest put should survive coalescing"),
        }
    }

    #[test]
    fn key_set_dirty_count_tracks_coalescing_and_flush_lifecycle() {
        let delta = WriteDelta::default();
        delta.stage_put(7, [1; 16], b"a", b"v", 1, false);
        assert_eq!(delta.tree_key_set_len(7), 0);
        delta.stage_put(7, [1; 16], b"b", b"v", 2, true);
        assert_eq!(delta.tree_key_set_len(7), 1);
        delta.stage_put(7, [1; 16], b"b", b"v2", 3, false);
        assert_eq!(delta.tree_key_set_len(7), 0);
        delta.stage_delete(7, [1; 16], b"c", 4);
        assert_eq!(delta.tree_key_set_len(7), 1);
        let ops = delta.begin_flush_tree(7);
        assert_eq!(delta.tree_key_set_len(7), 1);
        delta.finish_flush(&ops);
        assert_eq!(delta.tree_key_set_len(7), 0);
    }

    #[test]
    fn zero_debt_flag_tracks_finish_and_abort() {
        let delta = WriteDelta::default();
        assert!(!delta.has_any());

        delta.stage_put(7, [1; 16], b"a", b"v", 1, false);
        assert!(delta.has_any());
        let ops = delta.begin_flush_tree(7);
        assert!(delta.has_any(), "flushing debt is still visible");
        delta.finish_flush(&ops);
        assert!(!delta.has_any());

        delta.stage_delete(7, [1; 16], b"a", 2);
        let ops = delta.begin_flush_tree(7);
        delta.abort_flush(ops);
        assert!(delta.has_any(), "aborted debt returns to pending");
    }

    #[test]
    fn explicit_sort_keeps_seq_as_final_tiebreaker() {
        let mut ops = vec![
            put(1, [1; 16], b"k", 9),
            put(1, [1; 16], b"k", 7),
            put(1, [1; 16], b"k", 8),
        ];
        sort_tree_ops(&mut ops);
        let seqs: Vec<_> = ops.iter().map(|op| op.entry.seq()).collect();
        assert_eq!(seqs, vec![7, 8, 9]);
    }
}
