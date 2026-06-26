use std::collections::HashMap;
use std::sync::Mutex;

use crate::layout::BlobGuid;

#[derive(Clone)]
pub(crate) enum DeltaEntry {
    Put { value: Vec<u8>, seq: u64 },
    Delete { seq: u64 },
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
}

#[derive(Default)]
struct DeltaMaps {
    pending: HashMap<u64, HashMap<Vec<u8>, DeltaOp>>,
    flushing: HashMap<u64, HashMap<Vec<u8>, DeltaOp>>,
}

impl WriteDelta {
    pub(crate) fn stage_put(
        &self,
        tree_id: u64,
        root_guid: BlobGuid,
        key: &[u8],
        value: &[u8],
        seq: u64,
    ) {
        let op = DeltaOp {
            tree_id,
            root_guid,
            key: key.to_vec(),
            entry: DeltaEntry::Put {
                value: value.to_vec(),
                seq,
            },
        };
        self.inner.lock().unwrap().insert_pending(op);
    }

    pub(crate) fn stage_delete(&self, tree_id: u64, root_guid: BlobGuid, key: &[u8], seq: u64) {
        let op = DeltaOp {
            tree_id,
            root_guid,
            key: key.to_vec(),
            entry: DeltaEntry::Delete { seq },
        };
        self.inner.lock().unwrap().insert_pending(op);
    }

    pub(crate) fn get(&self, tree_id: u64, key: &[u8]) -> Option<DeltaEntry> {
        self.inner.lock().unwrap().get(tree_id, key).cloned()
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub(crate) fn tree_len(&self, tree_id: u64) -> usize {
        self.inner.lock().unwrap().tree_len(tree_id)
    }

    pub(crate) fn begin_flush_tree(&self, tree_id: u64) -> Vec<DeltaOp> {
        let mut guard = self.inner.lock().unwrap();
        let Some(tree) = guard.pending.remove(&tree_id) else {
            return Vec::new();
        };
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
    }
}

impl DeltaEntry {
    pub(crate) fn seq(&self) -> u64 {
        match self {
            Self::Put { seq, .. } | Self::Delete { seq } => *seq,
        }
    }
}

impl DeltaMaps {
    fn insert_pending(&mut self, op: DeltaOp) {
        self.pending
            .entry(op.tree_id)
            .or_default()
            .insert(op.key.clone(), op);
    }

    fn publish_flushing(&mut self, ops: &[DeltaOp]) {
        for op in ops {
            self.flushing
                .entry(op.tree_id)
                .or_default()
                .insert(op.key.clone(), op.clone());
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
            tree.remove(&op.key);
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
}

fn map_len(map: &HashMap<u64, HashMap<Vec<u8>, DeltaOp>>) -> usize {
    map.values().map(HashMap::len).sum()
}

fn sort_tree_ops(out: &mut [DeltaOp]) {
    out.sort_by(|a, b| {
        let a_seq = a.entry.seq();
        let b_seq = b.entry.seq();
        a_seq.cmp(&b_seq).then_with(|| a.key.cmp(&b.key))
    });
}

fn sort_all_ops(out: &mut [DeltaOp]) {
    out.sort_by(|a, b| {
        let a_seq = a.entry.seq();
        let b_seq = b.entry.seq();
        a_seq
            .cmp(&b_seq)
            .then_with(|| a.tree_id.cmp(&b.tree_id))
            .then_with(|| a.key.cmp(&b.key))
    });
}
