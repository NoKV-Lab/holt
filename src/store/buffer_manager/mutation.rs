use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::layout::BlobGuid;

pub(super) const BOOKKEEPING_SHARDS: usize = 64;

#[derive(Default)]
pub(super) struct MutationState {
    /// New dirty entries not yet claimed by a checkpoint round.
    pub(super) dirty: HashMap<BlobGuid, u64>,
    /// Number of checkpoint epochs that still own a cloned or
    /// clone-pending image of this blob. Multi-epoch checkpoint
    /// pipelining can have the same blob in more than one in-flight
    /// epoch, so this is a reference count rather than a single seq.
    pub(super) flushing: HashMap<BlobGuid, usize>,
    /// Blobs unlinked from the tree but not yet deleted from the
    /// store manifest because WAL/checkpoint ordering still owns
    /// them.
    pub(super) pending_deletes: HashMap<BlobGuid, u64>,
    /// Pending deletes claimed by a checkpoint epoch but not yet
    /// completed. This keeps the delete fence visible while I/O is
    /// in flight, so stale walkers cannot reload or dirty a blob
    /// that has already been logically unlinked.
    pub(super) deleting: HashMap<BlobGuid, u64>,
    /// In-memory maintenance hints for blobs whose local garbage
    /// is worth checking before the next online compact pass.
    ///
    /// This is advisory only. Dirty / flushing / pending-delete own
    /// correctness; candidate loss can only delay maintenance until
    /// a later seed scan or explicit compact pass rediscovers it.
    pub(super) compact_candidates: MaintenanceQueue,
    /// In-memory maintenance hints for parent blobs that own at
    /// least one `BlobNode` crossing and may be worth a merge pass.
    pub(super) merge_candidates: MaintenanceQueue,
}

impl MutationState {
    pub(super) fn is_protected(&self, guid: &BlobGuid) -> bool {
        self.dirty.contains_key(guid) || self.flushing.contains_key(guid)
    }

    pub(super) fn is_protected_or_pending(&self, guid: &BlobGuid) -> bool {
        self.is_protected(guid) || self.has_delete_fence(guid)
    }

    pub(super) fn checkpoint_owned_or_pending(&self, guid: &BlobGuid) -> bool {
        self.flushing.contains_key(guid) || self.has_delete_fence(guid)
    }

    pub(super) fn remove_unclaimed_dirty(&mut self, guid: &BlobGuid) {
        self.dirty.remove(guid);
    }

    pub(super) fn add_flushing(&mut self, guid: BlobGuid) {
        *self.flushing.entry(guid).or_insert(0) += 1;
    }

    pub(super) fn remove_one_flushing(&mut self, guid: &BlobGuid) {
        if let Some(count) = self.flushing.get_mut(guid) {
            *count -= 1;
            if *count == 0 {
                self.flushing.remove(guid);
            }
        }
    }

    pub(super) fn has_delete_fence(&self, guid: &BlobGuid) -> bool {
        self.pending_deletes.contains_key(guid) || self.deleting.contains_key(guid)
    }

    pub(super) fn remove_maintenance_candidates(&mut self, guid: &BlobGuid) -> (bool, bool) {
        (
            self.compact_candidates.remove(guid),
            self.merge_candidates.remove(guid),
        )
    }
}

#[derive(Default)]
pub(super) struct MaintenanceQueue {
    set: HashSet<BlobGuid>,
    queue: VecDeque<BlobGuid>,
}

impl MaintenanceQueue {
    pub(super) fn insert(&mut self, guid: BlobGuid) -> bool {
        if self.set.insert(guid) {
            self.queue.push_back(guid);
            true
        } else {
            false
        }
    }

    pub(super) fn remove(&mut self, guid: &BlobGuid) -> bool {
        self.set.remove(guid)
    }

    fn pop_batch(&mut self, limit: usize) -> Vec<BlobGuid> {
        let mut out = Vec::new();
        while out.len() < limit {
            let Some(guid) = self.queue.pop_front() else {
                break;
            };
            if self.set.remove(&guid) {
                out.push(guid);
            }
        }
        out
    }
}

pub(super) fn bookkeeping_shard_idx(guid: &BlobGuid) -> usize {
    debug_assert!(BOOKKEEPING_SHARDS.is_power_of_two());

    let mut lo_bytes = [0u8; 8];
    let mut hi_bytes = [0u8; 8];
    lo_bytes.copy_from_slice(&guid[0..8]);
    hi_bytes.copy_from_slice(&guid[8..16]);
    let lo = u64::from_le_bytes(lo_bytes);
    let hi = u64::from_le_bytes(hi_bytes);
    let mut h = lo ^ hi.rotate_left(27);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as usize) & (BOOKKEEPING_SHARDS - 1)
}

#[derive(Debug, Clone, Copy)]
pub(super) enum CandidateKind {
    Compact,
    Merge,
}

pub(super) fn pop_candidate_batch(
    shards: &[Mutex<MutationState>; BOOKKEEPING_SHARDS],
    cursor: &AtomicUsize,
    total: &AtomicUsize,
    kind: CandidateKind,
    limit: usize,
) -> Vec<BlobGuid> {
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let start = cursor.fetch_add(1, Ordering::Relaxed) & (BOOKKEEPING_SHARDS - 1);
    for offset in 0..BOOKKEEPING_SHARDS {
        let idx = (start + offset) & (BOOKKEEPING_SHARDS - 1);
        let shard = &shards[idx];
        let mut state = shard.lock().unwrap();
        let queue = match kind {
            CandidateKind::Compact => &mut state.compact_candidates,
            CandidateKind::Merge => &mut state.merge_candidates,
        };
        let remaining = limit - out.len();
        let popped = queue.pop_batch(remaining);
        total.fetch_sub(popped.len(), Ordering::Relaxed);
        out.extend(popped);
        if out.len() == limit {
            return out;
        }
    }
    out
}
