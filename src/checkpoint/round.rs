//! One checkpoint round — the planner's main work unit, also
//! invoked synchronously by `Checkpointer::Drop` to drain in-flight
//! dirty state before the Tree handle disappears.
//!
//! ## Sequence
//!
//! 0. **Merge pass** (optional, controlled by
//!    `CheckpointConfig::auto_merge`) — drains queued parent-merge
//!    candidates and folds mergeable children back into parents.
//!    Merge mutations are staged through the same dirty /
//!    pending-delete sets as foreground writes, then flushed by
//!    this round after the WAL sync.
//! 1. **Snapshot dirty + pending deletes + content versions** under
//!    the exclusive side of the tree's commit-publish gate.
//! 2. **Flush WAL** through the journal worker so every record that
//!    mirrors a snapshotted seq is durable before we drop it.
//! 3. **Clone version-matched bytes** outside the commit gate.
//!    If a foreground writer changed a blob after step 1, restore
//!    that dirty entry and retry it in a later round.
//! 4. **Enqueue one checkpoint epoch** to the I/O worker.
//! 5. **Retire completed epochs** on later planner turns in FIFO
//!    order. This is the truncate watermark: a later epoch may not
//!    advance WAL trimming before every older epoch is known to
//!    have landed or restored.
//! 6. **Truncate WAL** only when the pipeline is empty and
//!    `bm.dirty_count() == 0 && bm.pending_delete_count() == 0`
//!    under the commit-publish gate.
//!
//! This function is called from two places:
//!
//! - The `checkpoint_thread` main loop in [`super::mod`]
//!   (background path).
//! - `Checkpointer::Drop` (synchronous final round on the calling
//!   thread, after the planner has joined and writers are
//!   guaranteed to be gone).

use crossbeam_channel::{bounded, Receiver, TryRecvError};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use crate::api::errors::{is_blob_store_not_found, Error, Result};
use crate::engine;
use crate::layout::BlobGuid;
use crate::store::blob_store::{AlignedBlobBuf, BlobStore};
use crate::store::{DirtySnapshotEntry, WriteThroughEntry};

use super::io::{CheckpointEpoch, CheckpointEpochReport, IoTask};
use super::Shared;

pub(super) struct Pipeline {
    in_flight: VecDeque<PendingEpoch>,
    max_in_flight: usize,
}

struct PendingEpoch {
    rx: Receiver<CheckpointEpochReport>,
    snap: HashMap<BlobGuid, u64>,
    pending: HashMap<BlobGuid, u64>,
}

type ClonedDirtyBytes = Vec<(BlobGuid, u64, u64, AlignedBlobBuf)>;

impl Pipeline {
    pub(super) fn new(max_in_flight: usize) -> Self {
        Self {
            in_flight: VecDeque::new(),
            max_in_flight: max_in_flight.max(1),
        }
    }

    pub(super) fn has_room(&self) -> bool {
        self.in_flight.len() < self.max_in_flight
    }

    pub(super) fn is_empty(&self) -> bool {
        self.in_flight.is_empty()
    }

    pub(super) fn reap_ready(&mut self, shared: &Arc<Shared>) -> Result<()> {
        while let Some(front) = self.in_flight.front() {
            match front.rx.try_recv() {
                Ok(report) => {
                    self.in_flight.pop_front().expect("front exists");
                    finish_epoch(shared, report)?;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    let pending = self.in_flight.pop_front().expect("front exists");
                    restore_unreported_epoch(shared, pending);
                    shared
                        .rounds_failed
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(Error::Internal(
                        "checkpoint: I/O worker dropped epoch completion",
                    ));
                }
            }
        }
        self.maybe_truncate(shared)
    }

    fn wait_for_room(&mut self, shared: &Arc<Shared>) -> Result<()> {
        if self.has_room() {
            return Ok(());
        }
        self.wait_one(shared)
    }

    pub(super) fn drain(&mut self, shared: &Arc<Shared>) -> Result<()> {
        let mut first_err = None;
        while !self.in_flight.is_empty() {
            if let Err(e) = self.wait_one(shared) {
                first_err.get_or_insert(e);
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        self.maybe_truncate(shared)
    }

    fn wait_one(&mut self, shared: &Arc<Shared>) -> Result<()> {
        let Some(pending) = self.in_flight.pop_front() else {
            return Ok(());
        };
        if let Ok(report) = pending.rx.recv() {
            finish_epoch(shared, report)
        } else {
            restore_unreported_epoch(shared, pending);
            shared
                .rounds_failed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(Error::Internal(
                "checkpoint: I/O worker dropped epoch completion",
            ))
        }
    }

    fn push(&mut self, pending: PendingEpoch) {
        debug_assert!(self.has_room());
        self.in_flight.push_back(pending);
    }

    fn maybe_truncate(&self, shared: &Arc<Shared>) -> Result<()> {
        if !self.in_flight.is_empty() {
            return Ok(());
        }
        let Some(journal) = &shared.journal else {
            return Ok(());
        };
        if !journal.needs_checkpoint() {
            return Ok(());
        }
        let _commit = shared.commit_gate.enter_checkpoint();
        // The dirty/flushing/pending counters are BufferManager state and do
        // NOT capture the store's own deferred durability: the I/O worker
        // retires a dirty entry right after the data `pwrite` but BEFORE the
        // data fsync + manifest-delta persist (`flush_inner`). So all three
        // counters can read 0 while `store.needs_flush()` is still true —
        // i.e. a just-written blob's new slot mapping is only in the in-memory
        // manifest, not yet in `manifest.log`. Truncating the WAL there leaves
        // a crashed reopen with the acked write in NEITHER the (truncated) WAL
        // nor the (un-persisted) manifest — a lost acknowledged write. The
        // SIGKILL crash soak hits exactly this (lazy routing keeps the root
        // blob a perpetual compaction target, so it is re-written every round).
        // `run_round`'s early-skip gates on the same `needs_flush()` for this
        // recovery edge; mirror it here so truncate waits for store durability.
        if shared.bm.dirty_count() == 0
            && shared.bm.flushing_count() == 0
            && shared.bm.pending_delete_count() == 0
            && shared.bm.write_delta_count() == 0
            && !shared.bm.needs_flush()
        {
            journal.truncate()?;
            use std::sync::atomic::Ordering;
            shared.truncates.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

pub(super) fn run_round_sync(shared: &Arc<Shared>) -> Result<()> {
    let mut pipeline = Pipeline::new(1);
    run_round(shared, &mut pipeline)?;
    pipeline.drain(shared)
}

// The round is intentionally a single linear submission function:
// it maps "what is durable enough to enqueue" without hiding the
// WAL watermark / dirty snapshot / byte clone interlock.
#[allow(clippy::too_many_lines)]
pub(super) fn run_round(shared: &Arc<Shared>, pipeline: &mut Pipeline) -> Result<()> {
    use std::sync::atomic::Ordering;

    pipeline.reap_ready(shared)?;
    pipeline.wait_for_room(shared)?;

    shared.rounds_attempted.fetch_add(1, Ordering::Relaxed);
    let round_start = Instant::now();

    // 0. Optional candidate-driven merge pass.
    //
    // Merge is the only background checkpoint action that can both
    // rewrite a parent BlobNode edge and queue the old child for
    // manifest deletion. Do not run it while older checkpoint epochs
    // are still in flight: an older parent snapshot may already have
    // passed content-version validation in the I/O worker, and letting a
    // later merge delete its child before that epoch is fully retired can
    // leave a durable parent -> missing-child edge after crash. Plain
    // dirty epochs may pipeline; structural delete epochs form a barrier.
    if shared.cfg.auto_merge && shared.bm.merge_candidate_count() != 0 {
        pipeline.drain(shared)?;
    }
    let merged = if shared.cfg.auto_merge {
        match run_merge_pass(shared) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("holt: checkpoint merge pass failed: {e}");
                0
            }
        }
    } else {
        0
    };
    shared.merges_total.fetch_add(merged, Ordering::Relaxed);

    // 1. Snapshot dirty + pending-deletes + WAL watermark under
    // the same commit-publish gate used by foreground persistent
    // writers. For each dirty blob we also capture its latch
    // content version; byte cloning happens after the gate is
    // released and accepts a blob only if that version is still
    // current under a shared blob latch.
    //
    // If `snapshot_pending_deletes` were taken outside this
    // commit-publish block, a writer could (a) enter its mutation,
    // (b) walker.erase that hits `SubtreeGone` (which calls
    // `mark_for_delete`), (c) submit the erase record, (d)
    // leave the gate, before we snapshot pending; we'd then
    // execute `store.delete_blob` and re-Sync manifest while
    // the writer's WAL record was still only in the writer's
    // buffer. A crash there would leave the manifest ahead of
    // WAL — exactly the W2D violation deferred-delete was
    // designed to prevent.
    //
    // No-WAL trees (memory mode, user-supplied store) skip the WAL
    // watermark but use the same version-checked clone path.
    let (mut snap, mut pending, versioned_snap, wal_up_to) = if let Some(journal) = &shared.journal
    {
        let _maintenance = shared.maintenance_gate.enter_shared();
        let wal_up_to = {
            let _commit = shared.commit_gate.enter_checkpoint();
            journal.queued_work()
        };
        if let Err(e) = shared.bm.flush_write_deltas() {
            shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
            shared
                .last_round_micros
                .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
            return Err(e);
        }
        let _commit = shared.commit_gate.enter_checkpoint();
        let snap = shared.bm.snapshot_dirty();
        let pending = shared.bm.snapshot_pending_deletes();
        let versioned_snap = match shared.bm.snapshot_dirty_versions(&snap) {
            Ok(versioned) => versioned,
            Err(e) => {
                shared.bm.restore_pending_deletes(pending);
                shared.bm.restore_dirty(snap);
                shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
                shared
                    .last_round_micros
                    .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                return Err(e);
            }
        };
        (snap, pending, versioned_snap, Some(wal_up_to))
    } else {
        let _maintenance = shared.maintenance_gate.enter_shared();
        if let Err(e) = shared.bm.flush_write_deltas() {
            shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
            shared
                .last_round_micros
                .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
            return Err(e);
        }
        let snap = shared.bm.snapshot_dirty();
        let pending = shared.bm.snapshot_pending_deletes();
        let versioned_snap = match shared.bm.snapshot_dirty_versions(&snap) {
            Ok(versioned) => versioned,
            Err(e) => {
                shared.bm.restore_pending_deletes(pending);
                shared.bm.restore_dirty(snap);
                shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
                shared
                    .last_round_micros
                    .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
                return Err(e);
            }
        };
        (snap, pending, versioned_snap, None)
    };

    // 2. Force the WAL watermark before data-file writes, but do
    // not hold `commit_gate` across the fsync. Later writers may
    // append more WAL records while this flush runs; that is safe
    // because this epoch only writes the cloned dirty snapshot and
    // write-through retirement keeps newer dirty entries alive.
    if let (Some(journal), Some(up_to)) = (&shared.journal, wal_up_to) {
        if let Err(e) = journal.flush_up_to(up_to) {
            shared.bm.restore_pending_deletes(pending);
            shared.bm.restore_dirty(snap);
            shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
            shared
                .last_round_micros
                .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
            return Err(e);
        }
    }

    let (snap_bytes, stale_dirty) = match clone_versioned_dirty(shared, &versioned_snap) {
        Ok(cloned) => cloned,
        Err(e) => {
            shared.bm.restore_pending_deletes(pending);
            shared.bm.restore_dirty(snap);
            shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
            shared
                .last_round_micros
                .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
            return Err(e);
        }
    };
    if !stale_dirty.is_empty() {
        for guid in stale_dirty.keys() {
            snap.remove(guid);
        }
        shared.bm.restore_dirty(stale_dirty);
        if !pending.is_empty() {
            shared
                .bm
                .restore_pending_deletes(std::mem::take(&mut pending));
        }
    }
    let snap_count = snap.len();
    let pending_count = pending.len();
    shared.last_dirty_count.store(snap_count, Ordering::Relaxed);
    shared
        .last_pending_delete_count
        .store(pending_count, Ordering::Relaxed);

    // Early-skip only when nothing at all needs attention. A
    // pending deferred-delete from a previous round (e.g. one
    // whose `store.delete_blob` or trailing Sync failed and
    // got restored) was already drained above; check the
    // snapshot's length so we don't bail out on something we
    // just picked up. `needs_flush` covers the other recovery
    // edge: a prior round may have retired dirty entries after a
    // successful write-through but failed the following store
    // Sync, so there is still durable work even when dirty/pending
    // are both empty. A WAL-only round can skip store Sync but
    // must still retry truncate.
    let needs_store_flush = pipeline.in_flight.is_empty() && shared.bm.needs_flush();
    if snap.is_empty() && merged == 0 && pending.is_empty() && !needs_store_flush {
        pipeline.maybe_truncate(shared)?;
        shared.rounds_succeeded.fetch_add(1, Ordering::Relaxed);
        shared
            .last_round_micros
            .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "tracing")]
        tracing::trace!(target: "holt::checkpoint", "round skipped — nothing dirty");
        return Ok(());
    }

    // 4. Hand the whole epoch to the I/O worker. The planner has
    // already snapshotted durable intent under the commit-publish
    // gate; the worker can now drive data writes, store sync, pending
    // manifest deletes, and trailing sync without holding up writers
    // or future snapshot rounds.
    let entries: Vec<_> = snap_bytes
        .into_iter()
        .map(|(guid, seq, content_version, bytes)| WriteThroughEntry {
            guid,
            bytes,
            expected_seq: seq,
            content_version: Some(content_version),
        })
        .collect();
    let pending_for_recovery = pending.clone();
    let (tx, rx) = bounded(1);
    let epoch = CheckpointEpoch { entries, pending };
    if shared
        .io_tx
        .send(IoTask::CommitEpoch { epoch, on_done: tx })
        .is_err()
    {
        shared.bm.restore_pending_deletes(pending_for_recovery);
        shared.bm.restore_dirty(snap);
        shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
        shared
            .last_round_micros
            .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        return Err(Error::Internal(
            "checkpoint: I/O worker channel closed mid-round",
        ));
    }
    pipeline.push(PendingEpoch {
        rx,
        snap,
        pending: pending_for_recovery,
    });

    #[cfg(feature = "tracing")]
    {
        let elapsed = round_start.elapsed();
        tracing::info!(
            target: "holt::checkpoint",
            dirty_snapshot = snap_count,
            merged = merged,
            in_flight = pipeline.in_flight.len(),
            elapsed_us = elapsed.as_micros() as u64,
            "round submitted",
        );
    }

    shared
        .last_round_micros
        .store(round_start.elapsed().as_micros() as u64, Ordering::Relaxed);
    Ok(())
}

fn clone_versioned_dirty(
    shared: &Arc<Shared>,
    versioned: &[DirtySnapshotEntry],
) -> Result<(ClonedDirtyBytes, HashMap<BlobGuid, u64>)> {
    let mut snap_bytes = Vec::with_capacity(versioned.len());
    let mut stale = HashMap::new();
    for entry in versioned {
        match shared
            .bm
            .snapshot_bytes_if_version(entry.guid, entry.content_version)?
        {
            Some(bytes) => {
                snap_bytes.push((entry.guid, entry.expected_seq, entry.content_version, bytes));
            }
            None => {
                stale.insert(entry.guid, entry.expected_seq);
            }
        }
    }
    Ok((snap_bytes, stale))
}

/// Candidate-driven merge pass — fold mergeable `BlobNode`
/// children back into their parents. Stages the mutations via the
/// unified `mark_dirty` + `mark_for_delete` protocol so the round's
/// later checkpoint epoch (WAL flush → data writes → store sync →
/// pending deletes → re-Sync → truncate) handles persistence under W2D.
/// Takes the exclusive maintenance gate around one parent at a
/// time so no foreground writer is lock-coupling through the child
/// edge being folded and queued for delete. Foreground spillovers
/// enqueue parent blobs. Candidates that inspect only too-large
/// children are consumed; future spillovers or manual maintenance
/// seeding will requeue the parent when there is fresh shape debt.
///
/// Returns the cumulative count of children folded.
///
/// An inline `bm.commit(parent)` + `bm.delete_blob(child)` would
/// be wrong here — both happen pre-Sync, pre-WAL. `bm.commit`
/// would push cache bytes (potentially including user mutations
/// whose WAL records aren't yet durable) directly to store, and
/// `bm.delete_blob` would mutate the manifest in-memory which a
/// later `store.flush` could persist while the corresponding
/// user WAL records still hadn't reached disk. Staging through
/// dirty / pending-delete avoids both: the only flush path is the
/// round's checkpoint epoch, which runs strictly after step 2's
/// WAL flush.
fn run_merge_pass(shared: &Arc<Shared>) -> Result<u64> {
    use crate::store::STRUCTURAL_SEQ;

    let parents = shared.bm.pop_merge_candidates(256);
    let mut merged_total = 0u64;
    for guid in parents {
        let _maintenance = shared.maintenance_gate.enter_exclusive();
        if !shared.bm.has_blob(guid)? {
            continue;
        }
        let _commit = shared
            .journal
            .as_ref()
            .map(|_| shared.commit_gate.enter_writer());
        let pin = match shared.bm.pin(guid) {
            Ok(pin) => pin,
            Err(e) if is_blob_store_not_found(&e) => continue,
            Err(e) => return Err(e),
        };
        let (stats, has_children) = {
            let mut guard = pin.write();
            let mut frame = guard.frame();
            let stats = engine::try_merge_children(shared.bm.as_ref(), &mut frame, STRUCTURAL_SEQ)?;
            (stats, frame.header().num_ext_blobs != 0)
        };
        if stats.merged > 0 {
            // Keep the parent pin alive until after dirty
            // publication; otherwise eviction can drop the updated
            // cache image before this round snapshots it.
            shared.bm.mark_dirty(guid, STRUCTURAL_SEQ);
            merged_total += u64::from(stats.merged);
            if has_children {
                shared.bm.note_merge_candidate(guid);
            }
        }
        drop(pin);
    }
    shared.bm.note_merges(merged_total);
    Ok(merged_total)
}

fn finish_epoch(shared: &Arc<Shared>, report: CheckpointEpochReport) -> Result<()> {
    use std::sync::atomic::Ordering;

    shared
        .blobs_flushed
        .fetch_add(report.dirty_flushed as u64, Ordering::Relaxed);
    let dirty_total = report.dirty_total;
    let dirty_flushed = report.dirty_flushed;
    let pending_total = report.pending_total;
    let applied_deletes = report.applied_deletes;
    if let Err(e) = report.result {
        shared.rounds_failed.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "holt: checkpoint epoch failed (dirty={dirty_flushed}/{dirty_total}, pending deleted={applied_deletes}/{pending_total}): {e}",
        );
        return Err(e);
    }
    shared.rounds_succeeded.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn restore_unreported_epoch(shared: &Arc<Shared>, pending: PendingEpoch) {
    shared.bm.restore_pending_deletes(pending.pending);
    shared.bm.restore_dirty(pending.snap);
}
