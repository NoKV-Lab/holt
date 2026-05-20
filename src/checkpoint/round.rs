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
//! 1. **Snapshot dirty + pending deletes** under the exclusive
//!    side of the tree's commit-publish gate.
//! 2. **Flush WAL** through the journal worker so every record that
//!    mirrors a snapshotted seq is durable before we drop it.
//! 3. **Clone snapshotted bytes** while still holding the same
//!    commit-publish gate, then submit one `IoTask::FlushBatch`.
//! 4. **Collect completion** — wait for the batch one-shot
//!    completion. On failure, restore the whole dirty snapshot via
//!    `bm.restore_dirty` because a backend batch may have written
//!    an arbitrary prefix.
//! 5. **Submit `Sync`** — one `IoTask::Sync` after the `FlushBatch`
//!    landed. `fdatasync` of the inner backend, including the
//!    PersistentBackend's manifest persist.
//! 6. **Truncate WAL** — only when (a) no `Flush` failed AND (b)
//!    `bm.dirty_count() == 0` checked under the commit-publish
//!    gate. The interlock with the writer-side dirty/journal
//!    publish order ensures we never drop a record whose effect
//!    isn't already in backend.
//!
//! This function is called from two places:
//!
//! - The `checkpoint_thread` main loop in [`super::mod`]
//!   (background path).
//! - `Checkpointer::Drop` (synchronous final round on the calling
//!   thread, after the planner has joined and writers are
//!   guaranteed to be gone).

use crossbeam_channel::bounded;
use std::collections::HashMap;
use std::sync::Arc;

use crate::api::errors::{Error, Result};
use crate::engine;
use crate::layout::BlobGuid;
use crate::store::backend::Backend;
use crate::store::buffer_manager::WriteThroughEntry;

use super::io::IoTask;
use super::Shared;

// The round is intentionally a single linear function so the 6
// phases stay readable as one story. Splitting it into helpers
// would hide the interlock between WAL flush / per-blob commit /
// dirty restore / truncate gate.
#[allow(clippy::too_many_lines)]
pub(super) fn run_round(shared: &Arc<Shared>) -> Result<()> {
    use std::sync::atomic::Ordering;

    shared.rounds_attempted.fetch_add(1, Ordering::Relaxed);

    // 0. Optional candidate-driven merge pass.
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

    #[cfg(feature = "tracing")]
    let round_start = std::time::Instant::now();

    // 1+2+3. Snapshot dirty AND pending-deletes, flush the journal,
    // then clone bytes under the same commit-publish gate used by
    // foreground persistent writers. Holding the gate through the
    // byte clone is load-bearing: a writer must not mutate a blob
    // between our dirty snapshot and `snapshot_bytes`, otherwise
    // the backend flush could include bytes whose WAL record was
    // not part of the durable snapshot.
    //
    // If `snapshot_pending_deletes` were taken outside this
    // commit-publish block, a writer could (a) enter its mutation,
    // (b) walker.erase that hits `SubtreeGone` (which calls
    // `mark_for_delete`), (c) submit the erase record, (d)
    // leave the gate, before we snapshot pending; we'd then
    // execute `backend.delete_blob` and re-Sync manifest while
    // the writer's WAL record was still only in the writer's
    // buffer. A crash there would leave the manifest ahead of
    // WAL — exactly the W2D violation deferred-delete was
    // designed to prevent.
    //
    // No-WAL trees (memory mode, user-supplied backend) skip the
    // journal flush but still clone immediately after draining.
    let (snap, pending, snap_bytes) = if let Some(journal) = &shared.journal {
        let _commit = shared.commit_gate.enter_checkpoint();
        let snap = shared.bm.snapshot_dirty();
        let pending = shared.bm.snapshot_pending_deletes();
        if let Err(e) = journal.flush() {
            shared.bm.restore_pending_deletes(pending);
            shared.bm.restore_dirty(snap);
            return Err(e);
        }
        let mut snap_bytes = Vec::with_capacity(snap.len());
        for (guid, txn_id) in &snap {
            let Some(bytes) = shared.bm.snapshot_bytes(*guid) else {
                let mut failed = HashMap::new();
                for (g, t) in &snap {
                    failed.entry(*g).or_insert(*t);
                }
                shared.bm.restore_pending_deletes(pending);
                shared.bm.restore_dirty(failed);
                return Err(Error::Internal(
                    "checkpoint: dirty entry lost cache image — invariant I1 violated",
                ));
            };
            snap_bytes.push((*guid, *txn_id, bytes));
        }
        (snap, pending, snap_bytes)
    } else {
        let snap = shared.bm.snapshot_dirty();
        let pending = shared.bm.snapshot_pending_deletes();
        let mut snap_bytes = Vec::with_capacity(snap.len());
        for (guid, txn_id) in &snap {
            let Some(bytes) = shared.bm.snapshot_bytes(*guid) else {
                let mut failed = HashMap::new();
                for (g, t) in &snap {
                    failed.entry(*g).or_insert(*t);
                }
                shared.bm.restore_pending_deletes(pending);
                shared.bm.restore_dirty(failed);
                return Err(Error::Internal(
                    "checkpoint: dirty entry lost cache image — invariant I1 violated",
                ));
            };
            snap_bytes.push((*guid, *txn_id, bytes));
        }
        (snap, pending, snap_bytes)
    };
    let snap_count = snap.len();
    shared.last_dirty_count.store(snap_count, Ordering::Relaxed);

    // Early-skip only when nothing at all needs attention. A
    // pending deferred-delete from a previous round (e.g. one
    // whose `backend.delete_blob` or trailing Sync failed and
    // got restored) was already drained above; check the
    // snapshot's length so we don't bail out on something we
    // just picked up.
    if snap.is_empty() && merged == 0 && pending.is_empty() {
        shared.rounds_succeeded.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "tracing")]
        tracing::trace!(target: "holt::checkpoint", "round skipped — nothing dirty");
        return Ok(());
    }

    // 3. Submit one batched Flush task. The snapshot bytes were
    // already cloned under the commit-publish gate above.
    let mut failed: HashMap<BlobGuid, u64> = HashMap::new();

    if !snap_bytes.is_empty() {
        let mut entries = Vec::with_capacity(snap_bytes.len());
        let mut expected = Vec::with_capacity(snap_bytes.len());
        for (guid, txn_id, bytes) in snap_bytes {
            expected.push((guid, txn_id));
            entries.push(WriteThroughEntry {
                guid,
                bytes,
                expected_seq: txn_id,
            });
        }
        let (tx, rx) = bounded(1);
        let task = IoTask::FlushBatch {
            entries,
            on_done: tx,
        };
        if shared.io_tx.send(task).is_err() {
            // I/O thread is gone (Drop is mid-sequence on another
            // path) — restore EVERYTHING we drained at step 1 so
            // the next round retries. The whole `pending` snapshot
            // is restored because phase 6 won't run.
            for (g, t) in &snap {
                failed.entry(*g).or_insert(*t);
            }
            shared.bm.restore_pending_deletes(pending);
            shared.bm.restore_dirty(failed);
            return Err(Error::Internal(
                "checkpoint: I/O worker channel closed mid-round",
            ));
        }

        // 4. Collect batch completion. On error, restore the whole
        // snapshot: `Backend::write_blobs` may have landed any
        // prefix, and retrying all entries is the only portable
        // recovery shape.
        match rx.recv() {
            Ok(Ok(())) => {
                shared
                    .blobs_flushed
                    .fetch_add(expected.len() as u64, Ordering::Relaxed);
            }
            Ok(Err(e)) => {
                eprintln!(
                    "holt: checkpoint flush batch failed ({} blobs): {e}",
                    expected.len()
                );
                for (guid, txn_id) in expected {
                    failed.insert(guid, txn_id);
                }
            }
            Err(_) => {
                // Sender dropped before sending — I/O thread died.
                for (guid, txn_id) in expected {
                    failed.insert(guid, txn_id);
                }
            }
        }
    }

    let had_dirty_failure = !failed.is_empty();
    if had_dirty_failure {
        shared.bm.restore_dirty(failed.clone());
    }

    // 5. Pre-delete Sync — a successful FlushBatch above retired
    //    dirty entries via write-through CAS; we must still
    //    fsync so those bytes are stable on disk before phase 6
    //    mutates the manifest. Each early-return path restores
    //    `pending` because phase 6 won't run.
    let (sync_tx, sync_rx) = bounded(1);
    if shared
        .io_tx
        .send(IoTask::Sync { on_done: sync_tx })
        .is_err()
    {
        shared.bm.restore_pending_deletes(pending);
        return Err(Error::Internal(
            "checkpoint: I/O worker channel closed before Sync",
        ));
    }
    match sync_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("holt: checkpoint backend Sync failed: {e}");
            shared.bm.restore_pending_deletes(pending);
            return Err(e);
        }
        Err(_) => {
            shared.bm.restore_pending_deletes(pending);
            return Err(Error::Internal(
                "checkpoint: I/O worker dropped Sync completion",
            ));
        }
    }

    // 5.5. Abort-on-dirty-failure gate. A failed parent write must
    //      NOT propagate to a manifest delete of its dependent
    //      child — that would orphan the parent's `BlobNode`
    //      pointer (parent on-disk still points to the child;
    //      manifest no longer has the child; WAL replay's walker
    //      descent would fail to read the deleted child). Restore
    //      `pending` and bail; the next round retries the parent
    //      write and only then processes its child's deletion.
    if had_dirty_failure {
        shared.bm.restore_pending_deletes(pending);
        return Err(Error::Internal(
            "checkpoint: dirty write failed — pending deletes deferred to next round",
        ));
    }

    // 6. Apply pending deletes — `pending` was already drained in
    //    step 1 under the commit-publish gate, so the writer-side
    //    WAL records covering each unlink op are durable on disk
    //    (via the step-2 journal flush). Phase 5 has fsync'd the
    //    per-blob writes that the manifest delete is allowed to
    //    follow. Safe to mutate the manifest now; the trailing
    //    re-Sync at step 7 persists it.
    let pending_count = pending.len();
    let mut pending_failed: HashMap<BlobGuid, u64> = HashMap::new();
    for (guid, seq) in &pending {
        if let Err(e) = shared.bm.execute_pending_delete(*guid) {
            eprintln!(
                "holt: checkpoint deferred delete failed for blob {:02x?} (seq={seq}): {e}",
                &guid[..4]
            );
            pending_failed.insert(*guid, *seq);
        }
    }
    if !pending_failed.is_empty() {
        shared.bm.restore_pending_deletes(pending_failed.clone());
    }

    // 7. Re-Sync iff we actually deleted anything — the manifest
    //    mutation at step 6 is in-memory until `backend.flush`
    //    rewrites the manifest file. Skip the syscall when the
    //    pending set was empty.
    let applied_deletes = pending_count - pending_failed.len();
    // Helper: on Sync failure here the manifest deletions we
    // already applied at step 6 are stuck in-memory. We can't
    // re-`execute_pending_delete` them (the slot is already
    // gone from the manifest map and the call is idempotent),
    // but we MUST keep them in the pending-delete set so the
    // truncate gate stays closed and the next round retries the
    // Sync. Re-registering with the same seq is idempotent
    // (min-merge in `restore_pending_deletes`).
    let restore_applied = || -> HashMap<BlobGuid, u64> {
        pending
            .iter()
            .filter(|(g, _)| !pending_failed.contains_key(*g))
            .map(|(g, s)| (*g, *s))
            .collect()
    };
    if applied_deletes > 0 {
        let (sync_tx2, sync_rx2) = bounded(1);
        if shared
            .io_tx
            .send(IoTask::Sync { on_done: sync_tx2 })
            .is_err()
        {
            shared.bm.restore_pending_deletes(restore_applied());
            return Err(Error::Internal(
                "checkpoint: I/O worker channel closed before Sync (deletes)",
            ));
        }
        match sync_rx2.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("holt: checkpoint backend Sync (deletes) failed: {e}");
                shared.bm.restore_pending_deletes(restore_applied());
                return Err(e);
            }
            Err(_) => {
                shared.bm.restore_pending_deletes(restore_applied());
                return Err(Error::Internal(
                    "checkpoint: I/O worker dropped Sync (deletes) completion",
                ));
            }
        }
    }

    // 8. Truncate WAL atomically iff every snapshot landed AND no
    //    racing writer has re-dirtied (under commit-gate check), AND
    //    no deferred deletes are still queued. The pending-delete
    //    gate is essential: a queued delete means a WAL record
    //    "this blob is unlinked" hasn't yet propagated to the
    //    manifest, so truncating would orphan the unlink.
    if failed.is_empty() && pending_failed.is_empty() {
        if let Some(journal) = &shared.journal {
            let _commit = shared.commit_gate.enter_checkpoint();
            if shared.bm.dirty_count() == 0 && shared.bm.pending_delete_count() == 0 {
                journal.truncate()?;
                shared.truncates.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    shared.rounds_succeeded.fetch_add(1, Ordering::Relaxed);

    #[cfg(feature = "tracing")]
    {
        let elapsed = round_start.elapsed();
        let truncated = failed.is_empty()
            && pending_failed.is_empty()
            && shared.journal.is_some()
            && shared.bm.dirty_count() == 0
            && shared.bm.pending_delete_count() == 0;
        tracing::info!(
            target: "holt::checkpoint",
            dirty_snapshot = snap_count,
            blobs_flushed = snap_count - failed.len(),
            blobs_failed = failed.len(),
            blobs_deleted = applied_deletes,
            merged = merged,
            truncated_wal = truncated,
            elapsed_us = elapsed.as_micros() as u64,
            "round complete",
        );
    }

    Ok(())
}

/// Candidate-driven merge pass — fold mergeable `BlobNode`
/// children back into their parents. Stages the mutations via the
/// unified `mark_dirty` + `mark_for_delete` protocol so the round's
/// later phases (WAL flush → FlushBatch → Sync → pending
/// deletes → re-Sync → truncate) handle persistence under W2D.
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
/// whose WAL records aren't yet durable) directly to backend, and
/// `bm.delete_blob` would mutate the manifest in-memory which a
/// later `backend.flush` could persist while the corresponding
/// user WAL records still hadn't reached disk. Staging through
/// dirty / pending-delete avoids both: the only flush path is
/// the round's own `IoTask::FlushBatch`, which runs strictly after
/// step 2's WAL flush.
fn run_merge_pass(shared: &Arc<Shared>) -> Result<u64> {
    use crate::store::buffer_manager::STRUCTURAL_SEQ;

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
        let pin = shared.bm.pin(guid)?;
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
