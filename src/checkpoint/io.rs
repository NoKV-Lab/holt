//! I/O worker thread — drains the bounded queue and runs
//! `store.write_blobs` / `store.flush` on behalf of the
//! checkpoint planner.
//!
//! ## Why a separate thread
//!
//! Decouples I/O execution from planning so the planner can:
//! 1. Snapshot bytes under a brief shared read guard, then move on.
//! 2. Enqueue checkpoint epochs without waiting for data writes.
//! 3. Let the worker coalesce adjacent epochs into one write/sync
//!    turn when the queue is already hot.
//!
//! For the current local-`pread`/`pwrite` store the parallelism
//! gain is modest (single thread, single FD). On Linux with the
//! `io-uring` feature, the I/O thread owns the SQ submit / CQ
//! drain path and feeds the ring with whole checkpoint batches.
//!
//! ## Shutdown
//!
//! The thread terminates on receiving [`IoTask::Stop`]. The
//! `Checkpointer` orchestrator sends one at the end of its `Drop`
//! sequence, after the final synchronous round has drained
//! everything through this same queue.

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::api::errors::{Error, Result};
use crate::layout::BlobGuid;
use crate::store::WriteThroughEntry;

use super::Shared;

/// One checkpoint epoch after the planner has drained dirty /
/// pending-delete state and cloned dirty blob bytes.
pub(crate) struct CheckpointEpoch {
    pub(crate) entries: Vec<WriteThroughEntry>,
    pub(crate) pending: HashMap<BlobGuid, u64>,
}

/// Completion payload for a checkpoint epoch.
pub(crate) struct CheckpointEpochReport {
    pub(crate) dirty_total: usize,
    pub(crate) dirty_flushed: usize,
    pub(crate) pending_total: usize,
    pub(crate) applied_deletes: usize,
    pub(crate) result: Result<()>,
}

pub(crate) type CheckpointEpochCompletion = Sender<CheckpointEpochReport>;

/// Work item handed to the I/O thread via the bounded queue.
pub(crate) enum IoTask {
    /// Commit one checkpoint epoch: write dirty blob images,
    /// run the pre-delete store sync, apply pending manifest
    /// deletes, then run the post-delete sync when needed.
    CommitEpoch {
        epoch: CheckpointEpoch,
        on_done: CheckpointEpochCompletion,
    },
    /// Graceful stop signal. Sent once during `Checkpointer::Drop`
    /// after the planner has joined and the final round has run.
    Stop,
}

struct EpochTask {
    epoch: CheckpointEpoch,
    on_done: CheckpointEpochCompletion,
}

#[derive(Clone, Copy)]
struct EpochProgress {
    dirty_total: usize,
    pending_total: usize,
}

const EPOCH_COALESCE_WINDOW: Duration = Duration::from_micros(100);
const MAX_COALESCED_EPOCHS: usize = 64;

/// Main loop for the I/O thread.
pub(crate) fn run(shared: &Arc<Shared>, rx: Receiver<IoTask>) {
    while let Ok(task) = rx.recv() {
        match task {
            IoTask::CommitEpoch { epoch, on_done } => {
                let mut batch = vec![EpochTask { epoch, on_done }];
                let stop_after_batch = collect_epoch_batch(&rx, &mut batch);
                let mut epochs = Vec::with_capacity(batch.len());
                let mut completions = Vec::with_capacity(batch.len());
                for task in batch {
                    epochs.push(task.epoch);
                    completions.push(task.on_done);
                }
                let reports = commit_epoch_batch(shared, &mut epochs);
                for (on_done, report) in completions.into_iter().zip(reports) {
                    let _ = on_done.send(report);
                }
                if stop_after_batch {
                    break;
                }
            }
            IoTask::Stop => break,
        }
    }
}

fn collect_epoch_batch(rx: &Receiver<IoTask>, batch: &mut Vec<EpochTask>) -> bool {
    let mut stop_after_batch = false;
    match rx.recv_timeout(EPOCH_COALESCE_WINDOW) {
        Ok(IoTask::CommitEpoch { epoch, on_done }) => batch.push(EpochTask { epoch, on_done }),
        Ok(IoTask::Stop) | Err(RecvTimeoutError::Disconnected) => return true,
        Err(RecvTimeoutError::Timeout) => return false,
    }
    while batch.len() < MAX_COALESCED_EPOCHS {
        match rx.try_recv() {
            Ok(IoTask::CommitEpoch { epoch, on_done }) => batch.push(EpochTask { epoch, on_done }),
            Ok(IoTask::Stop) | Err(TryRecvError::Disconnected) => {
                stop_after_batch = true;
                break;
            }
            Err(TryRecvError::Empty) => break,
        }
    }
    stop_after_batch
}

fn commit_epoch_batch(
    shared: &Arc<Shared>,
    epochs: &mut [CheckpointEpoch],
) -> Vec<CheckpointEpochReport> {
    let mut progresses = Vec::with_capacity(epochs.len());
    let mut all_entries = Vec::new();
    for epoch in epochs.iter_mut() {
        progresses.push(EpochProgress {
            dirty_total: epoch.entries.len(),
            pending_total: epoch.pending.len(),
        });
        all_entries.append(&mut epoch.entries);
    }

    if !all_entries.is_empty() {
        if let Err(e) = shared.bm.write_through_batch(&all_entries) {
            restore_dirty_entries(shared, &all_entries);
            restore_all_pending(shared, epochs);
            return reports_with_error(&progresses, false, e);
        }
    }
    if let Err(e) = shared.bm.flush_inner() {
        restore_all_pending(shared, epochs);
        return reports_with_error(&progresses, true, e);
    }

    let mut per_epoch_failed = Vec::with_capacity(epochs.len());
    let mut per_epoch_first_err = Vec::with_capacity(epochs.len());
    let mut applied_total = 0usize;
    for epoch in epochs.iter() {
        let mut pending_failed = HashMap::new();
        let mut first_pending_err = None;
        for (guid, seq) in &epoch.pending {
            if let Err(e) = shared.bm.execute_pending_delete(*guid) {
                pending_failed.insert(*guid, *seq);
                if first_pending_err.is_none() {
                    first_pending_err = Some(e);
                }
            }
        }
        applied_total += epoch.pending.len() - pending_failed.len();
        if !pending_failed.is_empty() {
            shared.bm.restore_pending_deletes(pending_failed.clone());
        }
        per_epoch_failed.push(pending_failed);
        per_epoch_first_err.push(first_pending_err);
    }
    if applied_total > 0 {
        if let Err(e) = shared.bm.flush_inner() {
            restore_applied_pending(shared, epochs, &per_epoch_failed);
            return reports_with_error(&progresses, true, e);
        }
    }

    epochs
        .iter()
        .zip(progresses)
        .zip(per_epoch_failed)
        .zip(per_epoch_first_err)
        .map(
            |(((epoch, progress), failed), first_err)| CheckpointEpochReport {
                dirty_total: progress.dirty_total,
                dirty_flushed: progress.dirty_total,
                pending_total: progress.pending_total,
                applied_deletes: epoch.pending.len() - failed.len(),
                result: first_err.map_or(Ok(()), Err),
            },
        )
        .collect()
}

fn restore_dirty_entries(shared: &Arc<Shared>, entries: &[WriteThroughEntry]) {
    if entries.is_empty() {
        return;
    }
    let mut failed = HashMap::with_capacity(entries.len());
    for entry in entries {
        failed.insert(entry.guid, entry.expected_seq);
    }
    shared.bm.restore_dirty(failed);
}

fn restore_all_pending(shared: &Arc<Shared>, epochs: &mut [CheckpointEpoch]) {
    let mut all_pending = HashMap::new();
    for epoch in epochs {
        all_pending.extend(std::mem::take(&mut epoch.pending));
    }
    shared.bm.restore_pending_deletes(all_pending);
}

fn restore_applied_pending(
    shared: &Arc<Shared>,
    epochs: &[CheckpointEpoch],
    per_epoch_failed: &[HashMap<BlobGuid, u64>],
) {
    let mut all_applied = HashMap::new();
    for (epoch, failed) in epochs.iter().zip(per_epoch_failed) {
        all_applied.extend(
            epoch
                .pending
                .iter()
                .filter(|(guid, _)| !failed.contains_key(*guid))
                .map(|(guid, seq)| (*guid, *seq)),
        );
    }
    shared.bm.restore_pending_deletes(all_applied);
}

fn reports_with_error(
    progresses: &[EpochProgress],
    dirty_flushed: bool,
    first_error: Error,
) -> Vec<CheckpointEpochReport> {
    let mut first_error = Some(first_error);
    progresses
        .iter()
        .map(|progress| CheckpointEpochReport {
            dirty_total: progress.dirty_total,
            dirty_flushed: if dirty_flushed {
                progress.dirty_total
            } else {
                0
            },
            pending_total: progress.pending_total,
            applied_deletes: 0,
            result: Err(first_error
                .take()
                .unwrap_or(Error::Internal("checkpoint epoch group failed"))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::CheckpointConfig;
    use crate::concurrency::{CommitGate, MaintenanceGate};
    use crate::store::blob_store::{AlignedBlobBuf, BlobStore, MemoryBlobStore};
    use crate::store::BufferManager;
    use crossbeam_channel::bounded;
    use std::io;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingBatchStore {
        inner: MemoryBlobStore,
        write_batches: AtomicUsize,
        flushes: AtomicUsize,
        fail_writes: bool,
    }

    impl CountingBatchStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                write_batches: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                fail_writes: false,
            }
        }

        fn failing_writes() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                write_batches: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                fail_writes: true,
            }
        }
    }

    impl BlobStore for CountingBatchStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            self.write_batches.fetch_add(1, Ordering::AcqRel);
            if self.fail_writes {
                return Err(Error::BlobStoreIo(io::Error::other(
                    "injected write failure",
                )));
            }
            self.inner.write_blobs(writes)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.inner.delete_blob(guid)
        }

        fn list_blobs(&self) -> Result<Vec<BlobGuid>> {
            self.inner.list_blobs()
        }

        fn flush(&self) -> Result<()> {
            self.flushes.fetch_add(1, Ordering::AcqRel);
            self.inner.flush()
        }

        fn needs_flush(&self) -> bool {
            self.inner.needs_flush()
        }
    }

    fn test_shared<S: BlobStore + 'static>(store: Arc<S>) -> Arc<Shared> {
        let (io_tx, _io_rx) = bounded(1);
        Arc::new(Shared {
            bm: Arc::new(BufferManager::new(store, 8)),
            journal: None,
            commit_gate: Arc::new(CommitGate::new()),
            maintenance_gate: Arc::new(MaintenanceGate::new()),
            cfg: CheckpointConfig::default(),
            io_tx,
            checkpoint_stop: AtomicBool::new(false),
            eviction_stop: AtomicBool::new(false),
            rounds_attempted: AtomicU64::new(0),
            rounds_succeeded: AtomicU64::new(0),
            blobs_flushed: AtomicU64::new(0),
            merges_total: AtomicU64::new(0),
            truncates: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            last_dirty_count: AtomicUsize::new(0),
        })
    }

    fn epoch(guid: BlobGuid, byte: u8) -> CheckpointEpoch {
        let mut buf = AlignedBlobBuf::zeroed();
        buf.as_mut_slice()[100] = byte;
        CheckpointEpoch {
            entries: vec![WriteThroughEntry {
                guid,
                bytes: buf,
                expected_seq: u64::from(byte),
            }],
            pending: HashMap::new(),
        }
    }

    #[test]
    fn coalesced_epochs_share_one_store_batch_and_sync() {
        let store = Arc::new(CountingBatchStore::new());
        let shared = test_shared(Arc::clone(&store));
        let first = epoch([0xA1; 16], 1);
        let second = epoch([0xA2; 16], 2);

        let mut epochs = vec![first, second];
        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| report.result.is_ok()));
        assert_eq!(store.write_batches.load(Ordering::Acquire), 1);
        assert_eq!(store.flushes.load(Ordering::Acquire), 1);
        assert_eq!(shared.bm.list_blobs().unwrap().len(), 2);
    }

    #[test]
    fn coalesced_epochs_preserve_repeated_blob_order() {
        let store = Arc::new(CountingBatchStore::new());
        let shared = test_shared(Arc::clone(&store));
        let guid = [0xC1; 16];
        let first = epoch(guid, 1);
        let second = epoch(guid, 2);

        let mut epochs = vec![first, second];
        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| report.result.is_ok()));
        assert_eq!(store.write_batches.load(Ordering::Acquire), 1);
        assert_eq!(store.flushes.load(Ordering::Acquire), 1);

        let mut out = AlignedBlobBuf::zeroed();
        shared.bm.read_blob(guid, &mut out).unwrap();
        assert_eq!(out.as_slice()[100], 2);
    }

    #[test]
    fn coalesced_epoch_write_error_restores_without_sync() {
        let store = Arc::new(CountingBatchStore::failing_writes());
        let shared = test_shared(Arc::clone(&store));
        let first = epoch([0xB1; 16], 1);

        let mut epochs = vec![first];
        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].result.is_err());
        assert_eq!(reports[0].dirty_flushed, 0);
        assert_eq!(store.write_batches.load(Ordering::Acquire), 1);
        assert_eq!(store.flushes.load(Ordering::Acquire), 0);
        assert_eq!(shared.bm.dirty_count(), 1);
    }
}
