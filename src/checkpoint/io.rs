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
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use crate::api::errors::{Error, Result};
use crate::engine;
use crate::layout::BlobGuid;
use crate::store::{BlobFrameRef, WriteThroughEntry, WriteThroughStatus};

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

struct BatchEntry {
    epoch_idx: usize,
    guid: BlobGuid,
    expected_seq: u64,
    entry: Option<WriteThroughEntry>,
    children: Vec<BlobGuid>,
    flushed: bool,
}

struct BatchWriteReport {
    dirty_flushed_by_epoch: Vec<usize>,
    deferred: bool,
}

struct PendingDeleteReport {
    per_epoch_failed: Vec<HashMap<BlobGuid, u64>>,
    per_epoch_first_err: Vec<Option<Error>>,
    applied_total: usize,
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
    let mut entries = Vec::new();
    let mut collect_error = None;
    for (epoch_idx, epoch) in epochs.iter_mut().enumerate() {
        progresses.push(EpochProgress {
            dirty_total: epoch.entries.len(),
            pending_total: epoch.pending.len(),
        });
        for entry in epoch.entries.drain(..) {
            let children = match collect_entry_children(&entry) {
                Ok(children) => children,
                Err(e) => {
                    collect_error.get_or_insert(e);
                    Vec::new()
                }
            };
            entries.push(BatchEntry {
                epoch_idx,
                guid: entry.guid,
                expected_seq: entry.expected_seq,
                entry: Some(entry),
                children,
                flushed: false,
            });
        }
    }
    if let Some(e) = collect_error {
        restore_batch_entries(shared, &entries);
        restore_all_pending(shared, epochs);
        return reports_with_error(&progresses, vec![0; progresses.len()], e);
    }

    let dirty_flushed_by_epoch = if entries.is_empty() {
        if let Err(e) = shared.bm.flush_inner() {
            restore_all_pending(shared, epochs);
            return reports_with_error(&progresses, vec![0; progresses.len()], e);
        }
        vec![0; epochs.len()]
    } else {
        match write_entries_in_dependency_order(shared, &mut entries, epochs.len()) {
            Ok(report) => {
                if report.deferred {
                    restore_unflushed_batch_entries(shared, &entries);
                    restore_all_pending(shared, epochs);
                    return reports_without_delete_phase(
                        &progresses,
                        report.dirty_flushed_by_epoch,
                    );
                }
                report.dirty_flushed_by_epoch
            }
            Err(e) => {
                restore_batch_entries(shared, &entries);
                restore_all_pending(shared, epochs);
                return reports_with_error(&progresses, vec![0; progresses.len()], e);
            }
        }
    };

    let pending_report = apply_pending_deletes(shared, epochs);
    if pending_report.applied_total > 0 {
        if let Err(e) = shared.bm.flush_inner() {
            restore_applied_pending(shared, epochs, &pending_report.per_epoch_failed);
            return reports_with_error(&progresses, dirty_flushed_by_epoch, e);
        }
    }

    epochs
        .iter()
        .zip(progresses)
        .zip(dirty_flushed_by_epoch)
        .zip(pending_report.per_epoch_failed)
        .zip(pending_report.per_epoch_first_err)
        .map(
            |((((epoch, progress), dirty_flushed), failed), first_err)| CheckpointEpochReport {
                dirty_total: progress.dirty_total,
                dirty_flushed,
                pending_total: progress.pending_total,
                applied_deletes: epoch.pending.len() - failed.len(),
                result: first_err.map_or(Ok(()), Err),
            },
        )
        .collect()
}

fn apply_pending_deletes(shared: &Arc<Shared>, epochs: &[CheckpointEpoch]) -> PendingDeleteReport {
    let mut per_epoch_failed = Vec::with_capacity(epochs.len());
    let mut per_epoch_first_err = Vec::with_capacity(epochs.len());
    let mut applied_total = 0usize;
    for epoch in epochs {
        let mut pending_failed = HashMap::new();
        let mut first_pending_err = None;
        for (guid, seq) in &epoch.pending {
            match shared.bm.execute_pending_delete(*guid) {
                Ok(true) => {}
                Ok(false) => {
                    pending_failed.insert(*guid, *seq);
                }
                Err(e) => {
                    pending_failed.insert(*guid, *seq);
                    first_pending_err.get_or_insert(e);
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
    PendingDeleteReport {
        per_epoch_failed,
        per_epoch_first_err,
        applied_total,
    }
}

fn collect_entry_children(entry: &WriteThroughEntry) -> Result<Vec<BlobGuid>> {
    let frame = BlobFrameRef::wrap(entry.bytes.as_slice());
    engine::collect_blob_children_from_frame(frame)
}

fn write_entries_in_dependency_order(
    shared: &Arc<Shared>,
    entries: &mut [BatchEntry],
    epoch_count: usize,
) -> Result<BatchWriteReport> {
    let mut remaining_by_guid = HashMap::<BlobGuid, usize>::new();
    for entry in entries.iter() {
        *remaining_by_guid.entry(entry.guid).or_insert(0) += 1;
    }
    let mut durable_this_batch = HashSet::new();
    let mut dirty_flushed_by_epoch = vec![0; epoch_count];

    loop {
        let mut wave = Vec::new();
        for (idx, entry) in entries.iter().enumerate() {
            if !entry.flushed
                && children_ready(
                    shared,
                    &entry.children,
                    &remaining_by_guid,
                    &durable_this_batch,
                )?
            {
                wave.push(idx);
            }
        }

        if wave.is_empty() {
            let deferred = entries.iter().any(|entry| !entry.flushed);
            return Ok(BatchWriteReport {
                dirty_flushed_by_epoch,
                deferred,
            });
        }

        let wave_entries: Vec<_> = wave
            .iter()
            .map(|idx| {
                entries[*idx]
                    .entry
                    .take()
                    .expect("unflushed batch entry owns its write")
            })
            .collect();
        let report = shared.bm.write_through_batch(&wave_entries)?;
        shared.bm.flush_inner()?;

        let mut saw_stale = false;
        for (idx, status) in wave.into_iter().zip(report.statuses) {
            let entry = &mut entries[idx];
            match status {
                WriteThroughStatus::Written => {
                    entry.flushed = true;
                    dirty_flushed_by_epoch[entry.epoch_idx] += 1;
                    if let Some(count) = remaining_by_guid.get_mut(&entry.guid) {
                        *count -= 1;
                        if *count == 0 {
                            remaining_by_guid.remove(&entry.guid);
                        }
                    }
                    durable_this_batch.insert(entry.guid);
                }
                WriteThroughStatus::Stale => {
                    saw_stale = true;
                }
            }
        }
        if saw_stale {
            return Ok(BatchWriteReport {
                dirty_flushed_by_epoch,
                deferred: true,
            });
        }
    }
}

fn children_ready(
    shared: &Arc<Shared>,
    children: &[BlobGuid],
    remaining_by_guid: &HashMap<BlobGuid, usize>,
    durable_this_batch: &HashSet<BlobGuid>,
) -> Result<bool> {
    for child in children {
        if durable_this_batch.contains(child) {
            continue;
        }
        if remaining_by_guid.contains_key(child) || shared.bm.has_unflushed_blob(*child) {
            return Ok(false);
        }
        if !shared.bm.store_has_blob(*child)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn restore_batch_entries(shared: &Arc<Shared>, entries: &[BatchEntry]) {
    if entries.is_empty() {
        return;
    }
    let mut failed = HashMap::with_capacity(entries.len());
    for entry in entries {
        failed.insert(entry.guid, entry.expected_seq);
    }
    shared.bm.restore_dirty(failed);
}

fn restore_unflushed_batch_entries(shared: &Arc<Shared>, entries: &[BatchEntry]) {
    let mut failed = HashMap::new();
    for entry in entries.iter().filter(|entry| !entry.flushed) {
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

fn reports_without_delete_phase(
    progresses: &[EpochProgress],
    dirty_flushed_by_epoch: Vec<usize>,
) -> Vec<CheckpointEpochReport> {
    progresses
        .iter()
        .zip(dirty_flushed_by_epoch)
        .map(|(progress, dirty_flushed)| CheckpointEpochReport {
            dirty_total: progress.dirty_total,
            dirty_flushed,
            pending_total: progress.pending_total,
            applied_deletes: 0,
            result: Ok(()),
        })
        .collect()
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
    dirty_flushed_by_epoch: Vec<usize>,
    first_error: Error,
) -> Vec<CheckpointEpochReport> {
    let mut first_error = Some(first_error);
    progresses
        .iter()
        .zip(dirty_flushed_by_epoch)
        .map(|(progress, dirty_flushed)| CheckpointEpochReport {
            dirty_total: progress.dirty_total,
            dirty_flushed,
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
    use crate::concurrency::{CommitGate, Gate};
    use crate::layout::{BlobNode, NodeType};
    use crate::store::blob_store::{AlignedBlobBuf, BlobStore, MemoryBlobStore};
    use crate::store::{BlobFrame, BufferManager};
    use crossbeam_channel::bounded;
    use std::io;
    use std::mem::size_of;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, PartialEq, Eq)]
    enum StoreEvent {
        Write(Vec<BlobGuid>),
        Flush,
    }

    struct CountingBatchStore {
        inner: MemoryBlobStore,
        write_batches: AtomicUsize,
        flushes: AtomicUsize,
        events: Mutex<Vec<StoreEvent>>,
        fail_writes: bool,
        fail_flush: bool,
    }

    impl CountingBatchStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                write_batches: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                events: Mutex::new(Vec::new()),
                fail_writes: false,
                fail_flush: false,
            }
        }

        fn failing_writes() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                write_batches: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                events: Mutex::new(Vec::new()),
                fail_writes: true,
                fail_flush: false,
            }
        }

        fn failing_flush() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                write_batches: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                events: Mutex::new(Vec::new()),
                fail_writes: false,
                fail_flush: true,
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
            self.events.lock().unwrap().push(StoreEvent::Write(
                writes.iter().map(|(guid, _)| *guid).collect(),
            ));
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
            self.events.lock().unwrap().push(StoreEvent::Flush);
            if self.fail_flush {
                return Err(Error::BlobStoreIo(io::Error::other(
                    "injected flush failure",
                )));
            }
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
            maintenance_gate: Arc::new(Gate::new()),
            cfg: CheckpointConfig::default(),
            io_tx,
            checkpoint_stop: AtomicBool::new(false),
            eviction_stop: AtomicBool::new(false),
            rounds_attempted: AtomicU64::new(0),
            rounds_succeeded: AtomicU64::new(0),
            rounds_failed: AtomicU64::new(0),
            blobs_flushed: AtomicU64::new(0),
            merges_total: AtomicU64::new(0),
            truncates: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            last_dirty_count: AtomicUsize::new(0),
            last_pending_delete_count: AtomicUsize::new(0),
            last_round_micros: AtomicU64::new(0),
        })
    }

    fn epoch(guid: BlobGuid, byte: u8) -> CheckpointEpoch {
        let mut buf = AlignedBlobBuf::zeroed();
        {
            let _frame = BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
        }
        buf.as_mut_slice()[100] = byte;
        CheckpointEpoch {
            entries: vec![WriteThroughEntry {
                guid,
                bytes: buf,
                expected_seq: u64::from(byte),
                content_version: None,
            }],
            pending: HashMap::new(),
        }
    }

    fn child_blob(guid: BlobGuid, byte: u8) -> AlignedBlobBuf {
        let mut buf = AlignedBlobBuf::zeroed();
        {
            let _frame = BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
        }
        buf.as_mut_slice()[100] = byte;
        buf
    }

    fn parent_blob(parent: BlobGuid, child: BlobGuid) -> AlignedBlobBuf {
        let mut buf = AlignedBlobBuf::zeroed();
        {
            let mut frame = BlobFrame::init(buf.as_mut_slice(), parent).unwrap();
            let out = frame.alloc_node(NodeType::Blob).unwrap();
            let node = BlobNode::new(&[], child);
            let body = frame.body_of_slot_mut(out.slot).unwrap();
            let bytes = unsafe {
                std::slice::from_raw_parts(std::ptr::from_ref(&node).cast(), size_of::<BlobNode>())
            };
            body.copy_from_slice(bytes);
            frame.header_mut().root_slot = out.slot;
        }
        buf
    }

    fn entry(guid: BlobGuid, seq: u64, bytes: AlignedBlobBuf) -> WriteThroughEntry {
        WriteThroughEntry {
            guid,
            bytes,
            expected_seq: seq,
            content_version: None,
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

    #[test]
    fn coalesced_epoch_flush_error_restores_dirty_entry() {
        let store = Arc::new(CountingBatchStore::failing_flush());
        let shared = test_shared(Arc::clone(&store));
        let first = epoch([0xD1; 16], 1);

        let mut epochs = vec![first];
        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].result.is_err());
        assert_eq!(reports[0].dirty_flushed, 0);
        assert_eq!(store.write_batches.load(Ordering::Acquire), 1);
        assert_eq!(store.flushes.load(Ordering::Acquire), 1);
        assert_eq!(shared.bm.dirty_count(), 1);
    }

    #[test]
    fn stale_dirty_write_defers_pending_deletes() {
        let store = Arc::new(CountingBatchStore::new());
        let shared = test_shared(Arc::clone(&store));
        let parent = [0xD3; 16];
        let child = [0xD4; 16];
        let old_parent = parent_blob(parent, child);
        store.inner.write_blob(parent, &old_parent).unwrap();
        store
            .inner
            .write_blob(child, &child_blob(child, 9))
            .unwrap();

        let pin = shared.bm.pin(parent).unwrap();
        let old_version = pin.content_version();
        {
            let mut guard = pin.write();
            guard.as_mut_slice()[100] = 0xEE;
        }
        drop(pin);

        let mut pending = HashMap::new();
        pending.insert(child, 77);
        let mut epochs = vec![CheckpointEpoch {
            entries: vec![WriteThroughEntry {
                guid: parent,
                bytes: old_parent,
                expected_seq: 77,
                content_version: Some(old_version),
            }],
            pending,
        }];

        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].result.is_ok());
        assert_eq!(reports[0].dirty_flushed, 0);
        assert_eq!(reports[0].applied_deletes, 0);
        assert_eq!(shared.bm.dirty_count(), 1);
        assert_eq!(shared.bm.pending_delete_count(), 1);
        assert!(
            store.inner.has_blob(child).unwrap(),
            "child delete must wait until parent write is durable"
        );
    }

    #[test]
    fn checkpoint_flushes_child_manifest_before_parent_reference() {
        let store = Arc::new(CountingBatchStore::new());
        let shared = test_shared(Arc::clone(&store));
        let parent = [0xE1; 16];
        let child = [0xE2; 16];
        let mut epochs = vec![CheckpointEpoch {
            entries: vec![
                entry(parent, 1, parent_blob(parent, child)),
                entry(child, 2, child_blob(child, 9)),
            ],
            pending: HashMap::new(),
        }];

        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].result.is_ok());
        assert_eq!(reports[0].dirty_flushed, 2);
        assert_eq!(
            *store.events.lock().unwrap(),
            vec![
                StoreEvent::Write(vec![child]),
                StoreEvent::Flush,
                StoreEvent::Write(vec![parent]),
                StoreEvent::Flush,
            ]
        );
    }
}
