//! I/O worker thread — drains the bounded queue and runs
//! `store.write_blobs` / `store.flush` on behalf of the
//! checkpoint planner.
//!
//! ## Why a separate thread
//!
//! Decouples I/O execution from planning so the planner can:
//! 1. Snapshot bytes under a brief shared read guard, then move on.
//! 2. Enqueue checkpoint epochs without waiting for data writes.
//! 3. Keep each epoch's write/sync/report boundary explicit, so
//!    WAL truncation can advance strictly in checkpoint FIFO order.
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

use crossbeam_channel::{Receiver, Sender};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::api::errors::{Error, Result};
use crate::engine;
use crate::layout::BlobGuid;
use crate::store::blob_store::BlobStore;
use crate::store::{
    BlobFrameRef, BufferManager, WriteThroughEntry, WriteThroughStatus, STRUCTURAL_SEQ,
};

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
    /// run the pre-delete store sync, apply pending delete fences,
    /// then run the post-delete sync when a manifest delete applied.
    CommitEpoch {
        epoch: CheckpointEpoch,
        on_done: CheckpointEpochCompletion,
    },
    /// Graceful stop signal. Sent once during `Checkpointer::Drop`
    /// after the planner has joined and the final round has run.
    Stop,
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
    manifest_deleted_total: usize,
}

/// Main loop for the I/O thread.
pub(crate) fn run(shared: &Arc<Shared>, rx: Receiver<IoTask>) {
    while let Ok(task) = rx.recv() {
        match task {
            IoTask::CommitEpoch { mut epoch, on_done } => {
                let mut reports = commit_epoch_batch(shared, std::slice::from_mut(&mut epoch));
                let _ = on_done.send(reports.remove(0));
            }
            IoTask::Stop => break,
        }
    }
}

fn commit_epoch_batch(
    shared: &Arc<Shared>,
    epochs: &mut [CheckpointEpoch],
) -> Vec<CheckpointEpochReport> {
    // Serialize the complete write/sync/delete/sync transaction with manual
    // checkpoints. A stale older epoch must never overwrite a newer durable
    // parent after that newer epoch has already deleted one of its children.
    let _checkpoint_io = shared.bm.enter_checkpoint_io();
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
        match write_entries_in_dependency_order(&shared.bm, &mut entries, epochs.len()) {
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
    if pending_report.manifest_deleted_total > 0 {
        if let Err(e) = shared.bm.flush_inner() {
            restore_applied_manifest_deletes(shared, epochs, &pending_report.per_epoch_failed);
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
    let mut manifest_deleted_total = 0usize;
    for epoch in epochs {
        let mut pending_failed = HashMap::new();
        let mut first_pending_err = None;
        for (guid, seq) in &epoch.pending {
            match shared.bm.execute_pending_delete(*guid, *seq) {
                Ok(true) => {
                    if *seq != STRUCTURAL_SEQ {
                        manifest_deleted_total += 1;
                    }
                }
                Ok(false) => {
                    pending_failed.insert(*guid, *seq);
                }
                Err(e) => {
                    pending_failed.insert(*guid, *seq);
                    first_pending_err.get_or_insert(e);
                }
            }
        }
        if !pending_failed.is_empty() {
            shared.bm.restore_pending_deletes(pending_failed.clone());
        }
        per_epoch_failed.push(pending_failed);
        per_epoch_first_err.push(first_pending_err);
    }
    PendingDeleteReport {
        per_epoch_failed,
        per_epoch_first_err,
        manifest_deleted_total,
    }
}

fn collect_entry_children(entry: &WriteThroughEntry) -> Result<Vec<BlobGuid>> {
    let frame = BlobFrameRef::wrap(entry.bytes.as_slice());
    engine::collect_blob_children_from_frame(frame)
}

fn write_entries_in_dependency_order(
    bm: &Arc<BufferManager>,
    entries: &mut [BatchEntry],
    epoch_count: usize,
) -> Result<BatchWriteReport> {
    // A previous data/manifest operation may have left store-local flush
    // debt without a corresponding dirty BM entry. Child readiness treats
    // any such debt as non-durable, so merely deferring the parent would
    // reproduce the same empty dependency wave forever. Cross that frontier
    // explicitly before evaluating dependencies; failure is reported to the
    // planner, which restores every entry and pending delete for retry.
    if bm.needs_flush() {
        bm.flush_inner()?;
    }

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
                && children_ready(bm, &entry.children, &remaining_by_guid, &durable_this_batch)?
            {
                wave.push(idx);
            }
        }

        if wave.is_empty() {
            let deferred = entries.iter().any(|entry| !entry.flushed);
            if deferred {
                validate_no_progress_dependencies(
                    bm,
                    entries,
                    &remaining_by_guid,
                    &durable_this_batch,
                )?;
            }
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
        let report = bm.write_through_batch(&wave_entries)?;
        bm.flush_inner()?;

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

/// Classify a dependency planner stall. Only dependencies owned by dirty or
/// flushing work outside this batch may be retried: a missing durable child
/// and any cycle wholly inside the remaining batch are persistent corruption,
/// not scheduling backpressure.
fn validate_no_progress_dependencies(
    bm: &Arc<BufferManager>,
    entries: &[BatchEntry],
    remaining_by_guid: &HashMap<BlobGuid, usize>,
    durable_this_batch: &HashSet<BlobGuid>,
) -> Result<()> {
    let mut graph = HashMap::<BlobGuid, HashSet<BlobGuid>>::new();
    for guid in remaining_by_guid.keys().copied() {
        graph.entry(guid).or_default();
    }
    let mut external_transient = false;

    for entry in entries.iter().filter(|entry| !entry.flushed) {
        for child in &entry.children {
            if durable_this_batch.contains(child) {
                continue;
            }
            if remaining_by_guid.contains_key(child) {
                graph.entry(entry.guid).or_default().insert(*child);
                continue;
            }
            if bm.has_unflushed_blob(*child) {
                external_transient = true;
                continue;
            }
            if !bm.store_has_blob(*child)? {
                return Err(
                    Error::node_corrupt("checkpoint dependency references missing child")
                        .with_blob_guid(entry.guid),
                );
            }
            if !bm.store_has_durable_blob(*child)? {
                external_transient = true;
            }
        }
    }

    if let Some(guid) = dependency_cycle(&graph) {
        return Err(Error::node_corrupt("checkpoint dependency cycle").with_blob_guid(guid));
    }
    if external_transient {
        return Ok(());
    }
    Err(Error::Internal(
        "checkpoint dependency planner made no progress",
    ))
}

fn dependency_cycle(graph: &HashMap<BlobGuid, HashSet<BlobGuid>>) -> Option<BlobGuid> {
    // The graph is derived from potentially corrupt persisted frames, so its
    // depth is attacker-/corruption-controlled. Kahn's algorithm keeps the
    // diagnostic path iterative and bounded to O(V + E) heap memory instead
    // of risking a process stack overflow on a long acyclic chain.
    let mut indegree = graph
        .keys()
        .copied()
        .map(|guid| (guid, 0usize))
        .collect::<HashMap<_, _>>();
    for children in graph.values() {
        for child in children {
            if let Some(degree) = indegree.get_mut(child) {
                *degree = degree.saturating_add(1);
            }
        }
    }

    let mut ready = indegree
        .iter()
        .filter_map(|(guid, degree)| (*degree == 0).then_some(*guid))
        .collect::<VecDeque<_>>();
    let mut visited = 0usize;
    while let Some(guid) = ready.pop_front() {
        visited += 1;
        if let Some(children) = graph.get(&guid) {
            for child in children {
                let Some(degree) = indegree.get_mut(child) else {
                    continue;
                };
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(*child);
                }
            }
        }
    }

    (visited != indegree.len())
        .then(|| {
            indegree
                .into_iter()
                .find_map(|(guid, degree)| (degree != 0).then_some(guid))
        })
        .flatten()
}

fn children_ready(
    bm: &Arc<BufferManager>,
    children: &[BlobGuid],
    remaining_by_guid: &HashMap<BlobGuid, usize>,
    durable_this_batch: &HashSet<BlobGuid>,
) -> Result<bool> {
    for child in children {
        if durable_this_batch.contains(child) {
            continue;
        }
        if remaining_by_guid.contains_key(child) || bm.has_unflushed_blob(*child) {
            return Ok(false);
        }
        if !bm.store_has_durable_blob(*child)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Write one synchronous checkpoint snapshot in the same child-before-parent
/// durability waves used by the background I/O worker.
///
/// Every returned entry was either stale or deferred because one of its child
/// references was not yet durable. The caller must restore those entries to
/// the dirty map and retry them in a later checkpoint round. The caller must
/// hold [`BufferManager::enter_checkpoint_io`] across this call and its
/// pending-delete phase.
pub(crate) fn write_entries_child_first(
    bm: &Arc<BufferManager>,
    entries: Vec<WriteThroughEntry>,
) -> Result<Vec<(BlobGuid, u64)>> {
    let mut batch_entries = Vec::with_capacity(entries.len());
    for entry in entries {
        let children = collect_entry_children(&entry)?;
        batch_entries.push(BatchEntry {
            epoch_idx: 0,
            guid: entry.guid,
            expected_seq: entry.expected_seq,
            entry: Some(entry),
            children,
            flushed: false,
        });
    }

    let _ = write_entries_in_dependency_order(bm, &mut batch_entries, 1)?;
    Ok(batch_entries
        .iter()
        .filter(|entry| !entry.flushed)
        .map(|entry| (entry.guid, entry.expected_seq))
        .collect())
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

fn restore_applied_manifest_deletes(
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
                .filter(|(guid, seq)| **seq != STRUCTURAL_SEQ && !failed.contains_key(*guid))
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
                .unwrap_or(Error::Internal("checkpoint epoch write failed"))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::CheckpointConfig;
    use crate::concurrency::{CommitGate, Gate};
    use crate::layout::{BlobNode, NodeType};
    use crate::store::blob_store::{AlignedBlobBuf, BlobStore, FileBlobStore, MemoryBlobStore};
    use crate::store::{BlobFrame, BufferManager};
    use crossbeam_channel::bounded;
    use std::io;
    use std::mem::size_of;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Mutex};

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
        pending_flush: AtomicBool,
        fail_writes: bool,
        fail_flush: bool,
    }

    struct BlockingFirstBatchStore {
        inner: MemoryBlobStore,
        write_batches: AtomicUsize,
        first_entered: Barrier,
        release_first: Barrier,
    }

    impl BlockingFirstBatchStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                write_batches: AtomicUsize::new(0),
                first_entered: Barrier::new(2),
                release_first: Barrier::new(2),
            }
        }
    }

    impl BlobStore for BlockingFirstBatchStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.inner.write_blob(guid, src)
        }

        fn write_blobs_with_data_sync(&self, writes: &[(BlobGuid, &AlignedBlobBuf)]) -> Result<()> {
            let ordinal = self.write_batches.fetch_add(1, Ordering::AcqRel) + 1;
            if ordinal == 1 {
                // Pause the older epoch after its BM content validation but
                // before its store write becomes visible.
                self.first_entered.wait();
                self.release_first.wait();
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
            self.inner.flush()
        }

        fn needs_flush(&self) -> bool {
            self.inner.needs_flush()
        }

        fn has_blob(&self, guid: BlobGuid) -> Result<bool> {
            self.inner.has_blob(guid)
        }
    }

    impl CountingBatchStore {
        fn new() -> Self {
            Self {
                inner: MemoryBlobStore::new(),
                write_batches: AtomicUsize::new(0),
                flushes: AtomicUsize::new(0),
                events: Mutex::new(Vec::new()),
                pending_flush: AtomicBool::new(false),
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
                pending_flush: AtomicBool::new(false),
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
                pending_flush: AtomicBool::new(false),
                fail_writes: false,
                fail_flush: true,
            }
        }

        fn mark_pending_flush(&self) {
            self.pending_flush.store(true, Ordering::Release);
        }
    }

    impl BlobStore for CountingBatchStore {
        fn read_blob(&self, guid: BlobGuid, dst: &mut AlignedBlobBuf) -> Result<()> {
            self.inner.read_blob(guid, dst)
        }

        fn write_blob(&self, guid: BlobGuid, src: &AlignedBlobBuf) -> Result<()> {
            self.pending_flush.store(true, Ordering::Release);
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
            self.pending_flush.store(true, Ordering::Release);
            self.inner.write_blobs(writes)
        }

        fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
            self.pending_flush.store(true, Ordering::Release);
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
            self.inner.flush()?;
            self.pending_flush.store(false, Ordering::Release);
            Ok(())
        }

        fn needs_flush(&self) -> bool {
            self.pending_flush.load(Ordering::Acquire) || self.inner.needs_flush()
        }
    }

    fn test_shared<S: BlobStore + 'static>(store: Arc<S>) -> Arc<Shared> {
        let (io_tx, _io_rx) = bounded(1);
        Arc::new(Shared {
            bm: Arc::new(BufferManager::new(store, 8)),
            journal: None,
            volatile_journal: None,
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
            let off = frame.offset_of_slot(out.slot).unwrap();
            let node = BlobNode::new(&[], child);
            // Write the BlobNode body by raw offset (its `node_type` byte
            // isn't set yet, so `body_at_offset_mut` can't resolve it).
            let body = frame
                .bytes_at_mut(off, size_of::<BlobNode>() as u32)
                .unwrap();
            let bytes = unsafe {
                std::slice::from_raw_parts(std::ptr::from_ref(&node).cast(), size_of::<BlobNode>())
            };
            body.copy_from_slice(bytes);
            frame.header_mut().root_slot = crate::store::encode_child_off(off);
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
    fn epoch_batch_shares_one_store_batch_and_sync() {
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
    fn epoch_batch_preserves_repeated_blob_order() {
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
    fn epoch_write_error_restores_without_sync() {
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
    fn epoch_flush_error_restores_dirty_entry() {
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

    #[test]
    fn child_first_checkpoint_rejects_self_dependency_cycle() {
        let bm = Arc::new(BufferManager::new(Arc::new(MemoryBlobStore::new()), 8));
        let guid = [0xE7; 16];

        let error = write_entries_child_first(&bm, vec![entry(guid, 1, parent_blob(guid, guid))])
            .unwrap_err();

        assert!(matches!(
            error,
            Error::NodeCorrupt {
                context: "checkpoint dependency cycle",
                blob_guid: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn child_first_checkpoint_rejects_two_blob_dependency_cycle() {
        let bm = Arc::new(BufferManager::new(Arc::new(MemoryBlobStore::new()), 8));
        let first = [0xE8; 16];
        let second = [0xE9; 16];

        let error = write_entries_child_first(
            &bm,
            vec![
                entry(first, 1, parent_blob(first, second)),
                entry(second, 1, parent_blob(second, first)),
            ],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::NodeCorrupt {
                context: "checkpoint dependency cycle",
                blob_guid: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn dependency_cycle_handles_long_corrupt_graph_without_recursion() {
        const NODES: u32 = 50_000;
        let guid = |index: u32| {
            let mut guid = [0xA7; 16];
            guid[..4].copy_from_slice(&index.to_le_bytes());
            guid
        };
        let mut graph = HashMap::<BlobGuid, HashSet<BlobGuid>>::with_capacity(NODES as usize);
        for index in 0..NODES {
            let mut children = HashSet::new();
            if index + 1 < NODES {
                children.insert(guid(index + 1));
            }
            graph.insert(guid(index), children);
        }

        assert_eq!(dependency_cycle(&graph), None);
        graph.get_mut(&guid(NODES - 1)).unwrap().insert(guid(1));
        assert!(dependency_cycle(&graph).is_some());
    }

    #[test]
    fn child_first_checkpoint_rejects_permanently_missing_child() {
        let bm = Arc::new(BufferManager::new(Arc::new(MemoryBlobStore::new()), 8));
        let parent = [0xEA; 16];
        let missing = [0xEB; 16];

        let error =
            write_entries_child_first(&bm, vec![entry(parent, 1, parent_blob(parent, missing))])
                .unwrap_err();

        assert!(matches!(
            error,
            Error::NodeCorrupt {
                context: "checkpoint dependency references missing child",
                blob_guid: Some(guid),
                ..
            } if guid == parent
        ));
    }

    #[test]
    fn child_first_checkpoint_defers_external_dirty_child() {
        let bm = Arc::new(BufferManager::new(Arc::new(MemoryBlobStore::new()), 8));
        let parent = [0xEC; 16];
        let child = [0xED; 16];
        bm.install_new_blob(child, child_blob(child, 9), 1);

        let deferred =
            write_entries_child_first(&bm, vec![entry(parent, 2, parent_blob(parent, child))])
                .unwrap();

        assert_eq!(deferred, vec![(parent, 2)]);
        assert!(bm.has_unflushed_blob(child));
    }

    #[test]
    fn checkpoint_flushes_existing_child_manifest_before_parent_reference() {
        let store = Arc::new(CountingBatchStore::new());
        let shared = test_shared(Arc::clone(&store));
        let parent = [0xE3; 16];
        let child = [0xE4; 16];
        store
            .inner
            .write_blob(child, &child_blob(child, 9))
            .unwrap();
        store.mark_pending_flush();
        let mut epochs = vec![CheckpointEpoch {
            entries: vec![entry(parent, 1, parent_blob(parent, child))],
            pending: HashMap::new(),
        }];

        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].result.is_ok());
        assert_eq!(reports[0].dirty_flushed, 1);
        assert_eq!(
            *store.events.lock().unwrap(),
            vec![
                StoreEvent::Flush,
                StoreEvent::Write(vec![parent]),
                StoreEvent::Flush,
            ],
            "outstanding child manifest debt must become durable before parent write",
        );
        assert_eq!(shared.bm.dirty_count(), 0);
    }

    #[test]
    fn checkpoint_preflush_failure_restores_parent_for_retry() {
        let store = Arc::new(CountingBatchStore::failing_flush());
        let shared = test_shared(Arc::clone(&store));
        let parent = [0xE5; 16];
        let child = [0xE6; 16];
        store
            .inner
            .write_blob(child, &child_blob(child, 9))
            .unwrap();
        store.mark_pending_flush();
        let mut epochs = vec![CheckpointEpoch {
            entries: vec![entry(parent, 1, parent_blob(parent, child))],
            pending: HashMap::new(),
        }];

        let reports = commit_epoch_batch(&shared, &mut epochs);

        assert_eq!(reports.len(), 1);
        assert!(reports[0].result.is_err());
        assert_eq!(reports[0].dirty_flushed, 0);
        assert_eq!(*store.events.lock().unwrap(), vec![StoreEvent::Flush]);
        assert_eq!(store.write_batches.load(Ordering::Acquire), 0);
        assert_eq!(shared.bm.dirty_count(), 1);
    }

    #[test]
    fn synchronous_child_first_torn_parent_set_recovers_old_parent() {
        let dir = tempfile::tempdir().unwrap();
        let parent: BlobGuid = [
            0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E,
            0x9F, 0xA0,
        ];
        let child: BlobGuid = [
            0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE,
            0xAF, 0xB0,
        ];

        {
            let file = match FileBlobStore::open(dir.path()) {
                Ok(store) => Arc::new(store),
                Err(Error::BlobStoreIo(e)) if e.raw_os_error() == Some(libc::EINVAL) => {
                    eprintln!("skipping: O_DIRECT not supported on this fs");
                    return;
                }
                Err(e) => panic!("unexpected open error: {e}"),
            };
            // Keep an old durable parent mapping in the manifest. The later
            // checkpoint writes a shadow parent slot that points at a newly
            // created child, then publishes that slot in the manifest.
            file.write_blob(parent, &child_blob(parent, 1)).unwrap();
            file.flush().unwrap();

            let file_dyn: Arc<dyn BlobStore> = file;
            let bm = Arc::new(BufferManager::new(file_dyn, 8));
            write_entries_child_first(
                &bm,
                vec![
                    entry(parent, 2, parent_blob(parent, child)),
                    entry(child, 2, child_blob(child, 2)),
                ],
            )
            .unwrap();
        }

        let log_path = dir.path().join("manifest.log");
        let log = std::fs::read(&log_path).unwrap();
        let positions = |needle: &BlobGuid| {
            log.windows(needle.len())
                .enumerate()
                .filter_map(|(idx, bytes)| (bytes == needle).then_some(idx))
                .collect::<Vec<_>>()
        };
        let parent_positions = positions(&parent);
        let child_positions = positions(&child);
        assert_eq!(parent_positions.len(), 2, "initial + rewritten parent Set");
        assert_eq!(child_positions.len(), 1, "one newly-created child Set");
        assert!(
            child_positions[0] < parent_positions[1],
            "the child manifest record must be durable before the rewritten parent record",
        );

        // Keep the child Set and only a torn prefix of the later parent Set.
        // Reopen truncates that tail to the last complete record, so the old
        // parent image remains authoritative. The published child is an
        // unreachable orphan that a later clean-frontier reclaim may delete.
        const MANIFEST_RECORD_HEADER_SIZE: u64 = 9;
        let rewritten_parent_record =
            u64::try_from(parent_positions[1]).unwrap() - MANIFEST_RECORD_HEADER_SIZE;
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&log_path)
            .unwrap();
        f.set_len(rewritten_parent_record + 3).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let reopened = FileBlobStore::open(dir.path()).unwrap();
        let mut parent_bytes = AlignedBlobBuf::zeroed();
        reopened.read_blob(parent, &mut parent_bytes).unwrap();
        let children =
            engine::collect_blob_children_from_frame(BlobFrameRef::wrap(parent_bytes.as_slice()))
                .unwrap();
        assert!(
            children.is_empty(),
            "the torn remap must preserve the old parent"
        );
        let mut child_bytes = AlignedBlobBuf::zeroed();
        reopened.read_blob(child, &mut child_bytes).unwrap();
    }

    #[test]
    fn checkpoint_io_serializes_old_parent_before_new_parent_delete_epoch() {
        let store = Arc::new(BlockingFirstBatchStore::new());
        let parent = [0xB1; 16];
        let child = [0xB2; 16];
        store
            .inner
            .write_blob(child, &child_blob(child, 1))
            .unwrap();
        let shared = test_shared(Arc::clone(&store));
        shared.bm.mark_for_delete(child, 2);

        let old_shared = Arc::clone(&shared);
        let old_epoch = std::thread::spawn(move || {
            let mut epochs = vec![CheckpointEpoch {
                entries: vec![entry(parent, 1, parent_blob(parent, child))],
                pending: HashMap::new(),
            }];
            commit_epoch_batch(&old_shared, &mut epochs)
        });
        store.first_entered.wait();

        let new_shared = Arc::clone(&shared);
        let (new_done_tx, new_done_rx) = std::sync::mpsc::sync_channel(1);
        let new_epoch = std::thread::spawn(move || {
            let mut epochs = vec![CheckpointEpoch {
                entries: vec![entry(parent, 2, child_blob(parent, 2))],
                pending: HashMap::from([(child, 2)]),
            }];
            let reports = commit_epoch_batch(&new_shared, &mut epochs);
            new_done_tx.send(()).unwrap();
            reports
        });

        assert!(
            new_done_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "the newer parent+delete epoch must wait for the older epoch's complete I/O phase",
        );
        assert_eq!(
            store.write_batches.load(Ordering::Acquire),
            1,
            "no second epoch may enter store I/O while the first is paused",
        );
        store.release_first.wait();

        let old_reports = old_epoch.join().unwrap();
        new_done_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let new_reports = new_epoch.join().unwrap();
        assert!(old_reports.iter().all(|report| report.result.is_ok()));
        assert!(new_reports.iter().all(|report| report.result.is_ok()));
        assert!(!store.inner.has_blob(child).unwrap());
        let mut durable_parent = AlignedBlobBuf::zeroed();
        store.inner.read_blob(parent, &mut durable_parent).unwrap();
        assert!(
            engine::collect_blob_children_from_frame(
                BlobFrameRef::wrap(durable_parent.as_slice(),)
            )
            .unwrap()
            .is_empty(),
            "the final durable parent must be the newer child-free epoch",
        );
    }
}
