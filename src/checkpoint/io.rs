//! I/O worker thread — drains the bounded queue and runs
//! `backend.write_blobs` / `backend.flush` on behalf of the
//! checkpoint planner.
//!
//! ## Why a separate thread
//!
//! Decouples I/O execution from planning so the planner can:
//! 1. Snapshot bytes under a brief shared read guard, then move on.
//! 2. Submit one batch flush task without serialising on each I/O.
//! 3. Plan the next round's merge pass while the previous round's
//!    Sync is still in flight on the I/O thread.
//!
//! For the current local-`pread`/`pwrite` backend the parallelism
//! gain is modest (single thread, single FD). The architecture
//! pays off once the io_uring backend lands (next commit) — the
//! I/O thread becomes the SQE submitter + CQE poller, and the
//! planner's submit-N-then-wait pattern naturally feeds the ring.
//!
//! ## Shutdown
//!
//! The thread terminates on receiving [`IoTask::Stop`]. The
//! `Checkpointer` orchestrator sends one at the end of its `Drop`
//! sequence, after the final synchronous round has drained
//! everything through this same queue.

use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;

use crate::api::errors::Result;
use crate::store::buffer_manager::WriteThroughEntry;

use super::Shared;

/// One-shot completion channel — sized `bounded(1)` so a `send`
/// never blocks. The I/O worker sends `Ok(())` on success and
/// `Err(_)` on failure; the orchestrator receives once.
pub(crate) type Completion = Sender<Result<()>>;

/// Work item handed to the I/O thread via the bounded queue.
pub(crate) enum IoTask {
    /// Push a whole dirty snapshot to the inner backend. Bytes are
    /// owned by the task (snapshotted from cache by the planner)
    /// so the I/O thread doesn't touch BM read guards during the
    /// write.
    ///
    /// Each entry carries the dirty-map value observed when the
    /// planner drained the snapshot. The I/O worker retires those
    /// values only after the whole backend batch succeeds, guarding
    /// against racing writers and arbitrary-prefix partial backend
    /// failures.
    FlushBatch {
        entries: Vec<WriteThroughEntry>,
        on_done: Completion,
    },
    /// `fdatasync` (via `Backend::flush`). The orchestrator sends
    /// this after a `FlushBatch` completes so every
    /// blob's bytes are stable on disk before the WAL is
    /// truncated.
    Sync { on_done: Completion },
    /// Graceful stop signal. Sent once during `Checkpointer::Drop`
    /// after the planner has joined and the final round has run.
    Stop,
}

/// Main loop for the I/O thread.
pub(crate) fn run(shared: &Arc<Shared>, rx: Receiver<IoTask>) {
    while let Ok(task) = rx.recv() {
        match task {
            IoTask::FlushBatch { entries, on_done } => {
                let result = shared.bm.write_through_batch(&entries);
                // `send` only fails if the orchestrator dropped
                // the receiver — which only happens if the round
                // aborted or the Tree is shutting down. Either
                // way, no recovery action here; we just move on.
                let _ = on_done.send(result);
            }
            IoTask::Sync { on_done } => {
                let result = shared.bm.backend_flush();
                let _ = on_done.send(result);
            }
            IoTask::Stop => break,
        }
    }
}
