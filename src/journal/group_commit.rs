//! WAL append coordinator — lock-free shared ring + single flusher.
//!
//! Concurrent writers reserve a byte range in the lock-free [`WalRing`]
//! (one `tail.fetch_add`), memcpy their encoded record in parallel, and
//! publish by folding the contiguous published prefix; a single background
//! **flusher** drains that prefix into the [`WalWriter`] (unchanged on-disk
//! format + replay) and fsyncs on the sync path. This replaced an earlier
//! per-record `Vec` + single crossbeam channel + single batching worker,
//! which serialized concurrent durable writes (see
//! `PERF_FINDINGS.md`: ~5–6× faster concurrent durable write, beating
//! RocksDB 2.8–5.5× at 1/4/8/16 threads).
//!
//! ## Watermarks live in the record-count domain
//!
//! `queued`/`written`/`flushed`/`checkpointed` count RECORDS (dense,
//! monotone), exactly mirroring the legacy work-id watermarks — so
//! `needs_checkpoint`, the reopen signal, and the checkpoint round-trip are
//! preserved bit-for-bit. `record_base` is the reopen offset (1 when the
//! file already had records, else 0). `written/flushed = record_base +
//! ring.committed_records()` at drain/fsync time; `queued = record_base +
//! records submitted this process`. They reconcile because every `submit`
//! both bumps `queued` and appends exactly one ring record.
//!
//! ## Durability: the flusher drains PROMPTLY
//!
//! Async (`wal_sync=false`) records must reach the OS page cache promptly so
//! they survive a *process* crash (as the legacy worker guarantees). The
//! flusher polls every `FLUSH_POLL` and on every wake drains the committed
//! prefix into `WalWriter` (whose 64 KB auto-drain reaches the page cache).
//! fsync happens only when a sync target is outstanding (sync write or
//! checkpoint barrier), exactly like legacy group commit.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};

use crate::api::errors::{Error, Result};

use super::ring::{ReserveTicket, WalRing};
use super::writer::WalWriter;

/// Production journal counters surfaced through `Tree::stats`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JournalStats {
    pub(crate) appends: u64,
    pub(crate) batches: u64,
    pub(crate) syncs: u64,
    pub(crate) queued_work: u64,
    pub(crate) written_work: u64,
    pub(crate) flushed_work: u64,
    pub(crate) checkpointed_work: u64,
    pub(crate) pending_work: u64,
    pub(crate) checkpoint_debt: u64,
}

/// In-RAM ring capacity. 16 MiB absorbs large metadata bursts between
/// checkpoints; records are ≤ ~512 KB so a single record always fits.
const RING_CAPACITY_BYTES: usize = 16 * 1024 * 1024;
/// Flusher idle poll. Bounds the async RAM→page-cache window (process-crash
/// durability) and the latency a sync waiter adds if a wake is ever missed.
const FLUSH_POLL: Duration = Duration::from_micros(50);
const RECORD_BUFFER_POOL_LIMIT: usize = 1024;
const RECORD_BUFFER_RETAIN_MAX: usize = 64 * 1024;

/// Control messages to the flusher (rare — never per async append).
enum Control {
    /// Wake to drain + (if a sync target is outstanding) fsync.
    Flush,
    /// Drain, then truncate the WAL to its header and reset the ring.
    Truncate(Sender<Result<()>>),
    Stop,
}

struct Shared {
    ring: WalRing,
    writer: Mutex<WalWriter>,
    /// Reopen offset: records already on disk before this process's ring.
    record_base: u64,

    queued: AtomicU64,
    written: AtomicU64,
    flushed: AtomicU64,
    checkpointed: AtomicU64,
    /// Highest record count some waiter needs fsync-durable.
    sync_target: AtomicU64,

    appends: AtomicU64,
    batches: AtomicU64,
    syncs: AtomicU64,

    /// Sticky flusher error message; fanned out to waiters and future
    /// barriers (a `&'static str` because `Error` is not `Clone`, matching
    /// how the legacy worker collapses failures to `Error::Internal`).
    err: Mutex<Option<&'static str>>,
    /// Condvar handshake for `flushed`/`err` waiters.
    flushed_mx: Mutex<()>,
    flushed_cv: Condvar,
    /// Condvar handshake for writers parked on ring backpressure.
    space_mx: Mutex<()>,
    space_cv: Condvar,

    control_tx: Sender<Control>,
    record_pool: Mutex<Vec<Vec<u8>>>,
}

impl Shared {
    fn sticky_err(&self) -> Option<&'static str> {
        *self.err.lock().unwrap()
    }

    fn set_err(&self, msg: &'static str) {
        let mut slot = self.err.lock().unwrap();
        if slot.is_none() {
            *slot = Some(msg);
        }
        drop(slot);
        // Wake any sync waiters so they observe the error.
        let _g = self.flushed_mx.lock().unwrap();
        self.flushed_cv.notify_all();
    }

    /// Drain the committed prefix into the writer; if a sync target is
    /// outstanding, fsync and advance `flushed`. Flusher thread only.
    fn drain_and_maybe_sync(&self) {
        if self.sticky_err().is_some() {
            return;
        }
        // Read the record count BEFORE copying: copy reads committed_addr,
        // which (addr-before-records publish ordering) is >= end(rc), so the
        // copy drains >= rc records and `base + rc` is a safe lower bound.
        let rc = self.ring.committed_records();
        let written_to = self.record_base + rc;

        let mut sink_err: Option<&'static str> = None;
        let mut freed_space = false;
        {
            let mut w = self.writer.lock().unwrap();
            let copied = self.ring.copy_committed_prefix(&mut |bytes| {
                if sink_err.is_none() && w.append_encoded(bytes).is_err() {
                    sink_err = Some("journal flusher append failed");
                }
            });
            if let Some(msg) = sink_err {
                drop(w);
                self.set_err(msg);
                return;
            }
            if copied > 0 {
                self.written.fetch_max(written_to, Ordering::AcqRel);
                self.batches.fetch_add(1, Ordering::Relaxed);
                freed_space = true;
            }
            let sync_target = self.sync_target.load(Ordering::Acquire);
            let flushed = self.flushed.load(Ordering::Acquire);
            let want_sync = sync_target > flushed && written_to >= sync_target;
            if want_sync {
                if w.flush().is_err() {
                    drop(w);
                    self.set_err("journal flusher fsync failed");
                    return;
                }
                self.syncs.fetch_add(1, Ordering::Relaxed);
                self.flushed.fetch_max(written_to, Ordering::AcqRel);
            }
        }
        if self.flushed.load(Ordering::Acquire) >= self.sync_target.load(Ordering::Acquire) {
            let _g = self.flushed_mx.lock().unwrap();
            self.flushed_cv.notify_all();
        }
        if freed_space {
            // The flusher advanced flush_cursor: wake any writer parked on
            // ring backpressure.
            let _g = self.space_mx.lock().unwrap();
            self.space_cv.notify_all();
        }
    }

    /// Block until the reserved range fits below `flush_cursor + capacity`
    /// (built-in backpressure). Parks on `space_cv` instead of spinning;
    /// rare in practice (the ring is sized to absorb bursts between
    /// checkpoints), but bounds RAM and CPU under sustained overload.
    fn wait_for_ring_space(&self, ticket: &ReserveTicket) -> Result<()> {
        let _ = self.control_tx.send(Control::Flush);
        let mut guard = self.space_mx.lock().unwrap();
        while !self.ring.reserve_space_ready(ticket) {
            if let Some(m) = self.sticky_err() {
                return Err(Error::Internal(m));
            }
            let _ = self.control_tx.send(Control::Flush);
            let (next, _timeout) = self
                .space_cv
                .wait_timeout(guard, FLUSH_POLL.saturating_mul(4))
                .unwrap();
            guard = next;
        }
        Ok(())
    }

    /// Block until `flushed >= target` (or a flusher error). Used by sync
    /// appends (`JournalAck::wait`) and `flush_up_to`.
    fn flush_to(&self, target: u64) -> Result<()> {
        if target <= self.flushed.load(Ordering::Acquire) {
            return match self.sticky_err() {
                Some(m) => Err(Error::Internal(m)),
                None => Ok(()),
            };
        }
        self.sync_target.fetch_max(target, Ordering::AcqRel);
        // Wake the flusher; the Flush is advisory (the loop recomputes the
        // sync target), so a full channel / closed receiver is non-fatal.
        let _ = self.control_tx.send(Control::Flush);
        let mut guard = self.flushed_mx.lock().unwrap();
        loop {
            if let Some(m) = self.sticky_err() {
                return Err(Error::Internal(m));
            }
            if self.flushed.load(Ordering::Acquire) >= target {
                return Ok(());
            }
            guard = self.flushed_cv.wait(guard).unwrap();
        }
    }

    fn record_buffer(&self, min_capacity: usize) -> Vec<u8> {
        if min_capacity <= RECORD_BUFFER_RETAIN_MAX {
            if let Ok(mut pool) = self.record_pool.try_lock() {
                while let Some(mut buf) = pool.pop() {
                    if buf.capacity() >= min_capacity {
                        buf.clear();
                        return buf;
                    }
                }
            }
        }
        Vec::with_capacity(min_capacity)
    }

    fn recycle(&self, mut buf: Vec<u8>) {
        if buf.capacity() == 0 || buf.capacity() > RECORD_BUFFER_RETAIN_MAX {
            return;
        }
        if let Ok(mut pool) = self.record_pool.try_lock() {
            if pool.len() < RECORD_BUFFER_POOL_LIMIT {
                buf.clear();
                pool.push(buf);
            }
        }
    }
}

/// Completion handle for one acknowledged journal append. Async appends
/// return `None`; sync appends return a handle whose `wait` blocks until the
/// record is fsync-durable.
pub(crate) struct JournalAck {
    shared: Arc<Shared>,
    target: u64,
}

impl JournalAck {
    pub(crate) fn wait(self) -> Result<()> {
        self.shared.flush_to(self.target)
    }
}

pub(crate) struct Journal {
    shared: Arc<Shared>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl Journal {
    pub(crate) fn open_or_create(path: &std::path::Path, tree_id: u64) -> Result<Self> {
        let writer = WalWriter::open_or_create(path, tree_id)?;
        let record_base = u64::from(writer.has_records());
        // Mirror legacy reopen seeding: a reopened non-empty WAL is queued
        // and unflushed, so the first checkpoint flushes before making
        // replayed effects durable.
        let initial_flushed = record_base.saturating_sub(1);

        let (control_tx, control_rx) = unbounded::<Control>();
        let shared = Arc::new(Shared {
            ring: WalRing::with_capacity(RING_CAPACITY_BYTES),
            writer: Mutex::new(writer),
            record_base,
            queued: AtomicU64::new(record_base),
            written: AtomicU64::new(record_base),
            flushed: AtomicU64::new(initial_flushed),
            checkpointed: AtomicU64::new(0),
            sync_target: AtomicU64::new(0),
            appends: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            syncs: AtomicU64::new(0),
            err: Mutex::new(None),
            flushed_mx: Mutex::new(()),
            flushed_cv: Condvar::new(),
            space_mx: Mutex::new(()),
            space_cv: Condvar::new(),
            control_tx,
            record_pool: Mutex::new(Vec::new()),
        });

        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("holt-journal-ring".to_owned())
            .spawn(move || run_flusher(worker_shared, control_rx))
            .map_err(|_| Error::Internal("OS rejected thread spawn for holt-journal-ring"))?;

        Ok(Self {
            shared,
            handle: Mutex::new(Some(handle)),
        })
    }

    /// Submit one fully encoded WAL record. The caller passes an owned buffer
    /// (recycled here after the ring copies it). Sync appends return an ack.
    pub(crate) fn submit(&self, bytes: Vec<u8>, sync: bool) -> Result<Option<JournalAck>> {
        if let Some(m) = self.shared.sticky_err() {
            return Err(Error::Internal(m));
        }
        if bytes.is_empty() {
            return Err(Error::Internal("journal record must not be empty"));
        }
        if bytes.len() as u64 > self.shared.ring.capacity() {
            return Err(Error::Internal("journal record exceeds WAL ring capacity"));
        }
        // Reserve → backpressure wait → memcpy → publish. Backpressure parks on
        // the flusher advancing `flush_cursor`, so a full ring does not spin.
        let ticket = self.shared.ring.reserve(bytes.len() as u64);
        if !self.shared.ring.reserve_space_ready(&ticket) {
            self.shared.wait_for_ring_space(&ticket)?;
        }
        self.shared.ring.fill(&ticket, &bytes);
        self.shared.ring.publish(&ticket);
        self.shared.recycle(bytes);

        let n = self.shared.queued.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.appends.fetch_add(1, Ordering::Relaxed);
        // No per-op flusher wake: the flusher's FLUSH_POLL drains async
        // records within the poll window (the async RAM→page-cache budget).
        // Sync appends + checkpoint barriers wake it explicitly via flush_to.

        if sync {
            Ok(Some(JournalAck {
                shared: Arc::clone(&self.shared),
                target: n,
            }))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn record_buffer(&self, min_capacity: usize) -> Vec<u8> {
        self.shared.record_buffer(min_capacity)
    }

    pub(crate) fn queued_work(&self) -> u64 {
        self.shared.queued.load(Ordering::Acquire)
    }

    pub(crate) fn flush_up_to(&self, observed: u64) -> Result<()> {
        self.shared.flush_to(observed)
    }

    pub(crate) fn truncate(&self) -> Result<()> {
        let observed = self.shared.queued.load(Ordering::Acquire);
        if observed == self.shared.checkpointed.load(Ordering::Acquire) {
            return Ok(());
        }
        let (ack, rx) = crossbeam_channel::bounded(1);
        self.shared
            .control_tx
            .send(Control::Truncate(ack))
            .map_err(|_| Error::Internal("journal flusher stopped before truncate"))?;
        rx.recv()
            .map_err(|_| Error::Internal("journal flusher dropped truncate acknowledgement"))??;
        self.shared
            .checkpointed
            .fetch_max(observed, Ordering::AcqRel);
        Ok(())
    }

    pub(crate) fn needs_checkpoint(&self) -> bool {
        self.shared.queued.load(Ordering::Acquire)
            != self.shared.checkpointed.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn needs_flush(&self) -> bool {
        self.shared.queued.load(Ordering::Acquire) > self.shared.flushed.load(Ordering::Acquire)
    }

    pub(crate) fn stats(&self) -> JournalStats {
        let queued_work = self.shared.queued.load(Ordering::Acquire);
        let written_work = self.shared.written.load(Ordering::Acquire);
        let flushed_work = self.shared.flushed.load(Ordering::Acquire);
        let checkpointed_work = self.shared.checkpointed.load(Ordering::Acquire);
        JournalStats {
            appends: self.shared.appends.load(Ordering::Relaxed),
            batches: self.shared.batches.load(Ordering::Relaxed),
            syncs: self.shared.syncs.load(Ordering::Relaxed),
            queued_work,
            written_work,
            flushed_work,
            checkpointed_work,
            pending_work: queued_work.saturating_sub(flushed_work),
            checkpoint_debt: queued_work.saturating_sub(checkpointed_work),
        }
    }
}

impl Drop for Journal {
    fn drop(&mut self) {
        let _ = self.shared.control_tx.send(Control::Stop);
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

fn run_flusher(shared: Arc<Shared>, control_rx: Receiver<Control>) {
    loop {
        shared.drain_and_maybe_sync();
        match control_rx.recv_timeout(FLUSH_POLL) {
            // Flush wake and idle-poll timeout both just re-loop to drain +
            // (re)check the sync target at the top.
            Ok(Control::Flush) | Err(RecvTimeoutError::Timeout) => {}
            Ok(Control::Truncate(ack)) => {
                // Drain anything outstanding, then reset. The caller holds
                // the commit gate exclusively at the checkpoint truncate
                // boundary, so no writer is mid-reserve here.
                shared.drain_and_maybe_sync();
                let result = do_truncate(&shared);
                let _ = ack.send(result);
            }
            Ok(Control::Stop) => {
                shared.drain_and_maybe_sync();
                break;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn do_truncate(shared: &Shared) -> Result<()> {
    if let Some(m) = shared.sticky_err() {
        return Err(Error::Internal(m));
    }
    let mut w = shared.writer.lock().unwrap();
    w.truncate()?;
    drop(w);
    // The ring is fully drained (the drain above caught up; no concurrent
    // writer under the checkpoint gate). Reset byte cursors; record count is
    // preserved as the stable cross-truncation order.
    shared.ring.reset_after_drain();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::codec::FILE_HEADER_SIZE;

    // The 6 legacy contract tests, retargeted at the ring-backed Journal.

    #[test]
    fn fresh_journal_flush_and_truncate_are_noops() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        assert!(!journal.needs_checkpoint());
        journal.flush_up_to(journal.queued_work()).unwrap();
        journal.truncate().unwrap();

        let stats = journal.stats();
        assert_eq!(stats.appends, 0);
        assert_eq!(stats.syncs, 0);
        assert!(!journal.needs_checkpoint());
    }

    #[test]
    fn append_requires_one_checkpoint_truncate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Journal::open_or_create(&path, 0).unwrap();

        journal.submit(vec![1, 2, 3, 4], false).unwrap();
        assert!(journal.needs_checkpoint());
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);

        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            FILE_HEADER_SIZE as u64
        );

        let syncs_after_truncate = journal.stats().syncs;
        journal.flush_up_to(journal.queued_work()).unwrap();
        journal.truncate().unwrap();
        assert_eq!(journal.stats().syncs, syncs_after_truncate);
    }

    #[test]
    fn durable_append_satisfies_later_flush_barrier() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        let ack = journal
            .submit(vec![5, 6, 7, 8], true)
            .unwrap()
            .expect("durable append returns an ack");
        ack.wait().unwrap();

        assert!(journal.needs_checkpoint());
        assert!(!journal.needs_flush());
        let syncs_after_append = journal.stats().syncs;
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert_eq!(journal.stats().syncs, syncs_after_append);

        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
    }

    #[test]
    fn enqueue_append_is_flushed_by_later_barrier() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        let journal = Journal::open_or_create(&path, 0).unwrap();

        let ack = journal.submit(vec![1, 3, 5, 7], false).unwrap();
        assert!(ack.is_none());

        journal.flush_up_to(journal.queued_work()).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);
        assert!(!journal.needs_flush());
        assert_eq!(journal.stats().syncs, 1);
        assert_eq!(journal.stats().appends, 1);
        assert!(journal.stats().batches >= 1);
    }

    #[test]
    fn invalid_record_size_is_rejected_without_poisoning_journal() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        assert!(journal.submit(Vec::new(), false).is_err());
        assert!(journal
            .submit(vec![0; RING_CAPACITY_BYTES + 1], false)
            .is_err());

        journal.submit(vec![1, 2, 3, 4], false).unwrap();
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert_eq!(journal.stats().appends, 1);
    }

    #[test]
    fn encoded_record_buffers_are_recycled_after_flusher_append() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        let mut record = journal.record_buffer(64);
        let capacity = record.capacity();
        assert!(capacity >= 64);
        record.extend_from_slice(&[1; 32]);

        journal.submit(record, false).unwrap();
        journal.flush_up_to(journal.queued_work()).unwrap();

        let reused = journal.record_buffer(16);
        assert!(reused.capacity() >= capacity);
    }

    #[test]
    fn reopened_nonempty_wal_still_needs_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        {
            let journal = Journal::open_or_create(&path, 0).unwrap();
            journal.submit(vec![9, 8, 7, 6], false).unwrap();
            journal.flush_up_to(journal.queued_work()).unwrap();
            assert!(journal.needs_checkpoint());
        }

        let journal = Journal::open_or_create(&path, 0).unwrap();
        assert!(journal.needs_checkpoint());
        assert!(journal.needs_flush());
        journal.flush_up_to(journal.queued_work()).unwrap();
        assert!(!journal.needs_flush());
        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
    }
}
