//! WAL append coordinator.
//!
//! `WalWriter` owns the file format and append/truncate mechanics.
//! This module owns concurrency around the WAL file. All append
//! records go through the worker, so foreground writers do not
//! perform per-operation WAL file writes.

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::api::errors::{Error, Result};

use super::writer::WalWriter;

const GROUP_COMMIT_WINDOW: Duration = Duration::from_micros(200);
const ENQUEUE_BATCH_WINDOW: Duration = Duration::from_micros(10);
const GROUP_COMMIT_MAX_BYTES: usize = 256 * 1024;
const RECORD_BUFFER_POOL_LIMIT: usize = 1024;
const RECORD_BUFFER_RETAIN_MAX: usize = 64 * 1024;
// Large enough to absorb metadata bursts without making async WAL
// enqueue a foreground backpressure point.
const JOURNAL_QUEUE_DEPTH: usize = 65_536;

type AckTx = Sender<Result<()>>;
type AckRx = Receiver<Result<()>>;

enum JournalCommand {
    Append {
        bytes: Vec<u8>,
        seq: u64,
        sync: bool,
        ack: Option<AckTx>,
    },
    Flush {
        up_to: u64,
        ack: AckTx,
    },
    Truncate {
        ack: AckTx,
    },
    Stop,
}

struct AppendRequest {
    bytes: Vec<u8>,
    seq: u64,
    sync: bool,
    ack: Option<AckTx>,
}

#[derive(Clone, Copy)]
struct WorkerState<'a> {
    writer: &'a Arc<Mutex<WalWriter>>,
    record_pool: &'a Mutex<Vec<Vec<u8>>>,
    batches: &'a AtomicU64,
    syncs: &'a AtomicU64,
    durable_work: &'a AtomicU64,
}

/// Completion handle for one acknowledged journal append.
///
/// Synchronous appends wait until their batch has reached the
/// `sync_data` durability boundary. Asynchronous appends return
/// without an acknowledgement handle.
pub(crate) struct JournalAck {
    rx: AckRx,
}

impl JournalAck {
    pub(crate) fn wait(self) -> Result<()> {
        self.rx
            .recv()
            .map_err(|_| Error::Internal("journal worker dropped append acknowledgement"))?
    }
}

/// WAL append coordinator.
///
/// Async records are serialized by the background worker, and
/// `wal_sync = true` waiters share one `sync_data` when they arrive
/// inside the short group window.
pub(crate) struct Journal {
    tx: Sender<JournalCommand>,
    handle: Mutex<Option<JoinHandle<()>>>,
    appends: Arc<AtomicU64>,
    batches: Arc<AtomicU64>,
    syncs: Arc<AtomicU64>,
    record_pool: Arc<Mutex<Vec<Vec<u8>>>>,
    next_work: Arc<AtomicU64>,
    wal_work: Arc<AtomicU64>,
    durable_work: Arc<AtomicU64>,
    checkpointed_work: Arc<AtomicU64>,
}

impl Journal {
    pub(crate) fn open_or_create(path: &std::path::Path, tree_id: u64) -> Result<Self> {
        let writer = WalWriter::open_or_create(path, tree_id)?;
        let initial_wal_work = u64::from(writer.has_records());
        let writer = Arc::new(Mutex::new(writer));
        let (tx, rx) = bounded::<JournalCommand>(JOURNAL_QUEUE_DEPTH);
        let batches = Arc::new(AtomicU64::new(0));
        let syncs = Arc::new(AtomicU64::new(0));
        // Existing WAL bytes are replayable, but we cannot prove
        // they were fsync-durable in the previous process. The
        // first checkpoint after reopen must issue a WAL flush
        // before it makes replayed effects durable in the store.
        let initial_durable_work = if initial_wal_work == 0 {
            0
        } else {
            initial_wal_work - 1
        };
        let record_pool = Arc::new(Mutex::new(Vec::new()));
        let durable_work = Arc::new(AtomicU64::new(initial_durable_work));
        let worker_batches = Arc::clone(&batches);
        let worker_syncs = Arc::clone(&syncs);
        let worker_record_pool = Arc::clone(&record_pool);
        let worker_durable_work = Arc::clone(&durable_work);
        let worker_writer = Arc::clone(&writer);
        let handle = thread::Builder::new()
            .name("holt-journal".to_owned())
            .spawn(move || {
                run_worker(
                    worker_writer,
                    rx,
                    worker_batches,
                    worker_syncs,
                    worker_record_pool,
                    worker_durable_work,
                );
            })
            .map_err(|_| Error::Internal("OS rejected thread spawn for holt-journal"))?;
        Ok(Self {
            tx,
            handle: Mutex::new(Some(handle)),
            appends: Arc::new(AtomicU64::new(0)),
            batches,
            syncs,
            record_pool,
            next_work: Arc::new(AtomicU64::new(initial_wal_work)),
            wal_work: Arc::new(AtomicU64::new(initial_wal_work)),
            durable_work,
            checkpointed_work: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Submit one fully encoded WAL record.
    pub(crate) fn submit(&self, bytes: Vec<u8>, sync: bool) -> Result<Option<JournalAck>> {
        let (ack, rx) = if sync {
            let (ack, rx) = bounded(1);
            (Some(ack), Some(rx))
        } else {
            (None, None)
        };
        let seq = self.next_work.fetch_add(1, Ordering::AcqRel) + 1;
        self.tx
            .send(JournalCommand::Append {
                bytes,
                seq,
                sync,
                ack,
            })
            .map_err(|_| Error::Internal("journal worker stopped before append"))?;
        self.appends.fetch_add(1, Ordering::Relaxed);
        self.wal_work.fetch_max(seq, Ordering::Release);
        Ok(rx.map(|rx| JournalAck { rx }))
    }

    /// Return a scratch buffer for one encoded WAL record.
    pub(crate) fn record_buffer(&self, min_capacity: usize) -> Vec<u8> {
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

    /// Highest WAL work id published by append paths.
    pub(crate) fn wal_work(&self) -> u64 {
        self.wal_work.load(Ordering::Acquire)
    }

    /// Force all WAL records up to `observed` durable.
    pub(crate) fn flush_up_to(&self, observed: u64) -> Result<()> {
        if observed <= self.durable_work.load(Ordering::Acquire) {
            return Ok(());
        }
        let (ack, rx) = bounded(1);
        self.tx
            .send(JournalCommand::Flush {
                up_to: observed,
                ack,
            })
            .map_err(|_| Error::Internal("journal worker stopped before flush"))?;
        recv_control_ack(rx, "journal worker dropped flush acknowledgement")
    }

    /// Reset the WAL to a fresh header-only file after checkpoint.
    pub(crate) fn truncate(&self) -> Result<()> {
        let observed = self.wal_work.load(Ordering::Acquire);
        if observed == self.checkpointed_work.load(Ordering::Acquire) {
            return Ok(());
        }
        let (ack, rx) = bounded(1);
        self.tx
            .send(JournalCommand::Truncate { ack })
            .map_err(|_| Error::Internal("journal worker stopped before truncate"))?;
        recv_control_ack(rx, "journal worker dropped truncate acknowledgement")?;
        self.checkpointed_work.fetch_max(observed, Ordering::AcqRel);
        Ok(())
    }

    pub(crate) fn needs_checkpoint(&self) -> bool {
        self.wal_work.load(Ordering::Acquire) != self.checkpointed_work.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn needs_flush(&self) -> bool {
        self.wal_work.load(Ordering::Acquire) > self.durable_work.load(Ordering::Acquire)
    }

    pub(crate) fn stats(&self) -> JournalStats {
        JournalStats {
            appends: self.appends.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            syncs: self.syncs.load(Ordering::Relaxed),
        }
    }
}

impl Drop for Journal {
    fn drop(&mut self) {
        let _ = self.tx.send(JournalCommand::Stop);
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

/// Production journal counters surfaced through `Tree::stats`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JournalStats {
    pub(crate) appends: u64,
    pub(crate) batches: u64,
    pub(crate) syncs: u64,
}

fn recv_control_ack(rx: AckRx, closed_msg: &'static str) -> Result<()> {
    rx.recv().map_err(|_| Error::Internal(closed_msg))?
}

fn run_worker(
    writer: Arc<Mutex<WalWriter>>,
    rx: Receiver<JournalCommand>,
    batches: Arc<AtomicU64>,
    syncs: Arc<AtomicU64>,
    record_pool: Arc<Mutex<Vec<Vec<u8>>>>,
    durable_work: Arc<AtomicU64>,
) {
    let mut backlog = VecDeque::new();
    let mut append_batch = Vec::with_capacity(256);
    let state = WorkerState {
        writer: &writer,
        record_pool: &record_pool,
        batches: &batches,
        syncs: &syncs,
        durable_work: &durable_work,
    };

    loop {
        let cmd = match backlog.pop_front() {
            Some(cmd) => cmd,
            None => match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            },
        };

        match cmd {
            JournalCommand::Append {
                bytes,
                seq,
                sync,
                ack,
            } => {
                process_append_batch(
                    AppendRequest {
                        bytes,
                        seq,
                        sync,
                        ack,
                    },
                    &rx,
                    &mut backlog,
                    &mut append_batch,
                    state,
                );
            }
            JournalCommand::Flush { up_to, ack } => {
                let res = writer
                    .lock()
                    .map_err(|_| Error::Internal("journal writer mutex poisoned"))
                    .and_then(|mut writer| writer.flush());
                if res.is_ok() {
                    syncs.fetch_add(1, Ordering::Relaxed);
                    durable_work.fetch_max(up_to, Ordering::AcqRel);
                }
                let _ = ack.send(res);
            }
            JournalCommand::Truncate { ack } => {
                let res = writer
                    .lock()
                    .map_err(|_| Error::Internal("journal writer mutex poisoned"))
                    .and_then(|mut writer| writer.truncate());
                let _ = ack.send(res);
            }
            JournalCommand::Stop => break,
        }
    }
}

fn process_append_batch(
    first: AppendRequest,
    rx: &Receiver<JournalCommand>,
    backlog: &mut VecDeque<JournalCommand>,
    batch: &mut Vec<AppendRequest>,
    state: WorkerState<'_>,
) {
    debug_assert!(batch.is_empty());
    batch.push(first);
    let mut needs_sync = batch[0].sync;
    let mut bytes = batch[0].bytes.len();
    let mut deadline = Instant::now()
        + if needs_sync {
            GROUP_COMMIT_WINDOW
        } else {
            ENQUEUE_BATCH_WINDOW
        };

    loop {
        if bytes >= GROUP_COMMIT_MAX_BYTES {
            break;
        }

        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let next = match rx.recv_timeout(deadline - now) {
            Ok(cmd) => cmd,
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
        };

        match next {
            JournalCommand::Append {
                bytes: record,
                seq,
                sync,
                ack,
            } => {
                bytes += record.len();
                let was_sync = needs_sync;
                needs_sync |= sync;
                if sync && !was_sync {
                    deadline = Instant::now() + GROUP_COMMIT_WINDOW;
                }
                batch.push(AppendRequest {
                    bytes: record,
                    seq,
                    sync,
                    ack,
                });
            }
            other => {
                backlog.push_back(other);
                break;
            }
        }
    }

    state.batches.fetch_add(1, Ordering::Relaxed);
    let result = write_append_batch(batch.as_slice(), needs_sync, state);
    notify_append_batch(batch, &result, state.record_pool);
}

fn write_append_batch(
    batch: &[AppendRequest],
    needs_sync: bool,
    state: WorkerState<'_>,
) -> Result<()> {
    let mut writer = state
        .writer
        .lock()
        .map_err(|_| Error::Internal("journal writer mutex poisoned"))?;
    for req in batch {
        writer.append_encoded(&req.bytes)?;
    }

    if needs_sync {
        writer.flush()?;
        state.syncs.fetch_add(1, Ordering::Relaxed);
        let durable_seq = batch.iter().map(|req| req.seq).max().unwrap_or(0);
        state.durable_work.fetch_max(durable_seq, Ordering::AcqRel);
    }
    Ok(())
}

fn notify_append_batch(
    batch: &mut Vec<AppendRequest>,
    result: &Result<()>,
    record_pool: &Mutex<Vec<Vec<u8>>>,
) {
    for mut req in batch.drain(..) {
        if let Some(ack) = req.ack.take() {
            let _ = ack.send(match result {
                Ok(()) => Ok(()),
                Err(Error::Internal(msg)) => Err(Error::Internal(msg)),
                Err(_) => Err(Error::Internal("journal worker failed")),
            });
        }
        recycle_record_buffer(record_pool, req.bytes);
    }
}

fn recycle_record_buffer(record_pool: &Mutex<Vec<Vec<u8>>>, mut buf: Vec<u8>) {
    if buf.capacity() > RECORD_BUFFER_RETAIN_MAX {
        return;
    }
    buf.clear();
    if let Ok(mut pool) = record_pool.try_lock() {
        if pool.len() < RECORD_BUFFER_POOL_LIMIT {
            pool.push(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::codec::FILE_HEADER_SIZE;

    #[test]
    fn fresh_journal_flush_and_truncate_are_noops() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        assert!(!journal.needs_checkpoint());
        journal.flush_up_to(journal.wal_work()).unwrap();
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
        journal.flush_up_to(journal.wal_work()).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);

        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            FILE_HEADER_SIZE as u64,
        );

        let syncs_after_truncate = journal.stats().syncs;
        journal.flush_up_to(journal.wal_work()).unwrap();
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
        journal.flush_up_to(journal.wal_work()).unwrap();
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

        journal.flush_up_to(journal.wal_work()).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);
        assert!(!journal.needs_flush());
        assert_eq!(journal.stats().syncs, 1);
        assert_eq!(journal.stats().appends, 1);
        assert!(journal.stats().batches >= 1);
    }

    #[test]
    fn encoded_record_buffers_are_recycled_after_worker_append() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal.wal"), 0).unwrap();

        let mut record = journal.record_buffer(64);
        let capacity = record.capacity();
        assert!(capacity >= 64);
        record.extend_from_slice(&[1; 32]);

        journal.submit(record, false).unwrap();
        journal.flush_up_to(journal.wal_work()).unwrap();

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
            journal.flush_up_to(journal.wal_work()).unwrap();
            assert!(journal.needs_checkpoint());
        }

        let journal = Journal::open_or_create(&path, 0).unwrap();
        assert!(journal.needs_checkpoint());
        assert!(journal.needs_flush());
        journal.flush_up_to(journal.wal_work()).unwrap();
        assert!(!journal.needs_flush());
        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
    }
}
