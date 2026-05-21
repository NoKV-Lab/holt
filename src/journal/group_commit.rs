//! Group-commit journal worker.
//!
//! `WalWriter` owns the file format and append/truncate mechanics.
//! This module owns concurrency: foreground writers enqueue fully
//! encoded WAL records, then wait outside the tree's commit-publish
//! critical section. Durable callers share one `sync_data` when they
//! arrive inside the short group window.

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::api::errors::{Error, Result};

use super::writer::WalWriter;

const GROUP_COMMIT_WINDOW: Duration = Duration::from_micros(200);
const GROUP_COMMIT_MAX_BYTES: usize = 256 * 1024;

type AckTx = Sender<Result<()>>;
type AckRx = Receiver<Result<()>>;

enum JournalCommand {
    Append {
        bytes: Vec<u8>,
        seq: u64,
        durable: bool,
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
    durable: bool,
    ack: Option<AckTx>,
}

/// Completion handle for one durable journal append.
///
/// Non-durable appends return after the in-process queue accepts
/// the record; durable appends return this handle and wait until
/// the worker has included the record in a `sync_data` batch.
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

/// Background WAL append worker with durable group commit.
pub(crate) struct Journal {
    tx: Sender<JournalCommand>,
    handle: Mutex<Option<JoinHandle<()>>>,
    appends: Arc<AtomicU64>,
    batches: Arc<AtomicU64>,
    syncs: Arc<AtomicU64>,
    next_work: Arc<AtomicU64>,
    wal_work: Arc<AtomicU64>,
    durable_work: Arc<AtomicU64>,
    checkpointed_work: Arc<AtomicU64>,
}

impl Journal {
    pub(crate) fn open_or_create(path: &std::path::Path, tree_id: u64) -> Result<Self> {
        let writer = WalWriter::open_or_create(path, tree_id)?;
        let initial_wal_work = u64::from(writer.has_records());
        let (tx, rx) = bounded::<JournalCommand>(1024);
        let batches = Arc::new(AtomicU64::new(0));
        let syncs = Arc::new(AtomicU64::new(0));
        // Existing WAL bytes are replayable, but we cannot prove
        // they were fsync-durable in the previous process. The
        // first checkpoint after reopen must issue a WAL flush
        // before it makes replayed effects durable in the backend.
        let initial_durable_work = if initial_wal_work == 0 {
            0
        } else {
            initial_wal_work - 1
        };
        let durable_work = Arc::new(AtomicU64::new(initial_durable_work));
        let worker_batches = Arc::clone(&batches);
        let worker_syncs = Arc::clone(&syncs);
        let worker_durable_work = Arc::clone(&durable_work);
        let handle = thread::Builder::new()
            .name("holt-journal".to_owned())
            .spawn(move || {
                run_worker(
                    writer,
                    rx,
                    worker_batches,
                    worker_syncs,
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
            next_work: Arc::new(AtomicU64::new(initial_wal_work)),
            wal_work: Arc::new(AtomicU64::new(initial_wal_work)),
            durable_work,
            checkpointed_work: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Submit one fully encoded WAL record.
    pub(crate) fn submit(&self, bytes: Vec<u8>, durable: bool) -> Result<Option<JournalAck>> {
        let (ack, rx) = if durable {
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
                durable,
                ack,
            })
            .map_err(|_| Error::Internal("journal worker stopped before append"))?;
        self.appends.fetch_add(1, Ordering::Relaxed);
        self.wal_work.fetch_max(seq, Ordering::Release);
        Ok(rx.map(|rx| JournalAck { rx }))
    }

    /// Drain every append submitted before this call and force the
    /// WAL file durable.
    pub(crate) fn flush(&self) -> Result<()> {
        let observed = self.wal_work.load(Ordering::Acquire);
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
    mut writer: WalWriter,
    rx: Receiver<JournalCommand>,
    batches: Arc<AtomicU64>,
    syncs: Arc<AtomicU64>,
    durable_work: Arc<AtomicU64>,
) {
    let mut backlog = VecDeque::new();

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
                durable,
                ack,
            } => {
                process_append_batch(
                    AppendRequest {
                        bytes,
                        seq,
                        durable,
                        ack,
                    },
                    &rx,
                    &mut backlog,
                    &mut writer,
                    &batches,
                    &syncs,
                    &durable_work,
                );
            }
            JournalCommand::Flush { up_to, ack } => {
                let res = writer.flush();
                if res.is_ok() {
                    syncs.fetch_add(1, Ordering::Relaxed);
                    durable_work.fetch_max(up_to, Ordering::AcqRel);
                }
                let _ = ack.send(res);
            }
            JournalCommand::Truncate { ack } => {
                let _ = ack.send(writer.truncate());
            }
            JournalCommand::Stop => break,
        }
    }
}

fn process_append_batch(
    first: AppendRequest,
    rx: &Receiver<JournalCommand>,
    backlog: &mut VecDeque<JournalCommand>,
    writer: &mut WalWriter,
    batches: &AtomicU64,
    syncs: &AtomicU64,
    durable_work: &AtomicU64,
) {
    let mut batch = vec![first];
    let mut durable = batch[0].durable;
    let mut bytes = batch[0].bytes.len();
    let mut deadline = durable.then(|| Instant::now() + GROUP_COMMIT_WINDOW);

    loop {
        if bytes >= GROUP_COMMIT_MAX_BYTES {
            break;
        }

        let next = match deadline {
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                match rx.recv_timeout(deadline - now) {
                    Ok(cmd) => cmd,
                    Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
                }
            }
            None => match rx.try_recv() {
                Ok(cmd) => cmd,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            },
        };

        match next {
            JournalCommand::Append {
                bytes: record,
                seq,
                durable: record_durable,
                ack,
            } => {
                bytes += record.len();
                durable |= record_durable;
                if record_durable && deadline.is_none() {
                    deadline = Some(Instant::now() + GROUP_COMMIT_WINDOW);
                }
                batch.push(AppendRequest {
                    bytes: record,
                    seq,
                    durable: record_durable,
                    ack,
                });
            }
            other => {
                backlog.push_back(other);
                break;
            }
        }
    }

    let mut ok = true;
    for req in &batch {
        if writer.append_encoded(&req.bytes).is_err() {
            ok = false;
            break;
        }
    }

    if ok && durable {
        ok = writer.flush().is_ok();
        if ok {
            syncs.fetch_add(1, Ordering::Relaxed);
            let durable_seq = batch.iter().map(|req| req.seq).max().unwrap_or(0);
            durable_work.fetch_max(durable_seq, Ordering::AcqRel);
        }
    }

    batches.fetch_add(1, Ordering::Relaxed);
    let result = if ok {
        Ok(())
    } else {
        Err(Error::Internal(
            "journal worker append or durable flush failed",
        ))
    };
    for req in batch {
        if let Some(ack) = req.ack {
            let _ = ack.send(match &result {
                Ok(()) => Ok(()),
                Err(Error::Internal(msg)) => Err(Error::Internal(msg)),
                Err(_) => Err(Error::Internal("journal worker failed")),
            });
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
        journal.flush().unwrap();
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
        journal.flush().unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);

        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            FILE_HEADER_SIZE as u64,
        );

        let syncs_after_truncate = journal.stats().syncs;
        journal.flush().unwrap();
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
        journal.flush().unwrap();
        assert_eq!(journal.stats().syncs, syncs_after_append);

        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
    }

    #[test]
    fn reopened_nonempty_wal_still_needs_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.wal");
        {
            let journal = Journal::open_or_create(&path, 0).unwrap();
            journal.submit(vec![9, 8, 7, 6], false).unwrap();
            journal.flush().unwrap();
            assert!(journal.needs_checkpoint());
        }

        let journal = Journal::open_or_create(&path, 0).unwrap();
        assert!(journal.needs_checkpoint());
        assert!(journal.needs_flush());
        journal.flush().unwrap();
        assert!(!journal.needs_flush());
        journal.truncate().unwrap();
        assert!(!journal.needs_checkpoint());
    }
}
