//! `io_uring`-backed I/O context for [`super::PersistentBackend`].
//!
//! Only compiled when **both** of the following hold:
//!
//! - Target is Linux (`cfg(target_os = "linux")`).
//! - The `io-uring` feature is enabled.
//!
//! Otherwise the persistent backend stays on the `pread`/`pwrite`
//! syscall path. The feature gate keeps the `io-uring` crate out of
//! the default dependency closure (smaller build times, smaller
//! attack surface on platforms that don't use it) but lets Linux
//! users opt in to syscall-free I/O submission.
//!
//! ## Why a separate file
//!
//! The `io_uring` types (`IoUring`, `SubmissionQueueEntry`,
//! `CompletionQueueEntry`, …) are heavily `unsafe`-bound — keeping
//! them isolated here lets the rest of `PersistentBackend` stay
//! safe-Rust. The module exports exactly four operations:
//! [`UringContext::pread_at`], [`UringContext::pwrite_at`],
//! [`UringContext::pwrite_many_at`], and [`UringContext::new`].
//!
//! ## Concurrency
//!
//! One [`UringContext`] per [`super::PersistentBackend`]. The
//! backend wraps it in a `Mutex` so multiple writers serialise on
//! the submission queue. With a single I/O worker thread
//! (`holt-ckpt-io`) the lock is uncontended on the hot path.
//!
//! ## SQE depth
//!
//! `RING_DEPTH = 64` — enough to keep a local NVMe queue fed by
//! checkpoint batches without exploding CQ drain latency.

use std::io;
use std::os::unix::io::AsRawFd;

use io_uring::{opcode, types, IoUring};

/// Number of SQEs / CQEs the ring is sized for. Each checkpoint
/// blob write is one SQE; larger dirty snapshots are submitted in
/// ring-sized chunks.
const RING_DEPTH: u32 = 64;

/// Owns a single `io_uring` plus the `RawFd` of the file we
/// submit against. The file itself is owned by
/// [`super::PersistentBackend::data_file`]; this struct only
/// borrows its descriptor.
pub(super) struct UringContext {
    ring: IoUring,
    raw_fd: i32,
    fixed_fd: types::Fixed,
}

impl std::fmt::Debug for UringContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't dump the ring — would print SQ/CQ internals;
        // the fd alone is enough to identify which backend file
        // this context drives.
        f.debug_struct("UringContext")
            .field("fd", &self.raw_fd)
            .finish_non_exhaustive()
    }
}

impl UringContext {
    /// Build a fresh ring bound to `file`'s descriptor. Fails with
    /// `io::Error` if `IORING_SETUP_*` is rejected by the kernel
    /// (e.g. kernel too old).
    pub(super) fn new(file: &std::fs::File) -> io::Result<Self> {
        let ring = IoUring::new(RING_DEPTH)?;
        let raw_fd = file.as_raw_fd();
        ring.submitter().register_files(&[raw_fd])?;
        Ok(Self {
            ring,
            raw_fd,
            fixed_fd: types::Fixed(0),
        })
    }

    /// Synchronous `pwrite` via `io_uring`: push one SQE,
    /// `submit_and_wait(1)`, drain the CQE.
    ///
    /// The caller's `Mutex` over the `UringContext` ensures we
    /// never push a second SQE before the first's CQE has been
    /// drained — i.e. the SQ + CQ never get out of sync.
    pub(super) fn pwrite_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_many_at(&[(offset, buf)])
    }

    /// Synchronous batched `pwrite` via `io_uring`: push up to
    /// `RING_DEPTH` SQEs, submit once, then drain all completions.
    pub(super) fn pwrite_many_at(&mut self, writes: &[(u64, &[u8])]) -> io::Result<()> {
        for chunk in writes.chunks(RING_DEPTH as usize) {
            for (idx, (offset, buf)) in chunk.iter().enumerate() {
                let entry = opcode::Write::new(self.fixed_fd, buf.as_ptr(), buf.len() as u32)
                    .offset(*offset)
                    .build()
                    .user_data(idx as u64);

                // SAFETY: every SQE references a slice borrowed
                // from `writes`; this function synchronously waits
                // for all completions before returning, so all
                // buffers outlive the kernel reads.
                unsafe {
                    self.ring
                        .submission()
                        .push(&entry)
                        .map_err(|_| io::Error::other("uring SQ full"))?;
                }
            }
            self.ring.submit_and_wait(chunk.len())?;
            let mut seen = vec![false; chunk.len()];
            let mut complete = 0usize;
            while complete < chunk.len() {
                let cqe = self
                    .ring
                    .completion()
                    .next()
                    .ok_or_else(|| io::Error::other("uring CQE missing"))?;
                let idx = usize::try_from(cqe.user_data())
                    .map_err(|_| io::Error::other("uring CQE user_data overflow"))?;
                if idx >= chunk.len() {
                    return Err(io::Error::other("uring CQE user_data out of batch"));
                }
                if seen[idx] {
                    return Err(io::Error::other("duplicate uring CQE user_data"));
                }
                let n = cqe.result();
                if n < 0 {
                    return Err(io::Error::from_raw_os_error(-n));
                }
                let expected = chunk[idx].1.len();
                if (n as usize) != expected {
                    return Err(io::Error::other(format!(
                        "short uring write: wrote {} of {}",
                        n, expected
                    )));
                }
                seen[idx] = true;
                complete += 1;
            }
        }
        Ok(())
    }

    /// Synchronous `pread` via `io_uring`: same shape as
    /// [`Self::pwrite_at`].
    pub(super) fn pread_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let entry = opcode::Read::new(self.fixed_fd, buf.as_mut_ptr(), buf.len() as u32)
            .offset(offset)
            .build()
            .user_data(0);

        // SAFETY: same argument as `pwrite_at` — `buf` outlives the
        // synchronous `submit_and_wait`.
        unsafe {
            self.ring
                .submission()
                .push(&entry)
                .map_err(|_| io::Error::other("uring SQ full"))?;
        }
        self.ring.submit_and_wait(1)?;

        let cqe = self
            .ring
            .completion()
            .next()
            .ok_or_else(|| io::Error::other("uring CQE missing"))?;
        let n = cqe.result();
        if n < 0 {
            return Err(io::Error::from_raw_os_error(-n));
        }
        if (n as usize) != buf.len() {
            return Err(io::Error::other(format!(
                "short uring read: read {} of {}",
                n,
                buf.len()
            )));
        }
        Ok(())
    }
}

impl Drop for UringContext {
    fn drop(&mut self) {
        let _ = self.ring.submitter().unregister_files();
    }
}
