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
//! safe-Rust. The module exports exactly three operations:
//! [`UringContext::pread_at`], [`UringContext::pwrite_at`], and
//! [`UringContext::new`].
//!
//! ## Concurrency
//!
//! One [`UringContext`] per [`super::PersistentBackend`]. The
//! backend wraps it in a `Mutex` so multiple writers serialise on
//! the submission queue. With a single I/O worker thread (v0.2's
//! `holt-ckpt-io`) the lock is uncontended on the hot path.
//!
//! ## SQE depth
//!
//! `RING_DEPTH = 8` — comfortably accommodates one submit + one
//! completion at a time plus head-room for future batching (v0.3
//! will explore SQE-batched flushes once the planner can prepare a
//! whole snapshot of dirty blobs in one go).

use std::io;
use std::os::unix::io::AsRawFd;

use io_uring::{opcode, types, IoUring};

/// Number of SQEs / CQEs the ring is sized for. Each blob is one
/// SQE today; v0.3's batched-flush mode will saturate this.
const RING_DEPTH: u32 = 8;

/// Owns a single `io_uring` plus the `RawFd` of the file we
/// submit against. The file itself is owned by
/// [`super::PersistentBackend::data_file`]; this struct only
/// borrows its descriptor.
pub(super) struct UringContext {
    ring: IoUring,
    fd: types::Fd,
}

impl UringContext {
    /// Build a fresh ring bound to `file`'s descriptor. Fails with
    /// `io::Error` if `IORING_SETUP_*` is rejected by the kernel
    /// (e.g. kernel too old).
    pub(super) fn new(file: &std::fs::File) -> io::Result<Self> {
        let ring = IoUring::new(RING_DEPTH)?;
        let fd = types::Fd(file.as_raw_fd());
        Ok(Self { ring, fd })
    }

    /// Synchronous `pwrite` via `io_uring`: push one SQE,
    /// `submit_and_wait(1)`, drain the CQE.
    ///
    /// The caller's `Mutex` over the `UringContext` ensures we
    /// never push a second SQE before the first's CQE has been
    /// drained — i.e. the SQ + CQ never get out of sync.
    pub(super) fn pwrite_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        let entry = opcode::Write::new(self.fd, buf.as_ptr(), buf.len() as u32)
            .offset(offset)
            .build()
            .user_data(0);

        // SAFETY: the SQE references `buf` for the duration of the
        // operation; we synchronously `submit_and_wait` below
        // before returning, so the borrow outlives the kernel's
        // read of the buffer.
        unsafe {
            self.ring
                .submission()
                .push(&entry)
                .map_err(|_| io::Error::other("uring SQ full"))?;
        }
        self.ring.submit_and_wait(1)?;

        // Exactly one CQE per submit_and_wait(1).
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
                "short uring write: wrote {} of {}",
                n,
                buf.len()
            )));
        }
        Ok(())
    }

    /// Synchronous `pread` via `io_uring`: same shape as
    /// [`Self::pwrite_at`].
    pub(super) fn pread_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let entry = opcode::Read::new(self.fd, buf.as_mut_ptr(), buf.len() as u32)
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
