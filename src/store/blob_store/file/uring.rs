//! `io_uring`-backed I/O context for [`super::FileBlobStore`].
//!
//! Only compiled when **both** of the following hold:
//!
//! - Target is Linux (`cfg(target_os = "linux")`).
//! - The `io-uring` feature is enabled.
//!
//! Otherwise the persistent store stays on the `pread`/`pwrite`
//! syscall path. The feature gate keeps the `io-uring` crate out of
//! the default dependency closure (smaller build times, smaller
//! attack surface on platforms that don't use it) but lets Linux
//! users opt in to ring-backed data-file I/O.
//! The data file is registered as a fixed file, and data flushes
//! use `IORING_OP_FSYNC` with `DATASYNC` so the Linux fast path does
//! not bounce out to `File::sync_data`.
//!
//! ## Why a separate file
//!
//! The `io_uring` types (`IoUring`, `SubmissionQueueEntry`,
//! `CompletionQueueEntry`, …) are heavily `unsafe`-bound — keeping
//! them isolated here lets the rest of `FileBlobStore` stay
//! safe-Rust. The module exports only the store operations:
//! [`UringContext::pread_at`], [`UringContext::pread_many_at`],
//! [`UringContext::pwrite_at`],
//! [`UringContext::pwrite_many_at`],
//! [`UringContext::pwrite_many_and_sync_at`],
//! [`UringContext::sync_data`],
//! and [`UringContext::new`].
//!
//! ## Concurrency
//!
//! One [`UringContext`] per [`super::FileBlobStore`]. The
//! store wraps it in a `Mutex` so multiple writers serialise on
//! the submission queue. With a single I/O worker thread
//! (`holt-ckpt-io`) the lock is uncontended on the hot path.
//!
//! ## SQE depth
//!
//! `RING_DEPTH = 256` — enough to keep a local NVMe queue fed by
//! large checkpoint batches. Batches that fit in one ring keep
//! fixed-size CQ bookkeeping on the stack; larger checkpoint
//! bursts use a streaming refill/drain loop so the device sees a
//! sustained queue instead of chunk-sized stop-and-wait waves.
//! Batched writes are sorted by data-file offset first, matching
//! the default `pwritev` store's sequential shape.

use std::io;
use std::os::unix::io::AsRawFd;

use io_uring::{opcode, squeue, types, IoUring};

use crate::store::blob_frame::persisted_extent;
use crate::store::blob_store::{AlignedBlobBuf, BlobBufPool};

/// Number of SQEs / CQEs the ring is sized for. Each checkpoint
/// blob write is one SQE; larger dirty snapshots are submitted in
/// ring-sized chunks.
const RING_DEPTH: u32 = 256;
const RING_DEPTH_USIZE: usize = RING_DEPTH as usize;
const CQ_BITMAP_WORDS: usize = RING_DEPTH_USIZE.div_ceil(64);
/// Owns a single `io_uring` plus the `RawFd` of the file we
/// submit against. The file itself is owned by
/// [`super::FileBlobStore::data_file`]; this struct only
/// borrows its descriptor.
pub(super) struct UringContext {
    ring: IoUring,
    raw_fd: i32,
    fixed_fd: types::Fixed,
    fixed_buffers: bool,
}

#[derive(Clone, Copy)]
struct OrderedWrite<'a> {
    offset: u64,
    buf: &'a AlignedBlobBuf,
    /// Persisted extent of `buf`, a whole number of 4 KB pages.
    len: usize,
    order: usize,
}

impl std::fmt::Debug for UringContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't dump the ring — would print SQ/CQ internals;
        // the fd alone is enough to identify which store file
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
    pub(super) fn new(file: &std::fs::File, buffers: Option<&BlobBufPool>) -> io::Result<Self> {
        let ring = build_ring()?;
        let raw_fd = file.as_raw_fd();
        ring.submitter().register_files(&[raw_fd])?;
        let fixed_buffers = if let Some(buffers) = buffers {
            let iovecs = buffers.iovecs();
            // SAFETY: BlobBufPool owns every iovec's backing memory
            // for at least as long as this ring is registered. The
            // store drops/unregisters the ring before its pool Arc
            // can release the slab.
            unsafe {
                ring.submitter().register_buffers(&iovecs)?;
            }
            true
        } else {
            false
        };
        Ok(Self {
            ring,
            raw_fd,
            fixed_fd: types::Fixed(0),
            fixed_buffers,
        })
    }

    /// Synchronous `pwrite` via `io_uring`: push one SQE,
    /// `submit_and_wait(1)`, drain the CQE.
    ///
    /// The caller's `Mutex` over the `UringContext` ensures we
    /// never push a second SQE before the first's CQE has been
    /// drained — i.e. the SQ + CQ never get out of sync.
    pub(super) fn pwrite_at(&mut self, offset: u64, buf: &AlignedBlobBuf) -> io::Result<()> {
        let write = [OrderedWrite {
            offset,
            buf,
            len: persisted_extent(buf.as_slice()),
            order: 0,
        }];
        self.submit_write_batch(&write)
    }

    /// Synchronous batched `pwrite` via `io_uring`.
    ///
    /// Small batches push once and drain once. Larger checkpoint
    /// bursts keep the ring filled until every write completes,
    /// avoiding a hard wait at each `RING_DEPTH` boundary.
    pub(super) fn pwrite_many_at(&mut self, writes: &[(u64, &AlignedBlobBuf)]) -> io::Result<()> {
        let ordered = ordered_writes(writes);
        if ordered.is_empty() {
            return Ok(());
        }
        if ordered.len() <= RING_DEPTH_USIZE {
            return self.submit_write_batch(&ordered);
        }
        self.submit_write_stream(&ordered)
    }

    /// Submit a checkpoint write batch and then run a data-only
    /// fsync on the same ring.
    ///
    /// This deliberately keeps write completion and durability
    /// completion separate. Linked/drained fsync modes looked
    /// attractive in the code, but did not improve Holt's short
    /// checkpoint batches and made the completion path harder to
    /// reason about.
    pub(super) fn pwrite_many_and_sync_at(
        &mut self,
        writes: &[(u64, &AlignedBlobBuf)],
    ) -> io::Result<()> {
        if writes.is_empty() {
            return Ok(());
        }
        let ordered = ordered_writes(writes);
        if ordered.is_empty() {
            return Ok(());
        }
        if ordered.len() <= RING_DEPTH_USIZE {
            self.submit_write_batch(&ordered)?;
        } else {
            self.submit_write_stream(&ordered)?;
        }
        self.sync_data()
    }

    /// Synchronous `pread` via `io_uring`: same shape as
    /// [`Self::pwrite_at`].
    pub(super) fn pread_at(&mut self, offset: u64, buf: &mut AlignedBlobBuf) -> io::Result<()> {
        let entry = self.read_entry(offset, buf).user_data(0);

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

    /// Synchronous ranged `pread` via `io_uring`.
    ///
    /// This is used by cold point lookups, where the caller already
    /// holds a page-aligned window inside a 512 KiB scratch frame. The
    /// buffer is not necessarily one of the store's registered full
    /// frames, so this deliberately uses a normal `READ` SQE against
    /// the registered file instead of `READ_FIXED`.
    pub(super) fn pread_slice_at(&mut self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let entry = opcode::Read::new(self.fixed_fd, dst.as_mut_ptr(), dst.len() as u32)
            .offset(offset)
            .build()
            .user_data(0);

        // SAFETY: `dst` is borrowed until the single CQE is drained below.
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
        if (n as usize) != dst.len() {
            return Err(io::Error::other(format!(
                "short uring range read: read {} of {}",
                n,
                dst.len()
            )));
        }
        Ok(())
    }

    /// Synchronous batched `pread` via `io_uring`: the read analogue
    /// of [`Self::pwrite_many_at`].
    ///
    /// One ring submission per `RING_DEPTH`-sized chunk feeds the
    /// device a deep read queue from a single SQ owner, instead of
    /// the per-store `Mutex` serialising N independent `pread_at`
    /// calls down to queue-depth 1. This is what restores cold-scan
    /// read-ahead parallelism on the `io_uring` backend (on the
    /// `pread` backend the same queue depth comes for free from
    /// concurrent lock-free `read_exact_at` calls).
    ///
    /// Each `reads[i]` is `(byte_offset, dst)`; every `dst` must be a
    /// distinct buffer (the kernel writes each independently). Reads
    /// are submitted in slice order — callers that care about device
    /// LBA sequencing pre-sort by offset.
    pub(super) fn pread_many_at(
        &mut self,
        reads: &mut [(u64, &mut AlignedBlobBuf)],
    ) -> io::Result<()> {
        let total = reads.len();
        let mut base = 0usize;
        while base < total {
            let end = (base + RING_DEPTH_USIZE).min(total);
            self.submit_read_batch(&mut reads[base..end])?;
            base = end;
        }
        Ok(())
    }

    /// Synchronous `fdatasync` equivalent via `io_uring`.
    ///
    /// Callers only submit this after every prior write in the
    /// checkpoint batch has completed, matching `File::sync_data`
    /// ordering while keeping the Linux fast path on the ring.
    pub(super) fn sync_data(&mut self) -> io::Result<()> {
        let entry = self.fdatasync_entry();

        // SAFETY: no borrowed user buffer is attached to this SQE.
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
        check_fsync_result(n)
    }

    fn submit_write_batch(&mut self, chunk: &[OrderedWrite<'_>]) -> io::Result<()> {
        debug_assert!(!chunk.is_empty());
        debug_assert!(chunk.len() <= RING_DEPTH_USIZE);

        for (idx, write) in chunk.iter().enumerate() {
            let entry = self.write_entry(write).user_data(idx as u64);

            // SAFETY: every SQE references a slice borrowed from
            // `chunk`; this function synchronously waits for all
            // completions before returning, so all buffers outlive
            // the kernel reads.
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .map_err(|_| io::Error::other("uring SQ full"))?;
            }
        }

        self.ring.submit_and_wait(chunk.len())?;
        self.drain_write_batch(chunk)
    }

    fn submit_read_batch(&mut self, chunk: &mut [(u64, &mut AlignedBlobBuf)]) -> io::Result<()> {
        debug_assert!(!chunk.is_empty());
        debug_assert!(chunk.len() <= RING_DEPTH_USIZE);

        for (idx, (offset, buf)) in chunk.iter_mut().enumerate() {
            let entry = self.read_entry(*offset, buf).user_data(idx as u64);

            // SAFETY: every SQE references a buffer borrowed from
            // `chunk`; this function synchronously waits for all
            // completions before returning, so all buffers outlive
            // the kernel writes into them.
            unsafe {
                self.ring
                    .submission()
                    .push(&entry)
                    .map_err(|_| io::Error::other("uring SQ full"))?;
            }
        }

        self.ring.submit_and_wait(chunk.len())?;
        self.drain_read_batch(chunk)
    }

    fn submit_write_stream(&mut self, writes: &[OrderedWrite<'_>]) -> io::Result<()> {
        debug_assert!(writes.len() > RING_DEPTH_USIZE);

        let mut next = 0usize;
        let mut in_flight = 0usize;
        let mut completed = 0usize;
        let mut seen = vec![0u64; writes.len().div_ceil(64)];
        let mut first_err: Option<io::Error> = None;

        while completed < writes.len() {
            let mut pushed = 0usize;
            if first_err.is_none() {
                while next < writes.len() && in_flight < RING_DEPTH_USIZE {
                    let write = writes[next];
                    let entry = self.write_entry(&write).user_data(next as u64);

                    // SAFETY: each SQE borrows from `writes`; this
                    // function synchronously drains every submitted
                    // CQE before returning, so buffers outlive the
                    // kernel reads.
                    let pushed_ok = unsafe { self.ring.submission().push(&entry).is_ok() };
                    if !pushed_ok {
                        break;
                    }
                    next += 1;
                    in_flight += 1;
                    pushed += 1;
                }
            }

            if pushed > 0 {
                self.ring.submit()?;
            }

            if in_flight == 0 {
                if let Some(err) = first_err {
                    return Err(err);
                }
                return Err(io::Error::other("uring SQ full with no writes in flight"));
            }

            let drained = self.drain_write_stream_available(
                writes,
                &mut seen,
                &mut first_err,
                &mut in_flight,
                &mut completed,
            )?;
            if drained == 0 {
                self.ring.submit_and_wait(1)?;
                self.drain_write_stream_available(
                    writes,
                    &mut seen,
                    &mut first_err,
                    &mut in_flight,
                    &mut completed,
                )?;
            }

            if first_err.is_some() && in_flight == 0 {
                break;
            }
        }

        if let Some(err) = first_err {
            return Err(err);
        }
        if completed != writes.len() {
            return Err(io::Error::other(format!(
                "missing uring write completions: completed {completed} of {}",
                writes.len()
            )));
        }
        Ok(())
    }

    fn write_entry(&self, write: &OrderedWrite<'_>) -> squeue::Entry {
        let len = write.len as u32;
        if self.fixed_buffers {
            if let Some(buffer_index) = write.buf.fixed_buffer_index() {
                return opcode::WriteFixed::new(
                    self.fixed_fd,
                    write.buf.as_ptr(),
                    len,
                    buffer_index,
                )
                .offset(write.offset)
                .build();
            }
        }
        opcode::Write::new(self.fixed_fd, write.buf.as_ptr(), len)
            .offset(write.offset)
            .build()
    }

    fn read_entry(&self, offset: u64, buf: &mut AlignedBlobBuf) -> squeue::Entry {
        if self.fixed_buffers {
            if let Some(buffer_index) = buf.fixed_buffer_index() {
                return opcode::ReadFixed::new(
                    self.fixed_fd,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                    buffer_index,
                )
                .offset(offset)
                .build();
            }
        }
        opcode::Read::new(self.fixed_fd, buf.as_mut_ptr(), buf.len() as u32)
            .offset(offset)
            .build()
    }

    fn fdatasync_entry(&self) -> squeue::Entry {
        opcode::Fsync::new(self.fixed_fd)
            .flags(types::FsyncFlags::DATASYNC)
            .build()
    }

    fn drain_write_batch(&mut self, chunk: &[OrderedWrite<'_>]) -> io::Result<()> {
        let mut seen = [0u64; CQ_BITMAP_WORDS];
        let mut first_err: Option<io::Error> = None;

        for _ in 0..chunk.len() {
            let cqe = self
                .ring
                .completion()
                .next()
                .ok_or_else(|| io::Error::other("uring CQE missing"))?;
            let user_data = cqe.user_data();
            let n = cqe.result();

            let Ok(idx) = usize::try_from(user_data) else {
                record_err(
                    &mut first_err,
                    io::Error::other("uring CQE user_data overflow"),
                );
                continue;
            };
            if idx >= chunk.len() {
                record_err(
                    &mut first_err,
                    io::Error::other("uring CQE user_data out of batch"),
                );
                continue;
            }
            if let Err(e) = mark_seen(&mut seen, idx) {
                record_err(&mut first_err, e);
                continue;
            }
            if n < 0 {
                record_err(&mut first_err, io::Error::from_raw_os_error(-n));
                continue;
            }
            let expected = chunk[idx].len;
            if (n as usize) != expected {
                record_err(
                    &mut first_err,
                    io::Error::other(format!("short uring write: wrote {n} of {expected}")),
                );
            }
        }

        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    }

    fn drain_read_batch(&mut self, chunk: &[(u64, &mut AlignedBlobBuf)]) -> io::Result<()> {
        let mut seen = [0u64; CQ_BITMAP_WORDS];
        let mut first_err: Option<io::Error> = None;

        for _ in 0..chunk.len() {
            let cqe = self
                .ring
                .completion()
                .next()
                .ok_or_else(|| io::Error::other("uring CQE missing"))?;
            let user_data = cqe.user_data();
            let n = cqe.result();

            let Ok(idx) = usize::try_from(user_data) else {
                record_err(
                    &mut first_err,
                    io::Error::other("uring CQE user_data overflow"),
                );
                continue;
            };
            if idx >= chunk.len() {
                record_err(
                    &mut first_err,
                    io::Error::other("uring CQE user_data out of batch"),
                );
                continue;
            }
            if let Err(e) = mark_seen(&mut seen, idx) {
                record_err(&mut first_err, e);
                continue;
            }
            if n < 0 {
                record_err(&mut first_err, io::Error::from_raw_os_error(-n));
                continue;
            }
            let expected = chunk[idx].1.len();
            if (n as usize) != expected {
                record_err(
                    &mut first_err,
                    io::Error::other(format!("short uring read: read {n} of {expected}")),
                );
            }
        }

        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    }

    fn drain_write_stream_available(
        &mut self,
        writes: &[OrderedWrite<'_>],
        seen: &mut [u64],
        first_err: &mut Option<io::Error>,
        in_flight: &mut usize,
        completed: &mut usize,
    ) -> io::Result<usize> {
        let mut drained = 0usize;
        while let Some(cqe) = self.ring.completion().next() {
            drained += 1;
            *completed += 1;
            *in_flight = in_flight
                .checked_sub(1)
                .ok_or_else(|| io::Error::other("uring CQE without matching in-flight write"))?;

            let user_data = cqe.user_data();
            let n = cqe.result();
            let Ok(idx) = usize::try_from(user_data) else {
                record_err(first_err, io::Error::other("uring CQE user_data overflow"));
                continue;
            };
            if idx >= writes.len() {
                record_err(
                    first_err,
                    io::Error::other("uring CQE user_data out of stream"),
                );
                continue;
            }
            if let Err(e) = mark_seen_dynamic(seen, idx) {
                record_err(first_err, e);
                continue;
            }
            if n < 0 {
                record_err(first_err, io::Error::from_raw_os_error(-n));
                continue;
            }
            let expected = writes[idx].len;
            if (n as usize) != expected {
                record_err(
                    first_err,
                    io::Error::other(format!("short uring write: wrote {n} of {expected}")),
                );
            }
        }
        Ok(drained)
    }
}

impl Drop for UringContext {
    fn drop(&mut self) {
        if self.fixed_buffers {
            let _ = self.ring.submitter().unregister_buffers();
        }
        let _ = self.ring.submitter().unregister_files();
    }
}

fn build_ring() -> io::Result<IoUring> {
    let mut builder = IoUring::builder();
    builder.setup_cqsize(RING_DEPTH * 2).setup_clamp();
    builder.build(RING_DEPTH)
}

fn ordered_writes<'a>(writes: &'a [(u64, &'a AlignedBlobBuf)]) -> Vec<OrderedWrite<'a>> {
    let mut ordered: Vec<_> = writes
        .iter()
        .enumerate()
        .map(|(order, (offset, buf))| OrderedWrite {
            offset: *offset,
            buf,
            len: persisted_extent(buf.as_slice()),
            order,
        })
        .collect();
    ordered.sort_by(|a, b| a.offset.cmp(&b.offset).then(a.order.cmp(&b.order)));

    let mut deduped = Vec::with_capacity(ordered.len());
    let mut idx = 0usize;
    while idx < ordered.len() {
        let offset = ordered[idx].offset;
        let mut last = ordered[idx];
        idx += 1;
        while idx < ordered.len() && ordered[idx].offset == offset {
            last = ordered[idx];
            idx += 1;
        }
        deduped.push(last);
    }
    deduped
}

fn check_fsync_result(n: i32) -> io::Result<()> {
    if n < 0 {
        return Err(io::Error::from_raw_os_error(-n));
    }
    if n != 0 {
        return Err(io::Error::other(format!(
            "unexpected uring fdatasync result: {n}",
        )));
    }
    Ok(())
}

fn record_err(slot: &mut Option<io::Error>, err: io::Error) {
    if slot.is_none() {
        *slot = Some(err);
    }
}

fn mark_seen(seen: &mut [u64; CQ_BITMAP_WORDS], idx: usize) -> io::Result<()> {
    let word = idx / 64;
    let bit = 1u64 << (idx % 64);
    if seen[word] & bit != 0 {
        return Err(io::Error::other("duplicate uring CQE user_data"));
    }
    seen[word] |= bit;
    Ok(())
}

fn mark_seen_dynamic(seen: &mut [u64], idx: usize) -> io::Result<()> {
    let word = idx / 64;
    let bit = 1u64 << (idx % 64);
    let Some(seen_word) = seen.get_mut(word) else {
        return Err(io::Error::other("uring CQE user_data out of bitmap"));
    };
    if *seen_word & bit != 0 {
        return Err(io::Error::other("duplicate uring CQE user_data"));
    }
    *seen_word |= bit;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    fn frame(fill: u8) -> AlignedBlobBuf {
        let mut buf = AlignedBlobBuf::zeroed();
        buf.as_mut_slice().fill(fill);
        buf
    }

    /// Batched read must drop each completed frame into *its own*
    /// buffer — submitting out of offset order checks the
    /// `user_data → buffer` mapping rather than relying on CQEs
    /// arriving in submission order.
    #[test]
    fn pread_many_at_reads_each_frame_into_its_own_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();

        let mut ring = UringContext::new(&file, None).unwrap();

        let a = frame(0xAA);
        let b = frame(0xBB);
        let c = frame(0xCC);
        let frame_len = a.len() as u64;
        ring.pwrite_many_at(&[(0, &a), (frame_len, &b), (2 * frame_len, &c)])
            .unwrap();
        ring.sync_data().unwrap();

        let mut r0 = AlignedBlobBuf::zeroed();
        let mut r1 = AlignedBlobBuf::zeroed();
        let mut r2 = AlignedBlobBuf::zeroed();
        ring.pread_many_at(&mut [(2 * frame_len, &mut r2), (0, &mut r0), (frame_len, &mut r1)])
            .unwrap();

        assert!(r0.as_slice().iter().all(|&x| x == 0xAA));
        assert!(r1.as_slice().iter().all(|&x| x == 0xBB));
        assert!(r2.as_slice().iter().all(|&x| x == 0xCC));
    }

    /// A batch wider than one ring exercises the chunk loop in
    /// [`UringContext::pread_many_at`].
    #[test]
    fn pread_many_at_spans_multiple_ring_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();

        let mut ring = UringContext::new(&file, None).unwrap();

        let count = RING_DEPTH_USIZE + 5;
        let frame_len = AlignedBlobBuf::zeroed().len() as u64;
        let writes: Vec<AlignedBlobBuf> = (0..count).map(|i| frame(i as u8)).collect();
        let write_refs: Vec<(u64, &AlignedBlobBuf)> = writes
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64 * frame_len, b))
            .collect();
        ring.pwrite_many_at(&write_refs).unwrap();
        ring.sync_data().unwrap();

        let mut reads: Vec<AlignedBlobBuf> = (0..count).map(|_| AlignedBlobBuf::zeroed()).collect();
        let mut read_refs: Vec<(u64, &mut AlignedBlobBuf)> = reads
            .iter_mut()
            .enumerate()
            .map(|(i, b)| (i as u64 * frame_len, b))
            .collect();
        ring.pread_many_at(&mut read_refs).unwrap();
        drop(read_refs);

        for (i, buf) in reads.iter().enumerate() {
            assert!(
                buf.as_slice().iter().all(|&x| x == i as u8),
                "frame {i} mismatched after multi-chunk batched read"
            );
        }
    }

    #[test]
    fn pread_many_at_empty_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let mut ring = UringContext::new(&file, None).unwrap();
        ring.pread_many_at(&mut []).unwrap();
    }
}
