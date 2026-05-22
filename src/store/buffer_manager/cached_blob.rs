use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::concurrency::{Guard as LatchGuard, HybridLatch};
use crate::store::BlobFrame;

use super::AlignedBlobBuf;

const CLEAN_DIRTY_SEQ: u64 = 0;

/// A single cached blob. Callers obtain one via
/// [`BufferManager::pin`](super::BufferManager::pin) and then take
/// an optimistic / shared / exclusive guard on it to access the
/// underlying 512 KB buffer with zero copies.
///
/// Holding the `Arc<CachedBlob>` prevents the entry from being
/// evicted, so traversals that pin a blob can borrow into it for
/// as long as the pin is alive.
pub struct CachedBlob {
    latch: HybridLatch,
    buf: UnsafeCell<AlignedBlobBuf>,
    /// Fast-path low-watermark for dirty tracking. `0` means no
    /// live dirty-map entry is known for this cached blob; any
    /// non-zero value is the lowest unflushed seq observed by
    /// `mark_dirty`.
    ///
    /// The authoritative enumeration source remains
    /// `MutationState::dirty`. This hint lets repeated writes to an
    /// already-dirty cached blob skip the shard mutex when the
    /// existing low-watermark already covers the new seq.
    dirty_seq_hint: AtomicU64,
    /// Stamp set by `BufferManager` on every `pin` / `get_cached`.
    /// Read by the eviction thread to decide if this entry is
    /// cold enough to drop. Relaxed reads/writes - see
    /// [`BufferManager::clock`](super::BufferManager::clock_tick).
    pub(super) last_touched: AtomicU64,
}

// SAFETY: every access to `buf` is gated by `latch`, which provides
// the standard reader-writer exclusion (plus an optimistic mode
// whose reads are revalidated by the caller before being trusted).
// The `UnsafeCell` only marks the interior-mutability; the actual
// concurrency contract is enforced by `HybridLatch`.
unsafe impl Sync for CachedBlob {}

impl CachedBlob {
    pub(super) fn new(buf: AlignedBlobBuf) -> Self {
        Self {
            latch: HybridLatch::new(),
            buf: UnsafeCell::new(buf),
            dirty_seq_hint: AtomicU64::new(CLEAN_DIRTY_SEQ),
            last_touched: AtomicU64::new(0),
        }
    }

    /// Try to cover `seq` with this blob's dirty hint.
    ///
    /// Returns `true` when the caller must still publish/merge the
    /// guid into `MutationState::dirty`; returns `false` when the
    /// existing hint already has a lower-or-equal unflushed seq and
    /// therefore the dirty map entry is already sufficient.
    pub(super) fn dirty_hint_needs_map_publish(&self, seq: u64) -> bool {
        let mut cur = self.dirty_seq_hint.load(Ordering::Acquire);
        loop {
            if cur != CLEAN_DIRTY_SEQ && cur <= seq {
                return false;
            }
            let next = if cur == CLEAN_DIRTY_SEQ {
                seq
            } else {
                cur.min(seq)
            };
            match self.dirty_seq_hint.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => cur = actual,
            }
        }
    }

    pub(super) fn take_dirty_hint(&self) -> Option<u64> {
        match self.dirty_seq_hint.swap(CLEAN_DIRTY_SEQ, Ordering::AcqRel) {
            CLEAN_DIRTY_SEQ => None,
            seq => Some(seq),
        }
    }

    pub(super) fn clear_dirty_hint(&self) {
        self.dirty_seq_hint
            .store(CLEAN_DIRTY_SEQ, Ordering::Release);
    }

    /// Logical tick at which this entry was last looked up. Used
    /// by the eviction thread to classify the entry as cold.
    #[must_use]
    pub(crate) fn last_touched(&self) -> u64 {
        self.last_touched.load(Ordering::Relaxed)
    }

    /// Wait-free read snapshot. No real lock taken - the caller
    /// reads bytes through [`OptimisticGuard::as_slice`] and then
    /// calls [`OptimisticGuard::validate`] to confirm no writer
    /// lapped the snapshot. If validation fails the caller must
    /// discard everything read and restart.
    pub fn read_optimistic(&self) -> OptimisticGuard<'_> {
        OptimisticGuard {
            latch: LatchGuard::optimistic(&self.latch),
            buf: &self.buf,
        }
    }

    /// Shared read access - blocks while a writer holds the latch
    /// exclusively, but N shared readers run concurrently.
    pub fn read(&self) -> BlobReadGuard<'_> {
        BlobReadGuard {
            _latch: LatchGuard::shared(&self.latch),
            buf: &self.buf,
        }
    }

    /// Current blob content version. For route validation, read it
    /// while holding a shared guard on the same blob so the version
    /// and parent edge are stable until the child is pinned.
    #[must_use]
    pub(crate) fn content_version(&self) -> u64 {
        self.latch.current_version()
    }

    /// Validate a previously observed content version without
    /// taking a shared latch. Returns `false` while an exclusive
    /// writer is active or after any exclusive writer has released.
    #[must_use]
    pub(crate) fn validate_content_version(&self, version: u64) -> bool {
        self.latch.validate(version)
    }

    /// Exclusive write access - blocks until idle, then runs
    /// alone. Bumps the version on release so concurrent
    /// optimistic readers detect the change and restart.
    pub fn write(&self) -> BlobWriteGuard<'_> {
        BlobWriteGuard {
            _latch: LatchGuard::exclusive(&self.latch),
            buf: &self.buf,
        }
    }
}

/// Wait-free guard returned by [`CachedBlob::read_optimistic`].
///
/// Reads from `as_slice()` may be torn (a concurrent writer could
/// be mid-mutation). The caller must finish reading and call
/// [`OptimisticGuard::validate`]; if `validate` returns `false`,
/// every byte read through this guard is potentially stale and must
/// be discarded.
pub struct OptimisticGuard<'a> {
    latch: LatchGuard<'a>,
    buf: &'a UnsafeCell<AlignedBlobBuf>,
}

impl<'a> OptimisticGuard<'a> {
    /// Pointer-style view of the 512 KB buffer. Bytes may be torn;
    /// see the type-level docs.
    #[must_use]
    pub fn as_slice(&self) -> &'a [u8] {
        // SAFETY: the optimistic guard holds the latch in
        // `Optimistic` mode (no real lock); reads through this
        // borrow may race with a writer. The walker treats any
        // result derived from such a borrow as untrusted until
        // `validate()` confirms it; corrupt bodies surface as
        // `Error::NodeCorrupt` rather than panics because the
        // layout decoders bounds-check every field.
        unsafe { (&*self.buf.get()).as_slice() }
    }

    /// Returns `true` if no exclusive writer modified the buffer
    /// between the snapshot and now.
    #[must_use]
    pub fn validate(&self) -> bool {
        self.latch.validate()
    }
}

/// Shared-mode read guard returned by [`CachedBlob::read`].
///
/// Derefs to `&AlignedBlobBuf`; call `.as_slice()` for byte-level
/// access.
pub struct BlobReadGuard<'a> {
    _latch: LatchGuard<'a>,
    buf: &'a UnsafeCell<AlignedBlobBuf>,
}

impl Deref for BlobReadGuard<'_> {
    type Target = AlignedBlobBuf;
    fn deref(&self) -> &AlignedBlobBuf {
        // SAFETY: shared-mode latch excludes writers.
        unsafe { &*self.buf.get() }
    }
}

/// Exclusive-mode write guard returned by [`CachedBlob::write`].
///
/// Derefs to `&mut AlignedBlobBuf`; call `.as_mut_slice()` for
/// byte-level access. For walker paths that mutate the typed
/// [`BlobFrame`] view, prefer [`Self::frame`].
pub struct BlobWriteGuard<'a> {
    _latch: LatchGuard<'a>,
    buf: &'a UnsafeCell<AlignedBlobBuf>,
}

impl BlobWriteGuard<'_> {
    /// Construct a [`BlobFrame`] view over this guard's buffer.
    pub fn frame(&mut self) -> BlobFrame<'_> {
        BlobFrame::wrap(self.as_mut_slice())
    }
}

impl Deref for BlobWriteGuard<'_> {
    type Target = AlignedBlobBuf;
    fn deref(&self) -> &AlignedBlobBuf {
        // SAFETY: exclusive-mode latch excludes all other access.
        unsafe { &*self.buf.get() }
    }
}

impl DerefMut for BlobWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut AlignedBlobBuf {
        // SAFETY: exclusive-mode latch excludes all other access,
        // and `&mut self` ensures no other borrow of this guard
        // exists.
        unsafe { &mut *self.buf.get() }
    }
}
