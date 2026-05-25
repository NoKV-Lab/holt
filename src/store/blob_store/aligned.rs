//! `AlignedBlobBuf` — heap-allocated, 4 KB-aligned 512 KB buffer.
//!
//! All blob I/O in holt flows through this type so that:
//!
//! 1. Buffers can be handed directly to `O_DIRECT` files without a
//!    bounce copy — the kernel rejects unaligned submissions.
//! 2. Buffers can be registered with `io_uring`'s
//!    `register_buffers` for SQE-fast-path submission.
//! 3. `MemoryBlobStore` keeps an identical layout, so swapping
//!    stores never changes the on-the-wire shape of a blob.

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
#[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
use std::sync::Arc;

use crate::layout::PAGE_SIZE;

#[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
use super::buffer_pool::{BlobBufPool, BlobBufPoolInner};

/// Buffer alignment in bytes. Matches the smallest NVMe physical
/// block, satisfies `O_DIRECT`'s alignment requirement on Linux,
/// and is a multiple of the page size on every supported arch.
pub const BUF_ALIGN: usize = 4096;

/// Fixed-buffer index exposed to `io_uring`'s `*_FIXED`
/// opcodes. The kernel ABI stores this index as `u16`, so the
/// allocator refuses larger pools.
pub(crate) type FixedBufferIndex = u16;

/// A heap-allocated, 4 KB-aligned, `PAGE_SIZE`-byte buffer.
///
/// One per logical blob in flight. Cheap to construct (single
/// `alloc`), cheap to clone (single `memcpy`). `Send + Sync` — the
/// raw pointer is the sole owner of its allocation.
pub struct AlignedBlobBuf {
    ptr: NonNull<u8>,
    owner: BlobBufOwner,
}

enum BlobBufOwner {
    Heap,
    #[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
    Pool {
        pool: Arc<BlobBufPoolInner>,
        index: FixedBufferIndex,
    },
}

impl AlignedBlobBuf {
    /// Allocate a zero-filled buffer.
    #[must_use]
    pub fn zeroed() -> Self {
        let layout = Self::layout();
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self {
            ptr,
            owner: BlobBufOwner::Heap,
        }
    }

    /// Allocate an uninitialized buffer.
    ///
    /// # Safety
    ///
    /// The returned buffer's bytes are uninitialized. The caller
    /// must initialize all `PAGE_SIZE` bytes before any operation
    /// reads the buffer contents, including [`Self::as_slice`],
    /// [`Clone::clone`], or `BlobStore::write_blob`.
    #[must_use]
    pub(crate) unsafe fn uninit() -> Self {
        let layout = Self::layout();
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self {
            ptr,
            owner: BlobBufOwner::Heap,
        }
    }

    /// Allocate an uninitialized frame from `pool`.
    ///
    /// Returns `None` when every fixed slot is currently leased;
    /// callers should fall back to [`Self::uninit`] in that case.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::uninit`]: initialize every byte
    /// before any operation reads the buffer contents.
    #[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
    #[must_use]
    pub(crate) unsafe fn pooled_uninit(pool: &BlobBufPool) -> Option<Self> {
        let index = pool.inner.alloc_slot()?;
        let ptr = pool.inner.ptr_for_index(index);
        Some(Self {
            ptr,
            owner: BlobBufOwner::Pool {
                pool: Arc::clone(&pool.inner),
                index,
            },
        })
    }

    /// Allocate a zero-filled frame from `pool`, falling back to
    /// `None` when the pool is exhausted.
    #[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
    #[must_use]
    pub(crate) fn pooled_zeroed(pool: &BlobBufPool) -> Option<Self> {
        // SAFETY: `fill(0)` initializes the full PAGE_SIZE buffer
        // before the value escapes this function.
        let mut out = unsafe { Self::pooled_uninit(pool)? };
        out.as_mut_slice().fill(0);
        Some(out)
    }

    /// `io_uring` fixed-buffer slot index when this buffer comes
    /// from a registered pool.
    #[must_use]
    pub(crate) fn fixed_buffer_index(&self) -> Option<FixedBufferIndex> {
        match &self.owner {
            BlobBufOwner::Heap => None,
            #[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
            BlobBufOwner::Pool { index, .. } => Some(*index),
        }
    }

    /// Allocate a zero-filled buffer from the same pool when this
    /// buffer is pooled; otherwise allocate a normal heap buffer.
    #[must_use]
    pub(crate) fn zeroed_like(&self) -> Self {
        match &self.owner {
            BlobBufOwner::Heap => Self::zeroed(),
            #[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
            BlobBufOwner::Pool { pool, .. } => {
                let wrapper = BlobBufPool {
                    inner: Arc::clone(pool),
                };
                Self::pooled_zeroed(&wrapper).unwrap_or_else(Self::zeroed)
            }
        }
    }

    /// Raw pointer for FFI / io_uring `iovec`. Always non-null,
    /// always 4 KB-aligned, always `PAGE_SIZE` bytes valid.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Mutable raw pointer. Same invariants as [`Self::as_ptr`].
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Read-only view as a slice of `PAGE_SIZE` bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), PAGE_SIZE as usize) }
    }

    /// Mutable view as a slice of `PAGE_SIZE` bytes.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), PAGE_SIZE as usize) }
    }

    /// Length in bytes — always `PAGE_SIZE` (512 KB).
    #[must_use]
    pub const fn len(&self) -> usize {
        PAGE_SIZE as usize
    }

    /// `true` if `len() == 0` — always false; here for clippy.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    fn layout() -> Layout {
        Layout::from_size_align(PAGE_SIZE as usize, BUF_ALIGN)
            .expect("PAGE_SIZE/BUF_ALIGN both > 0 and BUF_ALIGN is a power of two")
    }
}

impl Drop for AlignedBlobBuf {
    fn drop(&mut self) {
        match &self.owner {
            BlobBufOwner::Heap => unsafe { dealloc(self.ptr.as_ptr(), Self::layout()) },
            #[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
            BlobBufOwner::Pool { pool, index } => pool.free_slot(*index),
        }
    }
}

impl Default for AlignedBlobBuf {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Clone for AlignedBlobBuf {
    fn clone(&self) -> Self {
        let mut out = match &self.owner {
            // SAFETY: copy_from_slice below initializes the full
            // PAGE_SIZE buffer before `out` is returned.
            BlobBufOwner::Heap => unsafe { Self::uninit() },
            #[cfg(any(test, all(target_os = "linux", feature = "io-uring")))]
            BlobBufOwner::Pool { pool, .. } => {
                let wrapper = BlobBufPool {
                    inner: Arc::clone(pool),
                };
                // SAFETY: copy_from_slice below initializes the full
                // PAGE_SIZE buffer before `out` is returned.
                if let Some(buf) = unsafe { Self::pooled_uninit(&wrapper) } {
                    buf
                } else {
                    // SAFETY: copy_from_slice below initializes the
                    // full PAGE_SIZE buffer before `out` is returned.
                    unsafe { Self::uninit() }
                }
            }
        };
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl std::fmt::Debug for AlignedBlobBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AlignedBlobBuf({:p}, {} B, fixed={:?})",
            self.ptr.as_ptr(),
            PAGE_SIZE,
            self.fixed_buffer_index(),
        )
    }
}

// SAFETY: AlignedBlobBuf owns its allocation exclusively (no
// aliasing) and exposes Rust's normal &/&mut borrow rules through
// as_slice / as_mut_slice. Sending the owning struct across threads
// is therefore sound.
unsafe impl Send for AlignedBlobBuf {}
unsafe impl Sync for AlignedBlobBuf {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_is_zeroed() {
        let b = AlignedBlobBuf::zeroed();
        assert_eq!(b.len(), PAGE_SIZE as usize);
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }

    #[test]
    fn pointer_is_4k_aligned() {
        for _ in 0..16 {
            let b = AlignedBlobBuf::zeroed();
            assert_eq!(b.as_ptr() as usize % BUF_ALIGN, 0);
        }
    }

    #[test]
    fn clone_is_independent_memcpy() {
        let mut a = AlignedBlobBuf::zeroed();
        a.as_mut_slice()[100] = 0xAB;
        let mut b = a.clone();
        assert_eq!(b.as_slice()[100], 0xAB);
        b.as_mut_slice()[100] = 0xCD;
        assert_eq!(a.as_slice()[100], 0xAB, "clone must not alias source");
        assert_eq!(b.as_slice()[100], 0xCD);
    }

    #[test]
    fn pooled_buffer_returns_fixed_index() {
        let pool = BlobBufPool::new(2).unwrap();
        let a = AlignedBlobBuf::pooled_zeroed(&pool).unwrap();
        let b = AlignedBlobBuf::pooled_zeroed(&pool).unwrap();
        assert_eq!(a.len(), PAGE_SIZE as usize);
        assert!(a.fixed_buffer_index().is_some());
        assert!(b.fixed_buffer_index().is_some());
        assert_ne!(a.fixed_buffer_index(), b.fixed_buffer_index());
        // SAFETY: no buffer is returned in this exhausted-pool case.
        assert!(unsafe { AlignedBlobBuf::pooled_uninit(&pool) }.is_none());
        drop(a);
        // SAFETY: the test does not read from the returned buffer.
        assert!(unsafe { AlignedBlobBuf::pooled_uninit(&pool) }.is_some());
    }

    #[test]
    fn pooled_buffer_free_list_survives_concurrent_churn() {
        let pool = BlobBufPool::new(8).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let pool = pool.clone();
                scope.spawn(move || {
                    for _ in 0..1000 {
                        let mut b = loop {
                            // SAFETY: the loop writes before dropping,
                            // and never reads the buffer contents.
                            if let Some(b) = unsafe { AlignedBlobBuf::pooled_uninit(&pool) } {
                                break b;
                            }
                            std::hint::spin_loop();
                        };
                        b.as_mut_slice()[0] = 0x7B;
                    }
                });
            }
        });
        let leased: Vec<_> = (0..8)
            .map(|_| {
                // SAFETY: leased buffers are used only to exhaust
                // the pool and are never read.
                unsafe { AlignedBlobBuf::pooled_uninit(&pool) }.unwrap()
            })
            .collect();
        // SAFETY: no buffer is returned in this exhausted-pool case.
        assert!(unsafe { AlignedBlobBuf::pooled_uninit(&pool) }.is_none());
        drop(leased);
        // SAFETY: the test does not read from the returned buffer.
        assert!(unsafe { AlignedBlobBuf::pooled_uninit(&pool) }.is_some());
    }

    #[test]
    fn uninit_is_writable() {
        // SAFETY: the test initializes every byte with fill() before
        // reading through as_slice().
        let mut b = unsafe { AlignedBlobBuf::uninit() };
        b.as_mut_slice().fill(0x42);
        assert!(b.as_slice().iter().all(|&x| x == 0x42));
    }

    #[test]
    fn default_equals_zeroed() {
        let b = AlignedBlobBuf::default();
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }
}
