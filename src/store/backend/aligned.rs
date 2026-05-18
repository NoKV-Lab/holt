//! `AlignedBlobBuf` — heap-allocated, 4 KB-aligned 512 KB buffer.
//!
//! All blob I/O in artisan flows through this type so that:
//!
//! 1. Buffers can be handed directly to `O_DIRECT` files without a
//!    bounce copy — the kernel rejects unaligned submissions.
//! 2. Buffers can be registered with `io_uring`'s
//!    `register_buffers` for SQE-fast-path submission.
//! 3. `MemoryBackend` keeps an identical layout, so swapping
//!    backends never changes the on-the-wire shape of a blob.

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

use crate::layout::PAGE_SIZE;

/// Buffer alignment in bytes. Matches the smallest NVMe physical
/// block, satisfies `O_DIRECT`'s alignment requirement on Linux,
/// and is a multiple of the page size on every supported arch.
pub const BUF_ALIGN: usize = 4096;

/// A heap-allocated, 4 KB-aligned, `PAGE_SIZE`-byte buffer.
///
/// One per logical blob in flight. Cheap to construct (single
/// `alloc`), cheap to clone (single `memcpy`). `Send + Sync` — the
/// raw pointer is the sole owner of its allocation.
pub struct AlignedBlobBuf {
    ptr: NonNull<u8>,
}

impl AlignedBlobBuf {
    /// Allocate a zero-filled buffer.
    #[must_use]
    pub fn zeroed() -> Self {
        let layout = Self::layout();
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr }
    }

    /// Allocate an uninitialized buffer. Caller must fill before
    /// reading (typical use: io_uring read fills it from disk).
    #[must_use]
    pub fn uninit() -> Self {
        let layout = Self::layout();
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr }
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

    /// Length in bytes — always [`PAGE_SIZE`].
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
        unsafe { dealloc(self.ptr.as_ptr(), Self::layout()) };
    }
}

impl Default for AlignedBlobBuf {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Clone for AlignedBlobBuf {
    fn clone(&self) -> Self {
        let mut out = Self::uninit();
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl std::fmt::Debug for AlignedBlobBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AlignedBlobBuf({:p}, {} B)",
            self.ptr.as_ptr(),
            PAGE_SIZE
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
    fn uninit_is_writable() {
        let mut b = AlignedBlobBuf::uninit();
        b.as_mut_slice().fill(0x42);
        assert!(b.as_slice().iter().all(|&x| x == 0x42));
    }

    #[test]
    fn default_equals_zeroed() {
        let b = AlignedBlobBuf::default();
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }
}
