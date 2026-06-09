//! SIMD hot paths used by the walker.
//!
//! Five operations dominate the ART walker's byte-scan cost:
//!
//! 1. **Node16 byte search** — find the index `i` in `keys[0..count]`
//!    such that `keys[i] == byte`. The scalar form is a 16-iteration
//!    loop; SSE2 / NEON do the same work in ~3 instructions.
//!    [`node16_find_byte`].
//! 2. **Longest common prefix** — find the first divergence between
//!    two byte slices (Leaf split, Prefix split). 16 bytes per
//!    iteration via vector compare. [`longest_common_prefix`].
//! 3. **Node48 index scan** — find the next non-zero byte in
//!    `index[256]` starting from `start`. Used by the range
//!    iterator to advance through Node48 children in lex order.
//!    [`find_next_nonzero_byte`].
//! 4. **Inner-node children scan** — find the next non-zero `u16`
//!    slot index in a `Node48` / `Node256` `children[]` array
//!    starting from `start`. Compares 8 (SSE2 / NEON) or 16 (AVX2)
//!    `u16` lanes per instruction. [`find_next_nonzero_u16`].
//! 5. **Delimiter byte scan** — find `/` or another delimiter in a
//!    leaf suffix during S3/POSIX rollup iteration.
//!    [`find_byte`].
//!
//! Dispatch is architecture-local: x86_64 uses SSE2 as the base
//! path and upgrades long scans to AVX2 when available; aarch64
//! uses NEON; other targets use scalar code. Behaviour is identical
//! across paths; the scalar form is the spec.

// ---------------------------------------------------------------
// Public API
// ---------------------------------------------------------------

/// Find the index `i` in `keys[0..count]` such that `keys[i] ==
/// byte`. Returns `None` if no such index exists. `count` is
/// clamped to 16.
#[inline]
pub fn node16_find_byte(keys: &[u8; 16], count: u8, byte: u8) -> Option<u8> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86::find_byte_in_16(keys.as_ptr(), count, byte)
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        arm::find_byte_in_16(keys.as_ptr(), count, byte)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        node16_find_byte_scalar(keys, count, byte)
    }
}

/// Reference scalar implementation — exposed only inside the crate
/// so the SSE2 / NEON paths can be cross-checked in tests and the
/// `cfg(not(...))` fallback path can call into it directly.
#[cfg(any(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
#[inline]
pub(crate) fn node16_find_byte_scalar(keys: &[u8; 16], count: u8, byte: u8) -> Option<u8> {
    let n = (count as usize).min(16);
    let mut i = 0;
    while i < n {
        if keys[i] == byte {
            return Some(i as u8);
        }
        i += 1;
    }
    None
}

/// Length of the longest common prefix between `a` and `b`. Equal
/// to `a.len().min(b.len())` when one is a prefix of the other.
#[inline]
pub fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
    let limit = a.len().min(b.len());
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    if limit >= 32 && x86::avx2_available() {
        while i + 32 <= limit {
            let mask = unsafe { x86::cmp_32_bytes_bitmask(a[i..].as_ptr(), b[i..].as_ptr()) };
            if mask != u32::MAX {
                return i + mask.trailing_ones() as usize;
            }
            i += 32;
        }
    }

    #[cfg(target_arch = "x86_64")]
    while i + 16 <= limit {
        let mask = unsafe { x86::cmp_16_bytes_bitmask(a[i..].as_ptr(), b[i..].as_ptr()) };
        if mask != 0xFFFF {
            // bit j = 1 iff a[i+j] == b[i+j]; first 0 bit = first
            // divergence. trailing_ones counts the leading 1s.
            return i + mask.trailing_ones() as usize;
        }
        i += 16;
    }

    #[cfg(target_arch = "aarch64")]
    while i + 16 <= limit {
        let mask = unsafe { arm::cmp_16_bytes_nibble(a[i..].as_ptr(), b[i..].as_ptr()) };
        if mask != u64::MAX {
            // Each input byte → 4 bits in mask (0xF = match,
            // 0x0 = mismatch). trailing_ones / 4 = first divergence
            // index inside this 16-byte chunk.
            return i + (mask.trailing_ones() / 4) as usize;
        }
        i += 16;
    }

    longest_common_prefix_tail(a, b, i, limit)
}

/// Find the smallest index `i ∈ [start, bytes.len())` with
/// `bytes[i] == needle`, or `None` if the byte is absent.
///
/// This is the generic slice version of the Node16 byte search. It
/// exists for delimiter-heavy metadata scans (`list_dir`,
/// S3-style `LIST prefix + delimiter`) where every yielded leaf used
/// to pay a scalar `position()` walk over the path suffix.
#[inline]
pub fn find_byte(bytes: &[u8], needle: u8, start: usize) -> Option<usize> {
    let len = bytes.len();
    if start >= len {
        return None;
    }
    let mut i = start;

    #[cfg(target_arch = "x86_64")]
    if len - i >= 32 && x86::avx2_available() {
        while i + 32 <= len {
            let ptr = unsafe { bytes.as_ptr().add(i) };
            let mask = unsafe { x86::cmp_byte_eq_mask_32(ptr, needle) };
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 32;
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    while i + 16 <= len {
        let ptr = unsafe { bytes.as_ptr().add(i) };
        let mask = unsafe { byte_eq_mask_16(ptr, needle) };
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 16;
    }

    while i < len {
        if bytes[i] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Find the smallest index `i ∈ [start, bytes.len())` with
/// `bytes[i] != 0`, or `None` if every byte from `start` onwards
/// is zero.
///
/// Drives the range iterator's Node48 lex-order walk (`index[256]`
/// where the entry at byte `b` is `0` if no child has key byte
/// `b`). The scalar loop is at most 256 iterations; SSE2 / NEON
/// scan 16 bytes per instruction.
#[inline]
pub fn find_next_nonzero_byte(bytes: &[u8], start: usize) -> Option<usize> {
    let len = bytes.len();
    if start >= len {
        return None;
    }
    let mut i = start;

    #[cfg(target_arch = "x86_64")]
    if len - i >= 32 && x86::avx2_available() {
        while i + 32 <= len {
            let ptr = unsafe { bytes.as_ptr().add(i) };
            let mask = unsafe { x86::cmp_byte_neq_zero_mask_32(ptr) };
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 32;
        }
    }

    // Align the SIMD window to whole 16-byte chunks. The tail (and
    // an unaligned `start`) falls through to the scalar loop below.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    while i + 16 <= len {
        let ptr = unsafe { bytes.as_ptr().add(i) };
        let mask = unsafe { nonzero_byte_mask_16(ptr) };
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 16;
    }

    while i < len {
        if bytes[i] != 0 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Hint the CPU that the cache line at `ptr` is likely to be read
/// soon. This is intentionally best-effort and a no-op on targets
/// where we do not have a stable cheap prefetch intrinsic.
#[inline]
#[cfg(target_arch = "x86_64")]
pub(crate) fn prefetch_read_data(ptr: *const u8) {
    unsafe {
        x86::prefetch_read_data(ptr);
    }
}

#[inline]
fn longest_common_prefix_tail(a: &[u8], b: &[u8], mut i: usize, limit: usize) -> usize {
    #[cfg(target_endian = "little")]
    {
        while i + 8 <= limit {
            let x = unsafe { read_u64_unaligned(a.as_ptr().add(i)) }
                ^ unsafe { read_u64_unaligned(b.as_ptr().add(i)) };
            if x != 0 {
                return i + (x.trailing_zeros() as usize / 8);
            }
            i += 8;
        }
    }

    while i < limit && a[i] == b[i] {
        i += 1;
    }
    i
}

#[cfg(target_endian = "little")]
#[inline]
unsafe fn read_u64_unaligned(ptr: *const u8) -> u64 {
    unsafe { std::ptr::read_unaligned(ptr.cast::<u64>()) }
}

/// Find the smallest index `i ∈ [start, words.len())` with
/// `words[i] != 0`, or `None` if every `u16` from `start` onwards
/// is zero.
///
/// Drives the inner-node lex-order walks: `Node48` / `Node256`
/// store `u16` child slot indices (entry `0` means "no child at
/// this key byte"). x86 upgrades to a 16-lane AVX2 pass when
/// available, then both x86 (SSE2) and aarch64 (NEON) compare 8
/// `u16` lanes per instruction; the lane helpers return a mask
/// with set bits at *even* positions (bit `2·lane`), so
/// `trailing_zeros() / 2` recovers the first non-zero lane. A
/// scalar tail handles the `< 8` remainder. All SIMD loads are
/// unaligned, so the function is sound on both 8-byte-aligned blob
/// bodies and 2-byte-aligned stack copies of node structs.
#[inline]
pub fn find_next_nonzero_u16(words: &[u16], start: usize) -> Option<usize> {
    let len = words.len();
    if start >= len {
        return None;
    }
    let mut i = start;

    #[cfg(target_arch = "x86_64")]
    if len - i >= 16 && x86::avx2_available() {
        while i + 16 <= len {
            let mask = unsafe { x86::cmp_u16_neq_zero_mask_16(words.as_ptr().add(i)) };
            if mask != 0 {
                return Some(i + (mask.trailing_zeros() as usize) / 2);
            }
            i += 16;
        }
    }

    // NEON is base aarch64, so the 16-lane (2×8) path is always
    // available — no feature gate. Handles the bulk of the 16 / 48 /
    // 256-element child arrays; the 8-lane loop below mops up the
    // 8..16 remainder.
    #[cfg(target_arch = "aarch64")]
    while i + 16 <= len {
        let mask = unsafe { nonzero_u16_mask_16(words.as_ptr().add(i)) };
        if mask != 0 {
            return Some(i + (mask.trailing_zeros() as usize) / 2);
        }
        i += 16;
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    while i + 8 <= len {
        let mask = unsafe { nonzero_u16_mask_8(words.as_ptr().add(i)) };
        if mask != 0 {
            return Some(i + (mask.trailing_zeros() as usize) / 2);
        }
        i += 8;
    }

    while i < len {
        if words[i] != 0 {
            return Some(i);
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------
// arch-specific glue for byte/u32 scans
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn byte_eq_mask_16(ptr: *const u8, needle: u8) -> u32 {
    unsafe { x86::cmp_byte_eq_mask_16(ptr, needle) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn nonzero_byte_mask_16(ptr: *const u8) -> u32 {
    // Compare 16 bytes against zero, then *invert* the mask so 1
    // means "non-zero".
    unsafe { x86::cmp_byte_neq_zero_mask_16(ptr) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn nonzero_u16_mask_8(ptr: *const u16) -> u32 {
    unsafe { x86::cmp_u16_neq_zero_mask_8(ptr) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn byte_eq_mask_16(ptr: *const u8, needle: u8) -> u32 {
    unsafe { arm::cmp_byte_eq_mask_16(ptr, needle) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn nonzero_byte_mask_16(ptr: *const u8) -> u32 {
    unsafe { arm::cmp_byte_neq_zero_mask_16(ptr) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn nonzero_u16_mask_8(ptr: *const u16) -> u32 {
    unsafe { arm::cmp_u16_neq_zero_mask_8(ptr) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn nonzero_u16_mask_16(ptr: *const u16) -> u32 {
    unsafe { arm::cmp_u16_neq_zero_mask_16(ptr) }
}

// ---------------------------------------------------------------
// x86_64 — SSE2 (always available in the base x86_64 ISA)
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_cmpeq_epi16, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_set1_epi8, _mm256_setzero_si256, _mm_cmpeq_epi16,
        _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_prefetch, _mm_set1_epi8,
        _mm_setzero_si128, _MM_HINT_T0,
    };

    #[inline]
    pub(super) fn avx2_available() -> bool {
        cfg!(target_feature = "avx2") || std::arch::is_x86_feature_detected!("avx2")
    }

    /// Compare 16 bytes from `a` against 16 from `b`. Returns a
    /// 16-bit bitmask: bit `i` = 1 iff `a[i] == b[i]`. Caller
    /// guarantees both pointers are at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn cmp_16_bytes_bitmask(a: *const u8, b: *const u8) -> u32 {
        let va = unsafe { _mm_loadu_si128(a.cast::<__m128i>()) };
        let vb = unsafe { _mm_loadu_si128(b.cast::<__m128i>()) };
        let cmp = _mm_cmpeq_epi8(va, vb);
        _mm_movemask_epi8(cmp) as u32
    }

    /// Compare 32 bytes from `a` against 32 from `b`. Returns a
    /// 32-bit bitmask: bit `i` = 1 iff `a[i] == b[i]`.
    /// Caller guarantees AVX2 support and both pointers are at
    /// least 32 bytes valid.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn cmp_32_bytes_bitmask(a: *const u8, b: *const u8) -> u32 {
        let va = unsafe { _mm256_loadu_si256(a.cast::<__m256i>()) };
        let vb = unsafe { _mm256_loadu_si256(b.cast::<__m256i>()) };
        let cmp = _mm256_cmpeq_epi8(va, vb);
        _mm256_movemask_epi8(cmp) as u32
    }

    /// 16-bit mask where bit `i = 1` iff byte `i` equals `needle`.
    /// Caller guarantees `ptr` is at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn cmp_byte_eq_mask_16(ptr: *const u8, needle: u8) -> u32 {
        let vec = unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) };
        let needle = _mm_set1_epi8(needle as i8);
        let cmp = _mm_cmpeq_epi8(vec, needle);
        _mm_movemask_epi8(cmp) as u32
    }

    /// 32-bit mask where bit `i = 1` iff byte `i` equals `needle`.
    /// Caller guarantees AVX2 support and `ptr` is at least 32
    /// bytes valid.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn cmp_byte_eq_mask_32(ptr: *const u8, needle: u8) -> u32 {
        let vec = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
        let needle = _mm256_set1_epi8(needle as i8);
        let cmp = _mm256_cmpeq_epi8(vec, needle);
        _mm256_movemask_epi8(cmp) as u32
    }

    /// Find `byte` in 16 keys; return the first matching index or
    /// `None`. Caller guarantees `keys` is at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn find_byte_in_16(keys: *const u8, count: u8, byte: u8) -> Option<u8> {
        let vec = unsafe { _mm_loadu_si128(keys.cast::<__m128i>()) };
        let needle = _mm_set1_epi8(byte as i8);
        let cmp = _mm_cmpeq_epi8(vec, needle);
        let mask = _mm_movemask_epi8(cmp) as u32;
        // Mask off any matches past `count` (unused slots may hold
        // arbitrary bytes — Node16::empty seeds 0, but defensive).
        let count_mask = if count >= 16 {
            0xFFFF
        } else {
            (1u32 << count) - 1
        };
        let masked = mask & count_mask;
        std::num::NonZeroU32::new(masked).map(|m| m.get().trailing_zeros() as u8)
    }

    /// 16-bit mask where bit `i = 1` iff byte `i` of the 16-byte
    /// window at `ptr` is **non-zero**. Caller guarantees `ptr` is
    /// at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn cmp_byte_neq_zero_mask_16(ptr: *const u8) -> u32 {
        let vec = unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) };
        let zero = _mm_setzero_si128();
        let cmp_eq_zero = _mm_cmpeq_epi8(vec, zero); // 0xFF where zero
                                                     // Invert via movemask + bitwise NOT trimmed to the low 16
                                                     // bits — avoids a `pxor` and lets the trailing_zeros caller
                                                     // see "non-zero positions" directly.
        let zero_mask = _mm_movemask_epi8(cmp_eq_zero) as u32;
        (!zero_mask) & 0xFFFF
    }

    /// 32-bit mask where bit `i = 1` iff byte `i` of the 32-byte
    /// window is **non-zero**. Caller guarantees AVX2 support and
    /// `ptr` is at least 32 bytes valid.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn cmp_byte_neq_zero_mask_32(ptr: *const u8) -> u32 {
        let vec = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
        let zero = _mm256_setzero_si256();
        let cmp_eq_zero = _mm256_cmpeq_epi8(vec, zero);
        let zero_mask = _mm256_movemask_epi8(cmp_eq_zero) as u32;
        !zero_mask
    }

    /// `u16`-lane non-zero mask (SSE2, 8 lanes). Returns a mask
    /// whose bit `2·i` is set iff `u16` lane `i` of the 8-lane
    /// window at `ptr` is **non-zero**, so `trailing_zeros() / 2`
    /// is the first non-zero lane. Caller guarantees `ptr` is at
    /// least 16 bytes (8 × `u16`) valid.
    #[inline]
    pub(super) unsafe fn cmp_u16_neq_zero_mask_8(ptr: *const u16) -> u32 {
        let vec = unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) };
        let zero = _mm_setzero_si128();
        // 0xFFFF in lanes that ARE zero → `movemask_epi8` sets both
        // bytes of a zero lane. Even bit `2·i` is set iff lane `i`
        // is zero; invert and keep even bits for the non-zero mask.
        let cmp_eq_zero = _mm_cmpeq_epi16(vec, zero);
        let zero_mask = _mm_movemask_epi8(cmp_eq_zero) as u32;
        (!zero_mask) & 0x5555
    }

    /// `u16`-lane non-zero mask (AVX2, 16 lanes). Same even-bit
    /// encoding as [`cmp_u16_neq_zero_mask_8`]. Caller guarantees
    /// AVX2 support and `ptr` is at least 32 bytes (16 × `u16`)
    /// valid.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn cmp_u16_neq_zero_mask_16(ptr: *const u16) -> u32 {
        let vec = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
        let zero = _mm256_setzero_si256();
        let cmp_eq_zero = _mm256_cmpeq_epi16(vec, zero);
        let zero_mask = _mm256_movemask_epi8(cmp_eq_zero) as u32;
        (!zero_mask) & 0x5555_5555
    }

    #[inline]
    pub(super) unsafe fn prefetch_read_data(ptr: *const u8) {
        unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0) };
    }
}

// ---------------------------------------------------------------
// aarch64 — NEON (always available in the base aarch64 ISA)
// ---------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::{
        uint8x16_t, vceqq_u16, vceqq_u8, vdupq_n_u16, vdupq_n_u8, vget_lane_u64, vld1q_u16,
        vld1q_u8, vmvnq_u16, vmvnq_u8, vreinterpret_u64_u8, vreinterpretq_u16_u8,
        vreinterpretq_u8_u16, vshrn_n_u16,
    };

    /// Pack a `uint8x16_t` byte-mask (each byte = 0xFF or 0x00)
    /// into a `u64` nibble-mask: byte i → nibble at bits
    /// `[i*4 .. i*4+4]`, value `0xF` (match) or `0x0` (no match).
    #[inline]
    unsafe fn byte_mask_to_nibble_u64(cmp: uint8x16_t) -> u64 {
        let narrow = vshrn_n_u16::<4>(vreinterpretq_u16_u8(cmp));
        vget_lane_u64::<0>(vreinterpret_u64_u8(narrow))
    }

    /// Compress the low bit of each 4-bit nibble into the low
    /// 16 bits. Input nibbles are expected to be either `0x0` or
    /// `0xF`, as produced by [`byte_mask_to_nibble_u64`].
    #[inline]
    fn nibble_mask_to_bitmask_16(nib: u64) -> u32 {
        let mut x = nib & 0x1111_1111_1111_1111;
        x = (x | (x >> 3)) & 0x0303_0303_0303_0303;
        x = (x | (x >> 6)) & 0x000f_000f_000f_000f;
        x = (x | (x >> 12)) & 0x0000_00ff_0000_00ff;
        x = (x | (x >> 24)) & 0x0000_0000_0000_ffff;
        x as u32
    }

    /// Compare 16 bytes; return a 64-bit nibble-mask (see
    /// [`byte_mask_to_nibble_u64`]).
    #[inline]
    pub(super) unsafe fn cmp_16_bytes_nibble(a: *const u8, b: *const u8) -> u64 {
        let va = unsafe { vld1q_u8(a) };
        let vb = unsafe { vld1q_u8(b) };
        let cmp = vceqq_u8(va, vb);
        unsafe { byte_mask_to_nibble_u64(cmp) }
    }

    /// Find `byte` in 16 keys; return the first matching index or
    /// `None`. Caller guarantees `keys` is at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn find_byte_in_16(keys: *const u8, count: u8, byte: u8) -> Option<u8> {
        let vec = unsafe { vld1q_u8(keys) };
        let needle = vdupq_n_u8(byte);
        let cmp = vceqq_u8(vec, needle);
        let mask64 = unsafe { byte_mask_to_nibble_u64(cmp) };
        // count nibbles → count * 4 bits.
        let count_bits = (count.min(16) as u32) * 4;
        let count_mask = if count_bits == 64 {
            u64::MAX
        } else {
            (1u64 << count_bits) - 1
        };
        let masked = mask64 & count_mask;
        // First non-zero nibble's position / 4 = byte index.
        std::num::NonZeroU64::new(masked).map(|m| (m.get().trailing_zeros() / 4) as u8)
    }

    /// 16-bit mask where bit `i = 1` iff byte `i` equals `needle`.
    /// Caller guarantees `ptr` is at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn cmp_byte_eq_mask_16(ptr: *const u8, needle: u8) -> u32 {
        let vec = unsafe { vld1q_u8(ptr) };
        let needle = vdupq_n_u8(needle);
        let cmp = vceqq_u8(vec, needle);
        let nib = unsafe { byte_mask_to_nibble_u64(cmp) };
        nibble_mask_to_bitmask_16(nib)
    }

    /// 16-bit mask where bit `i = 1` iff byte `i` of the 16-byte
    /// window at `ptr` is **non-zero**. Caller guarantees `ptr` is
    /// at least 16 bytes valid.
    ///
    /// Mirrors the SSE2 path: compare the lanes to a zero vector,
    /// invert, then narrow to a 1-bit-per-byte mask via the same
    /// nibble-shrink trick used in [`cmp_16_bytes_nibble`] but
    /// post-processed to one bit per byte (we want trailing_zeros
    /// to return the byte index, not the nibble index).
    #[inline]
    pub(super) unsafe fn cmp_byte_neq_zero_mask_16(ptr: *const u8) -> u32 {
        let vec = unsafe { vld1q_u8(ptr) };
        let zero = vdupq_n_u8(0);
        let cmp_eq_zero = vceqq_u8(vec, zero);
        let cmp_neq_zero = vmvnq_u8(cmp_eq_zero);
        // Each byte lane is 0xFF for non-zero, 0x00 for zero;
        // shrink to a 64-bit nibble mask, then compact one bit per
        // nibble with SWAR. This avoids the previous per-chunk
        // 16-iteration scalar collapse on Apple Silicon.
        let nib = unsafe { byte_mask_to_nibble_u64(cmp_neq_zero) };
        nibble_mask_to_bitmask_16(nib)
    }

    /// `u16`-lane non-zero mask (NEON, 8 lanes). Returns a mask
    /// whose bit `2·i` is set iff `u16` lane `i` is **non-zero**
    /// (so `trailing_zeros() / 2` is the first non-zero lane),
    /// matching the SSE2 encoding. Caller guarantees `ptr` is at
    /// least 16 bytes (8 × `u16`) valid.
    #[inline]
    pub(super) unsafe fn cmp_u16_neq_zero_mask_8(ptr: *const u16) -> u32 {
        let vec = unsafe { vld1q_u16(ptr) };
        let zero = vdupq_n_u16(0);
        let cmp_eq_zero = vceqq_u16(vec, zero);
        let cmp_neq_zero = vmvnq_u16(cmp_eq_zero);
        // Each u16 lane is 0xFFFF (non-zero) or 0x0000 (zero), so
        // its two bytes are both 0xFF or both 0x00. Shrink to a
        // per-byte bitmask (bit `b` set iff byte `b` is non-zero)
        // via the shared nibble trick, then keep even bits so bit
        // `2·lane` marks a non-zero u16 lane.
        let as_bytes = vreinterpretq_u8_u16(cmp_neq_zero);
        let nib = unsafe { byte_mask_to_nibble_u64(as_bytes) };
        nibble_mask_to_bitmask_16(nib) & 0x5555
    }

    /// `u16`-lane non-zero mask (NEON, 16 lanes = 2 × 8). Combines
    /// two 8-lane masks: low 8 lanes occupy even bits 0..16, high 8
    /// lanes even bits 16..32, so `trailing_zeros() / 2` is the
    /// first non-zero lane across all 16. Caller guarantees `ptr`
    /// is at least 32 bytes (16 × `u16`) valid.
    #[inline]
    pub(super) unsafe fn cmp_u16_neq_zero_mask_16(ptr: *const u16) -> u32 {
        let lo = unsafe { cmp_u16_neq_zero_mask_8(ptr) };
        let hi = unsafe { cmp_u16_neq_zero_mask_8(ptr.add(8)) };
        lo | (hi << 16)
    }
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- node16_find_byte ----

    #[test]
    fn find_byte_at_index_zero() {
        let mut keys = [0u8; 16];
        keys[0] = 0x42;
        assert_eq!(node16_find_byte(&keys, 1, 0x42), Some(0));
    }

    #[test]
    fn find_byte_at_last_valid_index() {
        let mut keys = [0u8; 16];
        keys[15] = 0xAB;
        assert_eq!(node16_find_byte(&keys, 16, 0xAB), Some(15));
    }

    #[test]
    fn find_byte_middle() {
        let mut keys = [0u8; 16];
        for (i, slot) in keys.iter_mut().enumerate().take(10) {
            *slot = b'a' + i as u8;
        }
        assert_eq!(node16_find_byte(&keys, 10, b'f'), Some(5));
    }

    #[test]
    fn find_byte_absent_returns_none() {
        let mut keys = [0u8; 16];
        for (i, slot) in keys.iter_mut().enumerate().take(8) {
            *slot = b'a' + i as u8;
        }
        assert_eq!(node16_find_byte(&keys, 8, b'z'), None);
    }

    #[test]
    fn find_byte_count_zero_returns_none() {
        let keys = [0xAB; 16];
        assert_eq!(node16_find_byte(&keys, 0, 0xAB), None);
    }

    #[test]
    fn find_byte_ignores_unused_tail() {
        // count=4, but byte present at index 10 — must NOT find it.
        let mut keys = [0u8; 16];
        keys[10] = 0x77;
        assert_eq!(node16_find_byte(&keys, 4, 0x77), None);
    }

    #[test]
    fn find_byte_first_of_duplicates() {
        // If a byte appears twice (shouldn't happen in valid Node16
        // but the routine is defined to return the first), index 3
        // wins over index 7.
        let mut keys = [0u8; 16];
        keys[3] = 0x55;
        keys[7] = 0x55;
        assert_eq!(node16_find_byte(&keys, 16, 0x55), Some(3));
    }

    #[test]
    fn find_byte_matches_scalar_random() {
        use std::collections::HashSet;
        // Generate pseudo-random Node16 contents and random queries;
        // SIMD and scalar must always agree.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let next = |s: &mut u64| -> u8 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*s >> 33) as u8
        };
        for _ in 0..1000 {
            let count = next(&mut state) % 17; // 0..=16
            let mut keys = [0u8; 16];
            let mut used = HashSet::new();
            for k in keys.iter_mut().take(count as usize) {
                loop {
                    let b = next(&mut state);
                    if used.insert(b) {
                        *k = b;
                        break;
                    }
                }
            }
            let query = next(&mut state);
            let got = node16_find_byte(&keys, count, query);
            let expected = node16_find_byte_scalar(&keys, count, query);
            assert_eq!(
                got, expected,
                "mismatch on keys={keys:?} count={count} q={query}"
            );
        }
    }

    // ---- longest_common_prefix ----

    #[test]
    fn lcp_empty_inputs() {
        assert_eq!(longest_common_prefix(b"", b""), 0);
        assert_eq!(longest_common_prefix(b"abc", b""), 0);
        assert_eq!(longest_common_prefix(b"", b"abc"), 0);
    }

    #[test]
    fn lcp_identical() {
        assert_eq!(longest_common_prefix(b"hello", b"hello"), 5);
    }

    #[test]
    fn lcp_strict_prefix() {
        assert_eq!(longest_common_prefix(b"abc", b"abcdef"), 3);
        assert_eq!(longest_common_prefix(b"abcdef", b"abc"), 3);
    }

    #[test]
    fn lcp_no_common() {
        assert_eq!(longest_common_prefix(b"abc", b"xyz"), 0);
    }

    #[test]
    fn lcp_divergence_at_boundary() {
        // Crosses the 16-byte SIMD boundary.
        let a = b"0123456789ABCDEFhello"; // 21 bytes
        let b = b"0123456789ABCDEFworld"; // 21 bytes
        assert_eq!(longest_common_prefix(a, b), 16);
    }

    #[test]
    fn lcp_long_match_then_diverge_in_chunk() {
        // 32 byte common prefix, then diverge at byte 35.
        let a = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa01"; // 37 bytes
        let b = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa99"; // 37 bytes
        assert_eq!(longest_common_prefix(a, b), 35);
    }

    #[test]
    fn lcp_match_then_diverge_at_byte_15() {
        // Diverge just before the 16-byte boundary.
        let a = b"aaaaaaaaaaaaaaaXrest";
        let b = b"aaaaaaaaaaaaaaaYrest";
        assert_eq!(longest_common_prefix(a, b), 15);
    }

    #[test]
    fn lcp_match_then_diverge_at_byte_16() {
        // Diverge at the boundary.
        let a = b"aaaaaaaaaaaaaaaaXrest";
        let b = b"aaaaaaaaaaaaaaaaYrest";
        assert_eq!(longest_common_prefix(a, b), 16);
    }

    // ---- find_byte ----

    fn scalar_find_byte(bytes: &[u8], needle: u8, start: usize) -> Option<usize> {
        bytes
            .iter()
            .enumerate()
            .skip(start)
            .find(|(_, b)| **b == needle)
            .map(|(i, _)| i)
    }

    #[test]
    fn find_byte_respects_start_and_boundaries() {
        let mut bytes = [b'a'; 96];
        for pos in [0usize, 1, 15, 16, 17, 31, 32, 63, 64, 95] {
            bytes.fill(b'a');
            bytes[pos] = b'/';
            assert_eq!(find_byte(&bytes, b'/', 0), Some(pos), "pos={pos}");
            assert_eq!(find_byte(&bytes, b'/', pos), Some(pos), "pos={pos}");
            if pos + 1 < bytes.len() {
                assert_eq!(find_byte(&bytes, b'/', pos + 1), None, "pos={pos}");
            }
        }
    }

    #[test]
    fn find_byte_random_matches_scalar() {
        let mut state: u64 = 0xA11C_E55E_D15C_0DED;
        let step = |s: &mut u64| -> u8 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*s >> 33) as u8
        };
        for len in [0usize, 1, 3, 15, 16, 17, 31, 32, 33, 127, 255] {
            let mut bytes = vec![0u8; len];
            for b in &mut bytes {
                *b = step(&mut state);
            }
            for _ in 0..32 {
                let needle = step(&mut state);
                let start = if len == 0 {
                    0
                } else {
                    (step(&mut state) as usize) % (len + 1)
                };
                assert_eq!(
                    find_byte(&bytes, needle, start),
                    scalar_find_byte(&bytes, needle, start),
                    "len={len} start={start} needle={needle}",
                );
            }
        }
    }

    // ---- find_next_nonzero_byte ----

    fn scalar_next_nonzero_byte(bytes: &[u8], start: usize) -> Option<usize> {
        bytes
            .iter()
            .enumerate()
            .skip(start)
            .find(|(_, b)| **b != 0)
            .map(|(i, _)| i)
    }

    #[test]
    fn find_next_nonzero_byte_empty() {
        let bytes = [0u8; 256];
        assert_eq!(find_next_nonzero_byte(&bytes, 0), None);
        assert_eq!(find_next_nonzero_byte(&bytes, 100), None);
        assert_eq!(find_next_nonzero_byte(&bytes, 256), None);
    }

    #[test]
    fn find_next_nonzero_byte_at_chunk_boundaries() {
        for pos in [0usize, 1, 15, 16, 17, 31, 32, 240, 254, 255] {
            let mut bytes = [0u8; 256];
            bytes[pos] = 0xAB;
            assert_eq!(find_next_nonzero_byte(&bytes, 0), Some(pos), "pos={pos}");
            assert_eq!(find_next_nonzero_byte(&bytes, pos), Some(pos), "pos={pos}");
            if pos + 1 < 256 {
                assert_eq!(find_next_nonzero_byte(&bytes, pos + 1), None, "pos+1={pos}");
            }
        }
    }

    #[test]
    fn find_next_nonzero_byte_random_matches_scalar() {
        let mut state: u64 = 0xCAFE_F00D_1234_5678;
        let step = |s: &mut u64| -> u8 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*s >> 33) as u8
        };
        for _ in 0..500 {
            let mut bytes = [0u8; 256];
            // Sparse: ~5% non-zero.
            for b in &mut bytes {
                if step(&mut state) < 13 {
                    *b = step(&mut state).max(1);
                }
            }
            let start = (step(&mut state) as usize) % 257;
            assert_eq!(
                find_next_nonzero_byte(&bytes, start),
                scalar_next_nonzero_byte(&bytes, start),
                "start={start}",
            );
        }
    }

    // ---- find_next_nonzero_u16 ----

    fn scalar_next_nonzero_u16(words: &[u16], start: usize) -> Option<usize> {
        (start..words.len()).find(|&i| words[i] != 0)
    }

    #[test]
    fn find_next_nonzero_u16_matches_scalar() {
        let mut state: u64 = 0x1357_9BDF_2468_ACE0;
        let step = |s: &mut u64| -> u32 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*s >> 32) as u32
        };
        // Cover both the even-length inner-node arrays (48 / 256)
        // and odd start offsets (which land mid-group in the
        // 4-lane fast path).
        for &len in &[4usize, 16, 48, 256] {
            for _ in 0..300 {
                let mut words = vec![0u16; len];
                for w in &mut words {
                    if step(&mut state).trailing_zeros() >= 3 {
                        *w = (step(&mut state) as u16).max(1);
                    }
                }
                for start in 0..=len {
                    assert_eq!(
                        find_next_nonzero_u16(&words, start),
                        scalar_next_nonzero_u16(&words, start),
                        "len={len} start={start}",
                    );
                }
            }
        }
    }
}
