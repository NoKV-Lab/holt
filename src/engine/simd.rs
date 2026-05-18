//! SIMD hot paths used by the walker.
//!
//! Two operations dominate the ART walker's read cost:
//!
//! 1. **Node16 byte search** — find the index `i` in `keys[0..count]`
//!    such that `keys[i] == byte`. The scalar form is a 16-iteration
//!    loop; SSE2 / NEON do the same work in ~3 instructions.
//! 2. **Longest common prefix** — find the first divergence between
//!    two byte slices (Leaf split, Prefix split). 16 bytes per
//!    iteration via vector compare.
//!
//! Dispatch is compile-time only — `cfg(target_arch = ...)` selects
//! the SSE2 path on x86_64 (always available — SSE2 is part of the
//! base x86_64 ISA), the NEON path on aarch64 (also always
//! available), and a scalar fallback otherwise. Behaviour is
//! identical across paths; the scalar form is the spec.

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

    while i < limit && a[i] == b[i] {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------
// x86_64 — SSE2 (always available in the base x86_64 ISA)
// ---------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Compare 16 bytes from `a` against 16 from `b`. Returns a
    /// 16-bit bitmask: bit `i` = 1 iff `a[i] == b[i]`. Caller
    /// guarantees both pointers are at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn cmp_16_bytes_bitmask(a: *const u8, b: *const u8) -> u32 {
        let va = unsafe { _mm_loadu_si128(a as *const __m128i) };
        let vb = unsafe { _mm_loadu_si128(b as *const __m128i) };
        let cmp = _mm_cmpeq_epi8(va, vb);
        _mm_movemask_epi8(cmp) as u32
    }

    /// Find `byte` in 16 keys; return the first matching index or
    /// `None`. Caller guarantees `keys` is at least 16 bytes valid.
    #[inline]
    pub(super) unsafe fn find_byte_in_16(keys: *const u8, count: u8, byte: u8) -> Option<u8> {
        let vec = unsafe { _mm_loadu_si128(keys as *const __m128i) };
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
        if masked == 0 {
            None
        } else {
            Some(masked.trailing_zeros() as u8)
        }
    }
}

// ---------------------------------------------------------------
// aarch64 — NEON (always available in the base aarch64 ISA)
// ---------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::{
        uint8x16_t, vceqq_u8, vdupq_n_u8, vget_lane_u64, vld1q_u8, vreinterpret_u64_u8,
        vreinterpretq_u16_u8, vshrn_n_u16,
    };

    /// Pack a `uint8x16_t` byte-mask (each byte = 0xFF or 0x00)
    /// into a `u64` nibble-mask: byte i → nibble at bits
    /// `[i*4 .. i*4+4]`, value `0xF` (match) or `0x0` (no match).
    #[inline]
    unsafe fn byte_mask_to_nibble_u64(cmp: uint8x16_t) -> u64 {
        let narrow = vshrn_n_u16::<4>(vreinterpretq_u16_u8(cmp));
        vget_lane_u64::<0>(vreinterpret_u64_u8(narrow))
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
        if masked == 0 {
            None
        } else {
            // First non-zero nibble's position / 4 = byte index.
            Some((masked.trailing_zeros() / 4) as u8)
        }
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
}
