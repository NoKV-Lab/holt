//! Search-key view used by point lookup / insert / erase.
//!
//! Public API calls pass user keys without the ART terminator. The
//! tree stores leaf keys with a trailing `0` byte so strict-prefix
//! pairs diverge inside the ART. `SearchKey` makes that terminator
//! virtual during descent, avoiding the old per-op copy into a
//! padded scratch buffer. When a new leaf is actually written, the
//! key is materialised once into the leaf extent.

use crate::engine::simd;

#[derive(Clone, Copy)]
pub(crate) struct SearchKey<'a> {
    bytes: &'a [u8],
    virtual_terminator: bool,
}

impl<'a> SearchKey<'a> {
    #[inline]
    pub(crate) fn user(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            virtual_terminator: true,
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn exact(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            virtual_terminator: false,
        }
    }

    #[inline]
    pub(crate) fn len(self) -> usize {
        self.bytes.len() + usize::from(self.virtual_terminator)
    }

    #[inline]
    pub(crate) fn user_prefix(self, len: usize) -> Option<&'a [u8]> {
        if len <= self.bytes.len() {
            Some(&self.bytes[..len])
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn remaining_len(self, depth: usize) -> usize {
        self.len().saturating_sub(depth)
    }

    #[inline]
    pub(crate) fn byte_at(self, idx: usize) -> Option<u8> {
        if idx < self.bytes.len() {
            Some(self.bytes[idx])
        } else if self.virtual_terminator && idx == self.bytes.len() {
            Some(0)
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn eq_slice(self, other: &[u8]) -> bool {
        if !self.virtual_terminator {
            return self.bytes.len() == other.len()
                && simd::longest_common_prefix(self.bytes, other) == self.bytes.len();
        }
        if other.len() != self.bytes.len() + 1 {
            return false;
        }
        simd::longest_common_prefix(self.bytes, &other[..self.bytes.len()]) == self.bytes.len()
            && other[self.bytes.len()] == 0
    }

    #[inline]
    pub(crate) fn range_eq(self, depth: usize, other: &[u8]) -> bool {
        if other.is_empty() {
            return depth <= self.len();
        }
        if other.len() > self.remaining_len(depth) {
            return false;
        }
        if !self.virtual_terminator || depth + other.len() <= self.bytes.len() {
            return simd::longest_common_prefix(&self.bytes[depth..depth + other.len()], other)
                == other.len();
        }

        if depth < self.bytes.len() {
            let raw_len = self.bytes.len() - depth;
            return simd::longest_common_prefix(&self.bytes[depth..], &other[..raw_len]) == raw_len
                && other[raw_len] == 0;
        }
        other[0] == 0
    }

    pub(crate) fn common_prefix_with_slice(self, depth: usize, other: &[u8]) -> usize {
        if depth >= self.len() || other.is_empty() {
            return 0;
        }
        if !self.virtual_terminator {
            return simd::longest_common_prefix(&self.bytes[depth..], other);
        }
        if depth < self.bytes.len() {
            let raw_tail = &self.bytes[depth..];
            let common = simd::longest_common_prefix(raw_tail, other);
            if common < raw_tail.len() || common == other.len() {
                return common;
            }
            return common + usize::from(other[common] == 0);
        }
        usize::from(other[0] == 0)
    }

    pub(crate) fn write_to_slice(self, dst: &mut [u8]) {
        debug_assert_eq!(dst.len(), self.len());
        if self.virtual_terminator {
            dst[..self.bytes.len()].copy_from_slice(self.bytes);
            dst[self.bytes.len()] = 0;
        } else {
            dst.copy_from_slice(self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SearchKey;

    #[test]
    fn user_key_has_virtual_terminator() {
        let key = SearchKey::user(b"abc");
        assert_eq!(key.len(), 4);
        assert_eq!(key.byte_at(0), Some(b'a'));
        assert_eq!(key.byte_at(3), Some(0));
        assert_eq!(key.byte_at(4), None);
        assert!(key.eq_slice(b"abc\0"));
    }

    #[test]
    fn virtual_key_compares_across_terminator() {
        let key = SearchKey::user(b"abc");
        assert!(key.range_eq(2, b"c\0"));
        assert_eq!(key.common_prefix_with_slice(0, b"abc\0def"), 4);
        assert_eq!(key.common_prefix_with_slice(0, b"abcdef"), 3);
    }
}
