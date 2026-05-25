//! `AtomicBatch` — buffer multiple ops for a single-record WAL commit.
//!
//! Companion to [`super::tree::Tree::atomic`]. The batch is a
//! mutation/guard accumulator: each `put` / `delete` / `rename`
//! call copies its inputs into the pending list, and each `assert_*`
//! call buffers a logical precondition. Nothing touches the tree
//! until the closure passed to `Tree::atomic` returns; then
//! `Tree::apply_batch` drains the pending list under the
//! mutation gate and emits one Batch WAL record.
//!
//! Atomicity contract is documented on `Tree::atomic` — short
//! version: logical preconditions are checked before mutation,
//! concurrent readers/writers cannot observe intermediate batch
//! state, and replay sees all-or-nothing.

/// Value plus the live record version observed by one lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// Value bytes stored under the key.
    pub value: Vec<u8>,
    /// Current compare-and-set token for the live record.
    pub version: RecordVersion,
}

/// Opaque per-record version returned by
/// [`Tree::get_version`](super::tree::Tree::get_version).
///
/// Holt uses leaf sequence numbers as lightweight compare-and-set
/// tokens. They are valid only for conditional writes against the
/// current tree state; they are **not** MVCC timestamps and do not
/// let callers read historical snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct RecordVersion(u64);

impl RecordVersion {
    /// Build a version from a raw sequence number previously
    /// obtained via [`Self::as_u64`].
    ///
    /// This exists so callers can persist or send a version token
    /// through their own metadata layer. Supplying an arbitrary
    /// value is safe: a conditional write simply returns `false`
    /// when no live record currently carries that sequence.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw sequence number backing this token.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Builder for an atomic batch. See [`super::tree::Tree::atomic`].
#[derive(Debug, Default)]
pub struct AtomicBatch {
    pub(crate) pending: Vec<BatchOp>,
}

#[derive(Debug)]
pub(crate) enum BatchOp {
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
    },
    PutIfAbsent {
        key: Vec<u8>,
        value: Vec<u8>,
    },
    CompareAndPut {
        key: Vec<u8>,
        expected: RecordVersion,
        value: Vec<u8>,
    },
    Delete {
        key: Vec<u8>,
    },
    DeleteIfVersion {
        key: Vec<u8>,
        expected: RecordVersion,
    },
    AssertVersion {
        key: Vec<u8>,
        expected: RecordVersion,
    },
    AssertPrefixEmpty {
        prefix: Vec<u8>,
    },
    Rename {
        src: Vec<u8>,
        dst: Vec<u8>,
        force: bool,
    },
}

impl BatchOp {
    pub(crate) const fn emits_wal(&self) -> bool {
        !matches!(
            self,
            Self::AssertVersion { .. } | Self::AssertPrefixEmpty { .. }
        )
    }
}

impl AtomicBatch {
    /// Buffer a `put(key, value)` to apply when the batch commits.
    pub fn put(&mut self, key: &[u8], value: &[u8]) {
        self.pending.push(BatchOp::Put {
            key: key.to_vec(),
            value: value.to_vec(),
        });
    }

    /// Buffer a create-only write. If `key` exists when the batch
    /// commits, the whole atomic batch returns `Ok(false)` and
    /// publishes no mutations.
    pub fn put_if_absent(&mut self, key: &[u8], value: &[u8]) {
        self.pending.push(BatchOp::PutIfAbsent {
            key: key.to_vec(),
            value: value.to_vec(),
        });
    }

    /// Buffer a version-guarded update. If `key` is missing or
    /// carries another version when the batch commits, the whole
    /// atomic batch returns `Ok(false)` and publishes no mutations.
    pub fn compare_and_put(&mut self, key: &[u8], expected: RecordVersion, value: &[u8]) {
        self.pending.push(BatchOp::CompareAndPut {
            key: key.to_vec(),
            expected,
            value: value.to_vec(),
        });
    }

    /// Buffer a `delete(key)` to apply when the batch commits.
    pub fn delete(&mut self, key: &[u8]) {
        self.pending.push(BatchOp::Delete { key: key.to_vec() });
    }

    /// Buffer a version-guarded delete. If `key` is missing or
    /// carries another version when the batch commits, the whole
    /// atomic batch returns `Ok(false)` and publishes no mutations.
    pub fn delete_if_version(&mut self, key: &[u8], expected: RecordVersion) {
        self.pending.push(BatchOp::DeleteIfVersion {
            key: key.to_vec(),
            expected,
        });
    }

    /// Require that `key` exists with `expected` when the batch
    /// commits. If the key is missing or carries another version,
    /// the whole atomic batch returns `Ok(false)` and publishes no
    /// mutations.
    ///
    /// This is a read-only compare-and-set guard: it validates a
    /// source record for copy / multipart-complete style metadata
    /// flows without rewriting that source or bumping its version.
    pub fn assert_version(&mut self, key: &[u8], expected: RecordVersion) {
        self.pending.push(BatchOp::AssertVersion {
            key: key.to_vec(),
            expected,
        });
    }

    /// Require that no live key starts with `prefix` when the batch
    /// commits. If the projected prefix is non-empty, the whole
    /// atomic batch returns `Ok(false)` and publishes no mutations.
    ///
    /// The check observes earlier operations in the same batch. For
    /// example, deleting the last child before this assertion lets a
    /// following directory-marker delete commit atomically.
    pub fn assert_prefix_empty(&mut self, prefix: &[u8]) {
        self.pending.push(BatchOp::AssertPrefixEmpty {
            prefix: prefix.to_vec(),
        });
    }

    /// Buffer a `rename(src, dst, force)` to apply when the batch
    /// commits. The semantics match
    /// [`super::tree::Tree::rename`] — missing `src` errors,
    /// `dst` collision errors unless `force` is `true`.
    pub fn rename(&mut self, src: &[u8], dst: &[u8], force: bool) {
        self.pending.push(BatchOp::Rename {
            src: src.to_vec(),
            dst: dst.to_vec(),
            force,
        });
    }

    /// Number of ops queued so far.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// `true` if nothing has been queued. A closure that leaves
    /// the batch empty makes [`super::tree::Tree::atomic`] return
    /// without taking endpoint locks or emitting a WAL record.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}
