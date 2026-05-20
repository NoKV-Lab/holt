//! TxnOp variants — one per ART mutation kind.
//!
//! Each variant carries the minimal info needed to replay the
//! operation deterministically during WAL recovery.

/// Reason a `compactBlob` (or `splitBlob`-triggered compact)
/// fired. Encoded into the WAL as the `reason` body of
/// [`TxnOp::Compact`]. Stable on-disk tag values are assigned
/// in [`super::codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactReason {
    /// Too many tombstone leaves; rebuild dropping them.
    SplitTombstone,
    /// Bump-allocator wasted space exceeds threshold; rebuild
    /// compactly.
    SplitGapSpace,
    /// Alloc failed in the current blob; spill a subtree out.
    OutOfBlobFrame,
}

/// 11 transaction-op variants emitted by the walker.
///
/// Variant tags are stable on-disk constants — see the `TY_*`
/// block in [`super::codec`]. Never renumber; only append.
// `seq` fields are populated on decode (from the record header) and
// verified via codec round-trip tests, but production replay consumes
// the per-record `seq` via the callback's separate parameter rather
// than re-reading it off the variant. Allow dead_code so the lint
// doesn't fire on those fields in non-test builds.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum TxnOp {
    /// Single-key insert / update.
    ///
    /// Replay only redoes from `(key, value)`; there is no
    /// `prev_value` field because replay never undoes (it's an
    /// idempotent forward redo) and holt does not provide a
    /// journal-scan audit surface.
    Insert {
        /// Owning tree root identifier.
        tree_id: u64,
        /// MVCC seq this op was committed at.
        seq: u64,
        /// Key bytes.
        key: Vec<u8>,
        /// New value bytes.
        value: Vec<u8>,
    },
    /// Single-key erase.
    ///
    /// Carries only the key — replay redoes the erase from `key`
    /// alone. The prior value is not retained on disk: the blind
    /// `Tree::delete` walker never reads it, and the returning
    /// `Tree::remove` walker hands it straight to the caller
    /// without round-tripping through the WAL.
    Erase {
        /// Owning tree root identifier.
        tree_id: u64,
        /// MVCC seq this op was committed at.
        seq: u64,
        /// Key bytes.
        key: Vec<u8>,
    },
    /// `splitBlob` — subtree moved to a new blob.
    Split {
        /// Parent blob's GUID.
        parent_blob: [u8; 16],
        /// Slot that pointed at the pre-split node.
        pre_split_node: u16,
        /// New child blob's GUID.
        new_child_blob: [u8; 16],
        /// Entry slot inside the new child blob.
        new_child_entry: u16,
    },
    /// `mergeBlob` — child blob's contents pulled back into parent.
    Merge {
        /// Parent blob's GUID.
        parent_blob: [u8; 16],
        /// Slot at which the merge-target sat.
        pre_merge_node: u16,
        /// Child blob that was merged + freed.
        child_blob: [u8; 16],
    },
    /// `compactBlob` — in-place rebuild dropping orphans.
    Compact {
        /// Compacted blob's GUID.
        blob: [u8; 16],
        /// Why we compacted.
        reason: CompactReason,
    },
    /// Atomic in-tree rename.
    RenameObject {
        /// Owning tree root identifier.
        tree_id: u64,
        /// MVCC seq.
        seq: u64,
        /// Source key.
        src_key: Vec<u8>,
        /// Destination key.
        dst_key: Vec<u8>,
        /// Overwrite if dst exists.
        force: bool,
    },
    /// Cross-tree rename (different bucket / root).
    Rename {
        /// Source tree.
        src_tree_id: u64,
        /// Destination tree.
        dst_tree_id: u64,
        /// MVCC seq.
        seq: u64,
        /// Source key.
        src_key: Vec<u8>,
        /// Destination key.
        dst_key: Vec<u8>,
        /// Overwrite if dst exists.
        force: bool,
    },
    /// Create a new tree (NewTreeTxnOp).
    NewTree {
        /// Tree root identifier to allocate.
        tree_id: u64,
        /// Tree's name (bucket name in S3 terms).
        name: Vec<u8>,
    },
    /// Drop a tree (RmTreeTxnOp).
    RmTree {
        /// Tree root identifier.
        tree_id: u64,
    },
    /// Memory-only twin: SplitMemOp, MergeMemOp, etc.
    /// (Post-replay-ack reconciliation; carries no durable state.)
    MemMarker {
        /// Sequence number for reconciliation.
        seq: u64,
    },
    /// Batch — one WAL record carrying multiple primitive ops so a
    /// crash either replays all of them or none.
    ///
    /// Emitted by [`crate::Tree::txn`]. Inner ops are primitive
    /// variants only (`Insert` / `Erase` / `RenameObject` today);
    /// nested `Batch`es are rejected at encode + decode. Each
    /// inner op carries `seq = outer_seq + index`; the outer
    /// record's header `SEQ` is the base, and the WAL allocator
    /// reserves a contiguous range of `ops.len()` seqs per batch.
    Batch {
        /// Owning tree root identifier.
        tree_id: u64,
        /// Inner ops, applied in order.
        ops: Vec<TxnOp>,
    },
}
