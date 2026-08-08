//! WalOp variants — durable logical redo records.
//!
//! Each variant carries the minimal info needed to replay the
//! operation deterministically during WAL recovery.

/// Logical WAL operation variants emitted by the public tree API.
///
/// Variant tags are stable on-disk constants — see the `TY_*`
/// block in [`super::codec`]. Never renumber; only append.
#[derive(Debug, Clone)]
pub enum WalOp {
    /// Single-key insert / update.
    ///
    /// Replay only redoes from `(key, value)`; there is no
    /// `prev_value` field because replay never undoes (it's an
    /// idempotent forward redo) and holt does not provide a
    /// journal-scan audit surface.
    Insert {
        /// Logical tree owner.
        tree_id: u64,
        /// Key bytes.
        key: Vec<u8>,
        /// New value bytes.
        value: Vec<u8>,
    },
    /// Single-key erase.
    ///
    /// Carries only the key because replay redoes the erase from
    /// `key` alone.
    Erase {
        /// Logical tree owner.
        tree_id: u64,
        /// Key bytes.
        key: Vec<u8>,
    },
    /// Atomic in-tree rename.
    RenameObject {
        /// Logical tree owner.
        tree_id: u64,
        /// Source key.
        src_key: Vec<u8>,
        /// Destination key.
        dst_key: Vec<u8>,
        /// Source value captured by the committing operation.
        ///
        /// Recovery must never re-read `src_key` from a checkpoint image that
        /// may already contain a later write.
        value: Vec<u8>,
        /// Overwrite if dst exists.
        #[allow(dead_code)]
        // parsed for v4 wire fidelity; redo always installs the bound value
        force: bool,
    },
    /// Batch — one WAL record carrying multiple primitive ops so a
    /// crash either replays all of them or none.
    ///
    /// Emitted by [`crate::Tree::atomic`]. Inner ops are primitive
    /// variants only (`Insert` / `Erase` / `RenameObject` today);
    /// nested `Batch`es are rejected at encode + decode. The outer
    /// record's header `SEQ` is the base, and replay derives each
    /// inner op's sequence as `outer_seq + index`.
    #[cfg_attr(not(test), allow(dead_code))] // production streams batch primitives directly
    Batch {
        /// Inner ops, applied in order.
        ops: Vec<WalOp>,
    },
}

impl WalOp {
    /// Tree id carried by primitive WAL ops.
    ///
    /// `Batch` does not have one owner: each inner op carries its
    /// own tree id so a future DB-level atomic record can cover
    /// multiple named trees.
    #[must_use]
    pub const fn tree_id(&self) -> Option<u64> {
        match self {
            Self::Insert { tree_id, .. }
            | Self::Erase { tree_id, .. }
            | Self::RenameObject { tree_id, .. } => Some(*tree_id),
            Self::Batch { .. } => None,
        }
    }
}
