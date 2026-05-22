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
        /// Key bytes.
        key: Vec<u8>,
    },
    /// Atomic in-tree rename.
    RenameObject {
        /// Source key.
        src_key: Vec<u8>,
        /// Destination key.
        dst_key: Vec<u8>,
        /// Overwrite if dst exists.
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
    Batch {
        /// Inner ops, applied in order.
        ops: Vec<WalOp>,
    },
}
