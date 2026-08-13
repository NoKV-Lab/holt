//! Journal — logical WAL + replay.
//!
//! Layered design:
//!
//! - [`wal_op`] — the `WalOp` variant union for durable logical
//!   mutations (`Insert`, `Erase`, `RenameObject`, `Batch`, and attached DB
//!   batches).
//! - [`codec`] — binary record codec + file header. Pure
//!   in-memory bytes ↔ `WalOp`.
//! - [`writer`] — append-only WAL file with
//!   `sync_data`-on-flush durability + 64 KB buffered auto-drain
//!   mechanics.
//! - [`group_commit`] — WAL append coordinator. Writers publish
//!   encoded records into a shared byte ring; one flusher drains the
//!   committed prefix and runs `sync_data` for `wal_sync = true`
//!   barriers.
//! - [`reader`] — forward replay scanner with graceful
//!   torn-tail handling. Unpacks batch records into per-inner callbacks for
//!   ordinary replay and validates the retained attached-envelope suffix for
//!   the public DB recovery API.
//!
//! Checkpoint (flush WAL → drain dirty → fdatasync → truncate
//! WAL) lives in [`crate::Tree::checkpoint`] and the background
//! [`crate::checkpoint`] module, not in here — it straddles the
//! tree + journal boundary.

pub mod codec;
// The WAL append coordinator: a lock-free shared ring (`ring`) + a single
// flusher (`group_commit`).
pub(crate) mod group_commit;
pub mod reader;
pub(crate) mod ring;
pub mod wal_op;
pub mod writer;

pub(crate) use group_commit::{Journal, JournalAck, JournalAppendGuard};

#[cfg(test)]
mod tests;
