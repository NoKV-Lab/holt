//! Journal — physiological WAL, replay, checkpoint.
//!
//! Layered design:
//!
//! - [`txn_op`] — the `TxnOp` variant union; one variant per
//!   walker-level mutation kind (`Insert`, `Erase`, `Split`,
//!   `Merge`, `Compact`, two `Rename` flavours, `NewTree`,
//!   `RmTree`, `MemMarker`).
//! - [`codec`] — binary record codec + file header. Pure
//!   in-memory bytes ↔ TxnOp.
//! - [`writer`] — append-only WAL file with
//!   `fdatasync`-on-flush.
//! - [`reader`] — forward replay scanner with graceful torn-tail
//!   handling.
//! - [`checkpoint`] (Stage 5c — queued) — trim the log past the
//!   last durable blob commit + integrate with the engine.

pub mod checkpoint;
pub mod codec;
pub mod reader;
pub mod txn_op;
pub mod writer;
