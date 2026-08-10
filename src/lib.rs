//! # holt — adaptive radix tree metadata storage engine
//!
//! `holt` is an embedded Rust library that stores **path-shaped
//! metadata** with sub-microsecond lookups, per-blob concurrency,
//! and crash-safe persistence. It is built around an Adaptive
//! Radix Tree that spans multiple 512 KB blob frames.
//!
//! See `README.md` for the elevator pitch and `ARCHITECTURE.md`
//! for the deep dive.
//!
//! ## Current status
//!
//! All core layers — layout, walker (insert / lookup / erase /
//! range / spillover / compact / merge), persistent store
//! (`O_DIRECT` + optional `io_uring`), WAL with replay, sharded
//! buffer manager, background checkpointer, copy-on-write
//! snapshots — are in place and covered by integration + property
//! tests.
//!
//! [`Durability::Wal`] controls how Holt acknowledges its local
//! write-ahead log. File-backed trees replay that WAL on open; manual
//! and background checkpoints make older log records redundant.
//!
//! ## Quick taste
//!
//! ```ignore
//! use holt::{RangeEntry, TreeBuilder};
//!
//! let tree = TreeBuilder::new("/var/lib/myapp/meta.holt").open()?;
//! tree.put(b"img/01.jpg", b"rgb_data")?;
//! let v: Vec<u8> = tree.get(b"img/01.jpg")?.unwrap();
//! for entry in tree.scan(b"img/").into_iter().take(10) {
//!     if let RangeEntry::Key { key, value, .. } = entry? {
//!         println!("{key:?} -> {value:?}");
//!     }
//! }
//! # Ok::<(), holt::Error>(())
//! ```
//!
//! ## Module map
//!
//! The supported import surface is the flat crate root:
//! [`Tree`], [`DB`], [`TreeBuilder`], [`AtomicBatch`], [`Record`],
//! [`RecordVersion`], [`View`], [`Snapshot`], [`KeyPathBuf`], [`KeyPrefixBuf`],
//! range iterator types, stats snapshots, and the optional
//! `metrics` renderer.
//!
//! All implementation modules are crate-private. This keeps the
//! on-disk format, WAL codec, walker, and buffer-manager internals
//! free to change in minor releases without breaking downstream
//! code.
//!
//! Internal modules (`pub(crate)`, not part of the SemVer surface):
//!
//! - `layout` — extern struct layouts (BlobHeader, SlotEntry,
//!   per-NodeType bodies). Pinned at compile time via
//!   `const _: () = assert!(...)` blocks per file.
//! - `journal` — WAL codec + replay scanner + writer.
//! - `store` — buffer manager + blob-frame allocator + store
//!   trait machinery. The supported store surface
//!   ([`BlobStore`], [`MemoryBlobStore`], [`FileBlobStore`],
//!   [`AlignedBlobBuf`]) is re-exported at the crate root for
//!   users who want to plug in a custom store.
//! - `engine` — recursive walker (insert / lookup / erase /
//!   scan / rename / compact). Record and key-only range iterator
//!   types and stats snapshots are re-exported at the crate root.
//! - `concurrency` — `HybridLatch` 3-mode lock plus the
//!   tree-wide maintenance gate.
//! - `checkpoint` — 3-thread background checkpointer for dirty blob
//!   drain, WAL flush/truncate, and cache eviction.
//!
//! ## Platform support
//!
//! holt is **Unix-only by design**: Linux (`O_DIRECT` fast path,
//! `io_uring` on the file store) and macOS (`F_NOCACHE`).
//! Windows is out of scope and the crate refuses to compile there
//! — see the platform stance in `ROADMAP.md`.

#![doc(html_no_source)]
#![deny(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(clippy::pedantic)]
// `clippy::pedantic` opt-in is on purpose — we want the high
// signal-to-noise lints firing. The blanket allows below are
// the categories we've reviewed and judged to be either
// intentional design choices in this crate or stylistic
// preferences that don't carry their weight here.
#![allow(
    // Intentional: many `T as U` casts are guarded by upstream
    // invariants (slot < MAX_SLOTS = 10240 < u16::MAX, value
    // length already validated < u16::MAX, etc.). Replacing all
    // with `try_into().unwrap()` would be net negative.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    // Misleading suggestion when we're explicitly choosing the
    // unchecked path for layout reasons.
    clippy::cast_ptr_alignment,
    // Many internal `Result`-returning helpers don't need an
    // `# Errors` section — the docstring already explains
    // failure modes inline.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Pedantic style nits that don't affect correctness:
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::stable_sort_primitive,
    clippy::large_types_passed_by_value,
    clippy::struct_field_names,
    // Reserved-counter fields in `BlobHeader` (RE'd byte
    // positions) intentionally aren't read from yet; the
    // compile-time offset asserts pin them in place.
    clippy::struct_excessive_bools,
    // Docstrings already use backticks for code spans; the
    // remaining flagged identifiers (e.g. "BufferManager",
    // "BlobFrame") are prose-level references where the
    // backtick noise is net-negative.
    clippy::doc_markdown,
    // Mixing items / statements is fine in small fns; refusing
    // it forces hoisting of locally-scoped const helpers.
    clippy::items_after_statements
)]

// Hard scope gate — see the "Platform support" section in the
// module docs. Building holt for Windows is intentionally
// unsupported; the file store's `O_DIRECT` / `F_NOCACHE`
// path has no Windows analog worth maintaining for this project.
#[cfg(not(unix))]
compile_error!(
    "holt is Unix-only — Linux and macOS are supported. Windows is out of scope; \
     see ROADMAP.md."
);

mod api;

pub(crate) mod checkpoint;
pub(crate) mod concurrency;
pub(crate) mod engine;
pub(crate) mod journal;
pub(crate) mod layout;
pub(crate) mod store;

/// Prometheus text-format renderer for [`TreeStats`]. Enabled via
/// the `metrics` feature flag.
#[cfg(feature = "metrics")]
pub mod metrics;

// -- Top-level re-exports -----------------------------------------
//
// The flat `holt::*` surface — every supported public name a user
// reaches for via `use holt::X` lives here.

// Core handle + configuration.
pub use api::builder::TreeBuilder;
pub use api::checkpoint::CheckpointImage;
pub use api::config::{AccessMode, Durability, Storage, TreeConfig};
pub use api::db::{DBAtomicBatch, DBView, DB};
pub use api::errors::{Error, Result};
pub use api::key::{KeyPathBuf, KeyPathError, KeyPrefixBuf};
pub use api::snapshot::Snapshot;
pub use api::tree::{PutOutcome, Tree};
pub use api::view::{View, ViewKeyRangeBuilder, ViewRangeBuilder};

// Range-scan iterator surface.
pub use engine::{
    KeyRangeBuilder, KeyRangeEntry, KeyRangeEntryRef, KeyRangeIter, KeyScanOutcome, PrefixCount,
    RangeBuilder, RangeEntry, RangeIter, ScanStats,
};

// Stats snapshots returned by `Tree::stats` and `DB::stats`.
pub use api::stats::{
    BlobStats, CheckpointerStats, DBStats, JournalStats, OpenStats, RouteCacheStats, StoreStats,
    TreeStats, VacuumStats,
};

// Single-record atomic batches.
pub use api::atomic::{AtomicBatch, Record, RecordVersion};

// Background checkpointer policy. The `Checkpointer` handle itself
// is crate-internal; users tune or disable it via this config.
pub use checkpoint::CheckpointConfig;

// BlobStore trait + bundled stores + zero-copy blob buffer.
// Users implementing a custom `BlobStore` need `BlobGuid` to name
// the blob they're storing.
pub use layout::BlobGuid;
pub use store::blob_store::{AlignedBlobBuf, BlobStore, FileBlobStore, MemoryBlobStore};
