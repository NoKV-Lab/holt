//! # artisan — adaptive radix tree metadata storage engine
//!
//! `artisan` is an embedded Rust library that stores **path-shaped
//! metadata** with sub-microsecond lookups, per-blob concurrency,
//! and crash-safe persistence. It is built around an Adaptive
//! Radix Tree that spans multiple 512 KB blob frames.
//!
//! See `README.md` for the elevator pitch and `ARCHITECTURE.md`
//! for the deep dive.
//!
//! ## Current status
//!
//! Early development. The crate skeleton is in place; the layout
//! layer (extern-struct types + slot encoding + 4 KB blob header)
//! is complete. The walker / persistence / journal layers are
//! being built out. See `ROADMAP.md`.
//!
//! ## Quick taste (when v0.1 ships)
//!
//! ```ignore
//! use artisan::TreeBuilder;
//!
//! let tree = TreeBuilder::new("/var/lib/myapp/meta.artisan").open()?;
//! tree.put(b"img/01.jpg", b"rgb_data")?;
//! let v = tree.get(b"img/01.jpg")?.unwrap();
//! for entry in tree.range(b"img/").take(10) {
//!     println!("{} -> {}", entry.key_str(), entry.value_str());
//! }
//! # Ok::<(), artisan::Error>(())
//! ```
//!
//! ## Module map
//!
//! - [`layout`] — extern struct layouts (BlobHeader, SlotEntry,
//!   per-NodeType bodies). Each struct has a
//!   `const _: () = assert!(...)` that pins its size + offsets.
//! - [`concurrency`] — `HybridLatch` 3-mode lock + guards.
//! - [`store`] — `BlobFrame` (single-blob ops) + backend trait.
//! - [`engine`] — recursive walker (insert / lookup / erase /
//!   scan / rename / compact).
//! - [`journal`] — WAL + replay + checkpoint.
//! - [`api`] — high-level `Tree` / `Txn` / `Iter` surface.

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

pub mod api;
pub mod concurrency;
pub mod engine;
pub mod journal;
pub mod layout;
pub mod store;

mod prelude_private {
    // Internal helpers shared across modules without exposing
    // them in the public API. Empty for now.
}

// -- Top-level re-exports -----------------------------------------

pub use api::config::{Storage, TreeConfig};
pub use api::errors::{Error, Result};

pub use api::builder::TreeBuilder;
pub use api::tree::Tree;
pub use store::backend::{AlignedBlobBuf, Backend, MemoryBackend, PersistentBackend};
pub use store::BufferManager;
