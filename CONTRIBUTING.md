# Contributing to holt

Thanks for your interest. This document spells out the build / test /
review loop so you can land a change with confidence.

## Build & test

```bash
# clone
git clone https://github.com/feichai0017/holt.git
cd holt

# everything CI runs:
cargo build  --workspace --all-targets
cargo test   --workspace --all-targets
cargo test   --workspace --doc
cargo fmt    --all --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

The CI matrix runs on **ubuntu-latest** and **macos-latest**.
Persistent-backend tests rely on `O_DIRECT` (Linux) / `F_NOCACHE`
(macOS); both work out of the box under the `tempfile` crate that
the integration tests use.

**Minimum supported Rust version: 1.82.** Bump it consciously — the
`msrv` CI job (which builds the library only — dev-dependencies
routinely require a newer toolchain) will fail otherwise.

## Examples

Each example demonstrates a real workload shape:

```bash
cargo run --example basic_kv
cargo run --example filesystem_meta
cargo run --example session_store
cargo run --example s3_metadata
```

All four are in-memory by default so they don't litter your cwd.

## Benchmarks

```bash
cargo bench --bench main                       # ~3 min full sweep
cargo bench --bench main -- --quick --noplot   # ~1 min smoke
cargo bench --bench main -- kv_get             # a single scenario
```

Results land in `target/criterion/`. See `benches/README.md` for the
methodology and the apples-to-apples ground rules vs RocksDB.

## Fuzzing

The repository follows the usual cargo-fuzz layout used by projects
such as sled: a separate `fuzz/` crate that depends on the root
library without joining the normal workspace.

```bash
cargo install cargo-fuzz
rustup toolchain install nightly
cargo +nightly fuzz run atomic_model
```

`atomic_model` drives a persistent tree against a `BTreeMap` oracle,
including reopen/checkpoint, range scans, conditional writes, and
atomic batches. Crash artifacts and generated corpora live under
`fuzz/artifacts/` and `fuzz/corpus/` and are ignored by git.

## Coverage

Use LLVM source coverage for release-facing changes. It exercises the
same lib + integration + example targets as CI and fails if line
coverage drops below the current project floor.

```bash
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
cargo llvm-cov --workspace --all-features --lib --tests --examples --locked --fail-under-lines 88
```

For a browsable report:

```bash
cargo llvm-cov report --html --output-dir coverage
```

## Conventions

### Code

- **Edition: 2021. MSRV: 1.82.**
- **Module layout** follows logical layers: `layout < store < engine
  < journal < api`. New abstractions should slot into the right
  layer, not span them.
- **`#![warn(clippy::pedantic)]`** is on. The vetted noise-allow list
  in [`src/lib.rs`](src/lib.rs) documents which categories we silence
  and why; before adding a new entry, try the local fix first.
- **No `unsafe` outside the `store` and `engine` layers** without a
  written justification. Every `unsafe` block in the crate has a
  `// SAFETY: ...` comment naming the invariant it relies on.
- **Compile-time-pinned layout offsets** — every `#[repr(C)]` layout
  struct in `src/layout/` has `const _: () = assert!(offset_of!(...))`
  blocks. Don't move fields; drift breaks the build, which is the
  point.
- **Walker submodule split** (`engine/walker/{types,readers,writers,
  lookup,insert,erase,spillover,migrate}`) keeps each file under ~700
  lines. Help us keep it that way — if a single file outgrows that,
  carve out a new submodule.

### Tests

- **One test per behaviour.** Tests are documentation; a unit test
  failing should immediately tell you which invariant broke.
- **Property tests** (`tests/properties.rs`) cross-check the tree
  against a `HashMap` oracle. Adding a new op variant? Extend the
  generator + oracle in lockstep.
- **WAL integration tests** (`tests/wal_tree_integration.rs` and
  `src/journal/tests.rs`) cover the crash-and-replay invariants.
  The `durable_cfg` helper enables `wal_sync_on_commit` for tests
  that simulate a crash without a checkpoint.

### Commits

- **One commit per logical change.** No "fix typo + rewrite scheduler
  + bump dep" mega-commits.
- **Subject line ≤ 70 chars**, imperative mood
  (`feat: add WAL group-commit auto-flush`, not "Added a thing").
- **Body wraps at ~72 chars** and answers *why*, not *what*. Diffs
  already show *what*.
- **Prefix tags we use**:
  `feat:` / `perf:` / `fix:` / `refactor:` / `docs:` / `chore:` /
  `test:`. Mixed-purpose commits split.
- **Commit attribution**: don't add tooling co-author trailers; the
  `Co-Authored-By` lines surface in the GitHub UI and crowd out the
  human authorship.

## Filing issues

If you've hit a bug, please include:

- The shortest reproducer you can manage (a failing test is ideal).
- The output of `rustc --version --verbose`.
- Whether you saw it on memory or persistent storage, and the
  config diff from the defaults if any.

For design discussions / "should we add X" questions, open an
issue rather than a PR — the architecture is still being shaped and
direction conversations beat surprise patches.

## Licence

holt is **MIT** licensed. By contributing, you agree your
contribution is licensed under the MIT licence. See
[`LICENSE`](LICENSE) for the legal text.
