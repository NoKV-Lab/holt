<!--
Thanks for sending a PR! A few checks before you click "Open":

- [ ] The change is a single logical commit (split mixed-purpose
      diffs per CONTRIBUTING.md).
- [ ] `cargo fmt --all --check` passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes.
- [ ] `cargo test --workspace --lib --tests --examples` passes.
- [ ] `cargo +nightly fuzz run atomic_model -- -runs=512` passes when fuzz-sensitive paths changed.
- [ ] `cargo llvm-cov --workspace --all-features --lib --tests --examples --locked --fail-under-lines 88` passes when coverage-sensitive paths changed.
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` passes.
- [ ] If you touched a `unsafe` block, the `// SAFETY: ...` comment
      still describes the invariant correctly.
- [ ] If you added a public API, it has rustdoc (the crate sets
      `#![deny(missing_docs)]`).
- [ ] If you changed on-disk layout, the compile-time
      `assert!(offset_of!(...))` blocks still match.

For non-trivial changes, please open an issue first — the
architecture is being shaped and we want to avoid churn on
already-rejected designs.
-->

## What changed

<!-- One short paragraph. The diff already shows *what*; this
     section says *why*. -->

## Test plan

<!-- New tests added? Existing tests that prove this works?
     Bench impact (if any)? Crash-and-replay coverage if you
     touched the WAL or persistence path? -->

## Related

<!-- Linked issues, ROADMAP items, prior PRs / commits. -->
