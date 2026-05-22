# Profile-Guided Optimization

`rustc` supports profile-guided optimization (PGO): record an
instrumented run against a representative workload, then rebuild
with feedback from the recorded profile to let the compiler
reorder branches, inline calls, and lay out code based on the
actual hot path.

For holt this matters most on the **read fast path** — the
`Tree::get` walker is a tight chain of `ntype_of` →
`body_of_slot` → SIMD scan → recurse. Branch prediction and
inlining decisions are often a visible slice of total cycles on
microbenchmarks. Treat PGO as a deployment-local tuning tool; the
v0.3 release numbers in `benches/RESULTS.md` are plain release
builds, not PGO-trained binaries.

## Setup

The Rust toolchain ships PGO support natively; no extra crates
needed. The [`cargo-pgo`][cargo-pgo] wrapper drives the
two-stage build cleanly.

```bash
rustup component add llvm-tools-preview
cargo install cargo-pgo
```

[cargo-pgo]: https://github.com/Kobzol/cargo-pgo

## Two-stage build

### 1. Instrumented build + training run

```bash
# Build with instrumentation; produces a binary that records
# `.profraw` files in `./target/pgo-profiles/` as it runs.
cargo pgo build

# Drive the instrumented binary through a representative
# workload. Use the bench binary so we exercise the same
# call shapes that release builds care about.
cargo pgo bench -- --bench main
```

`cargo pgo bench` emits one `.profraw` per criterion sample into
`target/pgo-profiles/`. Aggregate them into a single profile:

```bash
cargo pgo optimize merge
```

### 2. Optimized rebuild

```bash
# Reads the merged profile from `target/pgo-profiles/merged.profdata`
# and rebuilds with `-Cprofile-use=...`.
cargo pgo optimize build --release
```

The resulting `target/release/<binary>` is the PGO build. Run
the benches against it the same way you would with a normal
release build:

```bash
cargo pgo optimize bench -- --bench main
```

## Expected gains

PGO gains are workload- and compiler-version-dependent. The table
below is a yardstick for downstream measurement, not a published
holt claim:

| Workload pattern             | Typical PGO Δ | Why                                              |
| ---------------------------- | ------------- | ------------------------------------------------ |
| Walker `Tree::get` hot path  | -5 to -15 %   | Tight `ntype_of → body_of_slot → SIMD scan → recurse`; branch + inline decisions can dominate once data is hot |
| Range / list-style scans     | -5 to -12 %   | More uniform control flow, SIMD step inner loop already saturates |
| Write paths (`put` / spillover) | -5 to -8 %   | Spillover allocates + memcpys 512 KB blobs; LLVM PGO can't reorder a `memcpy` |
| WAL-fsync-bound workloads    | ≈0            | End-to-end latency is `sync_data`; CPU savings are rounding error |

If you publish numbers from your own runs, replace this table
with concrete measurements and the criterion baseline you
compared against (`cargo bench --bench main -- --baseline release`).

The public bench harness lives at `benches/main.rs` and covers KV,
object-store metadata, and filesystem metadata workloads against
RocksDB and SQLite. See `benches/README.md` for the exact scenario
matrix and `benches/RESULTS.md` for the v0.3 Linux release run.

## When PGO doesn't help

- **WAL-fsync-bound workloads** (`WalCommit::Sync`):
  end-to-end latency is dominated by `sync_data`, so the walker
  CPU time PGO saves is rounding error.
- **Bulk writes** that trigger spillover: dominated by 512 KB
  blob memcpy, which `glibc` / `libc++` already SIMD-optimize.
- **PGO-instrumented builds in CI**: the instrumentation adds
  ~3× overhead. Don't ship instrumented binaries; only use the
  optimized rebuild downstream.

## Profile staleness

The PGO profile must reflect realistic call ratios. Re-train
when:

1. The workload mix shifts (e.g. read:write ratio changes).
2. The walker or BM hot paths land a significant refactor —
   stale profiles can mis-inline.
3. The Rust toolchain bumps a major version (LLVM version
   change can invalidate the profile format).

Empirically a profile from the previous quarter is still good
for ±5 % of the fresh-profile gains.
