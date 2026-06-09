# Holt performance findings

A record of the read/write optimization pass on the `perf/u16-children`
branch: what landed, how holt measures against RocksDB/SQLite on a *fair*
benchmark, and — most importantly — the precisely-diagnosed remaining
bottleneck and the honest architectural tradeoffs. Read this before
chasing further perf work so you don't re-derive it.

## What landed (10 commits)

```
c6062da  bench: drop the holt Tree before its TempDir in persistent groups
9261965  perf(addressing)!: child body offsets — drop slot-table read indirection (2 loads→1/hop; manifest v4)
2845bec  perf(layout)!:   flatten leaf → one variable-size self-describing node (manifest v3; subsumes LeafInline)
4ffe973  perf(walker):    one-byte leaf key fingerprint (skip extent read on key mismatch)
10bfa0c  perf(range):     scan-ahead prefetch of the next sibling subtree
a1852d2  perf(walker):    software-prefetch the next node body during descent (+ aarch64 PRFM)
83b92df  perf(buffer-manager): cheap GUID hasher for the blob cache map (pin −37%)
945b9f9  ③ LeafInline   2ffe29a ① NEON 16-lane scan   abd4e69 ② spillover footprint memo
```

Two breaking on-disk format changes (R3 leaf flatten = v3, R1 offset
addressing = v4). Every step validated on **aarch64 (NEON)** and
**x86 (AVX2 + io_uring)** through the corruption gates: `proptest`
(randomized ops vs a BTreeMap/WAL-replay oracle) and
`checkpoint_failpoint` (crash injection). `④ pointer swizzling` was
**measured and rejected** (≤9–16% ceiling, large concurrency surface)
in favor of the GUID hasher.

## Fair benchmark vs RocksDB / SQLite

Methodology (benches/main.rs, holt-bench crate): **N = 20 000**
object-store metadata keys (~30 B path keys, ~60 B JSON values), spread
across ~7 × 512 KB blobs. Persistent groups run holt
`Durability::Wal { sync: false }` with the **journal + background
checkpointer threads running**, vs RocksDB **WAL on, sync off** — a fair
"hot service" durability profile (WAL to page cache, no per-op fsync).
Numbers from the x86 box (perf_event_paranoid=4, no perf sampling).

| operation (persistent, threads running) | holt | RocksDB | result |
|---|---|---|---|
| point read  (objstore_persist_get) | **210 ns** | 499 ns | **holt 2.4× faster** |
| point read  (memory)               | 219 ns | 487 ns | holt 2.2× |
| write       (objstore_persist_put, 1 thread) | 2.6 µs | 2.58 µs | **parity** |
| write       (memory, no WAL)        | 418 ns | 1397 ns | holt 3.3× (no durability) |
| prefix scan (objstore_list, 100 entries) | 16.35 µs | 15.74 µs | **~parity (within 4%)** |

R1 (offset addressing) was the highest-leverage single change: point
read −10.6%, **prefix scan −24.2%** (closed a 30%→4% gap), writes
unchanged.

### Concurrent write (1M keys, persistent WAL + checkpoint, 16-core x86)

| threads | holt | RocksDB |
|---|---|---|
| 1  | 8938 ns/op (0.11 Mops/s) | 3716 ns/op (0.27) |
| 4  | 2211 ns/op (0.45)        | 2044 ns/op (0.49) |
| 8  | 3089 ns/op (0.32)        | 1435 ns/op (0.70) |
| 16 | 3437 ns/op (0.29)        | 1532 ns/op (0.65) |

holt **peaks at 4 threads then negatively scales**; p99 tail at 16
threads ≈ 296 µs vs RocksDB 56 µs. RocksDB scales to ~0.65–0.70 Mops/s.

## Honest conclusions

**holt is a read engine.** It crushes RocksDB on reads (2.4×, durable or
not) because the ART + 512 KB self-describing blobs give one-load node
hops (post-R1) and subtree locality. That is the product story.

**Writes have two separate problems:**

1. **Architectural (hard to beat): in-place tree vs append-only LSM.**
   A holt `put` costs O(tree depth + route-cache miss + possible
   spillover); at 1M keys that's ~513 blobs, depth 2, ~78% route-cache
   miss, 512 spillovers — so single-thread put is ~2.4× RocksDB at scale.
   RocksDB's LSM append is ~O(1) (append memtable + WAL, defer
   reorganization to background compaction). Chasing LSM on raw write
   throughput fights the architecture; don't.

2. **Fixable: concurrent writes serialize on the root blob's latch.**
   The write path is **lock-coupled with exclusive latches**:
   `cross_and_insert` (src/engine/walker/insert.rs) takes the parent
   blob's `BlobWriteGuard` (exclusive), pins the child, takes the
   child's `BlobWriteGuard`, then drops the parent. So **every write to
   any child blob first exclusively latches the root blob** to traverse
   it → all writers serialize on the root's exclusive latch (classic
   lock-coupled-tree root bottleneck). This is why concurrency scales
   negatively and the tail explodes — and it is *not* architectural; it
   is fixable.

   **Attempted fix (REJECTED — measured regression):** optimistic write
   descent (LeanStore-style optimistic lock coupling) — traverse the
   upper blobs optimistically (snapshot `content_version`, read wait-free,
   validate) exactly like the read path, and take the **exclusive latch
   only on the target blob** where the mutation lands (revalidate the
   parent chain by version, restart on a miss, escalate to pessimistic
   after a 4-restart budget; CoW-fork escalates too). Fully implemented
   and **validated for correctness** (lib +3 tests, concurrent_stress 5,
   proptest 5, checkpoint_failpoint 8, restart counts bounded; aarch64
   20/20 + x86 12/12 hardened-stress loops clean, both arches).

   But a **clean same-machine A/B** (x86, `objstore put`, 50k ops/thread,
   RocksDB reproduced identically both runs → clean attribution) showed
   it is a **large regression at every thread count**, not a win:

   | threads | baseline Mops/s | with descent | Δ | baseline p50 | descent p50 |
   |--------:|----------------:|-------------:|---:|------------:|------------:|
   | 1  | 0.105 | 0.087 | **−17%** | 1.9µs | 4.8µs (2.5×) |
   | 4  | 0.453 | 0.138 | **−70%** | 2.9µs | 28.8µs (10×) |
   | 8  | 0.316 | 0.127 | **−60%** | 6.6µs | 62µs (9×) |
   | 16 | 0.287 | 0.134 | **−53%** | 7.3µs | 92µs (13×) |

   **Why it lost:** the workload *grows* the tree, so node splits and blob
   spillover are frequent. When a target mutation needs to split a node
   across a blob boundary (`TargetMutation::Crossing`) or hits a
   snapshot-shared child (`Escalate`), the optimistic attempt is wasted —
   it descends, takes the target latch, discovers it can't complete,
   restarts, burns the budget, and falls back to the full pessimistic
   path, **paying for both descents**. The −17% / 2.5× single-thread p50
   blowup (zero contention) is the proof: the fast path is mostly wasted
   work here, not a parallelism win. Optimistic descent only pays off when
   target blobs are disjoint *and* the mutation lands in-place (no split);
   for a path-shaped, growing object-store keyspace neither holds often
   enough. Code preserved at `docs/experiments/optimistic-write-descent.patch`;
   do not re-attempt without first making the bail path cheap (skip the
   optimistic attempt when the target node is full / a split is likely)
   or solving the real bottleneck below.

## Write-concurrency bottleneck: ISOLATED to the WAL group-commit path

The negative concurrent-write scaling (4t 0.453 → 16t 0.287 Mops/s, p99
55µs → 286µs) was diagnosed with two zero-/low-risk experiments instead of
guessing — and the answer overturns the earlier speculation (it is NOT the
root latch, and it does NOT need a ROWEX rewrite).

**Profile-by-stats (zero code).** A 16t `objstore put` run's `holt_shape`
line shows: tree `max_depth=2`, and during the measured overwrite phase
`route_hits +800000 / route_misses +0` → **100% route-cache hit, 0%
Phase-3 root-exclusive fallback**. `spillovers` flat (no tree growth, no
`Crossing`). So every put takes the Phase-1 fast path: a *shared* `.read()`
latch on the (single, depth-2) root + an exclusive `.write()` on the child.

**No-merge multi-root A/B (bench-only, `HOLT_SHARD_N`).** Open N independent
`Tree`s over ONE `DB` (each its own root blob, but sharing the DB's
`next_seq` + BufferManager + journal), route puts by `hash(key)%N`, point
ops only — no cross-shard merge. This splits the root latch N ways while
holding everything else constant:

   | config        | 1t | 4t | 8t | 16t |
   |---------------|----|----|----|-----|
   | sh1  (1 root) | 0.116 | 0.458 | 0.321 | **0.290** |
   | sh8  (8 roots)| 0.036*| 0.420 | 0.323 | **0.289** |
   | sh16 (16 roots, depth-1 ⇒ all root-*exclusive*) | 0.396 | 0.341 | 0.345 | **0.307** |

   At 16t every config converges to ~0.29–0.31 regardless of root count
   (+6% from 16× the roots). **The root latch — shared OR exclusive — is
   NOT the bottleneck**, so `prefix-sharded-forest` would not have helped.
   (`*`sh8 1t=0.036 is a one-off p99=688µs checkpoint spike.)

**Memory-mode A/B (bench-only, `HOLT_STORAGE=memory`).** `Storage::Memory`
attaches **no journal** (`wal_path`=None), so the put takes the else-branch
— same ART mutation, same `maintenance_gate`/`mutation_gate`/`endpoint_locks`/
`next_seq`/`mark_dirty`, but **no `commit_gate` + no `journal.submit`**:

   | threads | memory (no journal) | WAL mode | p99 mem | p99 WAL |
   |--------:|--------------------:|---------:|--------:|--------:|
   | 1  | 1.328 | 0.105 | 11µs | 13µs |
   | 4  | 3.768 | 0.453 | 9µs  | 55µs |
   | 8  | 4.567 | 0.316 | 7µs  | 145µs |
   | 16 | **5.782** | 0.287 | 9µs | 286µs |

   Removing the journal makes concurrent writes **scale near-linearly to
   5.78 Mops/s at 16t (20× the WAL-mode 0.287), with flat p99**. The ART
   write path + all three gates + `next_seq` + `mark_dirty` are all still
   present and scale fine.

**Conclusion (measured, not guessed).** The concurrent-write ceiling is
the **WAL group-commit plumbing**, nothing else: per-put `Vec` record
encode + crossbeam channel `send` to a single bounded channel + a single
worker thread draining it (the 286µs p99 = foreground blocked on a full
channel behind the saturated worker). `commit_gate.enter_writer()` is ruled
out — it is `gate.enter_shared()`, the same primitive `mutation_gate`/
`maintenance_gate` use, and those scale. **This overturns the "concede
writes, it's structural" verdict**: holt's *structure* scales (5.78 Mops/s
@ 16t ≈ 9× RocksDB's durable 0.62). Only the WAL plumbing doesn't — and
that is fixable without touching read/scan/cold locality.

**Fix — lock-free shared WAL ring (single ordered log): IMPLEMENTED &
MEASURED, beats RocksDB.** Replaced the per-record `Vec` + channel +
single-encoder-worker with a shared in-RAM ring: each writer reserves a byte
range via one atomic `fetch_add` on the tail (gap-free byte tiling = the
order key), memcpies its encoded record **in parallel**, and publishes by
folding the contiguous published byte interval into `committed_addr` under a
brief lock; a single background flusher drains the committed prefix into the
**unchanged** `WalWriter` (so on-disk format + replay reader are byte-for-byte
identical) and fsyncs on the sync path. ONE ordered log → trim-watermark /
single-pass-replay invariants preserved (unlike the rejected multi-lane
`wal-commit-sharding`). **This is now the sole WAL backend — the legacy
channel+worker has been removed (no feature flag).** See `src/journal/ring.rs`
+ `src/journal/group_commit.rs`, design in `docs/design/wal-ring.md`.

Measured A/B (x86, `objstore put`, 50k ops/thread, same machine; RocksDB the
fixed comparator):

   | threads | legacy Mops/s | **ring Mops/s** | RocksDB | ring vs RocksDB |
   |--------:|--------------:|----------------:|--------:|----------------:|
   | 1  | 0.105 | 0.112 | 0.271 | 0.41× (p50 1.5µs) |
   | 4  | 0.453 | **2.605** | 0.495 | **5.3×** |
   | 8  | 0.316 | **2.049** | 0.701 | **2.9×** |
   | 16 | 0.287 | **1.660** | 0.640 | **2.6×** |

Negative→positive scaling; **5.8–6.5× over legacy, 2.6–5.3× over RocksDB** at
4/8/16 threads; p99 51µs @16t (legacy 286µs). It does not reach the 5.78
memory-mode ceiling — one ordered file still funnels through a single flusher
+ the shared `tail.fetch_add` cacheline + the in-publish `advance` lock (the
4→16t taper). **holt now beats RocksDB on concurrent durable write** while
keeping its read/scan/cold edge.

Validated (both arches, ring LIVE in the engine under the feature): 6 ring
`Journal` contract tests, lib 286, **proptest BTreeMap/WAL-replay oracle 5**,
**checkpoint_failpoint crash-injection 8**, concurrent_stress 3 (+10× release
loop 0 flaky), loom gap-safety model, clippy clean. loom also **caught a real
design bug** (separate work-id counter could disagree with byte order →
unpublished-gap copy) → keyed on the byte tiling instead.

Hardening completed before making it the sole backend: a multi-process
**SIGKILL crash-soak** (`examples/wal_crash_soak.rs`, 40 rounds, every recovery
a contiguous valid prefix — covers the async RAM→page-cache window +
flusher-mid-drain + mid-checkpoint-truncate), a **2+3-publisher loom** model of
the `advance` lock (a leaf lock — no deadlock by construction), and **built-in
backpressure** (writers park on a `space_cv` the flusher signals). The 1-thread
~66ms outlier was root-caused (a per-op flusher wake = a channel send on every
write) and removed → 1t is now 1.12 Mops/s (p99 14.8µs), 4.1× RocksDB.

## Suggested next work (each its own focused session)

- ~~**Lock-free shared WAL ring**~~ — **SHIPPED as the sole WAL backend**
  (legacy removed; see "Write-concurrency bottleneck" above): beats RocksDB
  **2.8–5.5×** on concurrent durable write at 1/4/8/16 threads, dual-arch
  validated (proptest oracle + checkpoint_failpoint + 40-round SIGKILL
  crash-soak + loom). All hardening done.
- ~~**Optimistic write descent**~~ / ~~**prefix-sharded-forest**~~ — both
  **RULED OUT by measurement**: the root latch (shared or exclusive) is
  not the bottleneck (no-merge multi-root A/B: 16 roots ≈ 1 root at 16t),
  so neither helps. Optimistic-descent patch parked in `docs/experiments/`.
- **R2 — BlobNode prefix Bloom** — a per-edge Bloom (a Bloom *extent* in
  the parent, sized ~10 bits/key, not inline) so a negative lookup whose
  key matches a crossing's path prefix is answered without pinning +
  reading the 512 KB child. Targets cold-miss / existence checks; the
  crossing's inline path prefix already filters cross-prefix misses, so
  the marginal win is within-prefix existence misses. Write maintenance
  (update the parent edge on insert) is the cost; Bloom = no false
  negatives, so it is correctness-safe to skip on a miss.
- **Key-ordered leaf layout for cold scans** — compaction's `clone_subtree`
  DFS already lays leaves out in key order, so post-compaction/cold scans
  are already sequential; the remaining hot-scan ~4% is the optimistic
  restart-safe cursor's per-entry copy (diminishing returns).

## Benchmark reproduction notes

- RocksDB/SQLite comparators need libclang; the x86 box has only
  `libclang.so.1`, so symlink a shim and point clang-sys at it:
  `mkdir -p ~/libclang-shim && ln -sf /usr/lib/llvm-18/lib/libclang.so.1 ~/libclang-shim/libclang.so`
  then `export LIBCLANG_PATH=$HOME/libclang-shim`.
- Single-thread latency: `cargo bench --manifest-path benches/Cargo.toml --bench main -- --quick --noplot "objstore_persist_(get|put)/(holt|rocksdb)"`
- Concurrency: `HOLT_CONCURRENT_THREADS=1,4,8,16 HOLT_CONCURRENT_OPS_PER_THREAD=50000 HOLT_CONCURRENT_OPS=put cargo bench --manifest-path benches/Cargo.toml --bench concurrent -- objstore`
