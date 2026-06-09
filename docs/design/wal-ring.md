# Design: lock-free shared WAL ring (concurrent-append group commit)

Status: **SHIPPED — the sole WAL backend** (the legacy channel+worker has been
removed; there is no feature flag). A/B (x86, `objstore put`): concurrent
durable write **2.8–5.5× faster than RocksDB** at 1/4/8/16 threads (≈5–6× over
the old backend), positive scaling, p99 ≤56µs @16t, 1t 1.12 Mops/s. Validated
dual-arch: full lib + integration suite, proptest BTreeMap/WAL-replay oracle,
checkpoint_failpoint crash-injection, concurrent_stress, loom gap-safety
(2 + 3 publishers), and a **multi-process SIGKILL crash-soak** (40 rounds,
every recovery a contiguous valid prefix). Built-in backpressure parks on a
condvar. Note: the realized design keys on the **byte tiling**, not a separate
work-id (loom caught the work-id/byte-order disagreement — see below).

## Problem (measured, not assumed)

Concurrent durable writes scale negatively (4t 0.453 → 16t 0.287 Mops/s, p99
55µs → 286µs). Two diagnostics (commit `309d0be`, see `PERF_FINDINGS.md`)
isolated the cause:

- **No-merge multi-root A/B** (`HOLT_SHARD_N`): 1 / 8 / 16 independent roots
  all converge to ~0.29 Mops/s @16t → the root latch (shared *or* exclusive)
  is **not** the bottleneck; `prefix-sharded-forest` would not help.
- **Memory-mode A/B** (`HOLT_STORAGE=memory`, no journal): writes scale
  near-linearly to **5.78 Mops/s @16t (20× WAL-mode), flat p99** → the
  bottleneck is the **WAL group-commit plumbing**, nothing else (`commit_gate`
  is `gate.enter_shared`, same primitive the scaling gates use).

The plumbing today (`src/journal/group_commit.rs`): each put allocates a
per-record `Vec`, encodes into it, `submit()`s it down a **single crossbeam
channel** to a **single worker** that is the sole producer of bytes into the
WAL file. The 286µs p99 is foreground threads blocked on a full channel
behind the saturated single worker.

## Goal

Let N threads append to **one ordered log** concurrently, preserving **every**
durability/crash/recovery invariant. Explicitly NOT the multi-lane WAL
sharding (rejected: breaks the global trim watermark + silently corrupts
cross-lane Rename/Batch).

## Rejected alternatives

- **mmap-as-file** (conf 0.46, but `durability_safe=false`, `crash_safe=false`):
  a pre-`ftruncate`'d mmap makes file-size ≠ valid-data length, and the reader
  scans `while offset < bytes.len()` treating a magic mismatch as **hard
  corruption** (`reader.rs`) → recovery bricks on the zero tail. Worse,
  `MAP_SHARED` hands dirty pages to the kernel, whose writeback can flush a
  reserved-but-unpublished garbage page **independent of the flusher**, and a
  single `fdatasync` gives durability but **not inter-page ordering** → a
  durable EOF marker can outrun durable data. Changes the format+reader
  boundary. Rejected.
- **Per-thread sharded staging keyed on `next_seq`** (conf 0.18,
  `ordering_safe=false`): `next_seq` is the **gappy MVCC version** (burned on
  failed-guard `Ok(false)`, rename early-returns, batch range-burn — the
  *common* path), so a contiguous prefix keyed on it stalls forever on routine
  early returns → permanent durability freeze. It also forces file-order ==
  seq-order, a *stronger* linearization than the system produces today.
  Rejected.

## Chosen design: in-RAM byte ring + dense work-id-keyed committed prefix

Variant 1's in-RAM byte ring (one `fetch_add` reservation, parallel memcpy of
already-encoded records, single flusher copying a contiguous prefix into the
**UNCHANGED** `WalWriter::append_encoded`), with the fatal flaws excised by the
**one load-bearing decision**:

> Key the committed-prefix watermark and ALL checkpoint watermarks on the
> **byte tiling** — `tail.fetch_add(len)` hands out gap-free contiguous ranges
> (each start == the previous end), so the byte address *is* the dense,
> gap-free order key. Fold *published byte intervals* into `committed_addr`,
> synchronously, at the reservation point inside `commit_gate.enter_writer` —
> never on `next_seq` (gappy) and never on the *reserved* `tail` (decouples
> reservation from publish → breaks W2D).

> **Stage-1 correction (loom-driven).** The synthesized spec keyed the prefix
> on a *separate* dense `work_alloc` counter. Implementing it (stage 1) and
> running the loom model **caught the flaw the spec flagged as open question
> #3**: `work_alloc` and `tail` are independent `fetch_add`s whose orders can
> disagree, so folding by work-id can advance `committed_addr` over a byte
> range whose lower bytes aren't published yet → the contiguous-byte flusher
> copies an unpublished gap (silent corruption; CRC can't catch it). Keying on
> the byte tiling uses the **one** natural order and is immune — and is simpler
> (no second counter). File order = byte order = a valid linearization
> (conflicting same-key ops are ordered by the endpoint lock → ordered `tail`
> alloc). All W2D / no-stall properties below hold identically in the byte
> domain. See `src/journal/ring.rs` (behind the `wal_ring` feature).

On-disk format (`codec.rs`), `WalWriter` (`writer.rs`), and the replay reader
(`reader.rs`) are **byte-for-byte untouched**. `FORMAT_VERSION` stays 3.

### Why this is safe (defuses each killer objection)

- **W2D (no trim past un-fsynced data).** `publish()` folds the work-id into
  `committed_work` *synchronously while the writer still holds
  `commit_gate.enter_writer`*. Checkpoint captures `queued_work() ==
  committed_work` under `enter_checkpoint`, which waits for all shared writer
  guards to drop. So at capture, `committed_work ≥` the work-id of every record
  whose blob is in the dirty snapshot — `flush_up_to(committed_work)` targets a
  true contiguous, fully-written prefix. An in-flight writer that reserved but
  hasn't published has **not** released the gate, so `enter_checkpoint` is still
  waiting on it; once it releases, its work-id is already folded. (This is the
  exact equivalence today's `queued_work`-after-`tx.send`-inside-gate provides.)
- **No pervasive prefix stall.** The work-id is allocated *inside* `reserve()`,
  at the same program point as today's `journal.submit` (after the mutation
  succeeded, a record exists). Every allocated work-id maps to a real record
  that will be published. `next_seq`'s gaps (failed guards, early returns) are
  irrelevant — they never touch `work_alloc`.
- **Gap-safety (the silent-corruption class).** The flusher only copies
  `[flush_cursor, committed_addr)`, and `committed_work` advances **only over a
  contiguous run of published work-ids**, folded under `advance_lock`. The
  memcpy→`advance_lock` unlock(Release)→`committed_work.store`(Release)→flusher
  `committed_work.load`(Acquire)→flusher reads bytes chain guarantees the
  flusher observes every byte before copying it. **CRC will NOT catch an
  out-of-order gap copy** (each record is individually CRC-valid) → must be
  proven by loom model-checking, not relied on at runtime.
- **Reopen signal.** Seed watermarks from `WalWriter::has_records()` exactly as
  today (`needs_checkpoint == committed_work != checkpointed_work`).
- **Ordering unchanged.** File order == ascending work-id order. Conflicting
  same-key ops are ordered by the endpoint lock held across both `next_seq`
  alloc and reserve+publish → later op gets later work-id → later file offset.
  Identical to today's guarantee; we do NOT force global file-order==seq-order.

## Protocols (atomics + memory orderings)

New file `src/journal/ring.rs`. `WalRing` holds: `buf: Box<[u8]>` (pow2,
default 16 MiB), `tail` (logical byte addr, monotone, `&mask` for physical),
`work_alloc`, `committed_work`/`committed_addr`, `flush_cursor`,
`written_work`/`flushed_work`/`checkpointed_work`, `sync_target` +
`sync_park` condvar, `advance_lock: Mutex<AdvancerState{ pending:
BTreeMap<u64,(start,end)>, next_work }>`. All hot atomics `CachePadded`.

- **reserve()** [writer, inside `commit_gate.enter_writer`]:
  `work_id = work_alloc.fetch_add(1, Relaxed)+1`; `start = tail.fetch_add(
  total_len, Relaxed)` (sole byte-order-assigning op). `total_len =
  17 + body_len + 4`. Work-id order and byte order need not agree — flusher
  keys on work-id, copies each intent's stored disjoint range.
- **backpressure** [after reserve, before memcpy]: spin/park-with-timeout until
  `ticket.end ≤ flush_cursor.load(Acquire) + capacity` (gates on
  `flush_cursor` = copied-to-WalWriter, decoupled from fsync latency).
- **memcpy** [writer]: plain stores into `buf[start&mask ..]`, split on wrap.
- **publish()** [writer, STILL inside gate]: `advance_lock.lock()`;
  `pending.insert(work_id, (start,end))`; greedily fold `while pending.remove(
  &next_work)`: `committed_addr.store(end, Release)`,
  `committed_work.store(next_work, Release)`, `next_work += 1`. Unlock (Release).
  Unpark flusher. If sync: `sync_target.fetch_max(work_id, Release)`.
- **flusher copy pass** [single thread]: read `committed_work`/`committed_addr`
  (Acquire); if `== flush_cursor` idle; clamp `target_addr = min(committed_addr,
  flush_cursor + 256KB)` (matches `AUTO_FLUSH_THRESHOLD`); lock `WalWriter`,
  `append_encoded` the `[flush_cursor, target_addr)` slice (twice on wrap),
  unlock; `flush_cursor.store(Release)`, `written_work.store`; unpark
  backpressured writers; on sync window: `flush()` (`sync_data`),
  `flushed_work.store(Release)`, `sync_park.notify_all()`. Group-commit window
  (200µs) preserved for sync coalescing.
- **sync ack** [writer, after releasing gate, mirrors `tree.rs:749`]:
  `SyncWaiter{target_work}.wait()` blocks on `flushed_work ≥ target_work` via
  `sync_park` (replaces the per-op `bounded(1)` ack channel).
- **truncate** [flusher cmd, when pipeline clean per `round.rs:156`]: require
  `flush_cursor == committed_addr == tail` (ring drained); `WalWriter::truncate`;
  reset byte addrs to 0; **do NOT reset `work_alloc`** (work-id stays a stable
  global order across truncations); `checkpointed_work.fetch_max` after ack.
- **oversize** (`total_len > capacity`, rare — blobs ≤512KB, ring 16MiB): cold
  `Mutex`-serialized direct `append_encoded` with a synthetic full-range intent.

## Staged, independently-testable plan (flag `wal_ring`, default off)

1. `ring.rs` single-producer + single-flusher; unit + **loom** of
   reserve→memcpy→publish→advance→copy; tiny capacity forces wrap; assert
   `committed_work` dense, wrapped copy == linear copy. (Not wired to Journal.)
2. Wire into `Journal` behind flag; `submit→reserve+memcpy+publish`; flusher
   replaces `run_worker`; `queued_work()=committed_work`; reopen seeding.
   Validate: existing `group_commit.rs` suite passes with flag ON
   (`reopened_nonempty_wal_still_needs_checkpoint`); **golden-file diff** vs
   legacy path (byte-identical WAL).
3. Multi-writer: N concurrent reservers + `advance_lock` fold of out-of-order
   intents. **loom**: contiguous-only advance; flusher never copies a
   reserved-but-unpublished gap (inject parked-before-publish writer → assert
   `committed_work` stalls at id−1, resumes after publish); proptest
   conflicting same-key → seq order.
4. Sync path: `SyncWaiter` + `sync_park` + group-commit window; error fan-out.
   Validate: M sync writers in one window share one `sync_data`; async never
   blocks.
5. Backpressure: gate on `flush_cursor+capacity`; oversize fallback. Validate:
   parked mid-memcpy writer doesn't wedge reclamation; backpressure-vs-truncate
   no deadlock.
6. Checkpoint integration: `round.rs:226` capture `committed_work`; truncate
   reset; `debug_assert(observed_work ≤ committed_work)`. Validate:
   `checkpoint_failpoint` — capture under `enter_checkpoint` with concurrent
   in-gate writer; W2D assertion per snapshotted blob.
7. SIGKILL crash-soak matrix (reserved-not-published / published-not-flushed /
   flushed-fsynced / mid-truncate) + A/B benchmark 1/4/8/16t async & sync.

## Key invariants to test

PREFIX-DENSITY · GAP-SAFETY (loom, **CRC won't catch it**) · W2D-AT-CAPTURE ·
FLUSH-UP-TO-NON-BLOCKING · WORK-ID-DENSITY-VS-NEXT_SEQ-GAPS · WATERMARK-MONOTONE
(`work_alloc ≥ committed ≥ written ≥ flushed ≥ checkpointed`) · REOPEN-SIGNAL ·
FORMAT-IDENTICAL (golden diff) · SYNC-GROUP-COMMIT · BACKPRESSURE-LIVENESS ·
TRUNCATE-RING-CLEAN · SIGKILL-CRASH-SOAK · ORDERING-LINEARIZATION.

## Open questions (decide/probe before or during the relevant stage)

- **`advance_lock` contention** (the main risk): `publish()` takes a `Mutex`
  briefly inside the gate; under 16+ publishers this could become the new
  serialization point. Constraint: `committed_work` MUST advance for a writer's
  own record *before* it releases the gate (W2D) — so a flusher-only advancer is
  not acceptable. Decide structure (`Mutex<BTreeMap>` vs sharded `done[]` array
  keyed by work-id ordinal) at stage 3 with loom + microbench.
- Ring capacity default (16 MiB?) — probe backpressure-park rate at 16t.
- work-id-order vs tail-order disagreement — loom-confirm copying ascending
  work-id (non-contiguous byte) ranges yields correct file order.
- cargo feature vs runtime `TreeConfig` flag for A/B (lean: feature first).

## Honest expected payoff

- **Async durable (`wal_sync=false`)**: the real win. Removes the per-record
  `Vec` + single channel + single-byte-producer. Expect **~2–4 Mops/s @16t**
  (≈7–14× the current 0.29; **3–6× RocksDB's 0.62**). Will NOT reach the 5.78
  memory-mode ceiling — one ordered file still funnels through a single flusher
  + a shared `tail.fetch_add` cacheline + the in-gate fold. Realistic: close
  ~half-to-two-thirds of the 0.29→5.78 gap.
- **Sync durable (`wal_sync=true`)**: fsync-bound by design; group-commit
  coalescing preserved, so expect lower per-op CPU + tail-latency variance, NOT
  a throughput multiple.

Effort ~2–3 engineer-weeks; low integration risk because format/recovery and
`WalWriter` are provably unchanged, so the existing replay + checkpoint suites
are a strong regression net. The genuinely hard, model-check-required parts are
GAP-SAFETY and the `advance_lock` structure.
