# Holt Soak Harness

`holt-soak` is an explicit durability and lifecycle validation tool.
It is not part of the published crate and is intentionally kept out of
the parent workspace.

## Modes

- `normal`: multi-threaded point read/write/delete, key-only prefix
  scan, atomic batch, checkpoint, reopen, and oracle verification.
- `db-normal`: multi-threaded named-tree DB run with cross-tree atomic
  batches, per-tree point reads, key-only scans, DB views, checkpoint,
  reopen, and oracle verification.
- `crash`: parent process repeatedly starts a child writer, kills it
  with `SIGKILL`, reopens the tree, and verifies every operation the
  child acknowledged in `soak-ack.log`.
- `db-crash`: same kill/reopen protocol as `crash`, but each
  acknowledged operation is one cross-tree `DB::atomic` transaction
  recorded in `soak-db-ack.log`.
- `child`: internal mode used by `crash`.

## Quick Smoke

```sh
cargo run --manifest-path tools/soak/Cargo.toml --locked -- \
  --mode normal \
  --dir target/holt-soak \
  --reset \
  --duration-secs 60 \
  --keys 100000 \
  --ops 1000000 \
  --threads 4 \
  --buffer-pool 64 \
  --wal-sync false
```

## DB Smoke

```sh
cargo run --manifest-path tools/soak/Cargo.toml --locked -- \
  --mode db-normal \
  --dir target/holt-soak-db \
  --reset \
  --duration-secs 60 \
  --keys 100000 \
  --ops 1000000 \
  --threads 4 \
  --buffer-pool 64 \
  --wal-sync false
```

## Crash Campaign

Crash mode requires `--wal-sync true`: the verifier treats the ack log
as the source of acknowledged mutations, so each acknowledged Holt write
must have crossed the WAL durability boundary.

```sh
cargo run --manifest-path tools/soak/Cargo.toml --locked -- \
  --mode crash \
  --dir target/holt-soak-crash \
  --reset \
  --duration-secs 21600 \
  --keys 100000 \
  --ops 1000000 \
  --buffer-pool 64 \
  --wal-sync true \
  --kill-min-ms 100 \
  --kill-max-ms 5000
```

The DB crash campaign uses the same rule, but verifies that each
acknowledged cross-tree atomic transaction replays all of its tree
mutations:

```sh
cargo run --manifest-path tools/soak/Cargo.toml --locked -- \
  --mode db-crash \
  --dir target/holt-soak-db-crash \
  --reset \
  --duration-secs 21600 \
  --keys 100000 \
  --ops 1000000 \
  --buffer-pool 64 \
  --wal-sync true \
  --kill-min-ms 100 \
  --kill-max-ms 5000
```

The tool emits JSON lines with cache, WAL, checkpoint, route-cache, and
reopen-replay counters. CI runs only a short `normal` smoke; longer
normal/crash campaigns belong in nightly or release-gate runs.

## Validation Tiers

- PR CI: build the harness and run short `normal` and `db-normal`
  smokes so API or stats drift is caught quickly.
- Nightly: run `normal`, `db-normal`, `crash`, `db-crash`,
  checkpoint failpoints, WAL integration, and a longer fuzz campaign from
  `.github/workflows/nightly.yml`.
- Release gate: run `normal` for several hours on the target platform,
  then run `crash` with `--wal-sync true`; keep the JSON output so replay
  time, cache misses, WAL debt, and checkpoint debt can be compared
  across releases.

The complete project-level matrix lives in `TESTING.md`.

The crash verifier intentionally checks only acknowledged operations.
The child writes a key, Holt returns success, then the child appends that
key to `soak-ack.log` and fsyncs the ack log. After `SIGKILL`, the parent
reopens Holt and verifies every acknowledged key is present. Operations
that died before the ack log fsync are not part of the durability
contract.
