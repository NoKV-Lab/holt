# Agent workspace workflow: Holt vs SQLite

This independent, unpublished package runs the same application handler against
Holt and SQLite. It measures request/response JSON, consistent metadata reads,
guarded multi-family mutations, and optional 4 KiB local text-content I/O.
It does not run the NoKV SDK/RPC server or a remote object provider.

The eight metadata families are workspace marker, current path, revision,
object reference, operation, command dedupe, history, and change event. Holt
uses named trees and DB atomic/view; SQLite uses identically encoded values
in eight `WITHOUT ROWID` tables, prepared statements, and read/write
transactions. A publish touches all eight families. Both engines execute
the same condition checks, exact-replay rules, content verification, and
JSON handler. Rename is a roundtrip and remove recreates the original path;
each of these workflows commits twice.

The default mix is 40% read, 15% stat, 15% list, 20% publish, 5% rename
roundtrip, 3% remove+recreate, and 2% bounded consistent snapshot across all
families. Dataset: 8 workspaces × 512 paths. One client uses one workspace;
four clients use independent workspaces sharing one DB and its families.

```sh
cargo build --release --locked --manifest-path benches/agent_workspace/Cargo.toml
python3 benches/agent_workspace/run_comparison.py \
  --binary benches/agent_workspace/target/release/holt-agent-workspace-bench \
  --out /tmp/holt-sqlite-workspace-results
```

The output directory must not already contain same-named run stores. Raw
JSON, per-operation latency samples, exact commands, environment, resource
accounting, domain fingerprints, and a generated report are preserved. Validated
temporary databases are removed between runs; add `--keep-stores` to retain
them. A full suite with retained COW pressure stores can use tens of GiB. Eight
rounds run H,S,S,H / S,H,H,S, giving four independent processes per engine.
Use `--cases metadata_async_1,content_sync_1` for a subset; `--rounds 2` is a
smoke test, not the published comparison.

Holt is the local path dependency; the report records its commit. SQLite is
bundled through the pinned rusqlite dependency, with its runtime version in
each result. Holt has a shared 64 MiB cache target; SQLite has a 64 MiB target
per connection. Total RSS is not matched by these knobs. Results retain actual
maximum RSS, including Holt's protected dirty/COW frames above its cache target.

Async uses Holt `sync=false` and SQLite `synchronous=OFF`. Sync uses Holt
`sync=true` and SQLite `synchronous=FULL`, `fullfsync=ON`, and
`checkpoint_fullfsync=ON`. On macOS, Rust `File::sync_data()` uses
`F_FULLFSYNC`, so SQLite's default `fullfsync=OFF` is not the sync comparison.
Sync content publications sync the content file and parent directory on
both engines. These are configured durability boundaries, not a power-cut
qualification. Base cases disable background checkpointing. Cases marked
`background` enable Holt's native 500 ms/512-dirty-blob policy and SQLite's
1000-page WAL autocheckpoint. `_normal` cases additionally set SQLite WAL
`synchronous=NORMAL`, syncing at checkpoint/log reuse. `OFF` and `NORMAL`
results are kept separate; they differ in crash consistency. Each timed burst
ends with an explicit durable checkpoint, reported separately and also
included in drain-inclusive throughput. Setup and warmup are excluded.

Attached mode writes canonical mutation bytes with the metadata commit:
Holt's attached WAL envelope versus SQLite's transactional recovery table.
The final checkpoint expires this local recovery material. Ordinary and
attached results remain separate. The diagnostic `--read-mode point` omits
cross-tree snapshots for read/stat and must not be presented as having the
default view's consistency semantics.

Every run verifies exact replay causes no second commit, verifies all table
contents on clean reopen, and the runner rejects any cross-engine or
cross-repetition difference in complete ordered domain content. Objects,
history, and receipts are retained; no vacuum is included. This is a bounded
workspace workload, not indefinite retention or maintenance steady-state
evidence.

## Baseline / patched / SQLite regression comparison

`run_ab.py` uses the exact same driver and cases with two frozen binaries.
Twelve forward/reverse matched runs give four processes per variant and
balance each pair's order. It preserves full evidence and rejects mismatched
fingerprints or reopen failures before removing the generated store.

```sh
python3 benches/agent_workspace/run_ab.py \
  --baseline /absolute/path/to/baseline-binary \
  --patched /absolute/path/to/patched-binary \
  --out /tmp/holt-workspace-ab \
  --cases metadata_async_1,metadata_async_4,metadata_async_4_background_normal
```

The output directory must not exist. See
`RESULTS_OPTIMIZATION_2026-10-15.md` for the DB-view/COW optimization results,
sync follow-up, validation boundaries, and remaining costs. The original
Holt/SQLite comparison remains separately preserved.
