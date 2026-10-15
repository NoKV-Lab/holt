# 2026-10-15 evidence

`summary.json` contains per-case medians, per-run ranges, per-operation latency,
checkpoint/drain, allocated metadata space, and measured process maximum RSS.

The full raw archives are retained locally and intentionally not committed:
they include per-process desktop environment snapshots and multi-megabyte raw
latency samples. The checked-in reports and summary JSON contain the aggregate
results; the runners regenerate the complete evidence on a fresh run. Archive
checksums remain in `SHA256SUMS` and the corresponding `.sha256` file. A checksum
is provenance for the retained local bundle, not a public download link.

The local `evidence-2026-10-15.tar.gz` contains three groups:

- `comparison`: original 72-process matrix, including the checkpoint-disabled
  pressure reproduction, FULL and attached modes, and the point-read diagnostic.
- `followup`: 40 processes for equal-length checkpoint off/on comparisons and
  the balanced FULL replication after restoring free disk space.
- `native-checkpoint`: 16 processes with SQLite WAL NORMAL, fullfsync enabled,
  native WAL autocheckpoint, and Holt background checkpoint enabled.

Each group includes exact commands, per-operation latency samples in raw JSON,
complete ordered-domain CRC/count/byte fingerprints, resource accounting,
environment snapshots, and generated detailed reports. Binary hashes are in
each environment record; source archives capture the driver used for the
corresponding phase. `SHA256SUMS` checks the evidence files.

Generated test databases are removed after their data/reopen checks and
statistics have been captured. Their bytes are reproducible by running the
preserved command with the associated source and Holt commit. They are not
included in the archive. The current runner removes validated databases by
default; use `--keep-stores` when inspecting their physical topology.

The separate runner QA is excluded from the 128-process result set.

## DB::view and COW optimization

`optimization-2026-10-15/summary.json` and `RESULTS.md` summarize 132 balanced
baseline/patched/SQLite processes. `sync-followup-summary.json` retains the
additional 12-process attached-metadata sync replication. See
`../RESULTS_OPTIMIZATION_2026-10-15.md` for the implementation, caveats, validation
and source provenance.

The local `optimization-evidence-2026-10-15.tar.gz` also includes the raw runs,
frozen baseline source, exact driver, measured library patch, test/soak logs,
and explicitly separated failed preflight and correctness-only stress runs.
Its SHA256 is
`9fef9fed20e86412dee9a741742698868a0114c3b334f69e446ff947a4d9a62e`.
The unredacted environment and validation logs beside the summaries are likewise
local-only; no generated database is committed or included in the archive.
