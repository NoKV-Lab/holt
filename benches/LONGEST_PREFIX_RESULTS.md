# Longest-Prefix Results

Recorded on 2026-08-18 with:

- Holt based on `v0.9.1`;
- Rust 1.97.1;
- Intel Xeon Platinum 8457C, 2 sockets, 90 physical cores;
- release profile, in-memory tree;
- OrbitKV-shaped keys with 33 bytes per prefix chunk;
- 64 distractor paths per case.

The baseline performs exact `get_record` calls from the deepest chunk boundary
to the shallowest. Native uses one `longest_prefix_record` ART traversal.

## Single Thread

Criterion `--quick` median estimates:

| Chunks | Scenario | Native | Exact fallback | Speedup |
|---:|---|---:|---:|---:|
| 4 | deep hit | 94.8 ns | 301 ns | 3.18x |
| 16 | half hit | 99.8 ns | 5.48 us | 54.9x |
| 64 | shallow hit | 93.0 ns | 95.5 us | 1,027x |
| 64 | miss | 46.8 ns | 2.36 us | 50.5x |
| 256 | deep hit | 428 ns | 11.8 us | 27.6x |
| 256 | shallow hit | 162 ns | 1.47 ms | 9,071x |
| 256 | miss | 121 ns | 38.8 us | 322x |

The 256-chunk tier asserts that the benchmark tree contains at least two Holt
blobs, so those rows include cross-blob traversal.

## Concurrent

The concurrent harness uses a 64-chunk shallow hit and 100,000 operations per
thread. It is pinned with `taskset` to eight physical cores on NUMA node 0:

| Threads | Native ops/s | Exact fallback ops/s | Speedup |
|---:|---:|---:|---:|
| 1 | 14,167,854 | 10,864 | 1,304x |
| 2 | 26,010,681 | 21,419 | 1,214x |
| 4 | 55,808,087 | 43,019 | 1,297x |
| 8 | 107,395,294 | 79,579 | 1,350x |

These are metadata lookup microbenchmarks, not end-to-end inference results.
They exclude Capsule payload I/O, SGLang export/hydration, GPU transfer, and
attention execution.
