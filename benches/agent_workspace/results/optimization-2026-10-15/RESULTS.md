# Agent workspace: baseline / patched Holt / SQLite

Unchanged shared workspace handler, workload, dataset, consistency, durability and checkpoint policies. Four independent processes per variant in balanced order; medians of per-process metrics. No concurrent compiler/tests/benchmarks during these runs. Each database is fresh and removed only after full domain fingerprint and clean-reopen validation. Raw samples, commands, resource logs and environment remain alongside this report.

Async OFF is diagnostic, not crash-equivalent to Holt. Background `_normal` uses native checkpoint policies (Holt 500 ms/512 blobs, SQLite NORMAL/1000 pages), not matched checkpoint cadence. Sync uses Holt sync=true and SQLite FULL with macOS fullfsync/checkpoint_fullfsync. These are bounded desktop workloads, not long-running maintenance or power-cut qualification.

| Case | Old H ops/s | New H ops/s | SQLite ops/s | New/old | New/SQLite | Old/new p95 us |
|---|---:|---:|---:|---:|---:|---:|
| attached_content_sync_1 | 401 | 409 | 340 | 1.02x | 1.20x | 9834.4 / 9736.7 |
| attached_metadata_sync_1 | 804 | 780 | 838 | 0.97x | 0.93x | 5711.4 / 5921.2 |
| content_async_1 | 2,217 | 2,191 | 1,707 | 0.99x | 1.28x | 664.4 / 562.7 |
| content_async_4 | 761 | 4,593 | 4,128 | 6.03x | 1.11x | 12930.8 / 959.6 |
| content_async_4_background_normal | 1,746 | 3,596 | 1,860 | 2.06x | 1.93x | 5328.4 / 930.6 |
| content_sync_1 | 390 | 402 | 351 | 1.03x | 1.15x | 9695.6 / 9963.8 |
| metadata_async_1 | 19,674 | 23,188 | 14,952 | 1.18x | 1.55x | 79.9 / 73.4 |
| metadata_async_4 | 226 | 5,826 | 13,725 | 25.76x | 0.42x | 61299.6 / 2611.4 |
| metadata_async_4_background_normal | 1,971 | 22,098 | 1,877 | 11.21x | 11.78x | 7300.4 / 613.8 |
| metadata_async_4_short | 1,003 | 7,102 | 15,133 | 7.08x | 0.47x | 14277.2 / 2194.9 |
| metadata_sync_1 | 774 | 772 | 793 | 1.00x | 0.97x | 5965.5 / 5945.1 |

| Case | Old/new RSS MiB | Old/new metadata MiB | Old/new final checkpoint ms | Old/new post-drain orphan debt |
|---|---:|---:|---:|---:|
| attached_content_sync_1 | 47.6 / 46.3 | 25.1 / 21.4 | 27.7 / 25.9 | 0 / 0 |
| attached_metadata_sync_1 | 47.6 / 48.2 | 20.7 / 21.4 | 46.2 / 46.8 | 0 / 0 |
| content_async_1 | 52.0 / 49.1 | 26.3 / 24.1 | 50.6 / 46.3 | 0 / 0 |
| content_async_4 | 1824.1 / 75.4 | 1774.2 / 28.7 | 2033.3 / 54.8 | 1498 / 0 |
| content_async_4_background_normal | 405.9 / 72.3 | 1110.6 / 27.4 | 32.3 / 39.6 | 572 / 0 |
| content_sync_1 | 47.8 / 45.7 | 23.6 / 21.6 | 18.4 / 15.9 | 0 / 0 |
| metadata_async_1 | 62.6 / 63.9 | 30.2 / 29.2 | 49.6 / 48.2 | 0 / 0 |
| metadata_async_4 | 6800.3 / 107.0 | 8160.1 / 40.7 | 12619.7 / 70.1 | 7850 / 0 |
| metadata_async_4_background_normal | 423.4 / 79.9 | 1430.2 / 26.2 | 159.7 / 38.0 | 898 / 0 |
| metadata_async_4_short | 1388.2 / 87.6 | 1325.4 / 21.2 | 1561.3 / 40.3 | 1068 / 0 |
| metadata_sync_1 | 49.2 / 49.5 | 20.6 / 20.6 | 37.4 / 40.1 | 0 / 0 |
