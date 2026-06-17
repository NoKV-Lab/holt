#!/usr/bin/env bash
# Phase B — COLD regime + corrected sync-durable writes.
#   E1-cold : positive cold point read (read-amp + space), all 4 engines
#   E3      : negative (absent) cold point read — per-blob bloom, all 4
#   E2      : cold QD sweep (free device queue depth), holt only
#   E5-sync : sync-durable overwrite/mixed, all 4 (run last; lmdb msync is slow)
#
# holt + rocksdb-direct read O_DIRECT (device, page-cache-independent); lmdb +
# sqlite are page-cache engines, so the harness drops the page cache before
# timing via `sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`. That reuses a
# sudo *timestamp*, which only works from a pty — so launch this script under a
# pty AND detached so it survives disconnect, e.g.:
#   SUDO_PASS=*** nohup setsid script -qfc 'bash eval_cold.sh' /tmp/cold-pty.log &
# Matched 32 MB memory (bufpool=64 → sqlite 32 MB cache, rocksdb 32 MB cache).
set -u
cd /tmp/holt-cr/benches
BIN=$(ls -t target/release/deps/stress-* | grep -v '\.d$' | head -1)
OUT=/tmp/holt-eval
mkdir -p "$OUT"
ALL=holt,rocksdb,sqlite,lmdb
echo "BIN=$BIN"

# Warm the sudo timestamp once (password from env, never written to this file).
# We run under a pty, so child processes (the harness) reuse the ticket via
# `sudo -n`; a keepalive refresh keeps it valid for the whole run.
if [ -n "${SUDO_PASS:-}" ]; then echo "$SUDO_PASS" | sudo -S -v 2>/dev/null; unset SUDO_PASS; fi
( while true; do sudo -n -v 2>/dev/null || exit 0; sleep 50; done ) & KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null' EXIT

grab() { grep -E '^(holt|rocksdb|sqlite|lmdb) |read_amp |space |drop_caches:|settled|holt_shape final' "$1"; }

# E1-cold @1M, routed (holt reads the ~10KB routing region, not a 512KB blob).
echo "=== E1-COLD 1m routed (all 4) ==="
HOLT_STRESS_N=1000000 HOLT_STRESS_POINT_OPS=50000 HOLT_STRESS_LIST_OPS=1 \
HOLT_STRESS_BUFFER_POOL=64 HOLT_STRESS_ROCKSDB_DIRECT=1 HOLT_STRESS_DROP_CACHES=1 \
HOLT_STRESS_COMPACT_AFTER_PRELOAD=1 HOLT_STRESS_REOPEN_AFTER_PRELOAD=1 \
HOLT_STRESS_ENGINES=$ALL HOLT_STRESS_OPS=get \
"$BIN" objstore > "$OUT/cold_e1_1m_routed.log" 2>&1
grab "$OUT/cold_e1_1m_routed.log"

# E3 negative (absent keys) @1M, routed — bloom-reject vs leaf-miss.
echo "=== E3-NEG 1m routed (all 4) ==="
HOLT_STRESS_N=1000000 HOLT_STRESS_POINT_OPS=50000 HOLT_STRESS_LIST_OPS=1 \
HOLT_STRESS_BUFFER_POOL=64 HOLT_STRESS_ROCKSDB_DIRECT=1 HOLT_STRESS_DROP_CACHES=1 \
HOLT_STRESS_COMPACT_AFTER_PRELOAD=1 HOLT_STRESS_REOPEN_AFTER_PRELOAD=1 \
HOLT_STRESS_NEGATIVE=1 HOLT_STRESS_ENGINES=$ALL HOLT_STRESS_OPS=get \
"$BIN" objstore > "$OUT/cold_e3_neg_1m_routed.log" 2>&1
grab "$OUT/cold_e3_neg_1m_routed.log"

# E1-cold @10M, UNROUTED (routed compaction is glacial at 10M) — the honest
# at-scale case where holt reads full 512KB blobs cold.
echo "=== E1-COLD 10m unrouted (all 4) ==="
HOLT_STRESS_N=10000000 HOLT_STRESS_POINT_OPS=50000 HOLT_STRESS_LIST_OPS=1 \
HOLT_STRESS_BUFFER_POOL=64 HOLT_STRESS_ROCKSDB_DIRECT=1 HOLT_STRESS_DROP_CACHES=1 \
HOLT_STRESS_REOPEN_AFTER_PRELOAD=1 \
HOLT_STRESS_ENGINES=$ALL HOLT_STRESS_OPS=get \
"$BIN" objstore > "$OUT/cold_e1_10m_unrouted.log" 2>&1
grab "$OUT/cold_e1_10m_unrouted.log"

# E2 cold QD sweep, holt only @1M routed — free device queue depth via N
# concurrent lock-free gets (holt is O_DIRECT, no drop_caches needed).
for qd in 1 4 8 16; do
  echo "=== E2 qd=$qd 1m routed holt ==="
  HOLT_STRESS_N=1000000 HOLT_STRESS_POINT_OPS=200000 HOLT_STRESS_LIST_OPS=1 \
  HOLT_STRESS_BUFFER_POOL=64 HOLT_STRESS_COMPACT_AFTER_PRELOAD=1 \
  HOLT_STRESS_REOPEN_AFTER_PRELOAD=1 HOLT_STRESS_ENGINES=holt HOLT_STRESS_OPS=get \
  HOLT_STRESS_GET_THREADS=$qd \
  "$BIN" objstore > "$OUT/cold_e2_qd${qd}.log" 2>&1
  grep -E '^holt get|read_amp |settled' "$OUT/cold_e2_qd${qd}.log"
done

# E5-sync: sync-durable overwrite + mixed, warm, all 4 (LAST — lmdb msync slow).
echo "=== E5-SYNC 1m warm (wal_sync=true, all 4) ==="
HOLT_STRESS_N=1000000 HOLT_STRESS_POINT_OPS=50000 HOLT_STRESS_LIST_OPS=1 \
HOLT_STRESS_BUFFER_POOL=1200 HOLT_STRESS_ENGINES=$ALL HOLT_STRESS_OPS=put,mixed \
HOLT_STRESS_WAL_SYNC=true \
"$BIN" objstore > "$OUT/e5_sync_1m.log" 2>&1
grep -E '^(holt|rocksdb|sqlite|lmdb) ' "$OUT/e5_sync_1m.log"

echo ALL_COLD_DONE
