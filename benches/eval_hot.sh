#!/usr/bin/env bash
# Phase A — HOT regime (E1 read scale, E4 list/rollup, E5 overwrite, mixed).
# All four engines warm: holt buffer-pool sized to hold the dataset; the
# page-cache engines (lmdb/sqlite) + buffered rocksdb ride the OS page cache.
set -u
cd /tmp/holt-cr/benches
BIN=$(ls -t target/release/deps/stress-* | grep -v '\.d$' | head -1)
OUT=/tmp/holt-eval
mkdir -p "$OUT"
ENG=holt,rocksdb,sqlite,lmdb

run() { # n bufpool tag
  local n=$1 bp=$2 tag=$3
  echo "=== HOT scale=$n bufpool=$bp engines=$ENG ==="
  HOLT_STRESS_N=$n \
  HOLT_STRESS_POINT_OPS=500000 \
  HOLT_STRESS_LIST_OPS=50000 \
  HOLT_STRESS_BUFFER_POOL=$bp \
  HOLT_STRESS_ENGINES=$ENG \
  HOLT_STRESS_OPS=get,put,mixed,list,list_dir \
  HOLT_STRESS_WAL_SYNC=false \
  "$BIN" objstore > "$OUT/hot_${tag}.log" 2>&1
  echo "--- hot_${tag} ---"
  grep -E '^(holt|rocksdb|sqlite|lmdb) ' "$OUT/hot_${tag}.log"
}

run 100000   256  100k
run 1000000  1200 1m
run 10000000 8192 10m

# Sync-durable write profile (E5 sync) at 1M only — bounds the run.
echo "=== HOT-SYNC scale=1000000 bufpool=1200 (wal_sync=true) ==="
HOLT_STRESS_N=1000000 HOLT_STRESS_POINT_OPS=200000 HOLT_STRESS_LIST_OPS=1 \
HOLT_STRESS_BUFFER_POOL=1200 HOLT_STRESS_ENGINES=$ENG \
HOLT_STRESS_OPS=put,mixed HOLT_STRESS_WAL_SYNC=true \
"$BIN" objstore > "$OUT/hot_sync_1m.log" 2>&1
grep -E '^(holt|rocksdb|sqlite|lmdb) ' "$OUT/hot_sync_1m.log"

echo ALL_HOT_DONE
