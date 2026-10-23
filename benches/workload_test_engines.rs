//! Workload-test-only engine adapters, shared by the `ycsb` and
//! `ai_meta` benches.
//!
//! The `Engine` trait here is a measurement surface, not a
//! general-purpose storage abstraction: it is single-threaded
//! (`&mut self`), panics on engine errors, counts-and-discards scan
//! results, and exposes exactly the operations the workloads time —
//! no rename, no conditional batches, no key-only or cursored scans,
//! no version handles. Extend it only when a workload needs the op;
//! do not build on it as an application interface.
//!
//! Every engine runs the same persistent async-durable profile as the
//! stress harness: file-backed, WAL/journal on, no per-op fsync. A
//! committed op survives a process crash, not a power cut:
//!
//! - holt:     `Durability::Wal { sync: false }`, background checkpointer on
//! - rocksdb:  WAL on, `sync = false`, compression off
//! - sqlite:   `journal_mode=WAL`, `synchronous=OFF`, WITHOUT ROWID table
//! - sled:     `HighThroughput`, background flush off during timing
//! - lmdb:     `NO_SYNC | NO_META_SYNC` (mmap page-cache durability)
//! - redb:     `Durability::Eventual` commits
//! - postgres: `synchronous_commit = off` (server-side WAL, background sync)
//!
//! Each engine's private cache is capped to the same `cache_mb` budget
//! where the engine exposes a knob. LMDB is mmap-backed, so its working
//! set lives in the OS page cache and is NOT capped — a known caveat
//! carried over from the stress harness.
//!
//! PostgreSQL is a client/server engine behind a unix socket; its rows
//! carry per-op protocol round-trips the embedded engines do not pay.
//! It is included as the external-database baseline, not as an
//! embedded-engine peer.
#![allow(dead_code)]

use std::hint::black_box;
use std::path::Path;

use holt::{Tree, TreeConfig};

#[cfg(feature = "comparators")]
use heed::types::Bytes;
#[cfg(feature = "comparators")]
use heed::{Database as LmdbDatabase, Env as LmdbEnv, EnvFlags, EnvOpenOptions};
#[cfg(feature = "comparators")]
use rocksdb::{BlockBasedOptions, Cache, Options as RocksOptions, WriteBatch, WriteOptions, DB};
#[cfg(feature = "comparators")]
use rusqlite::{params, Connection, OptionalExtension};
#[cfg(feature = "comparators")]
use sled::{Db as SledDb, Mode as SledMode};
#[cfg(feature = "comparators")]
use std::ops::Bound;

pub const ALL_ENGINES: &[&str] = &[
    "holt", "rocksdb", "sqlite", "sled", "lmdb", "redb", "postgres",
];

#[derive(Debug, Clone)]
pub struct Opts {
    /// Per-engine private cache budget, in MiB.
    pub cache_mb: usize,
    /// LMDB map-size reservation, in GiB (sparse, virtual-only).
    pub map_gb: usize,
    /// PostgreSQL conninfo string (`HOLT_PG`), e.g.
    /// `host=/tmp/holt-pg port=54329 user=bench dbname=bench`.
    pub pg_conn: Option<String>,
}

impl Opts {
    pub fn from_env() -> Self {
        Self {
            cache_mb: env_usize("HOLT_BENCH_CACHE_MB", 256),
            map_gb: env_usize("HOLT_BENCH_MAP_GB", 32),
            pg_conn: std::env::var("HOLT_PG").ok(),
        }
    }
}

pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

/// One put/delete batch, applied atomically where the engine supports it.
#[derive(Default)]
pub struct Batch {
    pub puts: Vec<(Vec<u8>, Vec<u8>)>,
    pub deletes: Vec<Vec<u8>>,
}

pub trait Engine {
    fn name(&self) -> &'static str;
    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>>;
    fn put(&mut self, key: &[u8], value: &[u8]);
    fn delete(&mut self, key: &[u8]);

    /// Ordered scan of up to `limit` key/value entries with key > `start`
    /// (exclusive start, same semantic for every engine). Returns entries
    /// seen.
    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize;

    /// Bounded key/value prefix scan. Returns entries seen.
    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize;

    /// S3-style delimiter rollup under `prefix`: up to `limit` distinct
    /// entries (keys or CommonPrefix rollups). holt uses the native
    /// delimiter API; comparators use ordered iteration with the
    /// fast-forward seek the stress harness established.
    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize;

    /// Atomic multi-put/multi-delete.
    fn batch(&mut self, batch: &Batch);

    /// Conditional overwrite of an existing key (read, then write).
    /// holt overrides this with its native versioned `compare_and_put`;
    /// the generic form is the read-then-write a KV-store app would do.
    fn cas(&mut self, key: &[u8], value: &[u8]) -> bool {
        if self.get(key).is_some() {
            self.put(key, value);
            true
        } else {
            false
        }
    }

    /// Longest stored prefix of `query`, where `probes` holds the
    /// candidate key lengths in ascending order (chain-node boundaries).
    /// Returns the matched byte length, 0 on total miss. The generic
    /// form walks forward point-get by point-get until the first miss —
    /// what a chained KV-cache index (Mooncake/LMCache-style) does on a
    /// flat KV store with a prefix-closed key set. holt overrides it
    /// with the native single-descent `longest_prefix_record`.
    fn longest_prefix(&mut self, query: &[u8], probes: &[usize]) -> usize {
        let mut matched = 0usize;
        for &p in probes {
            if self.get(&query[..p]).is_some() {
                matched = p;
            } else {
                break;
            }
        }
        matched
    }

    /// Settle the preload (checkpoint/flush barrier) so timed phases
    /// start from a warm, drained service, like the stress harness.
    fn finish_load(&mut self) {}
}

pub fn open_engine(name: &str, dir: &Path, opts: &Opts) -> Box<dyn Engine> {
    match name {
        "holt" => Box::new(HoltEngine::open(dir, opts)),
        #[cfg(feature = "comparators")]
        "rocksdb" => Box::new(RocksEngine::open(dir, opts)),
        #[cfg(feature = "comparators")]
        "sqlite" => Box::new(SqliteEngine::open(dir, opts)),
        #[cfg(feature = "comparators")]
        "sled" => Box::new(SledEngine::open(dir, opts)),
        #[cfg(feature = "comparators")]
        "lmdb" => Box::new(LmdbEngine::open(dir, opts)),
        #[cfg(feature = "comparators")]
        "redb" => Box::new(RedbEngine::open(dir, opts)),
        #[cfg(feature = "pg")]
        "postgres" => Box::new(PgEngine::open(opts)),
        #[cfg(not(feature = "pg"))]
        "postgres" => panic!(
            "postgres requires `--features pg` and a running server; \
             set HOLT_PG to the conninfo string"
        ),
        other => panic!("unknown engine `{other}`; known: {ALL_ENGINES:?}"),
    }
}

// --------------------------------------------------------------------
// shared key math (same as the stress harness)
// --------------------------------------------------------------------

pub fn key_successor(key: &[u8]) -> Vec<u8> {
    let mut upper = key.to_vec();
    let last = upper.last_mut().expect("key must be non-empty");
    *last = last.checked_add(1).expect("key last byte must be < 0xff");
    upper
}

pub fn prefix_upper(prefix: &[u8]) -> Vec<u8> {
    key_successor(prefix)
}

pub fn delimiter_successor(key: &[u8], prefix_len: usize, delim: u8) -> Option<Vec<u8>> {
    let rest = &key[prefix_len..];
    let idx = rest.iter().position(|b| *b == delim)?;
    Some(key_successor(&key[..=prefix_len + idx]))
}

// --------------------------------------------------------------------
// holt
// --------------------------------------------------------------------

pub struct HoltEngine {
    tree: Tree,
}

impl HoltEngine {
    fn open(dir: &Path, opts: &Opts) -> Self {
        let mut cfg = TreeConfig::new(dir);
        cfg.durability = holt::Durability::Wal { sync: false };
        // buffer_pool_size counts 512 KiB frames.
        cfg.buffer_pool_size = (opts.cache_mb * 2).max(8);
        Self {
            tree: Tree::open(cfg).expect("holt open"),
        }
    }
}

impl Engine for HoltEngine {
    fn name(&self) -> &'static str {
        "holt"
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.tree.get(key).expect("holt get")
    }

    fn put(&mut self, key: &[u8], value: &[u8]) {
        self.tree.put(key, value).expect("holt put");
    }

    fn delete(&mut self, key: &[u8]) {
        self.tree.delete(key).expect("holt delete");
    }

    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize {
        let mut seen = 0usize;
        for entry in self.tree.range().start_after(start) {
            black_box(&entry.expect("holt scan_from"));
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize {
        let mut seen = 0usize;
        for entry in self.tree.range().prefix(prefix) {
            black_box(&entry.expect("holt scan_prefix"));
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize {
        let mut seen = 0usize;
        self.tree
            .scan_keys(prefix)
            .delimiter(delim)
            .visit(limit, |entry| {
                black_box(&entry);
                seen += 1;
                Ok(())
            })
            .expect("holt list_dir");
        seen
    }

    fn batch(&mut self, batch: &Batch) {
        self.tree
            .atomic(|b| {
                for (k, v) in &batch.puts {
                    b.put(k, v);
                }
                for k in &batch.deletes {
                    b.delete(k);
                }
            })
            .expect("holt atomic batch");
    }

    fn cas(&mut self, key: &[u8], value: &[u8]) -> bool {
        let Some(record) = self.tree.get_record(key).expect("holt cas read") else {
            return false;
        };
        self.tree
            .compare_and_put(key, record.version, value)
            .expect("holt compare_and_put")
    }

    fn longest_prefix(&mut self, query: &[u8], _probes: &[usize]) -> usize {
        self.tree
            .longest_prefix_record(query)
            .expect("holt longest_prefix_record")
            .map(|r| r.key.len())
            .unwrap_or(0)
    }

    fn finish_load(&mut self) {
        self.tree.checkpoint().expect("holt preload checkpoint");
    }
}

// --------------------------------------------------------------------
// rocksdb
// --------------------------------------------------------------------

#[cfg(feature = "comparators")]
pub struct RocksEngine {
    db: DB,
    wo: WriteOptions,
}

#[cfg(feature = "comparators")]
impl RocksEngine {
    fn open(dir: &Path, opts: &Opts) -> Self {
        let mut o = RocksOptions::default();
        o.create_if_missing(true);
        o.set_write_buffer_size(128 * 1024 * 1024);
        o.set_max_write_buffer_number(4);
        o.set_compression_type(rocksdb::DBCompressionType::None);
        let mut bbt = BlockBasedOptions::default();
        bbt.set_block_cache(&Cache::new_lru_cache(opts.cache_mb * 1024 * 1024));
        o.set_block_based_table_factory(&bbt);
        let mut wo = WriteOptions::default();
        wo.disable_wal(false);
        wo.set_sync(false);
        Self {
            db: DB::open(&o, dir).expect("rocksdb open"),
            wo,
        }
    }
}

#[cfg(feature = "comparators")]
impl Engine for RocksEngine {
    fn name(&self) -> &'static str {
        "rocksdb"
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.db.get(key).expect("rocksdb get")
    }

    fn put(&mut self, key: &[u8], value: &[u8]) {
        self.db.put_opt(key, value, &self.wo).expect("rocksdb put");
    }

    fn delete(&mut self, key: &[u8]) {
        self.db.delete_opt(key, &self.wo).expect("rocksdb delete");
    }

    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize {
        let mut seen = 0usize;
        let mut iter = self.db.raw_iterator();
        iter.seek(start);
        while let Some(key) = iter.key() {
            if key == start {
                iter.next();
                continue;
            }
            black_box(iter.value());
            seen += 1;
            if seen >= limit {
                break;
            }
            iter.next();
        }
        iter.status().expect("rocksdb scan_from status");
        seen
    }

    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize {
        let mut seen = 0usize;
        let mut iter = self.db.raw_iterator();
        iter.seek(prefix);
        while let Some(key) = iter.key() {
            if !key.starts_with(prefix) {
                break;
            }
            black_box(iter.value());
            seen += 1;
            if seen >= limit {
                break;
            }
            iter.next();
        }
        iter.status().expect("rocksdb scan_prefix status");
        seen
    }

    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize {
        let mut seen = 0usize;
        let mut iter = self.db.raw_iterator();
        iter.seek(prefix);
        while let Some(key) = iter.key() {
            if !key.starts_with(prefix) {
                break;
            }
            let next_seek = delimiter_successor(key, prefix.len(), delim);
            seen += 1;
            if seen >= limit {
                break;
            }
            match next_seek {
                Some(next) => iter.seek(next),
                None => iter.next(),
            }
        }
        iter.status().expect("rocksdb list_dir status");
        seen
    }

    fn batch(&mut self, batch: &Batch) {
        let mut wb = WriteBatch::default();
        for (k, v) in &batch.puts {
            wb.put(k, v);
        }
        for k in &batch.deletes {
            wb.delete(k);
        }
        self.db.write_opt(wb, &self.wo).expect("rocksdb batch");
    }

    fn finish_load(&mut self) {
        self.db.flush().expect("rocksdb flush");
    }
}

// --------------------------------------------------------------------
// sqlite
// --------------------------------------------------------------------

#[cfg(feature = "comparators")]
pub struct SqliteEngine {
    conn: Connection,
}

#[cfg(feature = "comparators")]
impl SqliteEngine {
    fn open(dir: &Path, opts: &Opts) -> Self {
        let conn = Connection::open(dir.join("sqlite.db")).expect("sqlite open");
        let cache_kib = opts.cache_mb * 1024;
        conn.execute_batch(&format!(
            "PRAGMA journal_mode = WAL;\n\
             PRAGMA synchronous = OFF;\n\
             PRAGMA temp_store = MEMORY;\n\
             PRAGMA cache_size = -{cache_kib};\n\
             CREATE TABLE IF NOT EXISTS kv (k BLOB PRIMARY KEY, v BLOB) WITHOUT ROWID;",
        ))
        .expect("sqlite pragmas + schema");
        Self { conn }
    }
}

#[cfg(feature = "comparators")]
impl Engine for SqliteEngine {
    fn name(&self) -> &'static str {
        "sqlite"
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT v FROM kv WHERE k = ?")
            .expect("sqlite get stmt");
        stmt.query_row(params![key], |row| row.get::<_, Vec<u8>>(0))
            .optional()
            .expect("sqlite get")
    }

    fn put(&mut self, key: &[u8], value: &[u8]) {
        let mut stmt = self
            .conn
            .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
            .expect("sqlite put stmt");
        stmt.execute(params![key, value]).expect("sqlite put");
    }

    fn delete(&mut self, key: &[u8]) {
        let mut stmt = self
            .conn
            .prepare_cached("DELETE FROM kv WHERE k = ?")
            .expect("sqlite delete stmt");
        stmt.execute(params![key]).expect("sqlite delete");
    }

    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT k, v FROM kv WHERE k > ? ORDER BY k LIMIT ?")
            .expect("sqlite scan_from stmt");
        let mut rows = stmt
            .query(params![start, limit as i64])
            .expect("sqlite scan_from query");
        let mut seen = 0usize;
        while let Some(row) = rows.next().expect("sqlite scan_from row") {
            black_box(row.get_ref(1).expect("sqlite scan_from value"));
            seen += 1;
        }
        seen
    }

    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize {
        let upper = prefix_upper(prefix);
        let mut stmt = self
            .conn
            .prepare_cached("SELECT k, v FROM kv WHERE k >= ? AND k < ? ORDER BY k LIMIT ?")
            .expect("sqlite scan_prefix stmt");
        let mut rows = stmt
            .query(params![prefix, upper, limit as i64])
            .expect("sqlite scan_prefix query");
        let mut seen = 0usize;
        while let Some(row) = rows.next().expect("sqlite scan_prefix row") {
            black_box(row.get_ref(1).expect("sqlite scan_prefix value"));
            seen += 1;
        }
        seen
    }

    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize {
        let upper = prefix_upper(prefix);
        let mut stmt = self
            .conn
            .prepare_cached("SELECT k FROM kv WHERE k >= ? AND k < ? ORDER BY k LIMIT 1")
            .expect("sqlite list_dir stmt");
        let mut seen = 0usize;
        let mut cursor = prefix.to_vec();
        while cursor.as_slice() < upper.as_slice() {
            let found = stmt
                .query_row(params![cursor.as_slice(), upper.as_slice()], |row| {
                    row.get::<_, Vec<u8>>(0)
                })
                .optional()
                .expect("sqlite list_dir query");
            let Some(key) = found else {
                break;
            };
            let next_seek = delimiter_successor(&key, prefix.len(), delim);
            seen += 1;
            if seen >= limit {
                break;
            }
            cursor = next_seek.unwrap_or_else(|| key_successor(&key));
        }
        seen
    }

    fn batch(&mut self, batch: &Batch) {
        let txn = self
            .conn
            .unchecked_transaction()
            .expect("sqlite batch txn");
        {
            let mut put = txn
                .prepare_cached("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)")
                .expect("sqlite batch put stmt");
            for (k, v) in &batch.puts {
                put.execute(params![k, v]).expect("sqlite batch put");
            }
            let mut del = txn
                .prepare_cached("DELETE FROM kv WHERE k = ?")
                .expect("sqlite batch delete stmt");
            for k in &batch.deletes {
                del.execute(params![k]).expect("sqlite batch delete");
            }
        }
        txn.commit().expect("sqlite batch commit");
    }

    fn finish_load(&mut self) {
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(PASSIVE);")
            .expect("sqlite wal checkpoint");
    }
}

// --------------------------------------------------------------------
// sled
// --------------------------------------------------------------------

#[cfg(feature = "comparators")]
pub struct SledEngine {
    db: SledDb,
}

#[cfg(feature = "comparators")]
impl SledEngine {
    fn open(dir: &Path, opts: &Opts) -> Self {
        let db = sled::Config::new()
            .path(dir)
            .mode(SledMode::HighThroughput)
            .cache_capacity((opts.cache_mb * 1024 * 1024) as u64)
            .flush_every_ms(None)
            .open()
            .expect("sled open");
        Self { db }
    }
}

#[cfg(feature = "comparators")]
impl Engine for SledEngine {
    fn name(&self) -> &'static str {
        "sled"
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.db.get(key).expect("sled get").map(|v| v.to_vec())
    }

    fn put(&mut self, key: &[u8], value: &[u8]) {
        self.db.insert(key, value).expect("sled put");
    }

    fn delete(&mut self, key: &[u8]) {
        self.db.remove(key).expect("sled delete");
    }

    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize {
        let mut seen = 0usize;
        let range = (Bound::Excluded(start.to_vec()), Bound::Unbounded);
        for item in self.db.range::<Vec<u8>, _>(range) {
            let (_k, v) = item.expect("sled scan_from");
            black_box(&v);
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize {
        let mut seen = 0usize;
        for item in self.db.scan_prefix(prefix) {
            let (_k, v) = item.expect("sled scan_prefix");
            black_box(&v);
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize {
        let mut seen = 0usize;
        let mut cursor = prefix.to_vec();
        let upper = prefix_upper(prefix);
        while cursor.as_slice() < upper.as_slice() {
            let found = self.db.range(cursor.as_slice()..upper.as_slice()).next();
            let Some(item) = found else {
                break;
            };
            let (key, _value) = item.expect("sled list_dir");
            let next_seek = delimiter_successor(&key, prefix.len(), delim);
            seen += 1;
            if seen >= limit {
                break;
            }
            cursor = next_seek.unwrap_or_else(|| key_successor(&key));
        }
        seen
    }

    fn batch(&mut self, batch: &Batch) {
        let mut wb = sled::Batch::default();
        for (k, v) in &batch.puts {
            wb.insert(k.as_slice(), v.as_slice());
        }
        for k in &batch.deletes {
            wb.remove(k.as_slice());
        }
        self.db.apply_batch(wb).expect("sled batch");
    }

    fn finish_load(&mut self) {
        self.db.flush().expect("sled flush");
    }
}

// --------------------------------------------------------------------
// lmdb (heed)
// --------------------------------------------------------------------

#[cfg(feature = "comparators")]
pub struct LmdbEngine {
    env: LmdbEnv,
    db: LmdbDatabase<Bytes, Bytes>,
}

#[cfg(feature = "comparators")]
impl LmdbEngine {
    fn open(dir: &Path, opts: &Opts) -> Self {
        let mut o = EnvOpenOptions::new();
        o.map_size(opts.map_gb << 30);
        o.max_dbs(1);
        // async-durable profile: no fsync on commit, like the peers.
        unsafe {
            o.flags(EnvFlags::NO_SYNC | EnvFlags::NO_META_SYNC);
        }
        let env = unsafe { o.open(dir) }.expect("lmdb open");
        let mut wtxn = env.write_txn().expect("lmdb create wtxn");
        let db: LmdbDatabase<Bytes, Bytes> =
            env.create_database(&mut wtxn, None).expect("lmdb create db");
        wtxn.commit().expect("lmdb create commit");
        Self { env, db }
    }
}

#[cfg(feature = "comparators")]
impl Engine for LmdbEngine {
    fn name(&self) -> &'static str {
        "lmdb"
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let rtxn = self.env.read_txn().expect("lmdb rtxn");
        self.db
            .get(&rtxn, key)
            .expect("lmdb get")
            .map(|v| v.to_vec())
    }

    fn put(&mut self, key: &[u8], value: &[u8]) {
        let mut wtxn = self.env.write_txn().expect("lmdb wtxn");
        self.db.put(&mut wtxn, key, value).expect("lmdb put");
        wtxn.commit().expect("lmdb put commit");
    }

    fn delete(&mut self, key: &[u8]) {
        let mut wtxn = self.env.write_txn().expect("lmdb wtxn");
        self.db.delete(&mut wtxn, key).expect("lmdb delete");
        wtxn.commit().expect("lmdb delete commit");
    }

    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize {
        let rtxn = self.env.read_txn().expect("lmdb rtxn");
        let range = (Bound::Excluded(start), Bound::Unbounded);
        let mut seen = 0usize;
        for item in self.db.range(&rtxn, &range).expect("lmdb scan_from") {
            let (_k, v) = item.expect("lmdb scan_from item");
            black_box(v);
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize {
        let rtxn = self.env.read_txn().expect("lmdb rtxn");
        let mut seen = 0usize;
        for item in self
            .db
            .prefix_iter(&rtxn, prefix)
            .expect("lmdb scan_prefix")
        {
            let (_k, v) = item.expect("lmdb scan_prefix item");
            black_box(v);
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize {
        let rtxn = self.env.read_txn().expect("lmdb rtxn");
        let mut seen = 0usize;
        let mut cursor = prefix.to_vec();
        let upper = prefix_upper(prefix);
        while cursor.as_slice() < upper.as_slice() {
            let range = (
                Bound::Included(cursor.as_slice()),
                Bound::Excluded(upper.as_slice()),
            );
            let found = self
                .db
                .range(&rtxn, &range)
                .expect("lmdb list_dir range")
                .next();
            let Some(item) = found else {
                break;
            };
            let (key, _value) = item.expect("lmdb list_dir item");
            let next_seek = delimiter_successor(key, prefix.len(), delim);
            seen += 1;
            if seen >= limit {
                break;
            }
            cursor = next_seek.unwrap_or_else(|| key_successor(key));
        }
        seen
    }

    fn batch(&mut self, batch: &Batch) {
        let mut wtxn = self.env.write_txn().expect("lmdb batch wtxn");
        for (k, v) in &batch.puts {
            self.db.put(&mut wtxn, k, v).expect("lmdb batch put");
        }
        for k in &batch.deletes {
            self.db.delete(&mut wtxn, k).expect("lmdb batch delete");
        }
        wtxn.commit().expect("lmdb batch commit");
    }

    fn finish_load(&mut self) {
        self.env.force_sync().expect("lmdb force_sync");
    }
}

// --------------------------------------------------------------------
// redb
// --------------------------------------------------------------------

#[cfg(feature = "comparators")]
const REDB_TABLE: redb::TableDefinition<&[u8], &[u8]> = redb::TableDefinition::new("kv");

#[cfg(feature = "comparators")]
pub struct RedbEngine {
    db: redb::Database,
}

#[cfg(feature = "comparators")]
impl RedbEngine {
    fn open(dir: &Path, opts: &Opts) -> Self {
        let db = redb::Builder::new()
            .set_cache_size(opts.cache_mb * 1024 * 1024)
            .create(dir.join("redb.db"))
            .expect("redb open");
        // Create the table up front so read txns never race a missing table.
        let txn = db.begin_write().expect("redb init txn");
        txn.open_table(REDB_TABLE).expect("redb init table");
        txn.commit().expect("redb init commit");
        Self { db }
    }

    fn write_txn(&self) -> redb::WriteTransaction {
        let mut txn = self.db.begin_write().expect("redb write txn");
        // async-durable profile: commit without fsync; the OS flushes later.
        txn.set_durability(redb::Durability::Eventual);
        txn
    }
}

#[cfg(feature = "comparators")]
impl Engine for RedbEngine {
    fn name(&self) -> &'static str {
        "redb"
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let txn = self.db.begin_read().expect("redb read txn");
        let table = txn.open_table(REDB_TABLE).expect("redb table");
        table
            .get(key)
            .expect("redb get")
            .map(|guard| guard.value().to_vec())
    }

    fn put(&mut self, key: &[u8], value: &[u8]) {
        let txn = self.write_txn();
        {
            let mut table = txn.open_table(REDB_TABLE).expect("redb table");
            table.insert(key, value).expect("redb put");
        }
        txn.commit().expect("redb put commit");
    }

    fn delete(&mut self, key: &[u8]) {
        let txn = self.write_txn();
        {
            let mut table = txn.open_table(REDB_TABLE).expect("redb table");
            table.remove(key).expect("redb delete");
        }
        txn.commit().expect("redb delete commit");
    }

    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize {
        let txn = self.db.begin_read().expect("redb read txn");
        let table = txn.open_table(REDB_TABLE).expect("redb table");
        let range = (Bound::Excluded(start), Bound::Unbounded);
        let mut seen = 0usize;
        for item in table.range::<&[u8]>(range).expect("redb scan_from") {
            let (_k, v) = item.expect("redb scan_from item");
            black_box(v.value());
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize {
        let txn = self.db.begin_read().expect("redb read txn");
        let table = txn.open_table(REDB_TABLE).expect("redb table");
        let upper = prefix_upper(prefix);
        let mut seen = 0usize;
        for item in table
            .range::<&[u8]>(prefix..upper.as_slice())
            .expect("redb scan_prefix")
        {
            let (_k, v) = item.expect("redb scan_prefix item");
            black_box(v.value());
            seen += 1;
            if seen >= limit {
                break;
            }
        }
        seen
    }

    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize {
        let txn = self.db.begin_read().expect("redb read txn");
        let table = txn.open_table(REDB_TABLE).expect("redb table");
        let mut seen = 0usize;
        let mut cursor = prefix.to_vec();
        let upper = prefix_upper(prefix);
        while cursor.as_slice() < upper.as_slice() {
            let found = table
                .range::<&[u8]>(cursor.as_slice()..upper.as_slice())
                .expect("redb list_dir range")
                .next();
            let Some(item) = found else {
                break;
            };
            let (key, _value) = item.expect("redb list_dir item");
            let key = key.value().to_vec();
            let next_seek = delimiter_successor(&key, prefix.len(), delim);
            seen += 1;
            if seen >= limit {
                break;
            }
            cursor = next_seek.unwrap_or_else(|| key_successor(&key));
        }
        seen
    }

    fn batch(&mut self, batch: &Batch) {
        let txn = self.write_txn();
        {
            let mut table = txn.open_table(REDB_TABLE).expect("redb table");
            for (k, v) in &batch.puts {
                table.insert(k.as_slice(), v.as_slice()).expect("redb batch put");
            }
            for k in &batch.deletes {
                table.remove(k.as_slice()).expect("redb batch delete");
            }
        }
        txn.commit().expect("redb batch commit");
    }

    fn finish_load(&mut self) {
        let mut txn = self.db.begin_write().expect("redb settle txn");
        txn.set_durability(redb::Durability::Immediate);
        txn.open_table(REDB_TABLE).expect("redb settle table");
        txn.commit().expect("redb settle commit");
    }
}

// --------------------------------------------------------------------
// postgres
// --------------------------------------------------------------------

#[cfg(feature = "pg")]
pub struct PgEngine {
    client: postgres::Client,
    get_stmt: postgres::Statement,
    put_stmt: postgres::Statement,
    del_stmt: postgres::Statement,
    scan_from_stmt: postgres::Statement,
    scan_prefix_stmt: postgres::Statement,
    first_in_range_stmt: postgres::Statement,
    multi_put_stmt: postgres::Statement,
    multi_del_stmt: postgres::Statement,
}

#[cfg(feature = "pg")]
impl PgEngine {
    fn open(opts: &Opts) -> Self {
        let conninfo = opts.pg_conn.as_deref().expect(
            "postgres engine needs HOLT_PG, e.g. \
             `host=/tmp/holt-pg port=54329 user=bench dbname=bench`",
        );
        let mut client =
            postgres::Client::connect(conninfo, postgres::NoTls).expect("postgres connect");
        client
            .batch_execute(
                "SET synchronous_commit = off;\n\
                 DROP TABLE IF EXISTS kv;\n\
                 CREATE TABLE kv (k BYTEA PRIMARY KEY, v BYTEA NOT NULL);",
            )
            .expect("postgres schema");
        let get_stmt = client
            .prepare("SELECT v FROM kv WHERE k = $1")
            .expect("pg get stmt");
        let put_stmt = client
            .prepare(
                "INSERT INTO kv (k, v) VALUES ($1, $2) \
                 ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v",
            )
            .expect("pg put stmt");
        let del_stmt = client
            .prepare("DELETE FROM kv WHERE k = $1")
            .expect("pg delete stmt");
        let scan_from_stmt = client
            .prepare("SELECT k, v FROM kv WHERE k > $1 ORDER BY k LIMIT $2")
            .expect("pg scan_from stmt");
        let scan_prefix_stmt = client
            .prepare("SELECT k, v FROM kv WHERE k >= $1 AND k < $2 ORDER BY k LIMIT $3")
            .expect("pg scan_prefix stmt");
        let first_in_range_stmt = client
            .prepare("SELECT k FROM kv WHERE k >= $1 AND k < $2 ORDER BY k LIMIT 1")
            .expect("pg first_in_range stmt");
        // One round trip per batch, the idiomatic bulk form — per-row
        // execute() calls would charge PostgreSQL a protocol round trip
        // per entry and benchmark the client loop, not the engine.
        let multi_put_stmt = client
            .prepare(
                "INSERT INTO kv (k, v) SELECT * FROM UNNEST($1::bytea[], $2::bytea[]) \
                 ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v",
            )
            .expect("pg multi_put stmt");
        let multi_del_stmt = client
            .prepare("DELETE FROM kv WHERE k = ANY($1::bytea[])")
            .expect("pg multi_del stmt");
        Self {
            client,
            get_stmt,
            put_stmt,
            del_stmt,
            scan_from_stmt,
            scan_prefix_stmt,
            first_in_range_stmt,
            multi_put_stmt,
            multi_del_stmt,
        }
    }
}

#[cfg(feature = "pg")]
impl Engine for PgEngine {
    fn name(&self) -> &'static str {
        "postgres"
    }

    fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.client
            .query_opt(&self.get_stmt, &[&key])
            .expect("pg get")
            .map(|row| row.get::<_, Vec<u8>>(0))
    }

    fn put(&mut self, key: &[u8], value: &[u8]) {
        self.client
            .execute(&self.put_stmt, &[&key, &value])
            .expect("pg put");
    }

    fn delete(&mut self, key: &[u8]) {
        self.client
            .execute(&self.del_stmt, &[&key])
            .expect("pg delete");
    }

    fn scan_from(&mut self, start: &[u8], limit: usize) -> usize {
        let rows = self
            .client
            .query(&self.scan_from_stmt, &[&start, &(limit as i64)])
            .expect("pg scan_from");
        for row in &rows {
            black_box(row.get::<_, &[u8]>(1));
        }
        rows.len()
    }

    fn scan_prefix(&mut self, prefix: &[u8], limit: usize) -> usize {
        let upper = prefix_upper(prefix);
        let rows = self
            .client
            .query(
                &self.scan_prefix_stmt,
                &[&prefix, &upper.as_slice(), &(limit as i64)],
            )
            .expect("pg scan_prefix");
        for row in &rows {
            black_box(row.get::<_, &[u8]>(1));
        }
        rows.len()
    }

    fn list_dir(&mut self, prefix: &[u8], delim: u8, limit: usize) -> usize {
        let upper = prefix_upper(prefix);
        let mut seen = 0usize;
        let mut cursor = prefix.to_vec();
        while cursor.as_slice() < upper.as_slice() {
            let found = self
                .client
                .query_opt(
                    &self.first_in_range_stmt,
                    &[&cursor.as_slice(), &upper.as_slice()],
                )
                .expect("pg list_dir")
                .map(|row| row.get::<_, Vec<u8>>(0));
            let Some(key) = found else {
                break;
            };
            let next_seek = delimiter_successor(&key, prefix.len(), delim);
            seen += 1;
            if seen >= limit {
                break;
            }
            cursor = next_seek.unwrap_or_else(|| key_successor(&key));
        }
        seen
    }

    fn batch(&mut self, batch: &Batch) {
        let mut txn = self.client.transaction().expect("pg batch txn");
        if !batch.puts.is_empty() {
            let keys: Vec<&[u8]> = batch.puts.iter().map(|(k, _)| k.as_slice()).collect();
            let values: Vec<&[u8]> = batch.puts.iter().map(|(_, v)| v.as_slice()).collect();
            txn.execute(&self.multi_put_stmt, &[&keys, &values])
                .expect("pg batch put");
        }
        if !batch.deletes.is_empty() {
            let keys: Vec<&[u8]> = batch.deletes.iter().map(|k| k.as_slice()).collect();
            txn.execute(&self.multi_del_stmt, &[&keys])
                .expect("pg batch delete");
        }
        txn.commit().expect("pg batch commit");
    }

    fn finish_load(&mut self) {
        self.client
            .batch_execute("ANALYZE kv;")
            .expect("pg analyze");
    }
}
