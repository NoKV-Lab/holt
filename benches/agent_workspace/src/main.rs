//! Application-level workspace workflow, with a shared handler and two stores.
//! This measures local content + metadata + JSON, not the NoKV SDK/RPC server.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use holt::{
    DBView, Durability, JournalAnchor, JournalEnvelope, RangeEntry, RecordVersion, TreeConfig, DB,
};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
type Pair = (Vec<u8>, Vec<u8>);
const TABLES: [&str; 8] = [
    "workspace",
    "path",
    "revision",
    "object_ref",
    "operation",
    "dedupe",
    "history",
    "event",
];
const WS: usize = 0;
const PATHS: usize = 1;
const REV: usize = 2;
const REFS: usize = 3;
const OPS: usize = 4;
const DEDUPE: usize = 5;
const HISTORY: usize = 6;
const EVENTS: usize = 7;

#[derive(Clone, Debug)]
struct Config {
    engine: String,
    directory: PathBuf,
    sync: bool,
    attached: bool,
    threads: usize,
    operations: usize,
    warmup: usize,
    workspaces: usize,
    files: usize,
    payload: usize,
    seed: u64,
    read_mode: String,
    background: bool,
    sqlite_normal: bool,
}

impl Config {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut values = BTreeMap::new();
        while let Some(key) = args.next() {
            let value = args.next().ok_or("every argument needs a value")?;
            values.insert(key, value);
        }
        let text = |key: &str, default: &str| {
            values
                .get(key)
                .cloned()
                .unwrap_or_else(|| default.to_owned())
        };
        let cfg = Self {
            engine: text("--engine", "holt"),
            directory: PathBuf::from(values.get("--dir").ok_or("--dir is required")?),
            sync: text("--sync", "false").parse()?,
            attached: text("--attached", "false").parse()?,
            threads: text("--threads", "1").parse()?,
            operations: text("--ops", "4000").parse()?,
            warmup: text("--warmup", "200").parse()?,
            workspaces: text("--workspaces", "8").parse()?,
            files: text("--files", "512").parse()?,
            payload: text("--payload", "4096").parse()?,
            seed: text("--seed", "456789").parse()?,
            read_mode: text("--read-mode", "view").to_owned(),
            background: text("--background", "false").parse()?,
            sqlite_normal: text("--sqlite-normal", "false").parse()?,
        };
        assert!(matches!(cfg.engine.as_str(), "holt" | "sqlite"));
        assert!(matches!(cfg.read_mode.as_str(), "view" | "point"));
        assert!(cfg.threads > 0 && cfg.threads <= cfg.workspaces && cfg.files >= 64);
        assert!(cfg.operations >= cfg.threads);
        Ok(cfg)
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Marker {
    generation: u64,
}
#[derive(Clone, Serialize, Deserialize)]
struct PathMeta {
    revision: String,
    size: usize,
    crc: u32,
    generation: u64,
}
#[derive(Clone, Serialize, Deserialize)]
struct Revision {
    object: String,
    size: usize,
    crc: u32,
}
#[derive(Clone, Serialize, Deserialize)]
struct Receipt {
    request_crc: u32,
    response: Response,
}

#[derive(Clone, Serialize, Deserialize)]
struct Request {
    op: String,
    workspace: String,
    path: String,
    id: String,
    content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Response {
    generation: u64,
    size: usize,
    crc: u32,
    items: usize,
    content: String,
}

#[derive(Serialize)]
struct Guard {
    table: usize,
    key: Vec<u8>,
    version: Option<u64>,
}
#[derive(Serialize)]
struct Change {
    table: usize,
    key: Vec<u8>,
    value: Option<Vec<u8>>,
}
#[derive(Default, Serialize)]
struct Mutation {
    guards: Vec<Guard>,
    changes: Vec<Change>,
}

impl Mutation {
    fn guard(&mut self, table: usize, key: &[u8], row: Option<&Row>) {
        self.guards.push(Guard {
            table,
            key: key.to_vec(),
            version: row.map(|r| r.version),
        });
    }
    fn put(&mut self, table: usize, key: &[u8], value: Vec<u8>) {
        self.changes.push(Change {
            table,
            key: key.to_vec(),
            value: Some(value),
        });
    }
    fn delete(&mut self, table: usize, key: &[u8]) {
        self.changes.push(Change {
            table,
            key: key.to_vec(),
            value: None,
        });
    }
}

struct Row {
    value: Vec<u8>,
    version: u64,
}

trait Reader {
    fn get(&self, table: usize, key: &[u8]) -> Result<Option<Row>>;
    fn list(&self, table: usize, prefix: &[u8], limit: usize) -> Result<Vec<Pair>>;
}

struct HoltReader<'a> {
    view: &'a DBView,
}
impl Reader for HoltReader<'_> {
    fn get(&self, table: usize, key: &[u8]) -> Result<Option<Row>> {
        Ok(self
            .view
            .tree(TABLES[table])
            .unwrap()
            .get_record(key)?
            .map(|r| Row {
                value: r.value,
                version: r.version.as_u64(),
            }))
    }
    fn list(&self, table: usize, prefix: &[u8], limit: usize) -> Result<Vec<Pair>> {
        self.view
            .tree(TABLES[table])
            .unwrap()
            .scan(prefix)?
            .into_iter()
            .take(limit)
            .map(|entry| match entry? {
                RangeEntry::Key { key, value, .. } => Ok((key, value)),
                _ => Err("unexpected delimiter entry".into()),
            })
            .collect()
    }
}

struct SqlReader<'a> {
    connection: &'a Connection,
}
impl Reader for SqlReader<'_> {
    fn get(&self, table: usize, key: &[u8]) -> Result<Option<Row>> {
        Ok(self
            .connection
            .prepare_cached(&format!("SELECT v,ver FROM t_{} WHERE k=?1", TABLES[table]))?
            .query_row([key], |row| {
                Ok(Row {
                    value: row.get(0)?,
                    version: row.get::<_, i64>(1)? as u64,
                })
            })
            .optional()?)
    }
    fn list(&self, table: usize, prefix: &[u8], limit: usize) -> Result<Vec<Pair>> {
        let mut upper = prefix.to_vec();
        let last = upper.last_mut().ok_or("empty prefix")?;
        assert!(*last < 255);
        *last += 1;
        let mut stmt = self.connection.prepare_cached(&format!(
            "SELECT k,v FROM t_{} WHERE k>=?1 AND k<?2 ORDER BY k LIMIT ?3",
            TABLES[table]
        ))?;
        let rows = stmt.query_map(params![prefix, upper, limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        Ok(rows.collect::<std::result::Result<Vec<Pair>, _>>()?)
    }
}

struct Shared {
    cfg: Config,
    db: Option<DB>,
    tail: Mutex<JournalAnchor>,
    version: AtomicU64,
}

#[derive(Default)]
struct Counters {
    view_ns: u64,
    views: u64,
    apply_ns: u64,
    commits: u64,
    object_ns: u64,
    object_reads: u64,
    object_writes: u64,
    retries: u64,
}

struct Session {
    shared: Arc<Shared>,
    sqlite: Option<Connection>,
    trees: Vec<holt::Tree>,
    counters: Counters,
}

fn sql_open(cfg: &Config) -> Result<Connection> {
    let conn = Connection::open(cfg.directory.join("metadata/sqlite.db"))?;
    conn.busy_timeout(Duration::from_secs(30))?;
    conn.set_prepared_statement_cache_capacity(128);
    conn.execute_batch(&format!("PRAGMA journal_mode=WAL; PRAGMA synchronous={}; PRAGMA fullfsync=ON; PRAGMA checkpoint_fullfsync=ON; PRAGMA cache_size=-65536; PRAGMA temp_store=MEMORY; PRAGMA wal_autocheckpoint={}; PRAGMA mmap_size=0;", if cfg.sync { "FULL" } else if cfg.sqlite_normal {"NORMAL"} else { "OFF" }, if cfg.background {1000} else {0}))?;
    Ok(conn)
}

impl Session {
    fn open(shared: Arc<Shared>) -> Result<Self> {
        let (sqlite, trees) = match &shared.db {
            Some(db) => (
                None,
                TABLES
                    .iter()
                    .map(|name| db.open_tree(name))
                    .collect::<holt::Result<Vec<_>>>()?,
            ),
            None => (Some(sql_open(&shared.cfg)?), Vec::new()),
        };
        Ok(Self {
            shared,
            sqlite,
            trees,
            counters: Counters::default(),
        })
    }
    fn current(&self, table: usize, key: &[u8]) -> Result<Option<Row>> {
        if let Some(connection) = &self.sqlite {
            SqlReader { connection }.get(table, key)
        } else {
            Ok(self.trees[table].get_record(key)?.map(|r| Row {
                value: r.value,
                version: r.version.as_u64(),
            }))
        }
    }
    fn view<R>(
        &mut self,
        tables: &[usize],
        workspace: &str,
        f: impl FnOnce(&dyn Reader) -> Result<R>,
    ) -> Result<R> {
        let started = Instant::now();
        let result = if let Some(conn) = &self.sqlite {
            let tx = conn.unchecked_transaction()?;
            let result = f(&SqlReader { connection: &tx })?;
            tx.commit()?;
            Ok(result)
        } else {
            let prefix = format!("{workspace}/");
            let scopes = tables
                .iter()
                .map(|&table| (TABLES[table], prefix.as_bytes()))
                .collect::<Vec<_>>();
            // The callback result is carried through Holt's native error type.
            self.shared
                .db
                .as_ref()
                .unwrap()
                .view(&scopes, |view| Ok(f(&HoltReader { view })))?
        };
        self.counters.views += 1;
        self.counters.view_ns += started.elapsed().as_nanos() as u64;
        result
    }
    fn apply(&mut self, mutation: &Mutation) -> Result<bool> {
        let started = Instant::now();
        let canonical = if self.shared.cfg.attached {
            Some(serde_json::to_vec(mutation)?)
        } else {
            None
        };
        let committed = if let Some(connection) = &self.sqlite {
            connection.execute_batch("BEGIN IMMEDIATE")?;
            let result = (|| -> Result<bool> {
                let reader = SqlReader { connection };
                for guard in &mutation.guards {
                    if reader.get(guard.table, &guard.key)?.map(|r| r.version) != guard.version {
                        return Ok(false);
                    }
                }
                let version = (self.shared.version.fetch_add(1, Ordering::Relaxed) + 1) as i64;
                for change in &mutation.changes {
                    match &change.value {
                        Some(value) => {
                            connection.prepare_cached(&format!("INSERT INTO t_{}(k,v,ver) VALUES(?1,?2,?3) ON CONFLICT(k) DO UPDATE SET v=excluded.v,ver=excluded.ver", TABLES[change.table]))?.execute(params![change.key, value, version])?;
                        }
                        None => {
                            connection
                                .prepare_cached(&format!(
                                    "DELETE FROM t_{} WHERE k=?1",
                                    TABLES[change.table]
                                ))?
                                .execute([&change.key])?;
                        }
                    }
                }
                if let Some(canonical) = &canonical {
                    connection
                        .prepare_cached("INSERT INTO recovery_log(seq,payload) VALUES(?1,?2)")?
                        .execute(params![version, canonical])?;
                }
                Ok(true)
            })();
            match result {
                Ok(true) => {
                    connection.execute_batch("COMMIT")?;
                    true
                }
                Ok(false) => {
                    connection.execute_batch("ROLLBACK")?;
                    false
                }
                Err(error) => {
                    let _ = connection.execute_batch("ROLLBACK");
                    return Err(error);
                }
            }
        } else {
            let db = self.shared.db.as_ref().unwrap();
            let build = |batch: &mut holt::DBAtomicBatch| {
                for guard in &mutation.guards {
                    match guard.version {
                        Some(version) => batch.assert_version(
                            TABLES[guard.table],
                            &guard.key,
                            RecordVersion::from_raw(version),
                        ),
                        None => batch.assert_absent(TABLES[guard.table], &guard.key),
                    }
                }
                for change in &mutation.changes {
                    match &change.value {
                        Some(value) => batch.put(TABLES[change.table], &change.key, value),
                        None => batch.delete(TABLES[change.table], &change.key),
                    }
                }
            };
            if let Some(canonical) = canonical {
                // One application recovery stream, with the same single-funnel
                // admission as the NoKV profile. Ordinary mode has no funnel.
                let mut tail = self.shared.tail.lock().unwrap();
                let mut digest = [0; 32];
                digest[..8].copy_from_slice(&(tail.sequence() + 1).to_le_bytes());
                digest[8..12].copy_from_slice(&crc32fast::hash(&canonical).to_le_bytes());
                let next = JournalAnchor::new(tail.sequence() + 1, digest);
                let committed = db.atomic_with_journal_envelope(
                    JournalEnvelope::new(*tail, next, canonical)?,
                    build,
                )?;
                if committed {
                    *tail = next;
                }
                committed
            } else {
                db.atomic(build)?
            }
        };
        self.counters.apply_ns += started.elapsed().as_nanos() as u64;
        if committed {
            self.counters.commits += 1;
        } else {
            self.counters.retries += 1;
        }
        Ok(committed)
    }
}

fn workspace_key(ws: &str) -> Vec<u8> {
    format!("{ws}/marker").into_bytes()
}
fn file_path(ws: &str, file: usize) -> String {
    format!("{ws}/src/dir-{:03}/file-{file:06}.txt", file / 64)
}
fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}
fn decode<T: for<'a> Deserialize<'a>>(bytes: &[u8]) -> Result<T> {
    Ok(serde_json::from_slice(bytes)?)
}

fn write_object(session: &mut Session, name: &str, content: &str) -> Result<()> {
    if session.shared.cfg.payload == 0 {
        return Ok(());
    }
    let started = Instant::now();
    let directory = session.shared.cfg.directory.join("objects");
    let path = directory.join(name);
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(content.as_bytes())?;
            if session.shared.cfg.sync {
                file.sync_all()?;
                File::open(&directory)?.sync_all()?;
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            assert_eq!(fs::read(directory.join(name))?, content.as_bytes());
        }
        Err(e) => return Err(e.into()),
    }
    session.counters.object_writes += 1;
    session.counters.object_ns += started.elapsed().as_nanos() as u64;
    Ok(())
}

fn mutate(
    session: &mut Session,
    request: &Request,
    kind: &str,
    from: &str,
    to: &str,
) -> Result<Response> {
    let request_crc = crc32fast::hash(&encode(&(request, kind, from, to))?);
    let receipt_key = format!("{}/requests/{}-{kind}", request.workspace, request.id).into_bytes();
    if let Some(row) = session.current(DEDUPE, &receipt_key)? {
        let receipt: Receipt = decode(&row.value)?;
        assert_eq!(request_crc, receipt.request_crc, "request replay mismatch");
        return Ok(receipt.response);
    }
    if kind == "publish" {
        write_object(session, &request.id, &request.content)?;
    }
    for _ in 0..32 {
        let (marker_row, source, destination) =
            session.view(&[WS, PATHS], &request.workspace, |reader| {
                Ok((
                    reader
                        .get(WS, &workspace_key(&request.workspace))?
                        .ok_or("workspace missing")?,
                    reader.get(PATHS, from.as_bytes())?,
                    if from != to {
                        reader.get(PATHS, to.as_bytes())?
                    } else {
                        None
                    },
                ))
            })?;
        let marker: Marker = decode(&marker_row.value)?;
        let generation = marker.generation + 1;
        let mut mutation = Mutation::default();
        mutation.guard(WS, &workspace_key(&request.workspace), Some(&marker_row));
        mutation.guard(PATHS, from.as_bytes(), source.as_ref());
        mutation.guard(DEDUPE, &receipt_key, None);
        mutation.put(
            WS,
            &workspace_key(&request.workspace),
            encode(&Marker { generation })?,
        );
        let old = source.as_ref().map(|row| row.value.clone());
        let (size, crc) = match kind {
            "publish" => {
                let revision_key = format!("{}/revision/{}", request.workspace, request.id);
                let meta = PathMeta {
                    revision: revision_key.clone(),
                    size: request.content.len(),
                    crc: crc32fast::hash(request.content.as_bytes()),
                    generation,
                };
                mutation.guard(REV, revision_key.as_bytes(), None);
                mutation.put(PATHS, to.as_bytes(), encode(&meta)?);
                mutation.put(
                    REV,
                    revision_key.as_bytes(),
                    encode(&Revision {
                        object: request.id.clone(),
                        size: meta.size,
                        crc: meta.crc,
                    })?,
                );
                mutation.put(
                    REFS,
                    revision_key.as_bytes(),
                    encode(&json!({"references": 1}))?,
                );
                (meta.size, meta.crc)
            }
            "rename_out" | "rename_back" => {
                assert!(destination.is_none());
                mutation.guard(PATHS, to.as_bytes(), None);
                let mut meta: PathMeta =
                    decode(&source.as_ref().ok_or("rename source missing")?.value)?;
                meta.generation = generation;
                mutation.delete(PATHS, from.as_bytes());
                mutation.put(PATHS, to.as_bytes(), encode(&meta)?);
                (meta.size, meta.crc)
            }
            "remove" => {
                let meta: PathMeta =
                    decode(&source.as_ref().ok_or("remove source missing")?.value)?;
                mutation.delete(PATHS, from.as_bytes());
                (meta.size, meta.crc)
            }
            _ => return Err("unknown mutation".into()),
        };
        let response = Response {
            generation,
            size,
            crc,
            items: 1,
            content: String::new(),
        };
        let receipt = encode(&Receipt {
            request_crc,
            response: response.clone(),
        })?;
        mutation.put(OPS, &receipt_key, receipt.clone());
        mutation.put(DEDUPE, &receipt_key, receipt);
        mutation.put(
            HISTORY,
            &receipt_key,
            old.unwrap_or_else(|| b"null".to_vec()),
        );
        mutation.put(
            EVENTS,
            &format!(
                "{}/events/{generation:020}/{}-{kind}",
                request.workspace, request.id
            )
            .into_bytes(),
            encode(&json!({"kind": kind, "path": to, "generation": generation}))?,
        );
        if session.apply(&mutation)? {
            return Ok(response);
        }
    }
    Err("workspace CAS retry exhausted".into())
}

fn read_metadata(session: &mut Session, request: &Request) -> Result<(Marker, PathMeta, Revision)> {
    if session.shared.cfg.read_mode == "point" {
        // Diagnostic only: no cross-tree snapshot guarantee. The report keeps
        // this separate from the default consistent-view workspace results.
        let marker = session
            .current(WS, &workspace_key(&request.workspace))?
            .ok_or("marker missing")?;
        let path = session
            .current(PATHS, request.path.as_bytes())?
            .ok_or("path missing")?;
        let path: PathMeta = decode(&path.value)?;
        let revision = session
            .current(REV, path.revision.as_bytes())?
            .ok_or("revision missing")?;
        return Ok((decode(&marker.value)?, path, decode(&revision.value)?));
    }
    session.view(&[WS, PATHS, REV], &request.workspace, |reader| {
        let marker = reader
            .get(WS, &workspace_key(&request.workspace))?
            .ok_or("marker missing")?;
        let path = reader
            .get(PATHS, request.path.as_bytes())?
            .ok_or("path missing")?;
        let path: PathMeta = decode(&path.value)?;
        let revision = reader
            .get(REV, path.revision.as_bytes())?
            .ok_or("revision missing")?;
        Ok((decode(&marker.value)?, path, decode(&revision.value)?))
    })
}

fn handle(session: &mut Session, request: &Request) -> Result<Response> {
    match request.op.as_str() {
        "read" | "stat" => {
            let (marker, path, revision) = read_metadata(session, request)?;
            assert!(path.generation <= marker.generation);
            assert_eq!((path.size, path.crc), (revision.size, revision.crc));
            let mut content = String::new();
            if request.op == "read" && session.shared.cfg.payload > 0 {
                let started = Instant::now();
                let bytes = fs::read(
                    session
                        .shared
                        .cfg
                        .directory
                        .join("objects")
                        .join(revision.object),
                )?;
                assert_eq!(
                    (bytes.len(), crc32fast::hash(&bytes)),
                    (path.size, path.crc)
                );
                content = String::from_utf8(bytes)?;
                session.counters.object_reads += 1;
                session.counters.object_ns += started.elapsed().as_nanos() as u64;
            }
            Ok(Response {
                generation: marker.generation,
                size: path.size,
                crc: path.crc,
                items: 1,
                content,
            })
        }
        "list" => {
            let prefix = format!("{}/", request.path.rsplit_once('/').unwrap().0);
            session.view(&[WS, PATHS], &request.workspace, |reader| {
                let marker: Marker = decode(
                    &reader
                        .get(WS, &workspace_key(&request.workspace))?
                        .unwrap()
                        .value,
                )?;
                let rows = reader.list(PATHS, prefix.as_bytes(), 64)?;
                let mut crc = crc32fast::Hasher::new();
                for (key, value) in &rows {
                    crc.update(key);
                    crc.update(value);
                }
                let entries = rows
                    .iter()
                    .map(|(key, value)| {
                        Ok((String::from_utf8(key.clone())?, decode::<PathMeta>(value)?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Response {
                    generation: marker.generation,
                    size: 0,
                    crc: crc.finalize(),
                    items: rows.len(),
                    content: serde_json::to_string(&entries)?,
                })
            })
        }
        "publish" => mutate(session, request, "publish", &request.path, &request.path),
        "rename_roundtrip" => {
            let temporary = format!("{}.moving", request.path);
            mutate(session, request, "rename_out", &request.path, &temporary)?;
            mutate(session, request, "rename_back", &temporary, &request.path)
        }
        "remove_recreate" => {
            mutate(session, request, "remove", &request.path, &request.path)?;
            mutate(session, request, "publish", &request.path, &request.path)
        }
        "snapshot" => session.view(&(0..8).collect::<Vec<_>>(), &request.workspace, |reader| {
            let mut crc = crc32fast::Hasher::new();
            let mut items = 0;
            for table in 0..8 {
                for (key, value) in
                    reader.list(table, format!("{}/", request.workspace).as_bytes(), 32)?
                {
                    crc.update(&key);
                    crc.update(&value);
                    items += 1;
                }
            }
            let marker: Marker = decode(
                &reader
                    .get(WS, &workspace_key(&request.workspace))?
                    .unwrap()
                    .value,
            )?;
            Ok(Response {
                generation: marker.generation,
                size: 0,
                crc: crc.finalize(),
                items,
                content: String::new(),
            })
        }),
        _ => Err("unknown operation".into()),
    }
}

fn call(session: &mut Session, request: &Request) -> Result<Response> {
    let encoded = encode(request)?;
    let decoded: Request = decode(&encoded)?;
    let response = handle(session, &decoded)?;
    let result = decode::<Response>(&encode(&response)?)?;
    assert_eq!(response, result);
    Ok(result)
}

fn contents(bytes: usize, token: usize) -> String {
    let mut out = vec![b'a' + (token % 26) as u8; bytes];
    for index in (0..bytes).step_by(80) {
        out[index] = b'\n';
    }
    String::from_utf8(out).unwrap()
}

fn preload(cfg: &Config) -> Result<Arc<Shared>> {
    fs::create_dir(&cfg.directory)?;
    fs::create_dir(cfg.directory.join("metadata"))?;
    fs::create_dir(cfg.directory.join("objects"))?;
    let db = if cfg.engine == "holt" {
        let mut config = TreeConfig::new(cfg.directory.join("metadata"));
        config.durability = Durability::Wal { sync: false };
        config.buffer_pool_size = 128;
        config.checkpoint.enabled = false;
        let db = DB::open(config)?;
        for table in TABLES {
            db.create_tree(table)?;
        }
        Some(db)
    } else {
        None
    };
    let connection = if db.is_none() {
        let conn = sql_open(cfg)?;
        for name in TABLES {
            conn.execute_batch(&format!("CREATE TABLE t_{name}(k BLOB PRIMARY KEY,v BLOB NOT NULL,ver INTEGER NOT NULL) WITHOUT ROWID"))?;
        }
        conn.execute_batch(
            "CREATE TABLE recovery_log(seq INTEGER PRIMARY KEY,payload BLOB NOT NULL)",
        )?;
        Some(conn)
    } else {
        None
    };
    let mut rows = Vec::<(usize, Vec<u8>, Vec<u8>)>::new();
    for ws in 0..cfg.workspaces {
        let name = format!("agent-{ws:03}");
        rows.push((WS, workspace_key(&name), encode(&Marker { generation: 0 })?));
        for file in 0..cfg.files {
            let id = format!("initial-{ws:03}-{file:06}");
            let revision = format!("{name}/revision/{id}");
            let content = contents(cfg.payload, file);
            let crc = crc32fast::hash(content.as_bytes());
            if cfg.payload > 0 {
                fs::write(cfg.directory.join("objects").join(&id), &content)?;
            }
            rows.push((
                PATHS,
                file_path(&name, file).into_bytes(),
                encode(&PathMeta {
                    revision: revision.clone(),
                    size: cfg.payload,
                    crc,
                    generation: 0,
                })?,
            ));
            rows.push((
                REV,
                revision.as_bytes().to_vec(),
                encode(&Revision {
                    object: id,
                    size: cfg.payload,
                    crc,
                })?,
            ));
            rows.push((
                REFS,
                revision.as_bytes().to_vec(),
                encode(&json!({"references": 1}))?,
            ));
        }
    }
    if let Some(db) = &db {
        for chunk in rows.chunks(256) {
            assert!(db.atomic(|batch| {
                for (table, key, value) in chunk {
                    batch.put(TABLES[*table], key, value);
                }
            })?);
        }
        db.checkpoint()?;
    } else {
        let conn = connection.as_ref().unwrap();
        conn.execute_batch("BEGIN IMMEDIATE")?;
        for (table, key, value) in &rows {
            conn.prepare_cached(&format!("INSERT INTO t_{} VALUES(?1,?2,0)", TABLES[*table]))?
                .execute(params![key, value])?;
        }
        conn.execute_batch("COMMIT; PRAGMA wal_checkpoint(TRUNCATE)")?;
    }
    drop(connection);
    // Reopen in the actual acknowledgement mode; setup is never timed.
    drop(db);
    let db = if cfg.engine == "holt" {
        let mut config = TreeConfig::new(cfg.directory.join("metadata"));
        config.durability = Durability::Wal { sync: cfg.sync };
        config.buffer_pool_size = 128;
        config.checkpoint.enabled = cfg.background;
        let db = DB::open(config)?;
        if cfg.attached {
            db.initialize_journal_stream(JournalAnchor::new(0, [0; 32]))?;
        }
        Some(db)
    } else {
        None
    };
    Ok(Arc::new(Shared {
        cfg: cfg.clone(),
        db,
        tail: Mutex::new(JournalAnchor::new(0, [0; 32])),
        version: AtomicU64::new(0),
    }))
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn request(cfg: &Config, worker: usize, index: usize, phase: &str, random: &mut u64) -> Request {
    let n = next_random(random);
    let op = match n % 100 {
        0..=39 => "read",
        40..=54 => "stat",
        55..=69 => "list",
        70..=89 => "publish",
        90..=94 => "rename_roundtrip",
        95..=97 => "remove_recreate",
        _ => "snapshot",
    };
    let workspace = format!("agent-{worker:03}");
    Request {
        op: op.to_owned(),
        path: file_path(&workspace, ((n >> 8) as usize) % cfg.files),
        workspace,
        id: format!("{phase}-{worker:03}-{index:08}"),
        content: if matches!(op, "publish" | "remove_recreate") {
            contents(cfg.payload, index)
        } else {
            String::new()
        },
    }
}

fn latency(values: &mut [u64]) -> Value {
    values.sort_unstable();
    let quantile = |p: f64| values[((values.len() - 1) as f64 * p).ceil() as usize] as f64 / 1000.0;
    json!({"count": values.len(), "p50_us": quantile(0.5), "p95_us": quantile(0.95), "p99_us": quantile(0.99), "max_us": quantile(1.0), "mean_us": values.iter().sum::<u64>() as f64 / values.len() as f64 / 1000.0})
}

fn space(directory: &Path) -> Result<Value> {
    let mut logical = 0;
    let mut allocated = 0;
    let mut files = serde_json::Map::new();
    fn walk(path: &Path, logical: &mut u64, allocated: &mut u64) -> Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                walk(&entry.path(), logical, allocated)?;
            } else {
                *logical += meta.len();
                *allocated += meta.blocks() * 512;
            }
        }
        Ok(())
    }
    walk(directory, &mut logical, &mut allocated)?;
    for entry in fs::read_dir(directory.join("metadata"))? {
        let entry = entry?;
        let meta = entry.metadata()?;
        files.insert(
            entry.file_name().to_string_lossy().into_owned(),
            json!({"logical": meta.len(), "allocated": meta.blocks() * 512}),
        );
    }
    let metadata_logical = files
        .values()
        .map(|v| v["logical"].as_u64().unwrap())
        .sum::<u64>();
    let metadata_allocated = files
        .values()
        .map(|v| v["allocated"].as_u64().unwrap())
        .sum::<u64>();
    Ok(json!({"logical": logical, "allocated": allocated,
        "metadata_logical": metadata_logical, "metadata_allocated": metadata_allocated,
        "objects_logical": logical - metadata_logical,
        "objects_allocated": allocated - metadata_allocated, "metadata_files": files}))
}

fn fingerprint(session: &mut Session) -> Result<Value> {
    let mut fingerprint = crc32fast::Hasher::new();
    let mut tables = serde_json::Map::new();
    for (table, name) in TABLES.iter().enumerate() {
        let mut count = 0;
        let mut bytes = 0;
        if let Some(conn) = &session.sqlite {
            let mut stmt = conn.prepare(&format!("SELECT k,v FROM t_{name} ORDER BY k"))?;
            let pairs = stmt.query_map([], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?;
            for pair in pairs {
                let (key, value) = pair?;
                fingerprint.update(&key);
                fingerprint.update(&value);
                count += 1;
                bytes += key.len() + value.len();
            }
            tables.insert(name.to_string(), json!({"rows": count, "bytes": bytes}));
        } else {
            for entry in session.trees[table].range() {
                if let RangeEntry::Key { key, value, .. } = entry? {
                    fingerprint.update(&key);
                    fingerprint.update(&value);
                    count += 1;
                    bytes += key.len() + value.len();
                }
            }
            let stats = session.trees[table].stats()?;
            tables.insert(
                name.to_string(),
                json!({"rows": count, "bytes": bytes, "blobs": stats.blob_count}),
            );
        }
    }
    Ok(json!({"crc32": fingerprint.finalize(), "tables": tables}))
}

fn checkpoint(shared: &Shared, keeper: &Session) -> Result<()> {
    if let Some(db) = &shared.db {
        db.checkpoint()?;
    } else {
        let conn = keeper.sqlite.as_ref().unwrap();
        conn.execute_batch("PRAGMA synchronous=FULL; PRAGMA wal_checkpoint(TRUNCATE)")?;
        if shared.cfg.attached {
            // The same local, checkpoint-bounded recovery retention as Holt.
            conn.execute_batch("DELETE FROM recovery_log; PRAGMA wal_checkpoint(TRUNCATE)")?;
        }
        if !shared.cfg.sync {
            conn.execute_batch(if shared.cfg.sqlite_normal {
                "PRAGMA synchronous=NORMAL"
            } else {
                "PRAGMA synchronous=OFF"
            })?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cfg = Config::parse()?;
    let setup = Instant::now();
    let shared = preload(&cfg)?;
    let mut keeper = Session::open(Arc::clone(&shared))?;
    // Sanity: exact replay has one commit and returns the exact response.
    let sanity = Request {
        op: "publish".into(),
        workspace: "agent-000".into(),
        path: file_path("agent-000", 0),
        id: "sanity".into(),
        content: contents(cfg.payload, 7),
    };
    let first = call(&mut keeper, &sanity)?;
    let commits = keeper.counters.commits;
    assert_eq!(first, call(&mut keeper, &sanity)?);
    assert_eq!(commits, keeper.counters.commits);
    checkpoint(&shared, &keeper)?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let barrier = Arc::new(Barrier::new(cfg.threads + 1));
    let (ready_tx, ready_rx) = mpsc::channel();
    let mut workers = Vec::new();
    for worker in 0..cfg.threads {
        let shared = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        let ready_tx = ready_tx.clone();
        workers.push(std::thread::spawn(
            move || -> Result<(BTreeMap<String, Vec<u64>>, Counters)> {
                let mut session = Session::open(Arc::clone(&shared))?;
                let mut random = shared.cfg.seed ^ ((worker as u64 + 1) * 0x09e3_779b);
                for index in 0..shared.cfg.warmup / shared.cfg.threads {
                    let req = request(&shared.cfg, worker, index, "warmup", &mut random);
                    call(&mut session, &req)?;
                }
                session.counters = Counters::default();
                ready_tx.send(())?;
                barrier.wait();
                let mut samples: BTreeMap<String, Vec<u64>> = BTreeMap::new();
                for index in 0..shared.cfg.operations / shared.cfg.threads {
                    // JSON and all domain/storage/content work are inside timing;
                    // caller-side request generation is outside each latency sample.
                    let req = request(&shared.cfg, worker, index, "timed", &mut random);
                    let started = Instant::now();
                    std::hint::black_box(call(&mut session, &req)?);
                    samples
                        .entry(req.op)
                        .or_default()
                        .push(started.elapsed().as_nanos() as u64);
                }
                Ok((samples, session.counters))
            },
        ));
    }
    for _ in 0..cfg.threads {
        ready_rx.recv_timeout(Duration::from_secs(120))?;
    }
    checkpoint(&shared, &keeper)?;
    let before = space(&cfg.directory)?;
    let before_stats = shared.db.as_ref().map(DB::stats);
    let started = Instant::now();
    barrier.wait();
    let mut samples: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut counters = Counters::default();
    for worker in workers {
        let (worker_samples, worker_counters) = worker.join().map_err(|_| "worker panicked")??;
        for (op, values) in worker_samples {
            samples.entry(op).or_default().extend(values);
        }
        counters.view_ns += worker_counters.view_ns;
        counters.views += worker_counters.views;
        counters.apply_ns += worker_counters.apply_ns;
        counters.commits += worker_counters.commits;
        counters.object_ns += worker_counters.object_ns;
        counters.object_reads += worker_counters.object_reads;
        counters.object_writes += worker_counters.object_writes;
        counters.retries += worker_counters.retries;
    }
    let elapsed = started.elapsed();
    let before_checkpoint = space(&cfg.directory)?;
    let checkpoint_start = Instant::now();
    checkpoint(&shared, &keeper)?;
    let checkpoint_elapsed = checkpoint_start.elapsed();
    let after = space(&cfg.directory)?;
    let final_fingerprint = fingerprint(&mut keeper)?;
    let stats = if let Some(db) = &shared.db {
        let stats = db.stats();
        let old = before_stats.as_ref().unwrap();
        json!({
            "walker_ops": stats.bm_walker_ops - old.bm_walker_ops,
            "walker_blob_hops": stats.bm_walker_blob_hops - old.bm_walker_blob_hops,
            "spillovers": stats.bm_spillovers - old.bm_spillovers,
            "live_slots": stats.store.live_slots, "next_slot": stats.store.next_slot,
            "orphan_backlog": stats.bm_gc_orphan_backlog_count,
            "wal_appends": stats.journal.as_ref().unwrap().appends - old.journal.as_ref().unwrap().appends,
            "wal_syncs": stats.journal.as_ref().unwrap().syncs - old.journal.as_ref().unwrap().syncs,
        })
    } else {
        json!({})
    };
    let mut all = samples.values().flatten().copied().collect::<Vec<_>>();
    let operations = all.len();
    let aggregate_latency = latency(&mut all);
    let per_op = samples
        .iter_mut()
        .map(|(name, samples)| (name.clone(), latency(samples)))
        .collect::<serde_json::Map<_, _>>();
    // Clean reopen verifies persisted domain state for both engines.
    drop(keeper);
    let shared = Arc::try_unwrap(shared).ok().ok_or("shared handle leaked")?;
    drop(shared);
    let reopen_start = Instant::now();
    let reopened = if cfg.engine == "holt" {
        let mut config = TreeConfig::new(cfg.directory.join("metadata"));
        config.checkpoint.enabled = false;
        config.buffer_pool_size = 128;
        Some(DB::open(config)?)
    } else {
        None
    };
    let reopened = Arc::new(Shared {
        cfg: cfg.clone(),
        db: reopened,
        tail: Mutex::new(JournalAnchor::new(0, [0; 32])),
        version: AtomicU64::new(0),
    });
    let mut session = Session::open(reopened)?;
    let reopen_ms = reopen_start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        final_fingerprint,
        fingerprint(&mut session)?,
        "reopen state differs"
    );
    println!(
        "{}",
        json!({
            "schema": "holt-sqlite.agent-workspace.v1", "engine": cfg.engine,
            "sqlite_version": rusqlite::version(), "sync": cfg.sync, "attached": cfg.attached,
            "threads": cfg.threads, "operations": operations, "warmup": cfg.warmup,
            "workspaces": cfg.workspaces, "files_per_workspace": cfg.files, "payload_bytes": cfg.payload,
        "read_mode": cfg.read_mode, "seed": cfg.seed,
        "background_checkpoints": cfg.background,
        "sqlite_synchronous": if cfg.sync {"FULL"} else if cfg.sqlite_normal {"NORMAL"} else {"OFF"},
        "holt_cache_target_bytes": 64 * 1024 * 1024,
        "sqlite_cache_target_bytes_per_connection": 64 * 1024 * 1024,
            "elapsed_seconds": elapsed.as_secs_f64(), "ops_per_second": operations as f64 / elapsed.as_secs_f64(),
            "drain_inclusive_ops_per_second": operations as f64 / (elapsed + checkpoint_elapsed).as_secs_f64(),
            "checkpoint_ms": checkpoint_elapsed.as_secs_f64() * 1000.0, "reopen_ms": reopen_ms,
        "setup_seconds": setup_seconds, "latency": aggregate_latency, "per_operation": per_op,
        "latencies_ns": samples,
            "stages": { "view_calls": counters.views, "view_ms": counters.view_ns as f64 / 1e6,
                "metadata_commits": counters.commits, "apply_ms": counters.apply_ns as f64 / 1e6,
                "object_reads": counters.object_reads, "object_writes": counters.object_writes,
                "object_ms": counters.object_ns as f64 / 1e6, "cas_retries": counters.retries },
            "space_before": before, "space_before_checkpoint": before_checkpoint, "space_after": after,
            "fingerprint": final_fingerprint, "holt_stats": stats, "verified_reopen": true,
            "qualification": "Shared application workflow; local metadata/content/JSON. No NoKV RPC, network, remote object store, or power-loss test. Checkpoint mode is recorded explicitly."
        })
    );
    Ok(())
}
