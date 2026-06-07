use std::env;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use holt::{CheckpointConfig, KeyRangeEntryRef, Tree, TreeConfig, DB};

type DynError = Box<dyn std::error::Error + Send + Sync + 'static>;
type Result<T> = std::result::Result<T, DynError>;

const DB_SOAK_TREES: [&str; 4] = ["objects", "inodes", "locks", "sessions"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Normal,
    DbNormal,
    Crash,
    DbCrash,
    Child,
    DbChild,
}

#[derive(Debug)]
struct Config {
    mode: Mode,
    dir: PathBuf,
    duration: Duration,
    keys: usize,
    ops: usize,
    threads: usize,
    buffer_pool: usize,
    wal_sync: bool,
    reset: bool,
    seed: u64,
    kill_min: Duration,
    kill_max: Duration,
}

impl Config {
    fn parse() -> Result<Self> {
        let mut cfg = Self {
            mode: Mode::Normal,
            dir: PathBuf::from("target/holt-soak"),
            duration: Duration::from_secs(60),
            keys: 100_000,
            ops: 1_000_000,
            threads: 4,
            buffer_pool: 256,
            wal_sync: false,
            reset: false,
            seed: 0xD15E_A5ED_500A_0001,
            kill_min: Duration::from_millis(100),
            kill_max: Duration::from_millis(2_000),
        };

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" => {
                    cfg.mode = match take(&mut args, "--mode")?.as_str() {
                        "normal" => Mode::Normal,
                        "db" | "db-normal" => Mode::DbNormal,
                        "crash" => Mode::Crash,
                        "db-crash" => Mode::DbCrash,
                        "child" | "crash-child" => Mode::Child,
                        "db-child" | "db-crash-child" => Mode::DbChild,
                        other => return Err(format!("unknown --mode `{other}`").into()),
                    };
                }
                "--dir" => cfg.dir = PathBuf::from(take(&mut args, "--dir")?),
                "--duration-secs" => {
                    cfg.duration =
                        Duration::from_secs(parse_u64(&take(&mut args, "--duration-secs")?)?)
                }
                "--keys" => cfg.keys = parse_usize(&take(&mut args, "--keys")?)?,
                "--ops" => cfg.ops = parse_usize(&take(&mut args, "--ops")?)?,
                "--threads" => cfg.threads = parse_usize(&take(&mut args, "--threads")?)?.max(1),
                "--buffer-pool" => {
                    cfg.buffer_pool = parse_usize(&take(&mut args, "--buffer-pool")?)?.max(1)
                }
                "--wal-sync" => cfg.wal_sync = parse_bool(&take(&mut args, "--wal-sync")?)?,
                "--reset" => cfg.reset = true,
                "--seed" => cfg.seed = parse_u64(&take(&mut args, "--seed")?)?,
                "--kill-min-ms" => {
                    cfg.kill_min =
                        Duration::from_millis(parse_u64(&take(&mut args, "--kill-min-ms")?)?)
                }
                "--kill-max-ms" => {
                    cfg.kill_max =
                        Duration::from_millis(parse_u64(&take(&mut args, "--kill-max-ms")?)?)
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument `{other}`").into()),
            }
        }
        if cfg.keys == 0 {
            return Err("--keys must be greater than zero".into());
        }
        if cfg.kill_max < cfg.kill_min {
            return Err("--kill-max-ms must be >= --kill-min-ms".into());
        }
        Ok(cfg)
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("holt-soak: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cfg = Config::parse()?;
    match cfg.mode {
        Mode::Normal => run_normal(&cfg),
        Mode::DbNormal => run_db_normal(&cfg),
        Mode::Crash => run_crash_parent(&cfg),
        Mode::DbCrash => run_db_crash_parent(&cfg),
        Mode::Child => run_crash_child(&cfg),
        Mode::DbChild => run_db_crash_child(&cfg),
    }
}

fn run_normal(cfg: &Config) -> Result<()> {
    prepare_dir(cfg)?;
    let tree = Arc::new(open_tree(cfg)?);
    let oracle = Arc::new((0..cfg.keys).map(|_| AtomicU64::new(0)).collect::<Vec<_>>());
    let deadline = Instant::now() + cfg.duration;
    let ops_per_thread = cfg.ops.div_ceil(cfg.threads);
    let mut handles = Vec::with_capacity(cfg.threads);

    for tid in 0..cfg.threads {
        let tree = Arc::clone(&tree);
        let oracle = Arc::clone(&oracle);
        let cfg = cfg.clone_for_thread();
        handles.push(thread::spawn(move || -> Result<u64> {
            let mut rng = Rng::new(cfg.seed ^ ((tid as u64) << 32) ^ 0x9E37_79B9_7F4A_7C15);
            let mut done = 0u64;
            while done < ops_per_thread as u64 && Instant::now() < deadline {
                let idx = partitioned_index(&mut rng, cfg.keys, cfg.threads, tid);
                let roll = rng.next_u64() % 100;
                if roll < 50 {
                    let rev = make_revision(tid, done);
                    tree.put(&key(idx), &value(idx, rev))?;
                    oracle[idx].store(rev, Ordering::Release);
                } else if roll < 75 {
                    verify_one(&tree, &oracle, idx)?;
                } else if roll < 85 {
                    tree.delete(&key(idx))?;
                    oracle[idx].store(0, Ordering::Release);
                } else if roll < 95 {
                    scan_small_prefix(&tree, idx)?;
                } else {
                    let rev = make_revision(tid, done);
                    let k = key(idx);
                    let v = value(idx, rev);
                    tree.atomic(|batch| batch.put(&k, &v))?;
                    oracle[idx].store(rev, Ordering::Release);
                }
                done += 1;
            }
            Ok(done)
        }));
    }

    let mut total_ops = 0u64;
    for handle in handles {
        total_ops += handle.join().map_err(|_| "worker thread panicked")??;
    }
    tree.checkpoint()?;
    print_stats("normal-pre-reopen", &tree, total_ops)?;
    drop(tree);

    let reopened = open_tree(cfg)?;
    verify_oracle(&reopened, &oracle)?;
    print_stats("normal-post-reopen", &reopened, total_ops)?;
    Ok(())
}

fn run_db_normal(cfg: &Config) -> Result<()> {
    prepare_dir(cfg)?;
    let db = Arc::new(open_db(cfg)?);
    for name in DB_SOAK_TREES {
        db.open_or_create_tree(name)?;
    }

    let oracle = Arc::new(
        DB_SOAK_TREES
            .iter()
            .map(|_| (0..cfg.keys).map(|_| AtomicU64::new(0)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );
    let deadline = Instant::now() + cfg.duration;
    let ops_per_thread = cfg.ops.div_ceil(cfg.threads);
    let mut handles = Vec::with_capacity(cfg.threads);

    for tid in 0..cfg.threads {
        let db = Arc::clone(&db);
        let oracle = Arc::clone(&oracle);
        let cfg = cfg.clone_for_thread();
        handles.push(thread::spawn(move || -> Result<u64> {
            let mut rng = Rng::new(cfg.seed ^ ((tid as u64) << 32) ^ 0xDBDB_0000_500A_0001);
            let mut done = 0u64;
            while done < ops_per_thread as u64 && Instant::now() < deadline {
                let tree_slot = (rng.next_u64() as usize) % DB_SOAK_TREES.len();
                let idx = partitioned_index(&mut rng, cfg.keys, cfg.threads, tid);
                let roll = rng.next_u64() % 100;
                if roll < 45 {
                    let rev = make_revision(tid, done);
                    let key = key(idx);
                    let value = value(idx, rev);
                    db.atomic(|batch| batch.put(DB_SOAK_TREES[tree_slot], &key, &value))
                        .map_err(|e| {
                            format!(
                                "db-normal put tree={} idx={idx} done={done}: {e}",
                                DB_SOAK_TREES[tree_slot]
                            )
                        })?;
                    oracle[tree_slot][idx].store(rev, Ordering::Release);
                } else if roll < 65 {
                    verify_db_one(&db, &oracle, tree_slot, idx).map_err(|e| {
                        format!(
                            "db-normal get tree={} idx={idx} done={done}: {e}",
                            DB_SOAK_TREES[tree_slot]
                        )
                    })?;
                } else if roll < 75 {
                    let key = key(idx);
                    db.atomic(|batch| batch.delete(DB_SOAK_TREES[tree_slot], &key))
                        .map_err(|e| {
                            format!(
                                "db-normal delete tree={} idx={idx} done={done}: {e}",
                                DB_SOAK_TREES[tree_slot]
                            )
                        })?;
                    oracle[tree_slot][idx].store(0, Ordering::Release);
                } else if roll < 88 {
                    scan_db_small_prefix(&db, tree_slot, idx).map_err(|e| {
                        format!(
                            "db-normal scan tree={} idx={idx} done={done}: {e}",
                            DB_SOAK_TREES[tree_slot]
                        )
                    })?;
                } else if roll < 96 {
                    let other =
                        (tree_slot + 1 + (rng.next_u64() as usize % (DB_SOAK_TREES.len() - 1)))
                            % DB_SOAK_TREES.len();
                    let rev = make_revision(tid, done);
                    let other_rev = rev ^ 0x5A5A_0000_0000_0000;
                    let key = key(idx);
                    let primary_value = value(idx, rev);
                    let other_value = value(idx, other_rev);
                    db.atomic(|batch| {
                        batch.put(DB_SOAK_TREES[tree_slot], &key, &primary_value);
                        batch.put(DB_SOAK_TREES[other], &key, &other_value);
                    })
                    .map_err(|e| {
                        format!(
                            "db-normal cross-put tree={} other={} idx={idx} done={done}: {e}",
                            DB_SOAK_TREES[tree_slot], DB_SOAK_TREES[other]
                        )
                    })?;
                    oracle[tree_slot][idx].store(rev, Ordering::Release);
                    oracle[other][idx].store(other_rev, Ordering::Release);
                } else {
                    view_db_prefix(&db, tree_slot, idx).map_err(|e| {
                        format!(
                            "db-normal view tree={} idx={idx} done={done}: {e}",
                            DB_SOAK_TREES[tree_slot]
                        )
                    })?;
                }
                done += 1;
            }
            Ok(done)
        }));
    }

    let mut total_ops = 0u64;
    for handle in handles {
        total_ops += handle.join().map_err(|_| "db worker thread panicked")??;
    }
    db.checkpoint()?;
    print_db_stats("db-normal-pre-reopen", &db, total_ops);
    drop(db);

    let reopened = open_db(cfg)?;
    verify_db_oracle(&reopened, &oracle)?;
    print_db_stats("db-normal-post-reopen", &reopened, total_ops);
    Ok(())
}

fn run_crash_parent(cfg: &Config) -> Result<()> {
    if !cfg.wal_sync {
        return Err("crash mode requires --wal-sync true so acked ops are durable".into());
    }
    prepare_dir(cfg)?;
    let ack_path = ack_log_path(&cfg.dir);
    let deadline = Instant::now() + cfg.duration;
    let exe = env::current_exe()?;
    let mut rng = Rng::new(cfg.seed ^ 0xC0FF_EE00_5150_0001);
    let mut rounds = 0u64;

    while Instant::now() < deadline {
        let mut child = Command::new(&exe)
            .arg("--mode")
            .arg("child")
            .arg("--dir")
            .arg(&cfg.dir)
            .arg("--keys")
            .arg(cfg.keys.to_string())
            .arg("--ops")
            .arg(cfg.ops.to_string())
            .arg("--buffer-pool")
            .arg(cfg.buffer_pool.to_string())
            .arg("--wal-sync")
            .arg("true")
            .arg("--seed")
            .arg((cfg.seed ^ rounds).to_string())
            .spawn()?;
        let sleep_for = random_duration(&mut rng, cfg.kill_min, cfg.kill_max);
        thread::sleep(sleep_for);
        let _ = child.kill();
        let _ = child.wait();
        rounds += 1;

        let expected = load_ack_log(&ack_path)?;
        let tree = open_tree(cfg)?;
        verify_ack_entries(&tree, &expected)?;
        print_stats("crash-reopen", &tree, rounds)?;
    }
    Ok(())
}

fn run_crash_child(cfg: &Config) -> Result<()> {
    fs::create_dir_all(&cfg.dir)?;
    let tree = open_tree(cfg)?;
    let mut ack = OpenOptions::new()
        .create(true)
        .append(true)
        .open(ack_log_path(&cfg.dir))?;
    let deadline = Instant::now() + cfg.duration;
    let mut rng = Rng::new(cfg.seed);
    let mut done = 0u64;

    while done < cfg.ops as u64 && Instant::now() < deadline {
        let rev = rng.next_u64().max(1);
        let key = format!("crash/{:016x}/{done:016x}", cfg.seed);
        tree.put(key.as_bytes(), &crash_value(rev))?;
        writeln!(ack, "P {key} {rev}")?;
        ack.sync_data()?;
        done += 1;
    }
    Ok(())
}

fn run_db_crash_parent(cfg: &Config) -> Result<()> {
    if !cfg.wal_sync {
        return Err(
            "db-crash mode requires --wal-sync true so acked transactions are durable".into(),
        );
    }
    prepare_dir(cfg)?;
    let ack_path = db_ack_log_path(&cfg.dir);
    let deadline = Instant::now() + cfg.duration;
    let exe = env::current_exe()?;
    let mut rng = Rng::new(cfg.seed ^ 0xD8C4_A5ED_500A_0001);
    let mut rounds = 0u64;

    while Instant::now() < deadline {
        let mut child = Command::new(&exe)
            .arg("--mode")
            .arg("db-child")
            .arg("--dir")
            .arg(&cfg.dir)
            .arg("--keys")
            .arg(cfg.keys.to_string())
            .arg("--ops")
            .arg(cfg.ops.to_string())
            .arg("--buffer-pool")
            .arg(cfg.buffer_pool.to_string())
            .arg("--wal-sync")
            .arg("true")
            .arg("--seed")
            .arg((cfg.seed ^ rounds).to_string())
            .spawn()?;
        let sleep_for = random_duration(&mut rng, cfg.kill_min, cfg.kill_max);
        thread::sleep(sleep_for);
        let _ = child.kill();
        let _ = child.wait();
        rounds += 1;

        let expected = load_db_ack_log(&ack_path)?;
        let db = open_db(cfg)?;
        verify_db_ack_entries(&db, &expected)?;
        print_db_stats("db-crash-reopen", &db, rounds);
    }
    Ok(())
}

fn run_db_crash_child(cfg: &Config) -> Result<()> {
    fs::create_dir_all(&cfg.dir)?;
    let db = open_db(cfg)?;
    db.open_or_create_tree("objects")?;
    db.open_or_create_tree("inodes")?;
    let mut ack = OpenOptions::new()
        .create(true)
        .append(true)
        .open(db_ack_log_path(&cfg.dir))?;
    let deadline = Instant::now() + cfg.duration;
    let mut rng = Rng::new(cfg.seed);
    let mut done = 0u64;

    while done < cfg.ops as u64 && Instant::now() < deadline {
        let object_rev = rng.next_u64().max(1);
        let inode_rev = rng.next_u64().max(1);
        let key = format!("db-crash/{:016x}/{done:016x}", cfg.seed);
        let object_value = crash_value(object_rev);
        let inode_value = crash_value(inode_rev);
        db.atomic(|batch| {
            batch.put("objects", key.as_bytes(), &object_value);
            batch.put("inodes", key.as_bytes(), &inode_value);
        })?;
        writeln!(ack, "D {key} {object_rev} {inode_rev}")?;
        ack.sync_data()?;
        done += 1;
    }
    Ok(())
}

fn open_tree(cfg: &Config) -> holt::Result<Tree> {
    let mut tree_cfg = TreeConfig::new(&cfg.dir);
    tree_cfg.buffer_pool_size = cfg.buffer_pool;
    tree_cfg.durability = holt::Durability::Wal { sync: cfg.wal_sync };
    tree_cfg.checkpoint = CheckpointConfig::default();
    Tree::open(tree_cfg)
}

fn open_db(cfg: &Config) -> holt::Result<DB> {
    let mut tree_cfg = TreeConfig::new(&cfg.dir);
    tree_cfg.buffer_pool_size = cfg.buffer_pool;
    tree_cfg.durability = holt::Durability::Wal { sync: cfg.wal_sync };
    tree_cfg.checkpoint = CheckpointConfig::default();
    DB::open(tree_cfg)
}

fn prepare_dir(cfg: &Config) -> Result<()> {
    if cfg.reset && cfg.dir.exists() {
        let name = cfg
            .dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if !name.contains("holt-soak") {
            return Err(format!(
                "refusing --reset on `{}`; path basename must contain `holt-soak`",
                cfg.dir.display()
            )
            .into());
        }
        fs::remove_dir_all(&cfg.dir)?;
    }
    fs::create_dir_all(&cfg.dir)?;
    Ok(())
}

fn verify_one(tree: &Tree, oracle: &[AtomicU64], idx: usize) -> Result<()> {
    let expected = oracle[idx].load(Ordering::Acquire);
    let got = tree.get(&key(idx))?;
    if expected == 0 {
        if got.is_some() {
            return Err(format!("key {idx} exists but oracle says deleted").into());
        }
    } else if got.as_deref() != Some(value(idx, expected).as_slice()) {
        return Err(format!("key {idx} value mismatch").into());
    }
    Ok(())
}

fn verify_oracle(tree: &Tree, oracle: &[AtomicU64]) -> Result<()> {
    for idx in 0..oracle.len() {
        verify_one(tree, oracle, idx)?;
    }
    Ok(())
}

fn verify_db_one(db: &DB, oracle: &[Vec<AtomicU64>], tree_slot: usize, idx: usize) -> Result<()> {
    let expected = oracle[tree_slot][idx].load(Ordering::Acquire);
    let tree = db.open_tree(DB_SOAK_TREES[tree_slot])?;
    let got = tree.get(&key(idx))?;
    if expected == 0 {
        if got.is_some() {
            return Err(format!(
                "tree `{}` key {idx} exists but oracle says deleted",
                DB_SOAK_TREES[tree_slot]
            )
            .into());
        }
    } else if got.as_deref() != Some(value(idx, expected).as_slice()) {
        return Err(format!(
            "tree `{}` key {idx} value mismatch",
            DB_SOAK_TREES[tree_slot]
        )
        .into());
    }
    Ok(())
}

fn verify_db_oracle(db: &DB, oracle: &[Vec<AtomicU64>]) -> Result<()> {
    for (tree_slot, tree_oracle) in oracle.iter().enumerate() {
        for idx in 0..tree_oracle.len() {
            verify_db_one(db, oracle, tree_slot, idx)?;
        }
    }
    Ok(())
}

fn verify_ack_entries(tree: &Tree, expected: &[(Vec<u8>, u64)]) -> Result<()> {
    for (key, rev) in expected {
        let got = tree.get(key)?;
        if got.as_deref() != Some(crash_value(*rev).as_slice()) {
            return Err(format!(
                "acknowledged crash key `{}` lost revision {rev}",
                String::from_utf8_lossy(key)
            )
            .into());
        }
    }
    Ok(())
}

fn verify_db_ack_entries(db: &DB, expected: &[(Vec<u8>, u64, u64)]) -> Result<()> {
    if expected.is_empty() {
        return Ok(());
    }
    let objects = db.open_tree("objects")?;
    let inodes = db.open_tree("inodes")?;
    for (key, object_rev, inode_rev) in expected {
        let object = objects.get(key)?;
        if object.as_deref() != Some(crash_value(*object_rev).as_slice()) {
            return Err(format!(
                "acknowledged DB object key `{}` lost revision {object_rev}",
                String::from_utf8_lossy(key)
            )
            .into());
        }
        let inode = inodes.get(key)?;
        if inode.as_deref() != Some(crash_value(*inode_rev).as_slice()) {
            return Err(format!(
                "acknowledged DB inode key `{}` lost revision {inode_rev}",
                String::from_utf8_lossy(key)
            )
            .into());
        }
    }
    Ok(())
}

fn scan_small_prefix(tree: &Tree, idx: usize) -> Result<()> {
    let prefix = format!("bucket-{:03}/", idx % 256);
    let mut seen = 0usize;
    tree.scan_keys(prefix.as_bytes())
        .delimiter(b'/')
        .visit(16, |entry| {
            match entry {
                KeyRangeEntryRef::Key { .. } | KeyRangeEntryRef::CommonPrefix(_) => {
                    seen += 1;
                }
                _ => {}
            }
            Ok(())
        })?;
    std::hint::black_box(seen);
    Ok(())
}

fn scan_db_small_prefix(db: &DB, tree_slot: usize, idx: usize) -> Result<()> {
    let tree = db.open_tree(DB_SOAK_TREES[tree_slot])?;
    scan_small_prefix(&tree, idx)
}

fn view_db_prefix(db: &DB, tree_slot: usize, idx: usize) -> Result<()> {
    let prefix = format!("bucket-{:03}/", idx % 256);
    let scopes = [(DB_SOAK_TREES[tree_slot], prefix.as_bytes())];
    db.view(&scopes, |view| {
        let Some(tree) = view.tree(DB_SOAK_TREES[tree_slot]) else {
            return Err(holt::Error::Internal("missing DB soak view tree"));
        };
        let mut seen = 0usize;
        tree.scan_keys(prefix.as_bytes())?
            .delimiter(b'/')
            .visit(16, |entry| {
                match entry {
                    KeyRangeEntryRef::Key { .. } | KeyRangeEntryRef::CommonPrefix(_) => {
                        seen += 1;
                    }
                    _ => {}
                }
                Ok(())
            })?;
        std::hint::black_box(seen);
        Ok(())
    })?;
    Ok(())
}

fn load_ack_log(path: &Path) -> Result<Vec<(Vec<u8>, u64)>> {
    let mut expected = Vec::new();
    let Ok(file) = OpenOptions::new().read(true).open(path) else {
        return Ok(expected);
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next(), fields.next(), fields.next()) {
            (Some("P"), Some(key), Some(rev), None) => {
                let rev = parse_u64(rev)?;
                expected.push((key.as_bytes().to_vec(), rev));
            }
            _ => {
                break;
            }
        }
    }
    Ok(expected)
}

fn load_db_ack_log(path: &Path) -> Result<Vec<(Vec<u8>, u64, u64)>> {
    let mut expected = Vec::new();
    let Ok(file) = OpenOptions::new().read(true).open(path) else {
        return Ok(expected);
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        let mut fields = line.split_whitespace();
        match (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) {
            (Some("D"), Some(key), Some(object_rev), Some(inode_rev), None) => {
                let object_rev = parse_u64(object_rev)?;
                let inode_rev = parse_u64(inode_rev)?;
                expected.push((key.as_bytes().to_vec(), object_rev, inode_rev));
            }
            _ => {
                break;
            }
        }
    }
    Ok(expected)
}

fn print_stats(label: &str, tree: &Tree, progress: u64) -> Result<()> {
    let s = tree.stats()?;
    let journal_debt = s.journal.map_or(0, |j| j.checkpoint_debt);
    let pending_work = s.journal.map_or(0, |j| j.pending_work);
    let ck_failed = s.checkpointer.map_or(0, |c| c.rounds_failed);
    println!(
        "{{\"event\":\"stats\",\"label\":\"{label}\",\"progress\":{progress},\
         \"blobs\":{},\"dirty\":{},\"pending_delete\":{},\"bm_hits\":{},\
         \"bm_misses\":{},\"route_hits\":{},\"route_misses\":{},\
         \"wal_pending_work\":{pending_work},\"wal_checkpoint_debt\":{journal_debt},\
         \"checkpoint_failed\":{ck_failed},\"replay_records\":{},\
         \"replay_micros\":{}}}",
        s.blob_count,
        s.bm_dirty_count,
        s.bm_pending_delete_count,
        s.bm_cache_hits,
        s.bm_cache_misses,
        s.route_cache.hits,
        s.route_cache.misses,
        s.open.wal_replay_records,
        s.open.wal_replay_micros,
    );
    Ok(())
}

fn print_db_stats(label: &str, db: &DB, progress: u64) {
    let s = db.stats();
    let journal_debt = s.journal.map_or(0, |j| j.checkpoint_debt);
    let pending_work = s.journal.map_or(0, |j| j.pending_work);
    let ck_failed = s.checkpointer.map_or(0, |c| c.rounds_failed);
    println!(
        "{{\"event\":\"stats\",\"label\":\"{label}\",\"progress\":{progress},\
         \"open_trees\":{},\"dirty\":{},\"pending_delete\":{},\"bm_hits\":{},\
         \"bm_misses\":{},\"walker_ops\":{},\"max_blob_hops\":{},\
         \"wal_pending_work\":{pending_work},\"wal_checkpoint_debt\":{journal_debt},\
         \"checkpoint_failed\":{ck_failed},\"replay_records\":{},\
         \"replay_micros\":{}}}",
        s.open_tree_count,
        s.bm_dirty_count,
        s.bm_pending_delete_count,
        s.bm_cache_hits,
        s.bm_cache_misses,
        s.bm_walker_ops,
        s.bm_max_blob_hops,
        s.open.wal_replay_records,
        s.open.wal_replay_micros,
    );
}

fn partitioned_index(rng: &mut Rng, keys: usize, threads: usize, tid: usize) -> usize {
    let slots = keys.div_ceil(threads);
    loop {
        let idx = tid + threads * ((rng.next_u64() as usize) % slots);
        if idx < keys {
            return idx;
        }
    }
}

fn key(idx: usize) -> Vec<u8> {
    format!(
        "bucket-{bucket:03}/tenant-{tenant:02}/path/object-{idx:010}.bin",
        bucket = idx % 256,
        tenant = idx % 32,
    )
    .into_bytes()
}

fn value(idx: usize, rev: u64) -> Vec<u8> {
    format!("{{\"idx\":{idx},\"rev\":{rev},\"size\":{}}}", idx * 4096).into_bytes()
}

fn crash_value(rev: u64) -> Vec<u8> {
    format!("crash-value:{rev}").into_bytes()
}

fn make_revision(tid: usize, op: u64) -> u64 {
    ((tid as u64) << 48) | (op + 1)
}

fn ack_log_path(dir: &Path) -> PathBuf {
    dir.join("soak-ack.log")
}

fn db_ack_log_path(dir: &Path) -> PathBuf {
    dir.join("soak-db-ack.log")
}

fn random_duration(rng: &mut Rng, min: Duration, max: Duration) -> Duration {
    let min_ms = min.as_millis() as u64;
    let max_ms = max.as_millis() as u64;
    let span = max_ms.saturating_sub(min_ms).saturating_add(1);
    Duration::from_millis(min_ms + (rng.next_u64() % span))
}

fn take(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value").into())
}

fn parse_bool(s: &str) -> Result<bool> {
    match s {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("invalid bool `{other}`").into()),
    }
}

fn parse_u64(s: &str) -> Result<u64> {
    s.parse::<u64>()
        .map_err(|e| format!("invalid integer `{s}`: {e}").into())
}

fn parse_usize(s: &str) -> Result<usize> {
    s.parse::<usize>()
        .map_err(|e| format!("invalid integer `{s}`: {e}").into())
}

fn print_help() {
    println!(
        "holt-soak\n\n\
         Modes:\n\
           --mode normal    multi-thread read/write/list/reopen validation\n\
           --mode db-normal multi-tree DB atomic/view/reopen validation\n\
           --mode crash     parent process repeatedly SIGKILLs child writers\n\
           --mode db-crash  parent process SIGKILLs cross-tree DB atomic writers\n\
           --mode child     internal crash-test child mode\n\n\
         Common options:\n\
           --dir PATH --duration-secs N --keys N --ops N --threads N\n\
           --buffer-pool N --wal-sync true|false --reset --seed N\n\n\
         Crash options:\n\
           --kill-min-ms N --kill-max-ms N\n"
    );
}

impl Config {
    fn clone_for_thread(&self) -> Self {
        Self {
            mode: self.mode,
            dir: self.dir.clone(),
            duration: self.duration,
            keys: self.keys,
            ops: self.ops,
            threads: self.threads,
            buffer_pool: self.buffer_pool,
            wal_sync: self.wal_sync,
            reset: false,
            seed: self.seed,
            kill_min: self.kill_min,
            kill_max: self.kill_max,
        }
    }
}

struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}
