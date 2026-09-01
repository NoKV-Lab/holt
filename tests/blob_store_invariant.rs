//! Invariant reproducer for NoKV issue #493:
//! "FileBlobStore::Manifest::duplicate slot" on reopen of a store whose
//! manifest.log is physically intact.
//!
//! Independently parses manifest.log from disk (b"MLG1" + u32le body_len +
//! u8 ty {1=Set: 16B guid + 8B le slot, 2=Delete: 16B guid} + u32le crc32
//! over magic..body) and replays it the same way `Manifest::load_or_create`
//! + `ReusableSlots::reconstruct` would, asserting:
//! - (1) FINAL state: no two live guids map to the same slot (= the #493
//!   corruption; reopen would fail).
//! - (2) EVERY PREFIX ending on a record boundary: same property. Any
//!   prefix P whose replay yields a duplicate-slot state is a SIGKILL
//!   landmine: a kill right after P is durable makes the next open fail.

use holt::{CheckpointConfig, Durability, Tree, TreeBuilder, TreeConfig, DB};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

// ---------- manifest.log parser / replayer ----------

const MAGIC: [u8; 4] = *b"MLG1";
const HEADER: usize = 4 + 4 + 1;
const FOOTER: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Guid([u8; 16]);

impl std::fmt::Debug for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum Rec {
    Set { guid: Guid, slot: u64 },
    Delete { guid: Guid },
}

#[derive(Debug)]
struct Conflict {
    record_idx: usize,
    slot: u64,
    incoming: Guid,
    holder: Guid,
    resolved_at: Option<usize>,
}

#[derive(Debug)]
struct Report {
    path: PathBuf,
    records: Vec<Rec>,
    prefix_conflicts: Vec<Conflict>,
    final_dup_slots: Vec<u64>,
    torn_tail_bytes: u64,
    crc_error: bool,
}

impl Report {
    fn is_clean(&self) -> bool {
        self.prefix_conflicts.is_empty() && self.final_dup_slots.is_empty() && !self.crc_error
    }

    fn dump(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "manifest.log {} : {} records, torn_tail_bytes={}, crc_error={}",
            self.path.display(),
            self.records.len(),
            self.torn_tail_bytes,
            self.crc_error
        );
        for c in &self.prefix_conflicts {
            let _ = writeln!(
                s,
                "  PREFIX CONFLICT at record #{}: Set {:?} -> slot {} while slot {} live-held by {:?} (resolved_at={:?})",
                c.record_idx, c.incoming, c.slot, c.slot, c.holder, c.resolved_at
            );
        }
        if !self.final_dup_slots.is_empty() {
            let _ = writeln!(
                s,
                "  FINAL-STATE DUPLICATE SLOTS: {:?}",
                self.final_dup_slots
            );
        }
        let _ = writeln!(s, "  full record trail:");
        for (i, r) in self.records.iter().enumerate() {
            match r {
                Rec::Set { guid, slot } => {
                    let _ = writeln!(s, "    #{i:04} Set    {guid:?} -> slot {slot}");
                }
                Rec::Delete { guid } => {
                    let _ = writeln!(s, "    #{i:04} Delete {guid:?}");
                }
            }
        }
        s
    }
}

fn parse_records(buf: &[u8]) -> (Vec<Rec>, u64, bool) {
    let mut records = Vec::new();
    let mut offset = 0usize;
    let mut crc_error = false;
    while offset < buf.len() {
        if buf.len() - offset < HEADER {
            break; // torn header
        }
        let start = offset;
        if buf[offset..offset + 4] != MAGIC {
            crc_error = true;
            break;
        }
        let body_len = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let ty = buf[offset + 8];
        let record_len = HEADER + body_len + FOOTER;
        if buf.len() - start < record_len {
            break; // torn tail (possible mid-append read)
        }
        let body = &buf[start + HEADER..start + HEADER + body_len];
        let expected_crc = u32::from_le_bytes(
            buf[start + HEADER + body_len..start + record_len]
                .try_into()
                .unwrap(),
        );
        let actual_crc = crc32fast_hash(&buf[start..start + HEADER + body_len]);
        if expected_crc != actual_crc {
            // Could be a mid-append read of the very last record; treat as
            // torn tail rather than corruption, but note it.
            break;
        }
        match ty {
            1 => {
                if body.len() != 24 {
                    crc_error = true;
                    break;
                }
                let mut g = [0u8; 16];
                g.copy_from_slice(&body[..16]);
                let slot = u64::from_le_bytes(body[16..24].try_into().unwrap());
                records.push(Rec::Set {
                    guid: Guid(g),
                    slot,
                });
            }
            2 => {
                if body.len() != 16 {
                    crc_error = true;
                    break;
                }
                let mut g = [0u8; 16];
                g.copy_from_slice(body);
                records.push(Rec::Delete { guid: Guid(g) });
            }
            _ => {
                crc_error = true;
                break;
            }
        }
        offset = start + record_len;
    }
    (records, (buf.len() - offset) as u64, crc_error)
}

// Local crc32 (IEEE) so the checker is fully independent of holt internals.
fn crc32fast_hash(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *entry = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ u32::from(b)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn check_manifest_log(path: &Path) -> Report {
    let buf = std::fs::read(path).unwrap_or_default();
    let (records, torn_tail_bytes, crc_error) = parse_records(&buf);

    let mut entries: HashMap<Guid, u64> = HashMap::new();
    let mut slot_count: HashMap<u64, u32> = HashMap::new();
    let mut slot_holder: HashMap<u64, Guid> = HashMap::new();
    let mut prefix_conflicts: Vec<Conflict> = Vec::new();

    for (i, rec) in records.iter().enumerate() {
        match *rec {
            Rec::Set { guid, slot } => {
                if let Some(old_slot) = entries.insert(guid, slot) {
                    let c = slot_count.entry(old_slot).or_default();
                    *c = c.saturating_sub(1);
                    if *c <= 1 {
                        for conf in prefix_conflicts.iter_mut() {
                            if conf.slot == old_slot && conf.resolved_at.is_none() {
                                conf.resolved_at = Some(i);
                            }
                        }
                    }
                }
                let c = slot_count.entry(slot).or_default();
                *c += 1;
                if *c > 1 {
                    prefix_conflicts.push(Conflict {
                        record_idx: i,
                        slot,
                        incoming: guid,
                        holder: *slot_holder.get(&slot).unwrap_or(&guid),
                        resolved_at: None,
                    });
                }
                slot_holder.insert(slot, guid);
            }
            Rec::Delete { guid } => {
                if let Some(old_slot) = entries.remove(&guid) {
                    let c = slot_count.entry(old_slot).or_default();
                    *c = c.saturating_sub(1);
                    if *c <= 1 {
                        for conf in prefix_conflicts.iter_mut() {
                            if conf.slot == old_slot && conf.resolved_at.is_none() {
                                conf.resolved_at = Some(i);
                            }
                        }
                    }
                }
            }
        }
    }

    let final_dup_slots: Vec<u64> = slot_count
        .iter()
        .filter(|(_, &c)| c > 1)
        .map(|(&s, _)| s)
        .collect();

    Report {
        path: path.to_path_buf(),
        records,
        prefix_conflicts,
        final_dup_slots,
        torn_tail_bytes,
        crc_error,
    }
}

fn find_manifest_logs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().map(|n| n == "manifest.log").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out
}

/// Check every manifest.log under `root`; panic with a full dump on any
/// violation. `allow_torn` tolerates a torn tail (mid-append concurrent read).
fn assert_invariant(root: &Path, ctx: &str, allow_torn: bool) {
    for log in find_manifest_logs(root) {
        let report = check_manifest_log(&log);
        if !allow_torn {
            assert_eq!(
                report.torn_tail_bytes,
                0,
                "[{ctx}] torn tail while quiescent:\n{}",
                report.dump()
            );
        }
        assert!(
            report.is_clean(),
            "[{ctx}] MANIFEST INVARIANT VIOLATION:\n{}",
            report.dump()
        );
    }
}

fn total_records(root: &Path) -> usize {
    find_manifest_logs(root)
        .iter()
        .map(|l| check_manifest_log(l).records.len())
        .sum()
}

// ---------- workload config mirroring nokv-meta-holt ----------

fn nokv_like_config(dir: &Path, background_checkpointer: bool) -> TreeConfig {
    let mut cfg = TreeConfig::new(dir);
    cfg.durability = Durability::Wal { sync: true };
    cfg.checkpoint = CheckpointConfig {
        enabled: background_checkpointer,
        auto_vacuum: false,
        ..CheckpointConfig::default()
    };
    cfg
}

fn key(i: u64) -> Vec<u8> {
    format!("ns/bucket-{:02}/object-{:04}", i % 4, i).into_bytes()
}

fn value(seed: u64, len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for b in v.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = x as u8;
    }
    v
}

// ---------- shape (a): same-page rewrite churn, manual checkpoints ----------

#[test]
fn shape_a_rewrite_churn_few_keys() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = nokv_like_config(dir.path(), false);
    let tree = Tree::open(cfg).unwrap();

    const OPS: u64 = 3000;
    for i in 0..OPS {
        let k = key(i % 8); // few keys => same-page rewrite churn
        tree.put(&k, &value(i, 512 + (i % 7) as usize * 128))
            .unwrap();
        if i % 25 == 24 {
            tree.checkpoint().unwrap();
            assert_invariant(dir.path(), &format!("shape_a op {i}"), false);
        }
    }
    tree.checkpoint().unwrap();
    assert_invariant(dir.path(), "shape_a final", false);
    drop(tree);
    assert_invariant(dir.path(), "shape_a after drop", false);
    let n = total_records(dir.path());
    assert!(n >= 50, "vacuous run: only {n} manifest records written");
    eprintln!("shape_a: {n} manifest records checked");
}

// ---------- shape (b): puts + deletes ----------

#[test]
fn shape_b_put_delete_churn() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = nokv_like_config(dir.path(), false);
    let tree = Tree::open(cfg).unwrap();

    const OPS: u64 = 3000;
    for i in 0..OPS {
        let k = if i % 10 < 6 {
            key(i % 16)
        } else {
            key(100 + i % 600)
        };
        if i % 7 == 3 {
            let _ = tree.delete(&k).unwrap();
        } else {
            tree.put(&k, &value(i, 256 + (i % 11) as usize * 200))
                .unwrap();
        }
        if i % 23 == 22 {
            tree.checkpoint().unwrap();
            assert_invariant(dir.path(), &format!("shape_b op {i}"), false);
        }
        if i % 500 == 499 {
            let _ = tree.gc().unwrap();
            tree.checkpoint().unwrap();
            assert_invariant(dir.path(), &format!("shape_b gc op {i}"), false);
        }
    }
    tree.checkpoint().unwrap();
    assert_invariant(dir.path(), "shape_b final", false);
}

// ---------- shape (c): reopen-heavy, cross-incarnation log append ----------

#[test]
fn shape_c_reopen_interleaved() {
    let dir = tempfile::tempdir().unwrap();

    const INCARNATIONS: u64 = 30;
    const OPS_PER_INC: u64 = 120;
    let mut op = 0u64;
    for inc in 0..INCARNATIONS {
        let cfg = nokv_like_config(dir.path(), false);
        let tree = Tree::open(cfg).unwrap();
        assert_invariant(dir.path(), &format!("shape_c inc {inc} after open"), false);

        for j in 0..OPS_PER_INC {
            let k = if op % 10 < 6 {
                key(op % 6)
            } else {
                key(100 + op % 500)
            };
            if op % 13 == 5 {
                let _ = tree.delete(&k).unwrap();
            } else {
                tree.put(&k, &value(op, 384 + (op % 5) as usize * 256))
                    .unwrap();
            }
            op += 1;
            // Checkpoint only for part of the incarnation, so the tail of
            // each incarnation's work lives only in the WAL at drop time
            // and gets replayed by the NEXT open (mid-recovery churn).
            if j < OPS_PER_INC / 2 && j % 20 == 19 {
                tree.checkpoint().unwrap();
                assert_invariant(dir.path(), &format!("shape_c inc {inc} op {op}"), false);
            }
        }
        // Drop WITHOUT an explicit final checkpoint: WAL carries the tail.
        drop(tree);
        assert_invariant(dir.path(), &format!("shape_c inc {inc} after drop"), false);
    }
}

// ---------- shape (c2): reopen with background checkpointer racing ----------

#[test]
fn shape_c2_reopen_with_background_checkpointer() {
    let dir = tempfile::tempdir().unwrap();

    const INCARNATIONS: u64 = 20;
    const OPS_PER_INC: u64 = 200;
    let mut op = 0u64;
    for inc in 0..INCARNATIONS {
        let cfg = nokv_like_config(dir.path(), true);
        let tree = Tree::open(cfg).unwrap();
        for _ in 0..OPS_PER_INC {
            let k = if op % 10 < 6 {
                key(op % 6)
            } else {
                key(100 + op % 500)
            };
            if op % 17 == 9 {
                let _ = tree.delete(&k).unwrap();
            } else {
                tree.put(&k, &value(op, 300 + (op % 9) as usize * 150))
                    .unwrap();
            }
            op += 1;
        }
        drop(tree); // background checkpointer stops on drop
        assert_invariant(dir.path(), &format!("shape_c2 inc {inc} after drop"), false);
    }
}

// ---------- shape (d): concurrent writers, multiple trees, bg checkpointer ----------

#[test]
fn shape_d_concurrent_multi_tree() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = nokv_like_config(dir.path(), true);
    let db = DB::open(cfg).unwrap();

    let trees: Vec<Tree> = (0..3)
        .map(|t| db.open_or_create_tree(&format!("t{t}")).unwrap())
        .collect();

    std::thread::scope(|s| {
        for (tid, tree) in trees.iter().enumerate() {
            let tree = tree.clone();
            s.spawn(move || {
                for i in 0..4000u64 {
                    let k = key(i % 8);
                    if i % 19 == 7 {
                        let _ = tree.delete(&k).unwrap();
                    } else {
                        tree.put(
                            &k,
                            &value(i.wrapping_mul(tid as u64 + 1), 256 + (i % 6) as usize * 128),
                        )
                        .unwrap();
                    }
                }
            });
        }
        // Poll the on-disk log while writers + background checkpointer run.
        s.spawn(|| {
            for round in 0..40 {
                std::thread::sleep(std::time::Duration::from_millis(150));
                assert_invariant(dir.path(), &format!("shape_d live round {round}"), true);
            }
        });
    });

    db.checkpoint().unwrap();
    assert_invariant(dir.path(), "shape_d final", false);
    drop(trees);
    drop(db);
    assert_invariant(dir.path(), "shape_d after drop", false);
}

// ---------- shape (e): real SIGKILL harness (child process) ----------
//
// The field failure (#493) was written under SIGKILL: serving owner killed,
// then several reopen attempts also killed mid-recovery. In-process `drop`
// is graceful (Checkpointer::drop runs a final sync round), so this shape
// re-executes the test binary as a writer child and SIGKILLs it at random
// times, including very early kills that land inside open()/WAL-replay.

#[test]
fn child_writer_entry() {
    let Ok(dir) = std::env::var("HOLT_INV_CHILD_DIR") else {
        return; // not in child mode; nothing to do
    };
    let dir = PathBuf::from(dir);
    let mut cfg = nokv_like_config(&dir, true);
    // Aggressive checkpoint cadence so manifest appends happen inside the
    // short kill windows (same code paths, higher frequency).
    cfg.checkpoint.idle_interval = std::time::Duration::from_millis(15);
    cfg.checkpoint.dirty_blob_threshold = 4;
    let tree = Tree::open(cfg).unwrap();
    let mut i: u64 = std::env::var("HOLT_INV_CHILD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    loop {
        // 70% hot 8-key churn (same-page rewrites), 30% wide 1200-key
        // spread (multi-frame tree, multi-guid slot competition).
        let k = if i % 10 < 7 {
            key(i % 8)
        } else {
            key(100 + i % 1200)
        };
        if i % 13 == 5 {
            let _ = tree.delete(&k).unwrap();
        } else {
            tree.put(&k, &value(i, 512 + (i % 5) as usize * 512))
                .unwrap();
        }
        i = i.wrapping_add(1);
    }
}

#[test]
#[ignore = "timing-based SIGKILL soak; deterministic torn-write coverage runs in the file-store process E2E"]
fn shape_e_sigkill_churn() {
    let dir = tempfile::tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();

    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    const INCARNATIONS: u32 = 40;
    for inc in 0..INCARNATIONS {
        let mut child = std::process::Command::new(&exe)
            .arg("--exact")
            .arg("child_writer_entry")
            .arg("--nocapture")
            .env("HOLT_INV_CHILD_DIR", dir.path())
            .env("HOLT_INV_CHILD_SEED", format!("{}", next() % 1000))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();

        // Every third incarnation: kill very fast, aiming at mid-recovery
        // (open + WAL replay of the previous incarnation's tail).
        let delay_ms = if inc % 3 == 2 {
            5 + next() % 60
        } else {
            80 + next() % 700
        };
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        child.kill().unwrap(); // SIGKILL on unix
        let _ = child.wait();

        // Post-kill: a torn last append is legitimate (repair_torn_tail's
        // job); duplicate-slot at any complete-record prefix is not.
        assert_invariant(dir.path(), &format!("shape_e inc {inc} post-kill"), true);

        // Half the time, run a graceful verification recovery ourselves —
        // this is exactly what the field's failed reopen attempted.
        if next() % 2 == 0 {
            let cfg = nokv_like_config(dir.path(), false);
            let tree = Tree::open(cfg).unwrap_or_else(|e| {
                let logs = find_manifest_logs(dir.path());
                let mut dumps = String::new();
                for l in logs {
                    dumps.push_str(&check_manifest_log(&l).dump());
                }
                panic!("shape_e inc {inc}: REOPEN FAILED (the #493 symptom): {e}\n{dumps}");
            });
            drop(tree);
            assert_invariant(
                dir.path(),
                &format!("shape_e inc {inc} post-recovery"),
                false,
            );
        }
    }
    let n = total_records(dir.path());
    assert!(n >= 50, "vacuous run: only {n} manifest records survived");
    eprintln!("shape_e: {n} manifest records checked after {INCARNATIONS} SIGKILLs");
}

// ---------- shape (f): WAL-heavy recovery kill storm ----------
//
// Child mode "burst": checkpointer DISABLED, writes a large burst so the
// WAL carries everything, then spins. Kill it; then run many incarnations
// of the normal child (checkpointer enabled) killed FAST, so most kills
// land inside open()/WAL-replay/first-checkpoint — the field's
// "SIGKILLed mid-recovery" scenario.

#[test]
fn child_burst_entry() {
    let Ok(dir) = std::env::var("HOLT_INV_BURST_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let cfg = nokv_like_config(&dir, false); // no checkpointer: WAL-only
    let tree = Tree::open(cfg).unwrap();
    for i in 0..2500u64 {
        let k = if i % 10 < 3 {
            key(i % 8)
        } else {
            key(100 + i % 1500)
        };
        if i % 13 == 5 {
            let _ = tree.delete(&k).unwrap();
        } else {
            tree.put(&k, &value(i, 512 + (i % 5) as usize * 512))
                .unwrap();
        }
    }
    // Signal readiness, then spin so the parent's kill is a true SIGKILL
    // on a live process with a fat WAL and no checkpoint.
    std::fs::write(dir.join("burst-ready"), b"1").unwrap();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[test]
fn shape_f_recovery_kill_storm() {
    let dir = tempfile::tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();

    // Phase 1: burst writer, killed with everything in the WAL.
    let mut child = std::process::Command::new(&exe)
        .arg("--exact")
        .arg("child_burst_entry")
        .arg("--nocapture")
        .env("HOLT_INV_BURST_DIR", dir.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let ready = dir.path().join("burst-ready");
    for _ in 0..600 {
        if ready.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(ready.exists(), "burst child never became ready");
    child.kill().unwrap();
    let _ = child.wait();
    assert_invariant(dir.path(), "shape_f post-burst-kill", true);

    // Phase 2: recovery kill storm. Each incarnation must replay the WAL
    // tail; fast kills land mid-recovery or mid-first-checkpoint.
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for inc in 0..30u32 {
        let mut child = std::process::Command::new(&exe)
            .arg("--exact")
            .arg("child_writer_entry")
            .arg("--nocapture")
            .env("HOLT_INV_CHILD_DIR", dir.path())
            .env("HOLT_INV_CHILD_SEED", format!("{}", next() % 1000))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        // Mix ultra-fast kills (mid WAL replay) with kills that give the
        // 15ms-cadence checkpointer a few rounds on the replayed state.
        let delay = if inc % 2 == 0 {
            3 + next() % 40
        } else {
            40 + next() % 260
        };
        std::thread::sleep(std::time::Duration::from_millis(delay));
        child.kill().unwrap();
        let _ = child.wait();
        assert_invariant(dir.path(), &format!("shape_f storm inc {inc}"), true);
    }

    // Final graceful recovery must succeed (an Err here = the #493 symptom).
    let cfg = nokv_like_config(dir.path(), true);
    let tree = Tree::open(cfg).unwrap_or_else(|e| {
        let mut dumps = String::new();
        for l in find_manifest_logs(dir.path()) {
            dumps.push_str(&check_manifest_log(&l).dump());
        }
        panic!("shape_f: FINAL REOPEN FAILED (the #493 symptom): {e}\n{dumps}");
    });
    tree.checkpoint().unwrap();
    drop(tree);
    assert_invariant(dir.path(), "shape_f final", false);
    let n = total_records(dir.path());
    // A multi-frame tree flushes ~1 Set per 512KB frame; storm incarnations
    // killed mid-WAL-replay add none. >=5 proves the tree is multi-frame.
    assert!(n >= 5, "vacuous run: only {n} manifest records written");
    eprintln!("shape_f: {n} manifest records checked after burst + 30 storm kills");
}

// ---------- negative control: flock must exclude a second writer ----------

#[test]
fn dual_writer_excluded_by_flock() {
    let dir = tempfile::tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();
    let mut child = std::process::Command::new(&exe)
        .arg("--exact")
        .arg("child_writer_entry")
        .arg("--nocapture")
        .env("HOLT_INV_CHILD_DIR", dir.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(400));
    // A second writable open in this process must NOT succeed while the
    // child holds the flock (5s acquire timeout in holt).
    let cfg = nokv_like_config(dir.path(), false);
    let second = Tree::open(cfg);
    assert!(
        second.is_err(),
        "SECOND WRITER OPENED CONCURRENTLY — flock exclusion failed; \
         dual-writer manifest interleave is possible"
    );
    child.kill().unwrap();
    let _ = child.wait();
}

// ---------- builder smoke: make sure the config path we exercise matches ----------

#[test]
fn builder_smoke_matches_nokv_shape() {
    let dir = tempfile::tempdir().unwrap();
    let tree = TreeBuilder::new(dir.path())
        .durability(Durability::Wal { sync: true })
        .checkpoint(CheckpointConfig {
            enabled: false,
            auto_vacuum: false,
            ..CheckpointConfig::default()
        })
        .open()
        .unwrap();
    tree.put(b"a/b", b"v").unwrap();
    tree.checkpoint().unwrap();
    assert_invariant(dir.path(), "builder smoke", false);
}
