//! Regression: a torn WAL tail (crash mid-write) must be physically
//! truncated on reopen. Otherwise the next session's `O_APPEND` writes a good
//! record *after* the torn bytes, turning them into a MID-log torn record; a
//! later replay's torn-tail `break` stops there and silently strands every
//! acked record written after it.

use holt::{Durability, Tree, TreeConfig};

fn cfg(dir: &std::path::Path) -> TreeConfig {
    let mut c = TreeConfig::new(dir);
    c.durability = Durability::Wal { sync: true };
    // Disable the checkpointer so the WAL is never truncated out from under
    // us — we want the torn tail to survive into the next session.
    c.checkpoint.enabled = false;
    c
}

#[test]
fn torn_tail_reopen_does_not_strand_later_appends() {
    let dir = tempfile::tempdir().unwrap();

    // Session 1: two complete, durable records.
    {
        let t = Tree::open(cfg(dir.path())).unwrap();
        t.put(b"a", b"1").unwrap();
        t.put(b"b", b"2").unwrap();
    }

    // Simulate a crash mid-write: lop the CRC off the last record so it
    // decodes as a torn tail ("record body truncated"), leaving "a" intact.
    let wal = cfg(dir.path()).wal_path().expect("file-backed WAL");
    let len = std::fs::metadata(&wal).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&wal)
        .unwrap()
        .set_len(len - 4)
        .unwrap();

    // Session 2: reopen (the fix truncates the torn tail), then append.
    {
        let t = Tree::open(cfg(dir.path())).unwrap();
        assert_eq!(t.get(b"a").unwrap().as_deref(), Some(&b"1"[..]));
        t.put(b"c", b"3").unwrap();
    }

    // Session 3: reopen again. With the torn tail dropped in session 2, "c"
    // sits cleanly after "a"; without it, "c" was appended behind a mid-log
    // torn record and is lost (or the log fails to replay).
    {
        let t = Tree::open(cfg(dir.path())).unwrap();
        assert_eq!(t.get(b"a").unwrap().as_deref(), Some(&b"1"[..]));
        assert_eq!(
            t.get(b"c").unwrap().as_deref(),
            Some(&b"3"[..]),
            "append after a torn-tail reopen was stranded behind a mid-log torn record",
        );
    }
}
