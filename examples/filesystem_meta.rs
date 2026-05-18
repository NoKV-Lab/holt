//! `filesystem_meta` — artisan as the inode table of a toy POSIX
//! filesystem. Keys are absolute paths; values are 32-byte packed
//! inodes (size + mtime + mode + uid + gid + nlink).
//!
//! Demonstrates the workload artisan was built for: short
//! hierarchical keys, fixed-size values, dense per-prefix
//! density, point-lookup + atomic rename.

use artisan::TreeBuilder;

/// 32 bytes — the shape the criterion `fs` bench also models.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct Inode {
    size: u64,
    mtime: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    nlink: u32,
}

impl Inode {
    fn to_bytes(self) -> [u8; 32] {
        // SAFETY: `#[repr(C, packed)]` over plain integer fields
        // gives a stable 32-byte layout matching the bench's
        // `fs-metadata` scenario.
        unsafe { std::mem::transmute(self) }
    }

    fn from_bytes(b: &[u8]) -> Self {
        assert_eq!(b.len(), 32, "inode is 32 bytes");
        let mut buf = [0u8; 32];
        buf.copy_from_slice(b);
        // SAFETY: same shape as `to_bytes`.
        unsafe { std::mem::transmute(buf) }
    }
}

fn main() {
    println!("=== artisan filesystem_meta example ===\n");

    // In-memory for the example so it doesn't litter cwd.
    let tree = TreeBuilder::new("scratch").memory().open().expect("open");

    // Populate a small directory tree.
    let now: u64 = 1_715_817_600;
    let entries: &[(&[u8], Inode)] = &[
        (
            b"/home/alice/notes.txt",
            Inode {
                size: 1024,
                mtime: now,
                mode: 0o644,
                uid: 1000,
                gid: 1000,
                nlink: 1,
            },
        ),
        (
            b"/home/alice/photos/sunset.jpg",
            Inode {
                size: 2_345_678,
                mtime: now - 3600,
                mode: 0o644,
                uid: 1000,
                gid: 1000,
                nlink: 1,
            },
        ),
        (
            b"/home/bob/.bashrc",
            Inode {
                size: 412,
                mtime: now - 86_400,
                mode: 0o600,
                uid: 1001,
                gid: 1001,
                nlink: 1,
            },
        ),
        (
            b"/etc/passwd",
            Inode {
                size: 2_048,
                mtime: now - 7 * 86_400,
                mode: 0o644,
                uid: 0,
                gid: 0,
                nlink: 1,
            },
        ),
    ];
    for (path, inode) in entries {
        tree.put(path, &inode.to_bytes()).unwrap();
    }
    println!("loaded {} inodes\n", entries.len());

    // Stat-style lookup.
    let path: &[u8] = b"/home/alice/photos/sunset.jpg";
    let bytes = tree.get(path).unwrap().expect("present");
    let i = Inode::from_bytes(&bytes);
    let size = i.size;
    let mode = i.mode;
    let nlink = i.nlink;
    println!(
        "stat {:?} -> size={size} mode={mode:o} nlink={nlink}",
        std::str::from_utf8(path).unwrap(),
    );

    // Atomic rename — `force=false` errors out if the destination
    // already exists (POSIX `rename(2)` allows overwrite, so a
    // real syscall layer would pass `true`).
    tree.rename(
        b"/home/alice/photos/sunset.jpg",
        b"/home/alice/photos/sunset-archived.jpg",
        false,
    )
    .unwrap();
    assert!(tree
        .get(b"/home/alice/photos/sunset.jpg")
        .unwrap()
        .is_none());
    assert!(tree
        .get(b"/home/alice/photos/sunset-archived.jpg")
        .unwrap()
        .is_some());
    println!("renamed: original path gone, new path present");

    // unlink.
    let prev = tree.delete(b"/home/bob/.bashrc").unwrap();
    println!(
        "unlink /home/bob/.bashrc -> previous {} bytes",
        prev.map(|v| v.len()).unwrap_or(0),
    );

    // Force durability on a persistent tree (no-op on memory).
    tree.checkpoint().unwrap();

    println!("\ndone");
}
