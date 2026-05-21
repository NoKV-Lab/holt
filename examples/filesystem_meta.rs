//! `filesystem_meta` — holt as a filesystem metadata index.
//!
//! This example uses the common POSIX split between dirents and
//! inodes:
//!
//! - `d/<absolute-path>` maps a path to a compact dirent value
//!   `{inode_id, kind}`.
//! - `i/<inode_id>` maps an inode id to opaque inode metadata bytes
//!   `{size, mtime, mode, uid, gid, nlink}`.
//!
//! Holt does not understand inode layouts. The binary encoding below
//! is just an application convention layered above the core engine.

use std::collections::BTreeSet;

use holt::{RangeEntry, Result, Tree, TreeBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    File,
    Directory,
}

#[derive(Debug, Clone, Copy)]
struct Dirent {
    inode: u64,
    kind: EntryKind,
}

#[derive(Debug, Clone, Copy)]
struct Inode {
    size: u64,
    mtime: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    nlink: u32,
}

fn main() -> Result<()> {
    println!("=== holt filesystem_meta example ===\n");

    let tree = TreeBuilder::new("scratch").memory().open()?;
    bootstrap_namespace(&tree)?;

    println!("create(O_CREAT|O_EXCL) /home/alice/notes.txt");
    let created = create_file_exclusive(
        &tree,
        "/home/alice/notes.txt",
        10,
        Inode {
            size: 1024,
            mtime: 1_779_360_000,
            mode: 0o100644,
            uid: 1000,
            gid: 1000,
            nlink: 1,
        },
    )?;
    println!("  committed? {created}");

    println!("\nstat /home/alice/notes.txt");
    let (dirent, inode) = stat_path(&tree, "/home/alice/notes.txt")?.expect("file exists");
    println!(
        "  inode={} size={} mode={:o} nlink={}",
        dirent.inode, inode.size, inode.mode, inode.nlink
    );

    println!("\nlink /home/alice/notes.txt -> /home/bob/notes-hardlink.txt");
    let linked = hardlink(
        &tree,
        "/home/alice/notes.txt",
        "/home/bob/notes-hardlink.txt",
    )?;
    println!("  committed? {linked}");
    let (_, inode_after_link) = stat_path(&tree, "/home/alice/notes.txt")?.expect("file exists");
    println!("  source nlink after link={}", inode_after_link.nlink);

    println!("\nrename dirent /home/alice/notes.txt -> /home/alice/todo.txt");
    rename_no_replace(&tree, "/home/alice/notes.txt", "/home/alice/todo.txt")?;
    println!(
        "  old exists? {} new exists? {}",
        stat_path(&tree, "/home/alice/notes.txt")?.is_some(),
        stat_path(&tree, "/home/alice/todo.txt")?.is_some()
    );

    println!("\nreaddir /home/alice");
    for child in read_dir(&tree, "/home/alice")? {
        println!("  {child}");
    }

    println!("\nrmdir /home/alice/projects while non-empty");
    mkdir(&tree, "/home/alice/projects", 20, 1000)?;
    create_file_exclusive(
        &tree,
        "/home/alice/projects/readme.md",
        21,
        Inode {
            size: 42,
            mtime: 1_779_360_030,
            mode: 0o100644,
            uid: 1000,
            gid: 1000,
            nlink: 1,
        },
    )?;
    let removed = rmdir(&tree, "/home/alice/projects")?;
    println!("  removed? {removed}");

    println!("\nunlink child, then rmdir again");
    unlink(&tree, "/home/alice/projects/readme.md")?;
    let removed = rmdir(&tree, "/home/alice/projects")?;
    println!("  removed? {removed}");

    tree.checkpoint()?;
    Ok(())
}

fn bootstrap_namespace(tree: &Tree) -> Result<()> {
    mkdir(tree, "/", 1, 0)?;
    mkdir(tree, "/home", 2, 0)?;
    mkdir(tree, "/home/alice", 3, 1000)?;
    mkdir(tree, "/home/bob", 4, 1001)?;
    Ok(())
}

fn mkdir(tree: &Tree, path: &str, inode_id: u64, uid: u32) -> Result<bool> {
    let inode = Inode {
        size: 0,
        mtime: 1_779_360_000,
        mode: 0o040755,
        uid,
        gid: uid,
        nlink: 2,
    };
    let dirent = Dirent {
        inode: inode_id,
        kind: EntryKind::Directory,
    };
    let dirent_key = dirent_key(path);
    let inode_key = inode_key(inode_id);
    tree.atomic(|batch| {
        batch.put_if_absent(&dirent_key, &dirent.to_bytes());
        batch.put(&inode_key, &inode.to_bytes());
    })
}

fn create_file_exclusive(tree: &Tree, path: &str, inode_id: u64, inode: Inode) -> Result<bool> {
    let dirent = Dirent {
        inode: inode_id,
        kind: EntryKind::File,
    };
    let dirent_key = dirent_key(path);
    let inode_key = inode_key(inode_id);
    tree.atomic(|batch| {
        batch.put_if_absent(&dirent_key, &dirent.to_bytes());
        batch.put(&inode_key, &inode.to_bytes());
    })
}

fn stat_path(tree: &Tree, path: &str) -> Result<Option<(Dirent, Inode)>> {
    let Some(dirent_record) = tree.get_record(&dirent_key(path))? else {
        return Ok(None);
    };
    let dirent = Dirent::from_bytes(&dirent_record.value);
    let Some(inode) = tree.get(&inode_key(dirent.inode))? else {
        return Ok(None);
    };
    Ok(Some((dirent, Inode::from_bytes(&inode))))
}

fn hardlink(tree: &Tree, src: &str, dst: &str) -> Result<bool> {
    let src_key = dirent_key(src);
    let Some(src_record) = tree.get_record(&src_key)? else {
        return Ok(false);
    };
    let dirent = Dirent::from_bytes(&src_record.value);
    if dirent.kind != EntryKind::File {
        return Ok(false);
    }

    let inode_key = inode_key(dirent.inode);
    let Some(inode_record) = tree.get_record(&inode_key)? else {
        return Ok(false);
    };
    let mut inode = Inode::from_bytes(&inode_record.value);
    inode.nlink += 1;
    let dst_key = dirent_key(dst);

    tree.atomic(|batch| {
        batch.assert_version(&src_key, src_record.version);
        batch.put_if_absent(&dst_key, &dirent.to_bytes());
        batch.compare_and_put(&inode_key, inode_record.version, &inode.to_bytes());
    })
}

fn rename_no_replace(tree: &Tree, src: &str, dst: &str) -> Result<()> {
    tree.rename(&dirent_key(src), &dirent_key(dst), false)?;
    Ok(())
}

fn unlink(tree: &Tree, path: &str) -> Result<bool> {
    let key = dirent_key(path);
    let Some(dirent_record) = tree.get_record(&key)? else {
        return Ok(false);
    };
    let dirent = Dirent::from_bytes(&dirent_record.value);
    let inode_key = inode_key(dirent.inode);
    let Some(inode_record) = tree.get_record(&inode_key)? else {
        return Ok(false);
    };
    let mut inode = Inode::from_bytes(&inode_record.value);

    tree.atomic(|batch| {
        batch.assert_version(&key, dirent_record.version);
        batch.delete_if_version(&key, dirent_record.version);
        if inode.nlink > 1 {
            inode.nlink -= 1;
            batch.compare_and_put(&inode_key, inode_record.version, &inode.to_bytes());
        } else {
            batch.delete_if_version(&inode_key, inode_record.version);
        }
    })
}

fn rmdir(tree: &Tree, path: &str) -> Result<bool> {
    let key = dirent_key(path);
    let Some(record) = tree.get_record(&key)? else {
        return Ok(false);
    };
    let dirent = Dirent::from_bytes(&record.value);
    if dirent.kind != EntryKind::Directory {
        return Ok(false);
    }
    let children = child_prefix(path);
    let inode_key = inode_key(dirent.inode);

    tree.atomic(|batch| {
        batch.assert_prefix_empty(&children);
        batch.delete_if_version(&key, record.version);
        batch.delete(&inode_key);
    })
}

fn read_dir(tree: &Tree, path: &str) -> Result<Vec<String>> {
    let prefix = child_prefix(path);
    let mut children = BTreeSet::new();

    for entry in tree.range().prefix(&prefix).delimiter(b'/') {
        match entry? {
            RangeEntry::Key { key, value, .. } => {
                let name = child_name(&prefix, &key);
                if !name.is_empty() {
                    let dirent = Dirent::from_bytes(&value);
                    let suffix = match dirent.kind {
                        EntryKind::File => "",
                        EntryKind::Directory => "/",
                    };
                    children.insert(format!("{name}{suffix}"));
                }
            }
            RangeEntry::CommonPrefix(common) => {
                let name = child_name(&prefix, &common);
                if !name.is_empty() {
                    children.insert(format!("{name}/"));
                }
            }
            _ => {}
        }
    }

    Ok(children.into_iter().collect())
}

fn child_name(prefix: &[u8], key: &[u8]) -> String {
    let tail = key
        .strip_prefix(prefix)
        .expect("range key starts with prefix");
    let tail = tail.strip_suffix(b"/").unwrap_or(tail);
    std::str::from_utf8(tail)
        .expect("example paths are utf8")
        .to_owned()
}

fn dirent_key(path: &str) -> Vec<u8> {
    assert!(path.starts_with('/'), "path must be absolute");
    if path == "/" {
        b"d/".to_vec()
    } else {
        format!("d{path}").into_bytes()
    }
}

fn child_prefix(path: &str) -> Vec<u8> {
    let mut key = dirent_key(path);
    if !key.ends_with(b"/") {
        key.push(b'/');
    }
    key
}

fn inode_key(inode: u64) -> Vec<u8> {
    format!("i/{inode:016x}").into_bytes()
}

impl Dirent {
    fn to_bytes(self) -> [u8; 9] {
        let mut out = [0; 9];
        out[..8].copy_from_slice(&self.inode.to_le_bytes());
        out[8] = match self.kind {
            EntryKind::File => 1,
            EntryKind::Directory => 2,
        };
        out
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 9, "dirent value is 9 bytes");
        let mut inode = [0; 8];
        inode.copy_from_slice(&bytes[..8]);
        let kind = match bytes[8] {
            1 => EntryKind::File,
            2 => EntryKind::Directory,
            other => panic!("unknown dirent kind {other}"),
        };
        Self {
            inode: u64::from_le_bytes(inode),
            kind,
        }
    }
}

impl Inode {
    fn to_bytes(self) -> [u8; 32] {
        let mut out = [0; 32];
        out[0..8].copy_from_slice(&self.size.to_le_bytes());
        out[8..16].copy_from_slice(&self.mtime.to_le_bytes());
        out[16..20].copy_from_slice(&self.mode.to_le_bytes());
        out[20..24].copy_from_slice(&self.uid.to_le_bytes());
        out[24..28].copy_from_slice(&self.gid.to_le_bytes());
        out[28..32].copy_from_slice(&self.nlink.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 32, "inode value is 32 bytes");
        Self {
            size: read_u64(&bytes[0..8]),
            mtime: read_u64(&bytes[8..16]),
            mode: read_u32(&bytes[16..20]),
            uid: read_u32(&bytes[20..24]),
            gid: read_u32(&bytes[24..28]),
            nlink: read_u32(&bytes[28..32]),
        }
    }
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut out = [0; 8];
    out.copy_from_slice(bytes);
    u64::from_le_bytes(out)
}

fn read_u32(bytes: &[u8]) -> u32 {
    let mut out = [0; 4];
    out.copy_from_slice(bytes);
    u32::from_le_bytes(out)
}
