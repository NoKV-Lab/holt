//! `basic_kv` — smallest possible end-to-end demo of the public API.

use artisan::{TreeBuilder, TreeConfig};

fn main() {
    println!("=== artisan basic_kv example ===\n");

    // Default is persistent. Add `.memory()` to flip to in-memory.
    // (We use memory here so the example doesn't litter the cwd.)
    let tree = TreeBuilder::new("scratch")
        .memory()
        .buffer_pool_size(64)
        .open()
        .expect("open");

    println!("Tree opened: {tree:?}\n");

    // Round-trip a few keys. Values are just bytes — the engine
    // doesn't interpret them.
    tree.put(b"img/01.jpg", b"\x89PNG...").unwrap();
    tree.put(b"img/02.jpg", b"\xFF\xD8\xFF...").unwrap();
    tree.put(b"meta/owner", b"alice").unwrap();

    for key in [
        b"img/01.jpg".as_ref(),
        b"img/02.jpg",
        b"meta/owner",
        b"missing",
    ] {
        match tree.get(key).unwrap() {
            Some(v) => println!(
                "get {:?} -> {} bytes",
                String::from_utf8_lossy(key),
                v.len()
            ),
            None => println!("get {:?} -> (none)", String::from_utf8_lossy(key)),
        }
    }

    let prev = tree.delete(b"meta/owner").unwrap();
    println!("\ndelete meta/owner returned previous = {prev:?}");
    println!("get meta/owner -> {:?}", tree.get(b"meta/owner").unwrap());

    // Direct config form (without the builder):
    let _persistent_cfg = TreeConfig::new("/var/lib/myapp"); // persistent (default)
    let _memory_cfg = TreeConfig::memory();
}
