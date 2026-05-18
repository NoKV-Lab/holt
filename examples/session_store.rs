//! `session_store` — multi-tenant session table.
//!
//! Each user has many active sessions; each session has a small
//! state blob (last-seen, IP, agent). Keys are prefixed by user
//! id so a single tenant's data sits in a contiguous slice of
//! the key space — exactly the shape ART path-compression
//! optimises for.

use artisan::TreeBuilder;

/// Compose the on-disk key for a session record.
fn session_key(user_id: u64, session_id: &str) -> Vec<u8> {
    // `u<16-hex>/` keeps the prefix fixed-width so two
    // neighbouring users' keys share a 1-byte radix-tree fork
    // (the `u` byte) and one tenant's sessions cluster under
    // their own Prefix node.
    let mut k = format!("u{user_id:016x}/").into_bytes();
    k.extend_from_slice(session_id.as_bytes());
    k
}

fn main() {
    println!("=== artisan session_store example ===\n");

    let tree = TreeBuilder::new("scratch").memory().open().expect("open");

    // Three users, two sessions each.
    let rows: &[(u64, &str, &str)] = &[
        (1, "abc123", "last_seen=1715817600 ip=10.0.0.1 ua=Chrome"),
        (1, "def456", "last_seen=1715818000 ip=10.0.0.2 ua=Safari"),
        (2, "ghi789", "last_seen=1715819000 ip=10.0.0.5 ua=Firefox"),
        (2, "jkl012", "last_seen=1715819100 ip=10.0.0.6 ua=Chrome"),
        (3, "mno345", "last_seen=1715819200 ip=10.0.0.9 ua=Edge"),
    ];
    for (user, sid, state) in rows {
        tree.put(&session_key(*user, sid), state.as_bytes())
            .unwrap();
    }
    println!("seeded {} sessions across 3 users", rows.len());

    // Authenticate-and-touch: look up the session, replace the
    // state. Same-size value → in-place leaf update; zero
    // allocator activity (the walker's `insert_into_leaf` fast
    // path detects same-size and overwrites in place).
    let touched_state = b"last_seen=1715820000 ip=10.0.0.2 ua=Safari";
    tree.put(&session_key(1, "def456"), touched_state).unwrap();
    println!(
        "touched session 1/def456 -> {:?}",
        std::str::from_utf8(touched_state).unwrap(),
    );

    // Look up a session.
    let s = tree
        .get(&session_key(2, "ghi789"))
        .unwrap()
        .expect("present");
    println!("lookup 2/ghi789 -> {:?}", std::str::from_utf8(&s).unwrap(),);

    // Revoke a session.
    let prev = tree.delete(&session_key(1, "abc123")).unwrap();
    println!(
        "revoked 1/abc123 -> previous = {:?}",
        prev.as_deref().map(String::from_utf8_lossy),
    );
    assert!(tree.get(&session_key(1, "abc123")).unwrap().is_none());

    // Verify the other tenant's data is untouched.
    assert!(tree.get(&session_key(2, "ghi789")).unwrap().is_some());

    tree.checkpoint().unwrap();

    println!("\ndone");
}
