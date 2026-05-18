//! `s3_metadata` — artisan as an S3-compatible object metadata
//! backend.
//!
//! Keys are `bucket/object/path`; values are the small JSON-ish
//! manifest you'd hand back from `HeadObject` (size, etag,
//! storage class). The criterion `objstore` bench is shaped like
//! this — the access pattern is point lookup + occasional
//! atomic move (rename for "copy + delete").

use artisan::TreeBuilder;

fn main() {
    println!("=== artisan s3_metadata example ===\n");

    let tree = TreeBuilder::new("scratch").memory().open().expect("open");

    // Three buckets; small clusters of objects per bucket.
    let rows: &[(&[u8], &[u8])] = &[
        (
            b"photos/users/alice/01.jpg",
            br#"{"size":2345678,"etag":"\"d41d8cd98f00\"","class":"STANDARD"}"#,
        ),
        (
            b"photos/users/alice/02.jpg",
            br#"{"size":1234567,"etag":"\"098f6bcd4621\"","class":"STANDARD"}"#,
        ),
        (
            b"photos/users/bob/profile.png",
            br#"{"size":456789,"etag":"\"7f8c0a3b4d5e\"","class":"STANDARD"}"#,
        ),
        (
            b"backups/db/2026-05/snapshot.tar.zst",
            br#"{"size":1073741824,"etag":"\"deadbeefcafe\"","class":"GLACIER"}"#,
        ),
        (
            b"logs/2026-05-19/api.log.gz",
            br#"{"size":98765,"etag":"\"fedcba987654\"","class":"STANDARD_IA"}"#,
        ),
    ];
    for (key, value) in rows {
        tree.put(key, value).unwrap();
    }
    println!("indexed {} objects across 3 buckets\n", rows.len());

    // `HeadObject`.
    let key: &[u8] = b"photos/users/alice/01.jpg";
    let meta = tree.get(key).unwrap().expect("present");
    println!(
        "HeadObject {:?} -> {}",
        std::str::from_utf8(key).unwrap(),
        std::str::from_utf8(&meta).unwrap(),
    );

    // `MoveObject` — S3 doesn't expose this as a single API
    // call, but the metadata layer can do it atomically with
    // `tree.rename` (the object bytes live in an external blob
    // store; only the metadata moves).
    tree.rename(
        b"photos/users/alice/01.jpg",
        b"photos/archive/users/alice/01.jpg",
        false,
    )
    .unwrap();
    assert!(tree.get(b"photos/users/alice/01.jpg").unwrap().is_none());
    println!("moved alice/01.jpg into archive/");

    // `DeleteObject`.
    let prev = tree
        .delete(b"logs/2026-05-19/api.log.gz")
        .unwrap()
        .map(|v| v.len())
        .unwrap_or(0);
    println!("deleted log.gz -> previous = {prev} bytes");

    // Persist the changes (no-op on memory).
    tree.checkpoint().unwrap();

    println!("\ndone");
}
