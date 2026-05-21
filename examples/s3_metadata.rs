//! `s3_metadata` — holt as the metadata layer behind a small
//! S3-compatible object store.
//!
//! The object bytes live elsewhere. Holt stores only namespace
//! metadata:
//!
//! - `o/<bucket>/<object>`: object metadata returned by HeadObject.
//! - `u/<bucket>/<upload_id>/meta`: multipart-upload state.
//! - `u/<bucket>/<upload_id>/parts/<part_no>`: uploaded part metadata.
//!
//! The values are opaque bytes to Holt. This example uses a compact
//! text encoding so the mapping stays readable; a real service would
//! usually use protobuf, bincode, or a fixed binary layout.

use holt::{RangeEntry, RecordVersion, Result, Tree, TreeBuilder};

#[derive(Debug, Clone, Copy)]
struct ObjectMeta<'a> {
    size: u64,
    etag: &'a str,
    storage_class: &'a str,
    content_type: &'a str,
    modified_unix: u64,
}

fn main() -> Result<()> {
    println!("=== holt s3_metadata example ===\n");

    let tree = TreeBuilder::new("scratch").memory().open()?;

    put_object(
        &tree,
        "photos",
        "users/alice/01.jpg",
        ObjectMeta {
            size: 2_345_678,
            etag: "d41d8cd98f00b204e9800998ecf8427e",
            storage_class: "STANDARD",
            content_type: "image/jpeg",
            modified_unix: 1_779_360_000,
        },
    )?;
    put_object_if_none_match(
        &tree,
        "photos",
        "users/alice/02.jpg",
        ObjectMeta {
            size: 1_234_567,
            etag: "098f6bcd4621d373cade4e832627b4f6",
            storage_class: "STANDARD",
            content_type: "image/jpeg",
            modified_unix: 1_779_360_060,
        },
    )?;
    put_object(
        &tree,
        "photos",
        "users/bob/profile.png",
        ObjectMeta {
            size: 456_789,
            etag: "7f8c0a3b4d5e6f001122334455667788",
            storage_class: "STANDARD",
            content_type: "image/png",
            modified_unix: 1_779_360_120,
        },
    )?;

    println!("HeadObject + version token");
    let alice01 = object_key("photos", "users/alice/01.jpg");
    let head = tree.get_record(&alice01)?.expect("object exists");
    println!(
        "  users/alice/01.jpg version={} meta={}",
        head.version.as_u64(),
        std::str::from_utf8(&head.value).expect("example metadata is utf8")
    );

    println!("\nPutObject with If-None-Match: *");
    let created = put_object_if_none_match(
        &tree,
        "photos",
        "users/alice/01.jpg",
        ObjectMeta {
            size: 999,
            etag: "should-not-win",
            storage_class: "STANDARD",
            content_type: "image/jpeg",
            modified_unix: 1_779_360_180,
        },
    )?;
    println!("  duplicate create committed? {created}");

    println!("\nCopyObject with source If-Match and destination no-overwrite");
    let copied = copy_object_if_source_version(
        &tree,
        "photos",
        "users/alice/01.jpg",
        "archive/2026/users/alice/01.jpg",
        head.version,
    )?;
    println!("  copy committed? {copied}");

    println!("\nListObjectsV2 prefix=users/ delimiter=/");
    for entry in list_objects_v2(&tree, "photos", "users/", Some(b'/'), None)? {
        println!("  {entry}");
    }

    println!("\nMultipart upload metadata");
    let upload_id = "upload-000001";
    create_multipart_upload(&tree, "photos", upload_id)?;
    put_upload_part(&tree, "photos", upload_id, 1, 4_194_304, "part-etag-1")?;
    put_upload_part(&tree, "photos", upload_id, 2, 2_097_152, "part-etag-2")?;
    let complete = complete_multipart_upload(
        &tree,
        "photos",
        upload_id,
        "users/alice/video.mov",
        "complete-etag",
    )?;
    println!("  complete committed? {complete}");
    println!(
        "  new object exists? {}",
        tree.get(&object_key("photos", "users/alice/video.mov"))?
            .is_some()
    );

    tree.checkpoint()?;
    Ok(())
}

fn put_object(tree: &Tree, bucket: &str, object: &str, meta: ObjectMeta<'_>) -> Result<()> {
    tree.put(&object_key(bucket, object), &encode_object_meta(meta))?;
    Ok(())
}

fn put_object_if_none_match(
    tree: &Tree,
    bucket: &str,
    object: &str,
    meta: ObjectMeta<'_>,
) -> Result<bool> {
    tree.put_if_absent(&object_key(bucket, object), &encode_object_meta(meta))
}

fn copy_object_if_source_version(
    tree: &Tree,
    bucket: &str,
    src_object: &str,
    dst_object: &str,
    expected_src: RecordVersion,
) -> Result<bool> {
    let src_key = object_key(bucket, src_object);
    let dst_key = object_key(bucket, dst_object);
    let src = tree.get_record(&src_key)?.expect("source exists");

    tree.atomic(|batch| {
        batch.assert_version(&src_key, expected_src);
        batch.put_if_absent(&dst_key, &src.value);
    })
}

fn create_multipart_upload(tree: &Tree, bucket: &str, upload_id: &str) -> Result<bool> {
    tree.put_if_absent(
        &upload_meta_key(bucket, upload_id),
        format!("state=open;bucket={bucket};upload_id={upload_id}").as_bytes(),
    )
}

fn put_upload_part(
    tree: &Tree,
    bucket: &str,
    upload_id: &str,
    part_no: u32,
    size: u64,
    etag: &str,
) -> Result<()> {
    let key = upload_part_key(bucket, upload_id, part_no);
    let value = format!("part={part_no};size={size};etag={etag}");
    tree.put(&key, value.as_bytes())?;
    Ok(())
}

fn complete_multipart_upload(
    tree: &Tree,
    bucket: &str,
    upload_id: &str,
    object: &str,
    final_etag: &str,
) -> Result<bool> {
    let upload_key = upload_meta_key(bucket, upload_id);
    let upload = tree.get_record(&upload_key)?.expect("upload exists");
    let part_prefix = upload_parts_prefix(bucket, upload_id);
    let mut parts: Vec<(Vec<u8>, RecordVersion, u64)> = Vec::new();

    for entry in tree.range().prefix(&part_prefix) {
        if let RangeEntry::Key {
            key,
            value,
            version,
        } = entry?
        {
            parts.push((key, version, parse_size(&value)));
        }
    }

    let total_size = parts.iter().map(|(_, _, size)| *size).sum();
    let final_meta = encode_object_meta(ObjectMeta {
        size: total_size,
        etag: final_etag,
        storage_class: "STANDARD",
        content_type: "video/quicktime",
        modified_unix: 1_779_360_240,
    });
    let final_key = object_key(bucket, object);

    tree.atomic(|batch| {
        batch.assert_version(&upload_key, upload.version);
        for (part_key, part_version, _) in &parts {
            batch.assert_version(part_key, *part_version);
        }
        batch.put(&final_key, &final_meta);
        for (part_key, part_version, _) in &parts {
            batch.delete_if_version(part_key, *part_version);
        }
        batch.delete_if_version(&upload_key, upload.version);
    })
}

fn list_objects_v2(
    tree: &Tree,
    bucket: &str,
    prefix: &str,
    delimiter: Option<u8>,
    start_after: Option<&str>,
) -> Result<Vec<String>> {
    let object_prefix = object_key(bucket, prefix);
    let mut range = tree.range().prefix(&object_prefix);
    if let Some(delimiter) = delimiter {
        range = range.delimiter(delimiter);
    }
    if let Some(start_after) = start_after {
        range = range.start_after(&object_key(bucket, start_after));
    }

    let mut out = Vec::new();
    for entry in range {
        match entry? {
            RangeEntry::Key { key, value, .. } => {
                out.push(format!(
                    "object {} bytes={}",
                    strip_object_key(bucket, &key),
                    parse_size(&value)
                ));
            }
            RangeEntry::CommonPrefix(prefix) => {
                out.push(format!(
                    "common-prefix {}",
                    strip_object_key(bucket, &prefix)
                ));
            }
            _ => {}
        }
    }
    Ok(out)
}

fn encode_object_meta(meta: ObjectMeta<'_>) -> Vec<u8> {
    format!(
        "size={};etag={};class={};content_type={};mtime={}",
        meta.size, meta.etag, meta.storage_class, meta.content_type, meta.modified_unix
    )
    .into_bytes()
}

fn parse_size(value: &[u8]) -> u64 {
    let text = std::str::from_utf8(value).expect("example metadata is utf8");
    text.split(';')
        .find_map(|field| field.strip_prefix("size="))
        .expect("metadata has size")
        .parse()
        .expect("size is numeric")
}

fn object_key(bucket: &str, object: &str) -> Vec<u8> {
    format!("o/{bucket}/{object}").into_bytes()
}

fn upload_meta_key(bucket: &str, upload_id: &str) -> Vec<u8> {
    format!("u/{bucket}/{upload_id}/meta").into_bytes()
}

fn upload_parts_prefix(bucket: &str, upload_id: &str) -> Vec<u8> {
    format!("u/{bucket}/{upload_id}/parts/").into_bytes()
}

fn upload_part_key(bucket: &str, upload_id: &str, part_no: u32) -> Vec<u8> {
    format!("u/{bucket}/{upload_id}/parts/{part_no:05}").into_bytes()
}

fn strip_object_key<'a>(bucket: &str, key: &'a [u8]) -> &'a str {
    let prefix = format!("o/{bucket}/");
    let object = key
        .strip_prefix(prefix.as_bytes())
        .expect("key belongs to bucket");
    std::str::from_utf8(object).expect("example keys are utf8")
}
