//! End-to-end unit tests for the WAL writer + replay scanner —
//! write some records, flush, scan, verify what comes back
//! matches what went in.

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

use tempfile::tempdir;

use super::codec::{
    crc32, encode_file_header, encode_record, BatchEncoder, FileHeader, FILE_HEADER_SIZE,
    FORMAT_VERSION, MAX_ATOMIC_WAL_OPS, RECORD_FOOTER_SIZE, RECORD_HEADER_SIZE, RECORD_MAGIC,
};
use super::reader::{replay, replay_bytes, replay_file};
use super::wal_op::WalOp;
use super::writer::{WalWriter, AUTO_FLUSH_THRESHOLD};
use crate::api::errors::Error;

fn wal_path(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("test.wal")
}

fn append_raw_record(out: &mut Vec<u8>, seq: u64, ty: u8, body: &[u8]) {
    let start = out.len();
    out.extend_from_slice(&RECORD_MAGIC.to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.push(ty);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
}

fn sample_ops() -> Vec<WalOp> {
    vec![
        WalOp::Insert {
            tree_id: 0,
            key: b"img/01.jpg".to_vec(),
            value: vec![0xAA; 64],
        },
        WalOp::Insert {
            tree_id: 0,
            key: b"img/02.jpg".to_vec(),
            value: vec![0xBB; 64],
        },
        WalOp::Erase {
            tree_id: 0,
            key: b"img/01.jpg".to_vec(),
        },
        WalOp::RenameObject {
            tree_id: 0,
            src_key: b"img/02.jpg".to_vec(),
            dst_key: b"img/02-renamed.jpg".to_vec(),
            value: vec![0xBB; 64],
            force: false,
        },
    ]
}

#[test]
fn create_open_round_trip_all_variants() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let ops = sample_ops();

    let mut w = WalWriter::create(&path, 42).unwrap();
    assert_eq!(w.header().tree_id, 42);
    assert_eq!(w.bytes_written(), FILE_HEADER_SIZE as u64);

    for (i, op) in ops.iter().enumerate() {
        w.append(op, i as u64 + 1).unwrap();
    }
    w.flush().unwrap();

    let mut collected = Vec::new();
    let (header, stats) = replay(&path, |op, seq, _off| {
        collected.push((format!("{op:?}"), seq));
        Ok(())
    })
    .unwrap();

    assert_eq!(header.tree_id, 42);
    assert_eq!(stats.records_seen, ops.len() as u64);
    assert_eq!(stats.highest_seq, Some(ops.len() as u64));
    assert_eq!(stats.torn_tail_at, None);
    assert_eq!(collected.len(), ops.len());
    for (i, (decoded_dbg, seq)) in collected.iter().enumerate() {
        assert_eq!(*seq, i as u64 + 1);
        assert_eq!(decoded_dbg, &format!("{:?}", ops[i]));
    }
}

#[test]
fn replay_rejects_removed_structural_tag() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    append_raw_record(&mut bytes, 7, 2, &[]);
    fs::write(&path, &bytes).unwrap();

    match replay(&path, |_, _, _| Ok(())) {
        Err(Error::ReplaySanityFailed {
            context,
            record_offset,
        }) => {
            assert!(context.contains("variant"));
            assert_eq!(record_offset, FILE_HEADER_SIZE as u64);
        }
        other => panic!("expected removed structural tag rejection, got {other:?}"),
    }
}

#[test]
fn open_existing_resumes_append_position() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    {
        let mut w = WalWriter::create(&path, 7).unwrap();
        w.append(
            &WalOp::Insert {
                tree_id: 0,
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
            },
            1,
        )
        .unwrap();
        w.flush().unwrap();
    }

    // Reopen and append more.
    {
        let mut w = WalWriter::open_existing(&path).unwrap();
        assert_eq!(w.header().tree_id, 7);
        w.append(
            &WalOp::Erase {
                tree_id: 0,
                key: b"k1".to_vec(),
            },
            2,
        )
        .unwrap();
        w.flush().unwrap();
    }

    // Replay sees both.
    let mut seen = Vec::new();
    let (_h, stats) = replay(&path, |op, seq, _| {
        seen.push((format!("{op:?}"), seq));
        Ok(())
    })
    .unwrap();
    assert_eq!(stats.records_seen, 2);
    assert_eq!(stats.highest_seq, Some(2));
}

#[test]
fn open_or_create_uses_existing_when_present() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let _ = WalWriter::create(&path, 99).unwrap();
    let w = WalWriter::open_or_create(&path, 99).unwrap();
    assert_eq!(w.header().tree_id, 99);
}

#[test]
fn open_or_create_rejects_mismatched_tree_id() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let _ = WalWriter::create(&path, 99).unwrap();
    match WalWriter::open_or_create(&path, 100) {
        Err(Error::ReplaySanityFailed { context, .. }) => {
            assert!(context.contains("tree_id"));
        }
        other => panic!("expected tree-id mismatch error, got {other:?}"),
    }
}

#[test]
fn unflushed_records_are_lost_after_drop() {
    // The WAL semantic is: bytes you didn't `flush` are not durable.
    // Drop without flush should leave the file at exactly the
    // header bytes.
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    {
        let mut w = WalWriter::create(&path, 0).unwrap();
        w.append(
            &WalOp::Insert {
                tree_id: 0,
                key: b"transient".to_vec(),
                value: b"never-persisted".to_vec(),
            },
            1,
        )
        .unwrap();
        // Intentionally no flush().
        drop(w);
    }

    let on_disk = fs::metadata(&path).unwrap().len();
    assert_eq!(on_disk, FILE_HEADER_SIZE as u64);

    // Replay sees zero records and no torn tail (file ends on
    // header boundary).
    let (_h, stats) = replay(&path, |_, _, _| Ok(())).unwrap();
    assert_eq!(stats.records_seen, 0);
    assert_eq!(stats.torn_tail_at, None);
}

#[test]
fn torn_tail_is_recovered_gracefully() {
    // Simulate a power loss in the middle of a `flush`: write
    // several records, then chop the last few bytes off the file
    // — the scanner should yield every complete record before the
    // chop and stop at the partial tail.
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let ops = sample_ops();
    {
        let mut w = WalWriter::create(&path, 0).unwrap();
        for (i, op) in ops.iter().enumerate() {
            w.append(op, i as u64 + 1).unwrap();
        }
        w.flush().unwrap();
    }

    // Chop off the last 8 bytes — guaranteed to fall inside the
    // CRC/body of the last record for any of the variants in
    // `sample_ops`.
    {
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        let len = file.metadata().unwrap().len();
        file.set_len(len - 8).unwrap();
    }

    let mut seen = Vec::new();
    let (_h, stats) = replay(&path, |_, seq, _| {
        seen.push(seq);
        Ok(())
    })
    .unwrap();

    assert!(stats.torn_tail_at.is_some());
    // All but the last record should have been replayed.
    assert_eq!(seen.len(), ops.len() - 1);
    assert_eq!(stats.records_seen, ops.len() as u64 - 1);
    assert_eq!(stats.highest_seq, Some(ops.len() as u64 - 1));
}

#[test]
fn mid_file_corruption_propagates_with_offset() {
    // Flip a bit in the middle of an early record. The scanner
    // should error out — this isn't a torn tail, it's real data
    // corruption.
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let ops = sample_ops();
    {
        let mut w = WalWriter::create(&path, 0).unwrap();
        for (i, op) in ops.iter().enumerate() {
            w.append(op, i as u64 + 1).unwrap();
        }
        w.flush().unwrap();
    }

    // Flip a CRC byte inside the SECOND record (so the first
    // replays cleanly and we exercise the offset-patching path).
    let mut bytes = fs::read(&path).unwrap();
    // First record body length is encoded in bytes [FILE_HEADER+4 .. +8].
    let len_pos = FILE_HEADER_SIZE + 4;
    let first_body_len =
        u32::from_le_bytes(bytes[len_pos..len_pos + 4].try_into().unwrap()) as usize;
    let first_record_end =
        FILE_HEADER_SIZE + RECORD_HEADER_SIZE + first_body_len + RECORD_FOOTER_SIZE;
    // Flip a bit deep inside the second record's body.
    bytes[first_record_end + 20] ^= 0xFF;
    fs::write(&path, &bytes).unwrap();

    match replay(&path, |_, _, _| Ok(())) {
        Err(Error::ReplaySanityFailed {
            context,
            record_offset,
        }) => {
            assert!(record_offset > 0, "offset should be patched in");
            assert!(record_offset >= first_record_end as u64);
            // CRC was the most likely catch — but any "byte
            // present but invalid" outcome is acceptable.
            assert!(
                context.contains("CRC") || context.contains("magic") || context.contains("variant"),
                "unexpected sanity context: {context}",
            );
        }
        other => panic!("expected mid-file sanity failure, got {other:?}"),
    }
}

#[test]
fn whole_wal_validation_rejects_late_corruption_before_any_callback() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    encode_record(
        &WalOp::Insert {
            tree_id: 0,
            key: b"good-before".to_vec(),
            value: b"value".to_vec(),
        },
        1,
        &mut bytes,
    );
    // A validly framed/CRC'd Insert with no body fails semantic validation.
    append_raw_record(&mut bytes, 2, 0, &[]);
    encode_record(
        &WalOp::Insert {
            tree_id: 0,
            key: b"good-after".to_vec(),
            value: b"value".to_vec(),
        },
        3,
        &mut bytes,
    );
    fs::write(&path, &bytes).unwrap();

    let mut callbacks = 0usize;
    assert!(matches!(
        replay(&path, |_, _, _| {
            callbacks += 1;
            Ok(())
        }),
        Err(Error::ReplaySanityFailed { .. })
    ));
    assert_eq!(
        callbacks, 0,
        "validation must precede every replay callback"
    );
    assert_eq!(
        fs::read(&path).unwrap(),
        bytes,
        "corruption must not truncate"
    );
}

#[test]
fn corrupt_declared_length_cannot_hide_a_valid_acknowledged_suffix() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"first".to_vec(),
        },
        1,
        &mut bytes,
    );
    let corrupt_offset = bytes.len();
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"middle".to_vec(),
        },
        2,
        &mut bytes,
    );
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"acknowledged-suffix".to_vec(),
        },
        3,
        &mut bytes,
    );
    // Stay under the native record ceiling but extend beyond EOF. The v4
    // footer and the independently valid suffix prove this is corruption,
    // not a safely truncatable torn tail.
    bytes[corrupt_offset + 4..corrupt_offset + 8].copy_from_slice(&1_000_000u32.to_le_bytes());
    fs::write(&path, &bytes).unwrap();

    let mut callbacks = 0usize;
    match replay(&path, |_, _, _| {
        callbacks += 1;
        Ok(())
    }) {
        Err(Error::ReplaySanityFailed {
            context,
            record_offset,
        }) => {
            assert!(context.contains("truncated record"));
            assert_eq!(record_offset, corrupt_offset as u64);
        }
        other => panic!("expected mid-log length corruption, got {other:?}"),
    }
    assert_eq!(callbacks, 0);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[test]
fn corrupt_length_cannot_disguise_a_complete_semantically_invalid_frame_as_torn() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    let record_offset = bytes.len();
    encode_record(
        &WalOp::RenameObject {
            tree_id: 0,
            src_key: b"src".to_vec(),
            dst_key: b"dst".to_vec(),
            value: b"bound-value".to_vec(),
            force: false,
        },
        1,
        &mut bytes,
    );

    // First create a complete CRC/LEN2-valid frame whose force byte is outside
    // the canonical 0/1 domain. Then corrupt only LEN in the leading header.
    // Reconstructing LEN from LEN2 proves a complete frame existed, so this is
    // corruption even though its semantic validation must fail.
    let body_end = bytes.len() - RECORD_FOOTER_SIZE;
    bytes[body_end - 1] = 2;
    let checksum = crc32(&bytes[record_offset..body_end]);
    bytes[body_end..body_end + 4].copy_from_slice(&checksum.to_le_bytes());
    bytes[record_offset + 4..record_offset + 8].copy_from_slice(&1_000_000u32.to_le_bytes());

    let mut slice_callbacks = 0;
    let slice_result = replay_bytes(&bytes, &mut |_, _, _| {
        slice_callbacks += 1;
        Ok(())
    });
    assert!(
        matches!(
            slice_result,
            Err(Error::ReplaySanityFailed {
                context: "truncated record precedes validated WAL data",
                ..
            })
        ),
        "unexpected slice result: {slice_result:?}"
    );
    assert_eq!(slice_callbacks, 0);

    fs::write(&path, &bytes).unwrap();
    let mut file_callbacks = 0;
    let file_result = replay(&path, |_, _, _| {
        file_callbacks += 1;
        Ok(())
    });
    assert!(matches!(
        file_result,
        Err(Error::ReplaySanityFailed {
            context: "truncated record precedes validated WAL data",
            ..
        })
    ));
    assert_eq!(file_callbacks, 0);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[test]
fn declared_length_above_native_ceiling_is_corruption_not_torn_tail() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    let record_offset = bytes.len();
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"bad-length".to_vec(),
        },
        1,
        &mut bytes,
    );
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"later".to_vec(),
        },
        2,
        &mut bytes,
    );
    bytes[record_offset + 4..record_offset + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    fs::write(&path, &bytes).unwrap();

    let mut callbacks = 0usize;
    let error = replay(&path, |_, _, _| {
        callbacks += 1;
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(
        error,
        Error::ReplaySanityFailed {
            context: "record exceeds native atomic WAL ceiling",
            record_offset: off,
        } if off == record_offset as u64
    ));
    assert_eq!(callbacks, 0);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

fn batch_insert_run_body(batch_count: u32, run_count: Option<u32>) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u64.to_le_bytes());
    body.extend_from_slice(&batch_count.to_le_bytes());
    if let Some(run_count) = run_count {
        body.push(11); // BatchInsertRun
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&run_count.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes()); // key_len
        body.extend_from_slice(&0u32.to_le_bytes()); // value_len
    }
    body
}

#[test]
fn logical_operation_ceiling_is_bounded_and_reader_accepts_its_boundary() {
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    let count = u32::try_from(MAX_ATOMIC_WAL_OPS).unwrap();
    append_raw_record(
        &mut bytes,
        1,
        10,
        &batch_insert_run_body(count, Some(count)),
    );

    let mut callbacks = 0usize;
    let (_, stats) = replay_bytes(&bytes, &mut |_, _, _| {
        callbacks += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(callbacks, MAX_ATOMIC_WAL_OPS);
    assert_eq!(stats.records_seen, 1);
    assert_eq!(stats.highest_seq, Some(MAX_ATOMIC_WAL_OPS as u64));

    let mut over = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut over,
    );
    append_raw_record(&mut over, 1, 10, &batch_insert_run_body(count + 1, None));
    let mut rejected_callbacks = 0usize;
    let error = replay_bytes(&over, &mut |_, _, _| {
        rejected_callbacks += 1;
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, Error::ReplaySanityFailed { .. }));
    assert_eq!(rejected_callbacks, 0);
}

#[test]
fn overflowing_batch_sequence_range_is_rejected_before_callbacks() {
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    // Sequence overflow is rejected from the outer count before the decoder
    // attempts to visit either primitive.
    append_raw_record(&mut bytes, u64::MAX, 10, &batch_insert_run_body(2, None));
    let mut callbacks = 0usize;
    let error = replay_bytes(&bytes, &mut |_, _, _| {
        callbacks += 1;
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, Error::ReplaySanityFailed { .. }));
    assert_eq!(callbacks, 0);

    let mut primitive = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut primitive,
    );
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"max-seq".to_vec(),
        },
        u64::MAX,
        &mut primitive,
    );
    let mut primitive_callbacks = 0usize;
    assert!(matches!(
        replay_bytes(&primitive, &mut |_, _, _| {
            primitive_callbacks += 1;
            Ok(())
        }),
        Err(Error::ReplaySanityFailed { .. })
    ));
    assert_eq!(primitive_callbacks, 0);

    let mut last_inner_is_max = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut last_inner_is_max,
    );
    append_raw_record(
        &mut last_inner_is_max,
        u64::MAX - 1,
        10,
        &batch_insert_run_body(2, Some(2)),
    );
    let mut batch_callbacks = 0usize;
    assert!(matches!(
        replay_bytes(&last_inner_is_max, &mut |_, _, _| {
            batch_callbacks += 1;
            Ok(())
        }),
        Err(Error::ReplaySanityFailed { .. })
    ));
    assert_eq!(batch_callbacks, 0);
}

#[test]
fn replay_high_water_is_maximum_not_physical_record_order() {
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"high".to_vec(),
        },
        10,
        &mut bytes,
    );
    encode_record(
        &WalOp::Erase {
            tree_id: 0,
            key: b"low".to_vec(),
        },
        3,
        &mut bytes,
    );
    let (_, stats) = replay_bytes(&bytes, &mut |_, _, _| Ok(())).unwrap();
    assert_eq!(stats.records_seen, 2);
    assert_eq!(stats.highest_seq, Some(10));
}

#[test]
#[ignore = "165 MiB full provider-envelope qualification"]
fn full_nokv_rename_envelope_streams_with_exact_v4_wire_size() {
    const KEY_BYTES: usize = 8_205;
    const VALUE_BYTES: usize = 61_493;
    const OPERATIONS: usize = 2_128;
    const EXPECTED_RECORD_BYTES: usize = 165_824_437;

    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    let record_start = bytes.len();
    let src = vec![0x51; KEY_BYTES];
    let dst = vec![0x52; KEY_BYTES];
    let value = vec![0x53; VALUE_BYTES];
    let mut encoder = BatchEncoder::begin(&mut bytes, 1, 0);
    for _ in 0..OPERATIONS {
        encoder.push_rename_object(0, &src, &dst, &value, true);
    }
    assert_eq!(encoder.finish() as usize, OPERATIONS);
    assert_eq!(bytes.len() - record_start, EXPECTED_RECORD_BYTES);
    const {
        assert!(EXPECTED_RECORD_BYTES < crate::DB::MAX_ATOMIC_RECORD_BYTES);
    }
    fs::write(&path, &bytes).unwrap();
    drop(bytes);

    let mut callbacks = 0usize;
    let (_, stats) = replay(&path, |op, seq, _| {
        let WalOp::RenameObject {
            src_key,
            dst_key,
            value,
            force,
            ..
        } = op
        else {
            panic!("expected RenameObject");
        };
        assert_eq!(src_key.len(), KEY_BYTES);
        assert_eq!(dst_key.len(), KEY_BYTES);
        assert_eq!(value.len(), VALUE_BYTES);
        assert!(*force);
        assert_eq!(seq, 1 + callbacks as u64);
        callbacks += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(callbacks, OPERATIONS);
    assert_eq!(stats.records_seen, 1);
    assert_eq!(stats.highest_seq, Some(OPERATIONS as u64));
}

#[test]
fn replay_callback_can_short_circuit() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let mut w = WalWriter::create(&path, 0).unwrap();
    for i in 0..10 {
        w.append(
            &WalOp::Insert {
                tree_id: 0,
                key: format!("k{i}").into_bytes(),
                value: vec![i as u8],
            },
            i + 1,
        )
        .unwrap();
    }
    w.flush().unwrap();

    // Force the callback to fail at the 4th record.
    let mut count = 0;
    let r = replay(&path, |_, _, _| {
        count += 1;
        if count == 4 {
            Err(Error::NotFound)
        } else {
            Ok(())
        }
    });
    assert!(matches!(r, Err(Error::NotFound)));
    assert_eq!(count, 4);
}

#[test]
fn rejected_file_with_wrong_magic() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    // Hand-craft a "WAL file" with bogus magic.
    let mut bogus = vec![0u8; FILE_HEADER_SIZE];
    bogus[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    bogus[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    fs::write(&path, &bogus).unwrap();

    match replay(&path, |_, _, _| Ok(())) {
        Err(Error::ReplaySanityFailed { context, .. }) => {
            assert!(context.contains("magic"));
        }
        other => panic!("expected magic mismatch, got {other:?}"),
    }
}

#[test]
fn rejected_file_with_unsupported_version() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let mut bogus = vec![0u8; FILE_HEADER_SIZE];
    bogus[0..4].copy_from_slice(&super::codec::FILE_MAGIC.to_le_bytes());
    // Format v4 deliberately has no dual-read path for v3: RenameObject's
    // payload and the record footer both changed, so startup must fail closed.
    bogus[4..8].copy_from_slice(&(FORMAT_VERSION - 1).to_le_bytes());
    fs::write(&path, &bogus).unwrap();

    match replay(&path, |_, _, _| Ok(())) {
        Err(Error::ReplaySanityFailed { context, .. }) => {
            assert!(context.contains("version"));
        }
        other => panic!("expected version mismatch, got {other:?}"),
    }
}

#[test]
fn owner_mismatch_is_rejected_before_any_callback_or_file_change() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut writer = WalWriter::create(&path, 7).unwrap();
    writer
        .append(
            &WalOp::Insert {
                tree_id: 7,
                key: b"wrong-owner".to_vec(),
                value: b"value".to_vec(),
            },
            1,
        )
        .unwrap();
    writer.flush().unwrap();
    drop(writer);

    let before = fs::read(&path).unwrap();
    let file = fs::File::open(&path).unwrap();
    let mut callbacks = 0;
    let result = replay_file(&file, 0, None, |_, _, _| {
        callbacks += 1;
        Ok(())
    });
    assert!(matches!(
        result,
        Err(Error::ReplaySanityFailed {
            context: "WAL file tree_id mismatch on open",
            record_offset: 0,
        })
    ));
    assert_eq!(callbacks, 0);
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn late_standalone_record_owner_mismatch_is_rejected_before_any_callback() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut writer = WalWriter::create(&path, 0).unwrap();
    for (seq, tree_id) in [(1, 0), (2, 9)] {
        writer
            .append(
                &WalOp::Insert {
                    tree_id,
                    key: format!("owner-{tree_id}").into_bytes(),
                    value: b"value".to_vec(),
                },
                seq,
            )
            .unwrap();
    }
    writer.flush().unwrap();
    drop(writer);

    let before = fs::read(&path).unwrap();
    let file = fs::File::open(&path).unwrap();
    let mut callbacks = 0;
    let result = replay_file(&file, 0, Some(0), |_, _, _| {
        callbacks += 1;
        Ok(())
    });
    assert!(matches!(
        result,
        Err(Error::ReplaySanityFailed {
            context: "WAL record tree_id does not belong to this Tree",
            ..
        })
    ));
    assert_eq!(callbacks, 0);
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn noncanonical_rename_force_is_rejected_before_any_callback() {
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            tree_id: 0,
            created_at: 0,
        },
        &mut bytes,
    );
    encode_record(
        &WalOp::RenameObject {
            tree_id: 0,
            src_key: b"src".to_vec(),
            dst_key: b"dst".to_vec(),
            value: b"value".to_vec(),
            force: false,
        },
        1,
        &mut bytes,
    );
    let record_start = FILE_HEADER_SIZE;
    let body_end = bytes.len() - RECORD_FOOTER_SIZE;
    bytes[body_end - 1] = 2;
    let checksum = crc32(&bytes[record_start..body_end]);
    bytes[body_end..body_end + 4].copy_from_slice(&checksum.to_le_bytes());

    let mut callbacks = 0;
    let result = replay_bytes(&bytes, &mut |_, _, _| {
        callbacks += 1;
        Ok(())
    });
    assert!(matches!(
        result,
        Err(Error::ReplaySanityFailed {
            context: "RenameObject force flag is not canonical",
            ..
        })
    ));
    assert_eq!(callbacks, 0);
}

#[test]
fn discard_pending_keeps_already_flushed_records() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut w = WalWriter::create(&path, 0).unwrap();

    w.append(
        &WalOp::Insert {
            tree_id: 0,
            key: b"k1".to_vec(),
            value: b"v1".to_vec(),
        },
        1,
    )
    .unwrap();
    w.flush().unwrap();

    w.append(
        &WalOp::Insert {
            tree_id: 0,
            key: b"k2".to_vec(),
            value: b"v2".to_vec(),
        },
        2,
    )
    .unwrap();
    w.discard_pending();
    drop(w);

    let (_h, stats) = replay(&path, |_, _, _| Ok(())).unwrap();
    assert_eq!(stats.records_seen, 1);
    assert_eq!(stats.highest_seq, Some(1));
}

#[test]
fn empty_wal_file_after_header_only() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut w = WalWriter::create(&path, 0).unwrap();
    w.flush().unwrap();
    drop(w);

    let (header, stats) = replay(&path, |_, _, _| Ok(())).unwrap();
    assert_eq!(header.tree_id, 0);
    assert_eq!(stats.records_seen, 0);
    assert_eq!(stats.highest_seq, None);
    assert_eq!(stats.torn_tail_at, None);
}

#[test]
fn truncate_reuses_live_wal_file_in_place() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut w = WalWriter::create(&path, 0).unwrap();
    w.append(
        &WalOp::Insert {
            tree_id: 0,
            key: b"before-truncate".to_vec(),
            value: b"v".to_vec(),
        },
        1,
    )
    .unwrap();
    w.flush().unwrap();
    assert!(fs::metadata(&path).unwrap().len() > FILE_HEADER_SIZE as u64);

    w.truncate().unwrap();
    assert_eq!(w.bytes_written(), FILE_HEADER_SIZE as u64);
    assert_eq!(fs::metadata(&path).unwrap().len(), FILE_HEADER_SIZE as u64);

    w.append(
        &WalOp::Insert {
            tree_id: 0,
            key: b"after-truncate".to_vec(),
            value: b"v2".to_vec(),
        },
        2,
    )
    .unwrap();
    w.flush().unwrap();
    drop(w);

    let mut seen = Vec::new();
    let (header, stats) = replay(&path, |op, seq, _| {
        seen.push((seq, op.clone()));
        Ok(())
    })
    .unwrap();
    assert_eq!(header.tree_id, 0);
    assert_eq!(stats.records_seen, 1);
    assert_eq!(stats.highest_seq, Some(2));
    assert_eq!(seen.len(), 1);
}

#[test]
fn many_records_stream_round_trip() {
    // ~5 KB of records, ensuring the buffered append path handles
    // many writes between flushes.
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    const N: u64 = 200;
    {
        let mut w = WalWriter::create(&path, 0).unwrap();
        for i in 1..=N {
            w.append(
                &WalOp::Insert {
                    tree_id: 0,
                    key: format!("k{i:04}").into_bytes(),
                    value: format!("v{i}").into_bytes(),
                },
                i,
            )
            .unwrap();
        }
        w.flush().unwrap();
    }

    let mut max_seq = 0u64;
    let (_h, stats) = replay(&path, |_, seq, _| {
        max_seq = max_seq.max(seq);
        Ok(())
    })
    .unwrap();
    assert_eq!(stats.records_seen, N);
    assert_eq!(stats.highest_seq, Some(N));
    assert_eq!(max_seq, N);
}

#[test]
fn auto_flush_keeps_user_space_buffer_bounded() {
    // Stress test buffered auto-drain: append records until the
    // per-record cost would otherwise pile up an
    // unbounded `Vec`. The file should grow past the auto-flush
    // threshold while the in-memory buffer stays small between
    // calls.
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let mut w = WalWriter::create(&path, 0).unwrap();
    // ~80 bytes per record means crossing the 64 KB threshold
    // happens roughly every ~800 appends. Push 3× that to see
    // the auto-flush fire multiple times.
    let target_records = (AUTO_FLUSH_THRESHOLD / 80) * 3;
    for i in 0..target_records as u64 {
        w.append(
            &WalOp::Insert {
                tree_id: 0,
                key: format!("k{i:06}").into_bytes(),
                value: vec![0xAB; 32],
            },
            i + 1,
        )
        .unwrap();
    }

    // The file on disk should already have grown well past the
    // header — the auto-flush is what drained the bytes there.
    // (We don't call `flush()` ourselves.)
    let on_disk_before_flush = fs::metadata(&path).unwrap().len();
    assert!(
        on_disk_before_flush > AUTO_FLUSH_THRESHOLD as u64,
        "auto-flush should have pushed bytes to disk: on-disk = {on_disk_before_flush}",
    );

    // The pending tail since the last auto-drain is bounded
    // by the threshold (the auto-flush triggers as soon as we
    // cross, so the next-cycle pending starts at 0 and never
    // exceeds threshold + one record's worth).
    let pending_upper_bound = AUTO_FLUSH_THRESHOLD + 256;
    let bytes_written_total = w.bytes_written();
    let pending_size = bytes_written_total - on_disk_before_flush;
    assert!(
        pending_size <= pending_upper_bound as u64,
        "pending tail should be bounded: {pending_size} bytes",
    );

    // Final flush ensures durability and the file holds every
    // record we appended.
    w.flush().unwrap();
    drop(w);

    let mut seen = Vec::new();
    let (_h, stats) = replay(&path, |_, seq, _| {
        seen.push(seq);
        Ok(())
    })
    .unwrap();
    assert_eq!(stats.records_seen, target_records as u64);
    assert_eq!(stats.highest_seq, Some(target_records as u64));
    assert_eq!(stats.torn_tail_at, None);
}

// Sanity: prevent the WAL writer from silently leaving the file
// in an over-extended state when an unsupported "seek + truncate"
// pattern races with the append cursor. Holding the writer in
// append-only mode means the OS keeps the cursor at EOF
// regardless of out-of-band manipulation.
#[test]
fn appending_after_external_truncate_grows_file_again() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let mut w = WalWriter::create(&path, 0).unwrap();
    w.append(
        &WalOp::Insert {
            tree_id: 0,
            key: b"keep".to_vec(),
            value: b"v".to_vec(),
        },
        1,
    )
    .unwrap();
    w.flush().unwrap();

    // Out-of-band truncate the file back to just the header.
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.set_len(FILE_HEADER_SIZE as u64).unwrap();
    // Touch the no-longer-relevant variables so clippy doesn't
    // squawk about unused.
    let mut f = f;
    f.seek(SeekFrom::Start(FILE_HEADER_SIZE as u64)).unwrap();
    let _ = f.write(&[]).unwrap();

    // The writer still appends successfully.
    w.append(
        &WalOp::Insert {
            tree_id: 0,
            key: b"after-truncate".to_vec(),
            value: b"v".to_vec(),
        },
        2,
    )
    .unwrap();
    w.flush().unwrap();
    drop(w);

    let mut seqs = Vec::new();
    let _ = replay(&path, |_, seq, _| {
        seqs.push(seq);
        Ok(())
    })
    .unwrap();
    assert_eq!(seqs.last().copied(), Some(2));
}
