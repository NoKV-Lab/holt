//! End-to-end unit tests for the WAL writer + replay scanner —
//! write some records, flush, scan, verify what comes back
//! matches what went in.
//!
//! Previously lived as `tests/wal_round_trip.rs`. Moved inward
//! when the `journal` module became `pub(crate)`.

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

use tempfile::tempdir;

use super::codec::{FILE_HEADER_SIZE, FORMAT_VERSION};
use super::reader::replay;
use super::txn_op::TxnOp;
use super::writer::{WalWriter, AUTO_FLUSH_THRESHOLD};
use crate::api::errors::Error;

fn wal_path(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("test.wal")
}

fn sample_ops() -> Vec<TxnOp> {
    vec![
        TxnOp::Insert {
            tree_id: 0,
            seq: 1,
            key: b"img/01.jpg".to_vec(),
            value: vec![0xAA; 64],
        },
        TxnOp::Insert {
            tree_id: 0,
            seq: 2,
            key: b"img/02.jpg".to_vec(),
            value: vec![0xBB; 64],
        },
        TxnOp::Erase {
            tree_id: 0,
            seq: 3,
            key: b"img/01.jpg".to_vec(),
        },
        TxnOp::RenameObject {
            tree_id: 0,
            seq: 4,
            src_key: b"img/02.jpg".to_vec(),
            dst_key: b"img/02-renamed.jpg".to_vec(),
            force: false,
        },
        TxnOp::Split {
            parent_blob: [0; 16],
            pre_split_node: 7,
            new_child_blob: [0xCD; 16],
            new_child_entry: 1,
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
fn open_existing_resumes_append_position() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    {
        let mut w = WalWriter::create(&path, 7).unwrap();
        w.append(
            &TxnOp::Insert {
                tree_id: 0,
                seq: 1,
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
            &TxnOp::Erase {
                tree_id: 0,
                seq: 2,
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
            &TxnOp::Insert {
                tree_id: 0,
                seq: 1,
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
    let first_record_end = FILE_HEADER_SIZE + 17 + first_body_len + 4; // header(17) + body + CRC(4)
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
fn replay_callback_can_short_circuit() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);

    let mut w = WalWriter::create(&path, 0).unwrap();
    for i in 0..10 {
        w.append(
            &TxnOp::Insert {
                tree_id: 0,
                seq: i + 1,
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
    bogus[4..8].copy_from_slice(&999u32.to_le_bytes());
    fs::write(&path, &bogus).unwrap();

    match replay(&path, |_, _, _| Ok(())) {
        Err(Error::ReplaySanityFailed { context, .. }) => {
            assert!(context.contains("version"));
        }
        other => panic!("expected version mismatch, got {other:?}"),
    }
}

#[test]
fn discard_pending_keeps_already_flushed_records() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut w = WalWriter::create(&path, 0).unwrap();

    w.append(
        &TxnOp::Insert {
            tree_id: 0,
            seq: 1,
            key: b"k1".to_vec(),
            value: b"v1".to_vec(),
        },
        1,
    )
    .unwrap();
    w.flush().unwrap();

    w.append(
        &TxnOp::Insert {
            tree_id: 0,
            seq: 2,
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
                &TxnOp::Insert {
                    tree_id: 0,
                    seq: i,
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
    // Stress test the group-commit auto-flush: append records
    // until the per-record cost would otherwise pile up an
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
            &TxnOp::Insert {
                tree_id: 0,
                seq: i + 1,
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
        &TxnOp::Insert {
            tree_id: 0,
            seq: 1,
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
        &TxnOp::Insert {
            tree_id: 0,
            seq: 2,
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
