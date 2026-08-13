//! End-to-end unit tests for the WAL writer + replay scanner —
//! write some records, flush, scan, verify what comes back
//! matches what went in.

use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

use tempfile::tempdir;

use super::codec::{
    crc32, decode_file_header, encode_file_header, BatchEncoder, FileHeader, FILE_HEADER_SIZE,
    FORMAT_VERSION, LEGACY_FILE_HEADER_SIZE, LEGACY_FORMAT_VERSION, RECORD_MAGIC,
};
use super::reader::replay;
use super::wal_op::WalOp;
use super::writer::{WalWriter, AUTO_FLUSH_THRESHOLD};
use crate::api::errors::Error;
use crate::{JournalAnchor, JournalEnvelope};

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
            force: false,
        },
    ]
}

fn sample_envelope() -> JournalEnvelope {
    JournalEnvelope::new(
        JournalAnchor::new(6, [0x66; 32]),
        JournalAnchor::new(7, [0x77; 32]),
        b"canonical-recovery-v1".to_vec(),
    )
    .unwrap()
}

#[test]
fn v3_and_v4_file_headers_have_stable_golden_bytes() {
    let mut expected_prefix = [
        0x57, 0x41, 0x4c, 0x41, 0x04, 0x00, 0x00, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02,
        0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ];
    let header_v4 = FileHeader {
        version: FORMAT_VERSION,
        tree_id: 0x0102_0304_0506_0708,
        created_at: 0x1112_1314_1516_1718,
        checkpoint_anchor: None,
        anchor_generation: 0,
    };
    let mut encoded = Vec::new();
    encode_file_header(&header_v4, &mut encoded);
    assert_eq!(encoded.len(), FILE_HEADER_SIZE);
    assert_eq!(&encoded[..LEGACY_FILE_HEADER_SIZE], expected_prefix);
    assert!(encoded[LEGACY_FILE_HEADER_SIZE..]
        .iter()
        .all(|byte| *byte == 0));
    assert_eq!(decode_file_header(&encoded).unwrap(), header_v4);

    expected_prefix[4] = 0x03;
    expected_prefix[24..32].fill(0);
    let header_v3 = FileHeader {
        version: LEGACY_FORMAT_VERSION,
        checkpoint_anchor: None,
        anchor_generation: 0,
        ..header_v4
    };
    encoded.clear();
    encode_file_header(&header_v3, &mut encoded);
    assert_eq!(encoded, expected_prefix);
    assert_eq!(decode_file_header(&encoded).unwrap(), header_v3);
}

#[test]
fn nonempty_format_3_is_replayable_but_rejected_for_append() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let mut bytes = Vec::new();
    encode_file_header(
        &FileHeader {
            version: LEGACY_FORMAT_VERSION,
            tree_id: 9,
            created_at: 0,
            checkpoint_anchor: None,
            anchor_generation: 0,
        },
        &mut bytes,
    );
    let mut body = Vec::new();
    body.extend_from_slice(&9u64.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(b"k");
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(b"v");
    append_raw_record(&mut bytes, 11, 0, &body);
    fs::write(&path, &bytes).unwrap();

    let mut seen = Vec::new();
    let (header, stats) = replay(&path, |op, seq, _| {
        seen.push((op.clone(), seq));
        Ok(())
    })
    .unwrap();
    assert_eq!(header.version, LEGACY_FORMAT_VERSION);
    assert_eq!(header.tree_id, 9);
    assert_eq!(stats.records_seen, 1);
    assert_eq!(stats.highest_seq, Some(11));
    assert!(matches!(
        seen.as_slice(),
        [(WalOp::Insert {
            tree_id: 9,
            key,
            value,
        }, 11)] if key == b"k" && value == b"v"
    ));

    let before = fs::read(&path).unwrap();
    assert!(matches!(
        WalWriter::open_existing(&path),
        Err(Error::ReplaySanityFailed {
            context: "nonempty WAL format 3 is replay-only; checkpoint it with a format-3 Holt binary before v4 writes",
            record_offset: 0,
        })
    ));
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn header_only_format_3_is_upgraded_before_append() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let legacy = FileHeader {
        version: LEGACY_FORMAT_VERSION,
        tree_id: 17,
        created_at: 23,
        checkpoint_anchor: None,
        anchor_generation: 0,
    };
    let mut bytes = Vec::new();
    encode_file_header(&legacy, &mut bytes);
    fs::write(&path, &bytes).unwrap();

    let writer = WalWriter::open_existing(&path).unwrap();
    assert_eq!(writer.header().version, FORMAT_VERSION);
    assert_eq!(writer.header().tree_id, legacy.tree_id);
    assert_eq!(writer.header().created_at, legacy.created_at);
    drop(writer);

    let upgraded = fs::read(&path).unwrap();
    assert_eq!(upgraded.len(), FILE_HEADER_SIZE);
    let header = decode_file_header(&upgraded).unwrap();
    assert_eq!(header.version, FORMAT_VERSION);
    assert_eq!(header.tree_id, legacy.tree_id);
    assert_eq!(header.created_at, legacy.created_at);
}

#[test]
fn format_4_attached_batch_flattens_for_replay_and_is_rejected_under_v3_header() {
    let envelope = sample_envelope();
    let mut attached_record = Vec::new();
    {
        let mut encoder = BatchEncoder::begin_with_envelope(&mut attached_record, 20, 0, &envelope);
        encoder.push_insert(3, b"a", b"one");
        encoder.push_erase(3, b"b");
        assert_eq!(encoder.finish(), 2);
    }

    let dir = tempdir().unwrap();
    let v4_path = dir.path().join("v4.wal");
    let mut v4 = Vec::new();
    encode_file_header(
        &FileHeader {
            version: FORMAT_VERSION,
            tree_id: 0,
            created_at: 0,
            checkpoint_anchor: None,
            anchor_generation: 0,
        },
        &mut v4,
    );
    v4.extend_from_slice(&attached_record);
    fs::write(&v4_path, &v4).unwrap();

    let mut seen = Vec::new();
    let (header, stats) = replay(&v4_path, |op, seq, _| {
        seen.push((op.clone(), seq));
        Ok(())
    })
    .unwrap();
    assert_eq!(header.version, FORMAT_VERSION);
    assert_eq!(stats.records_seen, 1);
    assert_eq!(stats.highest_seq, Some(21));
    assert_eq!(seen.len(), 2);
    assert!(matches!(seen[0], (WalOp::Insert { .. }, 20)));
    assert!(matches!(seen[1], (WalOp::Erase { .. }, 21)));

    let v3_path = dir.path().join("v3-with-tag-13.wal");
    let mut v3 = Vec::new();
    encode_file_header(
        &FileHeader {
            version: LEGACY_FORMAT_VERSION,
            tree_id: 0,
            created_at: 0,
            checkpoint_anchor: None,
            anchor_generation: 0,
        },
        &mut v3,
    );
    v3.extend_from_slice(&attached_record);
    fs::write(&v3_path, &v3).unwrap();
    assert!(matches!(
        replay(&v3_path, |_, _, _| Ok(())),
        Err(Error::ReplaySanityFailed {
            context: "attached batch requires WAL format version 4",
            record_offset,
        }) if record_offset == LEGACY_FILE_HEADER_SIZE as u64
    ));
}

#[test]
fn checkpoint_anchor_slots_fall_back_to_mirrored_anchor() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let first = JournalAnchor::new(1, [0x31; 32]);
    let second = JournalAnchor::new(2, [0x32; 32]);
    {
        let mut writer = WalWriter::create(&path, 0).unwrap();
        writer.persist_checkpoint_anchor(first).unwrap();
        writer.persist_checkpoint_anchor(second).unwrap();
    }

    let bytes = fs::read(&path).unwrap();
    let header = decode_file_header(&bytes).unwrap();
    assert_eq!(header.checkpoint_anchor, Some(second));
    assert_eq!(header.anchor_generation, 4);

    // Generation 4 lives in slot B. If it is lost, generation 3 in slot A
    // still names the same checkpoint anchor.
    let mut torn = bytes;
    let slot_b_crc_byte = super::codec::ANCHOR_SLOT_OFFSETS[1] + 64;
    torn[slot_b_crc_byte] ^= 0x80;
    fs::write(&path, &torn).unwrap();
    let recovered = decode_file_header(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(recovered.checkpoint_anchor, Some(second));
    assert_eq!(recovered.anchor_generation, 3);
}

#[test]
fn first_checkpoint_anchor_is_mirrored_before_initialization_returns() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let genesis = JournalAnchor::new(0, [0x39; 32]);
    {
        let mut writer = WalWriter::create(&path, 0).unwrap();
        writer.persist_checkpoint_anchor(genesis).unwrap();
    }

    let bytes = fs::read(&path).unwrap();
    let header = decode_file_header(&bytes).unwrap();
    assert_eq!(header.checkpoint_anchor, Some(genesis));
    assert_eq!(header.anchor_generation, 2);

    for slot in 0..2 {
        let mut damaged = bytes.clone();
        damaged[super::codec::ANCHOR_SLOT_OFFSETS[slot] + 64] ^= 0x80;
        let recovered = decode_file_header(&damaged).unwrap();
        assert_eq!(recovered.checkpoint_anchor, Some(genesis));
    }
}

#[test]
fn advanced_checkpoint_anchor_survives_either_slot_corruption_after_truncate() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let genesis = JournalAnchor::new(0, [0x3a; 32]);
    let advanced = JournalAnchor::new(7, [0x3b; 32]);
    {
        let mut writer = WalWriter::create(&path, 0).unwrap();
        writer.persist_checkpoint_anchor(genesis).unwrap();
        writer.checkpoint_and_truncate(Some(advanced)).unwrap();
    }

    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.len(), FILE_HEADER_SIZE);
    let header = decode_file_header(&bytes).unwrap();
    assert_eq!(header.checkpoint_anchor, Some(advanced));
    assert_eq!(header.anchor_generation, 4);

    for slot in 0..2 {
        let mut damaged = bytes.clone();
        damaged[super::codec::ANCHOR_SLOT_OFFSETS[slot] + 64] ^= 0x80;
        let recovered = decode_file_header(&damaged).unwrap();
        assert_eq!(recovered.checkpoint_anchor, Some(advanced));
    }
}

#[test]
fn interrupted_checkpoint_keeps_wal_and_retry_repairs_anchor_mirror() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let genesis = JournalAnchor::new(0, [0x40; 32]);
    let first = JournalAnchor::new(1, [0x41; 32]);
    let second = JournalAnchor::new(2, [0x42; 32]);
    let first_envelope = JournalEnvelope::new(genesis, first, b"first".to_vec()).unwrap();
    let second_envelope = JournalEnvelope::new(first, second, b"second".to_vec()).unwrap();

    let mut writer = WalWriter::create(&path, 0).unwrap();
    writer.persist_checkpoint_anchor(genesis).unwrap();
    for (seq, envelope) in [(10, &first_envelope), (11, &second_envelope)] {
        let mut record = Vec::new();
        BatchEncoder::begin_with_envelope(&mut record, seq, 0, envelope).finish();
        writer.append_encoded(&record).unwrap();
    }
    writer.flush().unwrap();
    let wal_len = fs::metadata(&path).unwrap().len();

    // Fail after the first new generation reaches disk. Checkpoint must leave
    // the old record stream intact so recovery can validate the new floor and
    // retain the suffix after it.
    WalWriter::fail_anchor_generation_after_for_test(1);
    assert!(matches!(
        writer.checkpoint_and_truncate(Some(first)),
        Err(Error::Internal("checkpoint anchor generation test failure"))
    ));
    assert_eq!(fs::metadata(&path).unwrap().len(), wal_len);
    drop(writer);

    let scan = super::reader::scan_attached_envelopes(&path).unwrap();
    assert_eq!(scan.checkpoint, Some(first));
    assert_eq!(scan.tail, Some(second));
    assert_eq!(scan.envelopes, vec![second_envelope]);

    // Reopen sees the first new generation. Retrying the same checkpoint must
    // write its sibling before truncation; either slot can then be lost.
    {
        let mut writer = WalWriter::open_existing(&path).unwrap();
        assert_eq!(writer.header().checkpoint_anchor, Some(first));
        assert_eq!(writer.header().anchor_generation, 3);
        writer.checkpoint_and_truncate(Some(first)).unwrap();
    }

    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.len(), FILE_HEADER_SIZE);
    let header = decode_file_header(&bytes).unwrap();
    assert_eq!(header.checkpoint_anchor, Some(first));
    assert_eq!(header.anchor_generation, 4);
    for slot in 0..2 {
        let mut damaged = bytes.clone();
        damaged[super::codec::ANCHOR_SLOT_OFFSETS[slot] + 64] ^= 0x80;
        let recovered = decode_file_header(&damaged).unwrap();
        assert_eq!(recovered.checkpoint_anchor, Some(first));
    }
}

#[test]
fn create_open_round_trip_all_variants() {
    let dir = tempdir().unwrap();
    let path = wal_path(&dir);
    let ops = sample_ops();

    let mut w = WalWriter::create(&path, 42).unwrap();
    assert_eq!(w.header().version, FORMAT_VERSION);
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
            version: FORMAT_VERSION,
            tree_id: 0,
            created_at: 0,
            checkpoint_anchor: None,
            anchor_generation: 0,
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
