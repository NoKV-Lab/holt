//! Per-page checksums for authoritative packed blob slots.
//!
//! A record is stored at the same slot number as its blob. The manifest may
//! reference a new slot only after both files have been synced. Reusable slots
//! are fenced by the manifest's existing reader drain before either record is
//! overwritten.

use crate::api::errors::{Error, Result};
use crate::layout::{BlobGuid, PAGE_SIZE};

const MAGIC: [u8; 8] = *b"HOLTCKS1";
const PAGE_BYTES: usize = 4096;
const PAGE_COUNT: usize = PAGE_SIZE as usize / PAGE_BYTES;
const PAGE_CHECKSUMS_OFFSET: usize = 32;
const COVERED_PAGES_OFFSET: usize = 544;
const RECORD_CHECKSUM_OFFSET: usize = 572;
pub(super) const RECORD_BYTES: usize = 576;

const _: () = assert!(PAGE_COUNT == 128);
const _: () = assert!(PAGE_CHECKSUMS_OFFSET + PAGE_COUNT * 4 <= RECORD_CHECKSUM_OFFSET);

pub(super) fn record_offset(slot: u64) -> Result<u64> {
    slot.checked_mul(RECORD_BYTES as u64)
        .ok_or(Error::node_corrupt("FileBlobStore::Integrity::slot offset"))
}

pub(super) fn encode(guid: BlobGuid, slot: u64, frame: &[u8]) -> [u8; RECORD_BYTES] {
    encode_prefix(guid, slot, frame, frame.len())
}

pub(super) fn encode_prefix(
    guid: BlobGuid,
    slot: u64,
    frame: &[u8],
    covered_bytes: usize,
) -> [u8; RECORD_BYTES] {
    debug_assert_eq!(frame.len(), PAGE_SIZE as usize);
    debug_assert!(
        covered_bytes > 0 && covered_bytes <= frame.len() && covered_bytes % PAGE_BYTES == 0
    );
    let mut record = [0u8; RECORD_BYTES];
    record[..8].copy_from_slice(&MAGIC);
    record[8..24].copy_from_slice(&guid);
    record[24..32].copy_from_slice(&slot.to_le_bytes());
    for (page, bytes) in frame[..covered_bytes].chunks_exact(PAGE_BYTES).enumerate() {
        let start = PAGE_CHECKSUMS_OFFSET + page * 4;
        record[start..start + 4].copy_from_slice(&crc32fast::hash(bytes).to_le_bytes());
    }
    record[COVERED_PAGES_OFFSET..COVERED_PAGES_OFFSET + 4]
        .copy_from_slice(&((covered_bytes / PAGE_BYTES) as u32).to_le_bytes());
    let checksum = crc32fast::hash(&record[..RECORD_CHECKSUM_OFFSET]);
    record[RECORD_CHECKSUM_OFFSET..].copy_from_slice(&checksum.to_le_bytes());
    record
}

fn validate_record(record: &[u8; RECORD_BYTES], guid: BlobGuid, slot: u64) -> Result<()> {
    let stored = u32::from_le_bytes(record[RECORD_CHECKSUM_OFFSET..].try_into().unwrap());
    let covered_pages = u32::from_le_bytes(
        record[COVERED_PAGES_OFFSET..COVERED_PAGES_OFFSET + 4]
            .try_into()
            .unwrap(),
    );
    if record[..8] != MAGIC
        || record[8..24] != guid
        || record[24..32] != slot.to_le_bytes()
        || !(1..=PAGE_COUNT as u32).contains(&covered_pages)
        || stored != crc32fast::hash(&record[..RECORD_CHECKSUM_OFFSET])
    {
        return Err(Error::node_corrupt("FileBlobStore::Integrity::record").with_blob_guid(guid));
    }
    Ok(())
}

pub(super) fn verify_full(
    record: &[u8; RECORD_BYTES],
    guid: BlobGuid,
    slot: u64,
    frame: &[u8],
) -> Result<()> {
    validate_record(record, guid, slot)?;
    debug_assert_eq!(frame.len(), PAGE_SIZE as usize);
    verify_pages(record, guid, 0, frame)
}

pub(super) fn verify_range(
    record: &[u8; RECORD_BYTES],
    guid: BlobGuid,
    slot: u64,
    byte_offset: u64,
    bytes: &[u8],
) -> Result<()> {
    validate_record(record, guid, slot)?;
    let start = usize::try_from(byte_offset)
        .map_err(|_| Error::node_corrupt("FileBlobStore::Integrity::range"))?;
    if start % PAGE_BYTES != 0
        || bytes.len() % PAGE_BYTES != 0
        || start
            .checked_add(bytes.len())
            .is_none_or(|end| end > PAGE_SIZE as usize)
    {
        return Err(Error::node_corrupt("FileBlobStore::Integrity::range").with_blob_guid(guid));
    }
    verify_pages(record, guid, start / PAGE_BYTES, bytes)
}

fn verify_pages(
    record: &[u8; RECORD_BYTES],
    guid: BlobGuid,
    first_page: usize,
    bytes: &[u8],
) -> Result<()> {
    let covered_pages = u32::from_le_bytes(
        record[COVERED_PAGES_OFFSET..COVERED_PAGES_OFFSET + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    for (index, page) in bytes.chunks_exact(PAGE_BYTES).enumerate() {
        if first_page + index >= covered_pages {
            continue;
        }
        let start = PAGE_CHECKSUMS_OFFSET + (first_page + index) * 4;
        let expected = u32::from_le_bytes(record[start..start + 4].try_into().unwrap());
        if crc32fast::hash(page) != expected {
            return Err(Error::node_corrupt("FileBlobStore::Integrity::page").with_blob_guid(guid));
        }
    }
    Ok(())
}

/// Legacy compact frames may leave stale, non-authoritative bytes beyond
/// their persisted prefix. A verified record makes that boundary explicit.
pub(super) fn clear_uncovered_tail(record: &[u8; RECORD_BYTES], frame: &mut [u8]) {
    let covered_pages = u32::from_le_bytes(
        record[COVERED_PAGES_OFFSET..COVERED_PAGES_OFFSET + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    frame[covered_pages * PAGE_BYTES..].fill(0);
}

/// Retarget a record after the caller has verified its source frame.
pub(super) fn relocate_verified(record: &[u8; RECORD_BYTES], new_slot: u64) -> [u8; RECORD_BYTES] {
    let mut relocated = *record;
    relocated[24..32].copy_from_slice(&new_slot.to_le_bytes());
    let checksum = crc32fast::hash(&relocated[..RECORD_CHECKSUM_OFFSET]);
    relocated[RECORD_CHECKSUM_OFFSET..].copy_from_slice(&checksum.to_le_bytes());
    relocated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_fault_is_rejected_in_full_and_range_reads() {
        let guid = [7; 16];
        let mut frame = vec![0u8; PAGE_SIZE as usize];
        frame[PAGE_BYTES * 19 + 31] = 0x42;
        let record = encode(guid, 3, &frame);
        verify_full(&record, guid, 3, &frame).unwrap();
        verify_range(
            &record,
            guid,
            3,
            (19 * PAGE_BYTES) as u64,
            &frame[19 * PAGE_BYTES..20 * PAGE_BYTES],
        )
        .unwrap();
        frame[PAGE_BYTES * 19 + 31] ^= 1;
        assert!(verify_full(&record, guid, 3, &frame).is_err());
        assert!(verify_range(
            &record,
            guid,
            3,
            (19 * PAGE_BYTES) as u64,
            &frame[19 * PAGE_BYTES..20 * PAGE_BYTES]
        )
        .is_err());
    }

    #[test]
    fn record_is_bound_to_guid_slot_and_checksum() {
        let frame = vec![0u8; PAGE_SIZE as usize];
        let guid = [7; 16];
        let mut record = encode(guid, 3, &frame);
        assert!(verify_full(&record, [8; 16], 3, &frame).is_err());
        assert!(verify_full(&record, guid, 4, &frame).is_err());
        record[100] ^= 1;
        assert!(verify_full(&record, guid, 3, &frame).is_err());
    }

    #[test]
    fn compact_prefix_ignores_stale_tail_but_clears_it_for_full_reads() {
        let guid = [9; 16];
        let mut frame = vec![3u8; PAGE_SIZE as usize];
        let record = encode_prefix(guid, 2, &frame, PAGE_BYTES * 2);
        frame[PAGE_BYTES * 3] ^= 1;
        verify_full(&record, guid, 2, &frame).unwrap();
        clear_uncovered_tail(&record, &mut frame);
        assert_eq!(frame[PAGE_BYTES * 2], 0);
        frame[PAGE_BYTES] ^= 1;
        assert!(verify_full(&record, guid, 2, &frame).is_err());
    }
}
