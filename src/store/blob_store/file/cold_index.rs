use crate::api::errors::{Error, Result};
use crate::engine::{lookup_cold_summary, summarize_blob_for_cold_index, ColdBlobSummary};
use crate::layout::BlobGuid;
use crate::store::{BlobFrameRef, ColdBlobLookup};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};

const COLD_MAGIC: [u8; 4] = *b"HCI1";
const COLD_HEADER_SIZE: usize = 12;
const COLD_TY_SET: u8 = 1;
const COLD_TY_DELETE: u8 = 2;
const COLD_INLINE_VALUE_LIMIT: usize = 4096;
const COLD_MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct ColdIndex {
    file: Mutex<File>,
    directory: RwLock<HashMap<BlobGuid, ColdIndexMeta>>,
    dirty: AtomicBool,
}

#[derive(Debug, Clone, Copy)]
struct ColdIndexMeta {
    generation: u64,
    payload_offset: u64,
    payload_len: u32,
    crc: u32,
}

impl ColdIndex {
    pub(super) fn open(path: PathBuf) -> Result<Self> {
        let (directory, valid_len) = replay(&path)?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)?;
        if file.metadata()?.len() != valid_len {
            file.set_len(valid_len)?;
        }
        Ok(Self {
            file: Mutex::new(file),
            directory: RwLock::new(directory),
            dirty: AtomicBool::new(false),
        })
    }

    pub(super) fn put_blob(&self, guid: BlobGuid, generation: u64, frame: &[u8]) -> Result<()> {
        let Ok(summary) =
            summarize_blob_for_cold_index(BlobFrameRef::wrap(frame), COLD_INLINE_VALUE_LIMIT)
        else {
            self.delete_blob(guid)?;
            return Ok(());
        };
        let payload = encode_set(guid, generation, &summary)?;
        let meta = self.append_payload(&payload)?;
        self.directory
            .write()
            .unwrap()
            .insert(guid, ColdIndexMeta { generation, ..meta });
        Ok(())
    }

    pub(super) fn delete_blob(&self, guid: BlobGuid) -> Result<()> {
        let payload = encode_delete(guid);
        self.append_payload(&payload)?;
        self.directory.write().unwrap().remove(&guid);
        Ok(())
    }

    pub(super) fn lookup_blob(
        &self,
        guid: BlobGuid,
        generation: u64,
        key: &[u8],
        depth: usize,
    ) -> Result<ColdBlobLookup> {
        let meta = {
            let directory = self.directory.read().unwrap();
            let Some(meta) = directory.get(&guid).copied() else {
                return Ok(ColdBlobLookup::Unknown);
            };
            if meta.generation != generation {
                return Ok(ColdBlobLookup::Unknown);
            }
            meta
        };

        let mut payload = vec![0u8; meta.payload_len as usize];
        {
            let file = self.file.lock().unwrap();
            file.read_exact_at(&mut payload, meta.payload_offset)?;
        }
        if crc32fast::hash(&payload) != meta.crc {
            return Ok(ColdBlobLookup::Unknown);
        }
        let summary = decode_set_payload(&payload, guid, generation)?;
        Ok(lookup_cold_summary(&summary, key, depth))
    }

    pub(super) fn flush(&self) -> Result<()> {
        if self.dirty.swap(false, Ordering::AcqRel) {
            if let Err(e) = self.file.lock().unwrap().sync_data() {
                self.dirty.store(true, Ordering::Release);
                return Err(Error::BlobStoreIo(e));
            }
        }
        Ok(())
    }

    pub(super) fn needs_flush(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn append_payload(&self, payload: &[u8]) -> Result<ColdIndexMeta> {
        let payload_len = u32::try_from(payload.len()).map_err(|_| {
            Error::BlobStoreIo(io::Error::other("cold index record exceeds u32::MAX"))
        })?;
        let crc = crc32fast::hash(payload);
        let mut record = Vec::with_capacity(COLD_HEADER_SIZE + payload.len());
        record.extend_from_slice(&COLD_MAGIC);
        record.extend_from_slice(&payload_len.to_le_bytes());
        record.extend_from_slice(&crc.to_le_bytes());
        record.extend_from_slice(payload);

        let mut file = self.file.lock().unwrap();
        let record_offset = file.seek(SeekFrom::End(0))?;
        file.write_all(&record)?;
        self.dirty.store(true, Ordering::Release);
        Ok(ColdIndexMeta {
            generation: 0,
            payload_offset: record_offset + COLD_HEADER_SIZE as u64,
            payload_len,
            crc,
        })
    }
}

fn replay(path: &PathBuf) -> Result<(HashMap<BlobGuid, ColdIndexMeta>, u64)> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((HashMap::new(), 0)),
        Err(e) => return Err(Error::BlobStoreIo(e)),
    };
    let file_len = file.metadata()?.len();
    let mut directory = HashMap::new();
    let mut offset = 0u64;
    while offset < file_len {
        let mut header = [0u8; COLD_HEADER_SIZE];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Error::BlobStoreIo(e)),
        }
        offset += COLD_HEADER_SIZE as u64;
        if header[..4] != COLD_MAGIC {
            break;
        }
        let payload_len = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let crc = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if payload_len as usize > COLD_MAX_RECORD_BYTES {
            break;
        }
        let payload_offset = offset;
        let Some(next_offset) = offset.checked_add(u64::from(payload_len)) else {
            break;
        };
        if next_offset > file_len {
            break;
        }
        let mut payload = vec![0u8; payload_len as usize];
        file.read_exact(&mut payload)?;
        offset = next_offset;
        if crc32fast::hash(&payload) != crc {
            break;
        }
        match payload.first().copied() {
            Some(COLD_TY_SET) => {
                let Ok((guid, generation)) = decode_set_header(&payload) else {
                    break;
                };
                directory.insert(
                    guid,
                    ColdIndexMeta {
                        generation,
                        payload_offset,
                        payload_len,
                        crc,
                    },
                );
            }
            Some(COLD_TY_DELETE) => {
                let Ok(guid) = decode_delete_payload(&payload) else {
                    break;
                };
                directory.remove(&guid);
            }
            _ => break,
        }
    }
    Ok((directory, offset))
}

fn encode_set(guid: BlobGuid, generation: u64, summary: &ColdBlobSummary) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.push(COLD_TY_SET);
    out.extend_from_slice(&guid);
    out.extend_from_slice(&generation.to_le_bytes());
    let leaf_count = u32::try_from(summary.leaves.len())
        .map_err(|_| Error::BlobStoreIo(io::Error::other("cold index leaf count overflow")))?;
    let crossing_count = u32::try_from(summary.crossings.len())
        .map_err(|_| Error::BlobStoreIo(io::Error::other("cold index crossing count overflow")))?;
    out.extend_from_slice(&leaf_count.to_le_bytes());
    out.extend_from_slice(&crossing_count.to_le_bytes());
    for leaf in &summary.leaves {
        let key_len = u16::try_from(leaf.key.len())
            .map_err(|_| Error::BlobStoreIo(io::Error::other("cold index key too long")))?;
        let value_len = match &leaf.value {
            Some(value) => u16::try_from(value.len())
                .map_err(|_| Error::BlobStoreIo(io::Error::other("cold index value too long")))?,
            None => u16::MAX,
        };
        out.extend_from_slice(&key_len.to_le_bytes());
        out.extend_from_slice(&value_len.to_le_bytes());
        out.extend_from_slice(&leaf.seq.to_le_bytes());
        out.extend_from_slice(&leaf.key);
        if let Some(value) = &leaf.value {
            out.extend_from_slice(value);
        }
    }
    for crossing in &summary.crossings {
        let prefix_len = u16::try_from(crossing.prefix.len())
            .map_err(|_| Error::BlobStoreIo(io::Error::other("cold index prefix too long")))?;
        out.extend_from_slice(&prefix_len.to_le_bytes());
        out.extend_from_slice(&crossing.child_guid);
        out.extend_from_slice(&crossing.prefix);
    }
    Ok(out)
}

fn encode_delete(guid: BlobGuid) -> Vec<u8> {
    let mut out = Vec::with_capacity(17);
    out.push(COLD_TY_DELETE);
    out.extend_from_slice(&guid);
    out
}

fn decode_set_header(payload: &[u8]) -> Result<(BlobGuid, u64)> {
    if payload.len() < 33 || payload[0] != COLD_TY_SET {
        return Err(Error::node_corrupt("cold index: corrupt set header"));
    }
    let mut guid = [0u8; 16];
    guid.copy_from_slice(&payload[1..17]);
    let generation = u64::from_le_bytes(payload[17..25].try_into().unwrap());
    Ok((guid, generation))
}

fn decode_delete_payload(payload: &[u8]) -> Result<BlobGuid> {
    if payload.len() != 17 || payload[0] != COLD_TY_DELETE {
        return Err(Error::node_corrupt("cold index: corrupt delete record"));
    }
    let mut guid = [0u8; 16];
    guid.copy_from_slice(&payload[1..17]);
    Ok(guid)
}

fn decode_set_payload(
    payload: &[u8],
    expected_guid: BlobGuid,
    expected_generation: u64,
) -> Result<ColdBlobSummary> {
    let (guid, generation) = decode_set_header(payload)?;
    if guid != expected_guid || generation != expected_generation {
        return Err(Error::node_corrupt("cold index: stale set payload"));
    }
    let mut cursor = 25usize;
    let leaf_count = read_u32(payload, &mut cursor)? as usize;
    let crossing_count = read_u32(payload, &mut cursor)? as usize;
    let mut summary = ColdBlobSummary {
        leaves: Vec::with_capacity(leaf_count),
        crossings: Vec::with_capacity(crossing_count),
    };
    for _ in 0..leaf_count {
        let key_len = read_u16(payload, &mut cursor)? as usize;
        let value_len = read_u16(payload, &mut cursor)?;
        let seq = read_u64(payload, &mut cursor)?;
        let key = read_bytes(payload, &mut cursor, key_len)?.to_vec();
        let value = if value_len == u16::MAX {
            None
        } else {
            Some(read_bytes(payload, &mut cursor, value_len as usize)?.to_vec())
        };
        summary
            .leaves
            .push(crate::engine::ColdLeaf { key, value, seq });
    }
    for _ in 0..crossing_count {
        let prefix_len = read_u16(payload, &mut cursor)? as usize;
        let bytes = read_bytes(payload, &mut cursor, 16)?;
        let mut child_guid = [0u8; 16];
        child_guid.copy_from_slice(bytes);
        let prefix = read_bytes(payload, &mut cursor, prefix_len)?.to_vec();
        summary
            .crossings
            .push(crate::engine::ColdCrossing { prefix, child_guid });
    }
    if cursor != payload.len() {
        return Err(Error::node_corrupt("cold index: trailing bytes"));
    }
    Ok(summary)
}

fn read_u16(input: &[u8], cursor: &mut usize) -> Result<u16> {
    let bytes = read_bytes(input, cursor, 2)?;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u32(input: &[u8], cursor: &mut usize) -> Result<u32> {
    let bytes = read_bytes(input, cursor, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64(input: &[u8], cursor: &mut usize) -> Result<u64> {
    let bytes = read_bytes(input, cursor, 8)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_bytes<'a>(input: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| Error::node_corrupt("cold index: offset overflow"))?;
    if end > input.len() {
        return Err(Error::node_corrupt("cold index: truncated payload"));
    }
    let out = &input[*cursor..end];
    *cursor = end;
    Ok(out)
}
