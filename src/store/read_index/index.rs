use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use crate::api::errors::{Error, Result};
use crate::layout::{
    BlobGuid, BlobHeader, BlobNode, Leaf, Node16, Node256, Node4, Node48, NodeType, Prefix,
    BLOB_MAX_INLINE, DATA_AREA_START, PAGE_SIZE, PREFIX_MAX_INLINE,
};
use crate::store::{decode_child_off, BlobFrameRef};

const MAGIC: [u8; 8] = *b"HOLTCI01";
const VERSION: u16 = 12;
const HEADER_LEN: usize = 8 + 2 + 2 + 4 + ReadIndexStamp::ENCODED_LEN + 4 + 4 + 4 + 4 + 4 + 4;
const BLOOM_BYTES: usize = 2048;
const BLOOM_BITS: usize = BLOOM_BYTES * 8;
const BLOOM_PROBES: usize = 4;
const BUCKET_ENTRY_LEN: usize = 8;
const BUCKET_CRC_LEN: usize = 4;
const INLINE_VALUE_MAX: usize = 256;
const MAX_INDEX_BYTES: usize = PAGE_SIZE as usize;
const MAX_VALUE_BYTES: usize = PAGE_SIZE as usize;
const MAX_COMPONENT_BYTES: usize = 64 * 1024;
const MIN_BUCKETS: usize = 64;
const MAX_BUCKETS: usize = 4096;
const SHARDS: usize = 16;
const VALUE_INLINE: u32 = 0;
const VALUE_SEGMENT: u32 = 1;
const VALUE_BLOB: u32 = 2;
const COMPONENT_DELIMITER_PRIORITY: &[u8] = b"/:|#@\\";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadIndexStamp {
    pub(crate) root_slot: u16,
    pub(crate) num_slots: u16,
    pub(crate) space_used: u32,
    pub(crate) compact_times: u32,
    pub(crate) dead_bytes: u32,
    pub(crate) gap_space: u32,
    pub(crate) tombstone_leaf_cnt: u32,
    pub(crate) created_epoch: u64,
    pub(crate) blob_guid: BlobGuid,
    pub(crate) routing_off: u32,
    pub(crate) routing_len: u32,
    pub(crate) leaf_region_start: u32,
    pub(crate) routing_unfit: u32,
}

impl ReadIndexStamp {
    const ENCODED_LEN: usize = 2 + 2 + (9 * 4) + 8 + 16;

    pub(crate) fn new(header: &BlobHeader) -> Self {
        Self {
            root_slot: header.root_slot,
            num_slots: header.num_slots,
            space_used: header.space_used,
            compact_times: header.compact_times,
            dead_bytes: header.dead_bytes,
            gap_space: header.gap_space,
            tombstone_leaf_cnt: header.tombstone_leaf_cnt,
            created_epoch: header.created_epoch,
            blob_guid: header.blob_guid,
            routing_off: header.routing_off,
            routing_len: header.routing_len,
            leaf_region_start: header.leaf_region_start,
            routing_unfit: header.routing_unfit,
        }
    }

    fn encode(self, out: &mut Vec<u8>) {
        put_u16(out, self.root_slot);
        put_u16(out, self.num_slots);
        put_u32(out, self.space_used);
        put_u32(out, self.compact_times);
        put_u32(out, self.dead_bytes);
        put_u32(out, self.gap_space);
        put_u32(out, self.tombstone_leaf_cnt);
        put_u64(out, self.created_epoch);
        out.extend_from_slice(&self.blob_guid);
        put_u32(out, self.routing_off);
        put_u32(out, self.routing_len);
        put_u32(out, self.leaf_region_start);
        put_u32(out, self.routing_unfit);
    }

    fn decode(input: &mut &[u8]) -> Result<Self> {
        let root_slot = take_u16(input)?;
        let num_slots = take_u16(input)?;
        let space_used = take_u32(input)?;
        let compact_times = take_u32(input)?;
        let dead_bytes = take_u32(input)?;
        let gap_space = take_u32(input)?;
        let tombstone_leaf_cnt = take_u32(input)?;
        let created_epoch = take_u64(input)?;
        let blob_guid = take_guid(input)?;
        let routing_off = take_u32(input)?;
        let routing_len = take_u32(input)?;
        let leaf_region_start = take_u32(input)?;
        let routing_unfit = take_u32(input)?;
        Ok(Self {
            root_slot,
            num_slots,
            space_used,
            compact_times,
            dead_bytes,
            gap_space,
            tombstone_leaf_cnt,
            created_epoch,
            blob_guid,
            routing_off,
            routing_len,
            leaf_region_start,
            routing_unfit,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadIndexAnswer {
    NotFound,
    Crossing {
        child_guid: BlobGuid,
        child_depth: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct BucketEntry {
    off: u32,
    len: u32,
}

#[derive(Debug, Clone)]
struct CrossingEntry {
    prefix: Box<[u8]>,
    child_guid: BlobGuid,
}

#[derive(Debug, Clone)]
struct ComponentEntry {
    delimiter: u8,
    prefix: Box<[u8]>,
}

struct BuildLeaf {
    hash: u64,
    value_source: u32,
    value_source_off: u32,
    value_len: u32,
    value_crc32: u32,
    key: Box<[u8]>,
    value: Option<Box<[u8]>>,
    seq: u64,
}

pub(crate) struct ReadIndexBuild {
    pub(crate) index: Vec<u8>,
    pub(crate) values: Vec<u8>,
}

struct DecodedHeader {
    stamp: ReadIndexStamp,
    bucket_count: usize,
    crossing_count: usize,
    crossing_bytes: usize,
    base_prefix_len: usize,
    component_count: usize,
    component_bytes: usize,
    directory_len: usize,
    total_len: usize,
}

#[derive(Debug)]
pub(crate) struct ReadIndex {
    stamp: ReadIndexStamp,
    bloom: Box<[u8; BLOOM_BYTES]>,
    buckets: Box<[BucketEntry]>,
    base_prefix: Box<[u8]>,
    crossings: Box<[CrossingEntry]>,
    components: Box<[ComponentEntry]>,
    bytes: usize,
}

pub(crate) enum ReadIndexHit {
    Inline {
        value: Vec<u8>,
        seq: u64,
    },
    ValueSegment {
        value_off: u32,
        value_len: u32,
        value_crc32: u32,
        seq: u64,
    },
    BlobOffset {
        value_off: u32,
        value_len: u32,
        seq: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefixLiveness {
    Present,
    Absent,
    Unknown,
}

impl ReadIndex {
    pub(crate) const HEADER_LEN: usize = HEADER_LEN;

    pub(crate) fn build(frame: BlobFrameRef<'_>) -> Result<ReadIndexBuild> {
        let mut builder = IndexBuilder::default();
        let root = decode_child(frame.header().root_slot)?;
        walk(frame, root, &mut Vec::new(), &mut builder)?;
        let values = pack_value_segments(&mut builder.leaves)?;
        let bucket_count = choose_bucket_count(builder.leaves.len());
        builder
            .leaves
            .sort_by_key(|entry| (bucket_idx(entry.hash, bucket_count), entry.hash));
        builder
            .crossings
            .sort_by_key(|entry| std::cmp::Reverse(entry.prefix.len()));

        let bloom = encode_bloom(&builder.leaves);
        let base_prefix = common_leaf_prefix(&builder.leaves);
        let crossing_bytes = encode_crossings(&builder.crossings)?;
        let component_bytes = encode_components(&builder.leaves)?;
        let bucket_blocks = encode_bucket_blocks(&builder.leaves, bucket_count, &base_prefix)?;
        let directory_len = HEADER_LEN
            + BLOOM_BYTES
            + bucket_count * BUCKET_ENTRY_LEN
            + base_prefix.len()
            + crossing_bytes.len()
            + component_bytes.len();
        let total_len = directory_len + bucket_blocks.iter().map(Vec::len).sum::<usize>();
        if total_len > MAX_INDEX_BYTES {
            return Err(Error::node_corrupt("read index total length"));
        }
        let mut out = Vec::with_capacity(total_len);
        out.extend_from_slice(&MAGIC);
        put_u16(&mut out, VERSION);
        put_u16(&mut out, 0);
        put_u32_checked(&mut out, total_len, "read index total length")?;
        ReadIndexStamp::new(frame.header()).encode(&mut out);
        put_u32_checked(&mut out, bucket_count, "read index buckets")?;
        put_u32_checked(&mut out, builder.crossings.len(), "read index crossings")?;
        put_u32_checked(&mut out, crossing_bytes.len(), "read index crossings bytes")?;
        put_u32_checked(&mut out, base_prefix.len(), "read index base prefix")?;
        put_u32_checked(
            &mut out,
            component_count(&component_bytes)?,
            "read index components",
        )?;
        put_u32_checked(
            &mut out,
            component_bytes.len(),
            "read index components bytes",
        )?;
        out.extend_from_slice(&bloom[..]);
        let mut cursor = u32::try_from(directory_len)
            .map_err(|_| Error::node_corrupt("read index directory"))?;
        for block in &bucket_blocks {
            put_u32(&mut out, cursor);
            put_u32_checked(&mut out, block.len(), "read index bucket block")?;
            cursor = cursor
                .checked_add(
                    u32::try_from(block.len())
                        .map_err(|_| Error::node_corrupt("read index bucket block"))?,
                )
                .ok_or(Error::node_corrupt("read index bucket offset"))?;
        }
        out.extend_from_slice(&base_prefix);
        out.extend_from_slice(&crossing_bytes);
        out.extend_from_slice(&component_bytes);
        for block in bucket_blocks {
            out.extend_from_slice(&block);
        }
        Ok(ReadIndexBuild { index: out, values })
    }

    pub(crate) fn directory_len(header: &[u8]) -> Result<usize> {
        let mut input = header;
        decode_header(&mut input).map(|decoded| decoded.directory_len)
    }

    pub(crate) fn decode_directory(bytes: Vec<u8>) -> Result<Self> {
        let encoded_len = bytes.len();
        let mut input = bytes.as_slice();
        let decoded = decode_header(&mut input)?;
        if encoded_len != decoded.directory_len {
            return Err(Error::node_corrupt("read index: directory length"));
        }

        let mut bloom = Box::new([0u8; BLOOM_BYTES]);
        bloom.copy_from_slice(take(&mut input, BLOOM_BYTES)?);

        let mut buckets = Vec::with_capacity(decoded.bucket_count);
        for _ in 0..decoded.bucket_count {
            let off = take_u32(&mut input)?;
            let len = take_u32(&mut input)?;
            if usize::try_from(off)
                .ok()
                .and_then(|start| start.checked_add(len as usize))
                .is_none_or(|end| end > decoded.total_len)
                || (len as usize) < BUCKET_CRC_LEN
            {
                return Err(Error::node_corrupt("read index: bucket range"));
            }
            buckets.push(BucketEntry { off, len });
        }

        let base_prefix = take(&mut input, decoded.base_prefix_len)?
            .to_vec()
            .into_boxed_slice();
        let crossing_bytes = take(&mut input, decoded.crossing_bytes)?;
        let mut crossing_input = crossing_bytes;
        let mut crossings = Vec::with_capacity(decoded.crossing_count);
        for _ in 0..decoded.crossing_count {
            let prefix_len = take_u32(&mut crossing_input)? as usize;
            let child_guid = take_guid(&mut crossing_input)?;
            let prefix = take(&mut crossing_input, prefix_len)?
                .to_vec()
                .into_boxed_slice();
            crossings.push(CrossingEntry { prefix, child_guid });
        }
        if !crossing_input.is_empty() {
            return Err(Error::node_corrupt("read index: crossing bytes"));
        }
        let component_bytes = take(&mut input, decoded.component_bytes)?;
        let mut component_input = component_bytes;
        let mut components = Vec::with_capacity(decoded.component_count);
        for _ in 0..decoded.component_count {
            let delimiter = take_u8(&mut component_input)?;
            let len = take_u32(&mut component_input)? as usize;
            let component = take(&mut component_input, len)?.to_vec().into_boxed_slice();
            components.push(ComponentEntry {
                delimiter,
                prefix: component,
            });
        }
        if !component_input.is_empty() || !input.is_empty() {
            return Err(Error::node_corrupt("read index: directory trailing bytes"));
        }

        let bytes = size_of::<Self>()
            + BLOOM_BYTES
            + buckets.len() * size_of::<BucketEntry>()
            + base_prefix.len()
            + crossings
                .iter()
                .map(|entry| size_of::<CrossingEntry>() + entry.prefix.len())
                .sum::<usize>()
            + components
                .iter()
                .map(|entry| size_of::<ComponentEntry>() + entry.prefix.len())
                .sum::<usize>();
        Ok(Self {
            stamp: decoded.stamp,
            bloom,
            buckets: buckets.into_boxed_slice(),
            base_prefix,
            crossings: crossings.into_boxed_slice(),
            components: components.into_boxed_slice(),
            bytes,
        })
    }

    pub(crate) fn route_or_absent(&self, user_key: &[u8], depth: usize) -> ReadIndexAnswer {
        if let Some((child_guid, child_depth)) = self.crossing(user_key, depth) {
            return ReadIndexAnswer::Crossing {
                child_guid,
                child_depth,
            };
        }

        ReadIndexAnswer::NotFound
    }

    pub(crate) fn crossing(&self, user_key: &[u8], depth: usize) -> Option<(BlobGuid, usize)> {
        self.crossings.iter().find_map(|crossing| {
            key_has_prefix_at_depth(user_key, depth, &crossing.prefix)
                .then_some((crossing.child_guid, depth + crossing.prefix.len()))
        })
    }

    pub(crate) fn may_have_key(&self, user_key: &[u8]) -> bool {
        bloom_may_have(&self.bloom, hash_user_key(user_key))
    }

    pub(crate) fn has_indexed_leaf(&self) -> bool {
        self.buckets
            .iter()
            .any(|bucket| bucket.len as usize > BUCKET_CRC_LEN)
    }

    pub(crate) fn base_prefix(&self) -> &[u8] {
        &self.base_prefix
    }

    pub(crate) fn next_component_rollup(
        &self,
        prefix: &[u8],
        delimiter: u8,
        lower_bound: Option<(&[u8], bool)>,
    ) -> Option<&[u8]> {
        self.components.iter().find_map(|component| {
            if component.delimiter != delimiter {
                return None;
            }
            let component = component.prefix.as_ref();
            if component.len() <= prefix.len() || !component.starts_with(prefix) {
                return None;
            }
            if let Some((bound, inclusive)) = lower_bound {
                let allowed = if inclusive {
                    component >= bound
                } else {
                    component > bound
                };
                if !allowed {
                    return None;
                }
            }
            Some(component)
        })
    }

    pub(crate) fn prefix_liveness(&self, prefix: &[u8]) -> PrefixLiveness {
        if prefix.is_empty() && self.has_indexed_leaf() {
            return PrefixLiveness::Present;
        }

        if self.has_indexed_leaf() {
            let base = user_key_prefix(&self.base_prefix);
            if base.starts_with(prefix) {
                return PrefixLiveness::Present;
            }
            if !prefix.starts_with(base) {
                return if self.crossings.is_empty() {
                    PrefixLiveness::Absent
                } else {
                    PrefixLiveness::Unknown
                };
            }
            if component_prefix_present(&self.components, prefix) {
                return PrefixLiveness::Present;
            }
            if component_prefix_absent(&self.components, prefix) {
                return if self.crossings.is_empty() {
                    PrefixLiveness::Absent
                } else {
                    PrefixLiveness::Unknown
                };
            }
            return PrefixLiveness::Unknown;
        }

        if self.crossings.is_empty() {
            PrefixLiveness::Absent
        } else {
            PrefixLiveness::Unknown
        }
    }

    pub(crate) fn stamp(&self) -> ReadIndexStamp {
        self.stamp
    }

    pub(crate) fn bucket_range(&self, user_key: &[u8]) -> Option<(u32, u32)> {
        let key_hash = hash_user_key(user_key);
        let entry = self.buckets.get(bucket_idx(key_hash, self.buckets.len()))?;
        Some((entry.off, entry.len))
    }

    pub(crate) fn lookup_leaf_in_bucket(
        &self,
        user_key: &[u8],
        block: &[u8],
    ) -> Result<Option<ReadIndexHit>> {
        if block.len() < BUCKET_CRC_LEN {
            return Err(Error::node_corrupt("read index: bucket block"));
        }
        let body_len = block.len() - BUCKET_CRC_LEN;
        let expected_crc32 = u32::from_le_bytes(block[body_len..].try_into().unwrap());
        let body = &block[..body_len];
        if crc32fast::hash(body) != expected_crc32 {
            return Err(Error::node_corrupt("read index: bucket crc"));
        }
        let key_hash = hash_user_key(user_key);
        let mut input = body;
        while !input.is_empty() {
            let hash = take_u64(&mut input)?;
            let value_source = take_u32(&mut input)?;
            let value_source_off = take_u32(&mut input)?;
            let full_value_len = take_u32(&mut input)?;
            let value_crc32 = take_u32(&mut input)?;
            let suffix_len = take_u32(&mut input)?;
            let inline_value_len = take_u32(&mut input)?;
            let seq = take_u64(&mut input)?;
            let suffix = take(&mut input, suffix_len as usize)?;
            let value = (inline_value_len != 0)
                .then(|| take(&mut input, inline_value_len as usize))
                .transpose()?;
            if hash != key_hash {
                continue;
            }
            if !exact_key_matches_base_and_suffix(user_key, &self.base_prefix, suffix) {
                continue;
            }
            return Ok(Some(match value {
                Some(value) => ReadIndexHit::Inline {
                    value: value.to_vec(),
                    seq,
                },
                None => match value_source {
                    VALUE_SEGMENT => {
                        if !valid_value_segment_range(value_source_off, full_value_len) {
                            return Err(Error::node_corrupt("read index: value segment range"));
                        }
                        ReadIndexHit::ValueSegment {
                            value_off: value_source_off,
                            value_len: full_value_len,
                            value_crc32,
                            seq,
                        }
                    }
                    VALUE_BLOB => {
                        if !valid_blob_value_range(value_source_off, full_value_len) {
                            return Err(Error::node_corrupt("read index: blob value range"));
                        }
                        ReadIndexHit::BlobOffset {
                            value_off: value_source_off,
                            value_len: full_value_len,
                            seq,
                        }
                    }
                    _ => return Err(Error::node_corrupt("read index: value source")),
                },
            }));
        }
        Ok(None)
    }

    fn memory_bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Default)]
struct IndexBuilder {
    leaves: Vec<BuildLeaf>,
    crossings: Vec<CrossingEntry>,
}

pub(crate) struct ReadIndexCache {
    shards: Box<[Mutex<IndexShard>]>,
    shard_budget_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ReadIndexCacheStats {
    pub(crate) entries: usize,
    pub(crate) bytes: usize,
    pub(crate) budget_bytes: usize,
}

#[derive(Default)]
struct IndexShard {
    entries: HashMap<BlobGuid, CacheEntry>,
    bytes: usize,
    clock: u64,
}

struct CacheEntry {
    index: Arc<ReadIndex>,
    bytes: usize,
    tick: u64,
}

impl ReadIndexCache {
    pub(crate) fn new(budget_bytes: usize) -> Self {
        Self {
            shards: (0..SHARDS)
                .map(|_| Mutex::new(IndexShard::default()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            shard_budget_bytes: budget_bytes / SHARDS,
        }
    }

    pub(crate) fn get(&self, guid: BlobGuid) -> Option<Arc<ReadIndex>> {
        if self.shard_budget_bytes == 0 {
            return None;
        }
        let mut shard = self.shard(guid).lock().unwrap();
        shard.clock = shard.clock.wrapping_add(1);
        let tick = shard.clock;
        let entry = shard.entries.get_mut(&guid)?;
        entry.tick = tick;
        Some(Arc::clone(&entry.index))
    }

    pub(crate) fn insert(&self, guid: BlobGuid, index: ReadIndex) -> Arc<ReadIndex> {
        let bytes = index.memory_bytes();
        let index = Arc::new(index);
        if self.shard_budget_bytes == 0 {
            return index;
        }
        if bytes > self.shard_budget_bytes {
            self.invalidate(guid);
            return index;
        }
        let mut shard = self.shard(guid).lock().unwrap();
        shard.clock = shard.clock.wrapping_add(1);
        let tick = shard.clock;
        if let Some(old) = shard.entries.remove(&guid) {
            shard.bytes = shard.bytes.saturating_sub(old.bytes);
        }
        shard.entries.insert(
            guid,
            CacheEntry {
                index: Arc::clone(&index),
                bytes,
                tick,
            },
        );
        shard.bytes += bytes;
        self.evict_if_needed(&mut shard);
        index
    }

    pub(crate) fn invalidate(&self, guid: BlobGuid) {
        let mut shard = self.shard(guid).lock().unwrap();
        if let Some(old) = shard.entries.remove(&guid) {
            shard.bytes = shard.bytes.saturating_sub(old.bytes);
        }
    }

    pub(crate) fn snapshot(&self) -> ReadIndexCacheStats {
        let mut out = ReadIndexCacheStats {
            budget_bytes: self.shard_budget_bytes.saturating_mul(SHARDS),
            ..ReadIndexCacheStats::default()
        };
        for shard in &self.shards {
            let shard = shard.lock().unwrap();
            out.entries += shard.entries.len();
            out.bytes += shard.bytes;
        }
        out
    }

    fn shard(&self, guid: BlobGuid) -> &Mutex<IndexShard> {
        &self.shards[guid_shard(guid)]
    }

    fn evict_if_needed(&self, shard: &mut IndexShard) {
        while shard.bytes > self.shard_budget_bytes {
            let Some((&victim, _)) = shard.entries.iter().min_by_key(|(_, entry)| entry.tick)
            else {
                break;
            };
            let Some(old) = shard.entries.remove(&victim) else {
                break;
            };
            shard.bytes = shard.bytes.saturating_sub(old.bytes);
        }
    }
}

fn walk(
    frame: BlobFrameRef<'_>,
    off: u32,
    path: &mut Vec<u8>,
    builder: &mut IndexBuilder,
) -> Result<()> {
    let body = frame
        .body_at_offset(off)
        .ok_or(Error::node_corrupt("read index: body"))?;
    let ntype = frame
        .ntype_at(off)
        .ok_or(Error::node_corrupt("read index: ntype"))?;
    match ntype {
        NodeType::Leaf => {
            let leaf = *cast::<Leaf>(&body[..size_of::<Leaf>()]);
            if leaf.tombstone == 0 {
                index_leaf(body, off, leaf, builder)?;
            }
        }
        NodeType::Prefix => {
            let prefix = *cast::<Prefix>(body);
            let len = usize::from(prefix.prefix_len);
            if len > PREFIX_MAX_INLINE {
                return Err(Error::node_corrupt("read index: prefix len"));
            }
            append_path(path, &prefix.bytes[..len], |path| {
                walk(frame, decode_prefix_child(prefix.child)?, path, builder)
            })?;
        }
        NodeType::Blob => {
            let blob = *cast::<BlobNode>(body);
            let len = usize::from(blob.prefix_len);
            if len > BLOB_MAX_INLINE {
                return Err(Error::node_corrupt("read index: blob prefix len"));
            }
            append_path(path, &blob.bytes[..len], |path| {
                builder.crossings.push(CrossingEntry {
                    prefix: path.clone().into_boxed_slice(),
                    child_guid: blob.child_blob_guid,
                });
                Ok(())
            })?;
        }
        NodeType::Node4 => {
            let node = *cast::<Node4>(body);
            for idx in 0..usize::from(node.count).min(4) {
                append_path(path, &[node.keys[idx]], |path| {
                    walk(frame, decode_child(node.children[idx])?, path, builder)
                })?;
            }
        }
        NodeType::Node16 => {
            let node = *cast::<Node16>(body);
            for idx in 0..usize::from(node.count).min(16) {
                append_path(path, &[node.keys[idx]], |path| {
                    walk(frame, decode_child(node.children[idx])?, path, builder)
                })?;
            }
        }
        NodeType::Node48 => {
            let node = *cast::<Node48>(body);
            for byte in 0..=u8::MAX {
                let child_idx = node.index[usize::from(byte)];
                if child_idx != 0 {
                    let idx = usize::from(child_idx - 1);
                    if idx >= 48 {
                        return Err(Error::node_corrupt("read index: node48 child index"));
                    }
                    append_path(path, &[byte], |path| {
                        walk(frame, decode_child(node.children[idx])?, path, builder)
                    })?;
                }
            }
        }
        NodeType::Node256 => {
            let node = *cast::<Node256>(body);
            for byte in 0..=u8::MAX {
                let child = node.children[usize::from(byte)];
                if child != 0 {
                    append_path(path, &[byte], |path| {
                        walk(frame, decode_child(child)?, path, builder)
                    })?;
                }
            }
        }
        NodeType::EmptyRoot => {}
        NodeType::Invalid => return Err(Error::node_corrupt("read index: invalid node")),
    }
    Ok(())
}

fn append_path(
    path: &mut Vec<u8>,
    bytes: &[u8],
    f: impl FnOnce(&mut Vec<u8>) -> Result<()>,
) -> Result<()> {
    let old_len = path.len();
    path.extend_from_slice(bytes);
    let result = f(path);
    path.truncate(old_len);
    result
}

fn decode_header(input: &mut &[u8]) -> Result<DecodedHeader> {
    if take(input, MAGIC.len())? != MAGIC {
        return Err(Error::node_corrupt("read index: magic"));
    }
    let version = take_u16(input)?;
    if version != VERSION {
        return Err(Error::node_corrupt("read index: version"));
    }
    let _flags = take_u16(input)?;
    let total_len = take_u32(input)? as usize;
    let stamp = ReadIndexStamp::decode(input)?;
    let bucket_count = take_u32(input)? as usize;
    let crossing_count = take_u32(input)? as usize;
    let crossing_bytes = take_u32(input)? as usize;
    let base_prefix_len = take_u32(input)? as usize;
    let component_count = take_u32(input)? as usize;
    let component_bytes = take_u32(input)? as usize;
    if total_len > MAX_INDEX_BYTES {
        return Err(Error::node_corrupt("read index: total length"));
    }
    if !(MIN_BUCKETS..=MAX_BUCKETS).contains(&bucket_count) || !bucket_count.is_power_of_two() {
        return Err(Error::node_corrupt("read index: bucket count"));
    }
    let directory_len = HEADER_LEN
        .checked_add(BLOOM_BYTES)
        .and_then(|len| len.checked_add(bucket_count.saturating_mul(BUCKET_ENTRY_LEN)))
        .and_then(|len| len.checked_add(base_prefix_len))
        .and_then(|len| len.checked_add(crossing_bytes))
        .and_then(|len| len.checked_add(component_bytes))
        .ok_or(Error::node_corrupt("read index: directory length"))?;
    if total_len < directory_len {
        return Err(Error::node_corrupt("read index: total length"));
    }
    Ok(DecodedHeader {
        stamp,
        bucket_count,
        crossing_count,
        crossing_bytes,
        base_prefix_len,
        component_count,
        component_bytes,
        directory_len,
        total_len,
    })
}

fn common_leaf_prefix(leaves: &[BuildLeaf]) -> Box<[u8]> {
    let Some(first) = leaves.first() else {
        return Box::new([]);
    };
    let mut prefix_len = first.key.len();
    for leaf in &leaves[1..] {
        let common = first
            .key
            .iter()
            .take(prefix_len)
            .zip(leaf.key.iter())
            .take_while(|(a, b)| a == b)
            .count();
        prefix_len = prefix_len.min(common);
        if prefix_len == 0 {
            break;
        }
    }
    first.key[..prefix_len].to_vec().into_boxed_slice()
}

fn choose_bucket_count(leaf_count: usize) -> usize {
    let target = (leaf_count / 4).next_power_of_two();
    target.clamp(MIN_BUCKETS, MAX_BUCKETS)
}

fn bucket_idx(hash: u64, bucket_count: usize) -> usize {
    debug_assert!(bucket_count.is_power_of_two());
    hash as usize & (bucket_count - 1)
}

fn encode_crossings(crossings: &[CrossingEntry]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for entry in crossings {
        put_u32_checked(&mut out, entry.prefix.len(), "read index prefix")?;
        out.extend_from_slice(&entry.child_guid);
        out.extend_from_slice(&entry.prefix);
    }
    Ok(out)
}

fn encode_components(leaves: &[BuildLeaf]) -> Result<Vec<u8>> {
    let mut grouped = HashMap::<u8, BTreeSet<Vec<u8>>>::new();
    for leaf in leaves {
        let key = leaf.key.strip_suffix(&[0]).unwrap_or(&leaf.key);
        for (idx, &byte) in key.iter().enumerate() {
            let end = idx + 1;
            if end < key.len() && is_component_delimiter(byte) {
                grouped.entry(byte).or_default().insert(key[..end].to_vec());
            }
        }
    }

    let mut out = Vec::new();
    let mut delimiters: Vec<u8> = grouped.keys().copied().collect();
    delimiters.sort_by_key(|&delimiter| component_delimiter_rank(delimiter));
    for delimiter in delimiters {
        let Some(components) = grouped.remove(&delimiter) else {
            continue;
        };
        let bytes = components
            .iter()
            .try_fold(0usize, |acc, component| {
                acc.checked_add(1)?
                    .checked_add(4)?
                    .checked_add(component.len())
            })
            .unwrap_or(MAX_COMPONENT_BYTES + 1);
        if out.len().saturating_add(bytes) > MAX_COMPONENT_BYTES {
            continue;
        }
        for component in components {
            out.push(delimiter);
            put_u32_checked(&mut out, component.len(), "read index component")?;
            out.extend_from_slice(&component);
        }
    }
    Ok(out)
}

fn component_count(bytes: &[u8]) -> Result<usize> {
    let mut input = bytes;
    let mut count = 0usize;
    while !input.is_empty() {
        let _delimiter = take_u8(&mut input)?;
        let len = take_u32(&mut input)? as usize;
        let _ = take(&mut input, len)?;
        count += 1;
    }
    Ok(count)
}

fn is_component_delimiter(byte: u8) -> bool {
    COMPONENT_DELIMITER_PRIORITY.contains(&byte)
}

fn component_delimiter_rank(byte: u8) -> usize {
    COMPONENT_DELIMITER_PRIORITY
        .iter()
        .position(|&candidate| candidate == byte)
        .unwrap_or(COMPONENT_DELIMITER_PRIORITY.len())
}

fn user_key_prefix(key: &[u8]) -> &[u8] {
    key.strip_suffix(&[0]).unwrap_or(key)
}

fn component_prefix_present(components: &[ComponentEntry], prefix: &[u8]) -> bool {
    components
        .iter()
        .any(|component| component.prefix.starts_with(prefix))
}

fn component_prefix_absent(components: &[ComponentEntry], prefix: &[u8]) -> bool {
    let Some(&delimiter) = prefix.last() else {
        return false;
    };
    if !is_component_delimiter(delimiter) {
        return false;
    }
    let mut saw_group = false;
    for component in components {
        if component.delimiter != delimiter {
            continue;
        }
        saw_group = true;
        if component.prefix.starts_with(prefix) {
            return false;
        }
    }
    saw_group
}

fn encode_bloom(leaves: &[BuildLeaf]) -> Box<[u8; BLOOM_BYTES]> {
    let mut bloom = Box::new([0u8; BLOOM_BYTES]);
    for leaf in leaves {
        bloom_insert(&mut bloom, leaf.hash);
    }
    bloom
}

fn bloom_insert(bloom: &mut [u8; BLOOM_BYTES], hash: u64) {
    for bit in bloom_bits(hash) {
        bloom[bit / 8] |= 1 << (bit % 8);
    }
}

fn bloom_may_have(bloom: &[u8; BLOOM_BYTES], hash: u64) -> bool {
    bloom_bits(hash)
        .into_iter()
        .all(|bit| (bloom[bit / 8] & (1 << (bit % 8))) != 0)
}

fn bloom_bits(hash: u64) -> [usize; BLOOM_PROBES] {
    let h1 = hash;
    let h2 = hash.rotate_left(31) | 1;
    let mut bits = [0; BLOOM_PROBES];
    for (i, bit) in bits.iter_mut().enumerate() {
        *bit = h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize & (BLOOM_BITS - 1);
    }
    bits
}

fn encode_bucket_blocks(
    leaves: &[BuildLeaf],
    bucket_count: usize,
    base_prefix: &[u8],
) -> Result<Vec<Vec<u8>>> {
    let mut blocks = (0..bucket_count).map(|_| Vec::new()).collect::<Vec<_>>();
    for leaf in leaves {
        let block = &mut blocks[bucket_idx(leaf.hash, bucket_count)];
        if !leaf.key.starts_with(base_prefix) {
            return Err(Error::node_corrupt("read index base prefix"));
        }
        let suffix = &leaf.key[base_prefix.len()..];
        put_u64(block, leaf.hash);
        put_u32(block, leaf.value_source);
        put_u32(block, leaf.value_source_off);
        put_u32(block, leaf.value_len);
        put_u32(block, leaf.value_crc32);
        put_u32_checked(block, suffix.len(), "read index key suffix")?;
        match &leaf.value {
            Some(value) => put_u32_checked(block, value.len(), "read index value")?,
            None => put_u32(block, 0),
        }
        put_u64(block, leaf.seq);
        block.extend_from_slice(suffix);
        if let Some(value) = &leaf.value {
            block.extend_from_slice(value);
        }
    }
    for block in &mut blocks {
        let crc = crc32fast::hash(block);
        put_u32(block, crc);
    }
    Ok(blocks)
}

fn pack_value_segments(leaves: &mut [BuildLeaf]) -> Result<Vec<u8>> {
    let mut values = Vec::new();
    for leaf in leaves {
        let Some(value) = leaf.value.as_ref() else {
            continue;
        };
        if value.len() <= INLINE_VALUE_MAX {
            leaf.value_source = VALUE_INLINE;
            leaf.value_source_off = 0;
            continue;
        }
        let Some(end) = values.len().checked_add(value.len()) else {
            leaf.value_source = VALUE_BLOB;
            continue;
        };
        if end > MAX_VALUE_BYTES {
            leaf.value_source = VALUE_BLOB;
            leaf.value = None;
            continue;
        }
        leaf.value_source = VALUE_SEGMENT;
        leaf.value_source_off =
            u32::try_from(values.len()).map_err(|_| Error::node_corrupt("value segment offset"))?;
        values.extend_from_slice(value);
        leaf.value = None;
    }
    Ok(values)
}

fn index_leaf(body: &[u8], off: u32, leaf: Leaf, builder: &mut IndexBuilder) -> Result<()> {
    let key_start = size_of::<Leaf>();
    let key_end = key_start + usize::from(leaf.key_len);
    let value_end = key_end + usize::from(leaf.value_len);
    if key_end > body.len() {
        return Err(Error::node_corrupt("read index: leaf key range"));
    }
    if value_end > body.len() {
        return Err(Error::node_corrupt("read index: leaf value range"));
    }
    let key = &body[key_start..key_end];
    let value = &body[key_end..value_end];
    let value_off = off
        .checked_add(u32::try_from(key_end).map_err(|_| Error::node_corrupt("read index value"))?)
        .ok_or(Error::node_corrupt("read index value offset"))?;
    let value_len = u32::from(leaf.value_len);
    let value_crc32 = crc32fast::hash(value);
    let value = Some(value.to_vec().into_boxed_slice());
    builder.leaves.push(BuildLeaf {
        hash: hash_exact_key(key),
        value_source: VALUE_BLOB,
        value_source_off: value_off,
        value_len,
        value_crc32,
        key: key.to_vec().into_boxed_slice(),
        value,
        seq: leaf.seq,
    });
    Ok(())
}

fn exact_key_matches_base_and_suffix(user_key: &[u8], base: &[u8], suffix: &[u8]) -> bool {
    let exact_len = user_key.len() + 1;
    if base.len() + suffix.len() != exact_len {
        return false;
    }
    exact_key_range_matches(user_key, 0, base)
        && exact_key_range_matches(user_key, base.len(), suffix)
}

fn exact_key_range_matches(user_key: &[u8], start: usize, want: &[u8]) -> bool {
    if start + want.len() > user_key.len() + 1 {
        return false;
    }
    want.iter().enumerate().all(|(idx, &byte)| {
        let pos = start + idx;
        let got = if pos < user_key.len() {
            user_key[pos]
        } else {
            0
        };
        got == byte
    })
}

fn decode_child(encoded: u16) -> Result<u32> {
    if encoded == 0 {
        return Err(Error::node_corrupt("read index: null child"));
    }
    Ok(decode_child_off(encoded))
}

fn decode_prefix_child(encoded: u32) -> Result<u32> {
    let encoded = u16::try_from(encoded).map_err(|_| Error::node_corrupt("read index: child"))?;
    decode_child(encoded)
}

fn key_has_prefix_at_depth(user_key: &[u8], depth: usize, prefix: &[u8]) -> bool {
    if depth + prefix.len() > user_key.len() + 1 {
        return false;
    }
    for (idx, &want) in prefix.iter().enumerate() {
        let pos = depth + idx;
        let got = match pos.cmp(&user_key.len()) {
            std::cmp::Ordering::Less => user_key[pos],
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => return false,
        };
        if got != want {
            return false;
        }
    }
    true
}

fn hash_user_key(user_key: &[u8]) -> u64 {
    let hash = fnv64(user_key);
    // The ART terminator byte is 0, so FNV-1a only needs the
    // multiply step for the virtual trailing byte.
    hash.wrapping_mul(0x0000_0100_0000_01b3)
}

fn hash_exact_key(exact_key: &[u8]) -> u64 {
    fnv64(exact_key)
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn valid_blob_value_range(value_off: u32, value_len: u32) -> bool {
    (DATA_AREA_START..=PAGE_SIZE).contains(&value_off)
        && value_off
            .checked_add(value_len)
            .is_some_and(|end| end <= PAGE_SIZE)
}

fn valid_value_segment_range(value_off: u32, value_len: u32) -> bool {
    value_off
        .checked_add(value_len)
        .is_some_and(|end| end <= PAGE_SIZE)
}

fn guid_shard(guid: BlobGuid) -> usize {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    guid.hash(&mut h);
    (h.finish() as usize) & (SHARDS - 1)
}

fn cast<T>(body: &[u8]) -> &T {
    debug_assert_eq!(body.len(), size_of::<T>());
    debug_assert_eq!(body.as_ptr() as usize % std::mem::align_of::<T>(), 0);
    unsafe { &*body.as_ptr().cast::<T>() }
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32_checked(out: &mut Vec<u8>, value: usize, what: &'static str) -> Result<()> {
    let value = u32::try_from(value).map_err(|_| Error::node_corrupt(what))?;
    put_u32(out, value);
    Ok(())
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn take<'a>(input: &mut &'a [u8], len: usize) -> Result<&'a [u8]> {
    if input.len() < len {
        return Err(Error::node_corrupt("read index: truncated"));
    }
    let (head, tail) = input.split_at(len);
    *input = tail;
    Ok(head)
}

fn take_u8(input: &mut &[u8]) -> Result<u8> {
    Ok(take(input, 1)?[0])
}

fn take_u16(input: &mut &[u8]) -> Result<u16> {
    Ok(u16::from_le_bytes(take(input, 2)?.try_into().unwrap()))
}

fn take_u32(input: &mut &[u8]) -> Result<u32> {
    Ok(u32::from_le_bytes(take(input, 4)?.try_into().unwrap()))
}

fn take_u64(input: &mut &[u8]) -> Result<u64> {
    Ok(u64::from_le_bytes(take(input, 8)?.try_into().unwrap()))
}

fn take_guid(input: &mut &[u8]) -> Result<BlobGuid> {
    let mut guid = [0u8; 16];
    guid.copy_from_slice(take(input, 16)?);
    Ok(guid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{leaf_body_size, BlobNode, Leaf};
    use crate::store::{encode_child_off, BlobFrame};

    fn install_leaf(
        frame: &mut BlobFrame<'_>,
        key: &[u8],
        value: &[u8],
        seq: u64,
        tombstone: bool,
    ) -> u32 {
        let total = leaf_body_size(key.len() as u32, value.len() as u32);
        let slot = frame.alloc_leaf(total).unwrap().slot;
        let off = frame.offset_of_slot(slot).unwrap();
        let body = frame.bytes_at_mut(off, total).unwrap();
        let mut header = Leaf::live(key.len() as u16, value.len() as u16, seq, 0);
        header.tombstone = u8::from(tombstone);
        body[..size_of::<Leaf>()].copy_from_slice(as_bytes(&header));
        body[size_of::<Leaf>()..size_of::<Leaf>() + key.len()].copy_from_slice(key);
        let value_start = size_of::<Leaf>() + key.len();
        body[value_start..value_start + value.len()].copy_from_slice(value);
        frame.header_mut().root_slot = encode_child_off(off);
        off
    }

    fn install_blob_node(frame: &mut BlobFrame<'_>, prefix: &[u8], child: BlobGuid) {
        let slot = frame.alloc_node(NodeType::Blob).unwrap().slot;
        let off = frame.offset_of_slot(slot).unwrap();
        let body = frame
            .bytes_at_mut(off, size_of::<BlobNode>() as u32)
            .unwrap();
        let node = BlobNode::new(prefix, child);
        body.copy_from_slice(as_bytes(&node));
        frame.header_mut().root_slot = encode_child_off(off);
    }

    fn as_bytes<T>(value: &T) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(value).cast::<u8>(), size_of::<T>())
        }
    }

    fn decode_dir(bytes: &[u8]) -> ReadIndex {
        let dir_len = ReadIndex::directory_len(&bytes[..ReadIndex::HEADER_LEN]).unwrap();
        ReadIndex::decode_directory(bytes[..dir_len].to_vec()).unwrap()
    }

    fn bucket<'a>(bytes: &'a [u8], index: &ReadIndex, key: &[u8]) -> &'a [u8] {
        let (off, len) = index.bucket_range(key).unwrap();
        &bytes[off as usize..off as usize + len as usize]
    }

    #[test]
    fn key_prefix_matches_virtual_terminator() {
        assert!(key_has_prefix_at_depth(b"abc", 0, b"abc\0"));
        assert!(key_has_prefix_at_depth(b"abc", 1, b"bc\0"));
        assert!(!key_has_prefix_at_depth(b"abc", 0, b"abcd"));
    }

    #[test]
    fn encode_decode_empty_index() {
        let guid = [0x11; 16];
        let mut buf = crate::store::blob_store::AlignedBlobBuf::zeroed();
        BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
        let bytes = ReadIndex::build(BlobFrameRef::wrap(buf.as_slice()))
            .unwrap()
            .index;
        let index = decode_dir(&bytes);
        assert_eq!(index.stamp.blob_guid, guid);
        assert!(matches!(
            index.route_or_absent(b"missing", 0),
            ReadIndexAnswer::NotFound
        ));
    }

    #[test]
    fn rejects_oversized_index_header() {
        let guid = [0x15; 16];
        let mut buf = crate::store::blob_store::AlignedBlobBuf::zeroed();
        BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
        let mut bytes = ReadIndex::build(BlobFrameRef::wrap(buf.as_slice()))
            .unwrap()
            .index;
        bytes[12..16].copy_from_slice(&(PAGE_SIZE + 1).to_le_bytes());

        assert!(ReadIndex::directory_len(&bytes[..ReadIndex::HEADER_LEN]).is_err());
    }

    #[test]
    fn indexes_live_leaves_and_tombstones() {
        let guid = [0x12; 16];
        let mut buf = crate::store::blob_store::AlignedBlobBuf::zeroed();
        {
            let mut frame = BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
            install_leaf(&mut frame, b"alpha\0", b"one", 1, false);
        }
        let bytes = ReadIndex::build(BlobFrameRef::wrap(buf.as_slice()))
            .unwrap()
            .index;
        let index = decode_dir(&bytes);
        assert!(index.may_have_key(b"alpha"));
        assert!(!index.may_have_key(b"alpha~"));
        assert_eq!(index.base_prefix.as_ref(), b"alpha\0");
        match index.lookup_leaf_in_bucket(b"alpha", bucket(&bytes, &index, b"alpha")) {
            Ok(Some(ReadIndexHit::Inline { value, seq })) => {
                assert_eq!(value, b"one");
                assert_eq!(seq, 1);
            }
            _ => panic!("live small leaf must be indexed inline"),
        }

        let mut tomb = crate::store::blob_store::AlignedBlobBuf::zeroed();
        {
            let mut frame = BlobFrame::init(tomb.as_mut_slice(), guid).unwrap();
            install_leaf(&mut frame, b"beta\0", b"two", 2, true);
        }
        let bytes = ReadIndex::build(BlobFrameRef::wrap(tomb.as_slice()))
            .unwrap()
            .index;
        let index = decode_dir(&bytes);
        assert!(matches!(
            index.route_or_absent(b"beta", 0),
            ReadIndexAnswer::NotFound
        ));
    }

    #[test]
    fn large_value_bucket_stores_value_segment_offset() {
        let guid = [0x1b; 16];
        let mut buf = crate::store::blob_store::AlignedBlobBuf::zeroed();
        let key = b"large-object\0";
        let value = vec![0x5a; INLINE_VALUE_MAX + 257];
        {
            let mut frame = BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
            install_leaf(&mut frame, key, &value, 42, false);
        }

        let build = ReadIndex::build(BlobFrameRef::wrap(buf.as_slice())).unwrap();
        assert_eq!(build.values, value);
        let bytes = build.index;
        let index = decode_dir(&bytes);
        match index.lookup_leaf_in_bucket(b"large-object", bucket(&bytes, &index, b"large-object"))
        {
            Ok(Some(ReadIndexHit::ValueSegment {
                value_off,
                value_len,
                value_crc32,
                seq,
            })) => {
                assert_eq!(value_off, 0);
                assert_eq!(value_len, value.len() as u32);
                assert_eq!(value_crc32, crc32fast::hash(&value));
                assert_eq!(seq, 42);
            }
            _ => panic!("large value should be indexed by value segment offset"),
        }
    }

    #[test]
    fn rejects_corrupt_bucket_block() {
        let guid = [0x1a; 16];
        let mut buf = crate::store::blob_store::AlignedBlobBuf::zeroed();
        {
            let mut frame = BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
            install_leaf(&mut frame, b"alpha\0", b"one", 1, false);
        }
        let bytes = ReadIndex::build(BlobFrameRef::wrap(buf.as_slice()))
            .unwrap()
            .index;
        let index = decode_dir(&bytes);
        let block = bucket(&bytes, &index, b"alpha");
        let mut corrupt = block.to_vec();
        *corrupt.last_mut().unwrap() ^= 0xff;

        assert!(index.lookup_leaf_in_bucket(b"alpha", &corrupt).is_err());
    }

    #[test]
    fn bucket_entries_store_suffix_after_common_prefix() {
        let leaves = vec![
            BuildLeaf {
                hash: hash_exact_key(b"bucket/a/file-0001\0"),
                value_source: VALUE_INLINE,
                value_source_off: 0,
                value_len: 3,
                value_crc32: crc32fast::hash(b"one"),
                key: b"bucket/a/file-0001\0".to_vec().into_boxed_slice(),
                value: Some(b"one".to_vec().into_boxed_slice()),
                seq: 7,
            },
            BuildLeaf {
                hash: hash_exact_key(b"bucket/a/file-0002\0"),
                value_source: VALUE_INLINE,
                value_source_off: 0,
                value_len: 3,
                value_crc32: crc32fast::hash(b"two"),
                key: b"bucket/a/file-0002\0".to_vec().into_boxed_slice(),
                value: Some(b"two".to_vec().into_boxed_slice()),
                seq: 8,
            },
        ];
        let base = common_leaf_prefix(&leaves);
        assert_eq!(base.as_ref(), b"bucket/a/file-000");

        let blocks = encode_bucket_blocks(&leaves, MIN_BUCKETS, &base).unwrap();
        let mut buckets = vec![BucketEntry { off: 0, len: 0 }; MIN_BUCKETS];
        let mut encoded = Vec::new();
        for (idx, block) in blocks.iter().enumerate() {
            buckets[idx] = BucketEntry {
                off: encoded.len() as u32,
                len: block.len() as u32,
            };
            encoded.extend_from_slice(block);
        }
        let index = ReadIndex {
            stamp: ReadIndexStamp {
                root_slot: 0,
                num_slots: 0,
                space_used: 0,
                compact_times: 0,
                dead_bytes: 0,
                gap_space: 0,
                tombstone_leaf_cnt: 0,
                created_epoch: 0,
                blob_guid: [0; 16],
                routing_off: 0,
                routing_len: 0,
                leaf_region_start: 0,
                routing_unfit: 0,
            },
            bloom: Box::new([0; BLOOM_BYTES]),
            buckets: buckets.into_boxed_slice(),
            base_prefix: base,
            crossings: Box::new([]),
            components: Box::new([]),
            bytes: 0,
        };

        let (off, len) = index.bucket_range(b"bucket/a/file-0001").unwrap();
        let block = &encoded[off as usize..off as usize + len as usize];
        match index.lookup_leaf_in_bucket(b"bucket/a/file-0001", block) {
            Ok(Some(ReadIndexHit::Inline { value, seq })) => {
                assert_eq!(value, b"one");
                assert_eq!(seq, 7);
            }
            _ => panic!("expected exact suffix hit"),
        }

        let (off, len) = index.bucket_range(b"bucket/a/file-0003").unwrap();
        let block = &encoded[off as usize..off as usize + len as usize];
        assert!(
            matches!(
                index.lookup_leaf_in_bucket(b"bucket/a/file-0003", block),
                Ok(None)
            ),
            "suffix comparison must reject a neighboring absent key"
        );
    }

    #[test]
    fn component_summary_returns_next_path_rollup() {
        let leaves = vec![
            BuildLeaf {
                hash: hash_exact_key(b"bucket/a/file-0001\0"),
                value_source: VALUE_INLINE,
                value_source_off: 0,
                value_len: 3,
                value_crc32: crc32fast::hash(b"one"),
                key: b"bucket/a/file-0001\0".to_vec().into_boxed_slice(),
                value: Some(b"one".to_vec().into_boxed_slice()),
                seq: 7,
            },
            BuildLeaf {
                hash: hash_exact_key(b"bucket/b/file-0001\0"),
                value_source: VALUE_INLINE,
                value_source_off: 0,
                value_len: 3,
                value_crc32: crc32fast::hash(b"two"),
                key: b"bucket/b/file-0001\0".to_vec().into_boxed_slice(),
                value: Some(b"two".to_vec().into_boxed_slice()),
                seq: 8,
            },
            BuildLeaf {
                hash: hash_exact_key(b"bucket/file-0001\0"),
                value_source: VALUE_INLINE,
                value_source_off: 0,
                value_len: 5,
                value_crc32: crc32fast::hash(b"plain"),
                key: b"bucket/file-0001\0".to_vec().into_boxed_slice(),
                value: Some(b"plain".to_vec().into_boxed_slice()),
                seq: 9,
            },
        ];
        let components = encode_components(&leaves).unwrap();
        assert_eq!(component_count(&components).unwrap(), 3);

        let index = ReadIndex {
            stamp: ReadIndexStamp {
                root_slot: 0,
                num_slots: 0,
                space_used: 0,
                compact_times: 0,
                dead_bytes: 0,
                gap_space: 0,
                tombstone_leaf_cnt: 0,
                created_epoch: 0,
                blob_guid: [0; 16],
                routing_off: 0,
                routing_len: 0,
                leaf_region_start: 0,
                routing_unfit: 0,
            },
            bloom: Box::new([0; BLOOM_BYTES]),
            buckets: Box::new([]),
            base_prefix: Box::new([]),
            crossings: Box::new([]),
            components: vec![
                ComponentEntry {
                    delimiter: b'/',
                    prefix: b"bucket/a/".to_vec().into_boxed_slice(),
                },
                ComponentEntry {
                    delimiter: b'/',
                    prefix: b"bucket/b/".to_vec().into_boxed_slice(),
                },
            ]
            .into_boxed_slice(),
            bytes: 0,
        };

        assert_eq!(
            index.next_component_rollup(b"bucket/", b'/', None),
            Some(b"bucket/a/".as_slice())
        );
        assert_eq!(
            index.next_component_rollup(b"bucket/", b'/', Some((b"bucket/a/", false))),
            Some(b"bucket/b/".as_slice())
        );
        assert_eq!(index.next_component_rollup(b"bucket/", b':', None), None);
    }

    #[test]
    fn component_summary_supports_colon_delimiter() {
        let leaves = vec![
            BuildLeaf {
                hash: hash_exact_key(b"tenant:bucket:object-0001\0"),
                value_source: VALUE_INLINE,
                value_source_off: 0,
                value_len: 3,
                value_crc32: crc32fast::hash(b"one"),
                key: b"tenant:bucket:object-0001\0".to_vec().into_boxed_slice(),
                value: Some(b"one".to_vec().into_boxed_slice()),
                seq: 7,
            },
            BuildLeaf {
                hash: hash_exact_key(b"tenant:logs:object-0001\0"),
                value_source: VALUE_INLINE,
                value_source_off: 0,
                value_len: 3,
                value_crc32: crc32fast::hash(b"two"),
                key: b"tenant:logs:object-0001\0".to_vec().into_boxed_slice(),
                value: Some(b"two".to_vec().into_boxed_slice()),
                seq: 8,
            },
        ];
        let components = encode_components(&leaves).unwrap();
        let mut input = components.as_slice();
        let mut decoded = Vec::new();
        while !input.is_empty() {
            let delimiter = take_u8(&mut input).unwrap();
            let len = take_u32(&mut input).unwrap() as usize;
            decoded.push((delimiter, take(&mut input, len).unwrap().to_vec()));
        }
        assert!(decoded.contains(&(b':', b"tenant:".to_vec())));
        assert!(decoded.contains(&(b':', b"tenant:bucket:".to_vec())));
    }

    #[test]
    fn prefix_liveness_uses_base_prefix_and_components() {
        let index = ReadIndex {
            stamp: ReadIndexStamp {
                root_slot: 0,
                num_slots: 0,
                space_used: 0,
                compact_times: 0,
                dead_bytes: 0,
                gap_space: 0,
                tombstone_leaf_cnt: 0,
                created_epoch: 0,
                blob_guid: [0; 16],
                routing_off: 0,
                routing_len: 0,
                leaf_region_start: 0,
                routing_unfit: 0,
            },
            bloom: Box::new([0; BLOOM_BYTES]),
            buckets: vec![BucketEntry { off: 0, len: 12 }].into_boxed_slice(),
            base_prefix: b"bucket/a".to_vec().into_boxed_slice(),
            crossings: Box::new([]),
            components: vec![ComponentEntry {
                delimiter: b'/',
                prefix: b"bucket/a/".to_vec().into_boxed_slice(),
            }]
            .into_boxed_slice(),
            bytes: 0,
        };

        assert_eq!(index.prefix_liveness(b"bucket/"), PrefixLiveness::Present);
        assert_eq!(index.prefix_liveness(b"bucket/a/"), PrefixLiveness::Present);
        assert_eq!(index.prefix_liveness(b"bucket/z/"), PrefixLiveness::Absent);
        assert_eq!(
            index.prefix_liveness(b"bucket/a/missing/"),
            PrefixLiveness::Absent
        );
        assert_eq!(
            index.prefix_liveness(b"bucket/a/file"),
            PrefixLiveness::Unknown
        );
    }

    #[test]
    fn indexes_blob_crossing_prefixes() {
        let guid = [0x13; 16];
        let child = [0x44; 16];
        let mut buf = crate::store::blob_store::AlignedBlobBuf::zeroed();
        {
            let mut frame = BlobFrame::init(buf.as_mut_slice(), guid).unwrap();
            install_blob_node(&mut frame, b"bucket/a/", child);
        }
        let bytes = ReadIndex::build(BlobFrameRef::wrap(buf.as_slice()))
            .unwrap()
            .index;
        let index = decode_dir(&bytes);
        assert!(!index.may_have_key(b"bucket/a/object"));
        assert!(matches!(
            index.route_or_absent(b"bucket/a/object", 0),
            ReadIndexAnswer::Crossing {
                child_guid,
                child_depth: 9,
                ..
            } if child_guid == child
        ));
    }
}
