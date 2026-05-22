//! Small root-to-child route cache for path-shaped metadata keys.
//!
//! This cache only remembers the first `BlobNode` crossing found
//! from the root blob. A hit is usable only while the root blob's
//! content version still equals the cached version; callers either
//! hold the root shared latch while acquiring the child or revalidate
//! the observed root version after acquiring the child. That keeps
//! the parent edge stable without re-running the root ART descent on
//! every large-tree metadata update.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

use crate::layout::BlobGuid;

use super::walker::SearchKey;

const ROUTE_CACHE_CAPACITY: usize = 16_384;
const ROUTE_PREFIX_MAX: usize = 96;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RouteCacheSnapshot {
    pub(crate) entries: usize,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) learns: u64,
    pub(crate) evictions: u64,
    pub(crate) invalidations: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RouteHit {
    pub(crate) child_guid: BlobGuid,
    pub(crate) child_depth: usize,
}

#[derive(Debug, Clone)]
struct RouteEntry {
    child_guid: BlobGuid,
    child_depth: usize,
}

#[derive(Debug)]
struct RouteEntries {
    map: HashMap<Vec<u8>, RouteEntry>,
    order: Vec<Vec<u8>>,
    lengths: Vec<usize>,
}

/// A tiny associative cache for top-level path routes.
#[derive(Debug)]
pub(crate) struct RouteCache {
    root_version: AtomicU64,
    entries: RwLock<RouteEntries>,
    replace_cursor: AtomicUsize,
    hits: AtomicU64,
    misses: AtomicU64,
    learns: AtomicU64,
    evictions: AtomicU64,
    invalidations: AtomicU64,
}

impl Default for RouteCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RouteCache {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            root_version: AtomicU64::new(u64::MAX),
            entries: RwLock::new(RouteEntries {
                map: HashMap::with_capacity(ROUTE_CACHE_CAPACITY),
                order: Vec::with_capacity(ROUTE_CACHE_CAPACITY),
                lengths: Vec::new(),
            }),
            replace_cursor: AtomicUsize::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            learns: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub(crate) fn stats(&self) -> RouteCacheSnapshot {
        RouteCacheSnapshot {
            entries: self.entries.read().unwrap().map.len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            learns: self.learns.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            invalidations: self.invalidations.load(Ordering::Relaxed),
        }
    }

    /// Return a cached first-blob crossing for `key` if the key is
    /// under a cached prefix and the entry was learned from the same
    /// root blob version the caller is currently holding stable.
    #[must_use]
    pub(crate) fn lookup(&self, key: SearchKey<'_>, root_version: u64) -> Option<RouteHit> {
        if self.root_version.load(Ordering::Acquire) != root_version {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let entries = self.entries.read().unwrap();

        // Try only prefix lengths that have been observed in learned
        // routes. Large metadata trees typically settle on a small
        // number of crossing depths; checking every possible byte
        // length would turn a cache hit into dozens of failed hash
        // probes on each update.
        for &len in &entries.lengths {
            if let Some(prefix) = key.user_prefix(len) {
                if let Some(entry) = entries.map.get(prefix) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Some(RouteHit {
                        child_guid: entry.child_guid,
                        child_depth: entry.child_depth,
                    });
                }
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Learn a root crossing just observed under a stable root read
    /// latch. Entries whose prefix would include the virtual
    /// terminator or exceed the inline budget are deliberately not
    /// cached; those shapes do not have useful route locality.
    pub(crate) fn learn(
        &self,
        key: SearchKey<'_>,
        root_version: u64,
        child_guid: BlobGuid,
        child_depth: usize,
    ) {
        let Some(prefix) = key.user_prefix(child_depth) else {
            return;
        };
        if prefix.len() > ROUTE_PREFIX_MAX {
            return;
        }

        let mut entries = self.entries.write().unwrap();
        if self.root_version.load(Ordering::Relaxed) != root_version {
            if !entries.map.is_empty() {
                self.invalidations
                    .fetch_add(entries.map.len() as u64, Ordering::Relaxed);
            }
            entries.map.clear();
            entries.order.clear();
            entries.lengths.clear();
            self.replace_cursor.store(0, Ordering::Relaxed);
            self.root_version.store(root_version, Ordering::Release);
        }
        if let Some(entry) = entries.map.get_mut(prefix) {
            entry.child_guid = child_guid;
            entry.child_depth = child_depth;
            self.learns.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let entry = RouteEntry {
            child_guid,
            child_depth,
        };
        remember_prefix_len(&mut entries.lengths, prefix.len());
        self.learns.fetch_add(1, Ordering::Relaxed);
        if entries.map.len() < ROUTE_CACHE_CAPACITY {
            entries.order.push(prefix.to_vec());
            entries.map.insert(prefix.to_vec(), entry);
            return;
        }
        let idx = self.replace_cursor.fetch_add(1, Ordering::Relaxed) % ROUTE_CACHE_CAPACITY;
        let new_prefix = prefix.to_vec();
        let old = std::mem::replace(&mut entries.order[idx], new_prefix.clone());
        entries.map.remove(old.as_slice());
        entries.map.insert(new_prefix, entry);
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
}

fn remember_prefix_len(lengths: &mut Vec<usize>, len: usize) {
    match lengths.binary_search_by(|known| known.cmp(&len).reverse()) {
        Ok(_) => {}
        Err(idx) => lengths.insert(idx, len),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHILD: BlobGuid = [7; 16];

    #[test]
    fn learns_and_matches_longest_prefix_for_same_root_version() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"bucket-01/a"), 3, [1; 16], 10);
        cache.learn(SearchKey::user(b"bucket-01/path/file"), 3, CHILD, 15);

        let hit = cache
            .lookup(SearchKey::user(b"bucket-01/path/other"), 3)
            .unwrap();
        assert_eq!(hit.child_guid, CHILD);
        assert_eq!(hit.child_depth, 15);
    }

    #[test]
    fn root_version_mismatch_misses() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"bucket-01/path/file"), 3, CHILD, 15);

        assert!(cache
            .lookup(SearchKey::user(b"bucket-01/path/other"), 4)
            .is_none());
    }

    #[test]
    fn new_root_version_drops_old_routes() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"bucket-01/path/file"), 3, CHILD, 15);
        cache.learn(SearchKey::user(b"bucket-02/path/file"), 4, [9; 16], 15);

        assert!(cache
            .lookup(SearchKey::user(b"bucket-01/path/other"), 4)
            .is_none());
        let hit = cache
            .lookup(SearchKey::user(b"bucket-02/path/other"), 4)
            .unwrap();
        assert_eq!(hit.child_guid, [9; 16]);
    }

    #[test]
    fn does_not_cache_prefix_past_user_key() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"abc"), 1, CHILD, 4);

        assert!(cache.lookup(SearchKey::user(b"abc"), 1).is_none());
    }

    #[test]
    fn stats_track_hits_misses_and_replacements() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"bucket-00/path/file"), 1, CHILD, 15);

        assert!(cache
            .lookup(SearchKey::user(b"bucket-00/path/other"), 1)
            .is_some());
        assert!(cache
            .lookup(SearchKey::user(b"bucket-01/path/other"), 1)
            .is_none());

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.learns, 1);

        for i in 0..=ROUTE_CACHE_CAPACITY {
            let key = format!("bucket-{i:03}/path/file");
            cache.learn(SearchKey::user(key.as_bytes()), 1, [i as u8; 16], 15);
        }

        let stats = cache.stats();
        assert_eq!(stats.entries, ROUTE_CACHE_CAPACITY);
        assert!(stats.evictions > 0);
    }

    #[test]
    fn stats_count_root_version_invalidations() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"bucket-00/path/file"), 1, CHILD, 15);
        cache.learn(SearchKey::user(b"bucket-01/path/file"), 2, [8; 16], 15);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.invalidations, 1);
    }
}
