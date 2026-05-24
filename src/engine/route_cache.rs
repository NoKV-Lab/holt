//! Small parent-validated route cache for path-shaped metadata keys.
//!
//! A hit is only a candidate. Callers must pin the cached parent,
//! hold its shared latch, verify the parent content version, and
//! then pin the child before using the shortcut. That keeps the
//! cached parent->child edge stable while still allowing deeper
//! prefix anchors than the root-only path.

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
    pub(crate) parent_guid: BlobGuid,
    pub(crate) parent_depth: usize,
    pub(crate) parent_version: u64,
    pub(crate) child_guid: BlobGuid,
    pub(crate) child_depth: usize,
}

#[derive(Debug, Clone)]
struct RouteEntry {
    parent_guid: BlobGuid,
    parent_depth: usize,
    parent_version: u64,
    child_guid: BlobGuid,
    child_depth: usize,
}

#[derive(Debug)]
struct RouteEntries {
    map: HashMap<Vec<u8>, RouteEntry>,
    order: Vec<Vec<u8>>,
    lengths: Vec<usize>,
}

/// A tiny associative cache for path-prefix routes.
#[derive(Debug)]
pub(crate) struct RouteCache {
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

    /// Return the longest cached crossing candidate for `key`.
    ///
    /// The caller validates the parent token before trusting the
    /// child edge. Lookup deliberately does not remove stale entries:
    /// invalidation is detected under the parent's shared latch.
    #[must_use]
    pub(crate) fn lookup(&self, key: SearchKey<'_>) -> Option<RouteHit> {
        {
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
                            parent_guid: entry.parent_guid,
                            parent_depth: entry.parent_depth,
                            parent_version: entry.parent_version,
                            child_guid: entry.child_guid,
                            child_depth: entry.child_depth,
                        });
                    }
                }
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Drop a caller-detected stale route candidate.
    pub(crate) fn invalidate(&self, key: SearchKey<'_>, route: RouteHit) {
        self.invalidations.fetch_add(1, Ordering::Relaxed);
        let Some(prefix) = key.user_prefix(route.child_depth) else {
            return;
        };
        let mut entries = self.entries.write().unwrap();
        let Some(entry) = entries.map.get(prefix) else {
            return;
        };
        if entry.parent_guid != route.parent_guid
            || entry.parent_depth != route.parent_depth
            || entry.parent_version != route.parent_version
            || entry.child_guid != route.child_guid
            || entry.child_depth != route.child_depth
        {
            return;
        }
        entries.map.remove(prefix);
        entries.order.retain(|known| known.as_slice() != prefix);
        rebuild_prefix_lengths(&mut entries);
    }

    /// Refresh the parent version after the caller revalidated that
    /// the cached parent edge still points at the same child.
    pub(crate) fn refresh_parent_version(
        &self,
        key: SearchKey<'_>,
        route: RouteHit,
        parent_version: u64,
    ) {
        let Some(prefix) = key.user_prefix(route.child_depth) else {
            return;
        };
        let mut entries = self.entries.write().unwrap();
        let Some(entry) = entries.map.get_mut(prefix) else {
            return;
        };
        if entry.parent_guid == route.parent_guid
            && entry.parent_depth == route.parent_depth
            && entry.parent_version == route.parent_version
            && entry.child_guid == route.child_guid
            && entry.child_depth == route.child_depth
        {
            entry.parent_version = parent_version;
        }
    }

    /// Learn a crossing just observed under a stable parent read
    /// latch. Entries whose prefix would include the virtual
    /// terminator or exceed the inline budget are deliberately not
    /// cached; those shapes do not have useful route locality.
    pub(crate) fn learn(
        &self,
        key: SearchKey<'_>,
        parent_guid: BlobGuid,
        parent_depth: usize,
        parent_version: u64,
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
        if let Some(entry) = entries.map.get_mut(prefix) {
            entry.parent_guid = parent_guid;
            entry.parent_depth = parent_depth;
            entry.parent_version = parent_version;
            entry.child_guid = child_guid;
            entry.child_depth = child_depth;
            self.learns.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if has_dominating_prefix(&entries, prefix, parent_guid, parent_depth, child_guid) {
            self.learns.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let entry = RouteEntry {
            parent_guid,
            parent_depth,
            parent_version,
            child_guid,
            child_depth,
        };
        prune_dominated_prefixes(&mut entries, prefix, parent_guid, parent_depth, child_guid);
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

fn has_dominating_prefix(
    entries: &RouteEntries,
    prefix: &[u8],
    parent_guid: BlobGuid,
    parent_depth: usize,
    child_guid: BlobGuid,
) -> bool {
    for &len in &entries.lengths {
        if len >= prefix.len() {
            continue;
        }
        if let Some(entry) = entries.map.get(&prefix[..len]) {
            if entry.parent_guid == parent_guid
                && entry.parent_depth == parent_depth
                && entry.child_guid == child_guid
            {
                return true;
            }
        }
    }
    false
}

fn prune_dominated_prefixes(
    entries: &mut RouteEntries,
    prefix: &[u8],
    parent_guid: BlobGuid,
    parent_depth: usize,
    child_guid: BlobGuid,
) {
    let mut removed = false;
    entries.order.retain(|known| {
        let dominated = known.len() > prefix.len()
            && known.starts_with(prefix)
            && entries.map.get(known.as_slice()).is_some_and(|entry| {
                entry.parent_guid == parent_guid
                    && entry.parent_depth == parent_depth
                    && entry.child_guid == child_guid
            });
        if dominated {
            entries.map.remove(known.as_slice());
            removed = true;
            false
        } else {
            true
        }
    });
    if removed {
        rebuild_prefix_lengths(entries);
    }
}

fn rebuild_prefix_lengths(entries: &mut RouteEntries) {
    entries.lengths.clear();
    let mut lens: Vec<_> = entries.map.keys().map(Vec::len).collect();
    lens.sort_unstable_by(|a, b| b.cmp(a));
    lens.dedup();
    entries.lengths = lens;
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

    const PARENT: BlobGuid = [1; 16];
    const OTHER_PARENT: BlobGuid = [2; 16];
    const CHILD: BlobGuid = [7; 16];

    #[test]
    fn learns_and_matches_longest_prefix() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"bucket-01/a"), PARENT, 0, 3, [1; 16], 10);
        cache.learn(
            SearchKey::user(b"bucket-01/path/file"),
            PARENT,
            0,
            3,
            CHILD,
            15,
        );

        let hit = cache
            .lookup(SearchKey::user(b"bucket-01/path/other"))
            .unwrap();
        assert_eq!(hit.parent_guid, PARENT);
        assert_eq!(hit.parent_depth, 0);
        assert_eq!(hit.parent_version, 3);
        assert_eq!(hit.child_guid, CHILD);
        assert_eq!(hit.child_depth, 15);
    }

    #[test]
    fn invalidate_removes_only_matching_stale_route() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-01/path/file"),
            PARENT,
            0,
            3,
            CHILD,
            15,
        );
        cache.learn(
            SearchKey::user(b"bucket-02/path/file"),
            OTHER_PARENT,
            0,
            4,
            [9; 16],
            15,
        );

        let stale = cache
            .lookup(SearchKey::user(b"bucket-01/path/other"))
            .unwrap();
        cache.invalidate(SearchKey::user(b"bucket-01/path/other"), stale);

        assert!(cache
            .lookup(SearchKey::user(b"bucket-01/path/other"))
            .is_none());
        let hit = cache
            .lookup(SearchKey::user(b"bucket-02/path/other"))
            .unwrap();
        assert_eq!(hit.child_guid, [9; 16]);
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.invalidations, 1);
    }

    #[test]
    fn does_not_cache_prefix_past_user_key() {
        let cache = RouteCache::new();
        cache.learn(SearchKey::user(b"abc"), PARENT, 0, 1, CHILD, 4);

        assert!(cache.lookup(SearchKey::user(b"abc")).is_none());
    }

    #[test]
    fn shorter_route_dominates_same_child_longer_route() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            PARENT,
            0,
            1,
            CHILD,
            10,
        );
        cache.learn(
            SearchKey::user(b"bucket-00/path/deeper/file"),
            PARENT,
            0,
            1,
            CHILD,
            15,
        );

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        let hit = cache
            .lookup(SearchKey::user(b"bucket-00/path/deeper/other"))
            .unwrap();
        assert_eq!(hit.child_guid, CHILD);
        assert_eq!(hit.child_depth, 10);
    }

    #[test]
    fn shorter_route_prunes_same_child_longer_route() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/deeper/file"),
            PARENT,
            0,
            1,
            CHILD,
            15,
        );
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            PARENT,
            0,
            1,
            CHILD,
            10,
        );

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        let hit = cache
            .lookup(SearchKey::user(b"bucket-00/path/deeper/other"))
            .unwrap();
        assert_eq!(hit.child_guid, CHILD);
        assert_eq!(hit.child_depth, 10);
    }

    #[test]
    fn stats_track_hits_misses_and_replacements() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            PARENT,
            0,
            1,
            CHILD,
            15,
        );

        assert!(cache
            .lookup(SearchKey::user(b"bucket-00/path/other"))
            .is_some());
        assert!(cache
            .lookup(SearchKey::user(b"bucket-01/path/other"))
            .is_none());

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.learns, 1);

        for i in 0..=ROUTE_CACHE_CAPACITY {
            let key = format!("bucket-{i:03}/path/file");
            cache.learn(
                SearchKey::user(key.as_bytes()),
                PARENT,
                0,
                1,
                [i as u8; 16],
                15,
            );
        }

        let stats = cache.stats();
        assert_eq!(stats.entries, ROUTE_CACHE_CAPACITY);
        assert!(stats.evictions > 0);
    }

    #[test]
    fn stats_count_stale_invalidations() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            PARENT,
            0,
            1,
            CHILD,
            15,
        );
        let hit = cache
            .lookup(SearchKey::user(b"bucket-00/path/other"))
            .unwrap();
        cache.invalidate(SearchKey::user(b"bucket-00/path/other"), hit);

        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.invalidations, 1);
    }

    #[test]
    fn refresh_parent_version_keeps_stable_edge_cached() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            PARENT,
            0,
            1,
            CHILD,
            15,
        );
        let hit = cache
            .lookup(SearchKey::user(b"bucket-00/path/other"))
            .unwrap();
        cache.refresh_parent_version(SearchKey::user(b"bucket-00/path/other"), hit, 7);

        let refreshed = cache
            .lookup(SearchKey::user(b"bucket-00/path/again"))
            .unwrap();
        assert_eq!(refreshed.parent_version, 7);
        assert_eq!(cache.stats().invalidations, 0);
    }

    #[test]
    fn stale_longer_route_can_be_removed_to_expose_shorter_route() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/deeper/file"),
            PARENT,
            0,
            1,
            [1; 16],
            21,
        );
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            OTHER_PARENT,
            0,
            2,
            CHILD,
            15,
        );

        let stale = cache
            .lookup(SearchKey::user(b"bucket-00/path/deeper/other"))
            .unwrap();
        assert_eq!(stale.child_guid, [1; 16]);
        cache.invalidate(SearchKey::user(b"bucket-00/path/deeper/other"), stale);

        let hit = cache
            .lookup(SearchKey::user(b"bucket-00/path/deeper/other"))
            .unwrap();
        assert_eq!(hit.child_guid, CHILD);
        assert_eq!(hit.child_depth, 15);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.invalidations, 1);
    }

    #[test]
    fn different_parent_token_does_not_prune_longer_route() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/deeper/file"),
            PARENT,
            0,
            1,
            [1; 16],
            21,
        );
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            OTHER_PARENT,
            0,
            1,
            CHILD,
            15,
        );

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
    }

    #[test]
    fn different_parent_depth_does_not_prune_longer_route() {
        let cache = RouteCache::new();
        cache.learn(
            SearchKey::user(b"bucket-00/path/deeper/file"),
            PARENT,
            12,
            1,
            [1; 16],
            21,
        );
        cache.learn(
            SearchKey::user(b"bucket-00/path/file"),
            PARENT,
            0,
            1,
            CHILD,
            15,
        );

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
    }
}
