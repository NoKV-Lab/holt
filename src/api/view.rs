//! Scoped read transaction.
//!
//! `View` captures a prefix subtree as immutable blob frames, then
//! reads from that private frame set. It gives stable list/readdir
//! semantics without keeping a live-tree read lock or MVCC chains.

use std::sync::Arc;

use super::atomic::{Record, RecordVersion};
use super::errors::{Error, Result};
use super::tree::{count_scan_limit, prefix_count_from_seen};
use crate::concurrency::Gate;
use crate::engine::{self, KeyRangeBuilder, KeyRangeEntryRef, RangeBuilder};
use crate::layout::BlobGuid;
use crate::store::{BufferManager, CachedBlob};

/// Immutable read transaction over one captured prefix.
///
/// Created by [`crate::Tree::view`]. Subsequent live-tree writes do
/// not affect it.
#[derive(Clone)]
pub struct View {
    scope: Vec<u8>,
    store: Arc<BufferManager>,
    root_guid: BlobGuid,
    root_pin: Arc<CachedBlob>,
    range_gate: Arc<Gate>,
    scan_fence: Option<(Arc<Gate>, Arc<Gate>)>,
}

impl View {
    pub(crate) fn new(
        scope: Vec<u8>,
        store: Arc<BufferManager>,
        root_guid: BlobGuid,
        root_pin: Arc<CachedBlob>,
        scan_fence: Option<(Arc<Gate>, Arc<Gate>)>,
    ) -> Self {
        Self {
            scope,
            store,
            root_guid,
            root_pin,
            range_gate: Arc::new(Gate::new()),
            scan_fence,
        }
    }

    /// Captured prefix for this view.
    #[must_use]
    pub fn scope(&self) -> &[u8] {
        &self.scope
    }

    /// Look up `key` in the view snapshot.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.ensure_in_scope(key)?;
        self.lookup_record(key)
            .map(|record| record.map(|record| record.value))
    }

    /// Look up `key` and return value plus the captured record
    /// version.
    pub fn get_record(&self, key: &[u8]) -> Result<Option<Record>> {
        self.ensure_in_scope(key)?;
        self.lookup_record(key)
    }

    /// Return the captured version token for `key`.
    pub fn get_version(&self, key: &[u8]) -> Result<Option<RecordVersion>> {
        self.ensure_in_scope(key)?;
        let search = engine::SearchKey::user(key);
        engine::lookup_multi_with(&self.store, &self.root_pin, None, search, |hit| {
            RecordVersion::new(hit.seq)
        })
    }

    /// Open a record range over the view's captured prefix.
    pub fn range(&self) -> ViewRangeBuilder {
        ViewRangeBuilder {
            inner: self.range_builder(&self.scope),
        }
    }

    /// Open a record range for a narrower prefix inside the view.
    pub fn scan(&self, prefix: &[u8]) -> Result<ViewRangeBuilder> {
        self.ensure_in_scope(prefix)?;
        Ok(ViewRangeBuilder {
            inner: self.range_builder(prefix),
        })
    }

    /// Open a key-only range over the view's captured prefix.
    pub fn range_keys(&self) -> ViewKeyRangeBuilder {
        ViewKeyRangeBuilder {
            inner: KeyRangeBuilder::new(self.range_builder(&self.scope)),
        }
    }

    /// Open a key-only range for a narrower prefix inside the view.
    pub fn scan_keys(&self, prefix: &[u8]) -> Result<ViewKeyRangeBuilder> {
        self.ensure_in_scope(prefix)?;
        Ok(ViewKeyRangeBuilder {
            inner: KeyRangeBuilder::new(self.range_builder(prefix)),
        })
    }

    /// Return `true` if no captured key starts with `prefix`.
    pub fn is_prefix_empty(&self, prefix: &[u8]) -> Result<bool> {
        let stats = self.scan_keys(prefix)?.visit(1, |_| Ok(()))?;
        Ok(stats.returned + stats.rollup == 0)
    }

    /// Count captured live keys under `prefix`, optionally capped by `limit`.
    ///
    /// `limit == 0` means exact / unbounded. Reads remain inside the view's
    /// copy-on-write snapshot and never observe later live-tree writes.
    pub fn prefix_count(&self, prefix: &[u8], limit: usize) -> Result<crate::PrefixCount> {
        let scan_limit = count_scan_limit(limit);
        let mut seen = 0u64;
        let outcome = self
            .scan_keys(prefix)?
            .visit_with_outcome(scan_limit, |entry| {
                if let KeyRangeEntryRef::Key { .. } = entry {
                    seen = seen.saturating_add(1);
                }
                Ok(())
            })?;
        Ok(prefix_count_from_seen(seen, limit, outcome))
    }

    fn lookup_record(&self, key: &[u8]) -> Result<Option<Record>> {
        let search = engine::SearchKey::user(key);
        engine::lookup_multi_with(&self.store, &self.root_pin, None, search, |hit| Record {
            value: hit.value.to_vec(),
            version: RecordVersion::new(hit.seq),
        })
    }

    fn range_builder(&self, prefix: &[u8]) -> RangeBuilder {
        let builder = RangeBuilder::new(
            Arc::clone(&self.store),
            Arc::clone(&self.root_pin),
            self.root_guid,
            self.scan_fence.as_ref().map_or_else(
                || Arc::clone(&self.range_gate),
                |(gate, _)| Arc::clone(gate),
            ),
        );
        let builder = if let Some((_, mutation_gate)) = &self.scan_fence {
            builder.with_mutation_gate(Arc::clone(mutation_gate))
        } else {
            builder.snapshot_cursor()
        };
        builder.prefix(prefix)
    }

    fn ensure_in_scope(&self, prefix_or_key: &[u8]) -> Result<()> {
        if self.scope.is_empty() || prefix_or_key.starts_with(&self.scope) {
            return Ok(());
        }
        Err(Error::OutsideViewScope {
            requested_len: prefix_or_key.len(),
            scope_len: self.scope.len(),
        })
    }
}

/// Record range builder scoped to a [`View`].
#[must_use = "ViewRangeBuilder is lazy — call `.into_iter()` or use it in a `for` loop"]
pub struct ViewRangeBuilder {
    inner: RangeBuilder,
}

impl ViewRangeBuilder {
    /// Strict-greater-than lower bound inside the view's range.
    pub fn start_after(mut self, key: &[u8]) -> Self {
        self.inner = self.inner.start_after(key);
        self
    }

    /// S3-style delimiter byte.
    pub fn delimiter(mut self, byte: u8) -> Self {
        self.inner = self.inner.delimiter(byte);
        self
    }
}

impl IntoIterator for ViewRangeBuilder {
    type Item = Result<crate::RangeEntry>;
    type IntoIter = crate::RangeIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

/// Key-only range builder scoped to a [`View`].
#[must_use = "ViewKeyRangeBuilder is lazy — call `.into_iter()`, `.visit()`, or use it in a `for` loop"]
pub struct ViewKeyRangeBuilder {
    inner: KeyRangeBuilder,
}

impl ViewKeyRangeBuilder {
    /// Strict-greater-than lower bound inside the view's range.
    pub fn start_after(mut self, key: &[u8]) -> Self {
        self.inner = self.inner.start_after(key);
        self
    }

    /// S3-style delimiter byte.
    pub fn delimiter(mut self, byte: u8) -> Self {
        self.inner = self.inner.delimiter(byte);
        self
    }

    /// Visit key-only entries with borrowed key bytes, returning the
    /// scan's [`crate::ScanStats`].
    pub fn visit<F>(self, limit: usize, visitor: F) -> Result<crate::ScanStats>
    where
        F: FnMut(crate::KeyRangeEntryRef<'_>) -> Result<()>,
    {
        self.inner.visit(limit, visitor)
    }

    /// Visit key-only entries and return scan/cache outcome metadata.
    pub fn visit_with_outcome<F>(self, limit: usize, visitor: F) -> Result<crate::KeyScanOutcome>
    where
        F: FnMut(crate::KeyRangeEntryRef<'_>) -> Result<()>,
    {
        self.inner.visit_with_outcome(limit, visitor)
    }
}

impl IntoIterator for ViewKeyRangeBuilder {
    type Item = Result<crate::KeyRangeEntry>;
    type IntoIter = crate::KeyRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}
