//! Copy-on-write frame snapshot.
//!
//! A [`Snapshot`] is a stable, point-in-time view of a tree (or a
//! prefix subtree). It copies only the root frame into memory up front
//! and shares all other frames with the live tree. Later writes *fork*
//! (copy-on-write) the individual frames a snapshot still references
//! instead of overwriting them in place, so the snapshot stays stable
//! with 1× read amplification and without MVCC version chains.
//!
//! Creation is O(one frame copy); the per-write cost is zero while no
//! snapshot is live, and bounded by the root→leaf frame path length on
//! the first write to each region while one is. Dropping the handle (or
//! calling [`Snapshot::retire`]) releases that handle's process-local lease
//! reference; the global fork barrier can lower only after every cloned view,
//! range builder, and owned cursor derived from that snapshot has also been
//! dropped.

use super::view::View;
use std::ops::Deref;

/// A stable copy-on-write snapshot of a tree or prefix subtree.
///
/// Created by [`crate::Tree::snapshot`]. Reads see the tree exactly as
/// it was at creation time regardless of concurrent or subsequent live
/// writes. All [`View`] read operations are available through `Deref`
/// (`snapshot.get(..)`, `snapshot.range()`, `snapshot.scan(..)`, …).
///
/// The snapshot epoch is retired after this handle and all cloned views,
/// range builders, or owned cursors derived from it are dropped. This lease is
/// process-local lifetime bookkeeping, not a persistent or cross-process
/// snapshot lease.
/// Persisted copy-on-write frames are not reclaimed inline on handle drop.
/// A later standalone or DB checkpoint reclaims a bounded exact batch, while
/// [`crate::Tree::gc`] and [`crate::DB::gc`] provide complete reachability
/// sweeps; [`crate::StoreStats::live_blobs`] may therefore remain elevated
/// until one of those maintenance frontiers completes.
pub struct Snapshot {
    view: Option<View>,
    epoch: u64,
}

impl Snapshot {
    pub(crate) fn new(view: View, epoch: u64) -> Self {
        Self {
            view: Some(view),
            epoch,
        }
    }

    /// This snapshot's epoch — its position on the global copy-on-write
    /// timeline. Frames with `created_epoch <= epoch` are the ones this
    /// snapshot may reference and that live writes must fork.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The underlying scoped read view.
    #[must_use]
    pub fn view(&self) -> &View {
        self.view.as_ref().expect("snapshot retired")
    }

    /// Release this snapshot handle now. The fork-barrier epoch retires
    /// after the last cloned [`View`], range builder, or owned cursor derived
    /// from it is also dropped.
    pub fn retire(mut self) {
        drop(self.view.take());
    }
}

impl Deref for Snapshot {
    type Target = View;

    fn deref(&self) -> &View {
        self.view()
    }
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("epoch", &self.epoch)
            .field("scope", &self.view.as_ref().map(View::scope))
            .finish_non_exhaustive()
    }
}
