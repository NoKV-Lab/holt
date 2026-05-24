//! Shared route-cache validation helpers.

use crate::api::errors::{Error, Result};
use crate::engine::RouteHit;
use crate::layout::ROOT_BLOB_GUID;
use crate::store::BlobFrameRef;
use crate::store::{BufferManager, CachedBlob};
use std::sync::Arc;

use super::lookup::lookup_at;
use super::types::LookupResult;
use super::SearchKey;

pub(super) fn validate_route_edge(
    frame: BlobFrameRef<'_>,
    key: SearchKey<'_>,
    route: RouteHit,
) -> Result<bool> {
    let root_slot = frame.header().root_slot;
    match lookup_at(frame, root_slot, key, route.parent_depth)? {
        LookupResult::Crossing(crossing) => Ok(
            crossing.child_guid == route.child_guid && crossing.child_depth == route.child_depth
        ),
        LookupResult::Found(_) | LookupResult::NotFound => Ok(false),
    }
}

pub(super) fn pin_route_parent(
    bm: &BufferManager,
    root_pin: &Arc<CachedBlob>,
    route: RouteHit,
) -> Result<Arc<CachedBlob>> {
    if route.parent_guid == ROOT_BLOB_GUID {
        Ok(Arc::clone(root_pin))
    } else {
        bm.pin(route.parent_guid)
    }
}

pub(super) fn route_pin_not_found(error: &Error) -> bool {
    matches!(
        error,
        Error::BlobStoreIo(err) if err.kind() == std::io::ErrorKind::NotFound
    )
}
