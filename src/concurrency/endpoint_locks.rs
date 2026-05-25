use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, MutexGuard};

const RENAME_LOCK_SHARDS: usize = 256;

/// Fixed-shard locks for multi-key operation endpoints.
///
/// Multi-key operations mutate two logical endpoints (`src`, `dst`).
/// Locking only those endpoint shards keeps unrelated operations
/// concurrent, while canonical shard ordering prevents AB/BA
/// deadlock.
pub(crate) struct EndpointLocks {
    shards: [Mutex<()>; RENAME_LOCK_SHARDS],
}

impl EndpointLocks {
    pub(crate) fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(())),
        }
    }

    pub(crate) fn lock_pair<'a>(&'a self, src: &[u8], dst: &[u8]) -> EndpointLockGuard<'a> {
        let src_idx = shard_index(src);
        let dst_idx = shard_index(dst);
        if src_idx == dst_idx {
            return EndpointLockGuard {
                _first: self.shards[src_idx].lock().unwrap(),
                _second: None,
            };
        }

        let (first_idx, second_idx) = if src_idx < dst_idx {
            (src_idx, dst_idx)
        } else {
            (dst_idx, src_idx)
        };
        let first = self.shards[first_idx].lock().unwrap();
        let second = self.shards[second_idx].lock().unwrap();
        EndpointLockGuard {
            _first: first,
            _second: Some(second),
        }
    }
}

pub(crate) struct EndpointLockGuard<'a> {
    _first: MutexGuard<'a, ()>,
    _second: Option<MutexGuard<'a, ()>>,
}

fn shard_index(key: &[u8]) -> usize {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    (h.finish() as usize) & (RENAME_LOCK_SHARDS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn reversed_endpoint_order_does_not_deadlock() {
        let locks = Arc::new(EndpointLocks::new());
        let a = b"bucket-a/object-1".to_vec();
        let b = b"bucket-b/object-2".to_vec();
        let (tx, rx) = std::sync::mpsc::channel();

        for _ in 0..2 {
            let locks = Arc::clone(&locks);
            let a = a.clone();
            let b = b.clone();
            let tx = tx.clone();
            thread::spawn(move || {
                for _ in 0..10_000 {
                    {
                        let _guard = locks.lock_pair(&a, &b);
                    }
                    {
                        let _guard = locks.lock_pair(&b, &a);
                    }
                }
                tx.send(()).unwrap();
            });
        }
        drop(tx);

        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    }
}
