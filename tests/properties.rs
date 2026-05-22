//! Property-based tests — random sequences of `put` / `delete` /
//! `rename`, cross-checked against a `HashMap` oracle. Catches
//! correctness bugs that hand-written cases miss: weird key/value
//! length combinations, interleaved insert+erase orderings, key
//! prefix relationships that exercise `Prefix` split / collapse
//! paths.
//!
//! Two suites:
//! 1. **In-memory tree** (`memory_round_trips_against_oracle`) —
//!    fast, hits every walker arm including spillover at multi-blob
//!    scale when the random input is long enough.
//! 2. **Persistent tree with reopen** (`persistent_round_trips_via_wal_replay`)
//!    — applies the ops to a tree, drops it, reopens, and verifies
//!    the post-replay state matches the oracle.

use std::collections::HashMap;

use proptest::collection::vec;
use proptest::prelude::*;

use holt::{RangeEntry, Tree, TreeConfig, WalCommit};

/// Drain a `RangeIter` into a `Vec<(key, value)>`. Panics on any
/// `Err` (proptest harness reports the panic up). `CommonPrefix`
/// entries are filtered out — they only appear under a
/// `delimiter` filter, which these tests don't set.
fn collect_kv(
    iter: impl IntoIterator<Item = Result<RangeEntry, holt::Error>>,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    iter.into_iter()
        .filter_map(|r| match r.unwrap() {
            RangeEntry::Key { key, value, .. } => Some((key, value)),
            RangeEntry::CommonPrefix(_) => None,
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect()
}

/// A single op in the random sequence.
#[derive(Debug, Clone)]
enum Op {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
    Rename(Vec<u8>, Vec<u8>, bool),
}

/// Generate a small key from a constrained alphabet so the
/// chance of two ops touching the same key is meaningful (the
/// interesting cases — update, delete-then-reinsert, rename
/// collision — only fire when keys actually collide).
fn key_strategy() -> impl Strategy<Value = Vec<u8>> {
    // 1..=8 bytes drawn from a 4-byte alphabet → ~65k distinct
    // keys but heavy collisions across a 200-op run.
    prop::collection::vec(prop::sample::select(vec![b'a', b'b', b'/', b'0']), 1..=8)
}

fn value_strategy() -> impl Strategy<Value = Vec<u8>> {
    // 0..=64 bytes of arbitrary bytes — exercise both empty
    // values and chunky leaf extents.
    prop::collection::vec(any::<u8>(), 0..=64)
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        // 50% put — most common in real workloads.
        5 => (key_strategy(), value_strategy()).prop_map(|(k, v)| Op::Put(k, v)),
        // 30% delete.
        3 => key_strategy().prop_map(Op::Delete),
        // 20% rename — split across `force=true/false`.
        1 => (key_strategy(), key_strategy()).prop_map(|(s, d)| Op::Rename(s, d, false)),
        1 => (key_strategy(), key_strategy()).prop_map(|(s, d)| Op::Rename(s, d, true)),
    ]
}

/// Drive `tree` through `ops`, mirroring each mutation onto the
/// oracle. Returns the oracle's final snapshot.
fn apply(tree: &Tree, ops: &[Op]) -> HashMap<Vec<u8>, Vec<u8>> {
    let mut oracle: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    for op in ops {
        match op {
            Op::Put(k, v) => {
                tree.put(k, v).unwrap();
                oracle.insert(k.clone(), v.clone());
            }
            Op::Delete(k) => {
                let prev = tree.remove(k).unwrap();
                assert_eq!(prev.as_deref(), oracle.get(k).map(|v| v.as_slice()));
                oracle.remove(k);
            }
            Op::Rename(s, d, force) => {
                // `Tree::rename` semantics:
                //   - src missing → NotFound
                //   - dst present + force=false → DstExists
                //   - src == dst → no-op
                //   - else → move bytes from src to dst
                let src_present = oracle.contains_key(s);
                let dst_present = oracle.contains_key(d);
                let result = tree.rename(s, d, *force);
                match result {
                    Ok(()) => {
                        // The Ok case is taken iff src was present AND
                        // (dst was absent OR force OR src == dst).
                        assert!(src_present, "rename Ok but oracle had no src");
                        if s == d {
                            // no-op — oracle unchanged.
                        } else {
                            assert!(*force || !dst_present);
                            let v = oracle.remove(s).unwrap();
                            oracle.insert(d.clone(), v);
                        }
                    }
                    Err(holt::Error::NotFound) => {
                        assert!(!src_present);
                    }
                    Err(holt::Error::DstExists) => {
                        assert!(src_present && dst_present && !force && s != d);
                    }
                    Err(e) => panic!("unexpected rename error: {e:?}"),
                }
            }
        }
    }
    oracle
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OracleAtomicErr {
    NotFound,
    DstExists,
}

fn apply_atomic_to_oracle(
    oracle: &mut HashMap<Vec<u8>, Vec<u8>>,
    ops: &[Op],
) -> std::result::Result<(), OracleAtomicErr> {
    let mut staged = oracle.clone();
    for op in ops {
        match op {
            Op::Put(k, v) => {
                staged.insert(k.clone(), v.clone());
            }
            Op::Delete(k) => {
                staged.remove(k);
            }
            Op::Rename(s, d, force) => {
                let Some(v) = staged.get(s).cloned() else {
                    return Err(OracleAtomicErr::NotFound);
                };
                if s == d {
                    continue;
                }
                if !*force && staged.contains_key(d) {
                    return Err(OracleAtomicErr::DstExists);
                }
                staged.remove(s);
                staged.insert(d.clone(), v);
            }
        }
    }
    *oracle = staged;
    Ok(())
}

/// Read every (key, value) pair back out of `tree` and assert
/// it matches the oracle bit-for-bit. Also checks that a key
/// not in the oracle returns `None`.
fn check(tree: &Tree, oracle: &HashMap<Vec<u8>, Vec<u8>>) {
    for (k, v) in oracle {
        let got = tree.get(k).unwrap();
        assert_eq!(
            got.as_deref(),
            Some(v.as_slice()),
            "tree.get({k:?}) returned {got:?}, oracle had {v:?}",
        );
    }
    // Negative-existence sanity probe: pick a key the oracle
    // doesn't have and verify the tree agrees.
    let absent: &[u8] = b"_PROPTEST_SENTINEL_ABSENT_/zzz";
    if !oracle.contains_key(absent) {
        assert_eq!(tree.get(absent).unwrap(), None);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        // Mutation traces with many key collisions catch the most
        // bugs; deeper sequences also exercise the Prefix split /
        // collapse paths. Cap at 200 ops to keep total runtime
        // reasonable in CI.
        max_shrink_iters: 64,
        ..ProptestConfig::default()
    })]

    /// In-memory tree: apply random ops, then verify the tree
    /// matches an oracle `HashMap`.
    #[test]
    fn memory_round_trips_against_oracle(
        ops in vec(op_strategy(), 1..=200),
    ) {
        let tree = Tree::open(TreeConfig::memory()).unwrap();
        let oracle = apply(&tree, &ops);
        check(&tree, &oracle);
    }

    /// Persistent tree: apply ops, drop without calling
    /// `checkpoint`, reopen, verify the WAL replay rebuilds
    /// exactly the oracle's state. `WalCommit::Sync` is required
    /// so every record is durable before drop.
    #[test]
    fn persistent_round_trips_via_wal_replay(
        ops in vec(op_strategy(), 1..=100),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = TreeConfig::new(dir.path());
        cfg.wal_commit = WalCommit::Sync;

        let oracle = {
            let tree = Tree::open(cfg.clone()).unwrap();
            apply(&tree, &ops)
        }; // tree dropped without checkpoint

        let tree = Tree::open(cfg).unwrap();
        check(&tree, &oracle);
    }

    /// `Tree::atomic` batches: preflight each random batch against a
    /// staged `HashMap` oracle, then verify holt either commits the
    /// same staged state or rejects the whole batch without partial
    /// publication.
    #[test]
    fn batch_round_trips_against_oracle(
        batches in vec(vec(op_strategy(), 1..=8), 1..=30),
    ) {
        let tree = Tree::open(TreeConfig::memory()).unwrap();
        let mut oracle: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

        for batch in &batches {
            let expected = apply_atomic_to_oracle(&mut oracle, batch);
            let got = tree.atomic(|tx| {
                for op in batch {
                    match op {
                        Op::Put(k, v) => tx.put(k, v),
                        Op::Delete(k) => tx.delete(k),
                        Op::Rename(s, d, force) => tx.rename(s, d, *force),
                    }
                }
            });
            match (got, expected) {
                (Ok(true), Ok(())) => {}
                (Err(holt::Error::NotFound), Err(OracleAtomicErr::NotFound)) => {}
                (Err(holt::Error::DstExists), Err(OracleAtomicErr::DstExists)) => {}
                (other, expected) => {
                    prop_assert!(
                        false,
                        "atomic result mismatch: tree={other:?}, oracle={expected:?}",
                    );
                }
            }
            check(&tree, &oracle);
        }
    }

    /// `Tree::range` enumeration must match a `BTreeMap` oracle:
    /// same keys in the same lex order, same values.
    #[test]
    fn range_iteration_matches_oracle(
        ops in vec(op_strategy(), 1..=200),
    ) {
        use std::collections::BTreeMap;

        let tree = Tree::open(TreeConfig::memory()).unwrap();
        let oracle_map = apply(&tree, &ops);
        let oracle: BTreeMap<Vec<u8>, Vec<u8>> = oracle_map.into_iter().collect();

        // Full enumeration.
        let actual = collect_kv(tree.range());
        let expected: Vec<(Vec<u8>, Vec<u8>)> = oracle
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        prop_assert_eq!(&actual, &expected);
    }

    /// `Tree::scan_prefix(p)` must match the BTreeMap-oracle's
    /// prefix range.
    #[test]
    fn scan_prefix_matches_oracle(
        ops in vec(op_strategy(), 1..=100),
        prefix in prop::collection::vec(prop::sample::select(vec![b'a', b'b', b'/', b'0']), 0..=3),
    ) {
        use std::collections::BTreeMap;

        let tree = Tree::open(TreeConfig::memory()).unwrap();
        let oracle_map = apply(&tree, &ops);
        let oracle: BTreeMap<Vec<u8>, Vec<u8>> = oracle_map.into_iter().collect();

        let actual = collect_kv(tree.scan_prefix(&prefix));
        let expected: Vec<(Vec<u8>, Vec<u8>)> = oracle
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        prop_assert_eq!(&actual, &expected);
    }
}
