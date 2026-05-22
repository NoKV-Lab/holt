#![no_main]

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::{Arbitrary, Result as ArbitraryResult, Unstructured};
use holt::{RangeEntry, RecordVersion, Tree, TreeConfig, WalCommit};
use libfuzzer_sys::fuzz_target;

const MAX_OPS: usize = 96;
const MAX_BATCH_OPS: usize = 12;
const KEYSPACE: u8 = 32;
const DIRS: u8 = 8;

#[derive(Debug)]
struct Ops(Vec<Op>);

#[derive(Debug)]
enum Op {
    Put { key: u8, value: u8 },
    Delete { key: u8 },
    Get { key: u8 },
    RangePrefix { dir: u8 },
    Checkpoint,
    Reopen,
    Atomic(Vec<AtomicOp>),
}

#[derive(Debug)]
enum AtomicOp {
    Put { key: u8, value: u8 },
    Delete { key: u8 },
    PutIfAbsent { key: u8, value: u8 },
    CompareAndPutCurrent { key: u8, value: u8 },
    DeleteIfCurrent { key: u8 },
    AssertCurrent { key: u8 },
    AssertStale { key: u8 },
    AssertPrefixEmpty { dir: u8 },
    Rename { src: u8, dst: u8, force: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelErr {
    GuardFailed,
    NotFound,
    DstExists,
}

impl<'a> Arbitrary<'a> for Ops {
    fn arbitrary(u: &mut Unstructured<'a>) -> ArbitraryResult<Self> {
        let len = u.int_in_range(0..=MAX_OPS)?;
        let mut ops = Vec::with_capacity(len);
        for _ in 0..len {
            ops.push(Op::arbitrary(u)?);
        }
        Ok(Self(ops))
    }
}

impl<'a> Arbitrary<'a> for Op {
    fn arbitrary(u: &mut Unstructured<'a>) -> ArbitraryResult<Self> {
        Ok(match u.int_in_range(0..=5u8)? {
            0 => Self::Put {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            1 => Self::Delete { key: key_id(u)? },
            2 => Self::Get { key: key_id(u)? },
            3 => Self::RangePrefix { dir: dir_id(u)? },
            4 => {
                if bool::arbitrary(u)? {
                    Self::Checkpoint
                } else {
                    Self::Reopen
                }
            }
            _ => {
                let len = u.int_in_range(0..=MAX_BATCH_OPS)?;
                let mut batch = Vec::with_capacity(len);
                for _ in 0..len {
                    batch.push(AtomicOp::arbitrary(u)?);
                }
                Self::Atomic(batch)
            }
        })
    }
}

impl<'a> Arbitrary<'a> for AtomicOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> ArbitraryResult<Self> {
        Ok(match u.int_in_range(0..=8u8)? {
            0 => Self::Put {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            1 => Self::Delete { key: key_id(u)? },
            2 => Self::PutIfAbsent {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            3 => Self::CompareAndPutCurrent {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            4 => Self::DeleteIfCurrent { key: key_id(u)? },
            5 => Self::AssertCurrent { key: key_id(u)? },
            6 => Self::AssertStale { key: key_id(u)? },
            7 => Self::AssertPrefixEmpty { dir: dir_id(u)? },
            _ => Self::Rename {
                src: key_id(u)?,
                dst: key_id(u)?,
                force: bool::arbitrary(u)?,
            },
        })
    }
}

fn key_id(u: &mut Unstructured<'_>) -> ArbitraryResult<u8> {
    u.int_in_range(0..=KEYSPACE - 1)
}

fn value_id(u: &mut Unstructured<'_>) -> ArbitraryResult<u8> {
    u.int_in_range(0..=63)
}

fn dir_id(u: &mut Unstructured<'_>) -> ArbitraryResult<u8> {
    u.int_in_range(0..=DIRS - 1)
}

fn key(id: u8) -> Vec<u8> {
    format!("dir/{:02}/file/{:02}", id % DIRS, id % KEYSPACE).into_bytes()
}

fn prefix(dir: u8) -> Vec<u8> {
    format!("dir/{:02}/", dir % DIRS).into_bytes()
}

fn value(id: u8) -> Vec<u8> {
    vec![id, id.wrapping_mul(3), id ^ 0x5A]
}

fn current_version(tree: &Tree, key: &[u8]) -> RecordVersion {
    tree.get_version(key)
        .unwrap()
        .unwrap_or_else(|| RecordVersion::from_raw(u64::MAX))
}

fn model_atomic(
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    ops: &[AtomicOp],
) -> Result<BTreeMap<Vec<u8>, Vec<u8>>, ModelErr> {
    let mut staged = model.clone();
    let mut touched = BTreeSet::new();

    for op in ops {
        match *op {
            AtomicOp::Put { key: id, value: v } => {
                let key = key(id);
                staged.insert(key.clone(), value(v));
                touched.insert(key);
            }
            AtomicOp::Delete { key: id } => {
                let key = key(id);
                staged.remove(&key);
                touched.insert(key);
            }
            AtomicOp::PutIfAbsent { key: id, value: v } => {
                let key = key(id);
                if staged.contains_key(&key) {
                    return Err(ModelErr::GuardFailed);
                }
                staged.insert(key.clone(), value(v));
                touched.insert(key);
            }
            AtomicOp::CompareAndPutCurrent { key: id, value: v } => {
                let key = key(id);
                if !model.contains_key(&key) || touched.contains(&key) {
                    return Err(ModelErr::GuardFailed);
                }
                staged.insert(key.clone(), value(v));
                touched.insert(key);
            }
            AtomicOp::DeleteIfCurrent { key: id } => {
                let key = key(id);
                if !model.contains_key(&key) || touched.contains(&key) {
                    return Err(ModelErr::GuardFailed);
                }
                staged.remove(&key);
                touched.insert(key);
            }
            AtomicOp::AssertCurrent { key: id } => {
                let key = key(id);
                if !model.contains_key(&key) || touched.contains(&key) {
                    return Err(ModelErr::GuardFailed);
                }
            }
            AtomicOp::AssertStale { .. } => return Err(ModelErr::GuardFailed),
            AtomicOp::AssertPrefixEmpty { dir } => {
                let prefix = prefix(dir);
                if staged.keys().any(|key| key.starts_with(&prefix)) {
                    return Err(ModelErr::GuardFailed);
                }
            }
            AtomicOp::Rename { src, dst, force } => {
                let src = key(src);
                let dst = key(dst);
                let Some(value) = staged.get(&src).cloned() else {
                    return Err(ModelErr::NotFound);
                };
                if src == dst {
                    continue;
                }
                if !force && staged.contains_key(&dst) {
                    return Err(ModelErr::DstExists);
                }
                staged.remove(&src);
                staged.insert(dst.clone(), value);
                touched.insert(src);
                touched.insert(dst);
            }
        }
    }
    Ok(staged)
}

fn apply_atomic(tree: &Tree, ops: &[AtomicOp]) -> Result<bool, holt::Error> {
    tree.atomic(|batch| {
        for op in ops {
            match *op {
                AtomicOp::Put { key: id, value: v } => batch.put(&key(id), &value(v)),
                AtomicOp::Delete { key: id } => batch.delete(&key(id)),
                AtomicOp::PutIfAbsent { key: id, value: v } => {
                    batch.put_if_absent(&key(id), &value(v));
                }
                AtomicOp::CompareAndPutCurrent { key: id, value: v } => {
                    let key = key(id);
                    batch.compare_and_put(&key, current_version(tree, &key), &value(v));
                }
                AtomicOp::DeleteIfCurrent { key: id } => {
                    let key = key(id);
                    batch.delete_if_version(&key, current_version(tree, &key));
                }
                AtomicOp::AssertCurrent { key: id } => {
                    let key = key(id);
                    batch.assert_version(&key, current_version(tree, &key));
                }
                AtomicOp::AssertStale { key: id } => {
                    batch.assert_version(&key(id), RecordVersion::from_raw(u64::MAX));
                }
                AtomicOp::AssertPrefixEmpty { dir } => batch.assert_prefix_empty(&prefix(dir)),
                AtomicOp::Rename { src, dst, force } => batch.rename(&key(src), &key(dst), force),
            }
        }
    })
}

fn assert_tree_matches_model(tree: &Tree, model: &BTreeMap<Vec<u8>, Vec<u8>>) {
    for (key, value) in model {
        assert_eq!(tree.get(key).unwrap().as_deref(), Some(value.as_slice()));
    }

    let got: BTreeMap<Vec<u8>, Vec<u8>> = tree
        .range()
        .into_iter()
        .map(|entry| match entry.unwrap() {
            RangeEntry::Key {
                key,
                value,
                version,
            } => {
                assert_eq!(tree.get_record(&key).unwrap().unwrap().version, version);
                (key, value)
            }
            RangeEntry::CommonPrefix(prefix) => {
                panic!("full range without delimiter returned prefix {prefix:?}");
            }
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect();
    assert_eq!(got, *model);
}

fn assert_prefix_matches_model(tree: &Tree, model: &BTreeMap<Vec<u8>, Vec<u8>>, dir: u8) {
    let prefix = prefix(dir);
    let expected: Vec<_> = model
        .iter()
        .filter(|(key, _)| key.starts_with(&prefix))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let got: Vec<_> = tree
        .scan_prefix(&prefix)
        .into_iter()
        .map(|entry| match entry.unwrap() {
            RangeEntry::Key {
                key,
                value,
                version,
            } => {
                assert_eq!(tree.get_record(&key).unwrap().unwrap().version, version);
                (key, value)
            }
            RangeEntry::CommonPrefix(prefix) => {
                panic!("prefix scan without delimiter returned prefix {prefix:?}");
            }
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect();
    assert_eq!(got, expected);
}

fuzz_target!(|ops: Ops| {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = TreeConfig::new(dir.path());
    cfg.wal_commit = WalCommit::Sync;

    let mut tree = Tree::open(cfg.clone()).unwrap();
    let mut model = BTreeMap::new();

    for op in ops.0 {
        match op {
            Op::Put { key: id, value: v } => {
                tree.put(&key(id), &value(v)).unwrap();
                model.insert(key(id), value(v));
            }
            Op::Delete { key: id } => {
                let deleted = tree.delete(&key(id)).unwrap();
                assert_eq!(deleted, model.remove(&key(id)).is_some());
            }
            Op::Get { key: id } => {
                assert_eq!(tree.get(&key(id)).unwrap(), model.get(&key(id)).cloned());
            }
            Op::RangePrefix { dir } => assert_prefix_matches_model(&tree, &model, dir),
            Op::Checkpoint => tree.checkpoint().unwrap(),
            Op::Reopen => {
                drop(tree);
                tree = Tree::open(cfg.clone()).unwrap();
            }
            Op::Atomic(batch) => {
                let expected = model_atomic(&model, &batch);
                let got = apply_atomic(&tree, &batch);
                match (got, expected) {
                    (Ok(true), Ok(staged)) => model = staged,
                    (Ok(false), Err(ModelErr::GuardFailed)) => {}
                    (Err(holt::Error::NotFound), Err(ModelErr::NotFound)) => {}
                    (Err(holt::Error::DstExists), Err(ModelErr::DstExists)) => {}
                    (got, expected) => panic!(
                        "atomic result mismatch: tree={got:?}, model={expected:?}, batch={batch:?}",
                    ),
                }
            }
        }
        assert_tree_matches_model(&tree, &model);
    }
});
