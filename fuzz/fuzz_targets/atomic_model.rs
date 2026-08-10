#![no_main]

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::{Arbitrary, Result as ArbitraryResult, Unstructured};
use holt::{KeyRangeEntry, KeyRangeEntryRef, RangeEntry, RecordVersion, Tree, TreeConfig};
use libfuzzer_sys::fuzz_target;

const MAX_OPS: usize = 96;
const MAX_BATCH_OPS: usize = 12;
const KEYSPACE: u8 = 32;
const DIRS: u8 = 8;
const PREFIXES: u8 = DIRS + 2;
const DELIMITERS: [u8; 6] = [b'/', b':', b'|', b'#', b'@', b'\\'];

#[derive(Debug)]
struct Ops(Vec<Op>);

#[derive(Debug)]
enum Op {
    Put { key: u8, value: u8 },
    Delete { key: u8 },
    Get { key: u8 },
    RangePrefix { dir: u8 },
    RangeDelimiter { dir: u8, delimiter: u8 },
    KeyScanPrefix { dir: u8 },
    KeyScanDelimiter { dir: u8, delimiter: u8 },
    PrefixEmpty { dir: u8 },
    ViewPrefix { dir: u8 },
    Checkpoint,
    Reopen,
    Atomic(Vec<AtomicOp>),
}

#[derive(Debug)]
enum AtomicOp {
    Put { key: u8, value: u8 },
    Delete { key: u8 },
    PutIfAbsent { key: u8, value: u8 },
    AssertAbsent { key: u8 },
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
        Ok(match u.int_in_range(0..=10u8)? {
            0 => Self::Put {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            1 => Self::Delete { key: key_id(u)? },
            2 => Self::Get { key: key_id(u)? },
            3 => Self::RangePrefix { dir: dir_id(u)? },
            4 => Self::RangeDelimiter {
                dir: dir_id(u)?,
                delimiter: delimiter_id(u)?,
            },
            5 => Self::KeyScanPrefix { dir: dir_id(u)? },
            6 => Self::KeyScanDelimiter {
                dir: dir_id(u)?,
                delimiter: delimiter_id(u)?,
            },
            7 => Self::PrefixEmpty { dir: dir_id(u)? },
            8 => Self::ViewPrefix { dir: dir_id(u)? },
            9 => {
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
        Ok(match u.int_in_range(0..=9u8)? {
            0 => Self::Put {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            1 => Self::Delete { key: key_id(u)? },
            2 => Self::PutIfAbsent {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            3 => Self::AssertAbsent { key: key_id(u)? },
            4 => Self::CompareAndPutCurrent {
                key: key_id(u)?,
                value: value_id(u)?,
            },
            5 => Self::DeleteIfCurrent { key: key_id(u)? },
            6 => Self::AssertCurrent { key: key_id(u)? },
            7 => Self::AssertStale { key: key_id(u)? },
            8 => Self::AssertPrefixEmpty { dir: dir_id(u)? },
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
    u.int_in_range(0..=PREFIXES - 1)
}

fn delimiter_id(u: &mut Unstructured<'_>) -> ArbitraryResult<u8> {
    u.int_in_range(0..=DELIMITERS.len() as u8 - 1)
}

fn delimiter(id: u8) -> u8 {
    DELIMITERS[id as usize % DELIMITERS.len()]
}

fn key(id: u8) -> Vec<u8> {
    format!(
        "dir/{:02}/tenant:{:02}|bucket#{:02}@shard\\file/{:02}",
        id % DIRS,
        id % 4,
        id % 8,
        id % KEYSPACE
    )
    .into_bytes()
}

fn prefix(dir: u8) -> Vec<u8> {
    if dir < DIRS {
        format!("dir/{:02}/", dir).into_bytes()
    } else {
        format!("missing/{:02}/", dir - DIRS).into_bytes()
    }
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
            AtomicOp::AssertAbsent { key: id } => {
                if staged.contains_key(&key(id)) {
                    return Err(ModelErr::GuardFailed);
                }
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
                AtomicOp::AssertAbsent { key: id } => batch.assert_absent(&key(id)),
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
        .scan(&prefix)
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExpectedKeyEntry {
    Key(Vec<u8>),
    CommonPrefix(Vec<u8>),
}

fn expected_key_entries(
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    dir: u8,
    delimiter: Option<u8>,
) -> Vec<ExpectedKeyEntry> {
    let prefix = prefix(dir);
    let mut emitted_prefixes = BTreeSet::new();
    let mut expected = Vec::new();

    for key in model.keys().filter(|key| key.starts_with(&prefix)) {
        if let Some(delimiter) = delimiter {
            if let Some(pos) = key[prefix.len()..]
                .iter()
                .position(|byte| *byte == delimiter)
            {
                let common = key[..prefix.len() + pos + 1].to_vec();
                if emitted_prefixes.insert(common.clone()) {
                    expected.push(ExpectedKeyEntry::CommonPrefix(common));
                }
                continue;
            }
        }
        expected.push(ExpectedKeyEntry::Key(key.clone()));
    }
    expected
}

fn assert_range_delimiter_matches_model_with(
    tree: &Tree,
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    dir: u8,
    delimiter: u8,
) {
    let prefix = prefix(dir);
    let expected = expected_key_entries(model, dir, Some(delimiter));
    let got: Vec<_> = tree
        .scan(&prefix)
        .delimiter(delimiter)
        .into_iter()
        .map(|entry| match entry.unwrap() {
            RangeEntry::Key { key, version, .. } => {
                assert_eq!(tree.get_record(&key).unwrap().unwrap().version, version);
                ExpectedKeyEntry::Key(key)
            }
            RangeEntry::CommonPrefix(prefix) => ExpectedKeyEntry::CommonPrefix(prefix),
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect();
    assert_eq!(got, expected);
}

fn assert_prefix_empty_matches_model(tree: &Tree, model: &BTreeMap<Vec<u8>, Vec<u8>>, dir: u8) {
    let prefix = prefix(dir);
    let expected = !model.keys().any(|key| key.starts_with(&prefix));
    assert_eq!(tree.is_prefix_empty(&prefix).unwrap(), expected);
}

fn assert_key_scan_matches_model(
    tree: &Tree,
    model: &BTreeMap<Vec<u8>, Vec<u8>>,
    dir: u8,
    delimiter: Option<u8>,
) {
    let prefix = prefix(dir);
    let expected = expected_key_entries(model, dir, delimiter);
    let mut builder = tree.scan_keys(&prefix);
    if let Some(delimiter) = delimiter {
        builder = builder.delimiter(delimiter);
    }

    let got: Vec<_> = builder
        .into_iter()
        .map(|entry| match entry.unwrap() {
            KeyRangeEntry::Key { key, version } => {
                assert_eq!(tree.get_record(&key).unwrap().unwrap().version, version);
                ExpectedKeyEntry::Key(key)
            }
            KeyRangeEntry::CommonPrefix(prefix) => ExpectedKeyEntry::CommonPrefix(prefix),
            _ => panic!("KeyRangeEntry got a new variant"),
        })
        .collect();
    assert_eq!(got, expected);

    let mut builder = tree.scan_keys(&prefix);
    if let Some(delimiter) = delimiter {
        builder = builder.delimiter(delimiter);
    }
    let mut visited = Vec::new();
    builder
        .visit(usize::MAX, |entry| {
            visited.push(match entry {
                KeyRangeEntryRef::Key { key, version } => {
                    assert_eq!(tree.get_record(key)?.unwrap().version, version);
                    ExpectedKeyEntry::Key(key.to_vec())
                }
                KeyRangeEntryRef::CommonPrefix(prefix) => {
                    ExpectedKeyEntry::CommonPrefix(prefix.to_vec())
                }
                _ => panic!("KeyRangeEntryRef got a new variant"),
            });
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, expected);
}

fn assert_view_matches_model(tree: &Tree, model: &BTreeMap<Vec<u8>, Vec<u8>>, dir: u8) {
    let prefix = prefix(dir);
    tree.view(&prefix, |view| {
        assert_eq!(view.scope(), prefix.as_slice());
        assert!(view.get(b"outside/scope").is_err());

        let expected_records: Vec<_> = model
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let got_records: Vec<_> = view
            .range()
            .into_iter()
            .map(|entry| match entry.unwrap() {
                RangeEntry::Key {
                    key,
                    value,
                    version,
                } => {
                    assert_eq!(view.get_record(&key).unwrap().unwrap().version, version);
                    (key, value)
                }
                RangeEntry::CommonPrefix(prefix) => {
                    panic!("view range without delimiter returned prefix {prefix:?}");
                }
                _ => panic!("RangeEntry got a new variant"),
            })
            .collect();
        assert_eq!(got_records, expected_records);

        let expected_keys = expected_key_entries(model, dir, Some(b'/'));
        let got_keys: Vec<_> = view
            .scan_keys(&prefix)?
            .delimiter(b'/')
            .into_iter()
            .map(|entry| match entry.unwrap() {
                KeyRangeEntry::Key { key, version } => {
                    assert_eq!(view.get_record(&key).unwrap().unwrap().version, version);
                    ExpectedKeyEntry::Key(key)
                }
                KeyRangeEntry::CommonPrefix(prefix) => ExpectedKeyEntry::CommonPrefix(prefix),
                _ => panic!("KeyRangeEntry got a new variant"),
            })
            .collect();
        assert_eq!(got_keys, expected_keys);
        Ok(())
    })
    .unwrap();
}

fuzz_target!(|ops: Ops| {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = TreeConfig::new(dir.path());
    cfg.durability = holt::Durability::Wal { sync: true };

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
            Op::RangeDelimiter { dir, delimiter } => {
                assert_range_delimiter_matches_model_with(
                    &tree,
                    &model,
                    dir,
                    self::delimiter(delimiter),
                );
            }
            Op::KeyScanPrefix { dir } => assert_key_scan_matches_model(&tree, &model, dir, None),
            Op::KeyScanDelimiter { dir, delimiter } => {
                assert_key_scan_matches_model(&tree, &model, dir, Some(self::delimiter(delimiter)));
            }
            Op::PrefixEmpty { dir } => assert_prefix_empty_matches_model(&tree, &model, dir),
            Op::ViewPrefix { dir } => assert_view_matches_model(&tree, &model, dir),
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
