#![no_main]

use std::collections::{BTreeMap, BTreeSet};

use arbitrary::{Arbitrary, Result as ArbitraryResult, Unstructured};
use holt::{KeyRangeEntry, KeyRangeEntryRef, RangeEntry, Tree, TreeConfig, View, DB};
use libfuzzer_sys::fuzz_target;

const TREE_NAMES: [&str; 4] = ["objects", "inodes", "locks", "sessions"];
const MAX_OPS: usize = 80;
const MAX_BATCH_OPS: usize = 12;
const KEYSPACE: u8 = 32;
const DIRS: u8 = 8;
const PREFIXES: u8 = DIRS + 2;
const DELIMITERS: [u8; 6] = [b'/', b':', b'|', b'#', b'@', b'\\'];

#[derive(Clone, Debug, PartialEq, Eq)]
enum TreeModel {
    Missing,
    Live(BTreeMap<Vec<u8>, Vec<u8>>),
    Dropping,
}

#[derive(Debug)]
struct Ops(Vec<Op>);

#[derive(Debug)]
enum Op {
    Create { tree: u8 },
    Drop { tree: u8 },
    Put { tree: u8, key: u8, value: u8 },
    Delete { tree: u8, key: u8 },
    Get { tree: u8, key: u8 },
    RangePrefix { tree: u8, dir: u8 },
    KeyScanDelimiter { tree: u8, dir: u8, delimiter: u8 },
    PrefixEmpty { tree: u8, dir: u8 },
    ViewPrefix { tree: u8, dir: u8 },
    Checkpoint,
    Reopen,
    Atomic(Vec<AtomicOp>),
}

#[derive(Debug)]
enum AtomicOp {
    Put {
        tree: u8,
        key: u8,
        value: u8,
    },
    Delete {
        tree: u8,
        key: u8,
    },
    PutIfAbsent {
        tree: u8,
        key: u8,
        value: u8,
    },
    AssertAbsent {
        tree: u8,
        key: u8,
    },
    AssertPrefixEmpty {
        tree: u8,
        dir: u8,
    },
    Rename {
        tree: u8,
        src: u8,
        dst: u8,
        force: bool,
    },
}

impl AtomicOp {
    fn tree(&self) -> u8 {
        match *self {
            Self::Put { tree, .. }
            | Self::Delete { tree, .. }
            | Self::PutIfAbsent { tree, .. }
            | Self::AssertAbsent { tree, .. }
            | Self::AssertPrefixEmpty { tree, .. }
            | Self::Rename { tree, .. } => tree,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelErr {
    GuardFailed,
    TreeNotFound,
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
        Ok(match u.int_in_range(0..=11u8)? {
            0 => Self::Create { tree: tree_id(u)? },
            1 => Self::Drop { tree: tree_id(u)? },
            2 => Self::Put {
                tree: tree_id(u)?,
                key: key_id(u)?,
                value: value_id(u)?,
            },
            3 => Self::Delete {
                tree: tree_id(u)?,
                key: key_id(u)?,
            },
            4 => Self::Get {
                tree: tree_id(u)?,
                key: key_id(u)?,
            },
            5 => Self::RangePrefix {
                tree: tree_id(u)?,
                dir: dir_id(u)?,
            },
            6 => Self::KeyScanDelimiter {
                tree: tree_id(u)?,
                dir: dir_id(u)?,
                delimiter: delimiter_id(u)?,
            },
            7 => Self::PrefixEmpty {
                tree: tree_id(u)?,
                dir: dir_id(u)?,
            },
            8 => Self::ViewPrefix {
                tree: tree_id(u)?,
                dir: dir_id(u)?,
            },
            9 => Self::Checkpoint,
            10 => Self::Reopen,
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
        Ok(match u.int_in_range(0..=5u8)? {
            0 => Self::Put {
                tree: tree_id(u)?,
                key: key_id(u)?,
                value: value_id(u)?,
            },
            1 => Self::Delete {
                tree: tree_id(u)?,
                key: key_id(u)?,
            },
            2 => Self::PutIfAbsent {
                tree: tree_id(u)?,
                key: key_id(u)?,
                value: value_id(u)?,
            },
            3 => Self::AssertAbsent {
                tree: tree_id(u)?,
                key: key_id(u)?,
            },
            4 => Self::AssertPrefixEmpty {
                tree: tree_id(u)?,
                dir: dir_id(u)?,
            },
            _ => Self::Rename {
                tree: tree_id(u)?,
                src: key_id(u)?,
                dst: key_id(u)?,
                force: bool::arbitrary(u)?,
            },
        })
    }
}

fn tree_id(u: &mut Unstructured<'_>) -> ArbitraryResult<u8> {
    u.int_in_range(0..=TREE_NAMES.len() as u8 - 1)
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

fn tree_name(id: u8) -> &'static str {
    TREE_NAMES[id as usize % TREE_NAMES.len()]
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
    vec![id, id.wrapping_mul(7), id ^ 0xA5]
}

fn model_atomic(model: &[TreeModel], ops: &[AtomicOp]) -> Result<Vec<TreeModel>, ModelErr> {
    let mut staged = model.to_vec();

    for op in ops {
        live_tree(&staged, op.tree())?;
    }

    for op in ops {
        match *op {
            AtomicOp::Put {
                tree,
                key: id,
                value: v,
            } => {
                live_tree_mut(&mut staged, tree)?.insert(key(id), value(v));
            }
            AtomicOp::Delete { tree, key: id } => {
                live_tree_mut(&mut staged, tree)?.remove(&key(id));
            }
            AtomicOp::PutIfAbsent {
                tree,
                key: id,
                value: v,
            } => {
                let tree = live_tree_mut(&mut staged, tree)?;
                let key = key(id);
                if tree.contains_key(&key) {
                    return Err(ModelErr::GuardFailed);
                }
                tree.insert(key, value(v));
            }
            AtomicOp::AssertAbsent { tree, key: id } => {
                if live_tree(&staged, tree)?.contains_key(&key(id)) {
                    return Err(ModelErr::GuardFailed);
                }
            }
            AtomicOp::AssertPrefixEmpty { tree, dir } => {
                let prefix = prefix(dir);
                if live_tree(&staged, tree)?
                    .keys()
                    .any(|key| key.starts_with(&prefix))
                {
                    return Err(ModelErr::GuardFailed);
                }
            }
            AtomicOp::Rename {
                tree,
                src,
                dst,
                force,
            } => {
                let tree = live_tree_mut(&mut staged, tree)?;
                let src = key(src);
                let dst = key(dst);
                let Some(value) = tree.get(&src).cloned() else {
                    return Err(ModelErr::NotFound);
                };
                if src == dst {
                    continue;
                }
                if !force && tree.contains_key(&dst) {
                    return Err(ModelErr::DstExists);
                }
                tree.remove(&src);
                tree.insert(dst, value);
            }
        }
    }
    Ok(staged)
}

fn apply_db_atomic(db: &DB, ops: &[AtomicOp]) -> Result<bool, holt::Error> {
    db.atomic(|batch| {
        for op in ops {
            match *op {
                AtomicOp::Put {
                    tree,
                    key: id,
                    value: v,
                } => batch.put(tree_name(tree), &key(id), &value(v)),
                AtomicOp::Delete { tree, key: id } => batch.delete(tree_name(tree), &key(id)),
                AtomicOp::PutIfAbsent {
                    tree,
                    key: id,
                    value: v,
                } => batch.put_if_absent(tree_name(tree), &key(id), &value(v)),
                AtomicOp::AssertAbsent { tree, key: id } => {
                    batch.assert_absent(tree_name(tree), &key(id));
                }
                AtomicOp::AssertPrefixEmpty { tree, dir } => {
                    batch.assert_prefix_empty(tree_name(tree), &prefix(dir));
                }
                AtomicOp::Rename {
                    tree,
                    src,
                    dst,
                    force,
                } => batch.rename(tree_name(tree), &key(src), &key(dst), force),
            }
        }
    })
}

fn live_tree(model: &[TreeModel], id: u8) -> Result<&BTreeMap<Vec<u8>, Vec<u8>>, ModelErr> {
    match &model[id as usize % model.len()] {
        TreeModel::Live(tree) => Ok(tree),
        TreeModel::Missing | TreeModel::Dropping => Err(ModelErr::TreeNotFound),
    }
}

fn live_tree_mut(
    model: &mut [TreeModel],
    id: u8,
) -> Result<&mut BTreeMap<Vec<u8>, Vec<u8>>, ModelErr> {
    let len = model.len();
    match &mut model[id as usize % len] {
        TreeModel::Live(tree) => Ok(tree),
        TreeModel::Missing | TreeModel::Dropping => Err(ModelErr::TreeNotFound),
    }
}

fn create_model_tree(model: &mut [TreeModel], tree: u8) -> Result<(), ModelErr> {
    let slot = &mut model[tree as usize % TREE_NAMES.len()];
    match slot {
        TreeModel::Missing => {
            *slot = TreeModel::Live(BTreeMap::new());
            Ok(())
        }
        TreeModel::Live(_) | TreeModel::Dropping => Err(ModelErr::GuardFailed),
    }
}

fn drop_model_tree(model: &mut [TreeModel], tree: u8) -> Result<(), ModelErr> {
    let slot = &mut model[tree as usize % TREE_NAMES.len()];
    match slot {
        TreeModel::Live(_) => {
            *slot = TreeModel::Dropping;
            Ok(())
        }
        TreeModel::Missing | TreeModel::Dropping => Err(ModelErr::TreeNotFound),
    }
}

fn checkpoint_model(model: &mut [TreeModel]) {
    for tree in model {
        if matches!(tree, TreeModel::Dropping) {
            *tree = TreeModel::Missing;
        }
    }
}

fn assert_db_matches_model(db: &DB, model: &[TreeModel]) {
    let mut expected_names = Vec::new();
    for (idx, state) in model.iter().enumerate() {
        let name = TREE_NAMES[idx];
        match state {
            TreeModel::Live(tree) => {
                expected_names.push(name.to_owned());
                let handle = db.open_tree(name).unwrap();
                assert_tree_matches_model(&handle, tree);
            }
            TreeModel::Missing | TreeModel::Dropping => {
                assert!(matches!(
                    db.open_tree(name),
                    Err(holt::Error::TreeNotFound { .. })
                ));
            }
        }
    }
    expected_names.sort();
    assert_eq!(db.list_trees().unwrap(), expected_names);
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

fn assert_prefix_matches_model(db: &DB, model: &[TreeModel], tree: u8, dir: u8) {
    let name = tree_name(tree);
    match live_tree(model, tree) {
        Ok(model_tree) => {
            let handle = db.open_tree(name).unwrap();
            let prefix = prefix(dir);
            let expected: Vec<_> = model_tree
                .iter()
                .filter(|(key, _)| key.starts_with(&prefix))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            let got: Vec<_> = handle
                .scan(&prefix)
                .into_iter()
                .map(|entry| match entry.unwrap() {
                    RangeEntry::Key {
                        key,
                        value,
                        version,
                    } => {
                        assert_eq!(handle.get_record(&key).unwrap().unwrap().version, version);
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
        Err(ModelErr::TreeNotFound) => {
            assert!(matches!(
                db.open_tree(name),
                Err(holt::Error::TreeNotFound { .. })
            ));
        }
        Err(e) => panic!("unexpected model error: {e:?}"),
    }
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

fn assert_key_scan_matches_model(
    db: &DB,
    model: &[TreeModel],
    tree: u8,
    dir: u8,
    delimiter: u8,
) {
    let name = tree_name(tree);
    match live_tree(model, tree) {
        Ok(model_tree) => {
            let handle = db.open_tree(name).unwrap();
            let prefix = prefix(dir);
            let expected = expected_key_entries(model_tree, dir, Some(delimiter));
            let got: Vec<_> = handle
                .scan_keys(&prefix)
                .delimiter(delimiter)
                .into_iter()
                .map(|entry| match entry.unwrap() {
                    KeyRangeEntry::Key { key, version } => {
                        assert_eq!(handle.get_record(&key).unwrap().unwrap().version, version);
                        ExpectedKeyEntry::Key(key)
                    }
                    KeyRangeEntry::CommonPrefix(prefix) => ExpectedKeyEntry::CommonPrefix(prefix),
                    _ => panic!("KeyRangeEntry got a new variant"),
                })
                .collect();
            assert_eq!(got, expected);

            let mut visited = Vec::new();
            handle
                .scan_keys(&prefix)
                .delimiter(delimiter)
                .visit(usize::MAX, |entry| {
                    visited.push(match entry {
                        KeyRangeEntryRef::Key { key, version } => {
                            assert_eq!(handle.get_record(key)?.unwrap().version, version);
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
        Err(ModelErr::TreeNotFound) => {
            assert!(matches!(
                db.open_tree(name),
                Err(holt::Error::TreeNotFound { .. })
            ));
        }
        Err(e) => panic!("unexpected model error: {e:?}"),
    }
}

fn assert_prefix_empty_matches_model(db: &DB, model: &[TreeModel], tree: u8, dir: u8) {
    let name = tree_name(tree);
    match live_tree(model, tree) {
        Ok(model_tree) => {
            let handle = db.open_tree(name).unwrap();
            let prefix = prefix(dir);
            let expected = !model_tree.keys().any(|key| key.starts_with(&prefix));
            assert_eq!(handle.is_prefix_empty(&prefix).unwrap(), expected);
        }
        Err(ModelErr::TreeNotFound) => {
            assert!(matches!(
                db.open_tree(name),
                Err(holt::Error::TreeNotFound { .. })
            ));
        }
        Err(e) => panic!("unexpected model error: {e:?}"),
    }
}

fn assert_db_view_matches_model(db: &DB, model: &[TreeModel], tree: u8, dir: u8) {
    let name = tree_name(tree);
    let prefix = prefix(dir);
    match live_tree(model, tree) {
        Ok(model_tree) => {
            let scopes = [(name, prefix.as_slice())];
            db.view(&scopes, |view| {
                let tree_view = view.tree(name).unwrap();
                assert_view_prefix_matches_model(tree_view, model_tree, dir);
                Ok(())
            })
            .unwrap();
        }
        Err(ModelErr::TreeNotFound) => {
            let scopes = [(name, prefix.as_slice())];
            assert!(matches!(
                db.view(&scopes, |_| Ok(())),
                Err(holt::Error::TreeNotFound { .. })
            ));
        }
        Err(e) => panic!("unexpected model error: {e:?}"),
    }
}

fn assert_view_prefix_matches_model(view: &View, model: &BTreeMap<Vec<u8>, Vec<u8>>, dir: u8) {
    let prefix = prefix(dir);
    let expected_records: Vec<_> = model
        .iter()
        .filter(|(key, _)| key.starts_with(&prefix))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let got_records: Vec<_> = view
        .scan(&prefix)
        .unwrap()
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
                panic!("view scan without delimiter returned prefix {prefix:?}");
            }
            _ => panic!("RangeEntry got a new variant"),
        })
        .collect();
    assert_eq!(got_records, expected_records);

    let expected_keys = expected_key_entries(model, dir, Some(b'/'));
    let got_keys: Vec<_> = view
        .scan_keys(&prefix)
        .unwrap()
        .delimiter(b'/')
        .into_iter()
        .map(|entry| match entry.unwrap() {
            KeyRangeEntry::Key { key, .. } => ExpectedKeyEntry::Key(key),
            KeyRangeEntry::CommonPrefix(prefix) => ExpectedKeyEntry::CommonPrefix(prefix),
            _ => panic!("KeyRangeEntry got a new variant"),
        })
        .collect();
    assert_eq!(got_keys, expected_keys);
}

fuzz_target!(|ops: Ops| {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = TreeConfig::new(dir.path());
    cfg.durability = holt::Durability::Wal { sync: true };
    cfg.checkpoint.enabled = false;

    let mut db = DB::open(cfg.clone()).unwrap();
    let mut model = vec![TreeModel::Missing; TREE_NAMES.len()];

    for op in ops.0 {
        match op {
            Op::Create { tree } => {
                let expected = create_model_tree(&mut model, tree);
                let got = db.create_tree(tree_name(tree));
                match (got, expected) {
                    (Ok(_), Ok(())) => {}
                    (Err(holt::Error::TreeExists { .. }), Err(ModelErr::GuardFailed)) => {}
                    (got, expected) => {
                        panic!("create mismatch: db={got:?}, model={expected:?}")
                    }
                }
            }
            Op::Drop { tree } => {
                let expected = drop_model_tree(&mut model, tree);
                let got = db.drop_tree(tree_name(tree));
                match (got, expected) {
                    (Ok(()), Ok(())) => {}
                    (Err(holt::Error::TreeNotFound { .. }), Err(ModelErr::TreeNotFound)) => {}
                    (got, expected) => panic!("drop mismatch: db={got:?}, model={expected:?}"),
                }
            }
            Op::Put {
                tree,
                key: id,
                value: v,
            } => match live_tree_mut(&mut model, tree) {
                Ok(model_tree) => {
                    db.open_tree(tree_name(tree))
                        .unwrap()
                        .put(&key(id), &value(v))
                        .unwrap();
                    model_tree.insert(key(id), value(v));
                }
                Err(ModelErr::TreeNotFound) => {
                    assert!(matches!(
                        db.open_tree(tree_name(tree)),
                        Err(holt::Error::TreeNotFound { .. })
                    ));
                }
                Err(e) => panic!("unexpected model error: {e:?}"),
            },
            Op::Delete { tree, key: id } => match live_tree_mut(&mut model, tree) {
                Ok(model_tree) => {
                    let deleted = db
                        .open_tree(tree_name(tree))
                        .unwrap()
                        .delete(&key(id))
                        .unwrap();
                    assert_eq!(deleted, model_tree.remove(&key(id)).is_some());
                }
                Err(ModelErr::TreeNotFound) => {
                    assert!(matches!(
                        db.open_tree(tree_name(tree)),
                        Err(holt::Error::TreeNotFound { .. })
                    ));
                }
                Err(e) => panic!("unexpected model error: {e:?}"),
            },
            Op::Get { tree, key: id } => match live_tree(&model, tree) {
                Ok(model_tree) => {
                    let got = db
                        .open_tree(tree_name(tree))
                        .unwrap()
                        .get(&key(id))
                        .unwrap();
                    assert_eq!(got, model_tree.get(&key(id)).cloned());
                }
                Err(ModelErr::TreeNotFound) => {
                    assert!(matches!(
                        db.open_tree(tree_name(tree)),
                        Err(holt::Error::TreeNotFound { .. })
                    ));
                }
                Err(e) => panic!("unexpected model error: {e:?}"),
            },
            Op::RangePrefix { tree, dir } => assert_prefix_matches_model(&db, &model, tree, dir),
            Op::KeyScanDelimiter {
                tree,
                dir,
                delimiter,
            } => {
                assert_key_scan_matches_model(&db, &model, tree, dir, self::delimiter(delimiter));
            }
            Op::PrefixEmpty { tree, dir } => {
                assert_prefix_empty_matches_model(&db, &model, tree, dir);
            }
            Op::ViewPrefix { tree, dir } => assert_db_view_matches_model(&db, &model, tree, dir),
            Op::Checkpoint => {
                db.checkpoint().unwrap();
                checkpoint_model(&mut model);
            }
            Op::Reopen => {
                drop(db);
                db = DB::open(cfg.clone()).unwrap();
            }
            Op::Atomic(batch) => {
                let expected = model_atomic(&model, &batch);
                let got = apply_db_atomic(&db, &batch);
                match (got, expected) {
                    (Ok(true), Ok(staged)) => model = staged,
                    (Ok(false), Err(ModelErr::GuardFailed)) => {}
                    (Err(holt::Error::TreeNotFound { .. }), Err(ModelErr::TreeNotFound)) => {}
                    (Err(holt::Error::NotFound), Err(ModelErr::NotFound)) => {}
                    (Err(holt::Error::DstExists), Err(ModelErr::DstExists)) => {}
                    (got, expected) => panic!(
                        "atomic result mismatch: db={got:?}, model={expected:?}, batch={batch:?}",
                    ),
                }
            }
        }
        assert_db_matches_model(&db, &model);
    }
});
