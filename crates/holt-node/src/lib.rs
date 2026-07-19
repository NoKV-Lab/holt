#![deny(unsafe_op_in_unsafe_fn)]

use holt::{
    KeyRangeEntry, RangeEntry, Tree as CoreTree, TreeBuilder, TreeConfig, DB as CoreDatabase,
};
use napi::bindgen_prelude::{BigInt, Buffer, Error, Result, Status, Uint8Array};
use napi_derive::napi;

fn js_error(error: holt::Error) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

fn delimiter(value: Option<u32>) -> Result<Option<u8>> {
    value
        .map(|value| {
            u8::try_from(value).map_err(|_| {
                Error::new(
                    Status::InvalidArg,
                    "delimiter must be an integer between 0 and 255",
                )
            })
        })
        .transpose()
}

fn into_buffer(bytes: Vec<u8>) -> Buffer {
    Buffer::from(bytes)
}

fn into_bigint(value: u64) -> BigInt {
    BigInt::from(value)
}

fn from_bigint(value: BigInt) -> Result<u64> {
    let (negative, value, lossless) = value.get_u64();
    if negative || !lossless {
        return Err(Error::new(
            Status::InvalidArg,
            "version must be a non-negative bigint that fits in uint64",
        ));
    }
    Ok(value)
}

/// Options for opening a file-backed tree or database.
#[napi(object)]
pub struct TreeOptions {
    /// Force the WAL to sync before acknowledging each write.
    pub wal_sync: Option<bool>,
}

/// Options shared by key and record scans.
#[napi(object)]
pub struct ScanOptions {
    /// Exclusive lower-bound key.
    pub start_after: Option<Uint8Array>,
    /// Optional delimiter byte used for common-prefix rollups.
    pub delimiter: Option<u32>,
}

/// A live value and its conditional-write version.
#[napi(object)]
pub struct Record {
    /// Value bytes stored at the key.
    pub value: Buffer,
    /// Version token for compare-and-put.
    pub version: BigInt,
}

/// One emitted scan entry.
#[napi(object)]
pub struct ScanEntry {
    /// `key` for a live record or `common_prefix` for a rollup.
    pub kind: String,
    /// Key or common-prefix bytes.
    pub path: Buffer,
    /// Value bytes; absent for key-only scans and common-prefix entries.
    pub value: Option<Buffer>,
    /// Conditional-write version. Zero for common-prefix entries.
    pub version: BigInt,
}

/// A Node.js handle to a Holt multi-tree database.
#[napi]
pub struct Database {
    inner: Option<CoreDatabase>,
}

#[napi]
impl Database {
    /// Open a file-backed multi-tree database.
    #[napi(factory)]
    pub fn open(path: String, options: Option<TreeOptions>) -> Result<Self> {
        let wal_sync = options
            .and_then(|options| options.wal_sync)
            .unwrap_or(false);
        let mut config = TreeConfig::new(path);
        config.durability = holt::Durability::Wal { sync: wal_sync };
        CoreDatabase::open(config)
            .map(|inner| Self { inner: Some(inner) })
            .map_err(js_error)
    }

    /// Open a volatile in-memory multi-tree database.
    #[napi(factory)]
    pub fn open_memory() -> Result<Self> {
        CoreDatabase::open(TreeConfig::memory())
            .map(|inner| Self { inner: Some(inner) })
            .map_err(js_error)
    }

    /// Explicitly release the database handle. Existing Tree handles remain
    /// usable until they are closed or their named tree is dropped.
    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
    }

    /// Create a new named tree.
    #[napi(js_name = "createTree")]
    pub fn create_tree(&self, name: String) -> Result<Tree> {
        self.core()?
            .create_tree(&name)
            .map(Tree::from_core)
            .map_err(js_error)
    }

    /// Open an existing named tree.
    #[napi(js_name = "openTree")]
    pub fn open_tree(&self, name: String) -> Result<Tree> {
        self.core()?
            .open_tree(&name)
            .map(Tree::from_core)
            .map_err(js_error)
    }

    /// Open a named tree, creating it when it does not exist.
    #[napi(js_name = "openOrCreateTree")]
    pub fn open_or_create_tree(&self, name: String) -> Result<Tree> {
        self.core()?
            .open_or_create_tree(&name)
            .map(Tree::from_core)
            .map_err(js_error)
    }

    /// List all live named trees.
    #[napi(js_name = "listTrees")]
    pub fn list_trees(&self) -> Result<Vec<String>> {
        self.core()?.list_trees().map_err(js_error)
    }

    /// Drop a named tree and fence existing handles to it.
    #[napi(js_name = "dropTree")]
    pub fn drop_tree(&self, name: String) -> Result<()> {
        self.core()?.drop_tree(&name).map_err(js_error)
    }

    /// Flush every named tree and the shared WAL to the backing store.
    #[napi]
    pub fn checkpoint(&self) -> Result<()> {
        self.core()?.checkpoint().map_err(js_error)
    }

    fn core(&self) -> Result<&CoreDatabase> {
        self.inner
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Holt database is closed"))
    }
}

/// A Node.js handle to one Holt tree.
#[napi]
pub struct Tree {
    inner: Option<CoreTree>,
}

#[napi]
impl Tree {
    fn from_core(inner: CoreTree) -> Self {
        Self { inner: Some(inner) }
    }

    /// Open a file-backed tree.
    #[napi(factory)]
    pub fn open(path: String, options: Option<TreeOptions>) -> Result<Self> {
        let wal_sync = options
            .and_then(|options| options.wal_sync)
            .unwrap_or(false);
        TreeBuilder::new(path)
            .durability(holt::Durability::Wal { sync: wal_sync })
            .open()
            .map(|inner| Self { inner: Some(inner) })
            .map_err(js_error)
    }

    /// Open a volatile in-memory tree.
    #[napi(factory)]
    pub fn open_memory() -> Result<Self> {
        TreeBuilder::new("holt-node-memory")
            .memory()
            .open()
            .map(|inner| Self { inner: Some(inner) })
            .map_err(js_error)
    }

    /// Explicitly release the native tree handle. This is idempotent.
    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
    }

    /// Return the value stored at `key`, or null when the key is absent.
    #[napi]
    pub fn get(&self, key: Uint8Array) -> Result<Option<Buffer>> {
        self.core()?
            .get(key.as_ref())
            .map(|value| value.map(into_buffer))
            .map_err(js_error)
    }

    /// Return the value and conditional-write version for `key`.
    #[napi(js_name = "getRecord")]
    pub fn get_record(&self, key: Uint8Array) -> Result<Option<Record>> {
        self.core()?
            .get_record(key.as_ref())
            .map(|record| {
                record.map(|record| Record {
                    value: into_buffer(record.value),
                    version: into_bigint(record.version.as_u64()),
                })
            })
            .map_err(js_error)
    }

    /// Insert or replace a value.
    #[napi]
    pub fn put(&self, key: Uint8Array, value: Uint8Array) -> Result<()> {
        self.core()?
            .put(key.as_ref(), value.as_ref())
            .map_err(js_error)
    }

    /// Delete a key and return whether a live record existed.
    #[napi]
    pub fn delete(&self, key: Uint8Array) -> Result<bool> {
        self.core()?.delete(key.as_ref()).map_err(js_error)
    }

    /// Compare the current version and replace the value if it matches.
    #[napi(js_name = "compareAndPut")]
    pub fn compare_and_put(
        &self,
        key: Uint8Array,
        version: BigInt,
        value: Uint8Array,
    ) -> Result<bool> {
        self.core()?
            .compare_and_put(
                key.as_ref(),
                holt::RecordVersion::from_raw(from_bigint(version)?),
                value.as_ref(),
            )
            .map_err(js_error)
    }

    /// Flush dirty frames and the WAL to the backing store.
    #[napi]
    pub fn checkpoint(&self) -> Result<()> {
        self.core()?.checkpoint().map_err(js_error)
    }

    /// Scan keys under an optional prefix.
    #[napi(js_name = "scanKeys")]
    pub fn scan_keys(
        &self,
        prefix: Option<Uint8Array>,
        options: Option<ScanOptions>,
    ) -> Result<Vec<ScanEntry>> {
        let options = options.unwrap_or(ScanOptions {
            start_after: None,
            delimiter: None,
        });
        let delimiter = delimiter(options.delimiter)?;
        let prefix = prefix.map(|prefix| prefix.to_vec()).unwrap_or_default();
        let mut builder = self.core()?.scan_keys(prefix.as_ref());
        if let Some(start_after) = options.start_after {
            builder = builder.start_after(start_after.as_ref());
        }
        if let Some(delimiter) = delimiter {
            builder = builder.delimiter(delimiter);
        }
        builder
            .into_iter()
            .map(|entry| match entry.map_err(js_error)? {
                KeyRangeEntry::Key { key, version } => Ok(ScanEntry {
                    kind: "key".to_owned(),
                    path: into_buffer(key),
                    value: None,
                    version: into_bigint(version.as_u64()),
                }),
                KeyRangeEntry::CommonPrefix(path) => Ok(ScanEntry {
                    kind: "common_prefix".to_owned(),
                    path: into_buffer(path),
                    value: None,
                    version: into_bigint(0),
                }),
                _ => Err(Error::new(
                    Status::GenericFailure,
                    "Holt returned an unsupported key scan entry",
                )),
            })
            .collect()
    }

    /// Scan records under an optional prefix.
    #[napi(js_name = "scanRecords")]
    pub fn scan_records(
        &self,
        prefix: Option<Uint8Array>,
        options: Option<ScanOptions>,
    ) -> Result<Vec<ScanEntry>> {
        let options = options.unwrap_or(ScanOptions {
            start_after: None,
            delimiter: None,
        });
        let delimiter = delimiter(options.delimiter)?;
        let prefix = prefix.map(|prefix| prefix.to_vec()).unwrap_or_default();
        let mut builder = self.core()?.scan(prefix.as_ref());
        if let Some(start_after) = options.start_after {
            builder = builder.start_after(start_after.as_ref());
        }
        if let Some(delimiter) = delimiter {
            builder = builder.delimiter(delimiter);
        }
        builder
            .into_iter()
            .map(|entry| match entry.map_err(js_error)? {
                RangeEntry::Key {
                    key,
                    value,
                    version,
                } => Ok(ScanEntry {
                    kind: "key".to_owned(),
                    path: into_buffer(key),
                    value: Some(into_buffer(value)),
                    version: into_bigint(version.as_u64()),
                }),
                RangeEntry::CommonPrefix(path) => Ok(ScanEntry {
                    kind: "common_prefix".to_owned(),
                    path: into_buffer(path),
                    value: None,
                    version: into_bigint(0),
                }),
                _ => Err(Error::new(
                    Status::GenericFailure,
                    "Holt returned an unsupported record scan entry",
                )),
            })
            .collect()
    }

    fn core(&self) -> Result<&CoreTree> {
        self.inner
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Holt tree is closed"))
    }
}
