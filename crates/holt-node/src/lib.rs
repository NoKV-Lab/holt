#![deny(unsafe_op_in_unsafe_fn)]

use holt::{
    KeyRangeEntry, RangeEntry, Tree as CoreTree, TreeBuilder, TreeConfig, DB as CoreDatabase,
};
use napi::bindgen_prelude::{
    spawn_blocking, AsyncTask, BigInt, Buffer, Error, Result, Status, ToNapiValue, TypeName,
    Uint8Array,
};
use napi::{Env, Task};
use napi_derive::napi;

fn js_error(error: holt::Error) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

fn worker_join_error(error: napi::tokio::task::JoinError) -> Error {
    Error::new(
        Status::GenericFailure,
        format!("Holt worker task failed: {error}"),
    )
}

async fn spawn_holt<T, F>(operation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> holt::Result<T> + Send + 'static,
{
    spawn_blocking(operation)
        .await
        .map_err(worker_join_error)?
        .map_err(js_error)
}

/// A libuv worker-pool task used by the generated Node-API Promise wrappers.
pub struct WorkerTask<T, J> {
    operation: Option<Box<dyn FnOnce() -> Result<T> + Send>>,
    resolver: Option<Box<dyn FnOnce(T) -> Result<J> + Send>>,
}

impl<T, J> WorkerTask<T, J> {
    fn new(
        operation: impl FnOnce() -> Result<T> + Send + 'static,
        resolver: impl FnOnce(T) -> Result<J> + Send + 'static,
    ) -> Self {
        Self {
            operation: Some(Box::new(operation)),
            resolver: Some(Box::new(resolver)),
        }
    }
}

impl<T, J> Task for WorkerTask<T, J>
where
    T: Send + 'static,
    J: ToNapiValue + TypeName,
{
    type Output = T;
    type JsValue = J;

    fn compute(&mut self) -> Result<Self::Output> {
        self.operation.take().ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                "Holt worker operation was already consumed",
            )
        })?()
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        self.resolver.take().ok_or_else(|| {
            Error::new(
                Status::GenericFailure,
                "Holt worker resolver was already consumed",
            )
        })?(output)
    }
}

/// Promise task resolving to no JavaScript value.
pub type VoidTask = AsyncTask<WorkerTask<(), ()>>;
/// Promise task resolving to a named Tree handle.
pub type TreeHandleTask = AsyncTask<WorkerTask<CoreTree, Tree>>;
/// Promise task resolving to a list of tree names.
pub type StringListTask = AsyncTask<WorkerTask<Vec<String>, Vec<String>>>;
/// Promise task resolving to an optional value buffer.
pub type OptionalBufferTask = AsyncTask<WorkerTask<Option<Vec<u8>>, Option<Buffer>>>;
/// Promise task resolving to an optional record.
pub type OptionalRecordTask = AsyncTask<WorkerTask<Option<holt::Record>, Option<Record>>>;
/// Promise task resolving to a boolean.
pub type BoolTask = AsyncTask<WorkerTask<bool, bool>>;
/// Promise task resolving to key-only scan entries.
pub type KeyScanTask = AsyncTask<WorkerTask<Vec<KeyRangeEntry>, Vec<ScanEntry>>>;
/// Promise task resolving to record scan entries.
pub type RecordScanTask = AsyncTask<WorkerTask<Vec<RangeEntry>, Vec<ScanEntry>>>;

fn worker_task<T, J>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
    resolver: impl FnOnce(T) -> Result<J> + Send + 'static,
) -> AsyncTask<WorkerTask<T, J>>
where
    T: Send + 'static,
    J: ToNapiValue + TypeName,
{
    AsyncTask::new(WorkerTask::new(operation, resolver))
}

fn holt_task<T, J>(
    operation: impl FnOnce() -> holt::Result<T> + Send + 'static,
    resolver: impl FnOnce(T) -> Result<J> + Send + 'static,
) -> AsyncTask<WorkerTask<T, J>>
where
    T: Send + 'static,
    J: ToNapiValue + TypeName,
{
    worker_task(move || operation().map_err(js_error), resolver)
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

fn key_scan_entry(entry: KeyRangeEntry) -> Result<ScanEntry> {
    match entry {
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
    }
}

fn record_scan_entry(entry: RangeEntry) -> Result<ScanEntry> {
    match entry {
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
    }
}

/// A Node.js handle to a Holt multi-tree database.
#[napi]
pub struct Database {
    inner: Option<CoreDatabase>,
}

#[napi]
impl Database {
    fn from_core(inner: CoreDatabase) -> Self {
        Self { inner: Some(inner) }
    }

    /// Open a file-backed multi-tree database without blocking the Node.js
    /// event loop.
    #[napi(factory)]
    pub async fn open(path: String, options: Option<TreeOptions>) -> Result<Self> {
        let wal_sync = options
            .and_then(|options| options.wal_sync)
            .unwrap_or(false);
        let mut config = TreeConfig::new(path);
        config.durability = holt::Durability::Wal { sync: wal_sync };
        spawn_holt(move || CoreDatabase::open(config))
            .await
            .map(Self::from_core)
    }

    /// Open a volatile in-memory multi-tree database without blocking the
    /// Node.js event loop.
    #[napi(factory)]
    pub async fn open_memory() -> Result<Self> {
        spawn_holt(|| CoreDatabase::open(TreeConfig::memory()))
            .await
            .map(Self::from_core)
    }

    /// Explicitly release the database handle on a worker thread. Existing
    /// Tree handles remain usable until they are closed or dropped.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close(&mut self) -> VoidTask {
        let inner = self.inner.take();
        worker_task(
            move || {
                drop(inner);
                Ok(())
            },
            Ok,
        )
    }

    /// Create a new named tree on a worker thread.
    #[napi(js_name = "createTree", ts_return_type = "Promise<Tree>")]
    pub fn create_tree(&self, name: String) -> Result<TreeHandleTask> {
        let database = self.clone_core()?;
        Ok(holt_task(
            move || database.create_tree(&name),
            |tree| Ok(Tree::from_core(tree)),
        ))
    }

    /// Open an existing named tree on a worker thread.
    #[napi(js_name = "openTree", ts_return_type = "Promise<Tree>")]
    pub fn open_tree(&self, name: String) -> Result<TreeHandleTask> {
        let database = self.clone_core()?;
        Ok(holt_task(
            move || database.open_tree(&name),
            |tree| Ok(Tree::from_core(tree)),
        ))
    }

    /// Open a named tree on a worker thread, creating it when absent.
    #[napi(js_name = "openOrCreateTree", ts_return_type = "Promise<Tree>")]
    pub fn open_or_create_tree(&self, name: String) -> Result<TreeHandleTask> {
        let database = self.clone_core()?;
        Ok(holt_task(
            move || database.open_or_create_tree(&name),
            |tree| Ok(Tree::from_core(tree)),
        ))
    }

    /// List all live named trees on a worker thread.
    #[napi(js_name = "listTrees", ts_return_type = "Promise<Array<string>>")]
    pub fn list_trees(&self) -> Result<StringListTask> {
        let database = self.clone_core()?;
        Ok(holt_task(move || database.list_trees(), Ok))
    }

    /// Drop a named tree and fence existing handles on a worker thread.
    #[napi(js_name = "dropTree", ts_return_type = "Promise<void>")]
    pub fn drop_tree(&self, name: String) -> Result<VoidTask> {
        let database = self.clone_core()?;
        Ok(holt_task(move || database.drop_tree(&name), Ok))
    }

    /// Flush every named tree and the shared WAL on a worker thread.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn checkpoint(&self) -> Result<VoidTask> {
        let database = self.clone_core()?;
        Ok(holt_task(move || database.checkpoint(), Ok))
    }

    fn clone_core(&self) -> Result<CoreDatabase> {
        self.inner
            .clone()
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

    /// Open a file-backed tree without blocking the Node.js event loop.
    #[napi(factory)]
    pub async fn open(path: String, options: Option<TreeOptions>) -> Result<Self> {
        let wal_sync = options
            .and_then(|options| options.wal_sync)
            .unwrap_or(false);
        spawn_holt(move || {
            TreeBuilder::new(path)
                .durability(holt::Durability::Wal { sync: wal_sync })
                .open()
        })
        .await
        .map(Self::from_core)
    }

    /// Open a volatile in-memory tree without blocking the Node.js event loop.
    #[napi(factory)]
    pub async fn open_memory() -> Result<Self> {
        spawn_holt(|| TreeBuilder::new("holt-node-memory").memory().open())
            .await
            .map(Self::from_core)
    }

    /// Explicitly release the native tree handle on a worker thread.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close(&mut self) -> VoidTask {
        let inner = self.inner.take();
        worker_task(
            move || {
                drop(inner);
                Ok(())
            },
            Ok,
        )
    }

    /// Return the value at `key` from a worker thread.
    #[napi(ts_return_type = "Promise<Buffer | null>")]
    pub fn get(&self, key: Uint8Array) -> Result<OptionalBufferTask> {
        let tree = self.clone_core()?;
        let key = key.to_vec();
        Ok(holt_task(
            move || tree.get(&key),
            |value| Ok(value.map(into_buffer)),
        ))
    }

    /// Return the value and conditional-write version from a worker thread.
    #[napi(js_name = "getRecord", ts_return_type = "Promise<Record | null>")]
    pub fn get_record(&self, key: Uint8Array) -> Result<OptionalRecordTask> {
        let tree = self.clone_core()?;
        let key = key.to_vec();
        Ok(holt_task(
            move || tree.get_record(&key),
            |record| {
                Ok(record.map(|record| Record {
                    value: into_buffer(record.value),
                    version: into_bigint(record.version.as_u64()),
                }))
            },
        ))
    }

    /// Insert or replace a value on a worker thread.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn put(&self, key: Uint8Array, value: Uint8Array) -> Result<VoidTask> {
        let tree = self.clone_core()?;
        let key = key.to_vec();
        let value = value.to_vec();
        Ok(holt_task(move || tree.put(&key, &value), Ok))
    }

    /// Delete a key on a worker thread and return whether it existed.
    #[napi(ts_return_type = "Promise<boolean>")]
    pub fn delete(&self, key: Uint8Array) -> Result<BoolTask> {
        let tree = self.clone_core()?;
        let key = key.to_vec();
        Ok(holt_task(move || tree.delete(&key), Ok))
    }

    /// Compare the current version and replace the value on a worker thread.
    #[napi(js_name = "compareAndPut", ts_return_type = "Promise<boolean>")]
    pub fn compare_and_put(
        &self,
        key: Uint8Array,
        version: BigInt,
        value: Uint8Array,
    ) -> Result<BoolTask> {
        let tree = self.clone_core()?;
        let key = key.to_vec();
        let version = holt::RecordVersion::from_raw(from_bigint(version)?);
        let value = value.to_vec();
        Ok(holt_task(
            move || tree.compare_and_put(&key, version, &value),
            Ok,
        ))
    }

    /// Flush dirty frames and the WAL on a worker thread.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn checkpoint(&self) -> Result<VoidTask> {
        let tree = self.clone_core()?;
        Ok(holt_task(move || tree.checkpoint(), Ok))
    }

    /// Scan keys on a worker thread.
    #[napi(js_name = "scanKeys", ts_return_type = "Promise<Array<ScanEntry>>")]
    pub fn scan_keys(
        &self,
        prefix: Option<Uint8Array>,
        options: Option<ScanOptions>,
    ) -> Result<KeyScanTask> {
        let tree = self.clone_core()?;
        let options = options.unwrap_or(ScanOptions {
            start_after: None,
            delimiter: None,
        });
        let delimiter = delimiter(options.delimiter)?;
        let prefix = prefix.map(|prefix| prefix.to_vec()).unwrap_or_default();
        let start_after = options.start_after.map(|key| key.to_vec());
        Ok(holt_task(
            move || {
                let mut builder = tree.scan_keys(&prefix);
                if let Some(start_after) = &start_after {
                    builder = builder.start_after(start_after);
                }
                if let Some(delimiter) = delimiter {
                    builder = builder.delimiter(delimiter);
                }
                builder.into_iter().collect()
            },
            |entries: Vec<KeyRangeEntry>| entries.into_iter().map(key_scan_entry).collect(),
        ))
    }

    /// Scan records on a worker thread.
    #[napi(js_name = "scanRecords", ts_return_type = "Promise<Array<ScanEntry>>")]
    pub fn scan_records(
        &self,
        prefix: Option<Uint8Array>,
        options: Option<ScanOptions>,
    ) -> Result<RecordScanTask> {
        let tree = self.clone_core()?;
        let options = options.unwrap_or(ScanOptions {
            start_after: None,
            delimiter: None,
        });
        let delimiter = delimiter(options.delimiter)?;
        let prefix = prefix.map(|prefix| prefix.to_vec()).unwrap_or_default();
        let start_after = options.start_after.map(|key| key.to_vec());
        Ok(holt_task(
            move || {
                let mut builder = tree.scan(&prefix);
                if let Some(start_after) = &start_after {
                    builder = builder.start_after(start_after);
                }
                if let Some(delimiter) = delimiter {
                    builder = builder.delimiter(delimiter);
                }
                builder.into_iter().collect()
            },
            |entries: Vec<RangeEntry>| entries.into_iter().map(record_scan_entry).collect(),
        ))
    }

    fn clone_core(&self) -> Result<CoreTree> {
        self.inner
            .clone()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Holt tree is closed"))
    }
}
