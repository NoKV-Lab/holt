//! C ABI for Holt.
//!
//! The ABI keeps Holt's Rust ownership model explicit:
//!
//! - `HoltTree` and `HoltIter` are opaque handles owned by the caller
//!   until `holt_tree_close` / `holt_iter_close`.
//! - Byte buffers returned through `HoltBytes`, `HoltRecord`, or
//!   `HoltEntry` are allocated by this crate and must be released with
//!   `holt_bytes_free`, `holt_record_free`, or `holt_entry_free`.
//! - Functions return `HOLT_OK`, `HOLT_ITER_END`, or `HOLT_ERR`.
//!   Diagnostic text for the current thread is available through
//!   `holt_last_error_message`.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use holt::{KeyRangeEntry, KeyRangeIter, RangeEntry, RangeIter, Tree, TreeBuilder};

/// Successful FFI call.
pub const HOLT_OK: i32 = 0;
/// Iterator is exhausted.
pub const HOLT_ITER_END: i32 = 1;
/// FFI call failed; inspect `holt_last_error_message`.
pub const HOLT_ERR: i32 = -1;

/// `HoltEntry` is a key/value record.
pub const HOLT_ENTRY_KEY: u32 = 1;
/// `HoltEntry` is a delimiter common-prefix rollup.
pub const HOLT_ENTRY_COMMON_PREFIX: u32 = 2;

/// `HoltAtomicOp` inserts or replaces one key/value pair.
pub const HOLT_ATOMIC_PUT: u32 = 1;
/// `HoltAtomicOp` deletes one key.
pub const HOLT_ATOMIC_DELETE: u32 = 2;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty CString"));
}

/// Opaque tree handle.
pub struct HoltTree {
    tree: Tree,
}

/// Opaque range iterator handle.
pub struct HoltIter {
    inner: HoltIterInner,
}

enum HoltIterInner {
    Keys(KeyRangeIter),
    Records(RangeIter),
}

/// Byte buffer owned by Holt FFI.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HoltBytes {
    /// Pointer to `len` bytes, or null for an empty buffer.
    pub ptr: *mut u8,
    /// Number of bytes at `ptr`.
    pub len: usize,
}

impl Default for HoltBytes {
    fn default() -> Self {
        Self {
            ptr: ptr::null_mut(),
            len: 0,
        }
    }
}

/// Result of `holt_tree_get`.
#[repr(C)]
#[derive(Default)]
pub struct HoltRecord {
    /// `1` if the key exists, `0` otherwise.
    pub found: u8,
    /// Value bytes. Empty when `found == 0`.
    pub value: HoltBytes,
    /// Live `RecordVersion` token. Zero when `found == 0`.
    pub version: u64,
}

/// One range iterator emission.
#[repr(C)]
#[derive(Default)]
pub struct HoltEntry {
    /// `HOLT_ENTRY_KEY` or `HOLT_ENTRY_COMMON_PREFIX`.
    pub kind: u32,
    /// Key bytes for `HOLT_ENTRY_KEY`, common-prefix bytes for
    /// `HOLT_ENTRY_COMMON_PREFIX`.
    pub path: HoltBytes,
    /// Value bytes for record scans. Empty for key-only scans and
    /// common-prefix entries.
    pub value: HoltBytes,
    /// Live `RecordVersion` token for key entries. Zero for
    /// common-prefix entries.
    pub version: u64,
}

/// One operation passed to `holt_tree_atomic`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HoltAtomicOp {
    /// `HOLT_ATOMIC_PUT` or `HOLT_ATOMIC_DELETE`.
    pub kind: u32,
    /// Key bytes.
    pub key: *const u8,
    /// Number of key bytes.
    pub key_len: usize,
    /// Value bytes for `HOLT_ATOMIC_PUT`.
    pub value: *const u8,
    /// Number of value bytes. Deletes require zero.
    pub value_len: usize,
}

enum ParsedAtomicOp<'a> {
    Put { key: &'a [u8], value: &'a [u8] },
    Delete { key: &'a [u8] },
}

type FfiResult<T> = Result<T, String>;

fn set_error(message: impl Into<String>) -> i32 {
    let raw = message.into().replace('\0', "\\0");
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(raw).expect("interior NUL removed");
    });
    HOLT_ERR
}

fn clear_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new("").expect("empty CString");
    });
}

fn boundary(f: impl FnOnce() -> FfiResult<i32>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(code)) => {
            clear_error();
            code
        }
        Ok(Err(err)) => set_error(err),
        Err(_) => set_error("holt ffi: panic crossed FFI boundary"),
    }
}

unsafe fn tree_ref<'a>(tree: *mut HoltTree) -> FfiResult<&'a HoltTree> {
    unsafe { tree.as_ref() }.ok_or_else(|| "holt ffi: null tree handle".to_owned())
}

unsafe fn iter_mut<'a>(iter: *mut HoltIter) -> FfiResult<&'a mut HoltIter> {
    unsafe { iter.as_mut() }.ok_or_else(|| "holt ffi: null iterator handle".to_owned())
}

unsafe fn out_mut<'a, T>(out: *mut T, name: &str) -> FfiResult<&'a mut T> {
    unsafe { out.as_mut() }.ok_or_else(|| format!("holt ffi: null {name} output pointer"))
}

unsafe fn bytes_arg<'a>(ptr: *const u8, len: usize, name: &str) -> FfiResult<&'a [u8]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(format!(
            "holt ffi: null {name} pointer with non-zero length"
        ));
    }
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

unsafe fn atomic_ops_arg<'a>(
    ptr: *const HoltAtomicOp,
    len: usize,
) -> FfiResult<&'a [HoltAtomicOp]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err("holt ffi: null atomic ops pointer with non-zero count".to_owned());
    }
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

unsafe fn optional_bytes_arg<'a>(
    ptr: *const u8,
    len: usize,
    name: &str,
) -> FfiResult<Option<&'a [u8]>> {
    if ptr.is_null() {
        if len == 0 {
            Ok(None)
        } else {
            Err(format!(
                "holt ffi: null {name} pointer with non-zero length"
            ))
        }
    } else {
        Ok(Some(unsafe { slice::from_raw_parts(ptr, len) }))
    }
}

unsafe fn cstr_arg<'a>(ptr: *const c_char, name: &str) -> FfiResult<&'a CStr> {
    if ptr.is_null() {
        return Err(format!("holt ffi: null {name} pointer"));
    }
    Ok(unsafe { CStr::from_ptr(ptr) })
}

fn into_ffi_bytes(bytes: Vec<u8>) -> HoltBytes {
    if bytes.is_empty() {
        return HoltBytes::default();
    }
    let mut boxed = bytes.into_boxed_slice();
    let out = HoltBytes {
        ptr: boxed.as_mut_ptr(),
        len: boxed.len(),
    };
    std::mem::forget(boxed);
    out
}

unsafe fn free_ffi_bytes(bytes: HoltBytes) {
    if bytes.ptr.is_null() {
        return;
    }
    let raw = ptr::slice_from_raw_parts_mut(bytes.ptr, bytes.len);
    drop(unsafe { Box::from_raw(raw) });
}

fn delimiter_from_raw(delimiter: i32) -> FfiResult<Option<u8>> {
    if delimiter < 0 {
        return Ok(None);
    }
    u8::try_from(delimiter)
        .map(Some)
        .map_err(|_| format!("holt ffi: delimiter {delimiter} is outside 0..=255"))
}

unsafe fn open_path(path: *const c_char) -> FfiResult<String> {
    let cstr = unsafe { cstr_arg(path, "path") }?;
    cstr.to_str()
        .map(str::to_owned)
        .map_err(|err| format!("holt ffi: path is not valid UTF-8: {err}"))
}

fn assign_tree(out: *mut *mut HoltTree, tree: Tree) -> FfiResult<i32> {
    let out = unsafe { out_mut(out, "tree") }?;
    *out = Box::into_raw(Box::new(HoltTree { tree }));
    Ok(HOLT_OK)
}

/// Return the last error message for the current thread.
#[no_mangle]
pub extern "C" fn holt_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

/// Open a file-backed tree with optional per-operation WAL sync.
///
/// # Safety
///
/// `path` must be a non-null NUL-terminated UTF-8 string. `out`
/// must be a valid writable pointer. On success, the caller owns
/// `*out` and must pass it to [`holt_tree_close`].
#[no_mangle]
pub unsafe extern "C" fn holt_tree_open_with_wal_sync(
    path: *const c_char,
    wal_sync: u8,
    out: *mut *mut HoltTree,
) -> i32 {
    boundary(|| {
        let path = unsafe { open_path(path) }?;
        let tree = TreeBuilder::new(path)
            .durability(holt::Durability::Wal {
                sync: wal_sync != 0,
            })
            .open()
            .map_err(|err| err.to_string())?;
        assign_tree(out, tree)
    })
}

/// Open an in-memory tree.
///
/// # Safety
///
/// `out` must be a valid writable pointer. On success, the caller
/// owns `*out` and must pass it to [`holt_tree_close`].
#[no_mangle]
pub unsafe extern "C" fn holt_tree_open_memory(out: *mut *mut HoltTree) -> i32 {
    boundary(|| {
        let tree = TreeBuilder::new("holt-ffi-memory")
            .memory()
            .open()
            .map_err(|err| err.to_string())?;
        assign_tree(out, tree)
    })
}

/// Close a tree handle. Null is ignored.
///
/// # Safety
///
/// `tree` must be null or a handle returned by this crate that has
/// not already been closed.
#[no_mangle]
pub unsafe extern "C" fn holt_tree_close(tree: *mut HoltTree) {
    if tree.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(tree) });
}

/// Insert or replace one key/value pair.
///
/// # Safety
///
/// `tree` must be a live Holt handle. `key` and `value` must point
/// to `key_len` / `value_len` readable bytes unless their length is
/// zero.
#[no_mangle]
pub unsafe extern "C" fn holt_tree_put(
    tree: *mut HoltTree,
    key: *const u8,
    key_len: usize,
    value: *const u8,
    value_len: usize,
) -> i32 {
    boundary(|| {
        let tree = unsafe { tree_ref(tree) }?;
        let key = unsafe { bytes_arg(key, key_len, "key") }?;
        let value = unsafe { bytes_arg(value, value_len, "value") }?;
        tree.tree.put(key, value).map_err(|err| err.to_string())?;
        Ok(HOLT_OK)
    })
}

/// Get one record by key.
///
/// # Safety
///
/// `tree` must be a live Holt handle. `key` must point to
/// `key_len` readable bytes unless `key_len` is zero. `out` must be
/// a valid writable pointer. If `out` already owns a previous result,
/// the caller must free it before reusing the storage.
#[no_mangle]
pub unsafe extern "C" fn holt_tree_get(
    tree: *mut HoltTree,
    key: *const u8,
    key_len: usize,
    out: *mut HoltRecord,
) -> i32 {
    boundary(|| {
        let tree = unsafe { tree_ref(tree) }?;
        let key = unsafe { bytes_arg(key, key_len, "key") }?;
        let out = unsafe { out_mut(out, "record") }?;
        *out = HoltRecord::default();

        if let Some(record) = tree.tree.get_record(key).map_err(|err| err.to_string())? {
            out.found = 1;
            out.version = record.version.as_u64();
            out.value = into_ffi_bytes(record.value);
        }
        Ok(HOLT_OK)
    })
}

/// Delete one key.
///
/// # Safety
///
/// `tree` must be a live Holt handle. `key` must point to
/// `key_len` readable bytes unless `key_len` is zero.
/// `existed_out` may be null; otherwise it must be writable.
#[no_mangle]
pub unsafe extern "C" fn holt_tree_delete(
    tree: *mut HoltTree,
    key: *const u8,
    key_len: usize,
    existed_out: *mut u8,
) -> i32 {
    boundary(|| {
        let tree = unsafe { tree_ref(tree) }?;
        let key = unsafe { bytes_arg(key, key_len, "key") }?;
        let existed = tree.tree.delete(key).map_err(|err| err.to_string())?;
        if !existed_out.is_null() {
            unsafe {
                *existed_out = u8::from(existed);
            }
        }
        Ok(HOLT_OK)
    })
}

/// Apply puts and deletes through one Holt atomic batch.
///
/// # Safety
///
/// `tree` must be a live Holt handle. `ops` must point to
/// `op_count` readable entries unless `op_count` is zero. Each key
/// and put value must remain readable for the call. `committed_out`
/// must be a valid writable pointer. The function writes it only on
/// success.
#[no_mangle]
pub unsafe extern "C" fn holt_tree_atomic(
    tree: *mut HoltTree,
    ops: *const HoltAtomicOp,
    op_count: usize,
    committed_out: *mut u8,
) -> i32 {
    boundary(|| {
        let tree = unsafe { tree_ref(tree) }?;
        if committed_out.is_null() {
            return Err("holt ffi: null committed output pointer".to_owned());
        }
        let ops = unsafe { atomic_ops_arg(ops, op_count) }?;
        let mut parsed = Vec::with_capacity(ops.len());

        for (index, op) in ops.iter().enumerate() {
            let key_name = format!("atomic op {index} key");
            let key = unsafe { bytes_arg(op.key, op.key_len, &key_name) }?;
            match op.kind {
                HOLT_ATOMIC_PUT => {
                    let value_name = format!("atomic op {index} value");
                    let value = unsafe { bytes_arg(op.value, op.value_len, &value_name) }?;
                    parsed.push(ParsedAtomicOp::Put { key, value });
                }
                HOLT_ATOMIC_DELETE => {
                    if op.value_len != 0 {
                        return Err(format!("holt ffi: atomic delete op {index} has a value"));
                    }
                    parsed.push(ParsedAtomicOp::Delete { key });
                }
                kind => {
                    return Err(format!(
                        "holt ffi: atomic op {index} has unknown kind {kind}"
                    ));
                }
            }
        }

        let committed = tree
            .tree
            .atomic(|batch| {
                for op in parsed {
                    match op {
                        ParsedAtomicOp::Put { key, value } => batch.put(key, value),
                        ParsedAtomicOp::Delete { key } => batch.delete(key),
                    }
                }
            })
            .map_err(|err| err.to_string())?;
        let committed_out = unsafe { out_mut(committed_out, "committed") }?;
        *committed_out = u8::from(committed);
        Ok(HOLT_OK)
    })
}

/// Force a checkpoint.
///
/// # Safety
///
/// `tree` must be a live Holt handle.
#[no_mangle]
pub unsafe extern "C" fn holt_tree_checkpoint(tree: *mut HoltTree) -> i32 {
    boundary(|| {
        let tree = unsafe { tree_ref(tree) }?;
        tree.tree.checkpoint().map_err(|err| err.to_string())?;
        Ok(HOLT_OK)
    })
}

/// Open a key-only range iterator.
///
/// # Safety
///
/// `tree` must be a live Holt handle. `prefix` must point to
/// `prefix_len` readable bytes unless `prefix_len` is zero.
/// `start_after` may be null only when `start_after_len` is zero.
/// `out` must be a valid writable pointer. On success, the caller
/// owns `*out` and must pass it to [`holt_iter_close`].
#[no_mangle]
pub unsafe extern "C" fn holt_tree_scan_keys(
    tree: *mut HoltTree,
    prefix: *const u8,
    prefix_len: usize,
    delimiter: i32,
    start_after: *const u8,
    start_after_len: usize,
    out: *mut *mut HoltIter,
) -> i32 {
    boundary(|| {
        let tree = unsafe { tree_ref(tree) }?;
        let prefix = unsafe { bytes_arg(prefix, prefix_len, "prefix") }?;
        let start_after =
            unsafe { optional_bytes_arg(start_after, start_after_len, "start_after") }?;
        let delimiter = delimiter_from_raw(delimiter)?;
        let out = unsafe { out_mut(out, "iterator") }?;

        let mut builder = tree.tree.scan_keys(prefix);
        if let Some(start_after) = start_after {
            builder = builder.start_after(start_after);
        }
        if let Some(delimiter) = delimiter {
            builder = builder.delimiter(delimiter);
        }

        *out = Box::into_raw(Box::new(HoltIter {
            inner: HoltIterInner::Keys(builder.into_iter()),
        }));
        Ok(HOLT_OK)
    })
}

/// Open a full-record range iterator.
///
/// # Safety
///
/// `tree` must be a live Holt handle. `prefix` must point to
/// `prefix_len` readable bytes unless `prefix_len` is zero.
/// `start_after` may be null only when `start_after_len` is zero.
/// `out` must be a valid writable pointer. On success, the caller
/// owns `*out` and must pass it to [`holt_iter_close`].
#[no_mangle]
pub unsafe extern "C" fn holt_tree_scan_records(
    tree: *mut HoltTree,
    prefix: *const u8,
    prefix_len: usize,
    delimiter: i32,
    start_after: *const u8,
    start_after_len: usize,
    out: *mut *mut HoltIter,
) -> i32 {
    boundary(|| {
        let tree = unsafe { tree_ref(tree) }?;
        let prefix = unsafe { bytes_arg(prefix, prefix_len, "prefix") }?;
        let start_after =
            unsafe { optional_bytes_arg(start_after, start_after_len, "start_after") }?;
        let delimiter = delimiter_from_raw(delimiter)?;
        let out = unsafe { out_mut(out, "iterator") }?;

        let mut builder = tree.tree.scan(prefix);
        if let Some(start_after) = start_after {
            builder = builder.start_after(start_after);
        }
        if let Some(delimiter) = delimiter {
            builder = builder.delimiter(delimiter);
        }

        *out = Box::into_raw(Box::new(HoltIter {
            inner: HoltIterInner::Records(builder.into_iter()),
        }));
        Ok(HOLT_OK)
    })
}

/// Advance an iterator.
///
/// # Safety
///
/// `iter` must be a live iterator handle. `out` must be a valid
/// writable pointer. If `out` already owns a previous entry, the
/// caller must free it before reusing the storage.
#[no_mangle]
pub unsafe extern "C" fn holt_iter_next(iter: *mut HoltIter, out: *mut HoltEntry) -> i32 {
    boundary(|| {
        let iter = unsafe { iter_mut(iter) }?;
        let out = unsafe { out_mut(out, "entry") }?;
        *out = HoltEntry::default();

        match &mut iter.inner {
            HoltIterInner::Keys(iter) => match iter.next() {
                Some(Ok(KeyRangeEntry::Key { key, version })) => {
                    out.kind = HOLT_ENTRY_KEY;
                    out.path = into_ffi_bytes(key);
                    out.version = version.as_u64();
                    Ok(HOLT_OK)
                }
                Some(Ok(KeyRangeEntry::CommonPrefix(prefix))) => {
                    out.kind = HOLT_ENTRY_COMMON_PREFIX;
                    out.path = into_ffi_bytes(prefix);
                    Ok(HOLT_OK)
                }
                Some(Err(err)) => Err(err.to_string()),
                Some(Ok(_)) => Err("holt ffi: unsupported key-range entry kind".to_owned()),
                None => Ok(HOLT_ITER_END),
            },
            HoltIterInner::Records(iter) => match iter.next() {
                Some(Ok(RangeEntry::Key {
                    key,
                    value,
                    version,
                })) => {
                    out.kind = HOLT_ENTRY_KEY;
                    out.path = into_ffi_bytes(key);
                    out.value = into_ffi_bytes(value);
                    out.version = version.as_u64();
                    Ok(HOLT_OK)
                }
                Some(Ok(RangeEntry::CommonPrefix(prefix))) => {
                    out.kind = HOLT_ENTRY_COMMON_PREFIX;
                    out.path = into_ffi_bytes(prefix);
                    Ok(HOLT_OK)
                }
                Some(Err(err)) => Err(err.to_string()),
                Some(Ok(_)) => Err("holt ffi: unsupported range entry kind".to_owned()),
                None => Ok(HOLT_ITER_END),
            },
        }
    })
}

/// Close an iterator handle. Null is ignored.
///
/// # Safety
///
/// `iter` must be null or a handle returned by this crate that has
/// not already been closed.
#[no_mangle]
pub unsafe extern "C" fn holt_iter_close(iter: *mut HoltIter) {
    if iter.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(iter) });
}

/// Free one byte buffer returned by Holt FFI.
///
/// # Safety
///
/// `bytes` must be empty or a buffer returned by this crate that
/// has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn holt_bytes_free(bytes: HoltBytes) {
    unsafe { free_ffi_bytes(bytes) };
}

/// Free the value buffer inside a record and reset the record.
///
/// # Safety
///
/// `record` must be null or a valid writable pointer to a
/// `HoltRecord` whose `value` buffer is empty or owned by this
/// crate.
#[no_mangle]
pub unsafe extern "C" fn holt_record_free(record: *mut HoltRecord) {
    if let Some(record) = unsafe { record.as_mut() } {
        unsafe { free_ffi_bytes(record.value) };
        *record = HoltRecord::default();
    }
}

/// Free all buffers inside an entry and reset the entry.
///
/// # Safety
///
/// `entry` must be null or a valid writable pointer to a
/// `HoltEntry` whose buffers are empty or owned by this crate.
#[no_mangle]
pub unsafe extern "C" fn holt_entry_free(entry: *mut HoltEntry) {
    if let Some(entry) = unsafe { entry.as_mut() } {
        unsafe {
            free_ffi_bytes(entry.path);
            free_ffi_bytes(entry.value);
        }
        *entry = HoltEntry::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::MaybeUninit;
    use std::ptr;
    use tempfile::tempdir;

    fn bytes(bytes: &HoltBytes) -> &[u8] {
        if bytes.ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(bytes.ptr, bytes.len) }
    }

    #[test]
    fn memory_tree_put_get_delete() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);
            assert!(!tree.is_null());

            let key = b"bucket-a/img/01.jpg";
            let value = b"size=42;etag=abc";
            assert_eq!(
                holt_tree_put(tree, key.as_ptr(), key.len(), value.as_ptr(), value.len()),
                HOLT_OK
            );

            let mut record = HoltRecord::default();
            assert_eq!(
                holt_tree_get(tree, key.as_ptr(), key.len(), &mut record),
                HOLT_OK
            );
            assert_eq!(record.found, 1);
            assert_eq!(bytes(&record.value), value);
            assert!(record.version > 0);
            holt_record_free(&mut record);

            let mut existed = 0;
            assert_eq!(
                holt_tree_delete(tree, key.as_ptr(), key.len(), &mut existed),
                HOLT_OK
            );
            assert_eq!(existed, 1);

            assert_eq!(
                holt_tree_get(tree, key.as_ptr(), key.len(), &mut record),
                HOLT_OK
            );
            assert_eq!(record.found, 0);

            holt_tree_close(tree);
        }
    }

    #[test]
    fn atomic_batch_applies_puts_and_deletes() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);

            let old_key = b"old";
            assert_eq!(
                holt_tree_put(tree, old_key.as_ptr(), old_key.len(), b"x".as_ptr(), 1),
                HOLT_OK
            );

            let key_a = b"a";
            let value_a = b"one";
            let key_b = b"b";
            let value_b = b"two";
            let ops = [
                HoltAtomicOp {
                    kind: HOLT_ATOMIC_PUT,
                    key: key_a.as_ptr(),
                    key_len: key_a.len(),
                    value: value_a.as_ptr(),
                    value_len: value_a.len(),
                },
                HoltAtomicOp {
                    kind: HOLT_ATOMIC_PUT,
                    key: key_b.as_ptr(),
                    key_len: key_b.len(),
                    value: value_b.as_ptr(),
                    value_len: value_b.len(),
                },
                HoltAtomicOp {
                    kind: HOLT_ATOMIC_DELETE,
                    key: old_key.as_ptr(),
                    key_len: old_key.len(),
                    value: ptr::null(),
                    value_len: 0,
                },
            ];

            let mut committed = 0;
            assert_eq!(
                holt_tree_atomic(tree, ops.as_ptr(), ops.len(), &mut committed),
                HOLT_OK
            );
            assert_eq!(committed, 1);

            for (key, value) in [(&key_a[..], &value_a[..]), (&key_b[..], &value_b[..])] {
                let mut record = HoltRecord::default();
                assert_eq!(
                    holt_tree_get(tree, key.as_ptr(), key.len(), &mut record),
                    HOLT_OK
                );
                assert_eq!(record.found, 1);
                assert_eq!(bytes(&record.value), value);
                holt_record_free(&mut record);
            }

            let mut record = HoltRecord::default();
            assert_eq!(
                holt_tree_get(tree, old_key.as_ptr(), old_key.len(), &mut record),
                HOLT_OK
            );
            assert_eq!(record.found, 0);

            committed = 0;
            assert_eq!(
                holt_tree_atomic(tree, ptr::null(), 0, &mut committed),
                HOLT_OK
            );
            assert_eq!(committed, 1);
            holt_tree_close(tree);
        }
    }

    #[test]
    fn atomic_batch_rejects_all_ops_before_mutation() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);

            let key = b"must-not-exist";
            let value = b"value";
            let ops = [
                HoltAtomicOp {
                    kind: HOLT_ATOMIC_PUT,
                    key: key.as_ptr(),
                    key_len: key.len(),
                    value: value.as_ptr(),
                    value_len: value.len(),
                },
                HoltAtomicOp {
                    kind: 99,
                    key: key.as_ptr(),
                    key_len: key.len(),
                    value: ptr::null(),
                    value_len: 0,
                },
            ];

            let mut committed = 9;
            assert_eq!(
                holt_tree_atomic(tree, ops.as_ptr(), ops.len(), &mut committed),
                HOLT_ERR
            );
            assert_eq!(committed, 9);
            let msg = CStr::from_ptr(holt_last_error_message()).to_str().unwrap();
            assert!(msg.contains("atomic op 1 has unknown kind 99"));

            let mut record = HoltRecord::default();
            assert_eq!(
                holt_tree_get(tree, key.as_ptr(), key.len(), &mut record),
                HOLT_OK
            );
            assert_eq!(record.found, 0);

            assert_eq!(
                holt_tree_atomic(tree, ptr::null(), 1, &mut committed),
                HOLT_ERR
            );
            let msg = CStr::from_ptr(holt_last_error_message()).to_str().unwrap();
            assert!(msg.contains("atomic ops"));

            let delete_with_value = HoltAtomicOp {
                kind: HOLT_ATOMIC_DELETE,
                key: key.as_ptr(),
                key_len: key.len(),
                value: value.as_ptr(),
                value_len: value.len(),
            };
            assert_eq!(
                holt_tree_atomic(tree, &delete_with_value, 1, &mut committed),
                HOLT_ERR
            );
            let msg = CStr::from_ptr(holt_last_error_message()).to_str().unwrap();
            assert!(msg.contains("atomic delete op 0 has a value"));
            holt_tree_close(tree);
        }
    }

    #[test]
    fn scan_keys_and_records() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);

            for (key, value) in [
                (&b"bucket-a/a.parquet"[..], &b"size=1"[..]),
                (&b"bucket-a/dir/b.parquet"[..], &b"size=2"[..]),
                (&b"bucket-b/c.parquet"[..], &b"size=3"[..]),
            ] {
                assert_eq!(
                    holt_tree_put(tree, key.as_ptr(), key.len(), value.as_ptr(), value.len()),
                    HOLT_OK
                );
            }

            let mut iter = ptr::null_mut();
            let prefix = b"bucket-a/";
            assert_eq!(
                holt_tree_scan_keys(
                    tree,
                    prefix.as_ptr(),
                    prefix.len(),
                    i32::from(b'/'),
                    ptr::null(),
                    0,
                    &mut iter,
                ),
                HOLT_OK
            );

            let mut entry = HoltEntry::default();
            assert_eq!(holt_iter_next(iter, &mut entry), HOLT_OK);
            assert_eq!(entry.kind, HOLT_ENTRY_KEY);
            assert_eq!(bytes(&entry.path), b"bucket-a/a.parquet");
            assert!(entry.value.ptr.is_null());
            holt_entry_free(&mut entry);

            assert_eq!(holt_iter_next(iter, &mut entry), HOLT_OK);
            assert_eq!(entry.kind, HOLT_ENTRY_COMMON_PREFIX);
            assert_eq!(bytes(&entry.path), b"bucket-a/dir/");
            holt_entry_free(&mut entry);

            assert_eq!(holt_iter_next(iter, &mut entry), HOLT_ITER_END);
            holt_iter_close(iter);

            let mut iter = ptr::null_mut();
            assert_eq!(
                holt_tree_scan_records(
                    tree,
                    prefix.as_ptr(),
                    prefix.len(),
                    -1,
                    ptr::null(),
                    0,
                    &mut iter
                ),
                HOLT_OK
            );
            assert_eq!(holt_iter_next(iter, &mut entry), HOLT_OK);
            assert_eq!(entry.kind, HOLT_ENTRY_KEY);
            assert!(!entry.value.ptr.is_null());
            holt_entry_free(&mut entry);
            holt_iter_close(iter);

            holt_tree_close(tree);
        }
    }

    #[test]
    fn output_structs_do_not_need_zero_initialization() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);

            let key = b"bucket-a/file";
            let value = b"size=9";
            assert_eq!(
                holt_tree_put(tree, key.as_ptr(), key.len(), value.as_ptr(), value.len()),
                HOLT_OK
            );

            let mut record = MaybeUninit::<HoltRecord>::uninit();
            assert_eq!(
                holt_tree_get(tree, key.as_ptr(), key.len(), record.as_mut_ptr()),
                HOLT_OK
            );
            let mut record = record.assume_init();
            assert_eq!(record.found, 1);
            assert_eq!(bytes(&record.value), value);
            holt_record_free(&mut record);

            let mut iter = ptr::null_mut();
            assert_eq!(
                holt_tree_scan_keys(tree, key.as_ptr(), 0, -1, ptr::null(), 0, &mut iter),
                HOLT_OK
            );
            let mut entry = MaybeUninit::<HoltEntry>::uninit();
            assert_eq!(holt_iter_next(iter, entry.as_mut_ptr()), HOLT_OK);
            let mut entry = entry.assume_init();
            assert_eq!(entry.kind, HOLT_ENTRY_KEY);
            holt_entry_free(&mut entry);
            holt_iter_close(iter);

            holt_tree_close(tree);
        }
    }

    #[test]
    fn reports_invalid_arguments() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);

            let mut iter = ptr::null_mut();
            assert_eq!(
                holt_tree_scan_keys(tree, ptr::null(), 1, -1, ptr::null(), 0, &mut iter),
                HOLT_ERR
            );
            let msg = CStr::from_ptr(holt_last_error_message()).to_str().unwrap();
            assert!(msg.contains("prefix"));

            holt_tree_close(tree);
        }
    }

    #[test]
    fn persistent_tree_reopens_checkpointed_data() {
        unsafe {
            let dir = tempdir().unwrap();
            let path = dir.path().join("ffi.holt");
            let path = CString::new(path.to_str().unwrap()).unwrap();

            let key = b"bucket-a/year=2026/file.parquet";
            let value = b"size=128;kind=file";

            let mut tree = ptr::null_mut();
            assert_eq!(
                holt_tree_open_with_wal_sync(path.as_ptr(), 0, &mut tree),
                HOLT_OK
            );
            assert_eq!(
                holt_tree_put(tree, key.as_ptr(), key.len(), value.as_ptr(), value.len()),
                HOLT_OK
            );
            assert_eq!(holt_tree_checkpoint(tree), HOLT_OK);
            holt_tree_close(tree);

            let mut reopened = ptr::null_mut();
            assert_eq!(
                holt_tree_open_with_wal_sync(path.as_ptr(), 0, &mut reopened),
                HOLT_OK
            );
            let mut record = HoltRecord::default();
            assert_eq!(
                holt_tree_get(reopened, key.as_ptr(), key.len(), &mut record),
                HOLT_OK
            );
            assert_eq!(record.found, 1);
            assert_eq!(bytes(&record.value), value);
            holt_record_free(&mut record);
            holt_tree_close(reopened);
        }
    }

    #[test]
    fn scan_records_honors_start_after_and_delimiter_validation() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);

            for key in [
                &b"bucket-a/a.parquet"[..],
                &b"bucket-a/b.parquet"[..],
                &b"bucket-a/dir/c.parquet"[..],
            ] {
                assert_eq!(
                    holt_tree_put(tree, key.as_ptr(), key.len(), b"x".as_ptr(), 1),
                    HOLT_OK
                );
            }

            let prefix = b"bucket-a/";
            let start_after = b"bucket-a/a.parquet";
            let mut iter = ptr::null_mut();
            assert_eq!(
                holt_tree_scan_records(
                    tree,
                    prefix.as_ptr(),
                    prefix.len(),
                    i32::from(b'/'),
                    start_after.as_ptr(),
                    start_after.len(),
                    &mut iter
                ),
                HOLT_OK
            );

            let mut entry = HoltEntry::default();
            assert_eq!(holt_iter_next(iter, &mut entry), HOLT_OK);
            assert_eq!(entry.kind, HOLT_ENTRY_KEY);
            assert_eq!(bytes(&entry.path), b"bucket-a/b.parquet");
            assert_eq!(bytes(&entry.value), b"x");
            holt_entry_free(&mut entry);

            assert_eq!(holt_iter_next(iter, &mut entry), HOLT_OK);
            assert_eq!(entry.kind, HOLT_ENTRY_COMMON_PREFIX);
            assert_eq!(bytes(&entry.path), b"bucket-a/dir/");
            holt_entry_free(&mut entry);
            assert_eq!(holt_iter_next(iter, &mut entry), HOLT_ITER_END);
            holt_iter_close(iter);

            assert_eq!(
                holt_tree_scan_keys(
                    tree,
                    prefix.as_ptr(),
                    prefix.len(),
                    256,
                    ptr::null(),
                    0,
                    &mut iter
                ),
                HOLT_ERR
            );
            let msg = CStr::from_ptr(holt_last_error_message()).to_str().unwrap();
            assert!(msg.contains("delimiter 256"));

            holt_tree_close(tree);
        }
    }

    #[test]
    fn delete_missing_and_null_outputs_are_well_defined() {
        unsafe {
            assert_eq!(holt_tree_open_memory(ptr::null_mut()), HOLT_ERR);
            let msg = CStr::from_ptr(holt_last_error_message()).to_str().unwrap();
            assert!(msg.contains("tree"));

            let mut tree = ptr::null_mut();
            assert_eq!(holt_tree_open_memory(&mut tree), HOLT_OK);

            let missing = b"missing";
            let mut existed = 99;
            assert_eq!(
                holt_tree_delete(tree, missing.as_ptr(), missing.len(), &mut existed),
                HOLT_OK
            );
            assert_eq!(existed, 0);
            assert_eq!(
                holt_tree_delete(tree, missing.as_ptr(), missing.len(), ptr::null_mut()),
                HOLT_OK
            );

            let mut record = HoltRecord {
                found: 1,
                value: into_ffi_bytes(b"owned".to_vec()),
                version: 42,
            };
            holt_record_free(&mut record);
            assert_eq!(record.found, 0);
            assert!(record.value.ptr.is_null());
            assert_eq!(record.version, 0);

            let mut entry = HoltEntry {
                kind: HOLT_ENTRY_KEY,
                path: into_ffi_bytes(b"path".to_vec()),
                value: into_ffi_bytes(b"value".to_vec()),
                version: 1,
            };
            holt_entry_free(&mut entry);
            assert_eq!(entry.kind, 0);
            assert!(entry.path.ptr.is_null());
            assert!(entry.value.ptr.is_null());

            holt_bytes_free(HoltBytes::default());
            holt_record_free(ptr::null_mut());
            holt_entry_free(ptr::null_mut());
            holt_iter_close(ptr::null_mut());
            holt_tree_close(ptr::null_mut());
            holt_tree_close(tree);
        }
    }

    #[test]
    fn reports_invalid_open_path() {
        unsafe {
            let mut tree = ptr::null_mut();
            assert_eq!(
                holt_tree_open_with_wal_sync(ptr::null(), 0, &mut tree),
                HOLT_ERR
            );
            let msg = CStr::from_ptr(holt_last_error_message()).to_str().unwrap();
            assert!(msg.contains("path"));
            assert!(tree.is_null());
        }
    }
}
