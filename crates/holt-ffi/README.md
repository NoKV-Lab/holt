# holt-ffi

`holt-ffi` is the C ABI wrapper for Holt. It is a separate crate so
the unsafe boundary, C header, and dynamic/static library build do
not leak into the Rust storage-engine core.

## Build

```sh
cargo build -p holt-ffi --release
```

This produces platform-native dynamic and static libraries under
`target/release/`:

- macOS: `libholt_ffi.dylib`, `libholt_ffi.a`
- Linux: `libholt_ffi.so`, `libholt_ffi.a`

The public header is:

```text
crates/holt-ffi/include/holt_ffi.h
```

`crates/holt-ffi/examples/abi_smoke.c` is a minimal C caller that
checks the header-level ABI surface.

## ABI Surface

Tree lifecycle:

- `holt_tree_open_with_wal_sync(path, wal_sync, &tree)`
- `holt_tree_open_memory(&tree)`
- `holt_tree_close(tree)`

Point operations:

- `holt_tree_put(tree, key, key_len, value, value_len)`
- `holt_tree_get(tree, key, key_len, &record)`
- `holt_tree_delete(tree, key, key_len, &existed)`
- `holt_tree_checkpoint(tree)`

Atomic mutations:

- `holt_tree_atomic(tree, ops, op_count, &committed)`

`HoltAtomicOp` supports `HOLT_ATOMIC_PUT` and
`HOLT_ATOMIC_DELETE`. The function validates the full input array
before it submits one `Tree::atomic` batch. Holt copies all key and
value bytes during the call. Operations keep their array order, and
delete operations require `value_len == 0`.

An empty batch commits successfully. The function writes `committed`
only when it returns `HOLT_OK`.

Range operations:

- `holt_tree_scan_keys(...)`
- `holt_tree_scan_records(...)`
- `holt_iter_next(iter, &entry)`
- `holt_iter_close(iter)`

Memory ownership:

- `holt_record_free(&record)`
- `holt_entry_free(&entry)`
- `holt_bytes_free(bytes)`

Output structs do not need to be zero-initialized before the first
call. If a caller wants to reuse a `HoltRecord` or `HoltEntry`
storage slot that already owns buffers from a previous call, it must
free the previous result before passing the slot back to Holt.

Every function returns `HOLT_OK`, `HOLT_ITER_END`, or `HOLT_ERR`.
Use `holt_last_error_message()` to inspect the current thread's
last error.

## Iterator Semantics

The ABI exposes the same semantics as Holt's Rust iterators:

- key-only scans use `Tree::scan_keys`.
- record scans use `Tree::scan`.
- delimiter values below zero mean "no delimiter".
- delimiter values `0..=255` enable common-prefix rollup.
- `start_after == NULL && start_after_len == 0` means "no marker".

These iterators are restart-on-conflict cursors, not MVCC snapshots.
