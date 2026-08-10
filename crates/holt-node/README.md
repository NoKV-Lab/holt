# @nokv-lab/holt

Node.js bindings for the [Holt](https://github.com/NoKV-Lab/holt)
path-shaped metadata engine.

The package is a Node-API native addon built directly on the Rust `holt`
crate. Keys and values are `Buffer`/`Uint8Array` instances. Scans return
`kind`, `path`, `value`, and `version` fields. `version` is a JavaScript
`bigint` in the generated typings.

The package exposes a focused subset of Holt's `Tree` and `Database` APIs. It
does not yet have full Rust API parity. It is currently Unix-only, matching
Holt's file-store support.

## Install

```sh
npm install @nokv-lab/holt
```

Prebuilt packages support these targets:

- macOS on x64 and arm64
- Linux with glibc on x64 and arm64

The release workflow does not publish Windows or Linux musl builds.

```ts
import { Tree } from "@nokv-lab/holt";

const tree = await Tree.openMemory();
await tree.put(Buffer.from("bucket/a"), Buffer.from("metadata"));
console.log((await tree.get(Buffer.from("bucket/a")))?.toString());
console.log(await tree.scanKeys(Buffer.from("bucket/")));
await tree.close();
```

Multiple named trees can share one database, WAL, and checkpoint boundary:

```ts
import { Database } from "@nokv-lab/holt";

const db = await Database.open("/var/lib/app/holt", { walSync: true });
const objects = await db.openOrCreateTree("objects");
const sessions = await db.openOrCreateTree("sessions");

await objects.put(Buffer.from("bucket/a"), Buffer.from("object metadata"));
await sessions.put(Buffer.from("session/1"), Buffer.from("session metadata"));

console.log(await db.listTrees());
await db.checkpoint();
await objects.close();
await sessions.close();
await db.close();
```

All storage operations return Promises and execute on native worker threads,
so file I/O, WAL sync, replay, checkpoints, and scans do not block the Node.js
event loop.

## API coverage

The Node package currently supports:

- file-backed and in-memory trees and databases
- named tree creation, lookup, listing, and removal
- point reads, writes, deletes, records with versions, and compare-and-put
- prefix scans with `startAfter` and delimiter rollups
- explicit tree and database checkpoints

The Node package does not expose these Rust APIs yet:

- conditional insert and delete, rename, and single-tree or cross-tree atomic batches
- snapshots and consistent scoped views
- prefix counts and empty-prefix checks
- garbage collection, vacuum, compaction, statistics, and metrics
- database checkpoint export and install
- scan limits, visitor-based streaming, and scan statistics
- buffer-pool and checkpoint tuning, custom storage backends, and durability modes beyond WAL sync

Node scans currently materialize all results in an array.

Build the native artifact locally with:

```sh
npm install
npm run build
```

The repository does not commit platform-specific `.node` artifacts. The
release workflow builds and tests each target before it publishes the platform
packages and the root package. Maintainers can find the release procedure in
[the repository release guide](https://github.com/NoKV-Lab/holt/blob/main/crates/holt-node/RELEASING.md).
