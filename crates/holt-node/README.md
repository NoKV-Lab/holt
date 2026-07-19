# @holt/node

Node.js bindings for the [Holt](https://github.com/EnderRomantice/holt)
path-shaped metadata engine.

The package is a Node-API native addon built directly on the Rust `holt`
crate. Keys and values are `Buffer`/`Uint8Array` instances; scans return
`kind`, `path`, `value`, and `version` fields. `version` is a JavaScript
`bigint` in the generated typings.

The package exposes the core `Tree` API plus `Database`, Holt's multi-tree
handle. It is currently Unix-only, matching Holt's file-store support.

```ts
import { Tree } from "@holt/node";

const tree = Tree.openMemory();
tree.put(Buffer.from("bucket/a"), Buffer.from("metadata"));
console.log(tree.get(Buffer.from("bucket/a"))?.toString());
console.log(tree.scanKeys(Buffer.from("bucket/")));
tree.close();
```

Multiple named trees can share one database, WAL, and checkpoint boundary:

```ts
import { Database } from "@holt/node";

const db = Database.open("/var/lib/app/holt", { walSync: true });
const objects = db.openOrCreateTree("objects");
const sessions = db.openOrCreateTree("sessions");

objects.put(Buffer.from("bucket/a"), Buffer.from("object metadata"));
sessions.put(Buffer.from("session/1"), Buffer.from("session metadata"));

console.log(db.listTrees());
db.checkpoint();
objects.close();
sessions.close();
db.close();
```

Build the native artifact locally with:

```sh
npm install
npm run build
```

The repository intentionally does not commit platform-specific `.node`
artifacts. A release pipeline should build and publish one napi-rs platform
package per supported Unix target, then attach them to the main package as
optional dependencies.
