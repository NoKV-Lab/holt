import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { Database, Tree } from "../index.js";

test("file-backed tree reopens after checkpoint", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "holt-node-"));
  const key = Buffer.from("objects/1");
  const first = Tree.open(dir, { walSync: true });
  first.put(key, Buffer.from("metadata"));
  first.checkpoint();
  first.close();

  const second = Tree.open(dir, { walSync: true });
  assert.deepEqual(second.get(key), Buffer.from("metadata"));
  second.close();
  await fs.rm(dir, { recursive: true, force: true });
});

test("file-backed database reopens named trees", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "holt-node-db-"));
  const key = Buffer.from("objects/1");
  const first = Database.open(dir, { walSync: true });
  const objects = first.createTree("objects");
  const metadata = first.createTree("metadata");
  objects.put(key, Buffer.from("object-value"));
  metadata.put(key, Buffer.from("metadata-value"));
  first.checkpoint();
  objects.close();
  metadata.close();
  first.close();

  const second = Database.open(dir, { walSync: true });
  assert.deepEqual(second.listTrees().sort(), ["metadata", "objects"]);
  const reopenedObjects = second.openTree("objects");
  const reopenedMetadata = second.openTree("metadata");
  assert.deepEqual(reopenedObjects.get(key), Buffer.from("object-value"));
  assert.deepEqual(reopenedMetadata.get(key), Buffer.from("metadata-value"));
  reopenedObjects.close();
  reopenedMetadata.close();
  second.close();
  await fs.rm(dir, { recursive: true, force: true });
});
