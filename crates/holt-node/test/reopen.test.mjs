import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { Database, Tree } from "../index.js";

test("file-backed tree reopens after checkpoint", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "holt-node-"));
  const key = Buffer.from("objects/1");
  const first = await Tree.open(dir, { walSync: true });
  await first.put(key, Buffer.from("metadata"));
  await first.checkpoint();
  await first.close();

  const second = await Tree.open(dir, { walSync: true });
  assert.deepEqual(await second.get(key), Buffer.from("metadata"));
  await second.close();
  await fs.rm(dir, { recursive: true, force: true });
});

test("file-backed database reopens named trees", async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "holt-node-db-"));
  const key = Buffer.from("objects/1");
  const first = await Database.open(dir, { walSync: true });
  const objects = await first.createTree("objects");
  const metadata = await first.createTree("metadata");
  await objects.put(key, Buffer.from("object-value"));
  await metadata.put(key, Buffer.from("metadata-value"));
  await first.checkpoint();
  await objects.close();
  await metadata.close();
  await first.close();

  const second = await Database.open(dir, { walSync: true });
  assert.deepEqual((await second.listTrees()).sort(), ["metadata", "objects"]);
  const reopenedObjects = await second.openTree("objects");
  const reopenedMetadata = await second.openTree("metadata");
  assert.deepEqual(
    await reopenedObjects.get(key),
    Buffer.from("object-value"),
  );
  assert.deepEqual(
    await reopenedMetadata.get(key),
    Buffer.from("metadata-value"),
  );
  await reopenedObjects.close();
  await reopenedMetadata.close();
  await second.close();
  await fs.rm(dir, { recursive: true, force: true });
});
