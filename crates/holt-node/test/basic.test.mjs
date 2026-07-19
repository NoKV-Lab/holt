import assert from "node:assert/strict";
import test from "node:test";
import { Database, Tree } from "../index.js";

test("storage operations expose native Promises", async () => {
  const open = Tree.openMemory();
  assert(open instanceof Promise);
  const tree = await open;

  const put = tree.put(Buffer.from("promise/key"), Buffer.from("value"));
  assert(put instanceof Promise);
  await put;

  const get = tree.get(Buffer.from("promise/key"));
  assert(get instanceof Promise);
  await get;

  const scan = tree.scanKeys(Buffer.from("promise/"));
  assert(scan instanceof Promise);
  await scan;

  const checkpoint = tree.checkpoint();
  assert(checkpoint instanceof Promise);
  await checkpoint;

  const close = tree.close();
  assert(close instanceof Promise);
  await close;
});

test("memory CRUD and conditional update", async () => {
  const tree = await Tree.openMemory();
  const key = Buffer.from("bucket/a.jpg");
  await tree.put(key, Buffer.from("old"));
  assert.deepEqual(await tree.get(key), Buffer.from("old"));

  const record = await tree.getRecord(key);
  assert.equal(typeof record.version, "bigint");
  assert.equal(
    await tree.compareAndPut(key, record.version, Buffer.from("new")),
    true,
  );
  assert.deepEqual(await tree.get(key), Buffer.from("new"));
  assert.equal(await tree.delete(key), true);
  assert.equal(await tree.get(key), null);
  await tree.close();
  await tree.close();
});

test("prefix and delimiter scans", async () => {
  const tree = await Tree.openMemory();
  for (const key of ["bucket/a/1", "bucket/a/2", "bucket/b/1"]) {
    await tree.put(Buffer.from(key), Buffer.from(key));
  }
  const entries = await tree.scanKeys(Buffer.from("bucket/"), {
    delimiter: 47,
  });
  assert.deepEqual(
    entries.map((entry) => [entry.kind, entry.path.toString()]),
    [
      ["common_prefix", "bucket/a/"],
      ["common_prefix", "bucket/b/"],
    ],
  );
  await tree.close();
});

test("database creates and isolates named trees", async () => {
  const db = await Database.openMemory();
  const objects = await db.createTree("objects");
  const sessions = await db.openOrCreateTree("sessions");
  const key = Buffer.from("same-key");

  await objects.put(key, Buffer.from("object-value"));
  await sessions.put(key, Buffer.from("session-value"));
  assert.deepEqual(await objects.get(key), Buffer.from("object-value"));
  assert.deepEqual(await sessions.get(key), Buffer.from("session-value"));
  assert.deepEqual((await db.listTrees()).sort(), ["objects", "sessions"]);

  const secondObjectsHandle = await db.openTree("objects");
  assert.deepEqual(
    await secondObjectsHandle.get(key),
    Buffer.from("object-value"),
  );
  await secondObjectsHandle.close();

  await db.dropTree("sessions");
  assert.deepEqual(await db.listTrees(), ["objects"]);
  await assert.rejects(sessions.get(key), /dropped/i);

  await sessions.close();
  await objects.close();
  await db.close();
  await db.close();
});
