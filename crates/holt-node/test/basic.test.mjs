import assert from "node:assert/strict";
import test from "node:test";
import { Database, Tree } from "../index.js";

test("memory CRUD and conditional update", () => {
  const tree = Tree.openMemory();
  const key = Buffer.from("bucket/a.jpg");
  tree.put(key, Buffer.from("old"));
  assert.deepEqual(tree.get(key), Buffer.from("old"));

  const record = tree.getRecord(key);
  assert.equal(typeof record.version, "bigint");
  assert.equal(tree.compareAndPut(key, record.version, Buffer.from("new")), true);
  assert.deepEqual(tree.get(key), Buffer.from("new"));
  assert.equal(tree.delete(key), true);
  assert.equal(tree.get(key), null);
  tree.close();
  tree.close();
});

test("prefix and delimiter scans", () => {
  const tree = Tree.openMemory();
  for (const key of ["bucket/a/1", "bucket/a/2", "bucket/b/1"]) {
    tree.put(Buffer.from(key), Buffer.from(key));
  }
  const entries = tree.scanKeys(Buffer.from("bucket/"), { delimiter: 47 });
  assert.deepEqual(
    entries.map((entry) => [entry.kind, entry.path.toString()]),
    [
      ["common_prefix", "bucket/a/"],
      ["common_prefix", "bucket/b/"],
    ],
  );
  tree.close();
});

test("database creates and isolates named trees", () => {
  const db = Database.openMemory();
  const objects = db.createTree("objects");
  const sessions = db.openOrCreateTree("sessions");
  const key = Buffer.from("same-key");

  objects.put(key, Buffer.from("object-value"));
  sessions.put(key, Buffer.from("session-value"));
  assert.deepEqual(objects.get(key), Buffer.from("object-value"));
  assert.deepEqual(sessions.get(key), Buffer.from("session-value"));
  assert.deepEqual(db.listTrees().sort(), ["objects", "sessions"]);

  const secondObjectsHandle = db.openTree("objects");
  assert.deepEqual(secondObjectsHandle.get(key), Buffer.from("object-value"));
  secondObjectsHandle.close();

  db.dropTree("sessions");
  assert.deepEqual(db.listTrees(), ["objects"]);
  assert.throws(() => sessions.get(key), /dropped/i);

  sessions.close();
  objects.close();
  db.close();
  db.close();
});
