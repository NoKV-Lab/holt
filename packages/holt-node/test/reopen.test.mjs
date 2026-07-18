import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { Tree } from "../index.js";

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
