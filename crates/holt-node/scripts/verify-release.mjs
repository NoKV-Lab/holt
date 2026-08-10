import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const targetMetadata = {
  "x86_64-apple-darwin": {
    suffix: "darwin-x64",
    os: "darwin",
    cpu: "x64",
  },
  "aarch64-apple-darwin": {
    suffix: "darwin-arm64",
    os: "darwin",
    cpu: "arm64",
  },
  "x86_64-unknown-linux-gnu": {
    suffix: "linux-x64-gnu",
    os: "linux",
    cpu: "x64",
    libc: "glibc",
  },
  "aarch64-unknown-linux-gnu": {
    suffix: "linux-arm64-gnu",
    os: "linux",
    cpu: "arm64",
    libc: "glibc",
  },
};

const readJson = (file) => JSON.parse(fs.readFileSync(file, "utf8"));
const root = process.cwd();
const packageJson = readJson(path.join(root, "package.json"));
const configuredTargets = packageJson.napi?.targets ?? [];
const license = fs.readFileSync(path.join(root, "LICENSE"));

assert.deepEqual(
  license,
  fs.readFileSync(path.resolve(root, "../../LICENSE")),
  "the Node package license must match the repository license",
);

assert.deepEqual(
  [...configuredTargets].sort(),
  Object.keys(targetMetadata).sort(),
  "package.json must configure the reviewed release targets",
);

const expectedDependencies = {};

for (const triple of configuredTargets) {
  const target = targetMetadata[triple];
  const binary = `index.${target.suffix}.node`;
  const rootBinary = path.join(root, binary);
  const packageDirectory = path.join(root, "npm", target.suffix);
  const platformBinary = path.join(packageDirectory, binary);
  const platformPackage = readJson(path.join(packageDirectory, "package.json"));
  const expectedName = `${packageJson.name}-${target.suffix}`;

  assert.ok(fs.statSync(rootBinary).size > 0, `${binary} is missing or empty`);
  assert.ok(
    fs.statSync(platformBinary).size > 0,
    `${path.relative(root, platformBinary)} is missing or empty`,
  );
  assert.deepEqual(
    fs.readFileSync(path.join(packageDirectory, "LICENSE")),
    license,
    `${path.relative(root, packageDirectory)} has the wrong license text`,
  );
  assert.equal(platformPackage.name, expectedName);
  assert.equal(platformPackage.version, packageJson.version);
  assert.equal(platformPackage.main, binary);
  assert.deepEqual(platformPackage.files, [binary]);
  assert.deepEqual(platformPackage.os, [target.os]);
  assert.deepEqual(platformPackage.cpu, [target.cpu]);
  assert.deepEqual(
    platformPackage.libc,
    target.libc ? [target.libc] : undefined,
  );
  assert.equal(platformPackage.publishConfig?.access, "public");
  assert.equal(
    platformPackage.repository?.url,
    packageJson.repository.url,
  );
  assert.equal(
    platformPackage.repository?.directory,
    packageJson.repository.directory,
  );

  expectedDependencies[expectedName] = packageJson.version;
}

assert.deepEqual(
  packageJson.optionalDependencies,
  expectedDependencies,
  "optionalDependencies must contain each platform package at the exact release version",
);

console.log(
  `Verified ${packageJson.name}@${packageJson.version} and ${configuredTargets.length} platform packages.`,
);
