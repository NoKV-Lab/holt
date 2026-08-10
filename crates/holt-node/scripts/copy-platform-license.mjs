import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const npmDirectory = path.join(root, "npm");
const license = path.join(root, "LICENSE");
const packageDirectories = fs
  .readdirSync(npmDirectory, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => path.join(npmDirectory, entry.name));

if (packageDirectories.length === 0) {
  throw new Error("napi did not create any platform package directories");
}

for (const packageDirectory of packageDirectories) {
  fs.copyFileSync(license, path.join(packageDirectory, "LICENSE"));
}

console.log(`Copied LICENSE into ${packageDirectories.length} platform packages.`);
