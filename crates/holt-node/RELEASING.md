# Releasing the Node package

The Node package has its own version and tag sequence. A tag named
`node-v0.1.0` releases package version `0.1.0`. Rust crate tags still use the
`v0.8.2` form.

The workflow publishes one root package and four platform packages. npm treats
each published version as immutable. Do not push a release tag until every
check in this file passes.

## Initial setup

1. Confirm that the npm organization `nokv-lab` exists and that the release
   account can publish `@nokv-lab/holt` and all four suffixed package names.
   An available package name alone does not grant scope access.
2. Add a GitHub Actions secret named `NPM_TOKEN`. Use a granular npm token with
   read and write access and bypass 2FA enabled for CI publishing.
3. Protect `node-v*` tags so that only release maintainers can create them.

After the initial release, configure `npm-release.yml` as the trusted publisher
for the root and platform packages. Then remove `NPM_TOKEN` from the workflow
and revoke the token.

## Prepare a release

1. Update `version` in `package.json` and `Cargo.toml` under this directory.
2. Run `npm install --package-lock-only` to update `package-lock.json`.
3. Run the local checks:

   ```sh
   npm ci
   npm run build
   npm test
   npm audit
   npm pack --dry-run --ignore-scripts
   ```

4. Merge the version change into `main` and wait for the Node package workflow
   to pass.
5. Create and push the release tag:

   ```sh
   git tag -s node-v0.1.0 -m "Release @nokv-lab/holt 0.1.0"
   git push origin node-v0.1.0
   ```

The workflow verifies the tag, builds each target, tests each native artifact,
checks every tarball, and publishes the platform packages before the root
package. A rerun skips package versions that already exist. This supports
recovery from a partial npm release without replacing published files.

## Verify the release

Check the root package and each platform package after the workflow completes:

```sh
npm view @nokv-lab/holt@0.1.0 version dist.integrity repository
npm view @nokv-lab/holt-darwin-x64@0.1.0 version dist.integrity
npm view @nokv-lab/holt-darwin-arm64@0.1.0 version dist.integrity
npm view @nokv-lab/holt-linux-x64-gnu@0.1.0 version dist.integrity
npm view @nokv-lab/holt-linux-arm64-gnu@0.1.0 version dist.integrity
```

Install the root package in a clean project on each supported target. Run a
memory operation and a file-backed checkpoint and reopen test.
