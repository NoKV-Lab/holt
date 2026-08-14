# Security Policy

## Supported versions

`holt` is pre-1.0; only the latest published `0.x.y` release on
[crates.io](https://crates.io/crates/holt) receives security
fixes. Older `0.x` releases are not patched — upgrade to the
current `0.x.y` to receive any fixes.

Once `1.0` ships, the support window expands to the current
minor release plus the previous minor.

| Version | Supported |
|---------|-----------|
| `0.9.1` | Yes (current) |
| `< 0.9.1` | No |

## Reporting a vulnerability

**Please do NOT open a public issue for security reports.**

Use GitHub's **[private vulnerability reporting](https://github.com/NoKV-Lab/holt/security/advisories/new)**
(Security tab → "Report a vulnerability"). The maintainers
receive a private advisory thread; you receive a CVE if one
gets assigned during the fix.

If you can't use GitHub's advisory flow, email the maintainer
listed in [`Cargo.toml`](Cargo.toml)'s `authors`. Encrypt with
the PGP key on the author's GitHub profile if the issue is
sensitive.

### What to include

- A minimal reproducer (failing test is ideal).
- Affected version (`cargo tree --invert holt` output if it's
  in a downstream dep chain).
- Impact assessment (memory safety, data loss, denial of service,
  …) and how you arrived at it.
- Output of `rustc --version --verbose`.

### What to expect

- Acknowledgement within 7 days.
- Fix + advisory + new release within 30 days for "confirmed,
  exploitable" issues; longer for low-severity / theoretical
  reports.
- Public credit in the advisory + CHANGELOG entry, unless you
  request anonymity.

## Out of scope

- **Windows support** — the crate fails to compile on Windows
  by design ([`#[cfg(not(unix))] compile_error!`](src/lib.rs));
  bug reports about Windows compatibility are closed as out of
  scope, not security issues.
- **Random-key workload performance** — ART is fundamentally
  `O(key.len)`; sub-optimal performance on random 32-byte
  keys (`*kv*` bench scenario) is a design constraint, not a
  vulnerability.
