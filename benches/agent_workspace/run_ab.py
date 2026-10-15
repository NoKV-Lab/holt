#!/usr/bin/env python3
"""Balanced baseline / patched / SQLite runs of the unchanged workspace driver."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from run_comparison import CASES, capture, domain_fingerprint, environment as base_environment


def environment():
    value = base_environment()
    value["disk_usage_bytes"] = shutil.disk_usage(Path(__file__).resolve())._asdict()
    return value


def report(out, rows):
    summary = {}
    for case in sorted({r["case"] for r in rows}):
        summary[case] = {}
        for variant in ["baseline", "patched", "sqlite"]:
            selected = [r for r in rows if r["case"] == case and r["variant"] == variant]
            if not selected:
                continue
            median = lambda f: statistics.median(f(r) for r in selected)
            summary[case][variant] = {
                "runs": len(selected),
                "ops_per_second": median(lambda r: r["ops_per_second"]),
                "ops_range": [min(r["ops_per_second"] for r in selected), max(r["ops_per_second"] for r in selected)],
                "drain_ops_per_second": median(lambda r: r["drain_inclusive_ops_per_second"]),
                "p50_us": median(lambda r: r["latency"]["p50_us"]),
                "p95_us": median(lambda r: r["latency"]["p95_us"]),
                "p99_us": median(lambda r: r["latency"]["p99_us"]),
                "rss_mib": median(lambda r: r["maximum_resident_set_bytes"] / 2**20),
                "metadata_mib": median(lambda r: r["space_after"]["metadata_allocated"] / 2**20),
                "checkpoint_ms": median(lambda r: r["checkpoint_ms"]),
                "view_ms": median(lambda r: r["stages"]["view_ms"]),
                "apply_ms": median(lambda r: r["stages"]["apply_ms"]),
                "cas_retries": sum(r["stages"]["cas_retries"] for r in selected),
                "orphan_backlog": median(lambda r: (r["holt_stats"] or {}).get("orphan_backlog", 0)),
                "live_slots": median(lambda r: (r["holt_stats"] or {}).get("live_slots", 0)),
                "per_operation": {
                    op: {metric: median(lambda r: r["per_operation"][op][metric]) for metric in ["p50_us", "p95_us", "p99_us", "mean_us"]}
                    for op in selected[0]["per_operation"]
                },
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Agent workspace: baseline / patched Holt / SQLite",
        "",
        "Unchanged shared workspace handler, workload, dataset, consistency, durability and checkpoint policies. Four independent processes per variant in balanced order; medians of per-process metrics. No concurrent compiler/tests/benchmarks during these runs. Each database is fresh and removed only after full domain fingerprint and clean-reopen validation. Raw samples, commands, resource logs and environment remain alongside this report.",
        "",
        "Async OFF is diagnostic, not crash-equivalent to Holt. Background `_normal` uses native checkpoint policies (Holt 500 ms/512 blobs, SQLite NORMAL/1000 pages), not matched checkpoint cadence. Sync uses Holt sync=true and SQLite FULL with macOS fullfsync/checkpoint_fullfsync. These are bounded desktop workloads, not long-running maintenance or power-cut qualification.",
        "",
        "| Case | Old H ops/s | New H ops/s | SQLite ops/s | New/old | New/SQLite | Old/new p95 us |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, variants in summary.items():
        if len(variants) != 3:
            continue
        a, b, s = (variants[k] for k in ["baseline", "patched", "sqlite"])
        lines.append(f"| {case} | {a['ops_per_second']:,.0f} | {b['ops_per_second']:,.0f} | {s['ops_per_second']:,.0f} | {b['ops_per_second']/a['ops_per_second']:.2f}x | {b['ops_per_second']/s['ops_per_second']:.2f}x | {a['p95_us']:.1f} / {b['p95_us']:.1f} |")
    lines += ["", "| Case | Old/new RSS MiB | Old/new metadata MiB | Old/new final checkpoint ms | Old/new post-drain orphan debt |", "|---|---:|---:|---:|---:|"]
    for case, variants in summary.items():
        if "baseline" not in variants or "patched" not in variants:
            continue
        a, b = variants["baseline"], variants["patched"]
        lines.append(f"| {case} | {a['rss_mib']:.1f} / {b['rss_mib']:.1f} | {a['metadata_mib']:.1f} / {b['metadata_mib']:.1f} | {a['checkpoint_ms']:.1f} / {b['checkpoint_ms']:.1f} | {a['orphan_backlog']:.0f} / {b['orphan_backlog']:.0f} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(k for k in CASES if "point" not in k and ("background" not in k or k.endswith("normal"))))
    parser.add_argument("--rounds", type=int, choices=[3, 6, 12], default=12)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected = args.cases.split(",")
    if any(k not in CASES for k in selected):
        parser.error("unknown case")
    binaries = {"baseline": args.baseline.resolve(), "patched": args.patched.resolve(), "sqlite": args.patched.resolve()}
    evidence = {
        "commit": capture(["git", "rev-parse", "HEAD"]),
        "diff": capture(["git", "diff", "--", "src", "tests", "ARCHITECTURE.md"]),
        "status": capture(["git", "status", "--short"]),
        "rust": capture(["rustc", "-Vv"]),
        "binary_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in binaries.items()},
        "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest() for name in ["Cargo.toml", "Cargo.lock", "src/main.rs", "run_ab.py", "run_comparison.py"]},
        "initial_environment": environment(),
        "cases": {k: CASES[k] for k in selected},
    }
    (out / "environment.json").write_text(json.dumps(evidence, indent=2) + "\n")
    # Forward/reverse matched triplets: each pair appears in each order twice.
    # Four processes per variant, twelve total.
    sequence = ["baseline", "patched", "sqlite", "sqlite", "patched", "baseline", "patched", "baseline", "sqlite", "sqlite", "baseline", "patched"][:args.rounds]
    rows = []
    for case in selected:
        expected = None
        for number, variant in enumerate(sequence, 1):
            label = f"{case}-{number:02}-{variant}"
            directory = out / "stores" / label
            directory.parent.mkdir(exist_ok=True)
            command = [str(binaries[variant]), "--engine", "sqlite" if variant == "sqlite" else "holt", "--dir", str(directory), "--warmup", "200", "--workspaces", "8", "--files", "512"]
            for key, value in CASES[case].items():
                command += [f"--{key}", str(value).lower()]
            before = environment()
            started = time.monotonic()
            (out / f"{label}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            process = subprocess.run(["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v", *command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out / f"{label}.stdout.json").write_text(process.stdout)
            (out / f"{label}.stderr.txt").write_text(process.stderr)
            (out / f"{label}.environment.json").write_text(json.dumps({"before": before, "after": environment(), "process_seconds": time.monotonic() - started}, indent=2) + "\n")
            if process.returncode:
                raise SystemExit(f"{label} failed: {process.returncode}\n{process.stderr}")
            row = json.loads(process.stdout)
            fingerprint = domain_fingerprint(row)
            if expected is None:
                expected = fingerprint
            if expected != fingerprint or not row["verified_reopen"]:
                raise SystemExit(f"fingerprint/reopen mismatch: {label}")
            match = re.search(r"(\d+)\s+maximum resident set size", process.stderr)
            if match:
                rss = int(match.group(1))
            else:
                match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", process.stderr)
                if not match:
                    raise SystemExit(f"missing peak RSS: {label}")
                rss = int(match.group(1)) * 1024
            row.update(case=case, round=number, variant=variant, maximum_resident_set_bytes=rss)
            rows.append(row)
            (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
            assert directory.parent == out / "stores" and not directory.is_symlink()
            shutil.rmtree(directory)
            print(f"{label}: {row['ops_per_second']:.0f} ops/s, RSS {rss/2**20:.1f} MiB, cp {row['checkpoint_ms']:.1f} ms, CRC {fingerprint['crc32']}", flush=True)
        report(out, rows)
    evidence["final_environment"] = environment()
    (out / "environment.json").write_text(json.dumps(evidence, indent=2) + "\n")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
