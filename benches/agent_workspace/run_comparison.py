#!/usr/bin/env python3
"""Serial, balanced runs of the shared workspace workflow; retain raw evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import tomllib


CASES = {
    "metadata_async_1": {"ops": 12000, "threads": 1, "payload": 0, "sync": False},
    "metadata_async_4": {"ops": 12000, "threads": 4, "payload": 0, "sync": False},
    "metadata_sync_1": {"ops": 4000, "threads": 1, "payload": 0, "sync": True},
    "content_async_1": {"ops": 6000, "threads": 1, "payload": 4096, "sync": False},
    "content_async_4": {"ops": 6000, "threads": 4, "payload": 4096, "sync": False},
    "content_sync_1": {"ops": 1600, "threads": 1, "payload": 4096, "sync": True},
    "attached_metadata_sync_1": {"ops": 4000, "threads": 1, "payload": 0, "sync": True, "attached": True},
    "attached_content_sync_1": {"ops": 1600, "threads": 1, "payload": 4096, "sync": True, "attached": True},
    "metadata_point_async_1": {"ops": 12000, "threads": 1, "payload": 0, "sync": False, "read-mode": "point"},
    "metadata_async_4_short": {"ops": 4000, "threads": 4, "payload": 0, "sync": False},
    "metadata_async_4_background": {"ops": 4000, "threads": 4, "payload": 0, "sync": False, "background": True},
    "content_async_4_background": {"ops": 4000, "threads": 4, "payload": 4096, "sync": False, "background": True},
    "metadata_async_4_background_normal": {"ops": 4000, "threads": 4, "payload": 0, "sync": False, "background": True, "sqlite-normal": True},
    "content_async_4_background_normal": {"ops": 4000, "threads": 4, "payload": 4096, "sync": False, "background": True, "sqlite-normal": True},
}


def capture(command):
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {"command": command, "exit_code": result.returncode, "output": result.stdout}
    except OSError as error:
        return {"command": command, "exit_code": None, "output": str(error)}


def environment():
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "cpu": capture(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu"]),
        "load": os.getloadavg(),
        "thermal": capture(["pmset", "-g", "therm"]),
        "processes": capture(["ps", "-A", "-o", "pid,pcpu,comm"]),
    }


def domain_fingerprint(row):
    value = json.loads(json.dumps(row["fingerprint"]))
    for table in value["tables"].values():
        table.pop("blobs", None)
    return value


def summarize(rows):
    summaries = {}
    for name in sorted({row["case"] for row in rows}):
        summary = {}
        for engine in ["holt", "sqlite"]:
            data = [row for row in rows if row["case"] == name and row["engine"] == engine]
            median = lambda f: statistics.median(f(row) for row in data)
            summary[engine] = {
                "runs": len(data),
                "ops_per_second": median(lambda r: r["ops_per_second"]),
                "ops_min": min(r["ops_per_second"] for r in data),
                "ops_max": max(r["ops_per_second"] for r in data),
                "p50_us": median(lambda r: r["latency"]["p50_us"]),
                "p95_us": median(lambda r: r["latency"]["p95_us"]),
                "p99_us": median(lambda r: r["latency"]["p99_us"]),
                "checkpoint_ms": median(lambda r: r["checkpoint_ms"]),
                "reopen_ms": median(lambda r: r["reopen_ms"]),
                "drain_ops_per_second": median(lambda r: r["drain_inclusive_ops_per_second"]),
                "metadata_allocated_mib": median(lambda r: r["space_after"]["metadata_allocated"] / 2**20),
                "metadata_logical_mib": median(lambda r: r["space_after"]["metadata_logical"] / 2**20),
                "view_ms": median(lambda r: r["stages"]["view_ms"]),
                "apply_ms": median(lambda r: r["stages"]["apply_ms"]),
                "object_ms": median(lambda r: r["stages"]["object_ms"]),
                "commits": median(lambda r: r["stages"]["metadata_commits"]),
                "cas_retries": sum(r["stages"]["cas_retries"] for r in data),
                "max_rss_mib": median(lambda r: r.get("maximum_resident_set_bytes", 0) / 2**20),
                "per_operation": {},
            }
            for operation in data[0]["per_operation"]:
                summary[engine]["per_operation"][operation] = {
                    key: median(lambda r: r["per_operation"][operation][key])
                    for key in ["mean_us", "p50_us", "p95_us", "p99_us", "count"]
                }
        summary["holt_vs_sqlite_throughput_ratio"] = summary["holt"]["ops_per_second"] / summary["sqlite"]["ops_per_second"]
        summaries[name] = summary
    return summaries


def write_report(out, rows, evidence):
    for row in rows:
        label = f"{row['case']}-{row['round']:02}-{row['engine']}"
        resource = (out / f"{label}.stderr.txt").read_text()
        match = re.search(r"(\d+)\s+maximum resident set size", resource)
        if match:
            row["maximum_resident_set_bytes"] = int(match.group(1))
        else:
            match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", resource)
            if match:
                row["maximum_resident_set_bytes"] = int(match.group(1)) * 1024
    summaries = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Holt / SQLite agent-workspace workflow comparison",
        "",
        "Application-level end-to-end boundary: JSON request encode/decode → shared workspace handler → metadata transaction/read view → optional local content file I/O + checksum → JSON response roundtrip. This is not a NoKV SDK/RPC or remote-object-provider benchmark.",
        "",
        f"Holt source: `{evidence['commit']}`; local library package {evidence.get('holt_version', '0.9.2')}. SQLite version is recorded in every raw row. All runs use the exact same optimized executable.",
        "",
        "Each run starts from 8 workspaces × 512 existing paths. Eight identically encoded metadata families map to Holt named trees and SQLite `WITHOUT ROWID` tables. Holt has a shared 64 MiB cache target; SQLite has a 64 MiB target per connection. These knobs are not a total-memory equivalence: measured SQLite RSS is below 20 MiB, while Holt can exceed its target through protected COW/dirty frames. Workload: 40% read, 15% stat, 15% list (64 records), 20% publish, 5% rename roundtrip, 3% remove+recreate, 2% consistent 8-family snapshot. Request choices are deterministic, so actual counts are reported per row.",
        "",
        "Single-client runs use one active workspace. Four clients use distinct workspaces in one shared DB; they still share metadata families. Rename and remove+recreate contain two metadata commits. Publish updates all eight families. Immutable content and history/receipts are retained. Setup and warmup are excluded; final checkpoint is reported separately and also included in a second throughput metric.",
        "",
        "Default read semantics: Holt `DB::view` and SQLite read transactions. Diagnostic `point` rows weaken only read/stat cross-tree consistency and are excluded from the main apples-to-apples result. Base cases disable background checkpoints and use an explicit final drain. Cases marked `background` enable Holt's native 500 ms/512-dirty-blob policy and SQLite's native 1000-page WAL autocheckpoint. Case `_normal` uses SQLite WAL synchronous=NORMAL; other async cases use OFF. These policies do not imply identical checkpoint cadence or bytes per round. These are bounded workload measurements, not a long-running maintenance qualification.",
        "",
        "Async diagnostic: Holt `Wal { sync:false }`, SQLite WAL `synchronous=OFF`; acknowledgements and crash consistency differ, so OFF is not a production-equivalent durability claim. Background `_normal` rows use SQLite WAL NORMAL (sync at checkpoint/log reuse) and Holt async WAL with background durable checkpoints; they still have different buffering and checkpoint policies. Sync: Holt `sync:true`, SQLite `synchronous=FULL`, **`fullfsync=ON` and `checkpoint_fullfsync=ON`** on macOS, matching Rust's F_FULLFSYNC path. Content writes in sync mode sync both the file and its parent directory for both engines. Final drains use durable checkpoint settings in every profile. No power-cut qualification was performed.",
        "",
        "Attached rows add the same canonical mutation bytes to Holt's attached WAL envelope and SQLite's transactional recovery table. A final durable checkpoint retires this local recovery material. The profile measures one ordered recovery funnel and does not claim shared-log or indefinite retention semantics.",
        "",
        "Every run verifies exact replay without another commit and checks the full ordered-domain fingerprint before and after reopen. Every engine and repetition within a case must produce identical domain fingerprints. Hardware/OS/load and raw command lines are preserved beside this report. Desktop background load limits portable claims about absolute timings.",
        "",
        "Reported values are medians across independent process runs. The balanced order is H,S,S,H / S,H,H,S; full per-run ranges remain in summary.json. p95/p99 are medians of per-run quantiles, not pooled quantiles.",
        "",
        "| Case | H ops/s | S ops/s | H/S | H p50 µs | S p50 µs | H p95 µs | S p95 µs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        h, s = summary["holt"], summary["sqlite"]
        lines.append(f"| {name} | {h['ops_per_second']:,.0f} | {s['ops_per_second']:,.0f} | {summary['holt_vs_sqlite_throughput_ratio']:.2f}× | {h['p50_us']:.1f} | {s['p50_us']:.1f} | {h['p95_us']:.1f} | {s['p95_us']:.1f} |")
    lines += ["", "## Checkpoint and final metadata footprint", "", "Content bytes are excluded from this table. No vacuum is run. Both logical bytes and allocated filesystem blocks are retained in the raw data.", "", "| Case | H checkpoint ms | S checkpoint ms | H metadata allocated MiB | S metadata allocated MiB | H peak RSS MiB | S peak RSS MiB | H drain-inclusive ops/s | S drain-inclusive ops/s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        h, s = summary["holt"], summary["sqlite"]
        lines.append(f"| {name} | {h['checkpoint_ms']:.1f} | {s['checkpoint_ms']:.1f} | {h['metadata_allocated_mib']:.2f} | {s['metadata_allocated_mib']:.2f} | {h['max_rss_mib']:.1f} | {s['max_rss_mib']:.1f} | {h['drain_ops_per_second']:,.0f} | {s['drain_ops_per_second']:,.0f} |")
    for name, summary in summaries.items():
        lines += ["", f"## {name}: operation latency", "", "| Operation | H p50 µs | S p50 µs | H p95 µs | S p95 µs |", "|---|---:|---:|---:|---:|"]
        h, s = summary["holt"], summary["sqlite"]
        for operation in h["per_operation"]:
            a, b = h["per_operation"][operation], s["per_operation"][operation]
            lines.append(f"| {operation} | {a['p50_us']:.1f} | {b['p50_us']:.1f} | {a['p95_us']:.1f} | {b['p95_us']:.1f} |")
        lines += ["", f"Stage totals (median, sums across clients): H view+reads {h['view_ms']:.1f} ms, apply {h['apply_ms']:.1f} ms, objects {h['object_ms']:.1f} ms; S view+reads {s['view_ms']:.1f} ms, apply {s['apply_ms']:.1f} ms, objects {s['object_ms']:.1f} ms. These are inclusive stage timers, not pure lock-wait or snapshot-copy timings. Metadata commits: H {h['commits']}, S {s['commits']}; CAS retries H {h['cas_retries']}, S {s['cas_retries']}."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--rounds", type=int, choices=[2, 4, 8], default=8)
    parser.add_argument("--keep-stores", action="store_true", help="Retain generated test databases; full suites can use tens of GiB")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = args.cases.split(",")
    for name in selected:
        if name not in CASES:
            parser.error(f"unknown case {name}")
    evidence = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "rust": capture(["rustc", "--version"]),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "holt_version": tomllib.loads((Path(__file__).resolve().parents[2] / "Cargo.toml").read_text())["package"]["version"],
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ["Cargo.toml", "Cargo.lock", "src/main.rs", "run_comparison.py"]
        },
        "cases": {name: CASES[name] for name in selected},
        "initial_environment": environment(),
    }
    (out / "environment.json").write_text(json.dumps(evidence, indent=2) + "\n")
    sequence = ["holt", "sqlite", "sqlite", "holt", "sqlite", "holt", "holt", "sqlite"][:args.rounds]
    rows = []
    for case in selected:
        expected = None
        for number, engine in enumerate(sequence, 1):
            label = f"{case}-{number:02}-{engine}"
            directory = out / "stores" / label
            directory.parent.mkdir(exist_ok=True)
            command = [str(args.binary.resolve()), "--engine", engine, "--dir", str(directory), "--warmup", "200", "--workspaces", "8", "--files", "512"]
            for key, value in CASES[case].items():
                command += [f"--{key}", str(value).lower()]
            (out / f"{label}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            before = environment()
            started = time.monotonic()
            resource_flag = "-l" if platform.system() == "Darwin" else "-v"
            process = subprocess.run(["/usr/bin/time", resource_flag, *command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out / f"{label}.stdout.json").write_text(process.stdout)
            (out / f"{label}.stderr.txt").write_text(process.stderr)
            (out / f"{label}.environment.json").write_text(json.dumps({"before": before, "after": environment(), "process_seconds": time.monotonic() - started}, indent=2) + "\n")
            if process.returncode:
                print(process.stderr, file=sys.stderr)
                raise SystemExit(f"{label} failed: {process.returncode}")
            row = json.loads(process.stdout)
            fingerprint = domain_fingerprint(row)
            if expected is None:
                expected = fingerprint
            if expected != fingerprint:
                raise SystemExit(f"domain state mismatch: {label}")
            row["case"], row["round"] = case, number
            rows.append(row)
            (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
            if not args.keep_stores:
                assert directory.parent == out / "stores" and not directory.is_symlink()
                shutil.rmtree(directory)
            print(f"{label}: {row['ops_per_second']:.0f} ops/s, p50={row['latency']['p50_us']:.1f}us p95={row['latency']['p95_us']:.1f}us, cp={row['checkpoint_ms']:.1f}ms, CRC={fingerprint['crc32']}", flush=True)
        write_report(out, rows, evidence)
    evidence["final_environment"] = environment()
    (out / "environment.json").write_text(json.dumps(evidence, indent=2) + "\n")
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(out / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
