#!/usr/bin/env python3
"""
write_run_manifest.py - Generates metadata manifest (meta.json) for a benchmark run.
Standard library only (json, os, sys, pathlib, hashlib, subprocess, datetime).
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def parse_build_info(bin_path: str) -> dict:
    if not bin_path or not os.path.exists(bin_path):
        return {}
    try:
        res = subprocess.run([bin_path, "--build-info"], capture_output=True, text=True, check=True)
        info = {}
        for line in res.stdout.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
        return info
    except Exception as e:
        return {"error": str(e)}


def get_binary_sha256(bin_path: str) -> str:
    if not bin_path or not os.path.exists(bin_path):
        return "unknown"
    try:
        h = hashlib.sha256()
        with open(bin_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unknown"


def get_cpu_model() -> str:
    # Try lscpu first
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    # Fallback /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name") or line.startswith("Hardware") or line.startswith("cpu model"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def get_isa_flags_summary() -> str:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith(("flags", "Features", "isa")):
                    val = line.split(":", 1)[1].strip()
                    return val[:512]
    except Exception:
        pass
    return "unknown"


def get_cache_info(core: int) -> list:
    caches = []
    base_dir = pathlib.Path(f"/sys/devices/system/cpu/cpu{core}/cache")
    if not base_dir.exists():
        base_dir = pathlib.Path("/sys/devices/system/cpu/cpu0/cache")
    if not base_dir.exists():
        return caches

    for idx_dir in sorted(base_dir.glob("index*")):
        try:
            level_str = (idx_dir / "level").read_text().strip()
            type_str = (idx_dir / "type").read_text().strip()
            size_str = (idx_dir / "size").read_text().strip()
            ways_str = (idx_dir / "ways_of_associativity").read_text().strip() if (idx_dir / "ways_of_associativity").exists() else "0"
            line_str = (idx_dir / "coherency_line_size").read_text().strip() if (idx_dir / "coherency_line_size").exists() else "0"

            caches.append({
                "level": int(level_str) if level_str.isdigit() else 0,
                "type": type_str,
                "size": size_str,
                "ways_of_associativity": int(ways_str) if ways_str.isdigit() else 0,
                "coherency_line_size": int(line_str) if line_str.isdigit() else 0,
            })
        except Exception:
            continue
    return caches


def get_governor(core: int) -> str | None:
    gov_file = pathlib.Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
    if not gov_file.exists():
        gov_file = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_file.exists():
        try:
            return gov_file.read_text().strip()
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate run metadata manifest JSON")
    parser.add_argument("--output", required=True, help="Destination JSON path")
    parser.add_argument("--bin", required=True, help="Path to GemmBench binary")
    parser.add_argument("--date-start", required=True, help="Run start ISO 8601 UTC timestamp")
    parser.add_argument("--date-end", required=True, help="Run end ISO 8601 UTC timestamp")
    parser.add_argument("--git-commit", default="unknown", help="Repository git commit")
    parser.add_argument("--git-dirty", type=int, default=-1, help="Repository git dirty status (0/1/-1)")
    parser.add_argument("--pin-core", default="", help="Pinned CPU core ID or empty")
    parser.add_argument("--taskset-used", type=lambda s: s.lower() in ("true", "1", "yes", "on"), default=None, help="Explicit taskset pinning status boolean")
    parser.add_argument("--freq-start", type=int, default=0, help="Initial CPU frequency in kHz")
    parser.add_argument("--freq-end", type=int, default=0, help="Final CPU frequency in kHz")
    parser.add_argument("--platform", required=True, help="Platform name (e.g. avx2, avx512)")
    parser.add_argument("--platform-tag", default=None, help="Platform tag (e.g. zen4)")
    parser.add_argument("--compiler-cmd", required=True, help="Compiler command/path")
    parser.add_argument("--compiler-version", default="unknown", help="Compiler version string")
    parser.add_argument("--cxx-flags", default="", help="Effective CXX flags")
    parser.add_argument("--fastmath", type=lambda s: s.lower() in ("true", "1", "yes", "on"), default=True)
    parser.add_argument("--precision", default="fp64", choices=["fp32", "fp64"])
    parser.add_argument("--crr-only", type=lambda s: s.lower() in ("true", "1", "yes", "on"), default=False)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--fastpath", type=lambda s: s.lower() in ("true", "1", "yes", "on"), default=True)
    parser.add_argument("--c-cold", type=lambda s: s.lower() in ("true", "1", "yes", "on"), default=False)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--cooldown-sec", type=float, default=2.0)
    parser.add_argument("--sizes-json", default="[]", help="JSON array of [K, iters, warmup]")
    parser.add_argument("--kernels-json", default="[]", help="JSON array of executed kernel names")
    parser.add_argument("--throttle-warnings-json", default="[]", help="JSON array of throttle warning strings")

    args = parser.parse_args()

    # Determine core for cache & governor
    target_core = 0
    pin_core_val = None
    if args.pin_core != "" and args.pin_core is not None:
        try:
            target_core = int(args.pin_core)
            pin_core_val = target_core
        except ValueError:
            pin_core_val = args.pin_core

    if args.taskset_used is not None:
        taskset_used = args.taskset_used
    elif pin_core_val is not None:
        taskset_used = True
    else:
        taskset_used = False

    bin_build_info = parse_build_info(args.bin)
    bin_sha256 = get_binary_sha256(args.bin)

    try:
        sizes = json.loads(args.sizes_json)
    except Exception:
        sizes = []

    try:
        kernels = json.loads(args.kernels_json)
    except Exception:
        kernels = []

    try:
        throttle_warnings = json.loads(args.throttle_warnings_json)
        throttle_warnings = [w for w in throttle_warnings if w]
    except Exception:
        throttle_warnings = []

    manifest = {
        "schema_version": 1,
        "date_utc_start": args.date_start,
        "date_utc_end": args.date_end,
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
        "binary_build_info": bin_build_info,
        "binary_sha256": bin_sha256,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST", None),
        "uname_a": " ".join(os.uname()),
        "cpu_model": get_cpu_model(),
        "isa_flags_summary": get_isa_flags_summary(),
        "caches": get_cache_info(target_core),
        "governor": get_governor(target_core),
        "freq_khz_start": args.freq_start if args.freq_start > 0 else None,
        "freq_khz_end": args.freq_end if args.freq_end > 0 else None,
        "pin_core": pin_core_val,
        "taskset_used": taskset_used,
        "platform": args.platform,
        "platform_tag": args.platform_tag if args.platform_tag else None,
        "compiler_cmd": args.compiler_cmd,
        "compiler_version_full": args.compiler_version,
        "cxx_flags_effective": args.cxx_flags,
        "fastmath": args.fastmath,
        "precision": args.precision,
        "crr_only": args.crr_only,
        "alpha": args.alpha,
        "beta": args.beta,
        "fastpath": args.fastpath,
        "c_cold": args.c_cold,
        "num_runs": args.num_runs,
        "repetitions": args.repetitions,
        "sizes": sizes,
        "kernels": kernels,
        "cooldown_sec": args.cooldown_sec,
        "throttle_warnings": throttle_warnings,
    }

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
