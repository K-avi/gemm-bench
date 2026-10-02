#!/usr/bin/env python3
"""
Updates measured_peak_flop_per_cycle in uarch_config.json
based on empirical double-precision (FP64) measurements from cpufp.
"""

import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "uarch_config.json"
CSV_PATH = ROOT_DIR / "gemm-bench-results" / "cpufp" / "cpufp_peak_dp_summary.csv"

def update_uarch_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CPUFP summary CSV not found: {CSV_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Read cpufp measurements
    measurements = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["Target"].strip().lower()
            peak_gflops = float(row["Peak_DP_GFLOPS"])
            freq_ghz = float(row["FreqGHz"])
            flop_per_cycle = round(peak_gflops / freq_ghz, 2)
            measurements[target] = {
                "peak_gflops": peak_gflops,
                "freq_ghz": freq_ghz,
                "measured_flop_per_cycle": flop_per_cycle,
                "computation": row.get("Computation", ""),
                "isa": row.get("ISA", "")
            }

    print("=" * 80)
    print("  Updating uarch_config.json with CPUFP Empirical Measurements")
    print("=" * 80)
    print(f"{'Target':<14} | {'Old FLOP/cyc':<14} | {'New Measured FLOP/cyc':<22} | {'Peak DP GFLOP/s':<16}")
    print("-" * 80)

    updated_count = 0
    for family, uarchs in config.items():
        for uarch_id, spec in uarchs.items():
            key = uarch_id.lower()
            if key in measurements:
                old_val = spec.get("measured_peak_flop_per_cycle")
                new_val = measurements[key]["measured_flop_per_cycle"]
                peak_gflops = measurements[key]["peak_gflops"]
                spec["measured_peak_flop_per_cycle"] = new_val
                updated_count += 1
                print(f"{uarch_id:<14} | {str(old_val):<14} | {new_val:<22.2f} | {peak_gflops:<16.2f}")

    print("=" * 80)
    print(f"Updated {updated_count} microarchitecture(s) in {CONFIG_PATH}\n")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    update_uarch_config()
