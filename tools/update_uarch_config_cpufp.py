#!/usr/bin/env python3
"""
Updates measured_peak_flop_per_cycle / measured_peak_flop_per_cycle_fp32 in uarch_config.json
based on empirical floating-point measurements from cpufp.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "uarch_config.json"

def update_uarch_config(is_fp32: bool = False, custom_csv: Path = None, custom_config: Path = None):
    config_path = custom_config or CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    csv_name = "cpufp_peak_sp_summary.csv" if is_fp32 else "cpufp_peak_dp_summary.csv"
    if custom_csv:
        csv_path = custom_csv
    else:
        csv_path = ROOT_DIR / "gemm-bench-results" / "cpufp" / csv_name
        if not csv_path.is_file():
            fallback_csv = ROOT_DIR / "results" / "cpufp" / csv_name
            if fallback_csv.is_file():
                csv_path = fallback_csv
            else:
                raise FileNotFoundError(f"CPUFP summary CSV not found: {csv_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Read cpufp measurements
    measurements = {}
    peak_col = "Peak_SP_GFLOPS" if is_fp32 else "Peak_DP_GFLOPS"
    target_field = "measured_peak_flop_per_cycle_fp32" if is_fp32 else "measured_peak_flop_per_cycle"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["Target"].strip().lower()
            peak_gflops = float(row[peak_col])
            freq_ghz = float(row["FreqGHz"])
            flop_per_cycle = round(peak_gflops / freq_ghz, 2)
            measurements[target] = {
                "peak_gflops": peak_gflops,
                "freq_ghz": freq_ghz,
                "measured_flop_per_cycle": flop_per_cycle,
                "computation": row.get("Computation", ""),
                "isa": row.get("ISA", "")
            }

    prec_label = "Single-Precision (FP32)" if is_fp32 else "Double-Precision (FP64)"
    print("=" * 80)
    print(f"  Updating uarch_config.json with CPUFP Empirical Measurements ({prec_label})")
    print("=" * 80)
    print(f"{'Target':<14} | {'Old FLOP/cyc':<14} | {'New Measured FLOP/cyc':<22} | {'Peak GFLOP/s':<16}")
    print("-" * 80)

    updated_count = 0
    for family, uarchs in config.items():
        for uarch_id, spec in uarchs.items():
            key = uarch_id.lower()
            if key in measurements:
                old_val = spec.get(target_field)
                new_val = measurements[key]["measured_flop_per_cycle"]
                peak_gflops = measurements[key]["peak_gflops"]
                spec[target_field] = new_val
                updated_count += 1
                old_str = str(old_val) if old_val is not None else "None"
                print(f"{uarch_id:<14} | {old_str:<14} | {new_val:<22.2f} | {peak_gflops:<16.2f}")

    print("=" * 80)
    print(f"Updated {updated_count} microarchitecture(s) [{target_field}] in {config_path}\n")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Update uarch_config.json with CPUFP empirical measurements")
    parser.add_argument("--fp32", action="store_true", help="Update FP32 measured peak FLOP/cycle from cpufp_peak_sp_summary.csv")
    parser.add_argument("--fp64", action="store_true", help="Update FP64 measured peak FLOP/cycle from cpufp_peak_dp_summary.csv (default)")
    parser.add_argument("--csv", type=Path, default=None, help="Custom path to CPUFP summary CSV")
    parser.add_argument("--config", type=Path, default=None, help="Custom path to uarch_config.json")
    args = parser.parse_args()

    is_fp32 = args.fp32
    update_uarch_config(is_fp32=is_fp32, custom_csv=args.csv, custom_config=args.config)

if __name__ == "__main__":
    main()
