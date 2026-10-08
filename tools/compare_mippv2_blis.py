#!/usr/bin/env python3
"""
tools/compare_mippv2_blis.py

Calculates and displays the percentage of BLIS reference micro-kernel peak performance
achieved by MIPPv2 champion micro-kernels across target microarchitectures.

Features:
- Reads CSVs from gemm-bench-results/blis_comparison/ (or specified --input-dir)
- Computes Peak & Median GFLOPS, Peak & Median FLOP/cycle, and MIPPv2/BLIS ratio
- Generates per-architecture markdown comparison tables and per-K scaling breakdowns
- Optionally outputs markdown reports, CSV summaries, and comparative SVG plots
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_filename(filepath: Path) -> Tuple[str, str, str]:
    """
    Extracts (platform, format, compiler) from filename.
    e.g. 'gemm_results_zen4_avx512_crr_clang.csv' -> ('zen4_avx512', 'crr', 'clang')
    """
    stem = filepath.stem
    m = re.match(r"gemm_results_(.+)_(crr|rrr)_(gcc|clang)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    parts = stem.split("_")
    compiler = parts[-1] if parts[-1] in ("gcc", "clang") else "unknown"
    layout = parts[-2] if parts[-2] in ("crr", "rrr") else "unknown"
    platform = "_".join(parts[2:-2]) if len(parts) > 4 else "unknown"
    return platform, layout, compiler


def load_dataset(input_dir: Path) -> pd.DataFrame:
    """Loads all CSV files in input_dir and annotates platform/compiler metadata."""
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"Warning: No CSV files found in {input_dir}", file=sys.stderr)
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            platform, layout, compiler = parse_filename(f)
            df["Platform"] = platform
            df["Layout"] = layout
            df["Compiler"] = compiler
            df["SourceFile"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def analyze_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Processes the benchmark dataframe, identifying for each (Platform, Compiler, Layout):
    - MIPPv2 Champion performance
    - BLIS Native performance
    - Ratio MIPPv2 / BLIS (%)
    """
    summary_rows = []
    detailed_comparisons = defaultdict(dict)

    groups = df.groupby(["Platform", "Compiler", "Layout"])

    for (platform, compiler, layout), group in groups:
        blis_data = group[group["Kernel"] == "blis_native"]
        mipp_data = group[group["Kernel"] != "blis_native"]

        if blis_data.empty:
            continue
        if mipp_data.empty:
            continue

        # Find champion MIPPv2 kernel by peak GFLOPS
        mipp_best_idx = mipp_data["GFLOPS_peak"].idxmax()
        mipp_best_kernel = mipp_data.loc[mipp_best_idx, "Kernel"]
        mipp_champ_data = mipp_data[mipp_data["Kernel"] == mipp_best_kernel]

        # Extract summary metrics
        mipp_peak_gflops = mipp_champ_data["GFLOPS_peak"].max()
        mipp_median_gflops = mipp_champ_data["GFLOPS_median"].max()
        mipp_peak_fc = mipp_champ_data["FLOP_per_cycle_peak"].max() if "FLOP_per_cycle_peak" in mipp_champ_data else 0.0
        mipp_med_fc = mipp_champ_data["FLOP_per_cycle_median"].max() if "FLOP_per_cycle_median" in mipp_champ_data else 0.0

        blis_peak_gflops = blis_data["GFLOPS_peak"].max()
        blis_median_gflops = blis_data["GFLOPS_median"].max()
        blis_peak_fc = blis_data["FLOP_per_cycle_peak"].max() if "FLOP_per_cycle_peak" in blis_data else 0.0
        blis_med_fc = blis_data["FLOP_per_cycle_median"].max() if "FLOP_per_cycle_median" in blis_data else 0.0

        ratio_peak_gflops = (mipp_peak_gflops / blis_peak_gflops * 100.0) if blis_peak_gflops > 0 else 0.0
        ratio_peak_fc = (mipp_peak_fc / blis_peak_fc * 100.0) if blis_peak_fc > 0 else 0.0
        ratio_med_fc = (mipp_med_fc / blis_med_fc * 100.0) if blis_med_fc > 0 else 0.0

        # Best dimensions
        best_row = mipp_champ_data.loc[mipp_best_idx]
        best_k = int(best_row["K"])

        summary_rows.append({
            "Platform": platform,
            "Compiler": compiler,
            "Layout": layout,
            "MIPPv2_Champion": mipp_best_kernel,
            "MIPPv2_Peak_GFLOPS": mipp_peak_gflops,
            "BLIS_Peak_GFLOPS": blis_peak_gflops,
            "Ratio_Peak_GFLOPS_pct": ratio_peak_gflops,
            "MIPPv2_Peak_FC": mipp_peak_fc,
            "BLIS_Peak_FC": blis_peak_fc,
            "Ratio_Peak_FC_pct": ratio_peak_fc,
            "MIPPv2_Med_FC": mipp_med_fc,
            "BLIS_Med_FC": blis_med_fc,
            "Ratio_Med_FC_pct": ratio_med_fc,
            "Best_K": best_k,
        })

        # Record per-K breakdown
        common_ks = sorted(set(mipp_champ_data["K"]).intersection(set(blis_data["K"])))
        for k in common_ks:
            m_sub = mipp_champ_data[mipp_champ_data["K"] == k]
            b_sub = blis_data[blis_data["K"] == k]
            if not m_sub.empty and not b_sub.empty:
                m_gflops = m_sub["GFLOPS_median"].values[0]
                b_gflops = b_sub["GFLOPS_median"].values[0]
                m_fc = m_sub["FLOP_per_cycle_median"].values[0] if "FLOP_per_cycle_median" in m_sub else 0.0
                b_fc = b_sub["FLOP_per_cycle_median"].values[0] if "FLOP_per_cycle_median" in b_sub else 0.0
                r_gflops = (m_gflops / b_gflops * 100.0) if b_gflops > 0 else 0.0
                detailed_comparisons[(platform, compiler, layout)][k] = {
                    "MIPPv2_GFLOPS": m_gflops,
                    "BLIS_GFLOPS": b_gflops,
                    "Ratio_pct": r_gflops,
                    "MIPPv2_FC": m_fc,
                    "BLIS_FC": b_fc,
                }

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detailed_comparisons


def format_markdown_report(summary_df: pd.DataFrame, details: Dict) -> str:
    """Generates a GitHub-flavored Markdown comparative analysis document."""
    lines = []
    lines.append("# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels\n")
    lines.append("Empirical comparison of sustained compute density and throughput between MIPPv2 champions")
    lines.append("and hand-tuned BLIS assembly micro-kernels across target microarchitectures.\n")

    if summary_df.empty:
        lines.append("> [!WARNING]\n> No paired MIPPv2 vs BLIS benchmark datasets were found.\n")
        return "\n".join(lines)

    lines.append("## 1. Executive Summary Table (% of BLIS Peak Reached)\n")
    lines.append("| Microarchitecture | Compiler | Layout | MIPPv2 Champion | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **% of BLIS Peak** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for _, row in summary_df.iterrows():
        ratio = row["Ratio_Peak_GFLOPS_pct"]
        if ratio >= 99.0:
            status = "🟢 Parity (≥99%)"
        elif ratio >= 95.0:
            status = "🟢 Near-Parity (≥95%)"
        elif ratio >= 90.0:
            status = "🟡 Competitive (≥90%)"
        else:
            status = "🔴 Delta (<90%)"

        m_name = row["MIPPv2_Champion"].replace("mippv2_", "")
        lines.append(
            f"| `{row['Platform']}` | `{row['Compiler']}` | `{row['Layout'].upper()}` | `{m_name}` | "
            f"**{row['MIPPv2_Peak_GFLOPS']:.2f}** | **{row['BLIS_Peak_GFLOPS']:.2f}** | "
            f"**{ratio:.1f}%** | {row['MIPPv2_Peak_FC']:.2f} | {row['BLIS_Peak_FC']:.2f} | {status} |"
        )

    lines.append("\n---\n")
    lines.append("## 2. Per-Size Scaling & Efficiency Analysis ($K \\in [32, 1024]$)\n")

    for (platform, compiler, layout), k_data in details.items():
        lines.append(f"### Platform: `{platform}` ({compiler}, {layout.upper()})\n")
        lines.append("| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |")
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for k in sorted(k_data.keys()):
            entry = k_data[k]
            r = entry["Ratio_pct"]
            adv = "MIPPv2 (+{:.1f}%)".format(r - 100) if r > 100.5 else ("BLIS (+{:.1f}%)".format(100 - r) if r < 99.5 else "Equivalent")
            lines.append(
                f"| K={k} | {entry['MIPPv2_GFLOPS']:.2f} | {entry['BLIS_GFLOPS']:.2f} | "
                f"**{r:.1f}%** | {entry['MIPPv2_FC']:.2f} | {entry['BLIS_FC']:.2f} | {adv} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare MIPPv2 champion kernels vs BLIS reference micro-kernels")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="gemm-bench-results/blis_comparison",
        help="Directory containing benchmark CSV files (default: gemm-bench-results/blis_comparison)",
    )
    parser.add_argument(
        "--fallback-crr",
        action="store_true",
        help="Fallback to gemm-bench-results/crr if blis_comparison is empty",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to save generated Markdown report (e.g. blis_vs_mippv2_report.md)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save summary CSV metrics",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists() or not list(input_path.glob("*.csv")):
        if args.fallback_crr and Path("gemm-bench-results/crr").exists():
            input_path = Path("gemm-bench-results/crr")
            print(f"Notice: Falling back to {input_path}", file=sys.stderr)
        else:
            print(f"Error: Directory '{input_path}' does not exist or contains no CSV files.", file=sys.stderr)
            print("Run the cluster benchmarks first: ./tools/run_cluster_benchmarks.sh all --crr-only --with-blis", file=sys.stderr)
            sys.exit(1)

    print(f"Loading benchmark datasets from {input_path}...")
    df = load_dataset(input_path)
    if df.empty:
        print("No valid data loaded.", file=sys.stderr)
        sys.exit(1)

    summary_df, details = analyze_dataset(df)
    md_report = format_markdown_report(summary_df, details)

    # Print to console
    print("\n" + md_report)

    if args.output_md:
        out_p = Path(args.output_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_report, encoding="utf-8")
        print(f"\n✓ Markdown report written to {out_p}")

    if args.output_csv and not summary_df.empty:
        out_c = Path(args.output_csv)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_c, index=False)
        print(f"✓ Summary CSV metrics written to {out_c}")


if __name__ == "__main__":
    main()
