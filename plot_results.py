#!/usr/bin/env python3
"""
GEMMBench — Microarchitectural Performance & Efficiency Analysis

Analyzes GEMM benchmark CSV results across microarchitectures and compilers.
- Automatically selects the best kernel implementation per microarchitecture based on peak GFLOP/s.
- Calculates microarchitectural FLOP/cycle and % of theoretical peak.
- Compares measured best kernels against theoretical expectations defined in uarch_config.json.
- Generates publication-quality SVG figures and summary tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================================================================
# Styling and Palette Configuration
# ==============================================================================

plt.rcParams.update(
    {
        "figure.figsize": (11, 6.5),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)

UARCH_COLORS = {
    "zen4": "#D32F2F",       # Deep Red (AMD)
    "zen5": "#B71C1C",       # Darker Red
    "m1": "#1976D2",         # Blue (Apple)
    "rpi5": "#E91E63",       # Raspberry Pink
    "meteorlake": "#0288D1", # Cyan/Blue (Intel)
    "skylake": "#0097A7",    # Teal (Intel)
    "x100": "#388E3C",       # Green (RISC-V)
    "x60": "#689F38",        # Light Green
    "a100": "#7B1FA2",       # Purple (High throughput)
}

UARCH_MARKERS = {
    "zen4": "s",
    "zen5": "D",
    "m1": "o",
    "rpi5": "^",
    "meteorlake": "v",
    "skylake": "<",
    "x100": "p",
    "x60": "h",
    "a100": "P",
}


# ==============================================================================
# Configuration Data Structures
# ==============================================================================

@dataclass
class UArchSpec:
    simd: str
    uarch_id: str
    name: str
    cpu_model: str
    frequency_ghz: float
    peak_flop_per_cycle: float
    expected_best_kernel: str
    csv_pattern: str

    @property
    def peak_gflops(self) -> float:
        return self.frequency_ghz * self.peak_flop_per_cycle


def load_uarch_config(config_path: Path) -> Dict[str, UArchSpec]:
    """Loads and flattens the hierarchical JSON configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    specs: Dict[str, UArchSpec] = {}
    for simd_family, uarchs in raw_data.items():
        for uarch_id, spec in uarchs.items():
            specs[uarch_id] = UArchSpec(
                simd=simd_family,
                uarch_id=uarch_id,
                name=spec.get("name", uarch_id),
                cpu_model=spec.get("cpu_model", ""),
                frequency_ghz=float(spec.get("frequency_ghz", 1.0)),
                peak_flop_per_cycle=float(spec.get("peak_flop_per_cycle", 1.0)),
                expected_best_kernel=spec.get("expected_best_kernel", ""),
                csv_pattern=spec.get("csv_pattern", f"*gemm_results_{uarch_id}_*.csv"),
            )
    return specs


# ==============================================================================
# Data Loading and Normalization
# ==============================================================================

def find_csv_files(input_dir: Path, spec: UArchSpec) -> List[Tuple[Path, str]]:
    """Finds CSV files matching the uarch specification and returns (path, compiler)."""
    matched: List[Tuple[Path, str]] = []
    for compiler in ["gcc", "clang"]:
        pattern = spec.csv_pattern.replace("{compiler}", compiler)
        for p in input_dir.glob("**/*.csv"):
            if fnmatch.fnmatch(p.name, pattern):
                matched.append((p, compiler))
            elif fnmatch.fnmatch(p.name, f"*gemm_results_{spec.uarch_id}_{compiler}.csv") or \
                 fnmatch.fnmatch(p.name, f"*gemm_results_{spec.uarch_id}_{spec.simd}_{compiler}.csv"):
                if (p, compiler) not in matched:
                    matched.append((p, compiler))
            # Legacy fallbacks for datasets where architecture wasn't prefixed
            elif spec.uarch_id == "meteorlake" and fnmatch.fnmatch(p.name, f"*gemm_results_avx2_{compiler}.csv"):
                if (p, compiler) not in matched:
                    matched.append((p, compiler))
            elif spec.uarch_id == "zen4" and fnmatch.fnmatch(p.name, f"*gemm_results_avx512_{compiler}.csv"):
                if (p, compiler) not in matched:
                    matched.append((p, compiler))

    return matched


def load_dataset_for_uarch(input_dir: Path, spec: UArchSpec) -> pd.DataFrame:
    """Loads and enriches benchmark CSV files for a given uarch."""
    files = find_csv_files(input_dir, spec)
    dfs: List[pd.DataFrame] = []

    for path, compiler in files:
        try:
            df = pd.read_csv(path)
            if df.empty or "Kernel" not in df.columns or "GFLOPS" not in df.columns:
                continue

            # Compute derived metrics cleanly without chained assignment
            flop_per_cycle = df["GFLOPS"] / spec.frequency_ghz
            efficiency_pct = (flop_per_cycle / spec.peak_flop_per_cycle) * 100.0

            df = df.assign(
                Compiler=compiler,
                UArchId=spec.uarch_id,
                SIMD=spec.simd,
                UArchName=spec.name,
                CpuModel=spec.cpu_model,
                FrequencyGHz=spec.frequency_ghz,
                PeakFlopPerCycle=spec.peak_flop_per_cycle,
                PeakGFLOPS=spec.peak_gflops,
                ExpectedKernel=spec.expected_best_kernel,
                FLOP_per_cycle=flop_per_cycle,
                Efficiency_pct=efficiency_pct,
            )

            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate measurements if any
    dedup_cols = ["UArchId", "Compiler", "Kernel", "M", "N", "K"]
    combined = combined.sort_values(by=["GFLOPS"], ascending=False).groupby(dedup_cols, as_index=False).first()
    return combined


# ==============================================================================
# Kernel Selection and Analysis
# ==============================================================================

@dataclass
class UArchChampion:
    uarch_id: str
    spec: UArchSpec
    kernel: str
    compiler: str
    peak_gflops: float
    peak_flop_per_cycle: float
    efficiency_pct: float
    matches_expected: bool
    data: pd.DataFrame


def analyze_uarch(df: pd.DataFrame, spec: UArchSpec) -> Optional[UArchChampion]:
    """Identifies the champion kernel for a uarch based on peak GFLOP/s."""
    if df.empty:
        return None

    # Identify the best (Kernel, Compiler) pair across all matrix sizes
    peak_by_kernel_compiler = df.groupby(["Kernel", "Compiler"], as_index=False)["GFLOPS"].max()
    best_row = peak_by_kernel_compiler.sort_values("GFLOPS", ascending=False).iloc[0]

    best_kernel = best_row["Kernel"]
    best_compiler = best_row["Compiler"]

    champion_data = df[(df["Kernel"] == best_kernel) & (df["Compiler"] == best_compiler)].sort_values("M")
    peak_gflops = champion_data["GFLOPS"].max()
    peak_flop_per_cycle = champion_data["FLOP_per_cycle"].max()
    efficiency_pct = (peak_flop_per_cycle / spec.peak_flop_per_cycle) * 100.0

    matches_expected = (best_kernel == spec.expected_best_kernel)

    return UArchChampion(
        uarch_id=spec.uarch_id,
        spec=spec,
        kernel=best_kernel,
        compiler=best_compiler,
        peak_gflops=peak_gflops,
        peak_flop_per_cycle=peak_flop_per_cycle,
        efficiency_pct=efficiency_pct,
        matches_expected=matches_expected,
        data=champion_data,
    )


# ==============================================================================
# Visualization Engine
# ==============================================================================

def plot_cross_uarch_efficiency(champions: List[UArchChampion], output_path: Path) -> None:
    """Generates the comparison plot of all uarch champions in % of theoretical peak."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Reference 100% line
    ax.axhline(100.0, color="#212121", linestyle="--", linewidth=1.5, label="Theoretical Peak (100%)", zorder=1)
    ax.set_ylim(0, 105)

    sizes = set()
    for champ in champions:
        sizes.update(champ.data["M"].tolist())
    sorted_sizes = sorted(sizes)

    for champ in champions:
        c = UARCH_COLORS.get(champ.uarch_id, "#455A64")
        m = UARCH_MARKERS.get(champ.uarch_id, "o")
        label = f"{champ.spec.name} ({champ.spec.simd}) : {champ.kernel} [{champ.compiler}] (Peak: {champ.efficiency_pct:.1f}%)"

        ax.plot(
            champ.data["M"],
            champ.data["Efficiency_pct"],
            label=label,
            color=c,
            marker=m,
            markersize=7,
            linewidth=2.2,
            zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted_sizes)
    ax.set_xticklabels([str(s) for s in sorted_sizes])
    ax.set_xlabel("Matrix dimension M=N=K")
    ax.set_ylabel("Efficiency (% of Theoretical Peak FLOP/cycle)")
    ax.set_title("Cross-Microarchitecture GEMM Efficiency (% of Theoretical Peak)")

    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_cross_uarch_flop_per_cycle(champions: List[UArchChampion], output_path: Path) -> None:
    """Generates comparison in raw DP FLOP/cycle."""
    fig, ax = plt.subplots(figsize=(12, 7))

    sizes = set()
    for champ in champions:
        sizes.update(champ.data["M"].tolist())
    sorted_sizes = sorted(sizes)

    for champ in champions:
        c = UARCH_COLORS.get(champ.uarch_id, "#455A64")
        m = UARCH_MARKERS.get(champ.uarch_id, "o")
        label = f"{champ.spec.name} : {champ.kernel} [{champ.compiler}] (Peak: {champ.peak_flop_per_cycle:.1f} FLOP/cyc)"

        ax.plot(
            champ.data["M"],
            champ.data["FLOP_per_cycle"],
            label=label,
            color=c,
            marker=m,
            markersize=7,
            linewidth=2.2,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted_sizes)
    ax.set_xticklabels([str(s) for s in sorted_sizes])
    ax.set_xlabel("Matrix dimension M=N=K")
    ax.set_ylabel("Performance (DP FLOP / cycle)")
    ax.set_title("Cross-Microarchitecture GEMM Compute Density (DP FLOP/cycle)")

    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_uarch_overview(df: pd.DataFrame, champ: UArchChampion, output_path: Path) -> None:
    """Generates an individual overview figure for a specific uarch."""
    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    # Dual axis: GFLOP/s on left, FLOP/cycle on right
    ax2 = ax1.twinx()

    freq = champ.spec.frequency_ghz
    peak_gflops = champ.spec.peak_gflops
    peak_flop_per_cycle = champ.spec.peak_flop_per_cycle

    # Peak lines
    ax1.axhline(peak_gflops, color="black", linestyle=":", linewidth=1.5,
                label=f"Peak: {peak_gflops:.1f} GFLOP/s ({peak_flop_per_cycle:.0f} FLOP/cyc)")

    # Find Top 3 kernels by peak GFLOPS
    kernel_peaks = df.groupby("Kernel")["GFLOPS"].max().sort_values(ascending=False)
    top_kernels = list(kernel_peaks.index[:4])
    if "ijk" in df["Kernel"].values and "ijk" not in top_kernels:
        top_kernels.append("ijk")

    sorted_sizes = sorted(df["M"].unique())

    palette = ["#D32F2F", "#1976D2", "#388E3C", "#7B1FA2", "#F57C00", "#616161"]
    markers = ["o", "s", "^", "D", "v", "x"]

    for idx, kern in enumerate(top_kernels):
        k_df = df[df["Kernel"] == kern]
        # Pick winning compiler for this kernel
        best_comp = k_df.groupby("Compiler")["GFLOPS"].max().idxmax()
        line_data = k_df[k_df["Compiler"] == best_comp].sort_values("M")

        is_champ = (kern == champ.kernel)
        lw = 2.6 if is_champ else 1.6
        label = f"{kern} [{best_comp}]"
        if is_champ:
            label += " ★ (Champion)"
        elif kern == "ijk":
            label += " (Scalar Baseline)"

        c = palette[idx % len(palette)]
        m = markers[idx % len(markers)]

        ax1.plot(line_data["M"], line_data["GFLOPS"], label=label,
                 color=c, marker=m, linewidth=lw, markersize=6)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sorted_sizes)
    ax1.set_xticklabels([str(s) for s in sorted_sizes])
    ax1.set_xlabel("Matrix dimension M=N=K")
    ax1.set_ylabel("Throughput (GFLOP/s)")

    # Synchronize right axis (FLOP/cycle = GFLOPS / freq)
    y1_min, y1_max = ax1.get_ylim()
    ax1.set_ylim(0, max(y1_max, peak_gflops * 1.05))
    ax2.set_ylim(0, ax1.get_ylim()[1] / freq)
    ax2.set_ylabel("Efficiency (DP FLOP / cycle)")

    title = f"{champ.spec.name} ({champ.spec.cpu_model}) — {champ.spec.simd} @ {freq:.1f} GHz"
    subtitle = f"Champion: {champ.kernel} [{champ.compiler}] | Peak: {champ.peak_gflops:.1f} GFLOP/s ({champ.efficiency_pct:.1f}% of Peak)"
    ax1.set_title(f"{title}\n{subtitle}", fontsize=12)

    ax1.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Generated: {output_path}")


# ==============================================================================
# Console Reporting
# ==============================================================================

def print_summary_table(champions: List[UArchChampion]) -> None:
    """Prints a formatted comparative table in the console."""
    print("\n" + "=" * 120)
    print("  GEMMBench — Microarchitectural Performance & Peak Efficiency Summary")
    print("=" * 120)
    header = (
        f"{'SIMD':<8} | {'Microarchitecture':<22} | {'Champion Kernel':<34} | "
        f"{'Comp':<5} | {'Peak GFLOP/s':<12} | {'FLOP/cyc':<9} | {'% Peak':<8} | {'Expected Match?':<15}"
    )
    print(header)
    print("-" * 120)

    for champ in champions:
        match_str = "✓ MATCH" if champ.matches_expected else f"✗ MISMATCH ({champ.spec.expected_best_kernel})"
        row = (
            f"{champ.spec.simd:<8} | "
            f"{champ.spec.name:<22} | "
            f"{champ.kernel:<34} | "
            f"{champ.compiler:<5} | "
            f"{champ.peak_gflops:<12.1f} | "
            f"{champ.peak_flop_per_cycle:<9.2f} | "
            f"{champ.efficiency_pct:<7.1f}% | "
            f"{match_str}"
        )
        print(row)
    print("=" * 120 + "\n")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and plot GEMM benchmarks with automatic best-kernel selection and FLOP/cycle efficiency."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("uarch_config.json"),
        help="Path to hierarchical uarch_config.json (default: uarch_config.json)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("gemm-bench-results"),
        help="Directory containing benchmark CSV files (default: gemm-bench-results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots"),
        help="Directory to save generated SVG plots (default: plots)",
    )
    parser.add_argument(
        "--uarch",
        nargs="+",
        help="Filter specific microarchitecture(s) by ID (e.g. --uarch zen4 m1 rvv_x100)",
    )
    parser.add_argument(
        "--simd",
        nargs="+",
        help="Filter specific SIMD family (e.g. --simd avx512 neon rvv)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        help="Override frequencies in GHz (e.g. --freq zen4=5.0,m1=3.2)",
    )
    args = parser.parse_args()

    # 1. Load microarchitectural specifications
    specs = load_uarch_config(args.config)

    # 2. Apply frequency overrides if specified
    if args.freq:
        for pair in args.freq.split(","):
            if "=" in pair:
                u_id, f_str = pair.strip().split("=", 1)
                if u_id in specs:
                    specs[u_id].frequency_ghz = float(f_str)

    # 3. Filter targets if requested
    target_specs = list(specs.values())
    if args.simd:
        target_specs = [s for s in target_specs if s.simd in args.simd]
    if args.uarch:
        target_specs = [s for s in target_specs if s.uarch_id in args.uarch]

    if not target_specs:
        print("No matching microarchitectures found with specified filters.")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # 4. Load data and analyze champions
    champions: List[UArchChampion] = []
    loaded_data: Dict[str, pd.DataFrame] = {}

    for spec in target_specs:
        df = load_dataset_for_uarch(args.input_dir, spec)
        if df.empty:
            print(f"Notice: No data found for {spec.name} ({spec.uarch_id}) in {args.input_dir}")
            continue

        champ = analyze_uarch(df, spec)
        if champ:
            champions.append(champ)
            loaded_data[spec.uarch_id] = df

    if not champions:
        print("No valid benchmark data could be loaded. Please check --input-dir.")
        return

    # 5. Generate Individual Overview Plots
    for champ in champions:
        u_df = loaded_data[champ.uarch_id]
        out_file = args.output / f"{champ.spec.simd}_{champ.uarch_id}_overview.svg"
        plot_uarch_overview(u_df, champ, out_file)

    # 6. Generate Cross-UArch Comparison Plots
    if len(champions) > 1:
        plot_cross_uarch_efficiency(champions, args.output / "cross_uarch_efficiency_comparison.svg")
        plot_cross_uarch_flop_per_cycle(champions, args.output / "cross_uarch_flop_cycle_comparison.svg")

    # 7. Print Console Summary
    print_summary_table(champions)


if __name__ == "__main__":
    main()