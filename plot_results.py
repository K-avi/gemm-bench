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
# Styling and Palette Configuration (Publication-Grade Light Mode)
# ==============================================================================

LIGHT_BG = "#ffffff"       # Clean white canvas
CARD_BG = "#ffffff"        # White plot axes
BORDER_COLOR = "#d0d7de"   # Subtle GitHub-style border
TEXT_PRIMARY = "#1f2328"   # Dark slate for primary text
TEXT_SECONDARY = "#656d76" # Muted slate for subtitles and labels
TEXT_MUTED = "#8c959f"     # Dim label
GRID_COLOR = "#eaeef2"     # Subtle grid lines

SIMD_FAMILY_COLORS = {
    "avx512": "#dc2626",   # Crimson Red
    "avx2": "#0284c7",     # Sky Blue
    "neon": "#7c3aed",     # Purple / Violet
    "rvv": "#059669",      # Emerald Green
}

UARCH_COLORS = {
    "zen4": "#dc2626",       # Crimson Red (AMD AVX-512)
    "zen5": "#b91c1c",       # Darker Red (AMD Zen 5)
    "m1": "#7c3aed",         # Purple (Apple M1 NEON)
    "rpi5": "#db2777",       # Pink (RPi5 Cortex-A76 NEON)
    "meteorlake": "#0284c7", # Sky Blue (Intel Meteor Lake AVX2)
    "skylake": "#0369a1",    # Deep Sky (Intel Skylake AVX2)
    "x100": "#059669",       # Emerald Green (SpacemiT X100 RVV)
    "x60": "#10b981",        # Sea Green (SpacemiT X60 RVV)
    "a100": "#047857",       # Dark Emerald (SpacemiT A100 RVV)
}

plt.rcParams.update(
    {
        "figure.facecolor": LIGHT_BG,
        "figure.edgecolor": LIGHT_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": BORDER_COLOR,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.9,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": BORDER_COLOR,
        "legend.fontsize": 9.5,
        "text.color": TEXT_PRIMARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial", "sans-serif"],
        "svg.fonttype": "none",
    }
)


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
    measured_peak_flop_per_cycle: Optional[float] = None
    hardware_max_frequency_ghz: Optional[float] = None
    notes: Optional[str] = None

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
                measured_peak_flop_per_cycle=float(spec["measured_peak_flop_per_cycle"]) if "measured_peak_flop_per_cycle" in spec else None,
                hardware_max_frequency_ghz=float(spec["hardware_max_frequency_ghz"]) if "hardware_max_frequency_ghz" in spec else None,
                notes=spec.get("notes"),
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
    measured_efficiency_pct: Optional[float]
    matches_expected: bool
    data: pd.DataFrame
    peak_m: int = 128


def analyze_uarch(df: pd.DataFrame, spec: UArchSpec) -> Optional[UArchChampion]:
    """Identifies the champion kernel for a uarch based on peak GFLOP/s."""
    if df.empty:
        return None

    peak_by_kernel_compiler = df.groupby(["Kernel", "Compiler"], as_index=False)["GFLOPS"].max()
    best_row = peak_by_kernel_compiler.sort_values("GFLOPS", ascending=False).iloc[0]

    best_kernel = best_row["Kernel"]
    best_compiler = best_row["Compiler"]

    champion_data = df[(df["Kernel"] == best_kernel) & (df["Compiler"] == best_compiler)].sort_values("M")
    peak_row = champion_data.sort_values("GFLOPS", ascending=False).iloc[0]
    peak_gflops = float(peak_row["GFLOPS"])
    peak_flop_per_cycle = float(peak_row["FLOP_per_cycle"])
    peak_m = int(peak_row["M"])

    efficiency_pct = (peak_flop_per_cycle / spec.peak_flop_per_cycle) * 100.0
    measured_eff = (peak_flop_per_cycle / spec.measured_peak_flop_per_cycle * 100.0) if spec.measured_peak_flop_per_cycle else None
    matches_expected = (best_kernel == spec.expected_best_kernel)

    return UArchChampion(
        uarch_id=spec.uarch_id,
        spec=spec,
        kernel=best_kernel,
        compiler=best_compiler,
        peak_gflops=peak_gflops,
        peak_flop_per_cycle=peak_flop_per_cycle,
        efficiency_pct=efficiency_pct,
        measured_efficiency_pct=measured_eff,
        matches_expected=matches_expected,
        data=champion_data,
        peak_m=peak_m,
    )


# ==============================================================================
# Visualization Engine (Publication-Grade Light Mode)
# ==============================================================================

def format_uarch_axis_label(c: UArchChampion) -> str:
    """Formats a concise two-line label for the Y-axis."""
    name = c.spec.name
    model = c.spec.cpu_model
    if "Strix Point" in name and "HX 370" in model:
        line1 = "AMD Zen 5 (Ryzen AI 9 HX 370)"
    elif model and model not in name:
        line1 = f"{name} ({model})"
    else:
        line1 = name
    line2 = f"{c.spec.simd.upper()} @ {c.spec.frequency_ghz:.1f} GHz"
    return f"{line1}\n{line2}"


def group_champions_by_family(champions: List[UArchChampion], sort_metric: str = "efficiency") -> List[UArchChampion]:
    """Groups champions by family: AVX-512, AVX2, NEON, RVV.

    Returns the list sorted so that AVX-512 appears at the top of a barh chart.
    """
    family_order = ["avx512", "avx2", "neon", "rvv"]
    grouped: List[UArchChampion] = []
    for fam in family_order:
        fam_champs = [c for c in champions if c.spec.simd == fam]
        if sort_metric == "efficiency":
            fam_champs.sort(key=lambda c: c.efficiency_pct, reverse=True)
        else:
            fam_champs.sort(key=lambda c: c.peak_flop_per_cycle, reverse=True)
        grouped.extend(fam_champs)

    # In barh, index 0 is at the bottom, so reverse to put AVX-512 at the top
    return list(reversed(grouped))


def plot_cross_uarch_efficiency(champions: List[UArchChampion], output_path: Path) -> None:
    """Generates an executive horizontal ranking bar chart grouped by SIMD family."""
    fig, ax = plt.subplots(figsize=(13.5, 7.5))

    sorted_champs = group_champions_by_family(champions, sort_metric="efficiency")
    y_positions = np.arange(len(sorted_champs))
    bar_height = 0.58

    bar_colors = [SIMD_FAMILY_COLORS.get(c.spec.simd, "#0284c7") for c in sorted_champs]

    bars = ax.barh(
        y_positions,
        [c.efficiency_pct for c in sorted_champs],
        height=bar_height,
        color=bar_colors,
        edgecolor=BORDER_COLOR,
        linewidth=0.8,
        zorder=3,
    )

    # 100% Theoretical Peak reference line
    ax.axvline(100.0, color="#1e293b", linestyle="--", linewidth=1.5, zorder=4)
    ax.text(
        100.2,
        len(sorted_champs) - 0.45,
        "100% Theoretical Peak",
        color="#1e293b",
        fontsize=9,
        fontweight="bold",
        va="center",
        zorder=5,
    )

    # Dock text cleanly on the left border inside the bar (white bold text):
    for idx, (champ, bar) in enumerate(zip(sorted_champs, bars)):
        eff = champ.efficiency_pct
        gflops = champ.peak_gflops
        flop_cyc = champ.peak_flop_per_cycle
        m_dim = champ.peak_m

        text_label = f" {eff:.1f}%  ({gflops:.1f} GFLOP/s @ M={m_dim} | {flop_cyc:.2f} FLOP/cyc)"
        ax.text(
            1.2,
            idx,
            text_label,
            va="center",
            ha="left",
            color="#ffffff",
            fontweight="bold",
            fontsize=9.5,
            zorder=5,
        )

    y_labels = [format_uarch_axis_label(c) for c in sorted_champs]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9.5)

    ax.set_xlim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax.set_xlabel("Hardware Efficiency (% of Theoretical Peak FLOP/cycle)", fontsize=11, fontweight="bold", labelpad=10)

    fig.text(
        0.04, 0.965,
        "Cross-Microarchitecture GEMM Peak Efficiency (% of Theoretical Peak)",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT_PRIMARY,
        ha="left",
    )
    fig.text(
        0.04, 0.935,
        "FP64 Canonical DGEMM (α=1, β=0) | Evaluated across M=N=K ∈ {32, 64, 96, 128} (L1/L2 cache resident)",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        ha="left",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SIMD_FAMILY_COLORS["avx512"], edgecolor=BORDER_COLOR, label="AVX-512"),
        Patch(facecolor=SIMD_FAMILY_COLORS["avx2"], edgecolor=BORDER_COLOR, label="AVX2"),
        Patch(facecolor=SIMD_FAMILY_COLORS["neon"], edgecolor=BORDER_COLOR, label="NEON"),
        Patch(facecolor=SIMD_FAMILY_COLORS["rvv"], edgecolor=BORDER_COLOR, label="RVV 1.0"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.97),
        ncol=4,
        framealpha=0.92,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.9)
    ax.yaxis.grid(False)

    plt.subplots_adjust(top=0.91, bottom=0.10, left=0.28, right=0.96)
    if output_path.suffix == ".png":
        fig.savefig(output_path, format="png", dpi=150)
    else:
        fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_cross_uarch_flop_per_cycle(champions: List[UArchChampion], output_path: Path) -> None:
    """Generates comparison in raw DP FLOP/cycle grouped by SIMD family."""
    fig, ax = plt.subplots(figsize=(13.5, 7.5))

    sorted_champs = group_champions_by_family(champions, sort_metric="flop_cycle")
    y_positions = np.arange(len(sorted_champs))
    bar_height = 0.58

    bar_colors = [SIMD_FAMILY_COLORS.get(c.spec.simd, "#0284c7") for c in sorted_champs]

    bars = ax.barh(
        y_positions,
        [c.peak_flop_per_cycle for c in sorted_champs],
        height=bar_height,
        color=bar_colors,
        edgecolor=BORDER_COLOR,
        linewidth=0.8,
        zorder=3,
    )

    max_val = max(c.peak_flop_per_cycle for c in sorted_champs)

    # Dock text cleanly on the left border inside the bar:
    for idx, (champ, bar) in enumerate(zip(sorted_champs, bars)):
        val = champ.peak_flop_per_cycle
        theo = champ.spec.peak_flop_per_cycle
        eff = champ.efficiency_pct
        gflops = champ.peak_gflops
        m_dim = champ.peak_m
        text_label = f" {val:.2f} / {theo:.0f} FLOP/cyc  ({eff:.1f}% peak | {gflops:.1f} GFLOP/s @ M={m_dim})"
        ax.text(
            max_val * 0.015,
            idx,
            text_label,
            va="center",
            ha="left",
            color="#ffffff",
            fontweight="bold",
            fontsize=9.5,
            zorder=5,
        )

    y_labels = [format_uarch_axis_label(c) for c in sorted_champs]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9.5)

    ax.set_xlim(0, max_val * 1.08)
    ax.set_xlabel("Compute Density (DP FLOP / cycle)", fontsize=11, fontweight="bold", labelpad=10)

    fig.text(
        0.04, 0.965,
        "Cross-Microarchitecture GEMM Compute Density (DP FLOP/cycle)",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT_PRIMARY,
        ha="left",
    )
    fig.text(
        0.04, 0.935,
        "Higher is better | Evaluated across M=N=K ∈ {32, 64, 96, 128} | Theoretical ceiling defined by datapath width & execution ports",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        ha="left",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SIMD_FAMILY_COLORS["avx512"], edgecolor=BORDER_COLOR, label="AVX-512"),
        Patch(facecolor=SIMD_FAMILY_COLORS["avx2"], edgecolor=BORDER_COLOR, label="AVX2"),
        Patch(facecolor=SIMD_FAMILY_COLORS["neon"], edgecolor=BORDER_COLOR, label="NEON"),
        Patch(facecolor=SIMD_FAMILY_COLORS["rvv"], edgecolor=BORDER_COLOR, label="RVV 1.0"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.97),
        ncol=4,
        framealpha=0.92,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.9)
    ax.yaxis.grid(False)

    plt.subplots_adjust(top=0.91, bottom=0.10, left=0.28, right=0.96)
    if output_path.suffix == ".png":
        fig.savefig(output_path, format="png", dpi=150)
    else:
        fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_uarch_overview(df: pd.DataFrame, champ: UArchChampion, output_path: Path) -> None:
    """Generates a single-panel publication figure showing scaling vs matrix dimensions (M=N=K)."""
    fig, ax1 = plt.subplots(figsize=(10.5, 6.0))
    ax2 = ax1.twinx()

    freq = champ.spec.frequency_ghz
    peak_gflops = champ.spec.peak_gflops
    peak_flop_per_cycle = champ.spec.peak_flop_per_cycle

    sorted_sizes = sorted(df["M"].unique())

    # Theoretical peak horizontal line
    ax1.axhline(
        peak_gflops,
        color="#1e293b",
        linestyle="--",
        linewidth=1.5,
        label=f"Theoretical Peak: {peak_gflops:.1f} GFLOP/s ({peak_flop_per_cycle:.0f} FLOP/cyc | 100%)",
        zorder=2,
    )

    # Top evaluated kernels + scalar baseline
    kernel_peaks = df.groupby("Kernel")["GFLOPS"].max().sort_values(ascending=False)
    top_kernels = list(kernel_peaks.index[:4])
    if "ijk" in df["Kernel"].values and "ijk" not in top_kernels:
        top_kernels.append("ijk")

    panel_kernels = []
    for k in top_kernels:
        k_df = df[df["Kernel"] == k]
        best_comp = k_df.groupby("Compiler")["GFLOPS"].max().idxmax()
        peak_k_gflops = k_df[k_df["Compiler"] == best_comp]["GFLOPS"].max()
        panel_kernels.append({
            "kernel": k,
            "compiler": best_comp,
            "gflops": peak_k_gflops,
            "is_champ": (k == champ.kernel),
            "is_scalar": (k == "ijk"),
        })

    # Sort so champion is plotted last (on top)
    panel_kernels = sorted(panel_kernels, key=lambda x: x["gflops"])

    line_palette = ["#0284c7", "#7c3aed", "#db2777", "#dc2626"]
    line_markers = ["s", "^", "D", "v"]

    for idx, k_info in enumerate(panel_kernels):
        k_df = df[(df["Kernel"] == k_info["kernel"]) & (df["Compiler"] == k_info["compiler"])].sort_values("M")
        is_champ = k_info["is_champ"]
        is_scalar = k_info["is_scalar"]

        c = "#059669" if is_champ else ("#d97706" if is_scalar else line_palette[idx % len(line_palette)])
        m = "o" if is_champ else ("x" if is_scalar else line_markers[idx % len(line_markers)])
        lw = 2.6 if is_champ else (1.6 if not is_scalar else 1.3)
        ls = "-" if is_champ else ("--" if not is_scalar else ":")
        ms = 7 if is_champ else 6

        lbl = f"{k_info['kernel']} [{k_info['compiler']}]"
        if is_champ:
            lbl = f"★ {lbl} (Champion — Peak: {champ.peak_gflops:.1f} GFLOP/s @ M={champ.peak_m})"
        elif is_scalar:
            lbl = f"ijk (Scalar Baseline) [{k_info['compiler']}]"

        ax1.plot(
            k_df["M"],
            k_df["GFLOPS"],
            label=lbl,
            color=c,
            marker=m,
            markersize=ms,
            linewidth=lw,
            linestyle=ls,
            zorder=4 if is_champ else 3,
        )

        if is_champ:
            ax1.fill_between(k_df["M"], 0, k_df["GFLOPS"], color=c, alpha=0.06, zorder=1)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sorted_sizes)
    ax1.set_xticklabels([str(s) for s in sorted_sizes], fontsize=10)
    ax1.set_xlabel("Matrix dimension M=N=K", fontsize=10.5, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Throughput (GFLOP/s)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax1.set_ylim(0, peak_gflops * 1.08)

    # Right axis: Hardware efficiency (% of theoretical peak)
    ax2.set_ylim(0, 108)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax2.set_ylabel("Hardware Efficiency (% of Theoretical Peak)", fontsize=10.5, fontweight="bold", labelpad=8)

    # Legend cleanly placed in center-right where data lines do not collide
    ax1.legend(
        loc="center right",
        framealpha=0.94,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        fontsize=9,
    )

    # Styling
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["left"].set_color(BORDER_COLOR)
    ax2.spines["right"].set_color(BORDER_COLOR)
    ax1.spines["bottom"].set_color(BORDER_COLOR)
    ax1.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.9)
    ax2.grid(False)

    # Title and subtitle
    title = f"{champ.spec.name} ({champ.spec.cpu_model}) — {champ.spec.simd.upper()} @ {freq:.1f} GHz"
    subtitle = (
        f"Champion: {champ.kernel} [{champ.compiler}] | "
        f"Peak Throughput: {champ.peak_gflops:.1f} GFLOP/s ({champ.efficiency_pct:.1f}% of Peak @ M={champ.peak_m})"
    )
    ax1.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=12)

    plt.subplots_adjust(top=0.89, bottom=0.11, left=0.10, right=0.90)
    if output_path.suffix == ".png":
        fig.savefig(output_path, format="png", dpi=150)
    else:
        fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Generated: {output_path}")


# ==============================================================================
# Console Reporting
# ==============================================================================

def print_summary_table(champions: List[UArchChampion]) -> None:
    """Prints a formatted comparative table in the console."""
    print("\n" + "=" * 135)
    print("  GEMMBench — Microarchitectural Performance & Peak Efficiency Summary")
    print("=" * 135)
    header = (
        f"{'SIMD':<8} | {'Microarchitecture':<22} | {'Champion Kernel':<34} | "
        f"{'Comp':<5} | {'Peak GFLOP/s':<12} | {'FLOP/cyc':<9} | {'% Theo':<8} | {'% Meas':<8} | {'Expected Match?':<15}"
    )
    print(header)
    print("-" * 135)

    for champ in champions:
        match_str = "✓ MATCH" if champ.matches_expected else f"✗ MISMATCH ({champ.spec.expected_best_kernel})"
        meas_str = f"{champ.measured_efficiency_pct:.1f}%" if champ.measured_efficiency_pct is not None else "N/A"
        row = (
            f"{champ.spec.simd:<8} | "
            f"{champ.spec.name:<22} | "
            f"{champ.kernel:<34} | "
            f"{champ.compiler:<5} | "
            f"{champ.peak_gflops:<12.1f} | "
            f"{champ.peak_flop_per_cycle:<9.2f} | "
            f"{champ.efficiency_pct:<7.1f}% | "
            f"{meas_str:<8} | "
            f"{match_str}"
        )
        print(row)
    print("=" * 135 + "\n")


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