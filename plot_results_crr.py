#!/usr/bin/env python3
"""
GEMMBench — Microarchitectural Performance & Efficiency Analysis for CRR Micro-Kernels

Analyzes GotoBLAS / BLIS-like CRR micro-kernel benchmark CSV results across microarchitectures and compilers.
- A is PackedColMajor (contiguous along MR), B is PackedRowMajor, C is PackedRowMajor.
- Evaluates native register tiles (MR x NR) across panel depth K ∈ {32, 64, 128, 256, 512, 1024}.
- Automatically selects the champion CRR kernel implementation per microarchitecture based on peak GFLOP/s.
- Calculates microarchitectural FLOP/cycle and % of theoretical peak.
- Optional side-by-side comparison against RRR baselines (--compare-rrr).
- Generates publication-quality SVG and PNG figures and summary tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
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
class UArchSpecCRR:
    simd: str
    uarch_id: str
    name: str
    cpu_model: str
    frequency_ghz: float
    peak_flop_per_cycle: float
    expected_best_kernel: str
    csv_pattern: str
    expected_best_kernel_rrr: str = ""
    csv_pattern_rrr: str = ""
    measured_peak_flop_per_cycle: Optional[float] = None
    hardware_max_frequency_ghz: Optional[float] = None
    notes: Optional[str] = None

    @property
    def peak_gflops(self) -> float:
        return self.frequency_ghz * self.peak_flop_per_cycle


def load_uarch_config_crr(config_path: Path) -> Dict[str, UArchSpecCRR]:
    """Loads hierarchical JSON configuration and specializes it for CRR kernels."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    specs: Dict[str, UArchSpecCRR] = {}
    for simd_family, uarchs in raw_data.items():
        for uarch_id, spec in uarchs.items():
            expected_crr = spec.get("expected_best_kernel_crr")
            base_exp = spec.get("expected_best_kernel", "")
            if not expected_crr:
                expected_crr = f"{base_exp}_crr" if base_exp and not base_exp.endswith("_crr") else base_exp

            csv_pattern = spec.get("crr_csv_pattern")
            if not csv_pattern:
                csv_pattern = f"*gemm_results_*{uarch_id}*crr*_{{compiler}}.csv"

            csv_pattern_rrr = spec.get("csv_pattern", f"*gemm_results_*{uarch_id}*_{{compiler}}.csv")

            specs[uarch_id] = UArchSpecCRR(
                simd=simd_family,
                uarch_id=uarch_id,
                name=spec.get("name", uarch_id),
                cpu_model=spec.get("cpu_model", ""),
                frequency_ghz=float(spec.get("frequency_ghz", 1.0)),
                peak_flop_per_cycle=float(spec.get("peak_flop_per_cycle", 1.0)),
                expected_best_kernel=expected_crr,
                csv_pattern=csv_pattern,
                expected_best_kernel_rrr=base_exp,
                csv_pattern_rrr=csv_pattern_rrr,
                measured_peak_flop_per_cycle=float(spec["measured_peak_flop_per_cycle"]) if "measured_peak_flop_per_cycle" in spec else None,
                hardware_max_frequency_ghz=float(spec["hardware_max_frequency_ghz"]) if "hardware_max_frequency_ghz" in spec else None,
                notes=spec.get("notes"),
            )
    return specs


# ==============================================================================
# Data Loading and Normalization
# ==============================================================================

def find_crr_csv_files(input_dir: Path, spec: UArchSpecCRR) -> List[Tuple[Path, str]]:
    """Finds CSV files matching the CRR uarch specification and returns (path, compiler)."""
    matched: List[Tuple[Path, str]] = []
    if not input_dir.exists():
        return matched

    for compiler in ["gcc", "clang"]:
        pattern = spec.csv_pattern.replace("{compiler}", compiler)
        for p in input_dir.glob("**/*.csv"):
            if fnmatch.fnmatch(p.name, pattern):
                if (p, compiler) not in matched:
                    matched.append((p, compiler))
            elif fnmatch.fnmatch(p.name, f"*gemm_results_*{spec.uarch_id}*crr*_{compiler}.csv") or \
                 fnmatch.fnmatch(p.name, f"*gemm_results_{spec.uarch_id}*crr*_{compiler}.csv") or \
                 fnmatch.fnmatch(p.name, f"*{spec.uarch_id}*crr*_{compiler}.csv") or \
                 (input_dir.name == "crr" and fnmatch.fnmatch(p.name, f"*gemm_results_*{spec.uarch_id}*_{compiler}.csv")):
                if (p, compiler) not in matched:
                    matched.append((p, compiler))

    return matched


def load_crr_dataset_for_uarch(input_dir: Path, spec: UArchSpecCRR) -> pd.DataFrame:
    """Loads and enriches CRR benchmark CSV files for a given uarch."""
    files = find_crr_csv_files(input_dir, spec)
    dfs: List[pd.DataFrame] = []

    for path, compiler in files:
        try:
            df = pd.read_csv(path)
            if df.empty or "Kernel" not in df.columns or "GFLOPS" not in df.columns:
                continue

            # Ensure we only retain CRR kernels
            df = df[df["Kernel"].astype(str).str.contains("_crr|panel", regex=True)].copy()
            if df.empty:
                continue

            if "FLOP_per_cycle" in df.columns and (pd.to_numeric(df["FLOP_per_cycle"], errors="coerce") > 0).any():
                flop_per_cycle = pd.to_numeric(df["FLOP_per_cycle"], errors="coerce").fillna(df["GFLOPS"] / spec.frequency_ghz)
            else:
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


def load_rrr_dataset_for_uarch(rrr_dir: Path, spec: UArchSpecCRR) -> pd.DataFrame:
    """Loads RRR benchmark CSV files for comparison."""
    if not rrr_dir.exists():
        return pd.DataFrame()

    matched: List[Tuple[Path, str]] = []
    for compiler in ["gcc", "clang"]:
        pattern = spec.csv_pattern_rrr.replace("{compiler}", compiler)
        for p in rrr_dir.glob("**/*.csv"):
            if fnmatch.fnmatch(p.name, pattern):
                if (p, compiler) not in matched:
                    matched.append((p, compiler))
            elif fnmatch.fnmatch(p.name, f"*gemm_results_*{spec.uarch_id}*_{compiler}.csv") or \
                 fnmatch.fnmatch(p.name, f"*gemm_results_{spec.uarch_id}*_{compiler}.csv"):
                if (p, compiler) not in matched:
                    matched.append((p, compiler))

    dfs: List[pd.DataFrame] = []
    for path, compiler in matched:
        try:
            df = pd.read_csv(path)
            if df.empty or "Kernel" not in df.columns or "GFLOPS" not in df.columns:
                continue

            # In RRR comparison, strictly exclude CRR / panel kernels
            df = df[~df["Kernel"].astype(str).str.contains("_crr|panel", regex=True)].copy()
            if df.empty:
                continue

            if "FLOP_per_cycle" in df.columns and (pd.to_numeric(df["FLOP_per_cycle"], errors="coerce") > 0).any():
                flop_per_cycle = pd.to_numeric(df["FLOP_per_cycle"], errors="coerce").fillna(df["GFLOPS"] / spec.frequency_ghz)
            else:
                flop_per_cycle = df["GFLOPS"] / spec.frequency_ghz
            efficiency_pct = (flop_per_cycle / spec.peak_flop_per_cycle) * 100.0

            df = df.assign(
                Compiler=compiler,
                UArchId=spec.uarch_id,
                SIMD=spec.simd,
                FrequencyGHz=spec.frequency_ghz,
                PeakFlopPerCycle=spec.peak_flop_per_cycle,
                PeakGFLOPS=spec.peak_gflops,
                FLOP_per_cycle=flop_per_cycle,
                Efficiency_pct=efficiency_pct,
            )
            dfs.append(df)
        except Exception:
            pass

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
class UArchChampionCRR:
    uarch_id: str
    spec: UArchSpecCRR
    kernel: str
    compiler: str
    mr: int
    nr: int
    peak_gflops: float
    peak_flop_per_cycle: float
    efficiency_pct: float
    measured_efficiency_pct: Optional[float]
    matches_expected: bool
    data: pd.DataFrame
    peak_k: int = 1024


@dataclass
class ComparisonPair:
    champ_crr: UArchChampionCRR
    rrr_kernel: str
    rrr_compiler: str
    rrr_peak_gflops: float
    rrr_peak_flop_per_cycle: float
    rrr_efficiency_pct: float
    speedup: float


def analyze_crr_uarch(df: pd.DataFrame, spec: UArchSpecCRR) -> Optional[UArchChampionCRR]:
    """Identifies the champion CRR kernel for a uarch based on peak GFLOP/s."""
    if df.empty:
        return None

    peak_by_kernel_compiler = df.groupby(["Kernel", "Compiler"], as_index=False)["GFLOPS"].max()
    best_row = peak_by_kernel_compiler.sort_values("GFLOPS", ascending=False).iloc[0]

    best_kernel = best_row["Kernel"]
    best_compiler = best_row["Compiler"]

    champion_data = df[(df["Kernel"] == best_kernel) & (df["Compiler"] == best_compiler)].sort_values("K")
    peak_row = champion_data.sort_values("GFLOPS", ascending=False).iloc[0]
    peak_gflops = float(peak_row["GFLOPS"])
    peak_flop_per_cycle = float(peak_row["FLOP_per_cycle"])
    peak_k = int(peak_row["K"])
    mr = int(peak_row["M"])
    nr = int(peak_row["N"])

    efficiency_pct = (peak_flop_per_cycle / spec.peak_flop_per_cycle) * 100.0
    measured_eff = (peak_flop_per_cycle / spec.measured_peak_flop_per_cycle * 100.0) if spec.measured_peak_flop_per_cycle else None
    matches_expected = (best_kernel == spec.expected_best_kernel)

    return UArchChampionCRR(
        uarch_id=spec.uarch_id,
        spec=spec,
        kernel=best_kernel,
        compiler=best_compiler,
        mr=mr,
        nr=nr,
        peak_gflops=peak_gflops,
        peak_flop_per_cycle=peak_flop_per_cycle,
        efficiency_pct=efficiency_pct,
        measured_efficiency_pct=measured_eff,
        matches_expected=matches_expected,
        data=champion_data,
        peak_k=peak_k,
    )


# ==============================================================================
# Visualization Engine (Publication-Grade Light Mode)
# ==============================================================================

def format_crr_axis_label(c: UArchChampionCRR) -> str:
    """Formats a concise two-line label for the Y-axis."""
    name = c.spec.name
    model = c.spec.cpu_model
    if "Strix Point" in name and "HX 370" in model:
        line1 = "AMD Zen 5 (Ryzen AI 9 HX 370)"
    elif "Meteor Lake" in name:
        line1 = "Intel Meteor Lake (Core Ultra 9 185H)"
    elif model and model not in name:
        line1 = f"{name} ({model})"
    else:
        line1 = name
    line2 = f"{c.spec.simd.upper()} @ {c.spec.frequency_ghz:.1f} GHz"
    return f"{line1}\n{line2}"


def group_crr_champions_by_family(champions: List[UArchChampionCRR], sort_metric: str = "efficiency") -> List[UArchChampionCRR]:
    """Groups champions by family: AVX-512, AVX2, NEON, RVV.

    Returns the list sorted so that AVX-512 appears at the top of a barh chart.
    """
    family_order = ["avx512", "avx2", "neon", "rvv"]
    grouped: List[UArchChampionCRR] = []
    for fam in family_order:
        fam_champs = [c for c in champions if c.spec.simd == fam]
        if sort_metric == "efficiency":
            fam_champs.sort(key=lambda c: c.efficiency_pct, reverse=True)
        else:
            fam_champs.sort(key=lambda c: c.peak_flop_per_cycle, reverse=True)
        grouped.extend(fam_champs)

    return list(reversed(grouped))


def plot_crr_cross_uarch_efficiency(champions: List[UArchChampionCRR], output_path: Path) -> None:
    """Generates an executive horizontal ranking bar chart grouped by SIMD family for CRR micro-kernels."""
    fig, ax = plt.subplots(figsize=(14.0, 7.5))

    sorted_champs = group_crr_champions_by_family(champions, sort_metric="efficiency")
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

    for idx, (champ, bar) in enumerate(zip(sorted_champs, bars)):
        eff = champ.efficiency_pct
        gflops = champ.peak_gflops
        flop_cyc = champ.peak_flop_per_cycle
        k_dim = champ.peak_k
        tile_str = f"{champ.mr}x{champ.nr}"

        text_label = f" {eff:.1f}%  ({gflops:.1f} GFLOP/s @ K={k_dim} | Tile MRxNR={tile_str} | {flop_cyc:.2f} FLOP/cyc)"
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

    y_labels = [format_crr_axis_label(c) for c in sorted_champs]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9.5)

    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax.set_xlabel("Hardware Efficiency (% of Theoretical Peak FLOP/cycle)", fontsize=11, fontweight="bold", labelpad=10)

    fig.text(
        0.04, 0.965,
        "Cross-Microarchitecture GEMM Peak Efficiency — CRR BLIS Micro-Kernels",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT_PRIMARY,
        ha="left",
    )
    fig.text(
        0.04, 0.935,
        "FP64 BLIS-like Micro-Kernel (A PackedColMajor, B PackedRowMajor) | Single tile MRxNR streaming over K ∈ [32..1024]",
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

    plt.subplots_adjust(top=0.91, bottom=0.10, left=0.30, right=0.96)
    if output_path.suffix == ".png":
        fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    else:
        fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_crr_cross_uarch_flop_per_cycle(champions: List[UArchChampionCRR], output_path: Path) -> None:
    """Generates comparison in raw DP FLOP/cycle grouped by SIMD family for CRR micro-kernels."""
    fig, ax = plt.subplots(figsize=(14.0, 7.5))

    sorted_champs = group_crr_champions_by_family(champions, sort_metric="flop_cycle")
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

    for idx, (champ, bar) in enumerate(zip(sorted_champs, bars)):
        val = champ.peak_flop_per_cycle
        tile_str = f"{champ.mr}x{champ.nr}"
        if val >= 10.0:
            text_label = f" {val:.2f} FLOP/cyc (Tile: {tile_str} | Peak: {champ.peak_gflops:.1f} GFLOP/s)"
            fsize = 9.5
        elif val >= 7.0:
            text_label = f" {val:.2f} FLOP/cyc ({tile_str} | {champ.peak_gflops:.1f} GFLOP/s)"
            fsize = 8.5
        elif val >= 6.0:
            text_label = f" {val:.2f} FLOP/cyc ({tile_str} | {champ.peak_gflops:.1f} GFLOP/s)"
            fsize = 8.5
        else:
            text_label = f" {val:.2f} FLOP/cyc ({tile_str} | {champ.peak_gflops:.1f} GFLOP/s)"
            fsize = 7.5
        ax.text(
            0.15,
            idx,
            text_label,
            va="center",
            ha="left",
            color="#ffffff",
            fontweight="bold",
            fontsize=fsize,
            zorder=5,
        )

    # Theoretical ceiling lines: 8 FLOP/cyc and 16 FLOP/cyc groups
    idx_8 = [i for i, c in enumerate(sorted_champs) if np.isclose(c.spec.peak_flop_per_cycle, 8.0)]
    idx_16 = [i for i, c in enumerate(sorted_champs) if np.isclose(c.spec.peak_flop_per_cycle, 16.0)]

    if idx_8 and idx_16:
        split_y = (max(idx_8) + min(idx_16)) / 2.0
        ax.axhline(split_y, color=BORDER_COLOR, linestyle=":", linewidth=1.0, zorder=2)

    if idx_8:
        ymin_8 = min(idx_8) - 0.45
        ymax_8 = max(idx_8) + 0.45
        ax.vlines(8.0, ymin_8, ymax_8, color="#334155", linestyle="--", linewidth=1.8, zorder=4)
        ax.text(
            8.15,
            (ymin_8 + ymax_8) / 2.0,
            "Theoretical Peak: 8 FLOP/cyc",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#334155",
            zorder=4,
        )

    if idx_16:
        ymin_16 = min(idx_16) - 0.45
        ymax_16 = max(idx_16) + 0.45
        ax.vlines(16.0, ymin_16, ymax_16, color="#334155", linestyle="--", linewidth=1.8, zorder=4)
        ax.text(
            16.0,
            ymax_16 + 0.15,
            "Theoretical Peak: 16 FLOP/cyc",
            ha="right",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#334155",
            zorder=4,
        )

    y_labels = [format_crr_axis_label(c) for c in sorted_champs]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9.5)

    ax.set_xlim(0, 18.0)
    ax.set_xticks(range(0, 19, 2))
    ax.set_ylim(-0.6, len(sorted_champs) - 0.4 + 0.6)
    ax.set_xlabel("Compute Density (DP FLOP / cycle)", fontsize=11, fontweight="bold", labelpad=10)

    fig.text(
        0.04, 0.965,
        "Cross-Microarchitecture GEMM Compute Density — CRR BLIS Micro-Kernels",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT_PRIMARY,
        ha="left",
    )
    fig.text(
        0.04, 0.935,
        "Higher is better | FP64 BLIS-like CRR micro-kernel streaming over K ∈ [32..1024] | Single native register tile",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        ha="left",
    )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SIMD_FAMILY_COLORS["avx512"], edgecolor=BORDER_COLOR, label="AVX-512"),
        Patch(facecolor=SIMD_FAMILY_COLORS["avx2"], edgecolor=BORDER_COLOR, label="AVX2"),
        Patch(facecolor=SIMD_FAMILY_COLORS["neon"], edgecolor=BORDER_COLOR, label="NEON"),
        Patch(facecolor=SIMD_FAMILY_COLORS["rvv"], edgecolor=BORDER_COLOR, label="RVV 1.0"),
        Line2D([0], [0], color="#334155", linestyle="--", linewidth=1.8, label="Theoretical Peak"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
        ncol=2,
        framealpha=0.95,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        fontsize=9.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.9)
    ax.yaxis.grid(False)

    plt.subplots_adjust(top=0.91, bottom=0.10, left=0.30, right=0.96)
    if output_path.suffix == ".png":
        fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    else:
        fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_crr_uarch_overview(df: pd.DataFrame, champ: UArchChampionCRR, output_path: Path) -> None:
    """Generates a single-panel publication figure showing scaling vs panel depth K for CRR micro-kernels."""
    fig, ax1 = plt.subplots(figsize=(10.5, 6.0))
    ax2 = ax1.twinx()

    freq = champ.spec.frequency_ghz
    peak_gflops = champ.spec.peak_gflops
    peak_flop_per_cycle = champ.spec.peak_flop_per_cycle

    sorted_k = sorted(df["K"].unique())

    # Theoretical peak horizontal line
    ax1.axhline(
        peak_gflops,
        color="#1e293b",
        linestyle="--",
        linewidth=1.5,
        label=f"Theoretical Peak: {peak_gflops:.1f} GFLOP/s ({peak_flop_per_cycle:.0f} FLOP/cyc | 100%)",
        zorder=2,
    )

    # Top evaluated CRR kernels
    kernel_peaks = df.groupby("Kernel")["GFLOPS"].max().sort_values(ascending=False)
    top_kernels = list(kernel_peaks.index[:5])

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
        })

    # Sort so champion is plotted last (on top)
    panel_kernels = sorted(panel_kernels, key=lambda x: x["gflops"])

    line_palette = ["#0284c7", "#7c3aed", "#db2777", "#dc2626", "#d97706"]
    line_markers = ["s", "^", "D", "v", "P"]

    for idx, k_info in enumerate(panel_kernels):
        k_df = df[(df["Kernel"] == k_info["kernel"]) & (df["Compiler"] == k_info["compiler"])].sort_values("K")
        is_champ = k_info["is_champ"]

        c = "#059669" if is_champ else line_palette[idx % len(line_palette)]
        m = "o" if is_champ else line_markers[idx % len(line_markers)]
        lw = 2.6 if is_champ else 1.6
        ls = "-" if is_champ else "--"
        ms = 7 if is_champ else 6

        lbl = f"{k_info['kernel']} [{k_info['compiler']}]"
        if is_champ:
            lbl = f"★ {lbl} (Champion — Peak: {champ.peak_gflops:.1f} GFLOP/s @ K={champ.peak_k} | Tile: {champ.mr}x{champ.nr})"

        ax1.plot(
            k_df["K"],
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
            ax1.fill_between(k_df["K"], 0, k_df["GFLOPS"], color=c, alpha=0.06, zorder=1)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sorted_k)
    ax1.set_xticklabels([str(s) for s in sorted_k], fontsize=10)
    ax1.set_xlabel("Panel Depth Dimension K (A: MR x K, B: K x NR)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Throughput (GFLOP/s)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax1.set_ylim(0, max(peak_gflops, champ.peak_gflops) * 1.08)

    # Right axis: Hardware efficiency (% of theoretical peak)
    ax2.set_ylim(0, 108)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax2.set_ylabel("Hardware Efficiency (% of Theoretical Peak)", fontsize=10.5, fontweight="bold", labelpad=8)

    ax1.legend(
        loc="lower right",
        framealpha=0.94,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        fontsize=9,
    )

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["left"].set_color(BORDER_COLOR)
    ax2.spines["right"].set_color(BORDER_COLOR)
    ax1.spines["bottom"].set_color(BORDER_COLOR)
    ax1.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.9)
    ax2.grid(False)

    title = f"{champ.spec.name} ({champ.spec.cpu_model}) — {champ.spec.simd.upper()} @ {freq:.1f} GHz"
    subtitle = (
        f"Champion: {champ.kernel} [{champ.compiler}] | Native Tile: MR={champ.mr}, NR={champ.nr} | "
        f"Peak: {champ.peak_gflops:.1f} GFLOP/s ({champ.efficiency_pct:.1f}% @ K={champ.peak_k})"
    )
    ax1.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=12)

    plt.subplots_adjust(top=0.89, bottom=0.11, left=0.10, right=0.90)
    if output_path.suffix == ".png":
        fig.savefig(output_path, format="png", dpi=150)
    else:
        fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"Generated: {output_path}")


def plot_crr_vs_rrr_comparison(pairs: List[ComparisonPair], output_path: Path) -> None:
    """Generates a grouped horizontal bar chart comparing RRR and CRR peak hardware efficiency."""
    fig, ax = plt.subplots(figsize=(14.5, 8.0))

    # Sort so AVX-512 is at top
    family_order = ["avx512", "avx2", "neon", "rvv"]
    sorted_pairs: List[ComparisonPair] = []
    for fam in family_order:
        fam_pairs = [p for p in pairs if p.champ_crr.spec.simd == fam]
        fam_pairs.sort(key=lambda p: p.champ_crr.efficiency_pct, reverse=True)
        sorted_pairs.extend(fam_pairs)

    sorted_pairs = list(reversed(sorted_pairs))
    y_positions = np.arange(len(sorted_pairs))
    bar_height = 0.36

    # Draw RRR bars (offset downward)
    rrr_y = y_positions - bar_height / 2.0 - 0.02
    crr_y = y_positions + bar_height / 2.0 + 0.02

    crr_colors = [SIMD_FAMILY_COLORS.get(p.champ_crr.spec.simd, "#0284c7") for p in sorted_pairs]
    rrr_colors = ["#94a3b8" for _ in sorted_pairs] # Slate gray for baseline

    bars_rrr = ax.barh(
        rrr_y,
        [p.rrr_efficiency_pct for p in sorted_pairs],
        height=bar_height,
        color=rrr_colors,
        edgecolor=BORDER_COLOR,
        linewidth=0.8,
        label="RRR (Full Matrix Baseline, ld=M)",
        zorder=3,
    )

    bars_crr = ax.barh(
        crr_y,
        [p.champ_crr.efficiency_pct for p in sorted_pairs],
        height=bar_height,
        color=crr_colors,
        edgecolor=BORDER_COLOR,
        linewidth=0.8,
        label="CRR (BLIS Micro-Kernel, ld=MR)",
        zorder=3,
    )

    # 100% Theoretical Peak line
    ax.axvline(100.0, color="#1e293b", linestyle="--", linewidth=1.5, zorder=4)
    ax.text(
        100.2,
        len(sorted_pairs) - 0.3,
        "100% Theoretical Peak",
        color="#1e293b",
        fontsize=9,
        fontweight="bold",
        va="center",
        zorder=5,
    )

    # Add text inside bars
    for idx, p in enumerate(sorted_pairs):
        # RRR label
        ax.text(
            1.2,
            rrr_y[idx],
            f" RRR: {p.rrr_efficiency_pct:.1f}% ({p.rrr_peak_gflops:.1f} GFLOP/s)",
            va="center",
            ha="left",
            color="#ffffff",
            fontweight="bold",
            fontsize=8.5,
            zorder=5,
        )
        # CRR label with speedup
        speedup_str = f"+{(p.speedup - 1.0) * 100.0:+.1f}%" if p.speedup >= 1.0 else f"{(p.speedup - 1.0) * 100.0:.1f}%"
        ax.text(
            1.2,
            crr_y[idx],
            f" CRR: {p.champ_crr.efficiency_pct:.1f}% ({p.champ_crr.peak_gflops:.1f} GFLOP/s | {p.speedup:.2f}x [{speedup_str}])",
            va="center",
            ha="left",
            color="#ffffff",
            fontweight="bold",
            fontsize=9.0,
            zorder=5,
        )

    y_labels = [format_crr_axis_label(p.champ_crr) for p in sorted_pairs]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9.5)

    ax.set_xlim(0, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax.set_xlabel("Hardware Efficiency (% of Theoretical Peak FLOP/cycle)", fontsize=11, fontweight="bold", labelpad=10)

    fig.text(
        0.04, 0.965,
        "Performance Impact: CRR BLIS Micro-Kernel vs RRR Full Matrix Baseline",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT_PRIMARY,
        ha="left",
    )
    fig.text(
        0.04, 0.935,
        "Quantifying speedup from contiguous Column-Major A access (ld=MR) and register spill elimination",
        fontsize=9.5,
        color=TEXT_SECONDARY,
        ha="left",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#94a3b8", edgecolor=BORDER_COLOR, label="RRR (Full Matrix Baseline, ld=M)"),
        Patch(facecolor="#0284c7", edgecolor=BORDER_COLOR, label="CRR (BLIS Micro-Kernel, ld=MR)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        framealpha=0.95,
        edgecolor=BORDER_COLOR,
        facecolor=CARD_BG,
        fontsize=9.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.9)
    ax.yaxis.grid(False)

    plt.subplots_adjust(top=0.91, bottom=0.10, left=0.30, right=0.96)
    if output_path.suffix == ".png":
        fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    else:
        fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


# ==============================================================================
# Console Reporting
# ==============================================================================

def print_crr_summary_table(champions: List[UArchChampionCRR]) -> None:
    """Prints a formatted comparative table in the console."""
    print("\n" + "=" * 145)
    print("  GEMMBench — Microarchitectural Performance & Peak Efficiency Summary (CRR BLIS Micro-Kernels)")
    print("=" * 145)
    header = (
        f"{'SIMD':<8} | {'Microarchitecture':<22} | {'Champion CRR Kernel':<36} | "
        f"{'Tile':<7} | {'Comp':<5} | {'Peak GFLOP/s':<12} | {'K_opt':<6} | {'FLOP/cyc':<9} | {'% Theo':<8} | {'Expected Match?':<15}"
    )
    print(header)
    print("-" * 145)

    for champ in champions:
        match_str = "✓ MATCH" if champ.matches_expected else f"✗ MISMATCH ({champ.spec.expected_best_kernel})"
        tile_str = f"{champ.mr}x{champ.nr}"
        row = (
            f"{champ.spec.simd:<8} | "
            f"{champ.spec.name:<22} | "
            f"{champ.kernel:<36} | "
            f"{tile_str:<7} | "
            f"{champ.compiler:<5} | "
            f"{champ.peak_gflops:<12.1f} | "
            f"{champ.peak_k:<6} | "
            f"{champ.peak_flop_per_cycle:<9.2f} | "
            f"{champ.efficiency_pct:<7.1f}% | "
            f"{match_str}"
        )
        print(row)
    print("=" * 145 + "\n")


def print_comparison_table(pairs: List[ComparisonPair]) -> None:
    """Prints a comparative table contrasting RRR and CRR peaks."""
    print("\n" + "=" * 145)
    print("  GEMMBench — Microarchitectural Comparison: RRR Baseline vs CRR BLIS Micro-Kernel")
    print("=" * 145)
    header = (
        f"{'SIMD':<8} | {'Microarchitecture':<22} | {'RRR Peak (GFLOP/s)':<18} | {'RRR %Theo':<10} | "
        f"{'CRR Peak (GFLOP/s)':<18} | {'CRR %Theo':<10} | {'Tile':<7} | {'Speedup':<9}"
    )
    print(header)
    print("-" * 145)

    for p in pairs:
        speedup_str = f"{p.speedup:.2f}x"
        tile_str = f"{p.champ_crr.mr}x{p.champ_crr.nr}"
        row = (
            f"{p.champ_crr.spec.simd:<8} | "
            f"{p.champ_crr.spec.name:<22} | "
            f"{p.rrr_peak_gflops:<18.1f} | "
            f"{p.rrr_efficiency_pct:<9.1f}% | "
            f"{p.champ_crr.peak_gflops:<18.1f} | "
            f"{p.champ_crr.efficiency_pct:<9.1f}% | "
            f"{tile_str:<7} | "
            f"{speedup_str:<9}"
        )
        print(row)
    print("=" * 145 + "\n")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and plot GEMM CRR (BLIS-like) benchmarks with automatic champion selection and FLOP/cycle efficiency."
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
        default=None,
        help="Directory containing CRR benchmark CSV files (default: auto-detected between results/crr, results, gemm-bench-results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots_crr"),
        help="Directory to save generated SVG plots (default: plots_crr)",
    )
    parser.add_argument(
        "--uarch",
        nargs="+",
        help="Filter specific microarchitecture(s) by ID (e.g. --uarch zen4 m1)",
    )
    parser.add_argument(
        "--simd",
        nargs="+",
        help="Filter specific SIMD family (e.g. --simd avx512 neon rvv avx2)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        help="Override frequencies in GHz (e.g. --freq zen4=5.0,m1=3.2)",
    )
    parser.add_argument(
        "--compare-rrr",
        type=Path,
        default=None,
        help="Optional directory containing RRR benchmark CSV files to generate speedup comparison",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export figures as PNG alongside SVG",
    )
    args = parser.parse_args()

    # 1. Determine input directory
    input_dir = args.input_dir
    if input_dir is None:
        if Path("gemm-bench-results/crr").is_dir():
            input_dir = Path("gemm-bench-results/crr")
        elif Path("results/crr").is_dir():
            input_dir = Path("results/crr")
        elif Path("results").is_dir():
            input_dir = Path("results")
        else:
            input_dir = Path("gemm-bench-results")

    print(f"Scanning for CRR benchmark results in: {input_dir}")

    # 2. Load microarchitectural specifications
    specs = load_uarch_config_crr(args.config)

    # 3. Apply frequency overrides if specified
    if args.freq:
        for pair in args.freq.split(","):
            if "=" in pair:
                u_id, f_str = pair.strip().split("=", 1)
                if u_id in specs:
                    specs[u_id].frequency_ghz = float(f_str)

    # 4. Filter targets if requested
    target_specs = list(specs.values())
    if args.simd:
        target_specs = [s for s in target_specs if s.simd in args.simd]
    if args.uarch:
        target_specs = [s for s in target_specs if s.uarch_id in args.uarch]

    if not target_specs:
        print("No matching microarchitectures found with specified filters.")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # 5. Load data and analyze champions
    champions: List[UArchChampionCRR] = []
    loaded_data: Dict[str, pd.DataFrame] = {}

    for spec in target_specs:
        df = load_crr_dataset_for_uarch(input_dir, spec)
        if df.empty:
            print(f"Notice: No CRR data found for {spec.name} ({spec.uarch_id}) in {input_dir}")
            continue

        champ = analyze_crr_uarch(df, spec)
        if champ:
            champions.append(champ)
            loaded_data[spec.uarch_id] = df

    if not champions:
        print("No valid CRR benchmark data could be loaded. Please check --input-dir.")
        return

    # 6. Generate Individual Overview Plots
    for champ in champions:
        u_df = loaded_data[champ.uarch_id]
        out_svg = args.output / f"{champ.spec.simd}_{champ.uarch_id}_crr_overview.svg"
        plot_crr_uarch_overview(u_df, champ, out_svg)
        if args.png:
            out_png = args.output / f"{champ.spec.simd}_{champ.uarch_id}_crr_overview.png"
            plot_crr_uarch_overview(u_df, champ, out_png)

    # 7. Generate Cross-UArch Comparison Plots
    if len(champions) > 1:
        plot_crr_cross_uarch_efficiency(champions, args.output / "cross_uarch_crr_efficiency_comparison.svg")
        plot_crr_cross_uarch_flop_per_cycle(champions, args.output / "cross_uarch_crr_flop_cycle_comparison.svg")
        if args.png:
            plot_crr_cross_uarch_efficiency(champions, args.output / "cross_uarch_crr_efficiency_comparison.png")
            plot_crr_cross_uarch_flop_per_cycle(champions, args.output / "cross_uarch_crr_flop_cycle_comparison.png")

    # 8. Print Console Summary
    print_crr_summary_table(champions)

    # 9. Optional RRR vs CRR comparison
    if args.compare_rrr:
        rrr_dir = args.compare_rrr
        print(f"Loading RRR baseline data from: {rrr_dir}")
        comp_pairs: List[ComparisonPair] = []

        for champ in champions:
            df_rrr = load_rrr_dataset_for_uarch(rrr_dir, champ.spec)
            if df_rrr.empty:
                continue

            # Identify best RRR kernel
            peak_row = df_rrr.sort_values("GFLOPS", ascending=False).iloc[0]
            rrr_gflops = float(peak_row["GFLOPS"])
            rrr_flop_cyc = float(peak_row["FLOP_per_cycle"])
            rrr_eff = float(peak_row["Efficiency_pct"])
            rrr_kernel = str(peak_row["Kernel"])
            rrr_comp = str(peak_row["Compiler"])
            speedup = champ.peak_gflops / rrr_gflops if rrr_gflops > 0 else 1.0

            comp_pairs.append(
                ComparisonPair(
                    champ_crr=champ,
                    rrr_kernel=rrr_kernel,
                    rrr_compiler=rrr_comp,
                    rrr_peak_gflops=rrr_gflops,
                    rrr_peak_flop_per_cycle=rrr_flop_cyc,
                    rrr_efficiency_pct=rrr_eff,
                    speedup=speedup,
                )
            )

        if comp_pairs:
            plot_crr_vs_rrr_comparison(comp_pairs, args.output / "cross_uarch_crr_vs_rrr_comparison.svg")
            if args.png:
                plot_crr_vs_rrr_comparison(comp_pairs, args.output / "cross_uarch_crr_vs_rrr_comparison.png")
            print_comparison_table(comp_pairs)


if __name__ == "__main__":
    main()
