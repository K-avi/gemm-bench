#!/usr/bin/env python3
"""
tools/compare_mippv2_blis.py

Calculates, analyzes, and visualizes MIPPv2 champion micro-kernels vs BLIS reference
micro-kernels with paper-grade metrology:
- Pre-registered champions from uarch_config.json (no post-hoc argmax by default)
- Matched-K ratios (FLOP/cycle at identical panel depth K)
- Hierarchical two-level bootstrap 95% confidence intervals
- Working-set memory footprint tracking and L1d cache boundary indicators
- Explicit per-compiler reporting (gcc vs clang) with dedicated best-compiler appendix
- Support for ablation variants: default, nofp, coldc, strict
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Headless Agg backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from gemmbench_common import (
    LIGHT_BG, CARD_BG, BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, GRID_COLOR,
    SIMD_FAMILY_COLORS, PLATFORM_META, L1_CACHE_BOUNDARY_K, get_platform_peak_fc,
    setup_matplotlib_theme, parse_benchmark_filename as parse_filename,
)
from paper_stats import (
    bootstrap_ratio_of_medians,
    flop_per_cycle_samples,
    geomean,
    get_l1d_bytes,
    load_raw,
    parse_filename_meta,
    working_set_bytes,
)

setup_matplotlib_theme()


def load_uarch_config(config_path: Optional[Path] = None) -> dict:
    """Loads uarch_config.json from repository root if available."""
    target = config_path or (REPO_ROOT / "uarch_config.json")
    if target.is_file():
        try:
            with open(target, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load uarch_config.json from {target}: {e}", file=sys.stderr)
    return {}


def get_expected_champion(
    platform: str,
    precision: str = "fp64",
    uarch_config: Optional[dict] = None,
) -> Optional[str]:
    """Resolves the pre-registered champion microkernel for a given platform and precision."""
    meta = PLATFORM_META.get(platform, {})
    uarch_id = meta.get("uarch_id")
    simd = meta.get("simd")

    if not uarch_config:
        uarch_config = load_uarch_config()

    if uarch_config and simd in uarch_config and uarch_id in uarch_config[simd]:
        spec = uarch_config[simd][uarch_id]
        if precision == "fp32" and spec.get("expected_best_kernel_crr_fp32"):
            return spec.get("expected_best_kernel_crr_fp32")
        if spec.get("expected_best_kernel_crr"):
            return spec.get("expected_best_kernel_crr")
        base = spec.get("expected_best_kernel", "")
        if base:
            return f"{base}_crr" if not base.endswith("_crr") else base

    # Fallback to general platform meta naming if uarch_config not resolved
    if platform.startswith("zen4") or platform.startswith("zen5"):
        return "mippv2_firestorm_mr4_nr4_fmaddi_crr"
    if platform.startswith("meteorlake") or platform.startswith("skylake"):
        return "mippv2_meteorlake_mr4_nr3_crr"
    if platform.startswith("m1"):
        return "mippv2_x60_mr6_nr4_fmaddi_crr"
    if platform.startswith("rpi5"):
        return "mippv2_a76_mr6_nr4_fmaddi_crr"
    if platform.startswith("x100"):
        return "mippv2_x100_register_blocked_apack4_crr"
    if platform.startswith("x60"):
        return "mippv2_x60_mr6_nr4_fmaddi_crr" if precision == "fp32" else "mippv2_a100_mr7_nr4_pipe_crr"
    if platform.startswith("a100"):
        return "mippv2_a100_mr7_nr2_lmul2_pipe_crr"

    return None


def load_dataset(input_dir: Path) -> pd.DataFrame:
    """Loads all benchmark CSV files in input_dir and annotates metadata."""
    csv_files = sorted([
        f for f in input_dir.glob("*.csv")
        if not f.name.endswith(".raw.csv") and f.name != "summary_metrics.csv"
    ])
    if not csv_files:
        print(f"Warning: No CSV files found in {input_dir}", file=sys.stderr)
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if df.empty or "Kernel" not in df.columns or "K" not in df.columns:
                continue
            meta = parse_filename_meta(f)
            df = df.assign(
                Platform=meta["Platform"],
                Layout=meta["Layout"],
                Compiler=meta["Compiler"],
                Variant=meta["Variant"],
                Precision=meta["Precision"],
                SourceFile=f.name,
                SourceDir=str(input_dir.resolve()),
            )
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}", file=sys.stderr)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def determine_parity_label(
    ratios: List[float],
    ci_lows: List[float],
    ci_highs: List[float],
    ci_available: bool,
) -> str:
    """Computes parity status label over a window of K."""
    if not ratios:
        return "no_data"

    if ci_available and all(not np.isnan(l) and not np.isnan(h) for l, h in zip(ci_lows, ci_highs)):
        # Parity: for all K, 95% CI overlaps [0.98, 1.02] -> low <= 1.02 and high >= 0.98
        if all(l <= 1.02 and h >= 0.98 for l, h in zip(ci_lows, ci_highs)):
            return "parity"
        if all(l > 1.02 for l in ci_lows):
            return "mipp_faster"
        if all(h < 0.98 for h in ci_highs):
            return "blis_faster"
        return "mixed"
    else:
        # Fallback without CI: suffix _noci
        if all(0.98 <= r <= 1.02 for r in ratios):
            return "parity_noci"
        if all(r > 1.02 for r in ratios):
            return "mipp_faster_noci"
        if all(r < 0.98 for r in ratios):
            return "blis_faster_noci"
        return "mixed_noci"


def analyze_dataset(
    df: pd.DataFrame,
    uarch_config: Optional[dict] = None,
    k_window: Tuple[int, int] = (128, 384),
    raw_dir: Optional[Path] = None,
    legacy_argmax: bool = False,
    variant_filter: str = "default",
    precision_filter: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Processes the benchmark dataframe with matched-K analysis and bootstrap CIs:
    Returns: (summary_df, detailed_df, details_dict)
    """
    if uarch_config is None:
        uarch_config = load_uarch_config()

    # Ensure required columns exist
    for col, dflt in [
        ("Variant", "default"),
        ("Precision", "fp64"),
        ("Layout", "crr"),
        ("Platform", "unknown"),
        ("Compiler", "unknown"),
    ]:
        if col not in df.columns:
            df[col] = dflt

    if variant_filter != "all":
        df = df[df["Variant"] == variant_filter]
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

    if precision_filter in ("fp32", "fp64"):
        df = df[df["Precision"] == precision_filter]
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

    summary_rows = []
    detailed_rows = []
    details_dict = defaultdict(dict)

    # Group strictly by (Platform, Compiler, Precision, Variant)
    groups = df.groupby(["Platform", "Compiler", "Precision", "Variant", "Layout"])

    for (platform, compiler, precision, variant, layout), group in groups:
        blis_data = group[group["Kernel"] == "blis_native"]
        mipp_data = group[group["Kernel"] != "blis_native"]

        if blis_data.empty or mipp_data.empty:
            continue

        if legacy_argmax:
            mipp_best_idx = mipp_data["GFLOPS_median"].idxmax()
            mipp_champ_kernel = str(mipp_data.loc[mipp_best_idx, "Kernel"])
        else:
            expected = get_expected_champion(platform, precision, uarch_config)
            if not expected or expected not in mipp_data["Kernel"].values:
                avail = sorted(mipp_data["Kernel"].unique())
                raise ValueError(
                    f"Pre-registered champion '{expected}' for platform '{platform}' ({precision}) "
                    f"not found in dataset. Available kernels: {avail}. "
                    f"Use --legacy-argmax-champion to override."
                )
            mipp_champ_kernel = expected

        mipp_champ_data = mipp_data[mipp_data["Kernel"] == mipp_champ_kernel]

        # Resolve raw samples if .raw.csv exists
        raw_df = None
        source_dir_val = group["SourceDir"].iloc[0] if "SourceDir" in group.columns else None
        source_file_val = group["SourceFile"].iloc[0] if "SourceFile" in group.columns else None
        meta_json_path = None

        if source_dir_val and source_file_val:
            s_dir = Path(source_dir_val)
            s_stem = Path(source_file_val).stem
            raw_path = s_dir / f"{s_stem}.raw.csv"
            if raw_path.is_file():
                try:
                    raw_df = load_raw(raw_path)
                except Exception:
                    raw_df = None
            cand_meta = s_dir / f"{s_stem}.meta.json"
            if cand_meta.is_file():
                meta_json_path = cand_meta

        elem_bytes = 4 if precision == "fp32" else 8
        uarch_id = PLATFORM_META.get(platform, {}).get("uarch_id")
        simd = PLATFORM_META.get(platform, {}).get("simd")
        uarch_spec = uarch_config.get(simd, {}).get(uarch_id, {}) if uarch_config and simd else {}
        l1d_bytes = get_l1d_bytes(meta_json_path, uarch_spec)

        common_ks = sorted(set(mipp_champ_data["K"]).intersection(set(blis_data["K"])))

        k_entries_for_group = []

        for k in common_ks:
            m_sub = mipp_champ_data[mipp_champ_data["K"] == k]
            b_sub = blis_data[blis_data["K"] == k]
            if m_sub.empty or b_sub.empty:
                continue

            m_row = m_sub.iloc[0]
            b_row = b_sub.iloc[0]

            m_fc = float(m_row["FLOP_per_cycle_median"]) if "FLOP_per_cycle_median" in m_row else 0.0
            b_fc = float(b_row["FLOP_per_cycle_median"]) if "FLOP_per_cycle_median" in b_row else 0.0

            if m_fc <= 0.0 and "GFLOPS_median" in m_row and "Eff_GHz" in m_row and float(m_row["Eff_GHz"]) > 0:
                ops_m = 2.0 * float(m_row["M"]) * float(m_row["N"]) * float(k)
                m_fc = float(m_row["GFLOPS_median"]) / float(m_row["Eff_GHz"])

            if b_fc <= 0.0 and "GFLOPS_median" in b_row and "Eff_GHz" in b_row and float(b_row["Eff_GHz"]) > 0:
                b_fc = float(b_row["GFLOPS_median"]) / float(b_row["Eff_GHz"])

            ratio = (m_fc / b_fc) if b_fc > 0 else float("nan")

            ci_low, ci_high = float("nan"), float("nan")
            ci_avail = False

            if raw_df is not None:
                samples_m = flop_per_cycle_samples(raw_df, mipp_champ_kernel, k)
                samples_b = flop_per_cycle_samples(raw_df, "blis_native", k)
                if samples_m and samples_b:
                    point_est, ci_low, ci_high = bootstrap_ratio_of_medians(samples_m, samples_b)
                    if not np.isnan(point_est):
                        m_fc = float(np.median(np.concatenate([v for v in samples_m.values() if len(v) > 0])))
                        b_fc = float(np.median(np.concatenate([v for v in samples_b.values() if len(v) > 0])))
                        ratio = point_est
                    ci_avail = True

            ws_m = working_set_bytes(int(m_row["M"]), int(m_row["N"]), k, elem_bytes)
            ws_b = working_set_bytes(int(b_row["M"]), int(b_row["N"]), k, elem_bytes)

            m_exceeds = (ws_m > l1d_bytes) if l1d_bytes else False
            b_exceeds = (ws_b > l1d_bytes) if l1d_bytes else False

            det_entry = {
                "Platform": platform,
                "Compiler": compiler,
                "Precision": precision,
                "Variant": variant,
                "Layout": layout,
                "K": int(k),
                "MIPP_kernel": mipp_champ_kernel,
                "MIPP_FC_med": m_fc,
                "BLIS_FC_med": b_fc,
                "Ratio": ratio,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "WS_MIPP_bytes": ws_m,
                "WS_BLIS_bytes": ws_b,
                "L1d_bytes": l1d_bytes if l1d_bytes is not None else np.nan,
                "MIPP_exceeds_L1d": m_exceeds,
                "BLIS_exceeds_L1d": b_exceeds,
                "ci_available": ci_avail,
            }
            detailed_rows.append(det_entry)
            k_entries_for_group.append(det_entry)

            details_dict[(platform, compiler, precision, variant, layout)][int(k)] = {
                "MIPPv2_GFLOPS": float(m_row["GFLOPS_median"]) if "GFLOPS_median" in m_row else 0.0,
                "BLIS_GFLOPS": float(b_row["GFLOPS_median"]) if "GFLOPS_median" in b_row else 0.0,
                "Ratio_pct": ratio * 100.0 if not np.isnan(ratio) else 0.0,
                "MIPPv2_FC": m_fc,
                "BLIS_FC": b_fc,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "CI_available": ci_avail,
                "WS_MIPP_bytes": ws_m,
                "WS_BLIS_bytes": ws_b,
                "MIPP_exceeds_L1d": m_exceeds,
                "BLIS_exceeds_L1d": b_exceeds,
                "L1d_bytes": l1d_bytes,
            }

        # Windowed summary calculation (k_window[0] <= K <= k_window[1])
        k_min, k_max = k_window
        window_entries = [e for e in k_entries_for_group if k_min <= e["K"] <= k_max and not np.isnan(e["Ratio"])]

        if window_entries:
            w_ratios = [e["Ratio"] for e in window_entries]
            w_ci_low = [e["CI_low"] for e in window_entries]
            w_ci_high = [e["CI_high"] for e in window_entries]
            w_ci_avail = all(e["ci_available"] for e in window_entries)

            r_geomean = geomean(w_ratios)
            r_min = float(np.min(w_ratios))
            r_max = float(np.max(w_ratios))
            valid_ci_low = [x for x in w_ci_low if not np.isnan(x)]
            valid_ci_high = [x for x in w_ci_high if not np.isnan(x)]
            ci_low_min = float(np.min(valid_ci_low)) if (w_ci_avail and valid_ci_low) else r_min
            ci_high_max = float(np.max(valid_ci_high)) if (w_ci_avail and valid_ci_high) else r_max
            n_k = len(window_entries)
            parity_label = determine_parity_label(w_ratios, w_ci_low, w_ci_high, w_ci_avail)
        else:
            r_geomean = float("nan")
            r_min = float("nan")
            r_max = float("nan")
            ci_low_min = float("nan")
            ci_high_max = float("nan")
            n_k = 0
            parity_label = "no_data"

        # Legacy peak-over-peak metric preserved under Legacy_PeakOverPeak_pct
        mipp_peak_gflops = mipp_champ_data["GFLOPS_median"].max() if "GFLOPS_median" in mipp_champ_data else 0.0
        blis_peak_gflops = blis_data["GFLOPS_median"].max() if "GFLOPS_median" in blis_data else 0.0
        legacy_pop_pct = (mipp_peak_gflops / blis_peak_gflops * 100.0) if blis_peak_gflops > 0 else 0.0

        best_row_idx = mipp_champ_data["GFLOPS_median"].idxmax() if "GFLOPS_median" in mipp_champ_data else 0
        best_k = int(mipp_champ_data.loc[best_row_idx, "K"]) if "GFLOPS_median" in mipp_champ_data else 0
        mipp_peak_fc = mipp_champ_data["FLOP_per_cycle_median"].max() if "FLOP_per_cycle_median" in mipp_champ_data else 0.0
        blis_peak_fc = blis_data["FLOP_per_cycle_median"].max() if "FLOP_per_cycle_median" in blis_data else 0.0

        summary_rows.append({
            "Platform": platform,
            "Compiler": compiler,
            "Precision": precision,
            "Variant": variant,
            "Layout": layout,
            "MIPP_Champion": mipp_champ_kernel,
            "Ratio_geomean": r_geomean,
            "Ratio_min": r_min,
            "Ratio_max": r_max,
            "CI_low_min": ci_low_min,
            "CI_high_max": ci_high_max,
            "N_K": n_k,
            "Parity_label": parity_label,
            "Legacy_PeakOverPeak_pct": legacy_pop_pct,
            # Backwards compatibility aliases
            "MIPPv2_Champion": mipp_champ_kernel,
            "MIPPv2_Peak_GFLOPS": mipp_peak_gflops,
            "BLIS_Peak_GFLOPS": blis_peak_gflops,
            "Ratio_Peak_GFLOPS_pct": legacy_pop_pct,
            "MIPPv2_Peak_FC": mipp_peak_fc,
            "BLIS_Peak_FC": blis_peak_fc,
            "Best_K": best_k,
        })

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    return summary_df, detailed_df, details_dict


# ==============================================================================
# Visualization Functions (Publication-Grade SVG & PNG)
# ==============================================================================

def plot_blis_ratio_comparison(
    summary_df: pd.DataFrame,
    out_dir: Path,
    enable_png: bool = True,
    is_fp32: bool = False,
    variant_name: str = "default",
) -> None:
    """
    Generates an executive horizontal bar chart of Ratio_geomean (% of BLIS)
    with [min CI_low, max CI_high] error bars over the K-window.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    sub_df = summary_df.copy()
    if "Precision" in sub_df.columns:
        target_prec = "fp32" if is_fp32 else "fp64"
        if (sub_df["Precision"] == target_prec).any():
            sub_df = sub_df[sub_df["Precision"] == target_prec]
    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.5, 2.2], hspace=0.35)

    ax_main = fig.add_subplot(gs[0])
    ax_a100 = fig.add_subplot(gs[1])

    main_plats = ["zen4_avx512", "zen5_avx512", "meteorlake_avx2", "m1_neon", "rpi5_neon", "x100_rvv", "x60_rvv"]

    def extract_rows(plats: List[str]):
        res = []
        for p in plats:
            p_sub = sub_df[sub_df["Platform"] == p]
            for comp in ["gcc", "clang"]:
                c_sub = p_sub[p_sub["Compiler"] == comp]
                if not c_sub.empty:
                    res.append((p, comp, c_sub.iloc[0]))
        return res

    main_rows = extract_rows(main_plats)
    y_pos = np.arange(len(main_rows))

    bar_colors = []
    ratios = []
    err_low = []
    err_high = []

    for p, comp, r in main_rows:
        simd = PLATFORM_META.get(p, {}).get("simd", "avx2")
        base_col = SIMD_FAMILY_COLORS.get(simd, "#0284c7")
        bar_colors.append(base_col if comp == "gcc" else matplotlib.colors.to_rgba(base_col, 0.72))

        val = (r["Ratio_geomean"] * 100.0) if not np.isnan(r["Ratio_geomean"]) else 0.0
        ratios.append(val)

        ci_l_raw = r.get("CI_low_min", r["Ratio_min"])
        ci_h_raw = r.get("CI_high_max", r["Ratio_max"])
        ci_l = (ci_l_raw * 100.0) if not np.isnan(ci_l_raw) else val
        ci_h = (ci_h_raw * 100.0) if not np.isnan(ci_h_raw) else val
        err_low.append(max(0.0, val - ci_l))
        err_high.append(max(0.0, ci_h - val))

    xerr = [err_low, err_high] if len(main_rows) > 0 else None

    bars = ax_main.barh(
        y_pos,
        ratios,
        xerr=xerr,
        capsize=3.5,
        height=0.62,
        color=bar_colors,
        edgecolor=BORDER_COLOR,
        linewidth=0.8,
        zorder=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#475569"},
    )

    ax_main.axvline(100.0, color="#475569", linestyle="--", linewidth=1.5, zorder=4)
    ax_main.text(100.5, len(main_rows) - 0.5, "100% BLIS Parity", color="#475569", fontsize=9.5, fontweight="bold", va="bottom", ha="left")

    for idx, ((p, comp, r), bar) in enumerate(zip(main_rows, bars)):
        val = ratios[idx]
        lbl = f"{val:.1f}% [{r['Ratio_min']*100:.1f}–{r['Ratio_max']*100:.1f}%] ({r['Parity_label']})"
        ax_main.text(val + 1.5, idx, lbl, va="center", ha="left", color=TEXT_PRIMARY, fontweight="bold", fontsize=9.0)

    labels = [f"{PLATFORM_META.get(p, {}).get('name', p)} ({comp.upper()})" for p, comp, _ in main_rows]
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(labels, fontsize=9.5)
    ax_main.invert_yaxis()
    ax_main.set_xlim(0, 130)
    prec_label = "SGEMM" if is_fp32 else "DGEMM"
    ax_main.set_xlabel(f"Matched-K Window Geometric Mean Ratio (% of BLIS)", fontsize=10.5, fontweight="bold")
    ax_main.set_title(f"Standard Architectural Targets — Matched-K Ratio ({variant_name})", fontsize=11.5, pad=8)

    # Panel 2: A100 Outlier
    a100_rows = extract_rows(["a100_rvv"])
    y_a100 = np.arange(len(a100_rows))
    a100_colors = [SIMD_FAMILY_COLORS["rvv"] if comp == "gcc" else matplotlib.colors.to_rgba(SIMD_FAMILY_COLORS["rvv"], 0.72) for _, comp, _ in a100_rows]
    a100_ratios = [(r["Ratio_geomean"] * 100.0) if not np.isnan(r["Ratio_geomean"]) else 0.0 for _, _, r in a100_rows]

    bars_a100 = ax_a100.barh(y_a100, a100_ratios, height=0.55, color=a100_colors, edgecolor=BORDER_COLOR, linewidth=0.8, zorder=3)
    ax_a100.axvline(100.0, color="#475569", linestyle="--", linewidth=1.5, zorder=4)
    ax_a100.text(101.5, 0.5, "100% BLIS (Fallback X60)", color="#475569", fontsize=9.0, fontweight="bold", va="center")

    for idx, ((p, comp, r), bar) in enumerate(zip(a100_rows, bars_a100)):
        val = a100_ratios[idx]
        speedup = val / 100.0
        lbl = f"{val:.1f}% ({speedup:.2f}× speedup)"
        ax_a100.text(val + 3.0, idx, lbl, va="center", ha="left", color=TEXT_PRIMARY, fontweight="bold", fontsize=9.0)

    labels_a100 = [f"{PLATFORM_META.get(p, {}).get('name', p)} ({comp.upper()})" for p, comp, _ in a100_rows]
    ax_a100.set_yticks(y_a100)
    ax_a100.set_yticklabels(labels_a100, fontsize=9.5)
    ax_a100.invert_yaxis()
    ax_a100.set_xlim(0, 420)
    ax_a100.set_xlabel(f"Matched-K Window Geometric Mean Ratio (% of BLIS)", fontsize=10.5, fontweight="bold")
    ax_a100.set_title("SpacemiT A100 VLEN=1024b Portability Study", fontsize=11.5, pad=8)

    fig.suptitle(f"GEMMBench — MIPPv2 vs BLIS Matched-K Geometric Mean Ratio ({prec_label}, {variant_name})", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005, "Bar represents Ratio_geomean; error whiskers show [min CI_low, max CI_high] over K window", ha="center", fontsize=8.5, color="#64748b", fontstyle="italic")

    svg_path = out_dir / "blis_vs_mippv2_ratio_comparison.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"✓ Generated SVG: {svg_path}")

    if enable_png:
        png_path = out_dir / "blis_vs_mippv2_ratio_comparison.png"
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        print(f"✓ Generated PNG: {png_path}")

    plt.close(fig)


def plot_k_scaling_grid(
    details: Dict,
    out_dir: Path,
    enable_png: bool = True,
    is_fp32: bool = False,
    variant_name: str = "default",
) -> None:
    """
    Generates a 2x4 multipanel grid showing ratio MIPPv2 / BLIS (%) scaling over K ∈ [32..1024]
    with 95% bootstrap confidence bands and vertical L1d cache boundary indicator.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.5), sharex=True)
    axes = axes.flatten()

    grid_plats = ["zen4_avx512", "zen5_avx512", "meteorlake_avx2", "m1_neon", "rpi5_neon", "x100_rvv", "a100_rvv", "x60_rvv"]

    for idx, p in enumerate(grid_plats):
        ax = axes[idx]
        meta = PLATFORM_META.get(p, {"name": p, "simd": "avx2", "peak_fc": 16.0})

        # Try gcc first, fallback to clang
        comp = "gcc"
        prec = "fp32" if is_fp32 else "fp64"
        key = (p, comp, prec, variant_name, "crr")

        if key not in details:
            comp = "clang"
            key = (p, comp, prec, variant_name, "crr")

        k_data = details.get(key, {})
        if not k_data:
            comp = "gcc" if comp == "clang" else "clang"
            key = (p, comp, prec, variant_name, "crr")
            k_data = details.get(key, {})

        ks = sorted(k_data.keys())
        ratios = [k_data[k]["Ratio_pct"] for k in ks]
        ci_lows = [(k_data[k]["CI_low"] * 100.0) if not np.isnan(k_data[k]["CI_low"]) else k_data[k]["Ratio_pct"] for k in ks]
        ci_highs = [(k_data[k]["CI_high"] * 100.0) if not np.isnan(k_data[k]["CI_high"]) else k_data[k]["Ratio_pct"] for k in ks]

        color = SIMD_FAMILY_COLORS.get(meta["simd"], "#0284c7")

        ax.axhline(100.0, color="#64748b", linestyle="--", linewidth=1.2, zorder=2)

        if ks:
            ax.plot(ks, ratios, marker="o", markersize=5.0, linewidth=2.0, color=color, label="MIPP / BLIS Ratio (%)", zorder=4)

            # Fill bootstrap CI band if available
            has_ci = any(k_data[k]["CI_available"] for k in ks)
            if has_ci:
                ax.fill_between(ks, ci_lows, ci_highs, color=color, alpha=0.22, zorder=3, label="95% Bootstrap CI")

            # Mark L1d spill transition
            l1d_spill_k = None
            for k in ks:
                if k_data[k]["MIPP_exceeds_L1d"] or k_data[k]["BLIS_exceeds_L1d"]:
                    l1d_spill_k = k
                    break

            if l1d_spill_k is not None:
                ax.axvline(l1d_spill_k, color="#f59e0b", linestyle=":", linewidth=1.4, alpha=0.9, zorder=2)
                ax.text(
                    l1d_spill_k * 1.05,
                    55.0 if p == "a100_rvv" else 65.0,
                    "L1d spill †",
                    color="#d97706",
                    fontsize=7.5,
                    fontweight="bold",
                    rotation=90,
                    va="bottom",
                )

        ax.set_title(f"{meta['name']} ({comp.upper()})", fontsize=11, fontweight="bold", pad=4)
        ax.set_xscale("log", base=2)
        if ks:
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks], fontsize=8.5)

        if p == "a100_rvv":
            ax.set_ylim(50, 420)
        else:
            ax.set_ylim(60, 140)

        if idx % 4 == 0:
            ax.set_ylabel("% of BLIS", fontsize=10, fontweight="bold")
        if idx >= 4:
            ax.set_xlabel("Panel Depth K", fontsize=10, fontweight="bold")

    axes[0].legend(loc="lower right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(f"Matched-K Ratio Across Panel Depth K ∈ [32, 1024] — MIPPv2 vs BLIS ({variant_name})", fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.91, bottom=0.09, left=0.06, right=0.98, hspace=0.28, wspace=0.22)
    fig.text(0.5, 0.015, "Shaded band: 95% hierarchical bootstrap CI; vertical line: first K exceeding L1d capacity", ha="center", fontsize=8.5, color="#64748b", fontstyle="italic")

    svg_path = out_dir / "blis_vs_mippv2_k_scaling_grid.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"✓ Generated SVG: {svg_path}")

    if enable_png:
        png_path = out_dir / "blis_vs_mippv2_k_scaling_grid.png"
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        print(f"✓ Generated PNG: {png_path}")

    plt.close(fig)


# ==============================================================================
# Markdown Report Generation
# ==============================================================================

def format_markdown_report(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    details: Dict,
    plot_dir: Optional[Path] = None,
    is_fp32: bool = False,
    out_md_path: Optional[Path] = None,
    k_window: Tuple[int, int] = (128, 384),
    variant_name: str = "default",
) -> str:
    """Generates a publication-grade GitHub-flavored Markdown comparative analysis document."""
    lines = []
    prec_str = "Single-Precision (FP32 / SGEMM)" if is_fp32 else "Double-Precision (FP64 / DGEMM)"
    k_min, k_max = k_window

    lines.append(f"# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels ({prec_str})\n")
    lines.append(f"**Ablation Variant:** `{variant_name}` | **Analysis Window:** $K \\in [{k_min}, {k_max}]$ (inclusive)\n")
    lines.append("Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions")
    lines.append(f"and hand-tuned BLIS assembly micro-kernels across target microarchitectures in {prec_str}.\n")

    lines.append(f"## 1. Summary: Matched-$K$ Window Ratio ($K \\in [{k_min}, {k_max}]$, Variant: `{variant_name}`)\n")
    lines.append("> [!NOTE]")
    lines.append("> **Statistical Methodology & Reporting Rules:**")
    lines.append(f"> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.")
    lines.append(f"> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \\text{{FLOP/cyc}}_{{\\text{{MIPP}}, K}} / \\text{{FLOP/cyc}}_{{\\text{{BLIS}}, K}}$).")
    lines.append(f"> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\\text{{geomean}}(R_K)$ for $K \\in [{k_min}, {k_max}]$.")
    lines.append(f"> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.\n")

    lines.append("| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |")
    lines.append("| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |")

    # Sort platforms for presentation
    for _, row in summary_df.iterrows():
        rg = f"**{row['Ratio_geomean']*100:.1f}%**" if not np.isnan(row["Ratio_geomean"]) else "N/A"
        rmin_max = f"[{row['Ratio_min']*100:.1f}%, {row['Ratio_max']*100:.1f}%]" if not np.isnan(row["Ratio_min"]) else "N/A"
        champ = str(row["MIPP_Champion"]).replace("mippv2_", "")

        label = row["Parity_label"]
        status_str = {
            "parity": "Parity (±2% CI)",
            "mipp_faster": "MIPPv2 Faster",
            "blis_faster": "BLIS Faster",
            "mixed": "Mixed Advantage",
            "parity_noci": "Parity (±2%, no CI)",
            "mipp_faster_noci": "MIPPv2 Faster (no CI)",
            "blis_faster_noci": "BLIS Faster (no CI)",
            "mixed_noci": "Mixed (no CI)",
            "no_data": "No Data",
        }.get(label, label)

        lines.append(
            f"| `{row['Platform']}` | `{row['Compiler']}` | `{row['Variant']}` | `{row['Precision']}` | "
            f"`{champ}` | {row['N_K']} | {rg} | {rmin_max} | {status_str} |"
        )

    lines.append("\n---\n")
    lines.append("## 2. Visual Comparative Figures\n")
    if plot_dir and plot_dir.exists():
        fig_prefix = "plots/"
        if out_md_path is not None:
            try:
                rel_dir = os.path.relpath(plot_dir.resolve(), Path(out_md_path).resolve().parent)
                fig_prefix = "" if rel_dir == "." else f"{rel_dir}/"
            except ValueError:
                pass
        lines.append(f"- **Matched-K Ratio Comparison:** [{fig_prefix}blis_vs_mippv2_ratio_comparison.svg]({fig_prefix}blis_vs_mippv2_ratio_comparison.svg)")
        lines.append(f"- **Scaling Grid Across $K \\in [32, 1024]$:** [{fig_prefix}blis_vs_mippv2_k_scaling_grid.svg]({fig_prefix}blis_vs_mippv2_k_scaling_grid.svg)\n")
    else:
        lines.append("*Figures can be generated by supplying the `--plot` flag.*\n")

    lines.append("\n---\n")
    lines.append(f"## 3. Per-Size Matched-$K$ Scaling & Working-Set Analysis (Variant: `{variant_name}`)\n")
    lines.append("> [!TIP]")
    lines.append(
        "> **Cache Boundary Semantics:** Rows marked with `†` denote operating points where panel working set "
        "$K \\times (M + N) \\times \\text{elem\\_bytes}$ exceeds L1d cache capacity. On architectures with large private L2 caches "
        "and aggressive hardware stream prefetchers (e.g. 1 MiB L2 on Zen 4 / Zen 5), L2-to-L1 bandwidth sustains near-100% FMA "
        "saturation beyond L1d; on cores with smaller L1d/L2 or stricter in-order/prefetch limits (e.g. X60, X100, Meteor Lake GCC), "
        "crossing the L1d boundary exposes cache-hierarchy throughput.\n"
    )

    for key, k_data in details.items():
        platform, compiler, precision, variant, layout = key
        lines.append(f"### Platform: `{platform}` ({compiler}, {precision}, {variant})\n")
        lines.append("| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |")
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for k in sorted(k_data.keys()):
            entry = k_data[k]
            r = entry["Ratio_pct"]
            if entry["CI_available"] and not np.isnan(entry["CI_low"]) and not np.isnan(entry["CI_high"]):
                if entry["CI_low"] <= 1.02 and entry["CI_high"] >= 0.98:
                    adv = "Parity (±2% CI)"
                elif entry["CI_low"] > 1.02:
                    adv = f"MIPPv2 (+{r - 100.0:.1f}%)"
                else:
                    adv = f"BLIS (+{100.0 - r:.1f}%)"
            else:
                adv = f"MIPPv2 (+{r - 100.0:.1f}%)" if r > 102.0 else (f"BLIS (+{100.0 - r:.1f}%)" if r < 98.0 else "Parity (±2%)")

            ci_str = f"[{entry['CI_low']*100:.1f}%, {entry['CI_high']*100:.1f}%]" if entry["CI_available"] else "N/A (no raw)"
            spill_mark = " †" if (entry["MIPP_exceeds_L1d"] or entry["BLIS_exceeds_L1d"]) else ""

            ws_m_kib = entry["WS_MIPP_bytes"] / 1024.0
            ws_b_kib = entry["WS_BLIS_bytes"] / 1024.0

            lines.append(
                f"| $K={k}${spill_mark} | {entry['MIPPv2_FC']:.2f} | {entry['BLIS_FC']:.2f} | "
                f"**{r:.1f}%** | {ci_str} | {ws_m_kib:.1f} | {ws_b_kib:.1f} | {adv} |"
            )
        lines.append("")

    lines.append("\n---\n")
    lines.append("## Appendix: Best-Compiler Envelope (Post-Hoc)\n")
    lines.append("> [!IMPORTANT]")
    lines.append("> This appendix computes the post-hoc per-$K$ envelope across compilers (`max(gcc, clang)` at each $K \\in [128, 384]$). It is presented exclusively for secondary reference.\n")

    lines.append("| Microarchitecture | Precision | Variant | Best Compiler by Ratio | Envelope Ratio Geomean | Envelope [Min, Max] |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Group by (Platform, Precision, Variant)
    env_groups = summary_df.groupby(["Platform", "Precision", "Variant"])
    for (plat, prec, var), grp in env_groups:
        if grp.empty:
            continue
        best_row = grp.loc[grp["Ratio_geomean"].idxmax()]
        if len(grp) > 1 and (grp["Ratio_geomean"].max() - grp["Ratio_geomean"].min()) < 0.001:
            comp_label = "`gcc` / `clang` (tie)"
        else:
            comp_label = f"`{best_row['Compiler']}`"

        d_sub = detailed_df[
            (detailed_df["Platform"] == plat)
            & (detailed_df["Precision"] == prec)
            & (detailed_df["Variant"] == var)
            & (detailed_df["K"] >= k_min)
            & (detailed_df["K"] <= k_max)
        ]
        if not d_sub.empty:
            per_k_max = d_sub.groupby("K")["Ratio"].max().values
            env_geo = geomean(per_k_max) * 100.0
            env_min = float(np.min(per_k_max)) * 100.0
            env_max = float(np.max(per_k_max)) * 100.0
        else:
            env_geo = best_row["Ratio_geomean"] * 100.0
            env_min = best_row["Ratio_min"] * 100.0
            env_max = best_row["Ratio_max"] * 100.0

        lines.append(
            f"| `{plat}` | `{prec}` | `{var}` | {comp_label} | "
            f"**{env_geo:.1f}%** | [{env_min:.1f}%, {env_max:.1f}%] |"
        )

    return "\n".join(lines)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare MIPPv2 champion kernels vs BLIS reference micro-kernels")
    parser.add_argument(
        "--fp32", "--float",
        action="store_true",
        dest="fp32",
        help="Run analysis on FP32 (SGEMM) datasets instead of default FP64",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing benchmark CSV files (default: gemm-bench-results/blis_comparison[_fp32])",
    )
    parser.add_argument(
        "--fallback-crr",
        action="store_true",
        help="Fallback to gemm-bench-results/crr[_fp32] if blis_comparison is empty",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        choices=["default", "nofp", "coldc", "strict", "all"],
        help="Filter results by ablation variant (default: default)",
    )
    parser.add_argument(
        "--k-window",
        type=str,
        default="128,384",
        help="Comma-separated K-window min,max for geometric mean summary (default: 128,384)",
    )
    parser.add_argument(
        "--legacy-argmax-champion",
        action="store_true",
        help="Select MIPPv2 champion by post-hoc argmax across kernels instead of pre-registered uarch_config.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Path to save generated Markdown report",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save summary CSV metrics",
    )
    parser.add_argument(
        "--output-detailed-csv",
        type=str,
        default=None,
        help="Path to save detailed per-K comparison CSV",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate publication-grade comparative SVG plots",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Custom directory to save plots (default: <input-dir>/plots)",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        default=True,
        help="Disable PNG raster generation (only generate SVG, default: True)",
    )
    parser.add_argument(
        "--png",
        action="store_false",
        dest="no_png",
        help="Enable PNG raster generation alongside SVG",
    )
    args = parser.parse_args()

    # Parse k-window
    try:
        kw_parts = [int(p.strip()) for p in args.k_window.split(",")]
        k_window = (kw_parts[0], kw_parts[1])
    except Exception:
        k_window = (128, 384)

    default_input = (
        "gemm-bench-results/paper_v1/default"
        if Path("gemm-bench-results/paper_v1/default").is_dir()
        else ("gemm-bench-results/blis_comparison_fp32" if args.fp32 else "gemm-bench-results/blis_comparison")
    )
    input_str = args.input_dir if args.input_dir else default_input
    input_path = Path(input_str)

    if not input_path.exists() or not list(input_path.glob("*.csv")):
        fallback_str = "gemm-bench-results/crr_fp32" if args.fp32 else "gemm-bench-results/crr"
        if args.fallback_crr and Path(fallback_str).exists():
            input_path = Path(fallback_str)
            print(f"Notice: Falling back to {input_path}", file=sys.stderr)
        else:
            print(f"Error: Directory '{input_path}' does not exist or contains no CSV files.", file=sys.stderr)
            print(f"Run the cluster benchmarks first: ./tools/run_cluster_benchmarks.sh all --crr-only --with-blis {'--fp32' if args.fp32 else ''}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading benchmark datasets from {input_path} (FP32={args.fp32}, Variant={args.variant})...")
    df = load_dataset(input_path)
    if df.empty:
        print("No valid data loaded.", file=sys.stderr)
        sys.exit(1)

    uarch_cfg = load_uarch_config()
    summary_df, detailed_df, details = analyze_dataset(
        df,
        uarch_config=uarch_cfg,
        k_window=k_window,
        legacy_argmax=args.legacy_argmax_champion,
        variant_filter=args.variant,
        precision_filter="fp32" if args.fp32 else "fp64",
    )

    plot_dir_path = None
    if args.plot or args.plot_dir:
        if args.input_dir:
            default_plot_dir = input_path / ("plots_fp32" if args.fp32 else "plots_fp64")
        else:
            default_plot_dir = Path("plots_blis_fp32") if args.fp32 else (input_path / "plots")
        plot_dir_path = Path(args.plot_dir) if args.plot_dir else default_plot_dir
        print(f"\nGenerating publication-grade comparative figures in {plot_dir_path}...")
        enable_png = not args.no_png
        plot_blis_ratio_comparison(summary_df, plot_dir_path, enable_png=enable_png, is_fp32=args.fp32, variant_name=args.variant)
        plot_k_scaling_grid(details, plot_dir_path, enable_png=enable_png, is_fp32=args.fp32, variant_name=args.variant)

    out_md_path = Path(args.output_md) if args.output_md else None
    if not out_md_path and args.plot:
        out_md_path = (plot_dir_path / "blis_fp32_summary_table.md") if args.fp32 else (input_path / "blis_vs_mippv2_report.md")

    md_report = format_markdown_report(
        summary_df,
        detailed_df,
        details,
        plot_dir=plot_dir_path,
        is_fp32=args.fp32,
        out_md_path=out_md_path,
        k_window=k_window,
        variant_name=args.variant,
    )

    print("\n" + md_report)

    if out_md_path:
        out_p = Path(out_md_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_report, encoding="utf-8")
        print(f"\n✓ Markdown report written to {out_p}")

    out_csv_path = args.output_csv
    if not out_csv_path and args.plot and not args.fp32 and not args.input_dir:
        out_csv_path = str(input_path / "summary_metrics.csv")

    if out_csv_path and not summary_df.empty:
        out_c = Path(out_csv_path)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_c, index=False)
        print(f"✓ Summary CSV metrics written to {out_c}")

    if args.output_detailed_csv and not detailed_df.empty:
        out_det = Path(args.output_detailed_csv)
        out_det.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(out_det, index=False)
        print(f"✓ Detailed per-K CSV written to {out_det}")


if __name__ == "__main__":
    main()
