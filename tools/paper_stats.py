#!/usr/bin/env python3
"""
tools/paper_stats.py

Statistical analysis and metrology utilities for GEMMBench paper-grade results:
- Hierarchical two-level bootstrap for ratio of medians
- Panel working set memory footprint calculation
- Raw repetition samples loader and FLOP/cycle extraction
- Geometric mean and parity classification
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def working_set_bytes(M: int, N: int, K: int, elem_bytes: int) -> int:
    """
    Computes the memory footprint in bytes of the active panels A and B for a micro-tile.
    Panel A: M x K, Panel B: K x N.
    Total = K * (M + N) * elem_bytes.
    (Tile C is excluded as its size M x N is constant with respect to K).
    """
    return int(K * (M + N) * elem_bytes)


def geomean(x: Union[np.ndarray, List[float]]) -> float:
    """Computes the geometric mean of positive numbers."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0 or np.any(arr <= 0):
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def parse_filename_meta(filepath: Union[str, Path]) -> Dict[str, str]:
    """
    Extracts platform, layout, compiler, variant, and precision from benchmark CSV filenames.
    Pattern: gemm_results_<platform>_<layout>[_<variant>...][_<precision>]_<compiler>.csv
    Variants can include: default, nofp, coldc, strict
    """
    stem = Path(filepath).stem
    if stem.endswith(".raw"):
        stem = stem[:-4]

    # Clean prefix
    name = stem
    if name.startswith("gemm_results_"):
        name = name[len("gemm_results_"):]

    parts = name.split("_")
    compiler = parts[-1] if parts[-1] in ("gcc", "clang") else "unknown"

    precision = "fp64"
    if "fp32" in parts:
        precision = "fp32"
    elif "fp64" in parts:
        precision = "fp64"

    layout = "crr" if "crr" in parts else ("rrr" if "rrr" in parts else "unknown")

    # Detect variant flags
    variants = []
    if "strict" in parts:
        variants.append("strict")
    if "nofp" in parts or "no_fastpath" in parts:
        variants.append("nofp")
    if "coldc" in parts:
        variants.append("coldc")
    variant = "_".join(variants) if variants else "default"

    # Platform is what precedes layout
    layout_indices = [i for i, p in enumerate(parts) if p in ("crr", "rrr")]
    if layout_indices:
        platform = "_".join(parts[:layout_indices[0]])
    else:
        platform = "unknown"

    return {
        "Platform": platform,
        "Layout": layout,
        "Compiler": compiler,
        "Variant": variant,
        "Precision": precision,
    }


def load_raw(raw_csv: Path) -> pd.DataFrame:
    """
    Loads a .raw.csv file and adds Platform, Layout, Compiler, Variant, and Precision
    columns derived from the filename.
    """
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv}")

    df = pd.read_csv(raw_csv)
    meta = parse_filename_meta(raw_csv)
    for k, v in meta.items():
        df[k] = v
    df["SourceFile"] = raw_csv.name
    return df


def flop_per_cycle_samples(df_raw: pd.DataFrame, kernel: str, K: int) -> Dict[int, np.ndarray]:
    """
    Extracts per-run arrays of FLOP/cycle for each measurement repetition.
    FLOP = 2 * M * N * K.
    Uses Cycles counter directly (frequency-independent metric).
    Returns: {run_id: np.ndarray of FLOP/cycle samples}
    """
    sub = df_raw[(df_raw["Kernel"] == kernel) & (df_raw["K"] == K)]
    if sub.empty:
        return {}

    run_col = "RunId" if "RunId" in sub.columns else ("run_id" if "run_id" in sub.columns else None)
    if run_col is None:
        return {}

    results: Dict[int, np.ndarray] = {}
    for r_id, grp in sub.groupby(run_col):
        # Calculate FLOP/cycle per sample
        m_val = grp["M"].iloc[0]
        n_val = grp["N"].iloc[0]
        k_val = grp["K"].iloc[0]
        ops = 2.0 * m_val * n_val * k_val

        cyc = grp["Cycles"].to_numpy(dtype=np.float64)
        valid = cyc > 0.0
        if np.any(valid):
            fc = np.full_like(cyc, np.nan)
            fc[valid] = ops / cyc[valid]
            results[int(r_id)] = fc[valid]

    return results


def bootstrap_ratio_of_medians(
    a: Dict[int, np.ndarray],
    b: Dict[int, np.ndarray],
    n_boot: int = 10000,
    seed: int = 12345,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Hierarchical two-level bootstrap for the ratio of medians: med(a) / med(b).
    Two-stage sampling independently for a and b:
      1) Resample with replacement len(runs) runs.
      2) Within each selected run, resample with replacement its repetition samples.
      3) Concatenate and compute median.
    Returns: (point_estimate, ci_low, ci_high) based on percentiles.
    """
    if not a or not b:
        return float("nan"), float("nan"), float("nan")

    all_a = np.concatenate(list(a.values()))
    all_b = np.concatenate(list(b.values()))
    if all_a.size == 0 or all_b.size == 0:
        return float("nan"), float("nan"), float("nan")

    med_a = float(np.median(all_a))
    med_b = float(np.median(all_b))
    if med_b <= 0 or np.isnan(med_b) or np.isnan(med_a):
        return float("nan"), float("nan"), float("nan")

    point_estimate = med_a / med_b

    rng = np.random.default_rng(seed)

    runs_a = list(a.values())
    num_runs_a = len(runs_a)
    runs_b = list(b.values())
    num_runs_b = len(runs_b)

    # Check if runs in a have equal length and runs in b have equal length (vectorized path)
    lens_a = [len(r) for r in runs_a]
    lens_b = [len(r) for r in runs_b]

    equal_a = len(set(lens_a)) == 1 and lens_a[0] > 0
    equal_b = len(set(lens_b)) == 1 and lens_b[0] > 0

    if equal_a and equal_b:
        arr_a = np.stack(runs_a)  # shape (num_runs_a, rep_a)
        rep_a = lens_a[0]
        rand_runs_a = rng.integers(0, num_runs_a, size=(n_boot, num_runs_a))
        rand_reps_a = rng.integers(0, rep_a, size=(n_boot, num_runs_a, rep_a))
        # Gather samples
        sampled_a = arr_a[rand_runs_a[:, :, None], rand_reps_a].reshape(n_boot, num_runs_a * rep_a)
        boot_med_a = np.median(sampled_a, axis=1)

        arr_b = np.stack(runs_b)
        rep_b = lens_b[0]
        rand_runs_b = rng.integers(0, num_runs_b, size=(n_boot, num_runs_b))
        rand_reps_b = rng.integers(0, rep_b, size=(n_boot, num_runs_b, rep_b))
        sampled_b = arr_b[rand_runs_b[:, :, None], rand_reps_b].reshape(n_boot, num_runs_b * rep_b)
        boot_med_b = np.median(sampled_b, axis=1)
    else:
        # General unvectorized path for varying sample counts
        boot_med_a = np.empty(n_boot, dtype=np.float64)
        boot_med_b = np.empty(n_boot, dtype=np.float64)

        for i in range(n_boot):
            # Sample a
            chosen_runs_a = rng.choice(num_runs_a, size=num_runs_a, replace=True)
            samples_a_list = [rng.choice(runs_a[r], size=len(runs_a[r]), replace=True) for r in chosen_runs_a]
            boot_med_a[i] = np.median(np.concatenate(samples_a_list))

            # Sample b
            chosen_runs_b = rng.choice(num_runs_b, size=num_runs_b, replace=True)
            samples_b_list = [rng.choice(runs_b[r], size=len(runs_b[r]), replace=True) for r in chosen_runs_b]
            boot_med_b[i] = np.median(np.concatenate(samples_b_list))

    valid_mask = (boot_med_b > 0) & ~np.isnan(boot_med_a) & ~np.isnan(boot_med_b)
    if not np.any(valid_mask):
        return point_estimate, float("nan"), float("nan")

    ratios = boot_med_a[valid_mask] / boot_med_b[valid_mask]

    alpha = 1.0 - ci
    pct_low = 100.0 * (alpha / 2.0)
    pct_high = 100.0 * (1.0 - alpha / 2.0)

    ci_low = float(np.percentile(ratios, pct_low))
    ci_high = float(np.percentile(ratios, pct_high))

    return point_estimate, ci_low, ci_high


def parse_cache_size_bytes(size_str: Union[str, int]) -> int:
    """Parses a cache size string like '32K', '256K', '3072K', '32M' into integer bytes."""
    if isinstance(size_str, (int, float)):
        return int(size_str)
    s = str(size_str).strip().upper()
    if s.endswith("K") or s.endswith("KB") or s.endswith("KIB"):
        m = re.match(r"^([0-9]+)", s)
        return int(m.group(1)) * 1024 if m else 0
    if s.endswith("M") or s.endswith("MB") or s.endswith("MIB"):
        m = re.match(r"^([0-9]+)", s)
        return int(m.group(1)) * 1024 * 1024 if m else 0
    try:
        return int(s)
    except ValueError:
        return 0


def get_l1d_bytes(
    meta_json_path: Optional[Path] = None,
    uarch_spec: Optional[dict] = None,
) -> Optional[int]:
    """
    Resolves L1 Data cache size in bytes.
    First tries meta.json caches index.
    Fallback: uarch_spec['l1d_kib'] if specified.
    """
    if meta_json_path and meta_json_path.is_file():
        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            caches = data.get("caches", [])
            for c in caches:
                if c.get("level") == 1 and "Data" in c.get("type", ""):
                    size_bytes = parse_cache_size_bytes(c.get("size", 0))
                    if size_bytes > 0:
                        return size_bytes
        except Exception:
            pass

    if uarch_spec and uarch_spec.get("l1d_kib") is not None:
        kib = uarch_spec.get("l1d_kib")
        if kib:
            return int(kib) * 1024

    return None
