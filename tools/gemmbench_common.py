#!/usr/bin/env python3
"""
tools/gemmbench_common.py

Shared utilities, visualization palettes, theme configuration, and platform metadata
for GEMMBench analysis and visualization tools.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Optional, Tuple, Union

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

SIMD_FAMILY_COLORS: Dict[str, str] = {
    "avx512": "#dc2626",   # Crimson Red
    "avx2": "#0284c7",     # Sky Blue
    "neon": "#7c3aed",     # Purple / Violet
    "rvv": "#059669",      # Emerald Green
}

UARCH_COLORS: Dict[str, str] = {
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

# Microarchitectural hardware specifications and ceilings
PLATFORM_META: Dict[str, Dict[str, Union[str, float]]] = {
    "zen4_avx512": {"name": "AMD Zen 4", "simd": "avx512", "isa": "AVX-512", "peak_fc": 16.0, "uarch_id": "zen4"},
    "zen5_avx512": {"name": "AMD Zen 5", "simd": "avx512", "isa": "AVX-512", "peak_fc": 16.0, "uarch_id": "zen5"},
    "meteorlake_avx2": {"name": "Intel Meteor Lake", "simd": "avx2", "isa": "AVX2", "peak_fc": 16.0, "uarch_id": "meteorlake"},
    "skylake_avx2": {"name": "Intel Skylake", "simd": "avx2", "isa": "AVX2", "peak_fc": 16.0, "uarch_id": "skylake"},
    "m1_neon": {"name": "Apple M1", "simd": "neon", "isa": "NEON", "peak_fc": 16.0, "uarch_id": "m1"},
    "rpi5_neon": {"name": "Raspberry Pi 5", "simd": "neon", "isa": "NEON", "peak_fc": 8.0, "uarch_id": "rpi5"},
    "x100_rvv": {"name": "SpacemiT X100", "simd": "rvv", "isa": "RVV 1.0 (256b)", "peak_fc": 8.0, "uarch_id": "x100"},
    "x60_rvv": {"name": "SpacemiT X60", "simd": "rvv", "isa": "RVV 1.0 (256b)", "peak_fc": 8.0, "uarch_id": "x60"},
    "a100_rvv": {"name": "SpacemiT A100", "simd": "rvv", "isa": "RVV 1.0 (1024b)", "peak_fc": 8.0, "uarch_id": "a100"},
}

# Empirical hardware cache residency boundary (elements in K panel dimension)
# K <= 128 corresponds to <= 48 KiB working set (L1d cache resident compute bound)
# K >= 256 spills into L2 cache (memory latency / bandwidth bound)
L1_CACHE_BOUNDARY_K = 128


def setup_matplotlib_theme() -> None:
    """Configures global Matplotlib parameters for publication-grade rendering."""
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": LIGHT_BG,
        "figure.edgecolor": LIGHT_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": BORDER_COLOR,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.labelsize": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.85,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": BORDER_COLOR,
        "legend.fontsize": 9.0,
        "text.color": TEXT_PRIMARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial", "sans-serif"],
        "svg.fonttype": "none",
    })


def parse_benchmark_filename(filepath: Union[str, Path]) -> Tuple[str, str, str]:
    """
    Extracts canonical (platform, format, compiler) from benchmark CSV filename.
    e.g. 'gemm_results_zen4_avx512_crr_clang.csv' -> ('zen4_avx512', 'crr', 'clang')
    """
    stem = Path(filepath).stem
    m = re.match(r"gemm_results_(.+)_(crr|rrr)_(gcc|clang)$", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    parts = stem.split("_")
    compiler = parts[-1] if parts[-1] in ("gcc", "clang") else "unknown"
    layout = parts[-2] if parts[-2] in ("crr", "rrr") else "unknown"
    platform = "_".join(parts[2:-2]) if len(parts) > 4 else "unknown"
    return platform, layout, compiler
