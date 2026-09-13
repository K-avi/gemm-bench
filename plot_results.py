#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MATRIX_SIZES = [32, 64, 128, 256, 512]

ARCHITECTURE_LABELS = {
    "Skylake": "skylake",
    "X100": "x100",
}

SCALAR_KERNELS = (
    "ijk",
    "ikj",
    "blocked_register_blocked",
    "blocked",
)

MIPPV2_PORTABLE_KERNELS = (
    "mippv2_dot",
)

SKYLAKE_REGISTER_KERNELS = (
    "mippv2_register_blocked",
    "mippv2_lmul_register_blocked",
    "mippv2_lmul2_register_blocked",
    "mippv2_lmul4_register_blocked",
)

X100_REGISTER_KERNELS = (
    "mippv2_x100_register_blocked_lmul1",
    "mippv2_x100_register_blocked_lmul2",
    "mippv2_x100_register_blocked_lmul4",
    "mippv2_x100_register_blocked_lmul8",
    "mippv2_x100_register_blocked_apack4",
)

SKYLAKE_PANEL_KERNELS = (
    "mippv2_panel",
    "mippv2_panel_lmul",
    "mippv2_panel_lmul2",
    "mippv2_panel_lmul4",
)

X100_PANEL_KERNELS = (
    "mippv2_panel_x100_lmul1",
    "mippv2_panel_x100_lmul2",
    "mippv2_panel_x100_lmul4",
    "mippv2_panel_x100_lmul8",
)

SKYLAKE_GENERIC_KERNELS = (
    "mippv2",
    "mippv2_lmul2",
    "mippv2_lmul4",
    "mippv2_lmul8",
    "mippv2_blocked",
    "mippv2_blocked_lmul2",
    "mippv2_blocked_lmul4",
    "mippv2_blocked_unroll2",
    "mippv2_blocked_unroll_jam4",
)

X100_GENERIC_KERNELS = (
    "mippv2_x100_lmul1",
    "mippv2_x100_lmul2",
    "mippv2_x100_lmul4",
    "mippv2_x100_lmul8",
)

KNOWN_KERNELS = tuple(
    dict.fromkeys(
        SCALAR_KERNELS
        + MIPPV2_PORTABLE_KERNELS
        + SKYLAKE_REGISTER_KERNELS
        + X100_REGISTER_KERNELS
        + SKYLAKE_PANEL_KERNELS
        + X100_PANEL_KERNELS
        + SKYLAKE_GENERIC_KERNELS
        + X100_GENERIC_KERNELS
    )
)

FAMILY_COLORS = {
    "Scalar": "black",
    "Generic": "tab:blue",
    "Register": "tab:red",
    "Panel": "tab:purple",
    "Blocked": "saddlebrown",
    "Dot": "tab:green",
}

LMUL_MARKERS = {
    "LMUL1": "o",
    "LMUL2": "s",
    "LMUL4": "^",
    "LMUL8": "D",
}

TARGET_LINESTYLES = {
    "Scalar": "solid",
    "Portable" : "dashdot",
    "Skylake-tuned": "dotted",
    "X100-tuned": "dashed",
}

plt.rcParams.update(
    {
        "figure.figsize": (12, 7),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
    }
)

from matplotlib import colors
import numpy as np


FAMILY_KERNEL_ORDER = {
    "Scalar": SCALAR_KERNELS,

    "Generic": (
        "mippv2",
        "mippv2_lmul2",
        "mippv2_lmul4",
        "mippv2_lmul8",
        "mippv2_blocked",
        "mippv2_blocked_lmul2",
        "mippv2_blocked_lmul4",
        "mippv2_blocked_unroll2",
        "mippv2_blocked_unroll_jam4",
        "mippv2_dot",
    ),

    "Register": (
        "mippv2_register_blocked",
        "mippv2_lmul_register_blocked",
        "mippv2_lmul2_register_blocked",
        "mippv2_lmul4_register_blocked",

        "mippv2_x100_register_blocked_lmul1",
        "mippv2_x100_register_blocked_lmul2",
        "mippv2_x100_register_blocked_lmul4",
        "mippv2_x100_register_blocked_lmul8",
        "mippv2_x100_register_blocked_apack4",
    ),

    "Panel": (
        "mippv2_panel",
        "mippv2_panel_lmul",
        "mippv2_panel_lmul2",
        "mippv2_panel_lmul4",

        "mippv2_panel_x100_lmul1",
        "mippv2_panel_x100_lmul2",
        "mippv2_panel_x100_lmul4",
        "mippv2_panel_x100_lmul8",
    ),

    "Blocked": (
        "mippv2_blocked",
        "mippv2_blocked_lmul2",
        "mippv2_blocked_lmul4",
        "mippv2_blocked_unroll2",
        "mippv2_blocked_unroll_jam4",
    ),

    "Dot": (
        "mippv2_dot",
    ),
}


def generate_family_colors(
    family: str,
    kernels: tuple[str, ...],
) -> dict[str, str]:
    base = np.array(colors.to_rgb(FAMILY_COLORS[family]))

    result = {}

    n = len(kernels)

    for i, kernel in enumerate(kernels):
        # Dark -> light
        # Keep the first variants visually stronger
        t = 0.15 + 0.7 * (i / max(1, n - 1))

        rgb = base * (1 - t) + np.ones(3) * t

        result[kernel] = colors.to_hex(rgb)

    return result


KERNEL_COLORS = {}

for family, kernels in FAMILY_KERNEL_ORDER.items():
    KERNEL_COLORS.update(
        generate_family_colors(
            family,
            kernels,
        )
    )


@dataclass(frozen=True)
class KernelMeta:
    kernel: str
    display_name: str
    family: str
    optimization_target: str
    lmul: str | None
    color: str
    marker: str
    linestyle: str


def dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def infer_lmul(kernel: str) -> str | None:
    if "lmul8" in kernel:
        return "LMUL8"
    if "lmul4" in kernel:
        return "LMUL4"
    if "lmul2" in kernel:
        return "LMUL2"
    if "lmul1" in kernel:
        return "LMUL1"
    if kernel.endswith("_lmul") or kernel == "mippv2_lmul_register_blocked":
        return "LMUL1"
    return None


def infer_family(kernel: str) -> str:
    if kernel in SCALAR_KERNELS:
        return "Scalar"
    if kernel == "blocked_register_blocked":
        return "Scalar"
    if "panel" in kernel:
        return "Panel"
    if "register_blocked" in kernel:
        return "Register"
    if kernel == "mippv2_dot":
        return "Dot"
    if "blocked" in kernel:
        return "Blocked"
    return "Generic"


def infer_optimization_target(kernel: str) -> str:
    if kernel in SCALAR_KERNELS:
        return "Scalar"
    if "x100" in kernel:
        return "X100-tuned"
    if "dot" in kernel:
        return "Portable"
    return "Skylake-tuned"


def infer_display_name(kernel: str) -> str:
    if kernel == "ijk":
        return "Scalar IJK"

    if kernel == "ikj":
        return "Scalar IKJ"

    if kernel == "blocked":
        return "Scalar blocked"

    if kernel == "blocked_register_blocked":
        return "Auto-vectorized blocked"

    prefix = "X100-tuned" if "x100" in kernel else "Skylake-tuned"
    family = infer_family(kernel)
    lmul = infer_lmul(kernel)

    if family == "Register":
        name = f"{prefix} MIPPv2 RB"

        if kernel.endswith("apack4"):
            return f"{name} A-panel pack"

        if lmul is not None:
            return f"{name} {lmul}"

        return name

    if family == "Panel":
        name = f"{prefix} MIPPv2 Panel"

        if lmul is not None:
            return f"{name} {lmul}"

        return name

    if family == "Blocked":
        name = f"{prefix} MIPPv2 blocked"

        if kernel == "mippv2_blocked":
            return name

        if kernel.endswith("unroll2"):
            return f"{name} unroll2"

        if kernel.endswith("unroll_jam4"):
            return f"{name} unroll jam4"

        if lmul is not None:
            return f"{name} {lmul}"

        return name

    if family == "Dot":
        return "MIPPv2 Dot"

    if family == "Generic":
        if "x100" in kernel:
            name = "MIPPv2 X100 baseline"
        else:
            name = "MIPPv2 Skylake baseline"

        if lmul is not None:
            return f"{name} {lmul}"

        return name

    return kernel.replace("_", " ")


def infer_marker(kernel: str) -> str:
    # A-panel packing is a different optimization axis than LMUL
    if kernel.endswith("apack4"):
        return "X"

    lmul = infer_lmul(kernel)
    if lmul is not None:
        return LMUL_MARKERS[lmul]

    return "o"

def infer_linestyle(kernel: str, optimization_target: str) -> str:
    if kernel.endswith("apack4"):
        return (0, (5, 5, 5, 5))  # custom dash pattern

    return TARGET_LINESTYLES[optimization_target]


# def infer_color(family: str) -> str:
#     return FAMILY_COLORS.get(family, "tab:blue")
def infer_color(kernel: str) -> str:
    family = infer_family(kernel)

    if kernel.endswith("apack4"):
        return "saddlebrown"

    if kernel in KERNEL_COLORS:
        return KERNEL_COLORS[kernel]

    return FAMILY_COLORS.get(family, "tab:blue")


def build_kernel_metadata(kernel: str) -> KernelMeta:
    family = infer_family(kernel)
    optimization_target = infer_optimization_target(kernel)
    return KernelMeta(
        kernel=kernel,
        display_name=infer_display_name(kernel),
        family=family,
        optimization_target=optimization_target,
        lmul=infer_lmul(kernel),
        color=infer_color(kernel),
        marker=infer_marker(kernel),
        linestyle=infer_linestyle(kernel,optimization_target),
    )


def add_kernel_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata = [build_kernel_metadata(kernel) for kernel in sorted(df["Kernel"].unique())]
    metadata_df = pd.DataFrame([meta.__dict__ for meta in metadata]).rename(
        columns={
            "kernel": "Kernel",
            "display_name": "DisplayName",
            "family": "Family",
            "optimization_target": "OptimizationTarget",
            "lmul": "LMUL",
            "color": "Color",
            "marker": "Marker",
            "linestyle": "LineStyle",
        }
    )
    return df.merge(metadata_df, on="Kernel", how="left")


def load_dataset(path: str, architecture: str, compiler: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    data = pd.read_csv(csv_path)
    data["Architecture"] = architecture
    data["Compiler"] = compiler
    return data


def load_all_datasets(args: argparse.Namespace) -> pd.DataFrame:
    datasets = [
        load_dataset(args.skylake_gcc, "Skylake", "GCC"),
        load_dataset(args.skylake_clang, "Skylake", "Clang"),
        load_dataset(args.x100_gcc, "X100", "GCC"),
        load_dataset(args.x100_clang, "X100", "Clang"),
    ]
    combined = pd.concat(datasets, ignore_index=True)
    return add_kernel_metadata(combined)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_axes(ax: plt.Axes, *, ylog: bool = False, ylabel: str | None = None) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(MATRIX_SIZES)
    ax.set_xticklabels([str(size) for size in MATRIX_SIZES])
    ax.grid(True, axis="y", which="both")
    ax.set_xlabel("Matrix size M=N=K")
    if ylog:
        ax.set_yscale("log")
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def filter_data(
    df: pd.DataFrame,
    *,
    architecture: str | None = None,
    compiler: str | None = None,
    kernels: Iterable[str] | None = None,
) -> pd.DataFrame:
    data = df
    if architecture is not None:
        data = data[data["Architecture"] == architecture]
    if compiler is not None:
        data = data[data["Compiler"] == compiler]
    if kernels is not None:
        data = data[data["Kernel"].isin(list(kernels))]
    return data.copy()


def aggregate_for_plot(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    aggregations: dict[str, str] = {column: "mean" for column in metric_cols}
    for column in [
        "Build",
        "Kernel",
        "DisplayName",
        "Family",
        "OptimizationTarget",
        "LMUL",
        "DisplayLabel",
        "Color",
        "Marker",
        "LineStyle",
        "Architecture",
        "Compiler",
    ]:
        if column in df.columns and column not in group_cols and column not in aggregations:
            aggregations[column] = "first"

    return df.groupby(group_cols, as_index=False).agg(aggregations)


# def select_best_versions_for_kernels(
#     df: pd.DataFrame,
#     *,
#     architecture: str,
#     kernels: Iterable[str],
#     metric: str = "GFLOPS",
# ) -> pd.DataFrame:
#     data = filter_data(df, architecture=architecture, kernels=kernels)
#     if data.empty:
#         return data.copy()

#     kernel_compiler_scores = data.groupby(["Kernel", "Compiler"], as_index=False)[metric].mean()
#     best_indices = kernel_compiler_scores.groupby("Kernel")[metric].idxmax()
#     best_compilers = kernel_compiler_scores.loc[best_indices, ["Kernel", "Compiler"]]

#     selected = data.merge(best_compilers, on=["Kernel", "Compiler"], how="inner")
#     if selected.empty:
#         return selected

#     selected["DisplayLabel"] = selected["DisplayName"] + " (" + selected["Compiler"] + ")"
#     return selected

def select_best_versions_for_kernels(
    df: pd.DataFrame,
    *,
    architecture: str,
    kernels: Iterable[str],
    metric: str = "GFLOPS",
) -> pd.DataFrame:
    data = filter_data(
        df,
        architecture=architecture,
        kernels=kernels,
    )

    if data.empty:
        return data.copy()

    # Pick the best compiler for each kernel based on peak throughput
    kernel_compiler_scores = (
        data
        .groupby(["Kernel", "Compiler"], as_index=False)[metric]
        .max()
    )


    best_indices = (
        kernel_compiler_scores
        .groupby("Kernel")[metric]
        .idxmax()
    )

    best_compilers = kernel_compiler_scores.loc[
        best_indices,
        ["Kernel", "Compiler"]
    ]

    # Keep ALL sizes/configurations from the winning compiler
    selected = data.merge(
        best_compilers,
        on=["Kernel", "Compiler"],
        how="inner",
    )

    if selected.empty:
        return selected

    # Remove duplicate measurements of the same configuration only
    config_cols = [
        "Kernel",
        "Compiler",
        "M",
        "N",
        "K",
    ]

    selected = (
        selected
        .sort_values(
            by=[metric, "Time_s"],
            ascending=[False, True],
        )
        .groupby(config_cols, as_index=False)
        .first()
    )

    selected["DisplayLabel"] = (
        selected["DisplayName"]
        + " ("
        + selected["Compiler"]
        + ")"
    )

    return selected

def plot_metric_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    ylog: bool = False,
    peak: float | None = None,
) -> None:
    if df.empty:
        ax.set_axis_off()
        ax.set_title(f"{title} (no data)")
        return

    setup_axes(ax, ylog=ylog, ylabel=ylabel)

    if peak is not None and metric == "GFLOPS":
        ax.axhline(
            y=peak,
            color="black",
            linestyle=":",
            label=f"Peak ({peak} GFLOP/s)",
            zorder=1,
        )

    for _, group in df.groupby("Kernel", sort=False):
        group = group.sort_values("M")
        meta = group.iloc[0]
        ax.plot(
            group["M"],
            group[metric],
            label=meta.get("DisplayLabel", meta["DisplayName"]),
            color=meta["Color"],
            marker=meta["Marker"],
            linestyle=meta["LineStyle"],
            linewidth=2,
            markersize=5,
        )

    ax.set_title(title)


def plot_metric_figure(
    df: pd.DataFrame,
    *,
    output: Path,
    title: str,
    metric: str,
    ylabel: str,
    architecture: str,
    kernels: Iterable[str],
    ylog: bool = False,
    compiler: str | None = None,
    peak: float | None = None,
) -> None:
    if compiler is None:
        data = select_best_versions_for_kernels(df, architecture=architecture, kernels=kernels, metric=metric)
    else:
        data = filter_data(df, architecture=architecture, compiler=compiler, kernels=kernels)
    data = aggregate_for_plot(data, ["Architecture", "Kernel", "M"], [metric])
    if data.empty:
        print(f"Skipping empty figure: {output}")
        return

    if metric == "Time_s":
        ylabel = "Runtime (ms)"
        data[metric] *= 1000.0

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_metric_lines(
        ax,
        data,
        metric=metric,
        title=title,
        ylabel=ylabel,
        ylog=ylog,
        peak=peak,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def compute_speedup(df: pd.DataFrame, baseline_kernel: str = "ijk") -> pd.DataFrame:
    baseline = df[df["Kernel"] == baseline_kernel][["Architecture", "M", "GFLOPS"]].rename(
        columns={"GFLOPS": "BaselineGFLOPS"}
    )
    merged = df.merge(baseline, on=["Architecture", "M"], how="left")
    merged = merged[merged["BaselineGFLOPS"].notna()].copy()
    merged["Speedup"] = merged["GFLOPS"] / merged["BaselineGFLOPS"]
    return merged


def plot_speedup_figure(
    df: pd.DataFrame,
    *,
    output: Path,
    title: str,
    architecture: str,
    kernels: Iterable[str],
    compiler: str | None = None,
    baseline_kernel: str = "ijk",
) -> None:
    if compiler is None:
        data = select_best_versions_for_kernels(df, architecture=architecture, kernels=kernels, metric="GFLOPS")
    else:
        data = filter_data(df, architecture=architecture, compiler=compiler, kernels=kernels)
    data = aggregate_for_plot(data, ["Architecture", "Kernel", "M"], ["GFLOPS"])
    data = compute_speedup(data, baseline_kernel=baseline_kernel)
    if data.empty:
        print(f"Skipping empty figure: {output}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    setup_axes(ax, ylabel="Speedup")
    for _, group in data.groupby("Kernel", sort=False):
        group = group.sort_values("M")
        meta = group.iloc[0]
        ax.plot(
            group["M"],
            group["Speedup"],
            label=meta.get("DisplayLabel", meta["DisplayName"]),
            color=meta["Color"],
            marker=meta["Marker"],
            linestyle=meta["LineStyle"],
            linewidth=2,
            markersize=5,
        )

    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def select_best_kernel(
    df: pd.DataFrame,
    *,
    architecture: str,
    kernels: Iterable[str],
    metric: str = "GFLOPS",
) -> str | None:
    data = select_best_versions_for_kernels(df, architecture=architecture, kernels=kernels, metric=metric)
    if data.empty:
        return None
    score = data.groupby("Kernel", as_index=False)[metric].mean().sort_values(metric, ascending=False)
    if score.empty:
        return None
    return str(score.iloc[0]["Kernel"])


# def choose_representative_kernels(df: pd.DataFrame, architecture: str) -> list[str]:
#     if architecture == "Skylake":
#         generic_candidates = list(SKYLAKE_GENERIC_KERNELS)
#         register_candidates = list(SKYLAKE_REGISTER_KERNELS)
#         panel_candidates = list(SKYLAKE_PANEL_KERNELS)
#     else:
#         generic_candidates = list(X100_GENERIC_KERNELS)
#         register_candidates = list(X100_REGISTER_KERNELS)
#         panel_candidates = list(X100_PANEL_KERNELS)

#     selected = ["ijk"]

#     for candidates in [
#         ["blocked_register_blocked"],
#         generic_candidates,
#         ["mippv2_blocked", "mippv2_blocked_lmul2", "mippv2_blocked_lmul4", "mippv2_blocked_unroll2", "mippv2_blocked_unroll_jam4"],
#         register_candidates,
#         panel_candidates,
#         ["mippv2_dot"],
#     ]:
#         best = select_best_kernel(df, architecture=architecture, kernels=candidates)
#         if best is not None:
#             selected.append(best)

#     return dedupe(selected)

SKYLAKE_PLOT_KERNELS = [
    "ijk",
    "blocked_register_blocked",
    "mippv2",
    "mippv2_dot",
    "mippv2_register_blocked",
    "mippv2_lmul4_register_blocked",
    "mippv2_panel",
    "mippv2_panel_lmul2",
    "mippv2_x100_register_blocked_apack4"
]

X100_PLOT_KERNELS = [
    "ijk",
    "blocked_register_blocked",
    "mippv2_x100_lmul4",
    "mippv2_dot",
    "mippv2_register_blocked",
    "mippv2_x100_register_blocked_lmul2",
    "mippv2_x100_register_blocked_apack4",
    
]


def choose_representative_kernels(df: pd.DataFrame, architecture: str) -> list[str]:
    """Return the kernels used in the.

    These kernels are selected manually to illustrate the successive
    optimization stages and the architecture-specific divergence between
    Skylake and X100. They are not chosen automatically based on
    performance.
    """
    kernels = (
        SKYLAKE_PLOT_KERNELS
        if architecture == "Skylake"
        else X100_PLOT_KERNELS
    )

    # Keep only kernels that are actually present in the dataset.
    available = set(
        df.loc[df["Architecture"] == architecture, "Kernel"].unique()
    )

    return [k for k in kernels if k in available]



def plot_architecture_overview(
    df: pd.DataFrame,
    output_dir: Path,
    architecture: str,
    peak: float | None = None,
) -> list[str]:
    selected = choose_representative_kernels(df, architecture)
    title_prefix = f"{architecture} architecture"
    plot_metric_figure(
        df,
        output=output_dir / "main_gflops.svg",
        title=f"{title_prefix}: GFLOP/s",
        metric="GFLOPS",
        ylabel="GFLOP/s",
        architecture=architecture,
        kernels=selected,
        peak=peak,
    )
    plot_metric_figure(
        df,
        output=output_dir / "main_runtime.svg",
        title=f"{title_prefix}: runtime",
        metric="Time_s",
        ylabel="Runtime (ms)",
        architecture=architecture,
        kernels=selected,
        ylog=True,
    )
    plot_speedup_figure(
        df,
        output=output_dir / "speedup_vs_scalar.svg",
        title=f"{title_prefix}: speedup vs scalar",
        architecture=architecture,
        kernels=selected,
        baseline_kernel="ijk",
    )
    plot_speedup_figure(
        df,
        output=output_dir / "speedup_vs_blocked_register_blocked.svg",
        title=f"{title_prefix}: speedup vs Auto-vectorized blocked",
        architecture=architecture,
        kernels=selected,
        baseline_kernel="blocked_register_blocked",
    )
    return selected


def plot_family_figure(
    df: pd.DataFrame,
    *,
    output: Path,
    title: str,
    architecture: str,
    kernels: Iterable[str],
    peak: float | None = None,
) -> None:
    plot_metric_figure(
        df,
        output=output,
        title=title,
        metric="GFLOPS",
        ylabel="GFLOP/s",
        architecture=architecture,
        kernels=kernels,
        peak=peak,
    )


def plot_register_and_panel_figures(
    df: pd.DataFrame,
    output_dir: Path,
    architecture: str,
    peak: float | None = None,
) -> None:
    if architecture == "Skylake":
        register_kernels = SKYLAKE_REGISTER_KERNELS
        panel_kernels = SKYLAKE_PANEL_KERNELS
    else:
        register_kernels = X100_REGISTER_KERNELS
        panel_kernels = X100_PANEL_KERNELS

    plot_family_figure(
        df,
        output=output_dir / "register.svg",
        title=f"{architecture} register family",
        architecture=architecture,
        kernels=register_kernels,
        peak=peak,
    )
    plot_family_figure(
        df,
        output=output_dir / "panel.svg",
        title=f"{architecture} panel family",
        architecture=architecture,
        kernels=panel_kernels,
        peak=peak,
    )


def plot_compiler_facets(
    df: pd.DataFrame,
    *,
    output: Path,
    title: str,
    architecture: str,
    kernels: Iterable[str],
    metric: str = "GFLOPS",
    ylog: bool = False,
    peak: float | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    plotted = False

    for ax, compiler in zip(axes, ["GCC", "Clang"]):
        data = filter_data(df, architecture=architecture, compiler=compiler, kernels=kernels)
        data = aggregate_for_plot(data, ["Architecture", "Compiler", "Kernel", "M"], [metric])
        if data.empty:
            ax.set_axis_off()
            ax.set_title(f"{compiler} (no data)")
            continue

        plotted = True
        setup_axes(ax, ylog=ylog, ylabel="GFLOP/s" if metric == "GFLOPS" else "Runtime (s)")

        if peak is not None and metric == "GFLOPS":
            ax.axhline(
                y=peak,
                color="black",
                linestyle=":",
                label=f"Peak ({peak} GFLOP/s)",
                zorder=1,
            )

        for _, group in data.groupby("Kernel", sort=False):
            group = group.sort_values("M")
            meta = group.iloc[0]
            ax.plot(
                group["M"],
                group[metric],
                label=meta["DisplayName"],
                color=meta["Color"],
                marker=meta["Marker"],
                linestyle=meta["LineStyle"],
                linewidth=2,
                markersize=5,
            )
        ax.set_title(compiler)

    if not plotted:
        plt.close(fig)
        print(f"Skipping empty figure: {output}")
        return

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, format="svg")
    plt.close(fig)


def plot_compiler_comparison_figures(
    df: pd.DataFrame,
    output_dir: Path,
    architecture: str,
    kernels: Iterable[str],
    peak: float | None = None,
) -> None:
    plot_compiler_facets(
        df,
        output=output_dir / f"compiler_{ARCHITECTURE_LABELS[architecture]}.svg",
        title=f"{architecture}: GCC vs Clang",
        architecture=architecture,
        kernels=kernels,
        metric="GFLOPS",
        ylog=False,
        peak=peak,
    )


def compute_compiler_ratio(df: pd.DataFrame, architecture: str, kernels: Iterable[str]) -> pd.DataFrame:
    subset = filter_data(df, architecture=architecture, kernels=kernels)
    subset = aggregate_for_plot(subset, ["Architecture", "Compiler", "Kernel", "M"], ["GFLOPS"])
    if subset.empty:
        return subset

    pivot = subset.pivot_table(
        index=["Kernel", "M"],
        columns="Compiler",
        values="GFLOPS",
        aggfunc="mean",
    ).reset_index()
    if "Clang" not in pivot.columns or "GCC" not in pivot.columns:
        return pd.DataFrame()
    pivot["CompilerRatio"] = pivot["Clang"] / pivot["GCC"]

    meta = subset[["Kernel", "DisplayName", "Family", "OptimizationTarget", "Color", "Marker", "LineStyle"]].drop_duplicates(
        subset=["Kernel"]
    )
    return pivot.merge(meta, on="Kernel", how="left")


def plot_compiler_ratio_figure(
    df: pd.DataFrame,
    *,
    output: Path,
    title: str,
    architectures: Iterable[str],
    kernels: dict[str, Iterable[str]],
) -> None:
    architectures = list(architectures)
    fig, axes = plt.subplots(1, len(architectures), figsize=(7 * len(architectures), 6), sharex=True, sharey=True)
    if len(architectures) == 1:
        axes = [axes]

    plotted = False
    for ax, architecture in zip(axes, architectures):
        data = compute_compiler_ratio(df, architecture, kernels[architecture])
        if data.empty:
            ax.set_axis_off()
            ax.set_title(f"{architecture} (no data)")
            continue

        plotted = True
        setup_axes(ax, ylabel="Clang / GCC")
        for _, group in data.groupby("Kernel", sort=False):
            group = group.sort_values("M")
            meta = group.iloc[0]
            ax.plot(
                group["M"],
                group["CompilerRatio"],
                label=meta["DisplayName"],
                color=meta["Color"],
                marker=meta["Marker"],
                linestyle=meta["LineStyle"],
                linewidth=2,
                markersize=5,
            )
        ax.axhline(1.0, color="gray", linewidth=1, linestyle="--")
        ax.set_title(architecture)

    if not plotted:
        plt.close(fig)
        print(f"Skipping empty figure: {output}")
        return

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, format="svg")
    plt.close(fig)


def select_best_kernel_per_architecture(df: pd.DataFrame, architecture: str) -> str | None:
    data = select_best_versions_for_kernels(df, architecture=architecture, kernels=df[df["Architecture"] == architecture]["Kernel"].unique())
    data = aggregate_for_plot(data, ["Architecture", "Kernel", "M"], ["GFLOPS"])
    if data.empty:
        return None
    scores = data.groupby("Kernel", as_index=False)["GFLOPS"].mean().sort_values("GFLOPS", ascending=False)
    if scores.empty:
        return None
    return str(scores.iloc[0]["Kernel"])


def plot_register_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    kernels = list(SKYLAKE_REGISTER_KERNELS) + list(X100_REGISTER_KERNELS)
    data = filter_data(df, kernels=kernels)
    data = aggregate_for_plot(data, ["Architecture", "Kernel", "M"], ["GFLOPS"])
    data = compute_speedup(data)
    if data.empty:
        print(f"Skipping empty figure: {output_dir / 'register_comparison.svg'}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    setup_axes(ax, ylabel="Speedup")
    for _, group in data.groupby("Kernel", sort=False):
        group = group.sort_values("M")
        meta = group.iloc[0]
        ax.plot(
            group["M"],
            group["Speedup"],
            label=meta["DisplayName"],
            color=meta["Color"],
            marker=meta["Marker"],
            linestyle=meta["LineStyle"],
            linewidth=2,
            markersize=5,
        )
    ax.set_title("Register comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "register_comparison.svg", format="svg")
    plt.close(fig)


def plot_panel_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    kernels = list(SKYLAKE_PANEL_KERNELS) + list(X100_PANEL_KERNELS)
    data = filter_data(df, kernels=kernels)
    data = aggregate_for_plot(data, ["Architecture", "Kernel", "M"], ["GFLOPS"])
    data = compute_speedup(data)
    if data.empty:
        print(f"Skipping empty figure: {output_dir / 'panel_comparison.svg'}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    setup_axes(ax, ylabel="Speedup")
    for _, group in data.groupby("Kernel", sort=False):
        group = group.sort_values("M")
        meta = group.iloc[0]
        ax.plot(
            group["M"],
            group["Speedup"],
            label=meta["DisplayName"],
            color=meta["Color"],
            marker=meta["Marker"],
            linestyle=meta["LineStyle"],
            linewidth=2,
            markersize=5,
        )
    ax.set_title("Panel comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "panel_comparison.svg", format="svg")
    plt.close(fig)


def plot_best_kernel_summary(
    df: pd.DataFrame,
    output_dir: Path,
    best_kernels: dict[str, str | None],
) -> None:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    for ax, architecture in zip(axes, ["Skylake", "X100"]):

        best_kernel = best_kernels.get(architecture)

        kernels = ["ijk"]
        if best_kernel is not None:
            kernels.append(best_kernel)

        data = df[
            (df["Architecture"] == architecture)
            & (df["Kernel"].isin(kernels))
        ].copy()

        data = aggregate_for_plot(
            data,
            ["Architecture", "Kernel", "M"],
            ["GFLOPS"],
        )

        data = compute_speedup(data)

        if data.empty:
            ax.set_axis_off()
            ax.set_title(f"{architecture} (no data)")
            continue

        setup_axes(
            ax,
            ylabel="Speedup",
        )

        for _, group in data.groupby("Kernel", sort=False):

            group = group.sort_values("M")
            meta = group.iloc[0]

            ax.plot(
                group["M"],
                group["Speedup"],
                label=meta.get(
                    "DisplayLabel",
                    meta["DisplayName"],
                ),
                color=meta["Color"],
                marker=meta["Marker"],
                linestyle=meta["LineStyle"],
                linewidth=2,
                markersize=5,
            )

        ax.set_title(architecture)

        ax.legend(fontsize=8)

    # Force labels on both axes after all matplotlib layout decisions
    for ax in axes:
        ax.yaxis.set_tick_params(
            labelleft=True
        )

        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            labelleft=True,
        )

    fig.suptitle(
        "Best-kernel summary"
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    fig.savefig(
        output_dir / "best_kernel_summary.svg",
        format="svg",
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GEMM benchmark results across architectures and compilers.")
    parser.add_argument("--skylake-gcc", required=True, help="Skylake GCC CSV")
    parser.add_argument("--skylake-clang", required=True, help="Skylake Clang CSV")
    parser.add_argument("--x100-gcc", required=True, help="X100 GCC CSV")
    parser.add_argument("--x100-clang", required=True, help="X100 Clang CSV")
    parser.add_argument("--output", default="plots", help="Output directory")
    parser.add_argument("--skylake-peak", type=float, help="Theoretical peak GFLOP/s for Skylake")
    parser.add_argument("--x100-peak", type=float, help="Theoretical peak GFLOP/s for X100")
    args = parser.parse_args()

    df = add_kernel_metadata(
        pd.concat(
            [
                load_dataset(args.skylake_gcc, "Skylake", "GCC"),
                load_dataset(args.skylake_clang, "Skylake", "Clang"),
                load_dataset(args.x100_gcc, "X100", "GCC"),
                load_dataset(args.x100_clang, "X100", "Clang"),
            ],
            ignore_index=True,
        )
    )


    df_full = df.copy()

    # Remove explicit ISA-targeted builds from the main analysis
    excluded_targets = {
        "sse",
        "avx",
    }
    df = df[~df["Build"].isin(excluded_targets)].copy()
    # df = add_kernel_metadata(df)


    output_root = Path(args.output)
    skylake_dir = output_root / "skylake"
    x100_dir = output_root / "x100"
    comparison_dir = output_root / "comparison"
    for directory in [output_root, skylake_dir, x100_dir, comparison_dir]:
        ensure_directory(directory)

    skylake_selected = plot_architecture_overview(df, skylake_dir, "Skylake", peak=args.skylake_peak)
    x100_selected = plot_architecture_overview(df, x100_dir, "X100", peak=args.x100_peak)

    plot_register_and_panel_figures(df, skylake_dir, "Skylake", peak=args.skylake_peak)
    plot_register_and_panel_figures(df, x100_dir, "X100", peak=args.x100_peak)

    plot_compiler_comparison_figures(df, skylake_dir, "Skylake", skylake_selected, peak=args.skylake_peak)
    plot_compiler_comparison_figures(df, x100_dir, "X100", x100_selected, peak=args.x100_peak)

    plot_register_comparison(df, comparison_dir)
    plot_panel_comparison(df, comparison_dir)

    best_kernels = {
        "Skylake": select_best_kernel_per_architecture(df, "Skylake"),
        "X100": select_best_kernel_per_architecture(df, "X100"),
    }
    plot_best_kernel_summary(df, comparison_dir, best_kernels)

    plot_compiler_ratio_figure(
        df,
        output=comparison_dir / "compiler_sensitivity.svg",
        title="Compiler sensitivity",
        architectures=["Skylake", "X100"],
        kernels={"Skylake": skylake_selected, "X100": x100_selected},
    )


if __name__ == "__main__":
    main()