from __future__ import annotations

"""Create quantitative-analysis graphs for one dataset or all supported datasets.

Normal mode processes all eligible full images. Passing ``--example`` selects
one low-cell full image per experimental condition. It consumes:

* the selected mask/image manifest;
* PCA outputs from ``align_cells_pca.py``;
* normalized-cell outputs from ``normalize_aligned_cells.py``.

Example mode is descriptive only. Full mode creates cell-level and ROI-level
measurement tables and uses ROI-level summaries for condition plots. Inferential
testing remains a separate stage. HADA time-course figures and WGA-DAPI
channel-comparison figures use the thesis plotting layouts.
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import ndimage as ndi

SCRIPT_FILE = Path(__file__).resolve()
EPS = np.finfo(np.float64).eps
EXAMPLE_MODE = False


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    channel_names: tuple[str, ...]
    condition_order: tuple[str, ...]


SPECS = {
    "2d_time": DatasetSpec(
        dataset="2d_time",
        channel_names=("HADA",),
        condition_order=(
            "THY_5min", "NHS_5min",
            "THY_20min", "NHS_20min",
            "THY_40min", "NHS_40min",
            "THY_120min", "NHS_120min",
        ),
    ),
    "2d_wga_dapi": DatasetSpec(
        dataset="2d_wga_dapi",
        # The Methods chapter defines channel 0 as DAPI and channel 1 as WGA.
        channel_names=("DAPI", "WGA"),
        condition_order=("THY_noCSP", "THY_CSP", "NHS_noCSP", "NHS_CSP"),
    ),
    "3d_mip": DatasetSpec(
        dataset="3d_mip",
        # The Methods chapter defines the stored channels as HADA, NADA and TADA.
        channel_names=("HADA", "NADA", "TADA"),
        condition_order=("THY", "NHS"),
    ),
}


def find_project_root(start: Path | None = None) -> Path:
    current = (start or SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT project root; provide --project-root.")


def configure_local_imports(project_root: Path) -> None:
    statistics_dir = project_root / "scripts" / "statistics"
    if str(statistics_dir) not in sys.path:
        sys.path.insert(0, str(statistics_dir))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_condition(dataset: str, sample_name: str) -> tuple[str, dict[str, object]]:
    upper = sample_name.upper()
    medium = "THY" if "THY" in upper else "NHS" if "NHS" in upper else "UNKNOWN"
    metadata: dict[str, object] = {"medium": medium}

    if dataset == "2d_time":
        match = re.search(r"(?<!\d)(5|20|40|120)\s*MIN", upper)
        if not match:
            match = re.search(r"_(5|20|40|120)MIN_", upper)
        time_min = int(match.group(1)) if match else -1
        metadata["time_min"] = time_min
        return f"{medium}_{time_min}min", metadata

    if dataset == "2d_wga_dapi":
        no_csp = any(token in upper for token in ("NOCSP", "NO_CSP", "WITHOUTCSP", "WITHOUT_CSP"))
        csp = ("CSP" in upper) and not no_csp
        csp_label = "CSP" if csp else "noCSP"
        metadata["csp"] = csp
        return f"{medium}_{csp_label}", metadata

    if dataset == "3d_mip":
        return medium, metadata

    raise ValueError(f"Unsupported dataset: {dataset}")


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def profile_minmax(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    finite = profile[np.isfinite(profile)]
    if finite.size == 0:
        return np.zeros_like(profile, dtype=np.float64)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(profile, dtype=np.float64)
    output = (profile - low) / (high - low)
    output[~np.isfinite(output)] = 0.0
    return np.clip(output, 0.0, 1.0)


def axial_profile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average intensity across cell width at each normalized axial position."""
    width = mask.shape[1]
    result = np.full(width, np.nan, dtype=np.float64)
    for x in range(width):
        valid = mask[:, x]
        if np.any(valid):
            values = image[:, x][valid]
            finite = values[np.isfinite(values)]
            if finite.size:
                result[x] = float(np.mean(finite))
    # Interpolate occasional empty columns created by transformed masks.
    valid_index = np.flatnonzero(np.isfinite(result))
    if valid_index.size == 0:
        return np.zeros(width, dtype=np.float64)
    if valid_index.size == 1:
        return np.full(width, result[valid_index[0]], dtype=np.float64)
    missing = ~np.isfinite(result)
    result[missing] = np.interp(np.flatnonzero(missing), valid_index, result[valid_index])
    return result


def radial_profile(image: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
    """Compute centre-to-boundary profile using the Methods radial coordinate."""
    distance = ndi.distance_transform_edt(mask)
    dmax = float(np.max(distance))
    output = np.full(bins, np.nan, dtype=np.float64)
    if dmax <= 0:
        return np.zeros(bins, dtype=np.float64)
    radial = 1.0 - distance / (dmax + EPS)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            selected = mask & (radial >= edges[index]) & (radial <= edges[index + 1])
        else:
            selected = mask & (radial >= edges[index]) & (radial < edges[index + 1])
        values = image[selected]
        finite = values[np.isfinite(values)]
        if finite.size:
            output[index] = float(np.mean(finite))
    valid_index = np.flatnonzero(np.isfinite(output))
    if valid_index.size == 0:
        return np.zeros(bins, dtype=np.float64)
    if valid_index.size == 1:
        return np.full(bins, output[valid_index[0]], dtype=np.float64)
    missing = ~np.isfinite(output)
    output[missing] = np.interp(np.flatnonzero(missing), valid_index, output[valid_index])
    return output


def pca_dimensions(binary: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(binary)
    if xx.size < 2:
        return 0.0, 0.0
    points = np.column_stack((xx.astype(np.float64), yy.astype(np.float64)))
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major = vectors[:, order[0]]
    minor = vectors[:, order[1]]
    major_projection = centred @ major
    minor_projection = centred @ minor
    length = float(np.max(major_projection) - np.min(major_projection) + 1.0)
    width = float(np.max(minor_projection) - np.min(minor_projection) + 1.0)
    return length, width


def background_parameters(image_cyx: np.ndarray, full_mask: np.ndarray) -> list[dict[str, float]]:
    background = full_mask == 0
    params: list[dict[str, float]] = []
    for channel in image_cyx:
        values = channel[background]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            median = mad = threshold = 0.0
        else:
            median = float(np.median(finite))
            mad = float(np.median(np.abs(finite - median)))
            threshold = median + 3.0 * mad
        params.append({"median": median, "mad": mad, "threshold": threshold})
    return params


def boundary_homogeneity(
    raw_channel: np.ndarray,
    corrected_channel: np.ndarray,
    cell_mask: np.ndarray,
    threshold: float,
) -> float:
    boundary = cell_mask & ~ndi.binary_erosion(cell_mask, iterations=1, border_value=0)
    if not np.any(boundary):
        return math.nan
    raw_values = raw_channel[boundary]
    corrected_values = corrected_channel[boundary]
    finite = np.isfinite(raw_values) & np.isfinite(corrected_values)
    if not np.any(finite):
        return math.nan
    raw_values = raw_values[finite]
    corrected_values = corrected_values[finite]
    coverage = float(np.mean(raw_values > threshold))
    mean = float(np.mean(corrected_values))
    std = float(np.std(corrected_values, ddof=1)) if corrected_values.size > 1 else 0.0
    cv = std / (mean + EPS)
    return coverage / (1.0 + cv)


def dapi_homogeneity(
    raw_channel: np.ndarray,
    corrected_channel: np.ndarray,
    cell_mask: np.ndarray,
    threshold: float,
) -> float:
    raw_values = raw_channel[cell_mask]
    corrected_values = corrected_channel[cell_mask]
    finite = np.isfinite(raw_values) & np.isfinite(corrected_values)
    if not np.any(finite):
        return math.nan
    raw_values = raw_values[finite]
    corrected_values = corrected_values[finite]
    area_fraction = float(np.mean(raw_values > threshold))
    mean = float(np.mean(corrected_values))
    std = float(np.std(corrected_values, ddof=1)) if corrected_values.size > 1 else 0.0
    cv = std / (mean + EPS)
    return area_fraction / (1.0 + cv)


def pearson_inside(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    finite = np.isfinite(av) & np.isfinite(bv)
    av = av[finite]
    bv = bv[finite]
    if av.size < 3 or np.std(av) <= EPS or np.std(bv) <= EPS:
        return math.nan
    return float(np.corrcoef(av, bv)[0, 1])


def condition_sort_key(spec: DatasetSpec, condition: str) -> tuple[int, str]:
    try:
        return spec.condition_order.index(condition), condition
    except ValueError:
        return len(spec.condition_order), condition


def condition_display(condition: str) -> str:
    labels = {
        "THY_noCSP": "THY",
        "THY_CSP": "THY + CSP",
        "NHS_noCSP": "NHS",
        "NHS_CSP": "NHS + CSP",
    }
    if condition in labels:
        return labels[condition]
    match = re.fullmatch(r"(THY|NHS)_(\d+)min", condition)
    if match:
        return f"{match.group(1)}, {match.group(2)} min"
    return condition.replace("_", " ")


def plot_2d_time_metric_over_time(
    records: list[dict[str, object]],
    key: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    available_times = sorted(
        {
            int(safe_float(row.get("time_min"), -1))
            for row in records
            if int(safe_float(row.get("time_min"), -1)) > 0
        }
    )
    if not available_times:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    rng = np.random.default_rng(42)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    plotted = False
    for medium_index, medium in enumerate(("THY", "NHS")):
        color = cycle[medium_index % len(cycle)]
        means: list[float] = []
        stds: list[float] = []
        times_used: list[int] = []
        for time in available_times:
            values = np.asarray(
                [
                    safe_float(row.get(key))
                    for row in records
                    if str(row.get("medium", "")) == medium
                    and int(safe_float(row.get("time_min"), -1)) == time
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            plotted = True
            times_used.append(time)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
            jitter_width = 0.55 if time <= 20 else 0.9
            jitter = rng.uniform(-jitter_width, jitter_width, size=values.size)
            ax.scatter(
                np.full(values.size, time) + jitter,
                values,
                s=14,
                alpha=0.45,
                color=color,
            )
        if times_used:
            ax.errorbar(
                times_used,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.5,
                capsize=3,
                color=color,
                label=medium,
            )
    if not plotted:
        plt.close(fig)
        return
    ax.set_xticks(available_times)
    ax.set_xlabel("Incubation time, min")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(title="Medium")
    add_note(fig)
    save_figure(fig, output_dir, stem)


def add_note(fig: plt.Figure) -> None:
    if not EXAMPLE_MODE:
        return
    fig.text(
        0.995,
        0.995,
        "Example mode: one image per condition",
        ha="right",
        va="top",
        fontsize=7,
        alpha=0.7,
    )


def jittered_scatter(ax, positions: Sequence[float], groups: Sequence[np.ndarray]) -> None:
    rng = np.random.default_rng(42)
    for position, values in zip(positions, groups):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            jitter = rng.uniform(-0.09, 0.09, size=values.size)
            ax.scatter(np.full(values.size, position) + jitter, values, s=11, alpha=0.55)


def boxplot_metric(
    records: list[dict[str, object]],
    conditions: list[str],
    key: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    groups = [np.asarray([safe_float(row.get(key)) for row in records if row["condition"] == cond]) for cond in conditions]
    if not any(np.any(np.isfinite(group)) for group in groups):
        return
    fig, ax = plt.subplots(figsize=(max(6.5, 1.25 * len(conditions)), 4.5), constrained_layout=True)
    clean = [group[np.isfinite(group)] for group in groups]
    ax.boxplot(clean, positions=np.arange(1, len(conditions) + 1), widths=0.55, showfliers=False)
    jittered_scatter(ax, np.arange(1, len(conditions) + 1), clean)
    ax.set_xticks(np.arange(1, len(conditions) + 1), [condition_display(condition).replace(" + ", " +\n") for condition in conditions])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    add_note(fig)
    save_figure(fig, output_dir, stem)


def mean_and_band(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    mean = np.nanmean(matrix, axis=0)
    if matrix.shape[0] <= 1:
        low = high = mean
    else:
        std = np.nanstd(matrix, axis=0, ddof=1)
        low = mean - std
        high = mean + std
    return mean, low, high


def plot_axial_profiles(
    profile_store: dict[tuple[str, str], list[np.ndarray]],
    conditions: list[str],
    channel_names: tuple[str, ...],
    output_dir: Path,
    dataset: str,
) -> None:
    for channel in channel_names:
        available = [condition for condition in conditions if profile_store.get((condition, channel))]
        if not available:
            continue

        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        if dataset == "2d_time":
            times = sorted(
                {
                    int(re.search(r"(\d+)min", condition).group(1))
                    for condition in available
                }
            )
            cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
                "color", [f"C{i}" for i in range(10)]
            )
            time_colors = {
                time: cycle[index % len(cycle)] for index, time in enumerate(times)
            }
            for time in times:
                for medium, linestyle in (("THY", "-"), ("NHS", "--")):
                    condition = f"{medium}_{time}min"
                    profiles = profile_store.get((condition, channel), [])
                    if not profiles:
                        continue
                    matrix = np.vstack(profiles)
                    mean = np.nanmean(matrix, axis=0)
                    x = np.linspace(0.0, 1.0, mean.size)
                    ax.plot(
                        x,
                        mean,
                        color=time_colors[time],
                        linestyle=linestyle,
                        linewidth=1.7,
                        label=f"{medium}, {time} min",
                    )
            ax.set_title(f"2D time-lapse {channel} axial boundary profile")
        else:
            cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
                "color", [f"C{i}" for i in range(10)]
            )
            for index, condition in enumerate(available):
                matrix = np.vstack(profile_store[(condition, channel)])
                mean, low, high = mean_and_band(matrix)
                x = np.linspace(0.0, 1.0, mean.size)
                color = cycle[index % len(cycle)]
                ax.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=1.7,
                    label=condition_display(condition),
                )
                ax.fill_between(x, low, high, color=color, alpha=0.10)
            ax.set_title(f"Axial {channel} fluorescence profiles")

        ax.axvline(0.5, linestyle="--", linewidth=0.8, color="0.45")
        ax.set_xticks([0.0, 0.5, 1.0], ["Pole 1", "Midcell", "Pole 2"])
        ax.set_xlabel("Normalized axial coordinate")
        ax.set_ylabel("Normalized fluorescence intensity")
        ax.grid(alpha=0.2)
        ax.legend()
        add_note(fig)
        save_figure(fig, output_dir, f"axial_profile_{channel.lower()}")


def plot_heatmaps(
    profile_store: dict[tuple[str, str], list[np.ndarray]],
    conditions: list[str],
    channel_names: tuple[str, ...],
    output_dir: Path,
    dataset: str,
) -> None:
    for channel in channel_names:
        if dataset == "2d_time":
            times = sorted(
                {
                    int(match.group(1))
                    for condition in conditions
                    for match in [re.search(r"(\d+)min", condition)]
                    if match is not None
                    and any(
                        profile_store.get(
                            (f"{medium}_{int(match.group(1))}min", channel)
                        )
                        for medium in ("THY", "NHS")
                    )
                }
            )
            if not times:
                continue
            ncols = 2 if len(times) > 1 else 1
            nrows = int(math.ceil(len(times) / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6.0 * ncols, 3.8 * nrows),
                sharex=True,
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes).reshape(-1)
            rendered = []
            for axis, time in zip(axes, times):
                profiles: list[np.ndarray] = []
                for medium in ("THY", "NHS"):
                    profiles.extend(
                        profile_store.get((f"{medium}_{time}min", channel), [])
                    )
                if not profiles:
                    axis.axis("off")
                    continue
                matrix = np.vstack(profiles)
                order = np.argsort(np.argmax(matrix, axis=1))
                matrix = matrix[order]
                image = axis.imshow(
                    matrix,
                    aspect="auto",
                    interpolation="nearest",
                    vmin=0.0,
                    vmax=1.0,
                )
                rendered.append(image)
                axis.set_title(f"{time} min")
                axis.set_xticks(
                    [0, (matrix.shape[1] - 1) / 2, matrix.shape[1] - 1],
                    ["0", "0.5", "1"],
                )
                axis.set_xlabel("Normalized axial coordinate")
                axis.set_ylabel("Individual cells")
            for axis in axes[len(times):]:
                axis.axis("off")
            if rendered:
                fig.colorbar(
                    rendered[0],
                    ax=list(axes[: len(times)]),
                    label="Within-cell normalized fluorescence",
                    shrink=0.88,
                )
            fig.suptitle(f"{channel} axial-profile heatmaps by time point")
            add_note(fig)
            save_figure(fig, output_dir, f"axial_heatmap_{channel.lower()}")
            continue

        blocks: list[np.ndarray] = []
        labels: list[tuple[str, int]] = []
        total = 0
        for condition in conditions:
            profiles = profile_store.get((condition, channel), [])
            if not profiles:
                continue
            block = np.vstack(profiles)
            blocks.append(block)
            total += block.shape[0]
            labels.append((condition, total))
        if not blocks:
            continue
        matrix = np.vstack(blocks)
        fig, ax = plt.subplots(
            figsize=(8.2, max(4.2, 0.055 * matrix.shape[0] + 2.2)),
            constrained_layout=True,
        )
        image = ax.imshow(
            matrix, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0
        )
        start = 0
        tick_positions = []
        tick_labels = []
        for condition, end in labels:
            tick_positions.append((start + end - 1) / 2.0)
            tick_labels.append(condition_display(condition))
            if end < matrix.shape[0]:
                ax.axhline(end - 0.5, linewidth=0.8, color="0.45")
            start = end
        ax.set_yticks(tick_positions, tick_labels)
        ax.set_xticks(
            [0, (matrix.shape[1] - 1) / 2, matrix.shape[1] - 1],
            ["0", "0.5", "1"],
        )
        ax.set_xlabel("Normalized axial coordinate")
        ax.set_ylabel("Condition and individual cells")
        ax.set_title(f"{channel} axial-profile population heatmap")
        fig.colorbar(image, ax=ax, label="Within-cell normalized fluorescence")
        add_note(fig)
        save_figure(fig, output_dir, f"axial_heatmap_{channel.lower()}")


def plot_radial_profiles(
    radial_store: dict[tuple[str, str], list[np.ndarray]],
    conditions: list[str],
    channel_names: tuple[str, ...],
    output_dir: Path,
    dataset: str,
) -> None:
    if not radial_store:
        return
    if dataset == "2d_wga_dapi":
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
            "color", [f"C{i}" for i in range(10)]
        )
        condition_handles: list[Line2D] = []
        channel_styles = {"WGA": "-", "DAPI": "--"}
        for index, condition in enumerate(conditions):
            color = cycle[index % len(cycle)]
            condition_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2.0,
                    label=condition_display(condition),
                )
            )
            for channel in ("WGA", "DAPI"):
                profiles = radial_store.get((condition, channel), [])
                if not profiles:
                    continue
                matrix = np.vstack(profiles)
                mean = np.nanmean(matrix, axis=0)
                x = np.linspace(0.0, 1.0, mean.size)
                ax.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=channel_styles[channel],
                    linewidth=1.8,
                )
        ax.set_xticks(
            [0.0, 0.5, 1.0],
            ["Cell centre", "Intermediate radius", "Cell boundary"],
        )
        ax.set_xlabel("Normalized radial distance")
        ax.set_ylabel("Normalized fluorescence intensity")
        ax.set_title("2D WGA-DAPI radial fluorescence profiles")
        ax.grid(alpha=0.2)
        condition_legend = ax.legend(
            handles=condition_handles, title="Condition", loc="center right"
        )
        ax.add_artist(condition_legend)
        channel_handles = [
            Line2D(
                [0], [0], color="black", linestyle="-", linewidth=1.8, label="WGA"
            ),
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=1.8, label="DAPI"
            ),
        ]
        ax.legend(handles=channel_handles, title="Channel", loc="lower center")
        add_note(fig)
        save_figure(fig, output_dir, "radial_profiles_wga_dapi")
        return

    if dataset == "3d_mip":
        pairs = (("HADA", "NADA"), ("HADA", "TADA"), ("NADA", "TADA"))
        for first, second in pairs:
            fig, axes = plt.subplots(
                1,
                len(conditions),
                figsize=(5 * len(conditions), 4.3),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            for axis, condition in zip(axes, conditions):
                for channel in (first, second):
                    profiles = radial_store.get((condition, channel), [])
                    if not profiles:
                        continue
                    matrix = np.vstack(profiles)
                    mean, low, high = mean_and_band(matrix)
                    x = np.linspace(0.0, 1.0, mean.size)
                    axis.plot(x, mean, label=channel)
                    axis.fill_between(x, low, high, alpha=0.12)
                axis.set_title(condition_display(condition))
                axis.set_xlabel("Normalized radial coordinate")
                axis.grid(alpha=0.2)
                axis.legend()
            axes[0].set_ylabel("Normalized fluorescence")
            fig.suptitle(f"Radial profiles: {first} versus {second}")
            add_note(fig)
            save_figure(fig, output_dir, f"radial_{first.lower()}_vs_{second.lower()}")


def plot_standardized_maps(
    map_store: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    conditions: list[str],
    channel_names: tuple[str, ...],
    output_dir: Path,
) -> None:
    for channel in channel_names:
        available = [condition for condition in conditions if (condition, channel) in map_store]
        if not available:
            continue
        ncols = 2 if len(available) > 1 else 1
        nrows = int(math.ceil(len(available) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.2 * nrows), constrained_layout=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for axis, condition in zip(axes, available):
            weighted_sum, weight = map_store[(condition, channel)]
            average = weighted_sum / np.maximum(weight, EPS)
            axis.imshow(average, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.axvline((average.shape[1] - 1) / 2, linestyle="--", linewidth=0.8)
            axis.set_xticks([0, (average.shape[1] - 1) / 2, average.shape[1] - 1], ["0", "0.5", "1"])
            axis.set_yticks([])
            axis.set_title(condition_display(condition))
        for axis in axes[len(available):]:
            axis.axis("off")
        fig.suptitle(f"standardized-cell maps: {channel}")
        add_note(fig)
        save_figure(fig, output_dir, f"standardized_map_{channel.lower()}")


def plot_wga_dapi_overlap_maps(
    map_store: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    conditions: list[str],
    output_dir: Path,
) -> None:
    available = [
        condition
        for condition in conditions
        if (condition, "WGA") in map_store and (condition, "DAPI") in map_store
    ]
    if not available:
        return
    ncols = 2 if len(available) > 1 else 1
    nrows = int(math.ceil(len(available) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 3.4 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, condition in zip(axes, available):
        wga_sum, wga_weight = map_store[(condition, "WGA")]
        dapi_sum, dapi_weight = map_store[(condition, "DAPI")]
        wga = np.zeros_like(wga_sum, dtype=np.float64)
        dapi = np.zeros_like(dapi_sum, dtype=np.float64)
        wga_valid = wga_weight > 0
        dapi_valid = dapi_weight > 0
        wga[wga_valid] = wga_sum[wga_valid] / np.maximum(
            wga_weight[wga_valid], EPS
        )
        dapi[dapi_valid] = dapi_sum[dapi_valid] / np.maximum(
            dapi_weight[dapi_valid], EPS
        )
        wga = np.clip(wga, 0.0, 1.0)
        dapi = np.clip(dapi, 0.0, 1.0)
        rgb = np.zeros((*wga.shape, 3), dtype=np.float64)
        rgb[..., 0] = dapi
        rgb[..., 1] = wga
        rgb[..., 2] = np.maximum(dapi, wga)
        axis.imshow(rgb, interpolation="nearest")
        axis.axvline(
            (wga.shape[1] - 1) / 2,
            linestyle="--",
            linewidth=0.8,
            color="yellow",
        )
        axis.set_xticks(
            [0, (wga.shape[1] - 1) / 2, wga.shape[1] - 1],
            ["0", "0.5", "1"],
        )
        axis.set_yticks([])
        axis.set_title(condition_display(condition))
    for axis in axes[len(available):]:
        axis.axis("off")
    fig.suptitle("Standardized WGA-DAPI spatial overlap maps")
    fig.text(
        0.5,
        0.01,
        "DAPI: magenta   WGA: cyan   spatial overlap: white",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    add_note(fig)
    save_figure(fig, output_dir, "wga_dapi_standardized_overlap_map")


def plot_detected_cells(
    dataset: str,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    parsed: list[dict[str, object]] = []
    for row in rows:
        if row.get("annotation_type", "full") != "full":
            continue
        if row.get("overall_status", "PASS") == "FAIL":
            continue
        sample = row.get("sample_name", "")
        condition, metadata = infer_condition(dataset, sample)
        parsed.append(
            {
                "sample": sample,
                "condition": condition,
                "medium": metadata.get("medium", "UNKNOWN"),
                "time_min": metadata.get("time_min", -1),
                "csp": bool(metadata.get("csp", False)),
                "n_cells": int(
                    float(
                        row.get("n_pca_eligible_instances")
                        or row.get("n_instances")
                        or 0
                    )
                ),
            }
        )
    if not parsed:
        return

    if dataset == "2d_time":
        fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
        rng = np.random.default_rng(42)
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
        all_times = sorted(
            {int(row["time_min"]) for row in parsed if int(row["time_min"]) > 0}
        )
        for medium_index, medium in enumerate(("THY", "NHS")):
            color = cycle[medium_index % len(cycle)]
            times_used: list[int] = []
            means: list[float] = []
            stds: list[float] = []
            for time in all_times:
                values = np.asarray(
                    [
                        row["n_cells"]
                        for row in parsed
                        if row["medium"] == medium
                        and int(row["time_min"]) == time
                    ],
                    dtype=float,
                )
                if values.size == 0:
                    continue
                times_used.append(time)
                means.append(float(np.mean(values)))
                stds.append(
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
                jitter = rng.uniform(-0.8, 0.8, values.size)
                ax.scatter(
                    np.full(values.size, time) + jitter,
                    values,
                    s=17,
                    alpha=0.45,
                    color=color,
                )
            if times_used:
                ax.errorbar(
                    times_used,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    color=color,
                    label=medium,
                )
        ax.set_xticks(all_times)
        ax.set_xlabel("Incubation time, min")
        ax.set_ylabel("Detected cells per image")
        ax.set_title("Detected cells over time")
        ax.grid(alpha=0.25)
        ax.legend(title="Medium")
        save_figure(fig, output_dir, "detected_cells_over_time_all_images")
        return

    if dataset == "2d_wga_dapi":
        fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
        rng = np.random.default_rng(42)
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
        x_positions = np.asarray([0.0, 1.0])
        for medium_index, medium in enumerate(("THY", "NHS")):
            color = cycle[medium_index % len(cycle)]
            offset = -0.07 if medium == "THY" else 0.07
            means: list[float] = []
            stds: list[float] = []
            valid_x: list[float] = []
            for csp_index, csp in enumerate((False, True)):
                values = np.asarray(
                    [
                        row["n_cells"]
                        for row in parsed
                        if row["medium"] == medium and bool(row["csp"]) == csp
                    ],
                    dtype=float,
                )
                if values.size == 0:
                    continue
                x = x_positions[csp_index] + offset
                valid_x.append(x)
                means.append(float(np.mean(values)))
                stds.append(
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
                jitter = rng.uniform(-0.035, 0.035, values.size)
                ax.scatter(
                    np.full(values.size, x) + jitter,
                    values,
                    s=18,
                    alpha=0.5,
                    color=color,
                )
            if valid_x:
                ax.errorbar(
                    valid_x,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    color=color,
                    label=medium,
                )
        ax.set_xticks(x_positions, ["No CSP", "CSP"])
        ax.set_ylabel("Detected cells per image")
        ax.set_title("Detected cells with and without CSP")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Medium")
        save_figure(fig, output_dir, "detected_cells_all_images")
        return

    spec = SPECS[dataset]
    conditions = [
        condition
        for condition in spec.condition_order
        if any(row["condition"] == condition for row in parsed)
    ]
    groups = [
        np.asarray(
            [row["n_cells"] for row in parsed if row["condition"] == condition],
            dtype=float,
        )
        for condition in conditions
    ]
    fig, ax = plt.subplots(
        figsize=(max(6.5, 1.35 * len(conditions)), 4.5), constrained_layout=True
    )
    ax.boxplot(
        groups,
        positions=np.arange(1, len(groups) + 1),
        widths=0.55,
        showfliers=False,
    )
    jittered_scatter(ax, np.arange(1, len(groups) + 1), groups)
    ax.set_xticks(
        np.arange(1, len(groups) + 1),
        [condition_display(condition) for condition in conditions],
    )
    ax.set_ylabel("Detected cells per image")
    ax.set_title("Detected cells by condition")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "detected_cells_all_images")


def _import_common():
    local_dir = SCRIPT_FILE.parent
    if str(local_dir) not in sys.path:
        sys.path.insert(0, str(local_dir))
    from statistics_pipeline_common import (
        DATASETS,
        find_project_root as common_find_project_root,
        graph_output_path,
        mask_summary_path,
        normalization_output_path,
        pca_output_path,
        selected_datasets,
        source_mode_for,
    )

    return {
        "DATASETS": DATASETS,
        "find_project_root": common_find_project_root,
        "graph_output_path": graph_output_path,
        "mask_summary_path": mask_summary_path,
        "normalization_output_path": normalization_output_path,
        "pca_output_path": pca_output_path,
        "selected_datasets": selected_datasets,
        "source_mode_for": source_mode_for,
    }


COMMON = _import_common()


def _prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output folder is not empty: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def aggregate_roi_records(cell_records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cell_records:
        grouped[str(row["annotation_id"])].append(row)

    excluded_numeric = {"source_label"}
    output: list[dict[str, object]] = []
    for annotation_id, rows in grouped.items():
        first = rows[0]
        summary: dict[str, object] = {
            "dataset": first.get("dataset", ""),
            "annotation_id": annotation_id,
            "sample_name": first.get("sample_name", ""),
            "condition": first.get("condition", ""),
            "medium": first.get("medium", ""),
            "time_min": first.get("time_min", ""),
            "csp": first.get("csp", ""),
            "n_cells": len(rows),
        }
        candidate_keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in candidate_keys:
                    candidate_keys.append(key)
        for key in candidate_keys:
            if key in excluded_numeric or key in summary:
                continue
            values = np.asarray([safe_float(row.get(key)) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                summary[key] = float(np.mean(finite))
        output.append(summary)
    output.sort(key=lambda row: (str(row.get("condition", "")), str(row.get("sample_name", ""))))
    return output


def _mean_profile_store(
    store: dict[tuple[str, str, str], list[np.ndarray]],
) -> dict[tuple[str, str], list[np.ndarray]]:
    output: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for (_annotation_id, condition, channel), profiles in store.items():
        if profiles:
            output[(condition, channel)].append(np.nanmean(np.vstack(profiles), axis=0))
    return output


def _condition_maps_from_annotations(
    annotation_maps: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    result: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (_annotation_id, condition, channel), (weighted_sum, weight) in annotation_maps.items():
        valid = weight > 0
        average = np.zeros_like(weighted_sum, dtype=np.float64)
        average[valid] = weighted_sum[valid] / np.maximum(weight[valid], EPS)
        key = (condition, channel)
        if key not in result:
            result[key] = (
                np.zeros_like(average, dtype=np.float64),
                np.zeros_like(weight, dtype=np.float64),
            )
        condition_sum, condition_weight = result[key]
        condition_sum[valid] += average[valid]
        condition_weight[valid] += 1.0
    return result


def process_dataset(
    *,
    project_root: Path,
    dataset: str,
    source_mode: str,
    example: bool,
    manifest: Path,
    pca_root: Path,
    normalized_root: Path,
    all_mask_summary: Path | None,
    output_root: Path,
    radial_bins: int,
    pixel_size_um: float | None,
    overwrite: bool,
) -> dict[str, object]:
    global EXAMPLE_MODE
    EXAMPLE_MODE = example
    _prepare_output(output_root, overwrite)
    configure_local_imports(project_root)
    from check_masks_for_statistics import load_image_cyx, load_label_mask

    if radial_bins < 3:
        raise ValueError("--radial-bins must be at least 3")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing PCA input manifest: {manifest}")
    if not normalized_root.exists():
        raise FileNotFoundError(f"Missing normalized-cell output: {normalized_root}")

    spec = SPECS[dataset]
    manifest_rows = read_csv(manifest)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest}")

    cell_records: list[dict[str, object]] = []
    cell_profile_store: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    cell_radial_store: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    annotation_profile_store: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    annotation_radial_store: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    annotation_map_store: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    profile_archive: dict[str, np.ndarray] = {}

    for row in manifest_rows:
        annotation_id = row["annotation_id"]
        sample_name = row["sample_name"]
        condition, sample_metadata = infer_condition(dataset, sample_name)
        mask_path = Path(row["mask_path"])
        image_path = Path(row["image_path"])
        full_mask, _mask_dtype = load_label_mask(mask_path)
        image_cyx, _axes = load_image_cyx(image_path, project_root)
        if image_cyx.shape[-2:] != full_mask.shape:
            raise ValueError(f"Image-mask mismatch for {annotation_id}")
        if image_cyx.shape[0] < len(spec.channel_names):
            raise ValueError(
                f"{annotation_id} contains {image_cyx.shape[0]} channels, but {dataset} "
                f"requires at least {len(spec.channel_names)}"
            )
        image_cyx = image_cyx[: len(spec.channel_names)].astype(np.float32, copy=False)
        bg = background_parameters(image_cyx, full_mask)

        annotation_normalized = normalized_root / Path(*annotation_id.split("/"))
        cell_files = sorted(
            annotation_normalized.glob("cells/label_*/normalized_cell_data.npz")
        )
        if not cell_files:
            raise FileNotFoundError(
                f"No normalized cells found for {annotation_id}: {annotation_normalized}"
            )

        for cell_file in cell_files:
            with np.load(cell_file) as data:
                source_label = int(data["source_label"])
                standardized_mask = np.asarray(data["mask_standardized"], dtype=bool)
                standardized_absolute = np.asarray(
                    data["image_standardized_absolute"], dtype=np.float32
                )
                standardized_normalized = np.asarray(
                    data["image_standardized_normalized"], dtype=np.float32
                )
            if source_label <= 0:
                continue
            cell_mask = full_mask == source_label
            if not np.any(cell_mask):
                continue
            length_px, width_px = pca_dimensions(cell_mask)
            area_px = int(np.count_nonzero(cell_mask))

            corrected_channels: list[np.ndarray] = []
            channel_metrics: dict[str, object] = {}
            for channel_index, channel_name in enumerate(spec.channel_names):
                raw = image_cyx[channel_index]
                corrected = np.maximum(raw - bg[channel_index]["median"], 0.0)
                corrected_channels.append(corrected)
                values = corrected[cell_mask]
                finite = values[np.isfinite(values)]
                channel_metrics[f"{channel_name.lower()}_mean_intensity"] = (
                    float(np.mean(finite)) if finite.size else math.nan
                )
                channel_metrics[f"{channel_name.lower()}_integrated_intensity"] = (
                    float(np.sum(finite)) if finite.size else math.nan
                )

                axial = profile_minmax(
                    axial_profile(standardized_absolute[channel_index], standardized_mask)
                )
                radial = profile_minmax(
                    radial_profile(
                        standardized_absolute[channel_index], standardized_mask, radial_bins
                    )
                )
                cell_profile_store[(condition, channel_name)].append(axial)
                cell_radial_store[(condition, channel_name)].append(radial)
                annotation_profile_store[(annotation_id, condition, channel_name)].append(axial)
                annotation_radial_store[(annotation_id, condition, channel_name)].append(radial)
                profile_archive[
                    f"{annotation_id}|{source_label}|{channel_name}|axial"
                ] = axial.astype(np.float32)
                profile_archive[
                    f"{annotation_id}|{source_label}|{channel_name}|radial"
                ] = radial.astype(np.float32)

                map_key = (annotation_id, condition, channel_name)
                if map_key not in annotation_map_store:
                    annotation_map_store[map_key] = (
                        np.zeros_like(
                            standardized_normalized[channel_index], dtype=np.float64
                        ),
                        np.zeros_like(standardized_mask, dtype=np.float64),
                    )
                weighted_sum, weight = annotation_map_store[map_key]
                weighted_sum += (
                    standardized_normalized[channel_index] * standardized_mask
                )
                weight += standardized_mask.astype(np.float64)

                if dataset in ("2d_time", "3d_mip"):
                    channel_metrics[f"{channel_name.lower()}_homogeneity"] = (
                        boundary_homogeneity(
                            raw,
                            corrected,
                            cell_mask,
                            bg[channel_index]["threshold"],
                        )
                    )

            if dataset == "2d_wga_dapi":
                dapi_raw = image_cyx[0]
                wga_raw = image_cyx[1]
                dapi_corrected = corrected_channels[0]
                wga_corrected = corrected_channels[1]
                channel_metrics["dapi_homogeneity"] = dapi_homogeneity(
                    dapi_raw, dapi_corrected, cell_mask, bg[0]["threshold"]
                )
                channel_metrics["wga_homogeneity"] = boundary_homogeneity(
                    wga_raw, wga_corrected, cell_mask, bg[1]["threshold"]
                )
                channel_metrics["wga_dapi_pearson"] = pearson_inside(
                    wga_corrected, dapi_corrected, cell_mask
                )

            record: dict[str, object] = {
                "dataset": dataset,
                "annotation_id": annotation_id,
                "sample_name": sample_name,
                "condition": condition,
                "source_label": source_label,
                "area_pixels": area_px,
                "length_pixels": length_px,
                "width_pixels": width_px,
                **sample_metadata,
                **channel_metrics,
            }
            if pixel_size_um is not None:
                record["pixel_size_um"] = pixel_size_um
                record["length_um"] = length_px * pixel_size_um
                record["width_um"] = width_px * pixel_size_um
                record["area_um2"] = area_px * pixel_size_um**2
            cell_records.append(record)

    if not cell_records:
        raise RuntimeError(f"No cell measurements were produced for {dataset}.")

    roi_records = aggregate_roi_records(cell_records)
    conditions = sorted(
        {str(row["condition"]) for row in cell_records},
        key=lambda condition: condition_sort_key(spec, condition),
    )
    roi_profile_store = _mean_profile_store(annotation_profile_store)
    roi_radial_store = _mean_profile_store(annotation_radial_store)
    condition_map_store = _condition_maps_from_annotations(annotation_map_store)

    write_csv(output_root / "cell_measurements.csv", cell_records)
    write_csv(output_root / "roi_summary.csv", roi_records)
    np.savez_compressed(output_root / "cell_profiles.npz", **profile_archive)

    roi_archive: dict[str, np.ndarray] = {}
    for (condition, channel), profiles in roi_profile_store.items():
        for index, profile in enumerate(profiles):
            roi_archive[f"{condition}|{channel}|roi_{index:04d}|axial"] = np.asarray(
                profile, dtype=np.float32
            )
    for (condition, channel), profiles in roi_radial_store.items():
        for index, profile in enumerate(profiles):
            roi_archive[f"{condition}|{channel}|roi_{index:04d}|radial"] = np.asarray(
                profile, dtype=np.float32
            )
    np.savez_compressed(output_root / "roi_profiles.npz", **roi_archive)

    count_rows = (
        read_csv(all_mask_summary)
        if all_mask_summary is not None and all_mask_summary.exists()
        else manifest_rows
    )
    plot_detected_cells(dataset, count_rows, output_root)

    plot_records = cell_records if example else roi_records
    profile_store_for_lines = cell_profile_store if example else roi_profile_store
    radial_store_for_lines = cell_radial_store if example else roi_radial_store
    length_key = "length_um" if pixel_size_um is not None else "length_pixels"
    if example:
        length_label = (
            "PCA-derived cell length, µm"
            if pixel_size_um is not None
            else "PCA-derived cell length, pixels"
        )
    else:
        length_label = (
            "ROI mean PCA-derived cell length, µm"
            if pixel_size_um is not None
            else "ROI mean PCA-derived cell length, pixels"
        )
    if dataset == "2d_time":
        plot_2d_time_metric_over_time(
            plot_records,
            length_key,
            length_label,
            "Cell length over time",
            output_root,
            "cell_length_over_time",
        )
    else:
        boxplot_metric(
            plot_records,
            conditions,
            length_key,
            length_label,
            "Cell length by condition",
            output_root,
            "cell_length",
        )

    prefix = "" if example else "ROI mean "
    if dataset == "2d_time":
        plot_2d_time_metric_over_time(
            plot_records,
            "hada_homogeneity",
            f"{prefix}HADA boundary homogeneity index",
            "HADA boundary homogeneity over time",
            output_root,
            "hada_boundary_homogeneity_over_time",
        )
    elif dataset == "2d_wga_dapi":
        boxplot_metric(
            plot_records,
            conditions,
            "wga_homogeneity",
            f"{prefix}WGA boundary homogeneity index",
            "WGA boundary homogeneity",
            output_root,
            "wga_homogeneity",
        )
        boxplot_metric(
            plot_records,
            conditions,
            "dapi_homogeneity",
            f"{prefix}DAPI homogeneity index",
            "DAPI homogeneity",
            output_root,
            "dapi_homogeneity",
        )
        boxplot_metric(
            plot_records,
            conditions,
            "wga_dapi_pearson",
            f"{prefix}within-cell Pearson correlation",
            "WGA-DAPI spatial intensity relationship",
            output_root,
            "wga_dapi_spatial_relationship",
        )
    else:
        long_rows: list[dict[str, object]] = []
        for row in plot_records:
            for channel in spec.channel_names:
                long_rows.append(
                    {
                        "condition": f"{channel}_{row['condition']}",
                        "homogeneity": row.get(
                            f"{channel.lower()}_homogeneity", math.nan
                        ),
                    }
                )
        channel_conditions = [
            f"{channel}_{condition}"
            for channel in spec.channel_names
            for condition in conditions
        ]
        boxplot_metric(
            long_rows,
            channel_conditions,
            "homogeneity",
            f"{prefix}boundary homogeneity index",
            "Three-channel homogeneity",
            output_root,
            "3d_channel_homogeneity",
        )

    plot_axial_profiles(
        profile_store_for_lines, conditions, spec.channel_names, output_root, dataset
    )
    # Heatmaps retain one row per cell in both modes.
    plot_heatmaps(cell_profile_store, conditions, spec.channel_names, output_root, dataset)
    plot_radial_profiles(
        radial_store_for_lines, conditions, spec.channel_names, output_root, dataset
    )
    plot_standardized_maps(
        condition_map_store, conditions, spec.channel_names, output_root
    )
    if dataset == "2d_wga_dapi":
        plot_wga_dapi_overlap_maps(condition_map_store, conditions, output_root)

    summary = {
        "dataset": dataset,
        "source_mode": source_mode,
        "run_mode": "example" if example else "full",
        "manifest": str(manifest),
        "pca_root": str(pca_root),
        "normalized_root": str(normalized_root),
        "output_root": str(output_root),
        "conditions": conditions,
        "n_images": len(manifest_rows),
        "n_cells": len(cell_records),
        "n_roi_summaries": len(roi_records),
        "channel_names": list(spec.channel_names),
        "radial_bins": radial_bins,
        "pixel_size_um": pixel_size_um,
        "plotting_unit": "cells" if example else "independent images/ROIs",
        "interpretation": (
            "Example mode uses one selected image per condition and is intended for "
            "workflow validation only."
            if example
            else "Full mode uses all processed images. Scalar condition plots and "
            "profile uncertainty are based on image/ROI-level summaries."
        ),
        "intensity_warning": (
            "When source images are normalized segmentation inputs, intensity and "
            "homogeneity figures are technical checks only. Final fluorescence "
            "quantification must use intensity-preserving processed images paired "
            "with final model masks."
        ),
    }
    (output_root / "graph_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"[GRAPHS] {dataset}: {len(manifest_rows)} images, "
        f"{len(cell_records)} cells, {len(roi_records)} ROI summaries"
    )
    print(f"Graphs and measurements: {output_root}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create statistics graphs for one dataset or all datasets. "
            "Use --example to use the one-image-per-condition PCA/normalization outputs."
        )
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--dataset",
        choices=("all", *COMMON["DATASETS"]),
        default="all",
    )
    parser.add_argument("--source-mode")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pca-root", type=Path)
    parser.add_argument("--normalized-root", type=Path)
    parser.add_argument("--all-mask-summary", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--radial-bins", type=int, default=20)
    parser.add_argument("--pixel-size-um", type=float)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = COMMON["find_project_root"](SCRIPT_FILE, args.project_root)
    if args.radial_bins < 3:
        raise ValueError("--radial-bins must be at least 3")
    explicit_paths = (
        args.manifest,
        args.pca_root,
        args.normalized_root,
        args.all_mask_summary,
    )
    if args.dataset == "all" and any(path is not None for path in explicit_paths):
        raise ValueError(
            "Explicit input paths require one concrete --dataset. Use automatic "
            "dataset paths with --dataset all."
        )
    if args.dataset == "all" and args.pixel_size_um is not None:
        raise ValueError(
            "--pixel-size-um with --dataset all is unsafe because datasets may have "
            "different calibration. Run each dataset separately with its metadata value."
        )

    datasets = COMMON["selected_datasets"](args.dataset)
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        source_mode = COMMON["source_mode_for"](dataset, args.source_mode, args.dataset)
        pca_root = (
            args.pca_root.expanduser().resolve()
            if args.pca_root
            else COMMON["pca_output_path"](
                project_root, dataset, source_mode, args.example
            )
        )
        normalized_root = (
            args.normalized_root.expanduser().resolve()
            if args.normalized_root
            else COMMON["normalization_output_path"](
                project_root, dataset, source_mode, args.example
            )
        )
        manifest = (
            args.manifest.expanduser().resolve()
            if args.manifest
            else pca_root / "input_manifest_used.csv"
        )
        all_mask_summary = (
            args.all_mask_summary.expanduser().resolve()
            if args.all_mask_summary
            else COMMON["mask_summary_path"](project_root, dataset, source_mode)
        )
        default_output = COMMON["graph_output_path"](
            project_root, dataset, source_mode, args.example
        )
        if args.output_root:
            explicit_output = args.output_root.expanduser().resolve()
            output_root = (
                explicit_output / dataset / source_mode
                if len(datasets) > 1
                else explicit_output
            )
        else:
            output_root = default_output

        required = (manifest, pca_root, normalized_root)
        missing = [path for path in required if not path.exists()]
        if missing:
            message = f"Missing required outputs for {dataset}: {', '.join(map(str, missing))}"
            if args.skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise FileNotFoundError(message)

        summaries.append(
            process_dataset(
                project_root=project_root,
                dataset=dataset,
                source_mode=source_mode,
                example=args.example,
                manifest=manifest,
                pca_root=pca_root,
                normalized_root=normalized_root,
                all_mask_summary=all_mask_summary,
                output_root=output_root,
                radial_bins=args.radial_bins,
                pixel_size_um=args.pixel_size_um,
                overwrite=args.overwrite,
            )
        )

    if not summaries:
        raise RuntimeError("No datasets were processed for graph generation.")

    batch_summary = {
        "requested_dataset": args.dataset,
        "run_mode": "example" if args.example else "full",
        "processed_datasets": [item["dataset"] for item in summaries],
        "summaries": summaries,
    }
    batch_path = (
        project_root
        / "results"
        / "statistics"
        / ("graphs_example" if args.example else "graphs")
        / "graph_batch_summary.json"
    )
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"\nGraph batch summary: {batch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
