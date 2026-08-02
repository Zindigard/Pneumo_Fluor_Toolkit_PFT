r"""Provide command-line and programmatic utilities for create statistics graphs.

Examples
--------
Show all command-line parameters:

    python scripts/statistics/create_statistics_graphs.py --help

Representative execution:

    python scripts/statistics/create_statistics_graphs.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --manifest results/statistics/manifest.csv \
        --example
"""

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
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from openpyxl import Workbook
from scipy import ndimage as ndi

SCRIPT_FILE = Path(__file__).resolve()
EPS = np.finfo(np.float64).eps
EXAMPLE_MODE = False


@dataclass(frozen=True)
class DatasetSpec:
    """Store validated configuration or result data for dataset spec."""
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
    """Find project root in the available data or project structure.

    Args:
        start (Path | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_project_root()
    """
    current = (start or SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT project root; provide --project-root.")


def configure_local_imports(project_root: Path) -> None:
    """Return configure local imports for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.

    Example:
        >>> configure_local_imports(project_root=Path("path/to/resource"))
    """
    statistics_dir = project_root / "scripts" / "statistics"
    if str(statistics_dir) not in sys.path:
        sys.path.insert(0, str(statistics_dir))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV data from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        list[dict[str, str]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = read_csv(path=Path("path/to/resource"))
    """
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write CSV data to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (list[dict[str, object]]): Text value specifying rows.

    Example:
        >>> write_csv(path=Path("path/to/resource"), rows="rows")
    """
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
    """Infer condition from the supplied model inputs.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample_name (str): Text value specifying sample name.

    Returns:
        tuple[str, dict[str, object]]: Mapping containing the generated or resolved values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = infer_condition(dataset="2d_time", sample_name="sample_name")
    """
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
    """Return safe float for the supplied inputs.

    Args:
        value (object): Value to validate, transform, store, or forward.
        default (float): Numerical value controlling default. Defaults to ``math.nan``.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = safe_float(value=...)
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def setup_style() -> None:
    """Return setup style for the supplied inputs.

    Example:
        >>> setup_style()
    """
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
    """Save figure to persistent storage.

    Args:
        fig (plt.Figure): Matplotlib figure object containing the generated visualization.
        output_dir (Path): Directory where generated resources are written.
        stem (str): Text value specifying stem.

    Example:
        >>> save_figure(
        ...     fig=...,
        ...     output_dir=Path("path/to/resource"),
        ...     stem="stem",
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def profile_minmax(profile: np.ndarray) -> np.ndarray:
    """Return profile minmax for the supplied inputs.

    Args:
        profile (np.ndarray): Array containing profile.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = profile_minmax(profile=image_array)
    """
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


def axial_profile_metrics(profile: np.ndarray) -> dict[str, float]:
    """Return scalar descriptors of a normalized pole-to-pole profile.

    The 50% axial extent is a profile-based signal length: the fraction of the
    normalized cell axis occupied by values at least 50% of the profile peak.

    Args:
        profile (np.ndarray): Array containing profile.

    Returns:
        dict[str, float]: Mapping containing the generated or resolved values.

    Example:
        >>> result = axial_profile_metrics(profile=image_array)
    """
    values = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            "axial_auc": math.nan,
            "axial_centroid": math.nan,
            "axial_spread": math.nan,
            "axial_peak_position": math.nan,
            "axial_extent_50_fraction": math.nan,
            "axial_extent_25_fraction": math.nan,
            "pole1_intensity": math.nan,
            "midcell_intensity": math.nan,
            "pole2_intensity": math.nan,
            "mean_pole_intensity": math.nan,
            "midcell_to_pole_ratio": math.nan,
            "pole_asymmetry": math.nan,
        }
    values = np.where(finite, values, 0.0)
    x = np.linspace(0.0, 1.0, values.size)
    total = float(np.sum(values))
    centroid = float(np.sum(x * values) / (total + EPS))
    spread = float(np.sqrt(np.sum(values * (x - centroid) ** 2) / (total + EPS)))

    def extent_fraction(level: float) -> float:
        """Return extent fraction for the supplied inputs.

        Args:
            level (float): Numerical value controlling level.

        Returns:
            float: Computed numerical result.

        Example:
            >>> result = extent_fraction(level=0.5)
        """
        selected = np.flatnonzero(values >= level)
        if selected.size == 0:
            return 0.0
        if values.size <= 1:
            return 0.0
        bin_width = 1.0 / (values.size - 1)
        return float(min(1.0, (selected[-1] - selected[0]) * bin_width + bin_width))

    region = max(1, int(round(values.size * 0.10)))
    centre_half = max(1, int(round(values.size * 0.05)))
    centre = values.size // 2
    centre_start = max(0, centre - centre_half)
    centre_end = min(values.size, centre + centre_half + 1)
    pole1 = float(np.mean(values[:region]))
    pole2 = float(np.mean(values[-region:]))
    midcell = float(np.mean(values[centre_start:centre_end]))
    mean_pole = 0.5 * (pole1 + pole2)
    return {
        "axial_auc": float(np.trapezoid(values, x=x)),
        "axial_centroid": centroid,
        "axial_spread": spread,
        "axial_peak_position": float(x[int(np.argmax(values))]),
        "axial_extent_50_fraction": extent_fraction(0.50),
        "axial_extent_25_fraction": extent_fraction(0.25),
        "pole1_intensity": pole1,
        "midcell_intensity": midcell,
        "pole2_intensity": pole2,
        "mean_pole_intensity": mean_pole,
        "midcell_to_pole_ratio": (midcell / mean_pole) if mean_pole > EPS else math.nan,
        "pole_asymmetry": abs(pole1 - pole2),
    }


def radial_profile_metrics(profile: np.ndarray) -> dict[str, float]:
    """Return scalar descriptors of a normalized centre-to-boundary profile.

    Args:
        profile (np.ndarray): Array containing profile.

    Returns:
        dict[str, float]: Mapping containing the generated or resolved values.

    Example:
        >>> result = radial_profile_metrics(profile=image_array)
    """
    values = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            "radial_auc": math.nan,
            "radial_centroid": math.nan,
            "centre_intensity": math.nan,
            "boundary_intensity": math.nan,
            "boundary_to_centre_ratio": math.nan,
        }
    values = np.where(finite, values, 0.0)
    x = np.linspace(0.0, 1.0, values.size)
    total = float(np.sum(values))
    region = max(1, int(round(values.size * 0.20)))
    centre_intensity = float(np.mean(values[:region]))
    boundary_intensity = float(np.mean(values[-region:]))
    return {
        "radial_auc": float(np.trapezoid(values, x=x)),
        "radial_centroid": float(np.sum(x * values) / (total + EPS)),
        "centre_intensity": centre_intensity,
        "boundary_intensity": boundary_intensity,
        "boundary_to_centre_ratio": (boundary_intensity / centre_intensity) if centre_intensity > EPS else math.nan,
    }


def axial_profile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average intensity across cell width at each normalized axial position.

    Args:
        image (np.ndarray): Input image array to process.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = axial_profile(image=image_array, mask=image_array)
    """
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
    """Compute centre-to-boundary profile using the Methods radial coordinate.

    Args:
        image (np.ndarray): Input image array to process.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        bins (int): Numerical value controlling bins.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = radial_profile(
        ...     image=image_array,
        ...     mask=image_array,
        ...     bins=1,
        ... )
    """
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
    """Return principal-component analysis result dimensions for the supplied inputs.

    Args:
        binary (np.ndarray): Array containing binary.

    Returns:
        tuple[float, float]: Collection containing the generated or selected values.

    Example:
        >>> result = pca_dimensions(binary=image_array)
    """
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
    """Return background parameters for the supplied inputs.

    Args:
        image_cyx (np.ndarray): Array containing image cyx.
        full_mask (np.ndarray): Array containing full mask.

    Returns:
        list[dict[str, float]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = background_parameters(image_cyx=image_array, full_mask=image_array)
    """
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
    """Return boundary homogeneity for the supplied inputs.

    Args:
        raw_channel (np.ndarray): Array containing raw channel.
        corrected_channel (np.ndarray): Array containing corrected channel.
        cell_mask (np.ndarray): Array containing cell mask.
        threshold (float): Numerical decision threshold used by the operation.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = boundary_homogeneity(
        ...     raw_channel=image_array,
        ...     corrected_channel=image_array,
        ...     cell_mask=image_array,
        ...     threshold=0.5,
        ... )
    """
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
    """Return dapi homogeneity for the supplied inputs.

    Args:
        raw_channel (np.ndarray): Array containing raw channel.
        corrected_channel (np.ndarray): Array containing corrected channel.
        cell_mask (np.ndarray): Array containing cell mask.
        threshold (float): Numerical decision threshold used by the operation.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = dapi_homogeneity(
        ...     raw_channel=image_array,
        ...     corrected_channel=image_array,
        ...     cell_mask=image_array,
        ...     threshold=0.5,
        ... )
    """
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
    """Return pearson inside for the supplied inputs.

    Args:
        a (np.ndarray): Array containing a.
        b (np.ndarray): Array containing b.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.

    Returns:
        float: Computed numerical result.

    Example:
        >>> result = pearson_inside(
        ...     a=image_array,
        ...     b=image_array,
        ...     mask=image_array,
        ... )
    """
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    finite = np.isfinite(av) & np.isfinite(bv)
    av = av[finite]
    bv = bv[finite]
    if av.size < 3 or np.std(av) <= EPS or np.std(bv) <= EPS:
        return math.nan
    return float(np.corrcoef(av, bv)[0, 1])


def channel_overlap_metrics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    first_positive_threshold: float,
    second_positive_threshold: float,
) -> dict[str, float]:
    """Compute descriptive intensity-overlap metrics inside one cell.

    Args:
        first (np.ndarray): Array containing first.
        second (np.ndarray): Array containing second.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        first_positive_threshold (float): Decision threshold applied to first positive.
        second_positive_threshold (float): Decision threshold applied to second positive.

    Returns:
        dict[str, float]: Mapping containing the generated or resolved values.

    Example:
        >>> result = channel_overlap_metrics(
        ...     first=image_array,
        ...     second=image_array,
        ...     mask=image_array,
        ...     first_positive_threshold=0.5,
        ...     second_positive_threshold=0.5,
        ... )
    """
    first_values = np.asarray(first[mask], dtype=np.float64)
    second_values = np.asarray(second[mask], dtype=np.float64)
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    first_values = first_values[finite]
    second_values = second_values[finite]
    if first_values.size == 0:
        return {
            "overlap_cosine": math.nan,
            "positive_overlap_fraction": math.nan,
            "positive_union_fraction": math.nan,
            "manders_first_in_second": math.nan,
            "manders_second_in_first": math.nan,
        }
    first_positive = first_values > first_positive_threshold
    second_positive = second_values > second_positive_threshold
    intersection = first_positive & second_positive
    union = first_positive | second_positive
    denominator = float(
        np.sqrt(np.sum(first_values**2) * np.sum(second_values**2))
    )
    return {
        "overlap_cosine": float(np.sum(first_values * second_values) / (denominator + EPS)),
        "positive_overlap_fraction": float(np.mean(intersection)),
        "positive_union_fraction": float(np.mean(union)),
        "manders_first_in_second": float(
            np.sum(first_values[second_positive]) / (np.sum(first_values) + EPS)
        ),
        "manders_second_in_first": float(
            np.sum(second_values[first_positive]) / (np.sum(second_values) + EPS)
        ),
    }


def condition_sort_key(spec: DatasetSpec, condition: str) -> tuple[int, str]:
    """Return condition sort key for the supplied inputs.

    Args:
        spec (DatasetSpec): Value specifying spec for the operation.
        condition (str): Text value specifying condition.

    Returns:
        tuple[int, str]: Collection containing the generated or selected values.

    Example:
        >>> result = condition_sort_key(spec=..., condition="condition")
    """
    try:
        return spec.condition_order.index(condition), condition
    except ValueError:
        return len(spec.condition_order), condition


def condition_display(condition: str) -> str:
    """Return condition display for the supplied inputs.

    Args:
        condition (str): Text value specifying condition.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = condition_display(condition="condition")
    """
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
    """Plot two-dimensional data time metric over time for visual assessment.

    Args:
        records (list[dict[str, object]]): Text value specifying records.
        key (str): Key used to access or identify an entry in a mapping.
        ylabel (str): Text value specifying ylabel.
        title (str): Title displayed on the generated figure or report section.
        output_dir (Path): Directory where generated resources are written.
        stem (str): Text value specifying stem.

    Example:
        >>> plot_2d_time_metric_over_time(
        ...     records="records",
        ...     key="key",
        ...     ylabel="ylabel",
        ...     title="title",
        ...     output_dir=Path("path/to/resource"),
        ...     stem="stem",
        ... )
    """
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
    """Add note to the current data structure.

    Args:
        fig (plt.Figure): Matplotlib figure object containing the generated visualization.

    Example:
        >>> add_note(fig=...)
    """
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
    """Return jittered scatter for the supplied inputs.

    Args:
        ax (Any): Matplotlib axes object on which graphical elements are drawn.
        positions (Sequence[float]): Numerical value controlling positions.
        groups (Sequence[np.ndarray]): Array containing groups.

    Example:
        >>> jittered_scatter(
        ...     ax=...,
        ...     positions=0.5,
        ...     groups=image_array,
        ... )
    """
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
    """Return boxplot metric for the supplied inputs.

    Args:
        records (list[dict[str, object]]): Text value specifying records.
        conditions (list[str]): Text value specifying conditions.
        key (str): Key used to access or identify an entry in a mapping.
        ylabel (str): Text value specifying ylabel.
        title (str): Title displayed on the generated figure or report section.
        output_dir (Path): Directory where generated resources are written.
        stem (str): Text value specifying stem.

    Example:
        >>> boxplot_metric(
        ...     records="records",
        ...     conditions="conditions",
        ...     key="key",
        ...     ylabel="ylabel",
        ...     title="title",
        ...     output_dir=Path("path/to/resource"),
        ...     stem="stem",
        ... )
    """
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
    """Return mean and band for the supplied inputs.

    Args:
        matrix (np.ndarray): Array containing matrix.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Example:
        >>> result = mean_and_band(matrix=image_array)
    """
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
    """Plot axial profiles for visual assessment.

    Args:
        profile_store (dict[tuple[str, str], list[np.ndarray]]): Array containing profile store.
        conditions (list[str]): Text value specifying conditions.
        channel_names (tuple[str, ...]): Text value specifying channel names.
        output_dir (Path): Directory where generated resources are written.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Example:
        >>> plot_axial_profiles(
        ...     profile_store=image_array,
        ...     conditions="conditions",
        ...     channel_names="channel_names",
        ...     output_dir=Path("path/to/resource"),
        ...     dataset="2d_time",
        ... )
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", [f"C{i}" for i in range(10)]
    )

    if dataset == "2d_time":
        times = sorted(
            {
                int(match.group(1))
                for condition in conditions
                for match in [re.search(r"(\d+)min", condition)]
                if match is not None
            }
        )
        for channel in channel_names:
            for time in times:
                fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
                plotted = False
                for medium, linestyle, color in (("THY", "-", cycle[0]), ("NHS", "--", cycle[1])):
                    condition = f"{medium}_{time}min"
                    profiles = profile_store.get((condition, channel), [])
                    if not profiles:
                        continue
                    plotted = True
                    matrix = np.vstack(profiles)
                    mean, low, high = mean_and_band(matrix)
                    x = np.linspace(0.0, 1.0, mean.size)
                    ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.8, label=medium)
                    ax.fill_between(x, low, high, color=color, alpha=0.12)
                if not plotted:
                    plt.close(fig)
                    continue
                ax.axvline(0.5, linestyle="--", linewidth=0.8, color="0.45")
                ax.set_xticks([0.0, 0.5, 1.0], ["Pole 1", "Midcell", "Pole 2"])
                ax.set_xlabel("Normalized axial coordinate")
                ax.set_ylabel("Normalized fluorescence intensity")
                ax.set_title(f"{channel} axial fluorescence profile, {time} min")
                ax.grid(alpha=0.2)
                ax.legend(title="Medium")
                add_note(fig)
                save_figure(fig, output_dir, f"axial_profile_{channel.lower()}_{time}min")
        return

    if dataset == "2d_wga_dapi":
        comparison_specs = [
            ("thy_vs_csp", "THY vs THY + CSP", [("THY_noCSP", "THY"), ("THY_CSP", "THY + CSP")]),
            ("nhs_vs_csp", "NHS vs NHS + CSP", [("NHS_noCSP", "NHS"), ("NHS_CSP", "NHS + CSP")]),
            ("thy_vs_nhs_pooled", "All THY vs all NHS", [(["THY_noCSP", "THY_CSP"], "All THY"), (["NHS_noCSP", "NHS_CSP"], "All NHS")]),
        ]
        for channel in channel_names:
            for stem_suffix, title, groups in comparison_specs:
                fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
                plotted = False
                for index, group in enumerate(groups):
                    key_or_keys, label = group
                    if isinstance(key_or_keys, str):
                        profiles = profile_store.get((key_or_keys, channel), [])
                    else:
                        profiles = []
                        for condition in key_or_keys:
                            profiles.extend(profile_store.get((condition, channel), []))
                    if not profiles:
                        continue
                    plotted = True
                    matrix = np.vstack(profiles)
                    mean, low, high = mean_and_band(matrix)
                    x = np.linspace(0.0, 1.0, mean.size)
                    color = cycle[index % len(cycle)]
                    ax.plot(x, mean, color=color, linewidth=1.8, label=label)
                    ax.fill_between(x, low, high, color=color, alpha=0.12)
                if not plotted:
                    plt.close(fig)
                    continue
                ax.axvline(0.5, linestyle="--", linewidth=0.8, color="0.45")
                ax.set_xticks([0.0, 0.5, 1.0], ["Pole 1", "Midcell", "Pole 2"])
                ax.set_xlabel("Normalized axial coordinate")
                ax.set_ylabel("Normalized fluorescence intensity")
                ax.set_title(f"{channel} axial fluorescence profiles: {title}")
                ax.grid(alpha=0.2)
                ax.legend()
                add_note(fig)
                save_figure(fig, output_dir, f"axial_profile_{channel.lower()}_{stem_suffix}")
        return

    if dataset == "3d_mip":
        for condition in conditions:
            fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
            plotted = False
            for index, channel in enumerate(channel_names):
                profiles = profile_store.get((condition, channel), [])
                if not profiles:
                    continue
                plotted = True
                matrix = np.vstack(profiles)
                mean, low, high = mean_and_band(matrix)
                x = np.linspace(0.0, 1.0, mean.size)
                color = cycle[index % len(cycle)]
                ax.plot(x, mean, color=color, linewidth=1.8, label=channel)
                ax.fill_between(x, low, high, color=color, alpha=0.12)
            if not plotted:
                plt.close(fig)
                continue
            ax.axvline(0.5, linestyle="--", linewidth=0.8, color="0.45")
            ax.set_xticks([0.0, 0.5, 1.0], ["Pole 1", "Midcell", "Pole 2"])
            ax.set_xlabel("Normalized axial coordinate")
            ax.set_ylabel("Normalized fluorescence intensity")
            ax.set_title(f"3D SIM axial fluorescence profiles: {condition_display(condition)}")
            ax.grid(alpha=0.2)
            ax.legend(title="Channel")
            add_note(fig)
            save_figure(fig, output_dir, f"axial_profile_{condition.lower()}")
        return


def plot_heatmaps(
    profile_store: dict[tuple[str, str], list[np.ndarray]],
    conditions: list[str],
    channel_names: tuple[str, ...],
    output_dir: Path,
    dataset: str,
) -> None:
    """Plot heatmaps for visual assessment.

    Args:
        profile_store (dict[tuple[str, str], list[np.ndarray]]): Array containing profile store.
        conditions (list[str]): Text value specifying conditions.
        channel_names (tuple[str, ...]): Text value specifying channel names.
        output_dir (Path): Directory where generated resources are written.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Example:
        >>> plot_heatmaps(
        ...     profile_store=image_array,
        ...     conditions="conditions",
        ...     channel_names="channel_names",
        ...     output_dir=Path("path/to/resource"),
        ...     dataset="2d_time",
        ... )
    """
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
    """Plot radial profiles for visual assessment.

    Args:
        radial_store (dict[tuple[str, str], list[np.ndarray]]): Array containing radial store.
        conditions (list[str]): Text value specifying conditions.
        channel_names (tuple[str, ...]): Text value specifying channel names.
        output_dir (Path): Directory where generated resources are written.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Example:
        >>> plot_radial_profiles(
        ...     radial_store=image_array,
        ...     conditions="conditions",
        ...     channel_names="channel_names",
        ...     output_dir=Path("path/to/resource"),
        ...     dataset="2d_time",
        ... )
    """
    if not radial_store:
        return
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", [f"C{i}" for i in range(10)]
    )

    if dataset == "2d_time":
        times = sorted(
            {
                int(match.group(1))
                for condition in conditions
                for match in [re.search(r"(\d+)min", condition)]
                if match is not None
            }
        )
        channel = channel_names[0]
        for time in times:
            fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
            plotted = False
            for medium, linestyle, color in (("THY", "-", cycle[0]), ("NHS", "--", cycle[1])):
                condition = f"{medium}_{time}min"
                profiles = radial_store.get((condition, channel), [])
                if not profiles:
                    continue
                plotted = True
                matrix = np.vstack(profiles)
                mean, low, high = mean_and_band(matrix)
                x = np.linspace(0.0, 1.0, mean.size)
                ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.8, label=medium)
                ax.fill_between(x, low, high, color=color, alpha=0.12)
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xticks([0.0, 0.5, 1.0], ["Cell centre", "Intermediate radius", "Cell boundary"])
            ax.set_xlabel("Normalized radial distance")
            ax.set_ylabel("Normalized fluorescence intensity")
            ax.set_title(f"{channel} radial fluorescence profile, {time} min")
            ax.grid(alpha=0.2)
            ax.legend(title="Medium")
            add_note(fig)
            save_figure(fig, output_dir, f"radial_profile_{channel.lower()}_{time}min")
        return

    if dataset == "2d_wga_dapi":
        comparison_specs = [
            ("thy_vs_csp", "THY vs THY + CSP", [("THY_noCSP", "THY"), ("THY_CSP", "THY + CSP")]),
            ("nhs_vs_csp", "NHS vs NHS + CSP", [("NHS_noCSP", "NHS"), ("NHS_CSP", "NHS + CSP")]),
            ("thy_vs_nhs_pooled", "All THY vs all NHS", [(["THY_noCSP", "THY_CSP"], "All THY"), (["NHS_noCSP", "NHS_CSP"], "All NHS")]),
        ]
        channel_styles = {"WGA": "-", "DAPI": "--"}
        for stem_suffix, title, groups in comparison_specs:
            fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
            condition_handles: list[Line2D] = []
            plotted = False
            for index, group in enumerate(groups):
                key_or_keys, label = group
                color = cycle[index % len(cycle)]
                condition_handles.append(Line2D([0], [0], color=color, linewidth=2.0, label=label))
                for channel in ("WGA", "DAPI"):
                    if isinstance(key_or_keys, str):
                        profiles = radial_store.get((key_or_keys, channel), [])
                    else:
                        profiles = []
                        for condition in key_or_keys:
                            profiles.extend(radial_store.get((condition, channel), []))
                    if not profiles:
                        continue
                    plotted = True
                    matrix = np.vstack(profiles)
                    mean, low, high = mean_and_band(matrix)
                    x = np.linspace(0.0, 1.0, mean.size)
                    ax.plot(x, mean, color=color, linestyle=channel_styles[channel], linewidth=1.8)
                    ax.fill_between(x, low, high, color=color, alpha=0.08)
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xticks([0.0, 0.5, 1.0], ["Cell centre", "Intermediate radius", "Cell boundary"])
            ax.set_xlabel("Normalized radial distance")
            ax.set_ylabel("Normalized fluorescence intensity")
            ax.set_title(f"WGA-DAPI radial fluorescence profiles: {title}")
            ax.grid(alpha=0.2)
            condition_legend = ax.legend(handles=condition_handles, title="Condition", loc="center right")
            ax.add_artist(condition_legend)
            channel_handles = [
                Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="WGA"),
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="DAPI"),
            ]
            ax.legend(handles=channel_handles, title="Channel", loc="lower center")
            add_note(fig)
            save_figure(fig, output_dir, f"radial_profiles_wga_dapi_{stem_suffix}")
        return

    if dataset == "3d_mip":
        for condition in conditions:
            fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
            plotted = False
            for index, channel in enumerate(channel_names):
                profiles = radial_store.get((condition, channel), [])
                if not profiles:
                    continue
                plotted = True
                matrix = np.vstack(profiles)
                mean, low, high = mean_and_band(matrix)
                x = np.linspace(0.0, 1.0, mean.size)
                color = cycle[index % len(cycle)]
                ax.plot(x, mean, color=color, linewidth=1.8, label=channel)
                ax.fill_between(x, low, high, color=color, alpha=0.12)
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xticks([0.0, 0.5, 1.0], ["Cell centre", "Intermediate radius", "Cell boundary"])
            ax.set_xlabel("Normalized radial distance")
            ax.set_ylabel("Normalized fluorescence intensity")
            ax.set_title(f"3D SIM radial fluorescence profiles: {condition_display(condition)}")
            ax.grid(alpha=0.2)
            ax.legend(title="Channel")
            add_note(fig)
            save_figure(fig, output_dir, f"radial_profile_{condition.lower()}")
        return


def plot_standardized_maps(
    map_store: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    conditions: list[str],
    channel_names: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Plot standardized maps for visual assessment.

    Args:
        map_store (dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]): Array containing map store.
        conditions (list[str]): Text value specifying conditions.
        channel_names (tuple[str, ...]): Text value specifying channel names.
        output_dir (Path): Directory where generated resources are written.

    Example:
        >>> plot_standardized_maps(
        ...     map_store=image_array,
        ...     conditions="conditions",
        ...     channel_names="channel_names",
        ...     output_dir=Path("path/to/resource"),
        ... )
    """
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
    """Plot wga dapi overlap maps for visual assessment.

    Args:
        map_store (dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]): Array containing map store.
        conditions (list[str]): Text value specifying conditions.
        output_dir (Path): Directory where generated resources are written.

    Example:
        >>> plot_wga_dapi_overlap_maps(
        ...     map_store=image_array,
        ...     conditions="conditions",
        ...     output_dir=Path("path/to/resource"),
        ... )
    """
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


def _numeric_metrics(records: list[dict[str, object]]) -> list[str]:
    """Return numeric metrics for the supplied inputs.

    Args:
        records (list[dict[str, object]]): Text value specifying records.

    Returns:
        list[str]: Collection containing the generated or selected values.

    Example:
        >>> result = _numeric_metrics(records="records")
    """
    excluded = {
        "dataset",
        "annotation_id",
        "sample_name",
        "condition",
        "medium",
        "time_min",
        "csp",
        "source_label",
        "pixel_size_um",
        "n_cells",
    }
    metrics: list[str] = []
    for row in records:
        for key, value in row.items():
            if key in excluded or key in metrics:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                metrics.append(key)
    return sorted(metrics)


def _profile_comparison_plan(dataset: str) -> list[tuple[str, str, list[str], str, list[str]]]:
    """Return profile comparison plan for the supplied inputs.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[tuple[str, str, list[str], str, list[str]]]: Collection containing the generated or selected values.

    Example:
        >>> result = _profile_comparison_plan(dataset="2d_time")
    """
    if dataset == "2d_time":
        return [
            (f"{time} min", "THY", [f"THY_{time}min"], "NHS", [f"NHS_{time}min"])
            for time in (5, 20, 40, 120)
        ]
    if dataset == "2d_wga_dapi":
        return [
            ("THY CSP effect", "THY", ["THY_noCSP"], "THY + CSP", ["THY_CSP"]),
            ("NHS CSP effect", "NHS", ["NHS_noCSP"], "NHS + CSP", ["NHS_CSP"]),
            ("No-CSP medium effect", "THY", ["THY_noCSP"], "NHS", ["NHS_noCSP"]),
            ("CSP medium effect", "THY + CSP", ["THY_CSP"], "NHS + CSP", ["NHS_CSP"]),
            ("Pooled medium comparison", "All THY", ["THY_noCSP", "THY_CSP"], "All NHS", ["NHS_noCSP", "NHS_CSP"]),
        ]
    if dataset == "3d_mip":
        return [("THY vs NHS", "THY", ["THY"], "NHS", ["NHS"])]
    return []


def _records_for_conditions(
    records: list[dict[str, object]], conditions: list[str]
) -> list[dict[str, object]]:
    """Return records for conditions for the supplied inputs.

    Args:
        records (list[dict[str, object]]): Text value specifying records.
        conditions (list[str]): Text value specifying conditions.

    Returns:
        list[dict[str, object]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _records_for_conditions(records="records", conditions="conditions")
    """
    condition_set = set(conditions)
    return [row for row in records if str(row.get("condition", "")) in condition_set]


def _profile_values(
    store: dict[tuple[str, str], list[np.ndarray]],
    conditions: list[str],
    channel: str,
) -> list[np.ndarray]:
    """Return profile values for the supplied inputs.

    Args:
        store (dict[tuple[str, str], list[np.ndarray]]): Array containing store.
        conditions (list[str]): Text value specifying conditions.
        channel (str): Channel index or channel identifier selected for processing.

    Returns:
        list[np.ndarray]: Collection containing the generated or selected values.

    Example:
        >>> result = _profile_values(
        ...     store=image_array,
        ...     conditions="conditions",
        ...     channel="channel",
        ... )
    """
    values: list[np.ndarray] = []
    for condition in conditions:
        values.extend(store.get((condition, channel), []))
    return values


def _append_profile_difference_sheet(
    workbook: Workbook,
    sheet_name: str,
    dataset: str,
    store: dict[tuple[str, str], list[np.ndarray]],
    channel_names: tuple[str, ...],
    coordinate_name: str,
) -> None:
    """Append profile difference sheet to the current collection.

    Args:
        workbook (Workbook): Value specifying workbook for the operation.
        sheet_name (str): Text value specifying sheet name.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        store (dict[tuple[str, str], list[np.ndarray]]): Array containing store.
        channel_names (tuple[str, ...]): Text value specifying channel names.
        coordinate_name (str): Text value specifying coordinate name.

    Example:
        >>> _append_profile_difference_sheet(
        ...     workbook=...,
        ...     sheet_name="sheet_name",
        ...     dataset="2d_time",
        ...     store=image_array,
        ...     channel_names="channel_names",
        ...     coordinate_name="coordinate_name",
        ... )
    """
    sheet = workbook.create_sheet(sheet_name)
    sheet.append([
        "dataset",
        "comparison",
        "channel",
        coordinate_name,
        "group_a",
        "n_a",
        "mean_a",
        "std_a",
        "group_b",
        "n_b",
        "mean_b",
        "std_b",
        "difference_b_minus_a",
        "propagated_sd",
        "percent_difference_vs_a",
    ])
    for comparison_name, label_a, conditions_a, label_b, conditions_b in _profile_comparison_plan(dataset):
        for channel in channel_names:
            profiles_a = _profile_values(store, conditions_a, channel)
            profiles_b = _profile_values(store, conditions_b, channel)
            if not profiles_a or not profiles_b:
                continue
            matrix_a = np.vstack(profiles_a)
            matrix_b = np.vstack(profiles_b)
            mean_a = np.nanmean(matrix_a, axis=0)
            mean_b = np.nanmean(matrix_b, axis=0)
            std_a = np.nanstd(matrix_a, axis=0, ddof=1) if matrix_a.shape[0] > 1 else np.zeros(matrix_a.shape[1])
            std_b = np.nanstd(matrix_b, axis=0, ddof=1) if matrix_b.shape[0] > 1 else np.zeros(matrix_b.shape[1])
            x = np.linspace(0.0, 1.0, mean_a.size)
            for index, coordinate in enumerate(x):
                difference = float(mean_b[index] - mean_a[index])
                propagated_sd = float(np.sqrt(std_a[index] ** 2 + std_b[index] ** 2))
                percent = float(100.0 * difference / (abs(mean_a[index]) + EPS))
                sheet.append([
                    dataset,
                    comparison_name,
                    channel,
                    float(coordinate),
                    label_a,
                    int(matrix_a.shape[0]),
                    float(mean_a[index]),
                    float(std_a[index]),
                    label_b,
                    int(matrix_b.shape[0]),
                    float(mean_b[index]),
                    float(std_b[index]),
                    difference,
                    propagated_sd,
                    percent,
                ])


def write_condition_metric_workbook(
    dataset: str,
    records: list[dict[str, object]],
    axial_store: dict[tuple[str, str], list[np.ndarray]],
    radial_store: dict[tuple[str, str], list[np.ndarray]],
    channel_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Write condition metric workbook to persistent storage.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        records (list[dict[str, object]]): Text value specifying records.
        axial_store (dict[tuple[str, str], list[np.ndarray]]): Array containing axial store.
        radial_store (dict[tuple[str, str], list[np.ndarray]]): Array containing radial store.
        channel_names (tuple[str, ...]): Text value specifying channel names.
        output_path (Path): Filesystem path where the generated result is written.

    Example:
        >>> write_condition_metric_workbook(
        ...     dataset="2d_time",
        ...     records="records",
        ...     axial_store=image_array,
        ...     radial_store=image_array,
        ...     channel_names="channel_names",
        ...     output_path=Path("path/to/resource"),
        ... )
    """
    if not records:
        return
    metrics = _numeric_metrics(records)
    if not metrics:
        return
    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "condition_summary"
    summary_sheet.append(["dataset", "condition", "metric", "n", "mean", "std"])
    condition_names = sorted({str(row.get("condition", "")) for row in records})
    for condition in condition_names:
        group_rows = _records_for_conditions(records, [condition])
        for metric in metrics:
            values = np.asarray([safe_float(row.get(metric)) for row in group_rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            summary_sheet.append([
                dataset,
                condition_display(condition),
                metric,
                int(values.size),
                float(np.mean(values)),
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            ])

    planned_sheet = workbook.create_sheet("planned_differences")
    planned_sheet.append([
        "dataset",
        "comparison",
        "metric",
        "group_a",
        "n_a",
        "mean_a",
        "std_a",
        "group_b",
        "n_b",
        "mean_b",
        "std_b",
        "difference_b_minus_a",
        "propagated_sd",
        "percent_difference_vs_a",
    ])
    for comparison_name, label_a, conditions_a, label_b, conditions_b in _profile_comparison_plan(dataset):
        rows_a = _records_for_conditions(records, conditions_a)
        rows_b = _records_for_conditions(records, conditions_b)
        for metric in metrics:
            values_a = np.asarray([safe_float(row.get(metric)) for row in rows_a], dtype=float)
            values_b = np.asarray([safe_float(row.get(metric)) for row in rows_b], dtype=float)
            values_a = values_a[np.isfinite(values_a)]
            values_b = values_b[np.isfinite(values_b)]
            if values_a.size == 0 or values_b.size == 0:
                continue
            mean_a = float(np.mean(values_a))
            mean_b = float(np.mean(values_b))
            std_a = float(np.std(values_a, ddof=1)) if values_a.size > 1 else 0.0
            std_b = float(np.std(values_b, ddof=1)) if values_b.size > 1 else 0.0
            difference = mean_b - mean_a
            planned_sheet.append([
                dataset,
                comparison_name,
                metric,
                label_a,
                int(values_a.size),
                mean_a,
                std_a,
                label_b,
                int(values_b.size),
                mean_b,
                std_b,
                difference,
                float(np.sqrt(std_a**2 + std_b**2)),
                float(100.0 * difference / (abs(mean_a) + EPS)),
            ])

    all_pairs_sheet = workbook.create_sheet("all_pairwise_conditions")
    all_pairs_sheet.append([
        "dataset",
        "metric",
        "condition_a",
        "n_a",
        "mean_a",
        "std_a",
        "condition_b",
        "n_b",
        "mean_b",
        "std_b",
        "difference_b_minus_a",
        "propagated_sd",
        "percent_difference_vs_a",
    ])
    for condition_a, condition_b in combinations(condition_names, 2):
        rows_a = _records_for_conditions(records, [condition_a])
        rows_b = _records_for_conditions(records, [condition_b])
        for metric in metrics:
            values_a = np.asarray([safe_float(row.get(metric)) for row in rows_a], dtype=float)
            values_b = np.asarray([safe_float(row.get(metric)) for row in rows_b], dtype=float)
            values_a = values_a[np.isfinite(values_a)]
            values_b = values_b[np.isfinite(values_b)]
            if values_a.size == 0 or values_b.size == 0:
                continue
            mean_a = float(np.mean(values_a))
            mean_b = float(np.mean(values_b))
            std_a = float(np.std(values_a, ddof=1)) if values_a.size > 1 else 0.0
            std_b = float(np.std(values_b, ddof=1)) if values_b.size > 1 else 0.0
            difference = mean_b - mean_a
            all_pairs_sheet.append([
                dataset,
                metric,
                condition_display(condition_a),
                int(values_a.size),
                mean_a,
                std_a,
                condition_display(condition_b),
                int(values_b.size),
                mean_b,
                std_b,
                difference,
                float(np.sqrt(std_a**2 + std_b**2)),
                float(100.0 * difference / (abs(mean_a) + EPS)),
            ])

    _append_profile_difference_sheet(
        workbook,
        "axial_profile_differences",
        dataset,
        axial_store,
        channel_names,
        "normalized_axial_coordinate",
    )
    _append_profile_difference_sheet(
        workbook,
        "radial_profile_differences",
        dataset,
        radial_store,
        channel_names,
        "normalized_radial_coordinate",
    )

    definitions_sheet = workbook.create_sheet("metric_definitions")
    definitions_sheet.append(["metric pattern", "definition"])
    definitions = [
        ("*_axial_extent_50_fraction", "Fraction of normalized pole-to-pole axis with profile intensity at least 50% of its peak; this is the profile-based signal length."),
        ("*_axial_extent_50_pixels", "50% axial signal extent multiplied by PCA-derived cell length in pixels."),
        ("*_axial_extent_50_um", "50% axial signal extent in micrometres when --pixel-size-um is supplied."),
        ("*_axial_spread", "Intensity-weighted standard deviation along the normalized pole-to-pole coordinate."),
        ("*_axial_centroid", "Intensity-weighted position along the normalized pole-to-pole coordinate."),
        ("*_radial_centroid", "Intensity-weighted radial position from cell centre 0 to boundary 1."),
        ("*_boundary_to_centre_ratio", "Mean boundary-region intensity divided by mean centre-region intensity."),
        ("wga_dapi_pearson", "Within-cell Pearson correlation of background-corrected WGA and DAPI intensities."),
        ("wga_dapi_overlap_cosine", "Cosine overlap coefficient of background-corrected WGA and DAPI intensities."),
        ("wga_dapi_positive_overlap_fraction", "Fraction of cell pixels positive in both WGA and DAPI."),
        ("wga_manders_in_dapi", "Fraction of WGA intensity located in DAPI-positive pixels."),
        ("dapi_manders_in_wga", "Fraction of DAPI intensity located in WGA-positive pixels."),
        ("propagated_sd", "Descriptive combination sqrt(SD_A^2 + SD_B^2), not the SD of paired differences."),
    ]
    for metric, definition in definitions:
        definitions_sheet.append([metric, definition])

    notes_sheet = workbook.create_sheet("notes")
    notes_sheet.append(["field", "value"])
    notes_sheet.append(["difference_definition", "difference_b_minus_a = mean(group_b) - mean(group_a)"])
    notes_sheet.append(["records_used", "Cell-level records in example mode; independent image/ROI-level records in full mode."])
    notes_sheet.append(["inference", "This workbook is descriptive. Significance testing should use full-mode ROI-level records."])
    notes_sheet.append(["DAPI_length", "DAPI length is represented by dapi_axial_extent_50_fraction and, when calibrated, dapi_axial_extent_50_um."])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def plot_detected_cells(
    dataset: str,
    rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    """Plot detected cells for visual assessment.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        rows (list[dict[str, str]]): Text value specifying rows.
        output_dir (Path): Directory where generated resources are written.

    Example:
        >>> plot_detected_cells(
        ...     dataset="2d_time",
        ...     rows="rows",
        ...     output_dir=Path("path/to/resource"),
        ... )
    """
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
    """Return import common for the supplied inputs.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = _import_common()
    """
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
    """Prepare output for downstream processing.

    Args:
        output_root (Path): Directory used for output.
        overwrite (bool): Whether an existing output may be replaced.

    Raises:
        FileExistsError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> _prepare_output(output_root=Path("path/to/resource"), overwrite=True)
    """
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output folder is not empty: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def aggregate_roi_records(cell_records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return aggregate region of interest records for the supplied inputs.

    Args:
        cell_records (list[dict[str, object]]): Text value specifying cell records.

    Returns:
        list[dict[str, object]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = aggregate_roi_records(cell_records="cell_records")
    """
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
    """Return mean profile store for the supplied inputs.

    Args:
        store (dict[tuple[str, str, str], list[np.ndarray]]): Array containing store.

    Returns:
        dict[tuple[str, str], list[np.ndarray]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _mean_profile_store(store=image_array)
    """
    output: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for (_annotation_id, condition, channel), profiles in store.items():
        if profiles:
            output[(condition, channel)].append(np.nanmean(np.vstack(profiles), axis=0))
    return output


def _condition_maps_from_annotations(
    annotation_maps: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Return condition maps from annotations for the supplied inputs.

    Args:
        annotation_maps (dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]): Array containing annotation maps.

    Returns:
        dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _condition_maps_from_annotations(annotation_maps=image_array)
    """
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
    """Process dataset using the configured workflow.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        example (bool): Boolean flag controlling example.
        manifest (Path): Filesystem path used for manifest.
        pca_root (Path): Directory used for principal-component analysis result.
        normalized_root (Path): Directory used for normalized.
        all_mask_summary (Path | None): Filesystem path used for all mask summary.
        output_root (Path): Directory used for output.
        radial_bins (int): Numerical value controlling radial bins.
        pixel_size_um (float | None): Physical pixel size in micrometres.
        overwrite (bool): Whether an existing output may be replaced.

    Returns:
        dict[str, object]: Mapping containing the generated or resolved values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = process_dataset(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     example=True,
        ...     manifest=Path("path/to/resource"),
        ...     pca_root=Path("path/to/resource"),
        ...     normalized_root=Path("path/to/resource"),
        ...     all_mask_summary=Path("path/to/resource"),
        ...     output_root=Path("path/to/resource"),
        ...     radial_bins=1,
        ...     pixel_size_um=0.5,
        ...     overwrite=True,
        ... )
    """
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
                axial_metrics = axial_profile_metrics(axial)
                radial_metrics = radial_profile_metrics(radial)
                for metric_name, metric_value in axial_metrics.items():
                    channel_metrics[f"{channel_name.lower()}_{metric_name}"] = metric_value
                for metric_name, metric_value in radial_metrics.items():
                    channel_metrics[f"{channel_name.lower()}_{metric_name}"] = metric_value
                extent_fraction = axial_metrics.get("axial_extent_50_fraction", math.nan)
                channel_metrics[f"{channel_name.lower()}_axial_extent_50_pixels"] = (
                    float(extent_fraction) * length_px
                    if np.isfinite(extent_fraction)
                    else math.nan
                )
                if pixel_size_um is not None:
                    channel_metrics[f"{channel_name.lower()}_axial_extent_50_um"] = (
                        float(extent_fraction) * length_px * pixel_size_um
                        if np.isfinite(extent_fraction)
                        else math.nan
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
                channel_metrics["dapi_positive_area_fraction"] = float(
                    np.mean(dapi_corrected[cell_mask] > (3.0 * bg[0]["mad"]))
                )
                channel_metrics["wga_positive_area_fraction"] = float(
                    np.mean(wga_corrected[cell_mask] > (3.0 * bg[1]["mad"]))
                )
                overlap = channel_overlap_metrics(
                    wga_corrected,
                    dapi_corrected,
                    cell_mask,
                    first_positive_threshold=3.0 * bg[1]["mad"],
                    second_positive_threshold=3.0 * bg[0]["mad"],
                )
                channel_metrics["wga_dapi_overlap_cosine"] = overlap["overlap_cosine"]
                channel_metrics["wga_dapi_positive_overlap_fraction"] = overlap["positive_overlap_fraction"]
                channel_metrics["wga_dapi_positive_union_fraction"] = overlap["positive_union_fraction"]
                channel_metrics["wga_manders_in_dapi"] = overlap["manders_first_in_second"]
                channel_metrics["dapi_manders_in_wga"] = overlap["manders_second_in_first"]

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
    write_condition_metric_workbook(
        dataset,
        plot_records,
        profile_store_for_lines,
        radial_store_for_lines,
        spec.channel_names,
        output_root / "condition_difference_metrics.xlsx",
    )
    length_key = "length_um" if pixel_size_um is not None else "length_pixels"
    if example:
        length_label = (
            "PCA-derived cell length, Âµm"
            if pixel_size_um is not None
            else "PCA-derived cell length, pixels"
        )
    else:
        length_label = (
            "ROI mean PCA-derived cell length, Âµm"
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

    extent_unit_suffix = "um" if pixel_size_um is not None else "fraction"
    extent_ylabel = (
        "Axial signal extent at 50% peak, Âµm"
        if pixel_size_um is not None
        else "Axial signal extent at 50% peak, fraction of cell length"
    )
    if dataset == "2d_time":
        plot_2d_time_metric_over_time(
            plot_records,
            f"hada_axial_extent_50_{extent_unit_suffix}",
            extent_ylabel,
            "HADA axial signal extent over time",
            output_root,
            "hada_axial_signal_extent_over_time",
        )
    elif dataset == "2d_wga_dapi":
        boxplot_metric(
            plot_records,
            conditions,
            f"dapi_axial_extent_50_{extent_unit_suffix}",
            extent_ylabel,
            "DAPI axial signal length by condition",
            output_root,
            "dapi_axial_signal_length",
        )
        boxplot_metric(
            plot_records,
            conditions,
            f"wga_axial_extent_50_{extent_unit_suffix}",
            extent_ylabel,
            "WGA axial signal extent by condition",
            output_root,
            "wga_axial_signal_extent",
        )
    elif dataset == "3d_mip":
        extent_rows: list[dict[str, object]] = []
        for row in plot_records:
            for channel in spec.channel_names:
                extent_rows.append(
                    {
                        "condition": f"{channel}_{row['condition']}",
                        "extent": row.get(
                            f"{channel.lower()}_axial_extent_50_{extent_unit_suffix}",
                            math.nan,
                        ),
                    }
                )
        extent_conditions = [
            f"{channel}_{condition}"
            for channel in spec.channel_names
            for condition in conditions
        ]
        boxplot_metric(
            extent_rows,
            extent_conditions,
            "extent",
            extent_ylabel,
            "Three-channel axial signal extent",
            output_root,
            "3d_channel_axial_signal_extent",
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
    """Build parser from the supplied inputs.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """
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
    """Execute the command-line workflow and return its process exit status.

    Args:
        argv (Sequence[str] | None): Optional command-line argument sequence. When omitted, arguments are read from ``sys.argv``. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
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

