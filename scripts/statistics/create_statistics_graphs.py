"""
Provide command-line and programmatic utilities for create statistics graphs.

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

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from openpyxl import Workbook
from scipy import ndimage as ndi

SCRIPT_FILE = Path(__file__).resolve()
EPS = np.finfo(np.float64).eps
EXAMPLE_MODE = False
THESIS_HEATMAP_ROWS = 50
TIME_POINTS_2D = (5, 20, 40, 120)
MEDIA_2D = ("THY", "NHS")
THREE_D_CHANNEL_PAIRS = (("HADA", "NADA"), ("NADA", "TADA"), ("HADA", "TADA"))



def trapezoidal_integral(
    values: np.ndarray,
    *,
    x: np.ndarray | None = None,
    axis: int = -1,
) -> float | np.ndarray:
    """Integrate values using a NumPy-version-compatible trapezoidal rule.

    NumPy 2.0 introduced ``numpy.trapezoid``. NumPy 1.x provides the
    equivalent ``numpy.trapz`` implementation. This helper selects the
    available implementation without changing the numerical method.

    Args:
        values: Values to integrate.
        x: Optional coordinates associated with ``values``.
        axis: Array axis along which integration is performed.

    Returns:
        Scalar or array containing the trapezoidal numerical integral.

    Example:
        >>> y = np.asarray([0.0, 1.0, 0.0])
        >>> float(trapezoidal_integral(y, x=np.asarray([0.0, 0.5, 1.0])))
        0.5
    """
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(values, x=x, axis=axis)
    return np.trapz(values, x=x, axis=axis)


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
        condition_order=(
            "WT_THY_0min",
            "WT_NHS_0min",
            "DpspA_THY_0min",
            "DpspA_NHS_0min",
            "WT_THY_40min",
            "WT_NHS_40min",
            "DpspA_THY_40min",
            "DpspA_NHS_40min",
        ),
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
    """Infer a canonical condition and metadata from a sample identifier.

    This parser matches ``statistics_pipeline_common.infer_condition``. It
    resolves ``noNHS`` as the THY/control group in WGA-DAPI data and keeps
    genotype, time, acquisition batch, and filename labeling sequence separate
    for 3D MIP data.

    Args:
        dataset (str): Supported dataset identifier.
        sample_name (str): Sample name or combined annotation/sample path.

    Returns:
        tuple[str, dict[str, object]]: Canonical condition and parsed metadata.

    Raises:
        ValueError: If ``dataset`` is unsupported.

    Example:
        >>> infer_condition("2d_wga_dapi", "WT_noCSP_noNHS_ROI1_SIM")[0]
        'THY_noCSP'
        >>> infer_condition(
        ...     "3d_mip",
        ...     "20220318_HADA_NADA_TADA_40min/DpspA_NHS_NADA_HADA_TADA_40min_ROI1_SIM",
        ... )[0]
        'DpspA_NHS_40min'
    """
    upper = str(sample_name).upper()
    compact_tokens = set(re.split(r"[^A-Z0-9]+", upper))

    if dataset == "2d_time":
        medium = "THY" if "THY" in compact_tokens else "NHS" if "NHS" in compact_tokens else "UNKNOWN"
        match = re.search(r"(?<!\d)(5|20|40|120)\s*MIN", upper)
        time_min = int(match.group(1)) if match else -1
        metadata: dict[str, object] = {"medium": medium, "time_min": time_min}
        return f"{medium}_{time_min}min", metadata

    if dataset == "2d_wga_dapi":
        has_no_nhs = "NONHS" in compact_tokens or bool(re.search(r"(?:^|_)NO_?NHS(?:_|$)", upper))
        if "THY" in compact_tokens or has_no_nhs:
            medium = "THY"
        elif "NHS" in compact_tokens:
            medium = "NHS"
        else:
            medium = "UNKNOWN"

        no_csp = "NOCSP" in compact_tokens or bool(re.search(r"(?:^|_)NO_?CSP(?:_|$)", upper))
        csp = ("CSP" in compact_tokens) and not no_csp
        metadata = {
            "medium": medium,
            "csp": csp,
            "naming_no_nhs_control": has_no_nhs,
        }
        return f"{medium}_{'CSP' if csp else 'noCSP'}", metadata

    if dataset == "3d_mip":
        genotype = "DpspA" if "DPSPA" in compact_tokens else "WT" if "WT" in compact_tokens else "UNKNOWN"
        medium = "THY" if "THY" in compact_tokens else "NHS" if "NHS" in compact_tokens else "UNKNOWN"
        time_match = re.search(r"(?<!\d)(0|40)\s*MIN", upper)
        time_min = int(time_match.group(1)) if time_match else -1
        date_match = re.search(r"(?<!\d)(20\d{6})(?!\d)", upper)
        acquisition_date = date_match.group(1) if date_match else "UNKNOWN"
        if "NADA_HADA_TADA" in upper:
            label_sequence = "NADA_HADA_TADA"
        elif "HADA_NADA_TADA" in upper:
            label_sequence = "HADA_NADA_TADA"
        else:
            label_sequence = "UNKNOWN"
        metadata = {
            "genotype": genotype,
            "medium": medium,
            "time_min": time_min,
            "acquisition_date": acquisition_date,
            "batch_id": acquisition_date,
            "label_sequence": label_sequence,
        }
        return f"{genotype}_{medium}_{time_min}min", metadata

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
        "axial_auc": float(trapezoidal_integral(values, x=x)),
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
        "radial_auc": float(trapezoidal_integral(values, x=x)),
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


def profile_relationship_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    """Describe similarity between two normalized one-dimensional profiles.

    The profiles are compared within the same segmented cell. Correlation
    captures similarity of profile shape, whereas RMSE and mean absolute
    difference quantify the magnitude of the shape difference.

    Args:
        first: First normalized axial or radial profile.
        second: Second normalized axial or radial profile.

    Returns:
        Mapping containing Pearson correlation, RMSE, mean absolute
        difference, and the signed integral of ``second - first``.

    Example:
        >>> result = profile_relationship_metrics(
        ...     np.asarray([0.0, 1.0, 0.0]),
        ...     np.asarray([0.0, 0.8, 0.2]),
        ... )
        >>> "correlation" in result
        True
    """
    first_values = np.asarray(first, dtype=np.float64).reshape(-1)
    second_values = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_values.size != second_values.size:
        raise ValueError("Profiles must have the same length.")
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    first_values = first_values[finite]
    second_values = second_values[finite]
    if first_values.size == 0:
        return {
            "correlation": math.nan,
            "rmse": math.nan,
            "mean_absolute_difference": math.nan,
            "signed_auc_difference": math.nan,
        }
    difference = second_values - first_values
    if (
        first_values.size >= 3
        and np.std(first_values) > EPS
        and np.std(second_values) > EPS
    ):
        correlation = float(np.corrcoef(first_values, second_values)[0, 1])
    else:
        correlation = math.nan
    coordinate = np.linspace(0.0, 1.0, first_values.size)
    return {
        "correlation": correlation,
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "signed_auc_difference": float(
            trapezoidal_integral(difference, x=coordinate)
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
    """Return a concise publication label for a canonical condition.

    Args:
        condition (str): Canonical condition identifier.

    Returns:
        str: Human-readable condition label.

    Example:
        >>> condition_display("DpspA_NHS_40min")
        'ΔpspA, NHS, 40 min'
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
    match = re.fullmatch(r"(WT|DpspA)_(THY|NHS)_(\d+)min", condition)
    if match:
        genotype = "ΔpspA" if match.group(1) == "DpspA" else "WT"
        return f"{genotype}, {match.group(2)}, {match.group(3)} min"
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
    tick_labels = [
        condition_display(condition).replace(" + ", " +\n")
        for condition in conditions
    ]
    ax.set_xticks(np.arange(1, len(conditions) + 1), tick_labels)
    if len(conditions) > 4:
        ax.tick_params(axis="x", labelrotation=28)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
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


def _sorted_profile_matrix(profiles: Sequence[np.ndarray]) -> np.ndarray:
    """Stack axial profiles and sort cells by fluorescence-peak position.

    Sorting gives the heatmap a reproducible biological order from profiles
    peaking near Pole 1 to profiles peaking near Pole 2.

    Args:
        profiles (Sequence[np.ndarray]): One-dimensional normalized axial
            fluorescence profiles with a common profile length.

    Returns:
        np.ndarray: Two-dimensional matrix with one sorted cell profile per row.

    Raises:
        ValueError: If no profiles are supplied or profile lengths differ.

    Example:
        >>> matrix = _sorted_profile_matrix([
        ...     np.array([0.0, 1.0, 0.2]),
        ...     np.array([0.1, 0.4, 1.0]),
        ... ])
        >>> matrix.shape
        (2, 3)
    """
    if not profiles:
        raise ValueError("At least one profile is required for a heatmap.")
    matrix = np.vstack([np.asarray(profile, dtype=np.float64) for profile in profiles])
    safe_matrix = np.nan_to_num(matrix, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    peak_positions = np.argmax(safe_matrix, axis=1)
    order = np.argsort(peak_positions, kind="stable")
    return matrix[order]


def _binned_population_matrix(
    sorted_matrix: np.ndarray,
    display_rows: int = THESIS_HEATMAP_ROWS,
) -> tuple[np.ndarray, int, str]:
    """Compress a sorted cell population into a fixed-height thesis heatmap.

    If the condition contains more cells than ``display_rows``, cells are split
    into equal-count groups and the median profile of every group is displayed.
    If the condition contains fewer cells, each real profile is retained and
    the unused rows are filled with ``NaN`` so that no artificial cells are
    created.

    Args:
        sorted_matrix (np.ndarray): Two-dimensional matrix containing one
            peak-position-sorted profile per row.
        display_rows (int): Fixed number of rows in each main heatmap panel.
            Defaults to ``THESIS_HEATMAP_ROWS`` (50).

    Returns:
        tuple[np.ndarray, int, str]: Display matrix, number of populated display
        rows, and a short description of the visualization method.

    Raises:
        ValueError: If the matrix is not two-dimensional or ``display_rows`` is
            smaller than one.

    Example:
        >>> source = np.arange(150, dtype=float).reshape(50, 3)
        >>> display, populated, method = _binned_population_matrix(source, 10)
        >>> display.shape, populated, method
        ((10, 3), 10, 'equal-count median bins')
    """
    matrix = np.asarray(sorted_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("The profile matrix must be two-dimensional.")
    if display_rows < 1:
        raise ValueError("display_rows must be at least 1.")

    n_cells, n_positions = matrix.shape
    if n_cells <= display_rows:
        display = np.full((display_rows, n_positions), np.nan, dtype=np.float64)
        display[:n_cells] = matrix
        return display, n_cells, "individual profiles"

    groups = np.array_split(matrix, display_rows, axis=0)
    display = np.vstack([np.nanmedian(group, axis=0) for group in groups])
    return display, display_rows, "equal-count median bins"


def _save_full_cell_heatmap(
    matrix: np.ndarray,
    *,
    channel: str,
    condition: str,
    output_dir: Path,
    cmap,
) -> None:
    """Save one all-cell axial heatmap for use as an appendix figure.

    Args:
        matrix (np.ndarray): Peak-position-sorted matrix with one row per cell.
        channel (str): Fluorescence-channel name shown in the title and file name.
        condition (str): Experimental condition represented by the matrix.
        output_dir (Path): Main graph output directory. Appendix figures are
            written to its ``appendix_heatmaps`` subdirectory.
        cmap: Matplotlib colour map used for normalized fluorescence values.

    Example:
        >>> _save_full_cell_heatmap(
        ...     np.random.default_rng(1).random((12, 90)),
        ...     channel="HADA",
        ...     condition="THY_5min",
        ...     output_dir=Path("results/statistics/graphs"),
        ...     cmap=plt.get_cmap("viridis"),
        ... )
    """
    n_cells, profile_length = matrix.shape
    figure_height = min(18.0, max(4.8, 2.8 + 0.0045 * n_cells))
    fig, ax = plt.subplots(figsize=(8.2, figure_height), constrained_layout=True)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )
    ax.set_xticks(
        [0, (profile_length - 1) / 2, profile_length - 1],
        ["Pole 1", "Midcell", "Pole 2"],
    )
    if n_cells == 1:
        ax.set_yticks([0], ["1"])
    else:
        ax.set_yticks([0, n_cells - 1], ["1", f"{n_cells:,}"])
    ax.set_xlabel("Normalized axial coordinate")
    ax.set_ylabel("Individual cells, ordered by peak position")
    ax.set_title(
        f"{channel} full-cell axial heatmap: {condition_display(condition)}\n"
        f"n = {n_cells:,} cells"
    )
    fig.colorbar(image, ax=ax, label="Within-cell normalized fluorescence")
    add_note(fig)
    stem = f"axial_heatmap_full_cells_{channel.lower()}_{condition.lower()}"
    save_figure(fig, output_dir / "appendix_heatmaps", stem)


def _plot_binned_heatmap_grid(
    *,
    profile_store: dict[tuple[str, str], list[np.ndarray]],
    channel: str,
    row_definitions: Sequence[tuple[str, str]],
    column_definitions: Sequence[tuple[str, str]],
    condition_builder,
    output_dir: Path,
    dataset: str,
    stem: str,
    title: str,
    cmap,
) -> None:
    """Create one fixed-height binned heatmap grid and appendix heatmaps.

    Every condition contributes at most ``THESIS_HEATMAP_ROWS`` displayed rows.
    Larger populations are represented by equal-count median bins. Smaller
    populations retain every individual profile and use blank padding. Full-cell
    heatmaps are saved separately for appendix use.

    Args:
        profile_store: Cell-level axial profiles indexed by condition and channel.
        channel: Channel represented by the figure.
        row_definitions: ``(value, display_label)`` pairs for grid rows.
        column_definitions: ``(value, display_label)`` pairs for grid columns.
        condition_builder: Callable converting row and column values to a condition.
        output_dir: Graph output directory.
        dataset: Dataset identifier written to the summary CSV.
        stem: Output filename stem.
        title: Figure title.
        cmap: Matplotlib colour map.

    Example:
        >>> _plot_binned_heatmap_grid(
        ...     profile_store={("THY_5min", "HADA"): [np.linspace(0, 1, 90)]},
        ...     channel="HADA",
        ...     row_definitions=(("THY", "THY"),),
        ...     column_definitions=(("5", "5 min"),),
        ...     condition_builder=lambda row, column: f"{row}_{column}min",
        ...     output_dir=Path("results/statistics/graphs"),
        ...     dataset="2d_time",
        ...     stem="axial_heatmap_hada",
        ...     title="HADA axial fluorescence",
        ...     cmap=plt.get_cmap("viridis"),
        ... )
    """
    profile_length = next(
        (
            int(np.asarray(profile).size)
            for row_value, _ in row_definitions
            for column_value, _ in column_definitions
            for profile in profile_store.get(
                (condition_builder(row_value, column_value), channel), []
            )
        ),
        90,
    )
    n_rows = len(row_definitions)
    n_columns = len(column_definitions)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(max(8.6, 3.75 * n_columns), max(5.8, 3.4 * n_rows)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    rendered = []
    summary_rows: list[dict[str, object]] = []

    for row_index, (row_value, row_label) in enumerate(row_definitions):
        for column_index, (column_value, column_label) in enumerate(column_definitions):
            axis = axes[row_index, column_index]
            condition = condition_builder(row_value, column_value)
            profiles = profile_store.get((condition, channel), [])
            if not profiles:
                axis.set_facecolor("0.92")
                axis.text(
                    0.5, 0.5, "No cells", transform=axis.transAxes,
                    ha="center", va="center", fontsize=9,
                )
                n_cells = 0
                populated_rows = 0
                method = "no profiles"
            else:
                sorted_matrix = _sorted_profile_matrix(profiles)
                n_cells = int(sorted_matrix.shape[0])
                display_matrix, populated_rows, method = _binned_population_matrix(
                    sorted_matrix, THESIS_HEATMAP_ROWS
                )
                image = axis.imshow(
                    display_matrix,
                    aspect="auto",
                    interpolation="nearest",
                    vmin=0.0,
                    vmax=1.0,
                    cmap=cmap,
                )
                rendered.append(image)
                _save_full_cell_heatmap(
                    sorted_matrix,
                    channel=channel,
                    condition=condition,
                    output_dir=output_dir,
                    cmap=cmap,
                )

            method_label = (
                f"{populated_rows} median bins"
                if method == "equal-count median bins"
                else f"{populated_rows} individual profiles"
                if method == "individual profiles"
                else "no profiles"
            )
            axis.text(
                0.02,
                0.98,
                f"n = {n_cells:,} cells\n{method_label}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.82,
                },
            )
            axis.set_yticks([])
            axis.set_xticks(
                [0, (profile_length - 1) / 2, profile_length - 1],
                ["Pole 1", "Midcell", "Pole 2"],
            )
            if row_index == 0:
                axis.set_title(column_label)
                if n_rows > 1:
                    axis.tick_params(labelbottom=False)
            if column_index == 0:
                axis.set_ylabel(f"{row_label}\nPopulation rows")
            if row_index == n_rows - 1:
                axis.set_xlabel("Normalized axial coordinate")

            summary_rows.append(
                {
                    "dataset": dataset,
                    "channel": channel,
                    "condition": condition,
                    "row_group": row_value,
                    "column_group": column_value,
                    "n_cells": n_cells,
                    "display_rows": THESIS_HEATMAP_ROWS,
                    "populated_display_rows": populated_rows,
                    "visualization_method": method,
                    "appendix_full_cell_heatmap": (
                        f"appendix_heatmaps/axial_heatmap_full_cells_"
                        f"{channel.lower()}_{condition.lower()}.png"
                        if n_cells > 0
                        else ""
                    ),
                }
            )

    if rendered:
        fig.colorbar(
            rendered[0],
            ax=axes.ravel().tolist(),
            label="Within-cell normalized fluorescence",
            shrink=0.88,
        )
    fig.suptitle(
        f"{title}: binned population heatmaps\n"
        + (
            f"All PCA-selected cells retained; maximum {THESIS_HEATMAP_ROWS} "
            "display rows per condition"
            if dataset == "3d_mip"
            else f"All cells retained; maximum {THESIS_HEATMAP_ROWS} "
            "display rows per condition"
        )
    )
    add_note(fig)
    save_figure(fig, output_dir, stem)
    write_csv(output_dir / f"{stem}_population_summary.csv", summary_rows)


def plot_heatmaps(
    profile_store: dict[tuple[str, str], list[np.ndarray]],
    conditions: list[str],
    channel_names: tuple[str, ...],
    output_dir: Path,
    dataset: str,
) -> None:
    """Create binned thesis heatmaps and full-cell appendix heatmaps for 2D data.

    ``2d_time`` uses a 2 × 4 layout with THY/NHS rows and 5, 20, 40,
    and 120 minute columns. ``2d_wga_dapi`` uses a 2 × 2 layout with THY/NHS
    rows and no-CSP/CSP columns. A separate main figure is generated for every
    available channel. Every panel is capped at 50 displayed profiles, while
    every detected cell remains represented either directly or through an
    equal-count median bin. Full-cell condition heatmaps are exported for the
    appendix. ``3d_mip`` uses a 2 × 4 layout with genotype rows and
    medium/time columns. The 3D heatmaps contain all PCA-selected cells,
    while the displayed population is capped at 50 rows per condition.

    Args:
        profile_store: Cell-level normalized axial profiles.
        conditions: Conditions detected in the processed dataset.
        channel_names: Channel names to plot independently.
        output_dir: Directory for graph outputs.
        dataset: Dataset identifier.

    Example:
        >>> plot_heatmaps(
        ...     {("THY_5min", "HADA"): [np.linspace(0, 1, 90)]},
        ...     ["THY_5min"],
        ...     ("HADA",),
        ...     Path("results/statistics/graphs/2d_time/filtered_unet"),
        ...     "2d_time",
        ... )
    """
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.92")

    for channel in channel_names:
        if dataset == "2d_time":
            _plot_binned_heatmap_grid(
                profile_store=profile_store,
                channel=channel,
                row_definitions=(("THY", "THY"), ("NHS", "NHS")),
                column_definitions=tuple((str(time), f"{time} min") for time in TIME_POINTS_2D),
                condition_builder=lambda medium, time: f"{medium}_{time}min",
                output_dir=output_dir,
                dataset=dataset,
                stem=f"axial_heatmap_{channel.lower()}",
                title=f"{channel} axial fluorescence",
                cmap=cmap,
            )
        elif dataset == "2d_wga_dapi":
            _plot_binned_heatmap_grid(
                profile_store=profile_store,
                channel=channel,
                row_definitions=(("THY", "THY"), ("NHS", "NHS")),
                column_definitions=(("noCSP", "No CSP"), ("CSP", "CSP")),
                condition_builder=lambda medium, csp: f"{medium}_{csp}",
                output_dir=output_dir,
                dataset=dataset,
                stem=f"axial_heatmap_{channel.lower()}",
                title=f"{channel} axial fluorescence",
                cmap=cmap,
            )
        elif dataset == "3d_mip":
            _plot_binned_heatmap_grid(
                profile_store=profile_store,
                channel=channel,
                row_definitions=(("WT", "WT"), ("DpspA", "ΔpspA")),
                column_definitions=(
                    ("THY_0", "THY, 0 min"),
                    ("NHS_0", "NHS, 0 min"),
                    ("THY_40", "THY, 40 min"),
                    ("NHS_40", "NHS, 40 min"),
                ),
                condition_builder=lambda genotype, medium_time: (
                    f"{genotype}_{medium_time}min"
                ),
                output_dir=output_dir,
                dataset=dataset,
                stem=f"axial_heatmap_{channel.lower()}",
                title=f"{channel} axial fluorescence in 3D MIPs",
                cmap=cmap,
            )

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


def plot_3d_pairwise_radial_profiles(
    radial_store: dict[tuple[str, str], list[np.ndarray]],
    conditions: list[str],
    output_dir: Path,
) -> None:
    """Plot all three pairwise radial channel comparisons for each 3D condition.

    Each panel compares ROI-level mean radial profiles from the same biological
    condition. The shaded band is one standard deviation across ROI profiles.

    Args:
        radial_store: ROI-level radial profiles indexed by condition and channel.
        conditions: Ordered 3D biological conditions.
        output_dir: Directory where figures are written.

    Example:
        >>> plot_3d_pairwise_radial_profiles({}, [], Path("results"))
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", [f"C{i}" for i in range(10)]
    )
    for condition in conditions:
        fig, axes = plt.subplots(
            1,
            len(THREE_D_CHANNEL_PAIRS),
            figsize=(15.0, 4.6),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        any_panel = False
        for axis, (first, second) in zip(axes, THREE_D_CHANNEL_PAIRS):
            panel_plotted = False
            for channel_index, channel in enumerate((first, second)):
                profiles = radial_store.get((condition, channel), [])
                if not profiles:
                    continue
                panel_plotted = True
                any_panel = True
                matrix = np.vstack(profiles)
                mean, low, high = mean_and_band(matrix)
                coordinate = np.linspace(0.0, 1.0, mean.size)
                color = cycle[channel_index % len(cycle)]
                axis.plot(
                    coordinate,
                    mean,
                    color=color,
                    linewidth=1.8,
                    label=f"{channel} (n={matrix.shape[0]} ROIs)",
                )
                axis.fill_between(coordinate, low, high, color=color, alpha=0.12)
            if not panel_plotted:
                axis.text(0.5, 0.5, "No profiles", ha="center", va="center")
            axis.set_title(f"{first} vs {second}")
            axis.set_xlabel("Normalized radial distance")
            axis.set_xticks([0.0, 0.5, 1.0], ["Centre", "0.5", "Boundary"])
            axis.grid(alpha=0.2)
            if panel_plotted:
                axis.legend(fontsize=8)
        axes[0].set_ylabel("Normalized fluorescence intensity")
        if not any_panel:
            plt.close(fig)
            continue
        fig.suptitle(
            "3D MIP pairwise radial fluorescence profiles: "
            f"{condition_display(condition)}"
        )
        add_note(fig)
        save_figure(
            fig,
            output_dir,
            f"radial_profile_pairwise_{condition.lower()}",
        )


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


def plot_3d_pairwise_overlap_maps(
    map_store: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    conditions: list[str],
    output_dir: Path,
) -> None:
    """Create standardized spatial-overlap maps for each 3D channel pair.

    The first channel is shown in magenta, the second in cyan, and coincident
    normalized signal appears white. A separate multi-condition figure is
    generated for HADA-NADA, NADA-TADA, and HADA-TADA.

    Args:
        map_store: Condition-level standardized maps.
        conditions: Ordered biological conditions.
        output_dir: Directory where figures are written.

    Example:
        >>> plot_3d_pairwise_overlap_maps({}, [], Path("results"))
    """
    for first, second in THREE_D_CHANNEL_PAIRS:
        available = [
            condition
            for condition in conditions
            if (condition, first) in map_store and (condition, second) in map_store
        ]
        if not available:
            continue
        ncols = 2 if len(available) > 1 else 1
        nrows = int(math.ceil(len(available) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.2 * ncols, 3.5 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.reshape(-1)
        for axis, condition in zip(axes, available):
            first_sum, first_weight = map_store[(condition, first)]
            second_sum, second_weight = map_store[(condition, second)]
            first_map = np.divide(
                first_sum,
                np.maximum(first_weight, EPS),
                out=np.zeros_like(first_sum, dtype=np.float64),
                where=first_weight > 0,
            )
            second_map = np.divide(
                second_sum,
                np.maximum(second_weight, EPS),
                out=np.zeros_like(second_sum, dtype=np.float64),
                where=second_weight > 0,
            )
            first_map = np.clip(first_map, 0.0, 1.0)
            second_map = np.clip(second_map, 0.0, 1.0)
            rgb = np.zeros((*first_map.shape, 3), dtype=np.float64)
            rgb[..., 0] = first_map
            rgb[..., 1] = second_map
            rgb[..., 2] = np.maximum(first_map, second_map)
            axis.imshow(rgb, interpolation="nearest")
            axis.axvline(
                (first_map.shape[1] - 1) / 2,
                linestyle="--",
                linewidth=0.8,
                color="yellow",
            )
            axis.set_xticks(
                [0, (first_map.shape[1] - 1) / 2, first_map.shape[1] - 1],
                ["Pole 1", "Midcell", "Pole 2"],
            )
            axis.set_yticks([])
            axis.set_title(condition_display(condition))
        for axis in axes[len(available):]:
            axis.axis("off")
        fig.suptitle(f"Standardized {first}-{second} spatial-overlap maps")
        fig.text(
            0.5,
            0.01,
            f"{first}: magenta   {second}: cyan   spatial overlap: white",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        add_note(fig)
        save_figure(
            fig,
            output_dir,
            f"standardized_overlap_map_{first.lower()}_{second.lower()}",
        )


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
        "naming_no_nhs_control",
        "genotype",
        "acquisition_date",
        "batch_id",
        "label_sequence",
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
        comparisons: list[tuple[str, str, list[str], str, list[str]]] = []
        for time in (0, 40):
            comparisons.extend(
                [
                    (
                        f"WT medium effect, {time} min",
                        "WT, THY",
                        [f"WT_THY_{time}min"],
                        "WT, NHS",
                        [f"WT_NHS_{time}min"],
                    ),
                    (
                        f"ΔpspA medium effect, {time} min",
                        "ΔpspA, THY",
                        [f"DpspA_THY_{time}min"],
                        "ΔpspA, NHS",
                        [f"DpspA_NHS_{time}min"],
                    ),
                    (
                        f"THY genotype effect, {time} min",
                        "WT, THY",
                        [f"WT_THY_{time}min"],
                        "ΔpspA, THY",
                        [f"DpspA_THY_{time}min"],
                    ),
                    (
                        f"NHS genotype effect, {time} min",
                        "WT, NHS",
                        [f"WT_NHS_{time}min"],
                        "ΔpspA, NHS",
                        [f"DpspA_NHS_{time}min"],
                    ),
                ]
            )
        return comparisons
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
        ("*_pearson", "Within-cell Pearson correlation of two background-corrected fluorescence channels."),
        ("*_overlap_cosine", "Cosine overlap coefficient for a fluorescence-channel pair."),
        ("*_positive_overlap_fraction", "Fraction of cell pixels positive in both channels of a pair."),
        ("*_manders_in_*", "Directional fraction of one channel intensity located in pixels positive for the paired channel."),
        ("*_axial_profile_correlation", "Within-cell correlation between two normalized axial profile shapes."),
        ("*_radial_profile_correlation", "Within-cell correlation between two normalized radial profile shapes."),
        ("*_profile_rmse", "Root mean square difference between two within-cell normalized profile shapes."),
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
    if dataset == "3d_mip":
        notes_sheet.append(["3D_channel_pairs", "Pairwise HADA-NADA, NADA-TADA, and HADA-TADA metrics are calculated within the same segmented cells and summarized per ROI."])
        notes_sheet.append(["3D_intensity_caution", "Raw intensity magnitudes are compared across biological conditions within a channel. Direct brightness comparison between different dyes requires matched acquisition settings and calibration."])
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
        parse_identifier = f"{row.get('annotation_id', '')}/{sample}"
        condition, metadata = infer_condition(dataset, parse_identifier)
        parsed.append(
            {
                "sample": sample,
                "condition": condition,
                "medium": metadata.get("medium", "UNKNOWN"),
                "time_min": metadata.get("time_min", -1),
                "csp": bool(metadata.get("csp", False)),
                "genotype": metadata.get("genotype", ""),
                "acquisition_date": metadata.get("acquisition_date", ""),
                "batch_id": metadata.get("batch_id", ""),
                "label_sequence": metadata.get("label_sequence", ""),
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
    if len(conditions) > 4:
        ax.tick_params(axis="x", labelrotation=28)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    save_figure(fig, output_dir, "detected_cells_all_images")

    if dataset == "3d_mip":
        batches = sorted(
            {str(row.get("acquisition_date", "")) for row in parsed}
            - {"", "UNKNOWN"}
        )
        if batches:
            batch_groups = [
                np.asarray(
                    [
                        row["n_cells"]
                        for row in parsed
                        if str(row.get("acquisition_date", "")) == batch
                    ],
                    dtype=float,
                )
                for batch in batches
            ]
            batch_fig, batch_ax = plt.subplots(
                figsize=(max(6.5, 1.5 * len(batches)), 4.5),
                constrained_layout=True,
            )
            batch_ax.boxplot(
                batch_groups,
                positions=np.arange(1, len(batches) + 1),
                widths=0.55,
                showfliers=False,
            )
            jittered_scatter(
                batch_ax, np.arange(1, len(batches) + 1), batch_groups
            )
            batch_ax.set_xticks(
                np.arange(1, len(batches) + 1),
                [f"{batch[:4]}-{batch[4:6]}-{batch[6:]}" for batch in batches],
            )
            batch_ax.set_xlabel("Acquisition date")
            batch_ax.set_ylabel("Detected cells per image")
            batch_ax.set_title("Detected cells by 3D acquisition batch")
            batch_ax.grid(axis="y", alpha=0.25)
            save_figure(
                batch_fig, output_dir, "detected_cells_by_acquisition_batch"
            )

            batch_rows: list[dict[str, object]] = []
            for batch, values in zip(batches, batch_groups):
                batch_rows.append(
                    {
                        "acquisition_date": batch,
                        "n_images": int(values.size),
                        "total_cells": int(np.sum(values)),
                        "mean_cells_per_image": float(np.mean(values)),
                        "sd_cells_per_image": (
                            float(np.std(values, ddof=1))
                            if values.size > 1
                            else 0.0
                        ),
                        "min_cells_per_image": int(np.min(values)),
                        "max_cells_per_image": int(np.max(values)),
                    }
                )
            write_csv(output_dir / "detected_cells_by_acquisition_batch.csv", batch_rows)


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
            "naming_no_nhs_control": first.get("naming_no_nhs_control", ""),
            "genotype": first.get("genotype", ""),
            "acquisition_date": first.get("acquisition_date", ""),
            "batch_id": first.get("batch_id", ""),
            "label_sequence": first.get("label_sequence", ""),
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
        parse_identifier = f"{annotation_id}/{sample_name}"
        condition, sample_metadata = infer_condition(dataset, parse_identifier)
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
            axial_by_channel: dict[str, np.ndarray] = {}
            radial_by_channel: dict[str, np.ndarray] = {}
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
                if dataset == "3d_mip":
                    channel_metrics[
                        f"{channel_name.lower()}_positive_area_fraction"
                    ] = float(
                        np.mean(
                            corrected[cell_mask] > (3.0 * bg[channel_index]["mad"])
                        )
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

                axial_by_channel[channel_name] = axial
                radial_by_channel[channel_name] = radial
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

            if dataset == "3d_mip":
                channel_index = {
                    channel: index
                    for index, channel in enumerate(spec.channel_names)
                }
                for first, second in THREE_D_CHANNEL_PAIRS:
                    first_index = channel_index[first]
                    second_index = channel_index[second]
                    first_corrected = corrected_channels[first_index]
                    second_corrected = corrected_channels[second_index]
                    pair_prefix = f"{first.lower()}_{second.lower()}"
                    channel_metrics[f"{pair_prefix}_pearson"] = pearson_inside(
                        first_corrected, second_corrected, cell_mask
                    )
                    overlap = channel_overlap_metrics(
                        first_corrected,
                        second_corrected,
                        cell_mask,
                        first_positive_threshold=3.0 * bg[first_index]["mad"],
                        second_positive_threshold=3.0 * bg[second_index]["mad"],
                    )
                    channel_metrics[f"{pair_prefix}_overlap_cosine"] = overlap[
                        "overlap_cosine"
                    ]
                    channel_metrics[
                        f"{pair_prefix}_positive_overlap_fraction"
                    ] = overlap["positive_overlap_fraction"]
                    channel_metrics[
                        f"{pair_prefix}_positive_union_fraction"
                    ] = overlap["positive_union_fraction"]
                    channel_metrics[
                        f"{first.lower()}_manders_in_{second.lower()}"
                    ] = overlap["manders_first_in_second"]
                    channel_metrics[
                        f"{second.lower()}_manders_in_{first.lower()}"
                    ] = overlap["manders_second_in_first"]

                    axial_relationship = profile_relationship_metrics(
                        axial_by_channel[first], axial_by_channel[second]
                    )
                    radial_relationship = profile_relationship_metrics(
                        radial_by_channel[first], radial_by_channel[second]
                    )
                    for metric_name, metric_value in axial_relationship.items():
                        channel_metrics[
                            f"{pair_prefix}_axial_profile_{metric_name}"
                        ] = metric_value
                    for metric_name, metric_value in radial_relationship.items():
                        channel_metrics[
                            f"{pair_prefix}_radial_profile_{metric_name}"
                        ] = metric_value

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

    paired_roi_archive: dict[str, np.ndarray] = {}
    roi_profile_index_rows: list[dict[str, object]] = []
    annotation_ids = sorted(
        {annotation_id for annotation_id, _condition, _channel in annotation_profile_store}
    )
    for roi_index, annotation_id in enumerate(annotation_ids):
        roi_id = f"roi_{roi_index:04d}"
        matching_keys = [
            key for key in annotation_profile_store if key[0] == annotation_id
        ]
        if not matching_keys:
            continue
        condition = matching_keys[0][1]
        roi_profile_index_rows.append(
            {
                "roi_id": roi_id,
                "annotation_id": annotation_id,
                "condition": condition,
            }
        )
        for channel in spec.channel_names:
            axial_profiles = annotation_profile_store.get(
                (annotation_id, condition, channel), []
            )
            radial_profiles = annotation_radial_store.get(
                (annotation_id, condition, channel), []
            )
            if axial_profiles:
                paired_roi_archive[
                    f"{roi_id}|{condition}|{channel}|axial"
                ] = np.nanmean(np.vstack(axial_profiles), axis=0).astype(np.float32)
            if radial_profiles:
                paired_roi_archive[
                    f"{roi_id}|{condition}|{channel}|radial"
                ] = np.nanmean(np.vstack(radial_profiles), axis=0).astype(np.float32)
    np.savez_compressed(
        output_root / "roi_profiles_by_annotation.npz", **paired_roi_archive
    )
    write_csv(output_root / "roi_profile_index.csv", roi_profile_index_rows)

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

    if dataset == "3d_mip":
        width_key = "width_um" if pixel_size_um is not None else "width_pixels"
        area_key = "area_um2" if pixel_size_um is not None else "area_pixels"
        width_label = (
            "ROI mean PCA-derived cell width, µm"
            if pixel_size_um is not None and not example
            else "PCA-derived cell width, µm"
            if pixel_size_um is not None
            else "ROI mean PCA-derived cell width, pixels"
            if not example
            else "PCA-derived cell width, pixels"
        )
        area_label = (
            "ROI mean cell area, µm²"
            if pixel_size_um is not None and not example
            else "Cell area, µm²"
            if pixel_size_um is not None
            else "ROI mean cell area, pixels²"
            if not example
            else "Cell area, pixels²"
        )
        boxplot_metric(
            plot_records,
            conditions,
            width_key,
            width_label,
            "Cell width by condition",
            output_root,
            "cell_width",
        )
        boxplot_metric(
            plot_records,
            conditions,
            area_key,
            area_label,
            "Cell area by condition",
            output_root,
            "cell_area",
        )
        for channel in spec.channel_names:
            channel_lower = channel.lower()
            scalar_specs = (
                (
                    f"{channel_lower}_mean_intensity",
                    f"ROI mean background-corrected {channel} intensity",
                    f"{channel} mean cellular intensity by condition",
                    f"{channel_lower}_mean_intensity",
                ),
                (
                    f"{channel_lower}_integrated_intensity",
                    f"ROI mean integrated background-corrected {channel} intensity",
                    f"{channel} integrated cellular intensity by condition",
                    f"{channel_lower}_integrated_intensity",
                ),
                (
                    f"{channel_lower}_positive_area_fraction",
                    f"ROI mean {channel}-positive cell-area fraction",
                    f"{channel} positive area fraction by condition",
                    f"{channel_lower}_positive_area_fraction",
                ),
                (
                    f"{channel_lower}_radial_centroid",
                    f"ROI mean {channel} radial centroid (centre 0, boundary 1)",
                    f"{channel} radial centroid by condition",
                    f"{channel_lower}_radial_centroid",
                ),
                (
                    f"{channel_lower}_radial_auc",
                    f"ROI mean normalized {channel} radial-profile AUC",
                    f"{channel} radial-profile area by condition",
                    f"{channel_lower}_radial_auc",
                ),
            )
            for key, ylabel, title, stem in scalar_specs:
                boxplot_metric(
                    plot_records,
                    conditions,
                    key,
                    ylabel,
                    title,
                    output_root,
                    stem,
                )

        for first, second in THREE_D_CHANNEL_PAIRS:
            pair_prefix = f"{first.lower()}_{second.lower()}"
            pair_specs = (
                (
                    f"{pair_prefix}_pearson",
                    "ROI mean within-cell Pearson correlation",
                    f"{first}-{second} spatial intensity correlation",
                    f"{pair_prefix}_spatial_correlation",
                ),
                (
                    f"{pair_prefix}_overlap_cosine",
                    "ROI mean cosine overlap coefficient",
                    f"{first}-{second} intensity overlap",
                    f"{pair_prefix}_overlap_cosine",
                ),
                (
                    f"{pair_prefix}_positive_overlap_fraction",
                    "ROI mean double-positive cell-area fraction",
                    f"{first}-{second} positive spatial overlap",
                    f"{pair_prefix}_positive_overlap",
                ),
                (
                    f"{pair_prefix}_axial_profile_correlation",
                    "ROI mean within-cell axial-profile correlation",
                    f"{first}-{second} axial profile similarity",
                    f"{pair_prefix}_axial_profile_similarity",
                ),
                (
                    f"{pair_prefix}_radial_profile_correlation",
                    "ROI mean within-cell radial-profile correlation",
                    f"{first}-{second} radial profile similarity",
                    f"{pair_prefix}_radial_profile_similarity",
                ),
                (
                    f"{first.lower()}_manders_in_{second.lower()}",
                    f"Fraction of {first} intensity in {second}-positive pixels",
                    f"{first} signal located in {second}-positive regions",
                    f"{first.lower()}_manders_in_{second.lower()}",
                ),
                (
                    f"{second.lower()}_manders_in_{first.lower()}",
                    f"Fraction of {second} intensity in {first}-positive pixels",
                    f"{second} signal located in {first}-positive regions",
                    f"{second.lower()}_manders_in_{first.lower()}",
                ),
            )
            for key, ylabel, title, stem in pair_specs:
                boxplot_metric(
                    plot_records,
                    conditions,
                    key,
                    ylabel,
                    title,
                    output_root,
                    stem,
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
        for channel in spec.channel_names:
            boxplot_metric(
                plot_records,
                conditions,
                f"{channel.lower()}_homogeneity",
                f"{prefix}{channel} boundary homogeneity index",
                f"{channel} boundary homogeneity by condition",
                output_root,
                f"{channel.lower()}_boundary_homogeneity",
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
        for channel in spec.channel_names:
            boxplot_metric(
                plot_records,
                conditions,
                f"{channel.lower()}_axial_extent_50_{extent_unit_suffix}",
                extent_ylabel,
                f"{channel} axial signal extent by condition",
                output_root,
                f"{channel.lower()}_axial_signal_extent",
            )

    plot_axial_profiles(
        profile_store_for_lines, conditions, spec.channel_names, output_root, dataset
    )
    # Every dataset uses at most 50 thesis-display rows per condition and
    # retains full selected-cell appendix files for every fluorescence channel.
    plot_heatmaps(cell_profile_store, conditions, spec.channel_names, output_root, dataset)
    plot_radial_profiles(
        radial_store_for_lines, conditions, spec.channel_names, output_root, dataset
    )
    if dataset == "3d_mip":
        plot_3d_pairwise_radial_profiles(
            radial_store_for_lines, conditions, output_root
        )
    plot_standardized_maps(
        condition_map_store, conditions, spec.channel_names, output_root
    )
    if dataset == "2d_wga_dapi":
        plot_wga_dapi_overlap_maps(condition_map_store, conditions, output_root)
    elif dataset == "3d_mip":
        plot_3d_pairwise_overlap_maps(condition_map_store, conditions, output_root)

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
        "heatmap_policy": (
            {
                "generated": True,
                "main_layout": (
                    "2 x 4: THY/NHS by 5/20/40/120 min"
                    if dataset == "2d_time"
                    else "2 x 2: THY/NHS by no-CSP/CSP"
                    if dataset == "2d_wga_dapi"
                    else "2 x 4: WT/ΔpspA by medium and labeling time"
                ),
                "channels": list(spec.channel_names),
                "display_rows_per_condition": THESIS_HEATMAP_ROWS,
                "large_group_method": "equal-count median bins after peak-position sorting",
                "small_group_method": "individual profiles with blank padding",
                "full_cell_outputs": "appendix_heatmaps",
            }
            if dataset in {"2d_time", "2d_wga_dapi", "3d_mip"}
            else {
                "generated": False,
                "reason": "Heatmaps not configured for this dataset",
            }
        ),
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
        "three_d_grouping": (
            {
                "condition_definition": "genotype + medium + labeling time",
                "batch_variable": "acquisition_date",
                "batch_warning": (
                    "Inspect acquisition-date distributions before interpreting pooled "
                    "3D condition differences; several conditions are batch-imbalanced."
                ),
                "filename_label_sequence_is_metadata_only": True,
            }
            if dataset == "3d_mip"
            else None
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

