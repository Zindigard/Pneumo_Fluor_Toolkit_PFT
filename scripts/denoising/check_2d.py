"""
Quantify 2D microscopy noise and Noise2Void restoration quality from OME-Zarr.

It operates on the highest-resolution, non-normalized OME-Zarr pixels and uses
manually prepared binary foreground masks for the thesis-defined region-based
signal-to-noise ratio (ROI SNR).

Two modes are provided:

``original``
    Analyse original OME-Zarr images below ``results/img/2d_time`` and
    ``results/img/2d_wga_dapi``. The script reports ROI SNR together with
    structured-noise diagnostics previously distributed across ``check_2d.py``,
    ``noise_checks.py``, and ``FFT.py``.

``n2v``
    Analyse Noise2Void outputs below ``results/N2V`` and pair each output with
    the exact source OME-Zarr frame and channel recorded in its provenance
    metadata. The same hand-labelled foreground mask is applied unchanged to
    the raw and denoised images. The script reports raw ROI SNR, N2V ROI SNR,
    delta SNR, and raw-to-N2V SSIM.


Outputs are written to ``results/noise_analysis/2d/<mode>`` as detailed CSV
files, a compact TXT summary, and an error report.
"""

from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``."""
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both 'scripts' and "
        f"'src/PFT/core_prog_parts'. Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"
if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import imageio.v3 as iio
import numpy as np
import tifffile

try:
    import zarr
except Exception as exc:  # pragma: no cover - depends on the runtime environment
    raise ImportError(
        "check_2d.py requires zarr. Install the project dependencies with "
        "'python -m pip install -e .'."
    ) from exc

try:
    from skimage.metrics import structural_similarity
except Exception as exc:  # pragma: no cover - depends on the runtime environment
    raise ImportError(
        "check_2d.py requires scikit-image for SSIM. Install the project "
        "dependencies with 'python -m pip install -e .'."
    ) from exc


DEFAULT_EPSILON = 1e-12
REPO_ROOT = _PFT_PROJECT_ROOT
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_ORIGINAL_ROOT = RESULTS_ROOT / "img"
DEFAULT_N2V_ROOT = RESULTS_ROOT / "N2V"
DEFAULT_MASK_ROOT = RESULTS_ROOT / "training_files" / "U-net"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "noise_analysis" / "2d"
DATASETS = ("2d_time", "2d_wga_dapi")

COMMON_MASK_NAMES = (
    "mask.tif",
    "mask.tiff",
    "masks.tif",
    "masks.tiff",
    "labels.tif",
    "labels.tiff",
    "label.tif",
    "label.tiff",
    "mask.png",
)


@dataclass(frozen=True)
class OmezarrImage:
    """Level-0 OME-Zarr array together with axes and root attributes."""

    path: Path
    axes: tuple[str, ...]
    array: Any
    attrs: dict[str, Any]


@dataclass(frozen=True)
class PlaneSelection:
    """One extracted 2D plane and the source indices used to obtain it."""

    plane_id: str
    time_index: int | None
    channel_index: int | None
    channel_name: str
    image: np.ndarray


@dataclass(frozen=True)
class MaskSelection:
    """Binary foreground mask selected for one image plane."""

    mask: np.ndarray
    path: Path
    selection_rule: str


@dataclass(frozen=True)
class PlaneMetrics:
    """ROI SNR and structured-noise metrics for one 2D image plane."""

    signal_pixels: int
    background_pixels: int
    signal_mean: float
    background_mean: float
    background_standard_deviation: float
    roi_snr: float
    mean_intensity: float
    variance: float
    standard_deviation: float
    robust_noise_sigma: float
    fano_factor: float
    neighbor_correlation: float
    row_adjacent_correlation: float
    column_adjacent_correlation: float
    fft_peak_score: float
    fft_directionality: float
    fft_direction_angle_deg: float


@dataclass(frozen=True)
class PlaneRecord:
    """Serializable per-plane result record."""

    source_kind: str
    dataset: str
    sample: str
    variant: str
    zarr_path: str
    plane_id: str
    time_index: int | None
    channel_index: int | None
    channel_name: str
    mask_path: str
    mask_selection_rule: str
    signal_pixels: int
    background_pixels: int
    signal_mean: float
    background_mean: float
    background_standard_deviation: float
    roi_snr: float
    mean_intensity: float
    variance: float
    standard_deviation: float
    robust_noise_sigma: float
    fano_factor: float
    neighbor_correlation: float
    row_adjacent_correlation: float
    column_adjacent_correlation: float
    fft_peak_score: float
    fft_directionality: float
    fft_direction_angle_deg: float


@dataclass(frozen=True)
class ImageRecord:
    """Per-image values obtained by averaging all evaluated planes."""

    source_kind: str
    dataset: str
    sample: str
    variant: str
    zarr_path: str
    axes: str
    shape: str
    plane_count: int
    roi_snr_mean: float
    roi_snr_standard_deviation: float
    signal_mean: float
    background_mean: float
    background_standard_deviation: float
    robust_noise_sigma: float
    fano_factor: float
    neighbor_correlation: float
    row_adjacent_correlation: float
    column_adjacent_correlation: float
    fft_peak_score: float
    fft_directionality: float


@dataclass(frozen=True)
class N2VPairRecord:
    """Matched raw-to-N2V result used for thesis-level reporting."""

    dataset: str
    sample: str
    variant: str
    n2v_zarr_path: str
    original_zarr_path: str
    source_frame_index: int
    source_channels: str
    channel_names: str
    mask_paths: str
    raw_roi_snr: float
    n2v_roi_snr: float
    delta_roi_snr: float
    ssim_raw_n2v: float
    raw_robust_noise_sigma: float
    n2v_robust_noise_sigma: float
    raw_neighbor_correlation: float
    n2v_neighbor_correlation: float
    raw_fano_factor: float
    n2v_fano_factor: float
    raw_fft_peak_score: float
    n2v_fft_peak_score: float
    provenance: str


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _finite(values: Iterable[float]) -> np.ndarray:
    """Return finite values as a one-dimensional float64 array."""
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of finite values, or NaN when none exist."""
    array = _finite(values)
    return float(np.mean(array)) if array.size else float("nan")


def _std(values: Iterable[float]) -> float:
    """Return the sample standard deviation of finite values."""
    array = _finite(values)
    if array.size == 0:
        return float("nan")
    if array.size == 1:
        return 0.0
    return float(np.std(array, ddof=1))


def _maximum(values: Iterable[float]) -> float:
    """Return the maximum finite value, or NaN when none exist."""
    array = _finite(values)
    return float(np.max(array)) if array.size else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    """Format one numeric value for terminal and text reporting."""
    if not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_mean_sd(values: Iterable[float], digits: int = 2) -> str:
    """Format values as ``mean ± sample standard deviation``."""
    values_list = list(values)
    return f"{_fmt(_mean(values_list), digits)} ± {_fmt(_std(values_list), digits)}"


def _json_attr(attrs: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a root attribute while tolerating JSON-encoded string values."""
    value = attrs.get(key, default)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _channel_names(image: OmezarrImage) -> list[str]:
    """Return stored channel names or deterministic fallback names."""
    count = image.array.shape[image.axes.index("c")] if "c" in image.axes else 1
    value = _json_attr(image.attrs, "channel_names")
    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value]
        if len(names) >= count:
            return names[:count]
    return [f"c{index}" for index in range(count)]


def read_omezarr_level0(path: Path) -> OmezarrImage:
    """Open the highest-resolution OME-Zarr array and read its axes metadata."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"OME-Zarr directory does not exist: {path}")

    root = zarr.open_group(str(path), mode="r")
    multiscales = root.attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"Missing OME-NGFF multiscales metadata: {path}")

    ms0 = multiscales[0]
    axes_meta = ms0.get("axes", [])
    axes = tuple(
        str(item.get("name", "")).lower() if isinstance(item, dict) else str(item).lower()
        for item in axes_meta
    )
    datasets = ms0.get("datasets", [])
    if not datasets:
        raise ValueError(f"No level-0 dataset is declared in: {path}")

    level_path = str(datasets[0]["path"])
    array = root[level_path]
    if len(axes) != array.ndim:
        raise ValueError(
            f"Axes/shape mismatch in {path}: axes={axes}, shape={tuple(array.shape)}"
        )
    if "y" not in axes or "x" not in axes:
        raise ValueError(f"OME-Zarr has no YX image axes: axes={axes}, path={path}")

    return OmezarrImage(path=path, axes=axes, array=array, attrs=dict(root.attrs))


def _axis_indices(size: int, requested: Sequence[int] | None) -> list[int]:
    """Validate requested axis indices or return every index for the axis."""
    if requested is None:
        return list(range(size))
    result = sorted(set(int(index) for index in requested))
    for index in result:
        if index < 0 or index >= size:
            raise IndexError(f"Axis index {index} is outside valid range 0..{size - 1}")
    return result


def iter_2d_planes(
    image: OmezarrImage,
    *,
    time_indices: Sequence[int] | None = None,
    channel_indices: Sequence[int] | None = None,
) -> Iterator[PlaneSelection]:
    """Yield 2D YX planes while preserving frame and channel identities."""
    axes = image.axes
    shape = tuple(int(value) for value in image.array.shape)
    names = _channel_names(image)
    non_spatial_axes = [axis for axis in axes if axis not in ("y", "x")]

    choices: list[list[int]] = []
    for axis in non_spatial_axes:
        size = shape[axes.index(axis)]
        if axis == "z" and size > 1:
            raise ValueError(
                f"check_2d.py cannot analyse non-singleton Z data: {image.path}, shape={shape}"
            )
        if axis == "t":
            choices.append(_axis_indices(size, time_indices))
        elif axis == "c":
            choices.append(_axis_indices(size, channel_indices))
        elif size == 1:
            choices.append([0])
        else:
            choices.append(list(range(size)))

    combinations = product(*choices) if choices else [tuple()]
    for combination in combinations:
        selected = dict(zip(non_spatial_axes, combination))
        index: list[Any] = []
        for axis in axes:
            index.append(slice(None) if axis in ("y", "x") else selected[axis])

        plane = np.asarray(image.array[tuple(index)])
        plane = np.squeeze(plane)
        if plane.ndim != 2:
            raise ValueError(
                f"Expected a 2D plane after selection, obtained {plane.shape} from {image.path}"
            )

        labels = [f"{axis}{selected[axis]}" for axis in non_spatial_axes]
        channel_index = selected.get("c")
        channel_name = names[channel_index] if channel_index is not None else names[0]
        yield PlaneSelection(
            plane_id="_".join(labels) if labels else "yx",
            time_index=selected.get("t"),
            channel_index=channel_index,
            channel_name=channel_name,
            image=plane,
        )


def _sanitize_name(value: str) -> str:
    """Convert a channel name into a conservative filename component."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "channel"


def _candidate_mask_names(selection: PlaneSelection) -> list[tuple[str, str]]:
    """Return mask filename candidates from most specific to least specific."""
    names: list[tuple[str, str]] = []
    suffixes = (".tif", ".tiff", ".png")

    if selection.time_index is not None and selection.channel_index is not None:
        for suffix in suffixes:
            names.extend(
                [
                    (f"mask_t{selection.time_index}_c{selection.channel_index}{suffix}", "frame-channel mask"),
                    (f"mask_frame{selection.time_index}_channel{selection.channel_index}{suffix}", "frame-channel mask"),
                ]
            )

    if selection.channel_index is not None:
        channel_alias = _sanitize_name(selection.channel_name)
        semantic_aliases: list[str] = []
        if selection.channel_index == 0:
            semantic_aliases.append("dapi")
        if selection.channel_index == 1:
            semantic_aliases.append("wga")
        for suffix in suffixes:
            names.extend(
                [
                    (f"mask_c{selection.channel_index}{suffix}", "channel-specific mask"),
                    (f"mask_ch{selection.channel_index}{suffix}", "channel-specific mask"),
                    (f"mask_channel{selection.channel_index}{suffix}", "channel-specific mask"),
                    (f"mask_{channel_alias}{suffix}", "channel-name mask"),
                ]
            )
            names.extend((f"mask_{alias}{suffix}", "semantic channel mask") for alias in semantic_aliases)

    if selection.time_index is not None:
        for suffix in suffixes:
            names.extend(
                [
                    (f"mask_t{selection.time_index}{suffix}", "frame-specific mask"),
                    (f"mask_frame{selection.time_index}{suffix}", "frame-specific mask"),
                ]
            )

    names.extend((name, "common sample mask") for name in COMMON_MASK_NAMES)

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, rule in names:
        lower = name.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append((name, rule))
    return unique


def _read_mask_file(path: Path) -> np.ndarray:
    """Read a TIFF or PNG mask and return its array without binarization."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return np.asarray(tifffile.imread(path))
    if suffix == ".png":
        return np.asarray(iio.imread(path))
    raise ValueError(f"Unsupported mask format: {path}")


def _select_mask_plane(
    mask_array: np.ndarray,
    *,
    selection: PlaneSelection,
    image: OmezarrImage,
    mask_path: Path,
) -> np.ndarray:
    """Select the YX mask corresponding to one image frame and channel.

    Supported layouts are YX, TYX, CYX, and TCYX. A two-dimensional common
    mask is reused unchanged for every selected frame and channel.
    """
    mask = np.asarray(mask_array)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]

    expected_yx = (
        int(image.array.shape[image.axes.index("y")]),
        int(image.array.shape[image.axes.index("x")]),
    )
    if mask.shape[-2:] != expected_yx:
        raise ValueError(
            f"Mask/image shape mismatch for {mask_path}: mask={mask.shape}, expected YX={expected_yx}"
        )

    if mask.ndim == 2:
        return mask

    time_size = int(image.array.shape[image.axes.index("t")]) if "t" in image.axes else 1
    channel_size = int(image.array.shape[image.axes.index("c")]) if "c" in image.axes else 1
    time_index = selection.time_index or 0
    channel_index = selection.channel_index or 0

    if mask.ndim == 3:
        first = mask.shape[0]
        time_match = time_size > 1 and first == time_size
        channel_match = channel_size > 1 and first == channel_size
        if time_match and channel_match:
            raise ValueError(
                f"Ambiguous 3D mask layout for {mask_path}: first dimension matches both "
                f"time ({time_size}) and channels ({channel_size}). Use specific 2D mask files."
            )
        if time_match:
            return mask[time_index]
        if channel_match:
            return mask[channel_index]
        raise ValueError(
            f"Unsupported 3D mask layout for {mask_path}: shape={mask.shape}, "
            f"time={time_size}, channels={channel_size}"
        )

    if mask.ndim == 4 and mask.shape[:2] == (time_size, channel_size):
        return mask[time_index, channel_index]

    raise ValueError(
        f"Unsupported mask dimensionality/layout for {mask_path}: shape={mask.shape}. "
        "Use YX, TYX, CYX, or TCYX."
    )


def resolve_mask(
    *,
    mask_root: Path,
    dataset: str,
    sample: str,
    selection: PlaneSelection,
    image: OmezarrImage,
) -> MaskSelection:
    """Locate and select the hand-labelled foreground mask for one plane."""
    sample_dir = mask_root / dataset / Path(sample)
    if not sample_dir.is_dir():
        raise FileNotFoundError(
            f"Mask directory not found: {sample_dir}. Expected a hand-labelled mask for {sample}."
        )

    available = {path.name.lower(): path for path in sample_dir.iterdir() if path.is_file()}
    for candidate, rule in _candidate_mask_names(selection):
        path = available.get(candidate.lower())
        if path is None:
            continue
        selected = _select_mask_plane(
            _read_mask_file(path), selection=selection, image=image, mask_path=path
        )
        binary = np.asarray(selected) > 0
        if binary.ndim != 2:
            raise ValueError(f"Selected mask is not 2D: {path}, shape={binary.shape}")
        return MaskSelection(mask=binary, path=path.resolve(), selection_rule=rule)

    expected = ", ".join(name for name, _ in _candidate_mask_names(selection)[:8])
    raise FileNotFoundError(
        f"No supported mask file was found in {sample_dir}. Example accepted names: {expected}"
    )


def roi_snr(image: np.ndarray, foreground_mask: np.ndarray, epsilon: float) -> tuple[int, int, float, float, float, float]:
    """Calculate the thesis-defined region-based SNR for one image plane.

    The foreground mask defines ``Omega_S``. Pixels outside the mask define
    ``Omega_B``. Non-finite image pixels are excluded. The background standard
    deviation is the sample standard deviation with ``ddof=1``.
    """
    array = np.asarray(image, dtype=np.float64)
    mask = np.asarray(foreground_mask, dtype=bool)
    if array.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"ROI SNR expects 2D image and mask, got {array.shape}, {mask.shape}")
    if array.shape != mask.shape:
        raise ValueError(f"ROI SNR shape mismatch: image={array.shape}, mask={mask.shape}")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"epsilon must be a finite positive value, got {epsilon}")

    finite = np.isfinite(array)
    signal_values = array[mask & finite]
    background_values = array[(~mask) & finite]
    if signal_values.size == 0:
        raise ValueError("The foreground mask contains no finite signal pixels.")
    if background_values.size < 2:
        raise ValueError("At least two finite background pixels are required for sample SD.")

    signal_mean = float(np.mean(signal_values))
    background_mean = float(np.mean(background_values))
    background_sd = float(np.std(background_values, ddof=1))
    snr_value = float((signal_mean - background_mean) / (background_sd + epsilon))
    return (
        int(signal_values.size),
        int(background_values.size),
        signal_mean,
        background_mean,
        background_sd,
        snr_value,
    )


def robust_noise_sigma(image: np.ndarray) -> float:
    """Estimate random-noise sigma from robust adjacent-pixel differences.

    This diagnostic is retained from the previous noise-checking scripts. It is
    not used as the denominator of the thesis ROI SNR.
    """
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"robust_noise_sigma expects 2D data, got {array.shape}")
    if min(array.shape) < 2:
        return float("nan")

    dx = array[:, 1:] - array[:, :-1]
    dy = array[1:, :] - array[:-1, :]
    differences = np.concatenate((dx.ravel(), dy.ravel()))
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")

    median = np.median(differences)
    mad = np.median(np.abs(differences - median))
    return float(1.4826 * mad / np.sqrt(2.0))


def _pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation for two equally sized arrays."""
    first_flat = np.asarray(first, dtype=np.float64).ravel()
    second_flat = np.asarray(second, dtype=np.float64).ravel()
    finite = np.isfinite(first_flat) & np.isfinite(second_flat)
    first_flat = first_flat[finite]
    second_flat = second_flat[finite]
    if first_flat.size < 2:
        return float("nan")

    first_flat = first_flat - np.mean(first_flat)
    second_flat = second_flat - np.mean(second_flat)
    denominator = np.sqrt(np.sum(first_flat**2) * np.sum(second_flat**2))
    if denominator <= DEFAULT_EPSILON:
        return 0.0
    return float(np.sum(first_flat * second_flat) / denominator)


def neighbor_correlation(image: np.ndarray) -> float:
    """Average Pearson correlation with horizontal and vertical neighbours."""
    array = np.asarray(image, dtype=np.float64)
    values: list[float] = []
    if array.shape[1] >= 2:
        values.append(_pearson_correlation(array[:, :-1], array[:, 1:]))
    if array.shape[0] >= 2:
        values.append(_pearson_correlation(array[:-1, :], array[1:, :]))
    return _mean(values)


def row_column_adjacent_correlations(image: np.ndarray) -> tuple[float, float]:
    """Return mean correlation between adjacent rows and adjacent columns."""
    array = np.asarray(image, dtype=np.float64)
    row_values = [
        _pearson_correlation(array[index, :], array[index + 1, :])
        for index in range(max(0, array.shape[0] - 1))
    ]
    column_values = [
        _pearson_correlation(array[:, index], array[:, index + 1])
        for index in range(max(0, array.shape[1] - 1))
    ]
    return _mean(row_values), _mean(column_values)


def fft_peak_score(image: np.ndarray) -> float:
    """Return strongest non-DC Fourier-power peak divided by median power."""
    array = np.asarray(image, dtype=np.float64)
    array = array - np.mean(array)
    power = np.abs(np.fft.fftshift(np.fft.fft2(array))) ** 2

    height, width = power.shape
    center_y, center_x = height // 2, width // 2
    radius = max(2, min(height, width) // 32)
    masked = power.astype(np.float64, copy=True)
    masked[
        max(0, center_y - radius) : min(height, center_y + radius + 1),
        max(0, center_x - radius) : min(width, center_x + radius + 1),
    ] = np.nan

    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) / (np.median(finite) + DEFAULT_EPSILON))


def fft_directionality(image: np.ndarray) -> tuple[float, float]:
    """Return FFT anisotropy score and dominant frequency-plane angle."""
    array = np.asarray(image, dtype=np.float64)
    array = array - np.mean(array)
    power = np.abs(np.fft.fftshift(np.fft.fft2(array))) ** 2

    height, width = power.shape
    center_y, center_x = height // 2, width // 2
    halfwidth = max(2, min(height, width) // 32)
    power[
        max(0, center_y - halfwidth) : min(height, center_y + halfwidth + 1),
        max(0, center_x - halfwidth) : min(width, center_x + halfwidth + 1),
    ] = 0.0

    total = float(np.sum(power))
    if total <= DEFAULT_EPSILON:
        return 0.0, 0.0

    yy, xx = np.indices((height, width))
    horizontal = (xx - center_x).astype(np.float64)
    vertical = (yy - center_y).astype(np.float64)
    mxx = float(np.sum(power * horizontal**2) / total)
    myy = float(np.sum(power * vertical**2) / total)
    mxy = float(np.sum(power * horizontal * vertical) / total)

    trace = mxx + myy
    determinant = mxx * myy - mxy**2
    discriminant = max(trace**2 - 4.0 * determinant, 0.0)
    root = float(np.sqrt(discriminant))
    lambda_1 = 0.5 * (trace + root)
    lambda_2 = 0.5 * (trace - root)
    score = float(
        np.clip(
            (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + DEFAULT_EPSILON),
            0.0,
            1.0,
        )
    )
    angle = float(np.degrees(0.5 * np.arctan2(2.0 * mxy, mxx - myy)))
    if angle >= 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return score, angle


def calculate_plane_metrics(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    epsilon: float,
) -> PlaneMetrics:
    """Calculate ROI SNR and all retained diagnostic metrics for one plane."""
    array = np.asarray(image, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if array.ndim != 2 or finite.size == 0:
        raise ValueError(f"Expected a finite 2D image plane, got {array.shape}")

    (
        signal_pixels,
        background_pixels,
        signal_mean,
        background_mean,
        background_sd,
        snr_value,
    ) = roi_snr(array, foreground_mask, epsilon)

    mean_intensity = float(np.mean(finite))
    variance = float(np.var(finite))
    standard_deviation = float(np.sqrt(variance))
    fano = float(variance / (mean_intensity + epsilon)) if mean_intensity > epsilon else float("nan")
    row_corr, column_corr = row_column_adjacent_correlations(array)
    directionality, direction_angle = fft_directionality(array)

    return PlaneMetrics(
        signal_pixels=signal_pixels,
        background_pixels=background_pixels,
        signal_mean=signal_mean,
        background_mean=background_mean,
        background_standard_deviation=background_sd,
        roi_snr=snr_value,
        mean_intensity=mean_intensity,
        variance=variance,
        standard_deviation=standard_deviation,
        robust_noise_sigma=robust_noise_sigma(array),
        fano_factor=fano,
        neighbor_correlation=neighbor_correlation(array),
        row_adjacent_correlation=row_corr,
        column_adjacent_correlation=column_corr,
        fft_peak_score=fft_peak_score(array),
        fft_directionality=directionality,
        fft_direction_angle_deg=direction_angle,
    )


def _dataset_channels(dataset: str) -> list[int] | None:
    """Return the intended source channel selection for a supported dataset."""
    if dataset == "2d_time":
        return [0]
    if dataset == "2d_wga_dapi":
        return None
    raise ValueError(f"Unsupported dataset: {dataset}")


def _aggregate_image_record(
    *,
    source_kind: str,
    dataset: str,
    sample: str,
    variant: str,
    image: OmezarrImage,
    plane_metrics: Sequence[PlaneMetrics],
) -> ImageRecord:
    """Average per-plane values into one per-image record."""
    if not plane_metrics:
        raise ValueError(f"No planes were analysed for {image.path}")

    def average(field_name: str) -> float:
        """Average one PlaneMetrics field across all planes in the image."""
        return _mean(getattr(metric, field_name) for metric in plane_metrics)

    return ImageRecord(
        source_kind=source_kind,
        dataset=dataset,
        sample=sample,
        variant=variant,
        zarr_path=str(image.path),
        axes="".join(image.axes),
        shape=str(tuple(int(value) for value in image.array.shape)),
        plane_count=len(plane_metrics),
        roi_snr_mean=average("roi_snr"),
        roi_snr_standard_deviation=_std(metric.roi_snr for metric in plane_metrics),
        signal_mean=average("signal_mean"),
        background_mean=average("background_mean"),
        background_standard_deviation=average("background_standard_deviation"),
        robust_noise_sigma=average("robust_noise_sigma"),
        fano_factor=average("fano_factor"),
        neighbor_correlation=average("neighbor_correlation"),
        row_adjacent_correlation=average("row_adjacent_correlation"),
        column_adjacent_correlation=average("column_adjacent_correlation"),
        fft_peak_score=average("fft_peak_score"),
        fft_directionality=average("fft_directionality"),
    )


def analyse_omezarr(
    path: Path,
    *,
    source_kind: str,
    dataset: str,
    sample: str,
    mask_root: Path,
    epsilon: float,
    variant: str = "-",
    time_indices: Sequence[int] | None = None,
    channel_indices: Sequence[int] | None = None,
) -> tuple[ImageRecord, list[PlaneRecord], list[PlaneSelection], list[PlaneMetrics], list[MaskSelection]]:
    """Analyse one OME-Zarr image using hand-labelled masks."""
    image = read_omezarr_level0(path)
    selections = list(
        iter_2d_planes(
            image,
            time_indices=time_indices,
            channel_indices=channel_indices,
        )
    )
    mask_selections = [
        resolve_mask(
            mask_root=mask_root,
            dataset=dataset,
            sample=sample,
            selection=selection,
            image=image,
        )
        for selection in selections
    ]
    metrics = [
        calculate_plane_metrics(selection.image, mask_selection.mask, epsilon=epsilon)
        for selection, mask_selection in zip(selections, mask_selections)
    ]

    records: list[PlaneRecord] = []
    for selection, mask_selection, metric in zip(selections, mask_selections, metrics):
        records.append(
            PlaneRecord(
                source_kind=source_kind,
                dataset=dataset,
                sample=sample,
                variant=variant,
                zarr_path=str(image.path),
                plane_id=selection.plane_id,
                time_index=selection.time_index,
                channel_index=selection.channel_index,
                channel_name=selection.channel_name,
                mask_path=str(mask_selection.path),
                mask_selection_rule=mask_selection.selection_rule,
                **asdict(metric),
            )
        )

    image_record = _aggregate_image_record(
        source_kind=source_kind,
        dataset=dataset,
        sample=sample,
        variant=variant,
        image=image,
        plane_metrics=metrics,
    )
    return image_record, records, selections, metrics, mask_selections


def find_original_omezarrs(original_root: Path, datasets: Sequence[str]) -> list[tuple[str, str, Path]]:
    """Discover original ``image.ome.zarr`` stores for selected datasets."""
    discovered: list[tuple[str, str, Path]] = []
    for dataset in datasets:
        dataset_root = original_root / dataset
        if not dataset_root.is_dir():
            continue
        for path in sorted(dataset_root.rglob("image.ome.zarr")):
            sample = path.parent.relative_to(dataset_root).as_posix()
            discovered.append((dataset, sample, path))
    return discovered


def _parse_n2v_path(n2v_root: Path, path: Path) -> tuple[str, str, str]:
    """Extract dataset, sample, and N2V variant from the standard output path."""
    relative = path.relative_to(n2v_root)
    parts = relative.parts
    if len(parts) < 5 or parts[-2] != "denoised":
        raise ValueError(
            "Expected N2V path '<dataset>/<sample>/<variant>/denoised/image.ome.zarr', "
            f"got: {relative}"
        )
    return parts[0], parts[1], parts[2]


def find_n2v_omezarrs(n2v_root: Path, datasets: Sequence[str]) -> list[tuple[str, str, str, Path]]:
    """Discover standard Noise2Void denoised OME-Zarr outputs."""
    discovered: list[tuple[str, str, str, Path]] = []
    if not n2v_root.is_dir():
        return discovered

    for path in sorted(n2v_root.rglob("image.ome.zarr")):
        try:
            dataset, sample, variant = _parse_n2v_path(n2v_root, path)
        except ValueError:
            continue
        if dataset in datasets:
            discovered.append((dataset, sample, variant, path))
    return discovered


def _variant_channels(dataset: str, variant: str) -> list[int]:
    """Map an N2V model variant to source channel indices."""
    if dataset == "2d_time" or variant == "time_ch0":
        return [0]
    if variant == "blue_only":
        return [0]
    if variant == "green_only":
        return [1]
    if variant == "joint":
        return [0, 1]
    return [0] if dataset == "2d_time" else [0, 1]


def _source_information(
    *,
    n2v_image: OmezarrImage,
    original_root: Path,
    dataset: str,
    sample: str,
    variant: str,
) -> tuple[Path, int, list[int], str]:
    """Resolve the original OME-Zarr and selected plane from N2V provenance."""
    attrs = n2v_image.attrs
    source_attr = (
        _json_attr(attrs, "pft_n2v_source_zarr")
        or _json_attr(attrs, "pft_source_omezarr")
        or _json_attr(attrs, "source_omezarr")
    )
    expected = original_root / dataset / sample / "image.ome.zarr"
    if source_attr:
        source_path = Path(str(source_attr)).expanduser()
        if not source_path.is_absolute():
            source_path = (REPO_ROOT / source_path).resolve()
        provenance = "OME-Zarr attributes"
    else:
        source_path = expected
        provenance = "path inference; legacy N2V output"

    if not source_path.exists() and expected.exists():
        source_path = expected
        provenance += "; expected results/img path used"

    frame_value = _json_attr(attrs, "pft_n2v_source_frame_index", 0)
    try:
        frame_index = int(frame_value)
    except Exception:
        frame_index = 0
        provenance += "; invalid frame attribute replaced by 0"

    channels_value = _json_attr(attrs, "pft_n2v_source_channels")
    if isinstance(channels_value, (list, tuple)):
        try:
            channels = [int(value) for value in channels_value]
        except Exception:
            channels = _variant_channels(dataset, variant)
    else:
        channels = _variant_channels(dataset, variant)
        if channels_value is None:
            provenance += "; channels inferred from variant"

    return source_path.resolve(), frame_index, channels, provenance


def _ssim(raw: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate structural similarity with a shared raw/denoised data range."""
    raw_float = np.asarray(raw, dtype=np.float64)
    denoised_float = np.asarray(denoised, dtype=np.float64)
    if raw_float.shape != denoised_float.shape:
        raise ValueError(f"SSIM shape mismatch: raw={raw_float.shape}, N2V={denoised_float.shape}")

    minimum = float(min(np.min(raw_float), np.min(denoised_float)))
    maximum = float(max(np.max(raw_float), np.max(denoised_float)))
    data_range = maximum - minimum
    if data_range <= DEFAULT_EPSILON:
        return 1.0 if np.array_equal(raw_float, denoised_float) else 0.0

    smallest = min(raw_float.shape)
    window_size: int | None = None
    if smallest < 7:
        window_size = smallest if smallest % 2 == 1 else smallest - 1
        if window_size < 3:
            raise ValueError(f"SSIM requires images of at least 3x3 pixels, got {raw_float.shape}")

    kwargs: dict[str, Any] = {"data_range": data_range}
    if window_size is not None:
        kwargs["win_size"] = window_size
    return float(structural_similarity(raw_float, denoised_float, **kwargs))


def _write_dataclass_csv(path: Path, rows: Sequence[Any]) -> None:
    """Write a sequence of dataclass instances to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = [field.name for field in fields(rows[0])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _dataset_summary_lines(records: Sequence[ImageRecord], dataset: str) -> list[str]:
    """Create compact original-or-denoised dataset summary lines."""
    subset = [record for record in records if record.dataset == dataset]
    if not subset:
        return [f"Dataset: {dataset}", "  No successful images.", ""]

    return [
        f"Dataset: {dataset}",
        f"  n images: {len(subset)}",
        f"  ROI SNR, mean ± SD: {_fmt_mean_sd(record.roi_snr_mean for record in subset)}",
        f"  Robust noise sigma, mean ± SD: "
        f"{_fmt_mean_sd(record.robust_noise_sigma for record in subset)}",
        f"  Neighbour correlation, mean: "
        f"{_fmt(_mean(record.neighbor_correlation for record in subset))}",
        f"  Neighbour correlation, maximum: "
        f"{_fmt(_maximum(record.neighbor_correlation for record in subset))}",
        f"  Fano factor, mean: {_fmt(_mean(record.fano_factor for record in subset), 2)}",
        f"  Fano factor, maximum: {_fmt(_maximum(record.fano_factor for record in subset), 2)}",
        f"  FFT peak score, mean: {_fmt(_mean(record.fft_peak_score for record in subset), 2)}",
        f"  FFT peak score, maximum: {_fmt(_maximum(record.fft_peak_score for record in subset), 2)}",
        f"  FFT directionality, mean: "
        f"{_fmt(_mean(record.fft_directionality for record in subset))}",
        "",
    ]


def write_original_summary(
    records: Sequence[ImageRecord],
    errors: Sequence[str],
    output_path: Path,
    datasets: Sequence[str],
    *,
    mask_root: Path,
    epsilon: float,
) -> str:
    """Write and return the complete original-image summary report."""
    lines = [
        "PFT 2D ORIGINAL OME-ZARR NOISE SUMMARY",
        "=" * 80,
        f"Generated (UTC): {_utc_now()}",
        f"Images analysed successfully: {len(records)}",
        f"Images failed: {len(errors)}",
        f"Mask root: {mask_root}",
        f"SNR epsilon: {epsilon}",
        "",
        "Thesis-defined ROI SNR",
        "----------------------",
        "Signal region: hand-labelled foreground mask (mask > 0).",
        "Background region: pixels outside the foreground mask.",
        "Background SD: sample standard deviation (ddof=1).",
        "ROI SNR = (mean signal - mean background) / (background SD + epsilon).",
        "The same mask must be used for raw and denoised versions.",
        "",
        "Additional structured-noise diagnostics",
        "---------------------------------------",
        "Robust noise sigma = 1.4826 * MAD(adjacent differences) / sqrt(2).",
        "Fano factor = full-plane pixel variance / full-plane mean intensity.",
        "Neighbour correlation = mean horizontal/vertical adjacent-pixel Pearson correlation.",
        "FFT peak score = maximum non-DC Fourier power / median non-DC Fourier power.",
        "These diagnostics do not replace the ROI SNR formula.",
        "",
        "Dataset results",
        "---------------",
    ]
    for dataset in datasets:
        lines.extend(_dataset_summary_lines(records, dataset))

    lines.extend(
        [
            "Thesis-table values",
            "-------------------",
            "Dataset | n | ROI SNR raw, mean ± SD",
        ]
    )
    for dataset in datasets:
        subset = [record for record in records if record.dataset == dataset]
        if subset:
            lines.append(
                f"{dataset} | {len(subset)} | "
                f"{_fmt_mean_sd(record.roi_snr_mean for record in subset)}"
            )

    if errors:
        lines.extend(["", "Errors", "------", *errors])

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def _preferred_n2v_group(records: Sequence[N2VPairRecord], dataset: str) -> list[N2VPairRecord]:
    """Select the thesis-relevant N2V variant for one dataset."""
    subset = [record for record in records if record.dataset == dataset]
    if not subset:
        return []

    preferred = "time_ch0" if dataset == "2d_time" else "joint"
    preferred_records = [record for record in subset if record.variant == preferred]
    if preferred_records:
        return preferred_records

    variants = sorted({record.variant for record in subset})
    return max(
        ([record for record in subset if record.variant == variant] for variant in variants),
        key=len,
    )


def _n2v_group_lines(records: Sequence[N2VPairRecord], title: str) -> list[str]:
    """Create thesis-compatible raw-versus-N2V lines for one record group."""
    if not records:
        return [f"Dataset: {title}", "  No successful pairs.", ""]
    return [
        f"Dataset: {title}",
        f"  n pairs: {len(records)}",
        f"  ROI SNR raw, mean ± SD: {_fmt_mean_sd(record.raw_roi_snr for record in records)}",
        f"  ROI SNR N2V, mean ± SD: {_fmt_mean_sd(record.n2v_roi_snr for record in records)}",
        f"  Delta ROI SNR, mean: {_fmt(_mean(record.delta_roi_snr for record in records), 2)}",
        f"  SSIM raw/N2V, mean: {_fmt(_mean(record.ssim_raw_n2v for record in records), 3)}",
        f"  Robust noise sigma raw -> N2V: "
        f"{_fmt(_mean(record.raw_robust_noise_sigma for record in records))} -> "
        f"{_fmt(_mean(record.n2v_robust_noise_sigma for record in records))}",
        f"  Neighbour correlation raw -> N2V: "
        f"{_fmt(_mean(record.raw_neighbor_correlation for record in records))} -> "
        f"{_fmt(_mean(record.n2v_neighbor_correlation for record in records))}",
        f"  Fano factor raw -> N2V: "
        f"{_fmt(_mean(record.raw_fano_factor for record in records), 2)} -> "
        f"{_fmt(_mean(record.n2v_fano_factor for record in records), 2)}",
        f"  FFT peak score raw -> N2V: "
        f"{_fmt(_mean(record.raw_fft_peak_score for record in records), 2)} -> "
        f"{_fmt(_mean(record.n2v_fft_peak_score for record in records), 2)}",
        "",
    ]


def write_n2v_summary(
    pairs: Sequence[N2VPairRecord],
    errors: Sequence[str],
    output_path: Path,
    datasets: Sequence[str],
    *,
    mask_root: Path,
    epsilon: float,
) -> str:
    """Write and return the raw-to-N2V comparison summary report."""
    lines = [
        "PFT 2D NOISE2VOID ROI-SNR COMPARISON SUMMARY",
        "=" * 80,
        f"Generated (UTC): {_utc_now()}",
        f"Pairs analysed successfully: {len(pairs)}",
        f"Pairs failed: {len(errors)}",
        f"Mask root: {mask_root}",
        f"SNR epsilon: {epsilon}",
        "",
        "The same hand-labelled foreground mask was applied unchanged to the",
        "raw and N2V-denoised version of each evaluated plane. Raw source frame",
        "and channel selection was read from N2V provenance attributes when present.",
        "",
        "Preferred thesis groups",
        "-----------------------",
    ]

    preferred_by_dataset: dict[str, list[N2VPairRecord]] = {}
    for dataset in datasets:
        preferred = _preferred_n2v_group(pairs, dataset)
        preferred_by_dataset[dataset] = preferred
        variant = preferred[0].variant if preferred else "-"
        lines.extend(_n2v_group_lines(preferred, f"{dataset} [{variant}]"))

    lines.extend(["All dataset/variant groups", "--------------------------"])
    groups = sorted({(record.dataset, record.variant) for record in pairs})
    for dataset, variant in groups:
        group = [
            record for record in pairs if record.dataset == dataset and record.variant == variant
        ]
        lines.extend(_n2v_group_lines(group, f"{dataset} [{variant}]"))

    lines.extend(
        [
            "Thesis-table values",
            "-------------------",
            "Dataset | n | ROI SNR raw | ROI SNR N2V | Delta SNR | SSIM raw/N2V",
        ]
    )
    for dataset in datasets:
        group = preferred_by_dataset[dataset]
        if group:
            lines.append(
                f"{dataset} | {len(group)} | "
                f"{_fmt_mean_sd(record.raw_roi_snr for record in group)} | "
                f"{_fmt_mean_sd(record.n2v_roi_snr for record in group)} | "
                f"{_fmt(_mean(record.delta_roi_snr for record in group), 2)} | "
                f"{_fmt(_mean(record.ssim_raw_n2v for record in group), 3)}"
            )

    if errors:
        lines.extend(["", "Errors", "------", *errors])

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def run_original_mode(
    *,
    original_root: Path,
    mask_root: Path,
    output_root: Path,
    datasets: Sequence[str],
    epsilon: float,
) -> int:
    """Analyse selected original 2D OME-Zarr images using ROI masks."""
    discovered = find_original_omezarrs(original_root, datasets)
    output_dir = output_root / "original"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_records: list[ImageRecord] = []
    plane_records: list[PlaneRecord] = []
    errors: list[str] = []

    print(f"Original OME-Zarr files found: {len(discovered)}")
    for index, (dataset, sample, path) in enumerate(discovered, start=1):
        print(f"  [{index}/{len(discovered)}] {dataset} | {sample}")
        try:
            image_record, records, _, _, _ = analyse_omezarr(
                path,
                source_kind="original",
                dataset=dataset,
                sample=sample,
                mask_root=mask_root,
                epsilon=epsilon,
                channel_indices=_dataset_channels(dataset),
            )
            image_records.append(image_record)
            plane_records.extend(records)
        except Exception as exc:
            errors.append(f"{path} | {type(exc).__name__}: {exc}")

    _write_dataclass_csv(output_dir / "original_per_image_metrics.csv", image_records)
    _write_dataclass_csv(output_dir / "original_per_plane_metrics.csv", plane_records)
    summary = write_original_summary(
        image_records,
        errors,
        output_dir / "ORIGINAL_2D_NOISE_SUMMARY.txt",
        datasets,
        mask_root=mask_root,
        epsilon=epsilon,
    )
    (output_dir / "errors.txt").write_text(
        "\n".join(errors) + ("\n" if errors else ""), encoding="utf-8"
    )

    print("\n" + summary)
    print(f"Detailed CSV files: {output_dir}")
    return 0 if not errors else 1


def run_n2v_mode(
    *,
    original_root: Path,
    n2v_root: Path,
    mask_root: Path,
    output_root: Path,
    datasets: Sequence[str],
    epsilon: float,
) -> int:
    """Compare N2V outputs with raw source planes using identical ROI masks."""
    discovered = find_n2v_omezarrs(n2v_root, datasets)
    output_dir = output_root / "n2v"
    output_dir.mkdir(parents=True, exist_ok=True)

    n2v_image_records: list[ImageRecord] = []
    n2v_plane_records: list[PlaneRecord] = []
    pair_records: list[N2VPairRecord] = []
    errors: list[str] = []

    print(f"N2V OME-Zarr files found: {len(discovered)}")
    for index, (dataset, sample, variant, n2v_path) in enumerate(discovered, start=1):
        print(f"  [{index}/{len(discovered)}] {dataset} | {sample} | {variant}")
        try:
            n2v_image = read_omezarr_level0(n2v_path)
            original_path, frame_index, channels, provenance = _source_information(
                n2v_image=n2v_image,
                original_root=original_root,
                dataset=dataset,
                sample=sample,
                variant=variant,
            )
            if not original_path.is_dir():
                raise FileNotFoundError(f"Matching original OME-Zarr was not found: {original_path}")

            original_image = read_omezarr_level0(original_path)
            original_selections = list(
                iter_2d_planes(
                    original_image,
                    time_indices=[frame_index] if "t" in original_image.axes else None,
                    channel_indices=channels if "c" in original_image.axes else None,
                )
            )
            n2v_selections = list(iter_2d_planes(n2v_image))
            if len(original_selections) != len(n2v_selections):
                raise ValueError(
                    "Raw/N2V plane-count mismatch: "
                    f"raw={len(original_selections)}, N2V={len(n2v_selections)}"
                )

            mask_selections = [
                resolve_mask(
                    mask_root=mask_root,
                    dataset=dataset,
                    sample=sample,
                    selection=selection,
                    image=original_image,
                )
                for selection in original_selections
            ]
            raw_metrics = [
                calculate_plane_metrics(raw.image, mask.mask, epsilon=epsilon)
                for raw, mask in zip(original_selections, mask_selections)
            ]
            n2v_metrics = [
                calculate_plane_metrics(denoised.image, mask.mask, epsilon=epsilon)
                for denoised, mask in zip(n2v_selections, mask_selections)
            ]
            similarities = [
                _ssim(raw.image, denoised.image)
                for raw, denoised in zip(original_selections, n2v_selections)
            ]

            n2v_records: list[PlaneRecord] = []
            for selection, mask, metric in zip(n2v_selections, mask_selections, n2v_metrics):
                n2v_records.append(
                    PlaneRecord(
                        source_kind="n2v",
                        dataset=dataset,
                        sample=sample,
                        variant=variant,
                        zarr_path=str(n2v_path),
                        plane_id=selection.plane_id,
                        time_index=selection.time_index,
                        channel_index=selection.channel_index,
                        channel_name=selection.channel_name,
                        mask_path=str(mask.path),
                        mask_selection_rule=mask.selection_rule,
                        **asdict(metric),
                    )
                )
            n2v_record = _aggregate_image_record(
                source_kind="n2v",
                dataset=dataset,
                sample=sample,
                variant=variant,
                image=n2v_image,
                plane_metrics=n2v_metrics,
            )

            raw_snr = _mean(metric.roi_snr for metric in raw_metrics)
            denoised_snr = _mean(metric.roi_snr for metric in n2v_metrics)
            pair_records.append(
                N2VPairRecord(
                    dataset=dataset,
                    sample=sample,
                    variant=variant,
                    n2v_zarr_path=str(n2v_path),
                    original_zarr_path=str(original_path),
                    source_frame_index=frame_index,
                    source_channels=",".join(str(channel) for channel in channels),
                    channel_names=",".join(selection.channel_name for selection in original_selections),
                    mask_paths=";".join(str(mask.path) for mask in mask_selections),
                    raw_roi_snr=raw_snr,
                    n2v_roi_snr=denoised_snr,
                    delta_roi_snr=denoised_snr - raw_snr,
                    ssim_raw_n2v=_mean(similarities),
                    raw_robust_noise_sigma=_mean(
                        metric.robust_noise_sigma for metric in raw_metrics
                    ),
                    n2v_robust_noise_sigma=_mean(
                        metric.robust_noise_sigma for metric in n2v_metrics
                    ),
                    raw_neighbor_correlation=_mean(
                        metric.neighbor_correlation for metric in raw_metrics
                    ),
                    n2v_neighbor_correlation=_mean(
                        metric.neighbor_correlation for metric in n2v_metrics
                    ),
                    raw_fano_factor=_mean(metric.fano_factor for metric in raw_metrics),
                    n2v_fano_factor=_mean(metric.fano_factor for metric in n2v_metrics),
                    raw_fft_peak_score=_mean(metric.fft_peak_score for metric in raw_metrics),
                    n2v_fft_peak_score=_mean(metric.fft_peak_score for metric in n2v_metrics),
                    provenance=provenance,
                )
            )
            n2v_image_records.append(n2v_record)
            n2v_plane_records.extend(n2v_records)
        except Exception as exc:
            errors.append(f"{n2v_path} | {type(exc).__name__}: {exc}")

    _write_dataclass_csv(output_dir / "n2v_per_image_metrics.csv", n2v_image_records)
    _write_dataclass_csv(output_dir / "n2v_per_plane_metrics.csv", n2v_plane_records)
    _write_dataclass_csv(output_dir / "n2v_raw_pair_comparison.csv", pair_records)
    summary = write_n2v_summary(
        pair_records,
        errors,
        output_dir / "N2V_2D_COMPARISON_SUMMARY.txt",
        datasets,
        mask_root=mask_root,
        epsilon=epsilon,
    )
    (output_dir / "errors.txt").write_text(
        "\n".join(errors) + ("\n" if errors else ""), encoding="utf-8"
    )

    print("\n" + summary)
    print(f"Detailed CSV files: {output_dir}")
    return 0 if not errors else 1


def _prompt_choice(title: str, items: Sequence[tuple[str, str]], default: int = 0) -> str:
    """Prompt for one labelled terminal selection and return its stored value."""
    print(f"\n{title}")
    for index, (label, _) in enumerate(items):
        suffix = " (default)" if index == default else ""
        print(f"   [{index}] {label}{suffix}")
    answer = input("Enter number: ").strip()
    selected = default if answer == "" else int(answer)
    if selected < 0 or selected >= len(items):
        raise ValueError("Invalid selection.")
    return items[selected][1]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "n2v"), help="Analysis mode.")
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset subset to analyse (default: all).",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=DEFAULT_ORIGINAL_ROOT,
        help="Root containing results/img/<dataset> (default: project results/img).",
    )
    parser.add_argument(
        "--n2v-root",
        type=Path,
        default=DEFAULT_N2V_ROOT,
        help="Root containing Noise2Void outputs (default: project results/N2V).",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=DEFAULT_MASK_ROOT,
        help=(
            "Root containing hand-labelled masks as <dataset>/<sample>/mask.tif "
            "(default: results/training_files/U-net)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination for noise-analysis CSV and TXT reports.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Positive numerical-stability constant in the ROI SNR denominator.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require --mode and do not display selection menus.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run original-image or Noise2Void 2D ROI-SNR analysis."""
    args = build_parser().parse_args(argv)

    mode = args.mode
    if mode is None:
        if args.non_interactive:
            raise SystemExit("--mode is required together with --non-interactive.")
        mode = _prompt_choice(
            "Select 2D noise-checking mode:",
            (
                ("Original OME-Zarr images", "original"),
                ("Noise2Void-denoised OME-Zarr images", "n2v"),
            ),
        )

    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        raise SystemExit("--epsilon must be a finite positive number.")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    original_root = args.original_root.expanduser().resolve()
    n2v_root = args.n2v_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if mode == "original":
        return run_original_mode(
            original_root=original_root,
            mask_root=mask_root,
            output_root=output_root,
            datasets=datasets,
            epsilon=args.epsilon,
        )
    return run_n2v_mode(
        original_root=original_root,
        n2v_root=n2v_root,
        mask_root=mask_root,
        output_root=output_root,
        datasets=datasets,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    raise SystemExit(main())
