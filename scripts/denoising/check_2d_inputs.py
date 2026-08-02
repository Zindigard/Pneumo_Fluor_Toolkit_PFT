r"""
Check whether 2D raw, mask, and Noise2Void data are ready for analysis.

The checker validates:
* original OME-Zarr stores below ``results/img/<dataset>/<sample>``;
* successful per-sample OME-Zarr validation reports;
* numerical U-Net ``mask.tif`` files below
  ``results/training_files/U-net/<dataset>/<sample>``;
* strict binary mask values, where ``0`` is background and ``1`` is foreground;
* mask and image YX dimensions;
* expected 2D axes and channel counts;
* readable, numeric, non-empty image samples;
* Noise2Void OME-Zarr outputs and their pairing with original images;
* raw/N2V plane count and YX compatibility;
* orphan masks that do not have a matching original OME-Zarr sample.

Exit code ``0`` means that no blocking failures were found. Exit code ``1``
means that at least one requested input is missing or incompatible.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/check_2d_inputs.py --help

Representative execution:

    python scripts/denoising/check_2d_inputs.py \
        --dataset 2d_time \
        --mode all \
        --original-root results/img \
        --n2v-root results/N2V \
        --mask-root results/training_files/U-net
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_SCRIPT_FILE = Path(__file__).resolve()


def _project_root(start: Path | None = None) -> Path:
    """Return the PFT repository root containing ``scripts`` and ``src/PFT``.

    Args:
        start (Path | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _project_root()
    """
    current = (start or _SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        if (
            (candidate / "scripts").is_dir()
            and (candidate / "src" / "PFT" / "core_prog_parts").is_dir()
        ):
            return candidate
    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both 'scripts' and "
        f"'src/PFT/core_prog_parts'. Script location: {_SCRIPT_FILE}"
    )


PROJECT_ROOT = _project_root()
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.denoising.check_2d import (  # noqa: E402
    DATASETS,
    DEFAULT_MASK_ROOT,
    DEFAULT_N2V_ROOT,
    DEFAULT_ORIGINAL_ROOT,
    OmezarrImage,
    PlaneSelection,
    _source_information,
    find_n2v_omezarrs,
    find_original_omezarrs,
    read_omezarr_level0,
    resolve_mask,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "noise_analysis" / "2d" / "input_check"
)
VALIDATION_REPORT_NAME = "ome_zarr_validation.txt"
STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class InputCheckRecord:
    """One machine-readable readiness result for an original or N2V item."""

    mode: str
    dataset: str
    sample: str
    variant: str
    status: str
    original_zarr: str
    n2v_zarr: str
    mask_path: str
    validation_report: str
    axes: str
    shape: str
    dtype: str
    selected_plane_count: int
    foreground_pixels: int
    background_pixels: int
    source_frame_index: str
    source_channels: str
    issues: str


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _utc_now()
    """
    return datetime.now(timezone.utc).isoformat()


def _status_from_issues(failures: Sequence[str], warnings: Sequence[str]) -> str:
    """Return FAIL, WARN, or PASS from accumulated messages.

    Args:
        failures (Sequence[str]): Text value specifying failures.
        warnings (Sequence[str]): Text value specifying warnings.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _status_from_issues(failures="failures", warnings="warnings")
    """
    if failures:
        return STATUS_FAIL
    if warnings:
        return STATUS_WARN
    return STATUS_PASS


def _render_issues(failures: Sequence[str], warnings: Sequence[str]) -> str:
    """Serialize blocking failures and non-blocking warnings into one field.

    Args:
        failures (Sequence[str]): Text value specifying failures.
        warnings (Sequence[str]): Text value specifying warnings.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _render_issues(failures="failures", warnings="warnings")
    """
    parts = [f"FAIL: {message}" for message in failures]
    parts.extend(f"WARN: {message}" for message in warnings)
    return " | ".join(parts)


def _validation_report_status(zarr_path: Path) -> tuple[Path, bool, str]:
    """Return the report path, PASS state, and explanatory message.

    Args:
        zarr_path (Path): Filesystem path associated with Zarr.

    Returns:
        tuple[Path, bool, str]: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _validation_report_status(zarr_path=Path("path/to/resource"))
    """
    report = zarr_path.parent / VALIDATION_REPORT_NAME
    if not report.is_file():
        return report, False, "per-sample OME-Zarr validation report is missing"
    text = report.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*Status\s*:\s*(\w+)\s*$", text, flags=re.I | re.M)
    if match is None:
        return report, False, "validation report has no readable Status field"
    status = match.group(1).upper()
    if status != STATUS_PASS:
        return report, False, f"OME-Zarr validation status is {status}, not PASS"
    return report, True, ""


def _axis_size(image: OmezarrImage, axis: str) -> int:
    """Return an axis size, or one when the axis is absent.

    Args:
        image (OmezarrImage): Input image array to process.
        axis (str): Array axis along which the operation is performed.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _axis_size(image=image_array, axis="axis")
    """
    if axis not in image.axes:
        return 1
    return int(image.array.shape[image.axes.index(axis)])


def _yx_shape(image: OmezarrImage) -> tuple[int, int]:
    """Return the YX dimensions of an opened OME-Zarr image.

    Args:
        image (OmezarrImage): Input image array to process.

    Returns:
        tuple[int, int]: Collection containing the generated or selected values.

    Example:
        >>> result = _yx_shape(image=image_array)
    """
    return _axis_size(image, "y"), _axis_size(image, "x")


def _selected_channels(image: OmezarrImage, dataset: str) -> list[int | None]:
    """Return the channel indices that ``check_2d.py`` will analyse.

    Args:
        image (OmezarrImage): Input image array to process.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        list[int | None]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _selected_channels(image=image_array, dataset="2d_time")
    """
    if "c" not in image.axes:
        if dataset == "2d_wga_dapi":
            raise ValueError("2d_wga_dapi requires an explicit two-channel C axis")
        return [None]

    count = _axis_size(image, "c")
    if dataset == "2d_time":
        if count < 1:
            raise ValueError("2d_time contains no channel")
        return [0]
    if dataset == "2d_wga_dapi":
        if count != 2:
            raise ValueError(
                f"2d_wga_dapi requires exactly two channels, found {count}"
            )
        return [0, 1]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _selected_times(image: OmezarrImage) -> list[int | None]:
    """Return all time indices, or one absent-axis marker.

    Args:
        image (OmezarrImage): Input image array to process.

    Returns:
        list[int | None]: Collection containing the generated or selected values.

    Example:
        >>> result = _selected_times(image=image_array)
    """
    if "t" not in image.axes:
        return [None]
    return list(range(_axis_size(image, "t")))


def _check_supported_axes(image: OmezarrImage) -> list[str]:
    """Return blocking axis-layout problems for 2D analysis.

    Args:
        image (OmezarrImage): Input image array to process.

    Returns:
        list[str]: Collection containing the generated or selected values.

    Example:
        >>> result = _check_supported_axes(image=image_array)
    """
    failures: list[str] = []
    allowed = {"t", "c", "z", "y", "x"}
    for axis, size in zip(image.axes, image.array.shape):
        if axis not in allowed and int(size) > 1:
            failures.append(f"unsupported non-singleton axis {axis}={int(size)}")
    if "z" in image.axes and _axis_size(image, "z") > 1:
        failures.append(f"non-singleton Z axis is not 2D: z={_axis_size(image, 'z')}")
    height, width = _yx_shape(image)
    if height < 3 or width < 3:
        failures.append(f"YX dimensions are too small for analysis: {(height, width)}")
    if not np.issubdtype(np.dtype(image.array.dtype), np.number):
        failures.append(f"image dtype is not numeric: {image.array.dtype}")
    return failures


def _small_plane_sample(
    image: OmezarrImage,
    *,
    time_index: int | None = None,
    channel_index: int | None = None,
) -> np.ndarray:
    """Read a small strided sample from one plane without loading a full image.

    Args:
        image (OmezarrImage): Input image array to process.
        time_index (int | None): Zero-based time-point index selected from a time series. ``None`` selects the function's default behavior.
        channel_index (int | None): Zero-based index selecting channel. ``None`` selects the function's default behavior.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _small_plane_sample(image=image_array)
    """
    index: list[Any] = []
    for axis, size_value in zip(image.axes, image.array.shape):
        size = int(size_value)
        if axis == "y" or axis == "x":
            step = max(1, size // 64)
            index.append(slice(0, size, step))
        elif axis == "t":
            index.append(0 if time_index is None else int(time_index))
        elif axis == "c":
            index.append(0 if channel_index is None else int(channel_index))
        else:
            index.append(0)
    return np.asarray(image.array[tuple(index)])


def _check_sample_values(
    image: OmezarrImage,
    *,
    time_index: int | None,
    channel_index: int | None,
) -> tuple[list[str], list[str]]:
    """Check that a representative image sample is finite and non-constant.

    Args:
        image (OmezarrImage): Input image array to process.
        time_index (int | None): Zero-based time-point index selected from a time series.
        channel_index (int | None): Zero-based index selecting channel.

    Returns:
        tuple[list[str], list[str]]: Collection containing the generated or selected values.

    Example:
        >>> result = _check_sample_values(
        ...     image=image_array,
        ...     time_index=1,
        ...     channel_index=1,
        ... )
    """
    failures: list[str] = []
    warnings: list[str] = []
    sample = _small_plane_sample(
        image, time_index=time_index, channel_index=channel_index
    )
    if sample.size == 0:
        failures.append("representative image sample is empty")
        return failures, warnings
    if not np.issubdtype(sample.dtype, np.number):
        failures.append(f"representative sample is not numeric: {sample.dtype}")
        return failures, warnings
    finite = np.isfinite(sample)
    if not np.any(finite):
        failures.append("representative image sample contains no finite values")
        return failures, warnings
    if not np.all(finite):
        warnings.append("representative image sample contains non-finite values")
    finite_values = sample[finite]
    if np.min(finite_values) == np.max(finite_values):
        warnings.append("representative image sample is constant")
    return failures, warnings


def _plane_selections(
    image: OmezarrImage,
    *,
    time_indices: Sequence[int | None],
    channel_indices: Sequence[int | None],
) -> list[PlaneSelection]:
    """Create lightweight plane descriptors for exact mask compatibility checks.

    Args:
        image (OmezarrImage): Input image array to process.
        time_indices (Sequence[int | None]): Zero-based indices selecting time.
        channel_indices (Sequence[int | None]): Zero-based indices selecting channel.

    Returns:
        list[PlaneSelection]: Collection containing the generated or selected values.

    Example:
        >>> result = _plane_selections(
        ...     image=image_array,
        ...     time_indices=1,
        ...     channel_indices=1,
        ... )
    """
    selections: list[PlaneSelection] = []
    for time_index, channel_index in product(time_indices, channel_indices):
        labels: list[str] = []
        if time_index is not None:
            labels.append(f"t{time_index}")
        if channel_index is not None:
            labels.append(f"c{channel_index}")
        selections.append(
            PlaneSelection(
                plane_id="_".join(labels) if labels else "yx",
                time_index=time_index,
                channel_index=channel_index,
                channel_name=(
                    f"c{channel_index}" if channel_index is not None else "c0"
                ),
                image=np.empty((0, 0), dtype=np.float32),
            )
        )
    return selections


def _check_masks(
    *,
    image: OmezarrImage,
    mask_root: Path,
    dataset: str,
    sample: str,
    time_indices: Sequence[int | None],
    channel_indices: Sequence[int | None],
) -> tuple[Path | None, int, int, int, list[str]]:
    """Validate exact ``mask.tif`` compatibility for every selected plane.

    Args:
        image (OmezarrImage): Input image array to process.
        mask_root (Path): Directory used for mask.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.
        time_indices (Sequence[int | None]): Zero-based indices selecting time.
        channel_indices (Sequence[int | None]): Zero-based indices selecting channel.

    Returns:
        tuple[Path | None, int, int, int, list[str]]: Resolved or generated filesystem path.

    Example:
        >>> result = _check_masks(
        ...     image=image_array,
        ...     mask_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     sample="sample",
        ...     time_indices=1,
        ...     channel_indices=1,
        ... )
    """
    failures: list[str] = []
    paths: set[Path] = set()
    foreground_counts: list[int] = []
    background_counts: list[int] = []
    selections = _plane_selections(
        image,
        time_indices=time_indices,
        channel_indices=channel_indices,
    )
    for selection in selections:
        try:
            resolved = resolve_mask(
                mask_root=mask_root,
                dataset=dataset,
                sample=sample,
                selection=selection,
                image=image,
            )
            paths.add(resolved.path)
            foreground_counts.append(int(np.count_nonzero(resolved.mask)))
            background_counts.append(int(resolved.mask.size - np.count_nonzero(resolved.mask)))
        except Exception as exc:
            failures.append(f"{selection.plane_id}: {type(exc).__name__}: {exc}")

    mask_path = next(iter(paths)) if len(paths) == 1 else None
    if len(paths) > 1:
        failures.append("analysis resolves more than one numerical mask file")
    return (
        mask_path,
        len(selections),
        min(foreground_counts) if foreground_counts else 0,
        min(background_counts) if background_counts else 0,
        failures,
    )


def _check_original_item(
    *,
    dataset: str,
    sample: str,
    zarr_path: Path,
    mask_root: Path,
) -> InputCheckRecord:
    """Check one original OME-Zarr sample and its exact U-Net mask.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.
        zarr_path (Path): Filesystem path associated with Zarr.
        mask_root (Path): Directory used for mask.

    Returns:
        InputCheckRecord: Result produced by the operation.

    Example:
        >>> result = _check_original_item(
        ...     dataset="2d_time",
        ...     sample="sample",
        ...     zarr_path=Path("path/to/resource"),
        ...     mask_root=Path("path/to/resource"),
        ... )
    """
    failures: list[str] = []
    warnings: list[str] = []
    image: OmezarrImage | None = None
    mask_path: Path | None = None
    report_path, validation_passed, validation_issue = _validation_report_status(zarr_path)
    if not validation_passed:
        failures.append(validation_issue)

    plane_count = 0
    foreground = 0
    background = 0
    axes = "-"
    shape = "-"
    dtype = "-"

    try:
        image = read_omezarr_level0(zarr_path)
        axes = "".join(image.axes)
        shape = str(tuple(int(value) for value in image.array.shape))
        dtype = str(image.array.dtype)
        failures.extend(_check_supported_axes(image))
        channels = _selected_channels(image, dataset)
        times = _selected_times(image)
        sample_failures, sample_warnings = _check_sample_values(
            image,
            time_index=times[0],
            channel_index=channels[0],
        )
        failures.extend(sample_failures)
        warnings.extend(sample_warnings)
        (
            mask_path,
            plane_count,
            foreground,
            background,
            mask_failures,
        ) = _check_masks(
            image=image,
            mask_root=mask_root,
            dataset=dataset,
            sample=sample,
            time_indices=times,
            channel_indices=channels,
        )
        failures.extend(mask_failures)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    return InputCheckRecord(
        mode="original",
        dataset=dataset,
        sample=sample,
        variant="-",
        status=_status_from_issues(failures, warnings),
        original_zarr=str(zarr_path.resolve()),
        n2v_zarr="",
        mask_path=str(mask_path) if mask_path else str(mask_root / dataset / sample / "mask.tif"),
        validation_report=str(report_path),
        axes=axes,
        shape=shape,
        dtype=dtype,
        selected_plane_count=plane_count,
        foreground_pixels=foreground,
        background_pixels=background,
        source_frame_index="",
        source_channels="",
        issues=_render_issues(failures, warnings),
    )


def _check_n2v_item(
    *,
    dataset: str,
    sample: str,
    variant: str,
    n2v_path: Path,
    original_root: Path,
    mask_root: Path,
) -> InputCheckRecord:
    """Check one N2V output, its raw source, and shared U-Net mask.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.
        variant (str): Text value specifying variant.
        n2v_path (Path): Filesystem path associated with Noise2Void result.
        original_root (Path): Directory used for original.
        mask_root (Path): Directory used for mask.

    Returns:
        InputCheckRecord: Result produced by the operation.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _check_n2v_item(
        ...     dataset="2d_time",
        ...     sample="sample",
        ...     variant="variant",
        ...     n2v_path=Path("path/to/resource"),
        ...     original_root=Path("path/to/resource"),
        ...     mask_root=Path("path/to/resource"),
        ... )
    """
    failures: list[str] = []
    warnings: list[str] = []
    original_path = original_root / dataset / sample / "image.ome.zarr"
    mask_path: Path | None = None
    report_path = original_path.parent / VALIDATION_REPORT_NAME
    axes = "-"
    shape = "-"
    dtype = "-"
    frame_index = 0
    channels: list[int] = []
    plane_count = 0
    foreground = 0
    background = 0

    try:
        n2v_image = read_omezarr_level0(n2v_path)
        axes = "".join(n2v_image.axes)
        shape = str(tuple(int(value) for value in n2v_image.array.shape))
        dtype = str(n2v_image.array.dtype)
        failures.extend(_check_supported_axes(n2v_image))
        original_path, frame_index, channels, provenance = _source_information(
            n2v_image=n2v_image,
            original_root=original_root,
            dataset=dataset,
            sample=sample,
            variant=variant,
        )
        if "legacy" in provenance or "inferred" in provenance:
            warnings.append(provenance)
        if not original_path.is_dir():
            raise FileNotFoundError(f"matching original OME-Zarr is missing: {original_path}")

        report_path, validation_passed, validation_issue = _validation_report_status(original_path)
        if not validation_passed:
            failures.append(validation_issue)

        original_image = read_omezarr_level0(original_path)
        failures.extend(_check_supported_axes(original_image))
        time_size = _axis_size(original_image, "t")
        if frame_index < 0 or frame_index >= time_size:
            failures.append(
                f"source frame index {frame_index} is outside valid range 0..{time_size - 1}"
            )
        channel_size = _axis_size(original_image, "c")
        if "c" in original_image.axes:
            invalid_channels = [value for value in channels if value < 0 or value >= channel_size]
            if invalid_channels:
                failures.append(
                    f"source channels are outside valid range 0..{channel_size - 1}: {invalid_channels}"
                )
        elif channels not in ([], [0]):
            failures.append("source channels are declared but the original has no C axis")

        raw_yx = _yx_shape(original_image)
        n2v_yx = _yx_shape(n2v_image)
        if raw_yx != n2v_yx:
            failures.append(f"raw/N2V YX mismatch: raw={raw_yx}, N2V={n2v_yx}")

        raw_plane_count = len(channels) if "c" in original_image.axes else 1
        n2v_plane_count = _axis_size(n2v_image, "t") * _axis_size(n2v_image, "c")
        if raw_plane_count != n2v_plane_count:
            failures.append(
                f"raw/N2V plane-count mismatch: raw={raw_plane_count}, N2V={n2v_plane_count}"
            )

        n2v_failures, n2v_warnings = _check_sample_values(
            n2v_image,
            time_index=0 if "t" in n2v_image.axes else None,
            channel_index=0 if "c" in n2v_image.axes else None,
        )
        failures.extend(n2v_failures)
        warnings.extend(n2v_warnings)

        raw_times: list[int | None] = [frame_index] if "t" in original_image.axes else [None]
        raw_channels: list[int | None] = channels if "c" in original_image.axes else [None]
        (
            mask_path,
            plane_count,
            foreground,
            background,
            mask_failures,
        ) = _check_masks(
            image=original_image,
            mask_root=mask_root,
            dataset=dataset,
            sample=sample,
            time_indices=raw_times,
            channel_indices=raw_channels,
        )
        failures.extend(mask_failures)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    return InputCheckRecord(
        mode="n2v",
        dataset=dataset,
        sample=sample,
        variant=variant,
        status=_status_from_issues(failures, warnings),
        original_zarr=str(original_path),
        n2v_zarr=str(n2v_path.resolve()),
        mask_path=str(mask_path) if mask_path else str(mask_root / dataset / sample / "mask.tif"),
        validation_report=str(report_path),
        axes=axes,
        shape=shape,
        dtype=dtype,
        selected_plane_count=plane_count,
        foreground_pixels=foreground,
        background_pixels=background,
        source_frame_index=str(frame_index),
        source_channels=",".join(str(value) for value in channels),
        issues=_render_issues(failures, warnings),
    )


def _orphan_mask_records(
    *,
    original_root: Path,
    mask_root: Path,
    datasets: Sequence[str],
) -> list[InputCheckRecord]:
    """Return warning rows for exact masks with no matching original sample.

    Args:
        original_root (Path): Directory used for original.
        mask_root (Path): Directory used for mask.
        datasets (Sequence[str]): Text value specifying datasets.

    Returns:
        list[InputCheckRecord]: Collection containing the generated or selected values.

    Example:
        >>> result = _orphan_mask_records(
        ...     original_root=Path("path/to/resource"),
        ...     mask_root=Path("path/to/resource"),
        ...     datasets="datasets",
        ... )
    """
    records: list[InputCheckRecord] = []
    for dataset in datasets:
        dataset_mask_root = mask_root / dataset
        if not dataset_mask_root.is_dir():
            continue
        for mask_path in sorted(dataset_mask_root.rglob("mask.tif")):
            sample = mask_path.parent.relative_to(dataset_mask_root).as_posix()
            original = original_root / dataset / sample / "image.ome.zarr"
            if original.is_dir():
                continue
            records.append(
                InputCheckRecord(
                    mode="orphan_mask",
                    dataset=dataset,
                    sample=sample,
                    variant="-",
                    status=STATUS_WARN,
                    original_zarr=str(original),
                    n2v_zarr="",
                    mask_path=str(mask_path.resolve()),
                    validation_report="",
                    axes="-",
                    shape="-",
                    dtype="-",
                    selected_plane_count=0,
                    foreground_pixels=0,
                    background_pixels=0,
                    source_frame_index="",
                    source_channels="",
                    issues="WARN: mask.tif has no matching original image.ome.zarr",
                )
            )
    return records


def _write_csv(path: Path, rows: Sequence[InputCheckRecord]) -> None:
    """Write all readiness records to CSV with a stable header.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (Sequence[InputCheckRecord]): Value specifying rows for the operation.

    Example:
        >>> _write_csv(path=Path("path/to/resource"), rows=[])
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(InputCheckRecord)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _count_status(rows: Iterable[InputCheckRecord], status: str) -> int:
    """Count records with one status value.

    Args:
        rows (Iterable[InputCheckRecord]): Value specifying rows for the operation.
        status (str): Text value specifying status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _count_status(rows=..., status="status")
    """
    return sum(1 for row in rows if row.status == status)


def _write_summary(
    path: Path,
    *,
    mode: str,
    datasets: Sequence[str],
    rows: Sequence[InputCheckRecord],
    original_count: int,
    n2v_count: int,
) -> str:
    """Write a compact human-readable readiness summary.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        mode (str): Text value specifying mode.
        datasets (Sequence[str]): Text value specifying datasets.
        rows (Sequence[InputCheckRecord]): Value specifying rows for the operation.
        original_count (int): Number of original used by the operation.
        n2v_count (int): Number of Noise2Void result used by the operation.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _write_summary(
        ...     path=Path("path/to/resource"),
        ...     mode="mode",
        ...     datasets="datasets",
        ...     rows=[],
        ...     original_count=1,
        ...     n2v_count=1,
        ... )
    """
    lines = [
        "PFT 2D analysis input readiness report",
        "=" * 78,
        f"Generated (UTC): {_utc_now()}",
        f"Requested mode: {mode}",
        f"Datasets: {', '.join(datasets)}",
        f"Original OME-Zarr files discovered: {original_count}",
        f"N2V OME-Zarr files discovered: {n2v_count}",
        f"Records checked: {len(rows)}",
        f"PASS: {_count_status(rows, STATUS_PASS)}",
        f"WARN: {_count_status(rows, STATUS_WARN)}",
        f"FAIL: {_count_status(rows, STATUS_FAIL)}",
        "",
        "Interpretation",
        "--------------",
        "PASS: the item is ready for the requested check_2d.py analysis.",
        "WARN: the item is usable, but provenance or folder cleanup is recommended.",
        "FAIL: the item must not be analysed until the reported problem is fixed.",
    ]
    failing = [row for row in rows if row.status == STATUS_FAIL]
    warning = [row for row in rows if row.status == STATUS_WARN]
    if failing:
        lines.extend(["", "Blocking failures", "-----------------"])
        lines.extend(
            f"{row.mode} | {row.dataset} | {row.sample} | {row.variant} | {row.issues}"
            for row in failing
        )
    if warning:
        lines.extend(["", "Warnings", "--------"])
        lines.extend(
            f"{row.mode} | {row.dataset} | {row.sample} | {row.variant} | {row.issues}"
            for row in warning
        )
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _prompt_mode() -> str:
    """Prompt for original, N2V, or combined readiness checks.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _prompt_mode()
    """
    print("\nSelect data to check:")
    print("   [0] Original OME-Zarr images and U-Net masks (default)")
    print("   [1] N2V outputs, original sources, and shared masks")
    print("   [2] Check both original and N2V inputs")
    value = input("Enter number: ").strip()
    return {"": "original", "0": "original", "1": "n2v", "2": "all"}.get(
        value, "original"
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the readiness checker.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """
    parser = argparse.ArgumentParser(
        description="Check whether 2D original, mask, and N2V data are ready for analysis."
    )
    parser.add_argument(
        "--mode",
        choices=("original", "n2v", "all"),
        help="Inputs to check. When omitted, an interactive menu is shown.",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset to check. Default: all supported 2D datasets.",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=DEFAULT_ORIGINAL_ROOT,
        help="Root containing results/img/<dataset>/<sample>/image.ome.zarr.",
    )
    parser.add_argument(
        "--n2v-root",
        type=Path,
        default=DEFAULT_N2V_ROOT,
        help="Root containing standard N2V outputs.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=DEFAULT_MASK_ROOT,
        help="Root containing U-net/<dataset>/<sample>/mask.tif.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for readiness CSV and TXT reports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested checks, save reports, and return a process exit code.

    Args:
        argv (Sequence[str] | None): Optional command-line argument sequence. When omitted, arguments are read from ``sys.argv``. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    args = build_parser().parse_args(argv)
    mode = args.mode or _prompt_mode()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    original_root = args.original_root.expanduser().resolve()
    n2v_root = args.n2v_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    records: list[InputCheckRecord] = []
    originals = find_original_omezarrs(original_root, datasets)
    n2v_items = find_n2v_omezarrs(n2v_root, datasets)

    if mode in ("original", "all"):
        for dataset in datasets:
            dataset_items = [item for item in originals if item[0] == dataset]
            if not dataset_items:
                records.append(
                    InputCheckRecord(
                        mode="original",
                        dataset=dataset,
                        sample="-",
                        variant="-",
                        status=STATUS_FAIL,
                        original_zarr=str(original_root / dataset),
                        n2v_zarr="",
                        mask_path=str(mask_root / dataset),
                        validation_report="",
                        axes="-",
                        shape="-",
                        dtype="-",
                        selected_plane_count=0,
                        foreground_pixels=0,
                        background_pixels=0,
                        source_frame_index="",
                        source_channels="",
                        issues="FAIL: no original image.ome.zarr files were found for this dataset",
                    )
                )
        for dataset, sample, path in originals:
            records.append(
                _check_original_item(
                    dataset=dataset,
                    sample=sample,
                    zarr_path=path,
                    mask_root=mask_root,
                )
            )
        records.extend(
            _orphan_mask_records(
                original_root=original_root,
                mask_root=mask_root,
                datasets=datasets,
            )
        )

    if mode in ("n2v", "all"):
        for dataset in datasets:
            dataset_items = [item for item in n2v_items if item[0] == dataset]
            if not dataset_items:
                records.append(
                    InputCheckRecord(
                        mode="n2v",
                        dataset=dataset,
                        sample="-",
                        variant="-",
                        status=STATUS_FAIL,
                        original_zarr=str(original_root / dataset),
                        n2v_zarr=str(n2v_root / dataset),
                        mask_path=str(mask_root / dataset),
                        validation_report="",
                        axes="-",
                        shape="-",
                        dtype="-",
                        selected_plane_count=0,
                        foreground_pixels=0,
                        background_pixels=0,
                        source_frame_index="",
                        source_channels="",
                        issues="FAIL: no standard N2V image.ome.zarr outputs were found for this dataset",
                    )
                )
        for dataset, sample, variant, path in n2v_items:
            records.append(
                _check_n2v_item(
                    dataset=dataset,
                    sample=sample,
                    variant=variant,
                    n2v_path=path,
                    original_root=original_root,
                    mask_root=mask_root,
                )
            )

    csv_path = output_dir / "analysis_input_readiness.csv"
    txt_path = output_dir / "ANALYSIS_INPUT_READINESS.txt"
    _write_csv(csv_path, records)
    summary = _write_summary(
        txt_path,
        mode=mode,
        datasets=datasets,
        rows=records,
        original_count=len(originals),
        n2v_count=len(n2v_items),
    )

    print("\n" + summary)
    print(f"Detailed CSV: {csv_path}")
    print(f"Summary TXT: {txt_path}")
    return 1 if any(row.status == STATUS_FAIL for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
