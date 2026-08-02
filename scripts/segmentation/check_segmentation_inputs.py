"""
Audit PFT instance-segmentation inputs, provenance, metadata, and masks.

The checker can be run before or after input preparation. It verifies:

* required filtered images, MIPs, and U-Net prediction masks exist;
* the prepared input, when present, is finite float32 in [0, 1];
* the prepared input is not completely black;
* spatial dimensions and effective channel counts agree with the quantitative source;
* ``2d_time`` always uses C=0 and ignores any additional stored channel;
* raw 2D and raw-MIP products preserve the original raw dtype;
* float32 is accepted for Richardson-Lucy-deconvolved MIPs;
* OME-NGFF axes and physical Y/X coordinate scales remain consistent;
* manual training masks exist below ``results/training_files/segmentation``;
* training masks are integer instance labels rather than only binary foreground;
* one random sample is visualized as raw data and prepared normalized input.

Missing manual masks are warnings, not fatal errors, because the checker is also
used before Napari annotation. The visual report is written as HTML and contains
two PNG images for one randomly selected sample: the raw source and either the
saved normalized segmentation input or an in-memory normalize-then-mask preview
when the prepared OME-Zarr has not yet been created. Raw data are display-scaled only for viewing; the
underlying raw values are not modified.

Examples
--------
Interactive::

    python scripts/segmentation/check_segmentation_inputs.py

Check all WGA-DAPI inputs::

    python scripts/segmentation/check_segmentation_inputs.py `
        --dataset 2d_wga_dapi `
        --source-mode filtered_unet

Check deconvolved MIPs::

    python scripts/segmentation/check_segmentation_inputs.py `
        --dataset 3d_mip `
        --source-mode deconv_masked
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile as tiff

SCRIPT_FILE = Path(__file__).resolve()
CHECKER_VERSION = "2026-07-31-v4-2d-time-channel0"


def _project_root() -> Path:
    """Return project root for the supplied inputs.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _project_root()
    """
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr  # noqa: E402
from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
    _load_mip_mask,
    _load_projection_mip,
    _processing_attrs,
    create_segmentation_input,
    discover_segmentation_sources,
    select_dataset_channels,
)


def _choose(title: str, values: Sequence[str]) -> str:
    """Choose the requested operation according to the configured criteria.

    Args:
        title (str): Title displayed on the generated figure or report section.
        values (Sequence[str]): Text value specifying values.

    Returns:
        str: Generated or resolved text value.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _choose(title="title", values="values")
    """
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    selected = int(input("Select number: ").strip())
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _root_attrs(path: Path) -> dict[str, Any]:
    """Return root attrs for the supplied inputs.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _root_attrs(path=Path("path/to/resource"))
    """
    try:
        import zarr

        return dict(zarr.open_group(str(path), mode="r").attrs.asdict())
    except Exception:
        return {}


def _coordinate_scale(path: Path) -> list[float] | None:
    """Return coordinate scale for the supplied inputs.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        list[float] | None: Collection containing the generated or selected values.

    Example:
        >>> result = _coordinate_scale(path=Path("path/to/resource"))
    """
    attrs = _root_attrs(path)
    try:
        transforms = attrs["multiscales"][0]["datasets"][0].get(
            "coordinateTransformations", []
        )
        for transform in transforms:
            if transform.get("type") == "scale":
                return [float(value) for value in transform["scale"]]
    except Exception:
        return None
    return None


def _inspect_zarr(path: Path, *, load_values: bool = False) -> dict[str, Any]:
    """Return inspect Zarr for the supplied inputs.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        load_values (bool): Boolean flag controlling load values. Defaults to ``False``.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _inspect_zarr(path=Path("path/to/resource"))
    """
    if not path.is_dir():
        raise FileNotFoundError(path)
    array, axes = load_ome_zarr(path, level=0, as_numpy=load_values)
    result: dict[str, Any] = {
        "path": str(path),
        "axes": str(axes).lower(),
        "shape": tuple(int(value) for value in array.shape),
        "dtype": str(np.dtype(array.dtype)),
        "scale": _coordinate_scale(path),
        "attrs": _root_attrs(path),
    }
    if load_values:
        values = np.asarray(array)
        result.update(
            {
                "finite": bool(np.all(np.isfinite(values))),
                "minimum": float(np.min(values)) if values.size else float("nan"),
                "maximum": float(np.max(values)) if values.size else float("nan"),
                "nonzero": int(np.count_nonzero(values)),
            }
        )
    return result


def _axis_size(info: dict[str, Any], axis: str) -> int | None:
    """Return axis size for the supplied inputs.

    Args:
        info (dict[str, Any]): Text value specifying info.
        axis (str): Array axis along which the operation is performed.

    Returns:
        int | None: Computed numerical result.

    Example:
        >>> result = _axis_size(info="info", axis="axis")
    """
    axes = info["axes"]
    if axis not in axes:
        return None
    return int(info["shape"][axes.index(axis)])


def _axis_scale(info: dict[str, Any], axis: str) -> float | None:
    """Return axis scale for the supplied inputs.

    Args:
        info (dict[str, Any]): Text value specifying info.
        axis (str): Array axis along which the operation is performed.

    Returns:
        float | None: Computed numerical result.

    Example:
        >>> result = _axis_scale(info="info", axis="axis")
    """
    scale = info.get("scale")
    axes = info["axes"]
    if scale is None or axis not in axes or len(scale) != len(axes):
        return None
    return float(scale[axes.index(axis)])


def _add(
    issues: list[dict[str, str]],
    severity: str,
    check: str,
    message: str,
) -> None:
    """Add the requested operation to the current data structure.

    Args:
        issues (list[dict[str, str]]): Text value specifying issues.
        severity (str): Text value specifying severity.
        check (str): Text value specifying check.
        message (str): Text value specifying message.

    Example:
        >>> _add(
        ...     issues="issues",
        ...     severity="severity",
        ...     check="check",
        ...     message="message",
        ... )
    """
    issues.append({"severity": severity, "check": check, "message": message})


def _channel_count(info: dict[str, Any]) -> int:
    """Return the stored numerical channel count, treating YX as one channel.

    Args:
        info (dict[str, Any]): Text value specifying info.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _channel_count(info="info")
    """
    size = _axis_size(info, "c")
    return 1 if size is None else int(size)


def _effective_channel_count(info: dict[str, Any], dataset: str) -> int:
    """Return the channel count after applying the dataset channel policy.

    Args:
        info (dict[str, Any]): Text value specifying info.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = _effective_channel_count(info="info", dataset="2d_time")
    """
    observed = _channel_count(info)
    if dataset == "2d_time":
        return 1 if observed >= 1 else 0
    return observed


def _compare_channel_count(
    left: dict[str, Any],
    right: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    dataset: str,
    check_prefix: str,
) -> None:
    """Compare effective channels after applying the dataset channel policy.

    Args:
        left (dict[str, Any]): Text value specifying left.
        right (dict[str, Any]): Text value specifying right.
        issues (list[dict[str, str]]): Text value specifying issues.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        check_prefix (str): Text value specifying check prefix.

    Example:
        >>> _compare_channel_count(
        ...     left="left",
        ...     right="right",
        ...     issues="issues",
        ...     dataset="2d_time",
        ...     check_prefix="check_prefix",
        ... )
    """
    left_channels = _effective_channel_count(left, dataset)
    right_channels = _effective_channel_count(right, dataset)
    if left_channels != right_channels:
        _add(
            issues,
            "FAIL",
            f"{check_prefix}_channel_count",
            f"Effective numerical channel count differs: {left_channels} versus "
            f"{right_channels} (stored axes {left['axes']} versus {right['axes']})",
        )


def _check_expected_channels(
    info: dict[str, Any],
    dataset: str,
    issues: list[dict[str, str]],
    *,
    check_prefix: str,
) -> None:
    """Validate stored channels under the defined dataset-specific policy.

    Args:
        info (dict[str, Any]): Text value specifying info.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        issues (list[dict[str, str]]): Text value specifying issues.
        check_prefix (str): Text value specifying check prefix.

    Example:
        >>> _check_expected_channels(
        ...     info="info",
        ...     dataset="2d_time",
        ...     issues="issues",
        ...     check_prefix="check_prefix",
        ... )
    """
    observed = _channel_count(info)
    if dataset == "2d_time":
        if observed < 1:
            _add(
                issues,
                "FAIL",
                f"{check_prefix}_expected_channels",
                "2d_time contains no usable numerical channel",
            )
        return

    expected = 2 if dataset == "2d_wga_dapi" else 1
    if observed != expected:
        _add(
            issues,
            "FAIL",
            f"{check_prefix}_expected_channels",
            f"{dataset} expects {expected} numerical channel(s), received {observed} "
            f"with axes {info['axes']}",
        )


def _compare_named_axes(
    left: dict[str, Any],
    right: dict[str, Any],
    axes: Sequence[str],
    issues: list[dict[str, str]],
    *,
    check_prefix: str,
) -> None:
    """Compare named axes across the supplied inputs.

    Args:
        left (dict[str, Any]): Text value specifying left.
        right (dict[str, Any]): Text value specifying right.
        axes (Sequence[str]): Axis specification describing the dimensional order of the image data.
        issues (list[dict[str, str]]): Text value specifying issues.
        check_prefix (str): Text value specifying check prefix.

    Example:
        >>> _compare_named_axes(
        ...     left="left",
        ...     right="right",
        ...     axes="axes",
        ...     issues="issues",
        ...     check_prefix="check_prefix",
        ... )
    """
    for axis in axes:
        left_size = _axis_size(left, axis)
        right_size = _axis_size(right, axis)
        if left_size is None or right_size is None:
            _add(
                issues,
                "FAIL",
                f"{check_prefix}_{axis}_axis",
                f"Missing {axis.upper()} axis: left={left['axes']}, right={right['axes']}",
            )
        elif left_size != right_size:
            _add(
                issues,
                "FAIL",
                f"{check_prefix}_{axis}_shape",
                f"{axis.upper()} size differs: {left_size} versus {right_size}",
            )


def _compare_yx_scale(
    left: dict[str, Any],
    right: dict[str, Any],
    issues: list[dict[str, str]],
    *,
    check_prefix: str,
) -> None:
    """Compare yx scale across the supplied inputs.

    Args:
        left (dict[str, Any]): Text value specifying left.
        right (dict[str, Any]): Text value specifying right.
        issues (list[dict[str, str]]): Text value specifying issues.
        check_prefix (str): Text value specifying check prefix.

    Example:
        >>> _compare_yx_scale(
        ...     left="left",
        ...     right="right",
        ...     issues="issues",
        ...     check_prefix="check_prefix",
        ... )
    """
    for axis in ("y", "x"):
        left_scale = _axis_scale(left, axis)
        right_scale = _axis_scale(right, axis)
        if left_scale is None or right_scale is None:
            _add(
                issues,
                "WARN",
                f"{check_prefix}_{axis}_scale",
                f"Could not compare {axis.upper()} physical scale metadata",
            )
        elif not np.isclose(left_scale, right_scale, rtol=1e-6, atol=1e-9):
            _add(
                issues,
                "FAIL",
                f"{check_prefix}_{axis}_scale",
                f"{axis.upper()} physical scale differs: {left_scale} versus {right_scale}",
            )


def _inspect_training_mask(
    mask_path: Path,
    expected_yx: tuple[int, int] | None,
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    """Return inspect training mask for the supplied inputs.

    Args:
        mask_path (Path): Filesystem path associated with mask.
        expected_yx (tuple[int, int] | None): Numerical value controlling expected yx.
        issues (list[dict[str, str]]): Text value specifying issues.

    Returns:
        dict[str, Any] | None: Mapping containing the generated or resolved values.

    Example:
        >>> result = _inspect_training_mask(
        ...     mask_path=Path("path/to/resource"),
        ...     expected_yx=1,
        ...     issues="issues",
        ... )
    """
    if not mask_path.is_file():
        _add(
            issues,
            "WARN",
            "training_mask_exists",
            f"Manual instance mask is not created yet: {mask_path}",
        )
        return None
    try:
        mask = np.asarray(tiff.imread(mask_path))
    except Exception as exc:
        _add(issues, "FAIL", "training_mask_readable", f"Cannot read mask: {exc}")
        return None
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        _add(
            issues,
            "FAIL",
            "training_mask_ndim",
            f"Manual MIP/2D mask must be YX, received shape={mask.shape}",
        )
    if not np.issubdtype(mask.dtype, np.integer):
        _add(
            issues,
            "FAIL",
            "training_mask_dtype",
            f"Instance mask must use an integer dtype, received {mask.dtype}",
        )
    if np.any(mask < 0):
        _add(issues, "FAIL", "training_mask_negative", "Instance mask contains negative labels")
    if expected_yx is not None and tuple(mask.shape[-2:]) != tuple(expected_yx):
        _add(
            issues,
            "FAIL",
            "training_mask_shape",
            f"Mask YX {mask.shape[-2:]} differs from input YX {expected_yx}",
        )
    labels = np.unique(mask)
    positive = labels[labels > 0]
    if positive.size == 0:
        _add(issues, "FAIL", "training_mask_foreground", "Mask contains no labelled cell")
    elif positive.size == 1 and positive[0] == 1:
        _add(
            issues,
            "WARN",
            "training_mask_instance_labels",
            "Mask is binary (0/1). Cellpose, Omnipose, and StarDist training require "
            "a unique positive integer for each cell instance.",
        )
    return {
        "dtype": str(mask.dtype),
        "shape": list(mask.shape),
        "positive_instance_ids": int(positive.size),
        "maximum_label": int(np.max(mask)) if mask.size else 0,
    }



def _reduce_to_preview_cyx(
    array: Any,
    axes: str,
    *,
    dataset: str,
) -> tuple[np.ndarray, str]:
    """Reduce an OME array to CYX for a report preview.

    Time and other non-spatial axes are sampled at their middle index. For a
    3D volume used in the MIP workflow, Z is reduced by maximum-intensity
    projection so that the raw preview is spatially comparable with the saved
    normalized MIP.

    Args:
        array (Any): Value specifying array for the operation.
        axes (str): Axis specification describing the dimensional order of the image data.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _reduce_to_preview_cyx(
        ...     array=image_array,
        ...     axes="axes",
        ...     dataset="2d_time",
        ... )
    """
    current = array
    current_axes = str(axes).lower()
    if "y" not in current_axes or "x" not in current_axes:
        raise ValueError(f"Preview requires Y and X axes, received {current_axes!r}")

    selections: list[str] = []
    for axis in tuple(current_axes):
        if axis in {"c", "y", "x"}:
            continue
        axis_index = current_axes.index(axis)
        if axis == "z" and dataset == "3d_mip":
            current = current.max(axis=axis_index)
            selections.append("Z=max projection")
        else:
            axis_size = int(current.shape[axis_index])
            selected_index = axis_size // 2
            slicing = [slice(None)] * int(current.ndim)
            slicing[axis_index] = selected_index
            current = current[tuple(slicing)]
            selections.append(f"{axis.upper()}={selected_index}")
        current_axes = current_axes[:axis_index] + current_axes[axis_index + 1 :]

    current = np.asarray(current)
    if "c" not in current_axes:
        current = current[None, ...]
        current_axes = "c" + current_axes

    order = [current_axes.index("c"), current_axes.index("y"), current_axes.index("x")]
    current = np.transpose(current, order)
    return np.asarray(current), ", ".join(selections) or "full YX plane"


def _load_preview_cyx(path: Path, *, dataset: str) -> tuple[np.ndarray, str, str]:
    """Load one display plane while avoiding full eager loading when possible.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        tuple[np.ndarray, str, str]: Collection containing the generated or selected values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_preview_cyx(path=Path("path/to/resource"), dataset="2d_time")
    """
    if not path.is_dir():
        raise FileNotFoundError(path)
    array, axes = load_ome_zarr(path, level=0, as_numpy=False)
    cyx, selection = _reduce_to_preview_cyx(array, axes, dataset=dataset)
    return cyx, str(axes).lower(), selection


def _display_normalize_plane(plane: np.ndarray) -> np.ndarray:
    """Robustly scale one raw channel for display only.

    Args:
        plane (np.ndarray): Array containing plane.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _display_normalize_plane(plane=image_array)
    """
    values = np.asarray(plane, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.8))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _channels_to_rgb(
    channels_cyx: np.ndarray,
    *,
    dataset: str,
    raw_display_scaling: bool,
) -> np.ndarray:
    """Create an RGB preview while preserving the defined fluorescence colours.

    Args:
        channels_cyx (np.ndarray): Array containing channels cyx.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        raw_display_scaling (bool): Boolean flag controlling raw display scaling.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _channels_to_rgb(
        ...     channels_cyx=image_array,
        ...     dataset="2d_time",
        ...     raw_display_scaling=True,
        ... )
    """
    channels = np.asarray(channels_cyx)
    if channels.ndim != 3:
        raise ValueError(f"Expected CYX preview data, received {channels.shape}")

    planes: list[np.ndarray] = []
    for channel in channels:
        if raw_display_scaling:
            planes.append(_display_normalize_plane(channel))
        else:
            plane = np.asarray(channel, dtype=np.float32)
            plane = np.nan_to_num(plane, nan=0.0, posinf=1.0, neginf=0.0)
            planes.append(np.clip(plane, 0.0, 1.0))

    height, width = channels.shape[-2:]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    if dataset == "2d_time":
        # HADA is displayed in blue.
        rgb[..., 2] = planes[0]
    elif dataset == "2d_wga_dapi" and len(planes) >= 2:
        # Numerical channel 0 is DAPI (blue); channel 1 is WGA (green).
        rgb[..., 2] = planes[0]
        rgb[..., 1] = planes[1]
    elif len(planes) == 1:
        rgb[...] = planes[0][..., None]
    elif len(planes) == 2:
        rgb[..., 1] = planes[0]
        rgb[..., 2] = planes[1]
    else:
        for channel_index in range(min(3, len(planes))):
            rgb[..., channel_index] = planes[channel_index]
    return np.clip(rgb, 0.0, 1.0)


def _save_preview_image(
    path: Path,
    rgb: np.ndarray | None,
    *,
    title: str,
    subtitle: str,
    placeholder: str | None = None,
) -> None:
    """Save one report image with an informative title.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rgb (np.ndarray | None): Array containing RGB representation.
        title (str): Title displayed on the generated figure or report section.
        subtitle (str): Text value specifying subtitle.
        placeholder (str | None): Text value specifying placeholder. ``None`` selects the function's default behavior.

    Example:
        >>> _save_preview_image(
        ...     path=Path("path/to/resource"),
        ...     rgb=image_array,
        ...     title="title",
        ...     subtitle="subtitle",
        ... )
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    if rgb is None:
        axis.set_facecolor("white")
        axis.text(
            0.5,
            0.5,
            placeholder or "Image unavailable",
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
            fontsize=12,
        )
    else:
        axis.imshow(rgb, interpolation="nearest")
    axis.set_title(f"{title}\n{subtitle}", fontsize=11)
    axis.set_axis_off()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _build_unsaved_normalized_preview(record: Any) -> tuple[np.ndarray, str]:
    """Build the exact normalize-then-mask input in memory without saving data.

    Args:
        record (Any): Value specifying record for the operation.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _build_unsaved_normalized_preview(record=...)
    """
    if record.dataset == "3d_mip" and record.source_mode in {"raw_masked", "deconv_masked"}:
        intensity, intensity_axes, _ = _load_projection_mip(record.intensity_source_zarr)
        if record.mask_zarr is None:
            raise FileNotFoundError("Masked MIP has no resolved U-Net mask")
        processing = _processing_attrs(record.source_zarr)
        target_slice = processing.get("target_slice_1based_for_qc_and_mask")
        mask, mask_axes = _load_mip_mask(
            record.mask_zarr,
            target_slice_1based=int(target_slice) if target_slice is not None else None,
            expected_yx=tuple(int(value) for value in intensity.shape[-2:]),
        )
    else:
        intensity, intensity_axes = load_ome_zarr(
            record.intensity_source_zarr,
            level=0,
            as_numpy=True,
        )
        intensity = np.asarray(intensity)
        mask = None
        mask_axes = None
        if record.mask_zarr is not None:
            mask, mask_axes = load_ome_zarr(record.mask_zarr, level=0, as_numpy=True)
            mask = np.asarray(mask)

    selected_intensity, selected_axes, _ = select_dataset_channels(
        np.asarray(intensity),
        str(intensity_axes).lower(),
        record.dataset,
    )
    normalized, _ = create_segmentation_input(
        selected_intensity,
        selected_axes,
        predicted_mask=mask,
        mask_axes=str(mask_axes).lower() if mask_axes is not None else None,
    )
    return normalized, selected_axes


def _create_random_visualization(
    records: Sequence[Any],
    report_root: Path,
    *,
    dataset: str,
    seed: int | None,
) -> dict[str, Any]:
    """Create raw and normalized PNG previews for one randomly selected sample.

    Args:
        records (Sequence[Any]): Value specifying records for the operation.
        report_root (Path): Directory used for report.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        seed (int | None): Random seed used to make sampling, splitting, or initialization reproducible.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _create_random_visualization(
        ...     records=[],
        ...     report_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     seed=1,
        ... )
    """
    if not records:
        raise ValueError("No records are available for visualization")

    prepared_records = [record for record in records if record.output_zarr.is_dir()]
    candidates = prepared_records or list(records)
    chooser: random.Random | random.SystemRandom
    chooser = random.Random(seed) if seed is not None else random.SystemRandom()
    record = chooser.choice(candidates)

    raw_path = record.raw_reference_zarr
    if raw_path is None or not raw_path.is_dir():
        raw_path = record.intensity_source_zarr
    raw_png = report_root / "random_sample_raw.png"
    normalized_png = report_root / "random_sample_normalized.png"

    raw_cyx, raw_axes, raw_selection = _load_preview_cyx(raw_path, dataset=dataset)
    raw_rgb = _channels_to_rgb(raw_cyx, dataset=dataset, raw_display_scaling=True)
    _save_preview_image(
        raw_png,
        raw_rgb,
        title="Raw/original quantitative image",
        subtitle=(
            f"Sample: {record.sample_key} | dtype={raw_cyx.dtype} | "
            f"axes={raw_axes} | {raw_selection} | display-scaled P1-P99.8"
        ),
    )

    normalized_available = record.output_zarr.is_dir()
    normalized_preview_generated = False
    normalized_axes = ""
    normalized_selection = ""
    normalized_dtype = ""
    normalized_shape: list[int] | None = None
    normalized_source = "saved segmentation_input.ome.zarr"
    try:
        if normalized_available:
            normalized_cyx, normalized_axes, normalized_selection = _load_preview_cyx(
                record.output_zarr,
                dataset=dataset,
            )
            title = "Prepared normalized segmentation input"
        else:
            normalized_array, normalized_axes = _build_unsaved_normalized_preview(record)
            normalized_cyx, normalized_selection = _reduce_to_preview_cyx(
                normalized_array,
                normalized_axes,
                dataset=dataset,
            )
            normalized_preview_generated = True
            normalized_source = "in-memory preview; no OME-Zarr was written"
            title = "Prospective normalized segmentation input"

        normalized_dtype = str(normalized_cyx.dtype)
        normalized_shape = [int(value) for value in normalized_cyx.shape]
        normalized_rgb = _channels_to_rgb(
            normalized_cyx,
            dataset=dataset,
            raw_display_scaling=False,
        )
        _save_preview_image(
            normalized_png,
            normalized_rgb,
            title=title,
            subtitle=(
                f"Sample: {record.sample_key} | dtype={normalized_cyx.dtype} | "
                f"axes={normalized_axes} | {normalized_selection} | actual [0,1] values | "
                f"{normalized_source}"
            ),
        )
    except Exception as exc:
        normalized_source = f"preview failed: {type(exc).__name__}: {exc}"
        _save_preview_image(
            normalized_png,
            None,
            title="Normalized segmentation input preview",
            subtitle=f"Sample: {record.sample_key}",
            placeholder=(
                "Normalized preview could not be generated.\n"
                f"{type(exc).__name__}: {exc}"
            ),
        )

    return {
        "sample_key": record.sample_key,
        "selection_method": "random among prepared samples" if prepared_records else "random among discovered samples",
        "seed": seed,
        "raw_source_zarr": str(raw_path),
        "raw_png": str(raw_png),
        "raw_axes": raw_axes,
        "raw_selection": raw_selection,
        "raw_dtype": str(raw_cyx.dtype),
        "raw_shape_cyx": [int(value) for value in raw_cyx.shape],
        "raw_display_note": "P1-P99.8 scaling is used only in the PNG visualization",
        "normalized_input_zarr": str(record.output_zarr),
        "normalized_png": str(normalized_png),
        "normalized_available": normalized_available,
        "normalized_preview_generated": normalized_preview_generated,
        "normalized_source": normalized_source,
        "normalized_axes": normalized_axes,
        "normalized_selection": normalized_selection,
        "normalized_dtype": normalized_dtype,
        "normalized_shape_cyx": normalized_shape,
        "normalized_display_note": (
            "The [0,1] segmentation input is displayed without additional normalization; "
            "it is read from disk when available or calculated in memory for the pre-check"
        ),
    }


def audit_record(record) -> dict[str, Any]:
    """Return audit record for the supplied inputs.

    Args:
        record (Any): Value specifying record for the operation.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = audit_record(record=...)
    """
    issues: list[dict[str, str]] = []
    source_info: dict[str, Any] | None = None
    intensity_info: dict[str, Any] | None = None
    raw_info: dict[str, Any] | None = None
    output_info: dict[str, Any] | None = None
    mask_info: dict[str, Any] | None = None

    try:
        source_info = _inspect_zarr(record.source_zarr, load_values=True)
    except Exception as exc:
        _add(issues, "FAIL", "source_exists", f"Source OME-Zarr is unavailable: {exc}")

    try:
        intensity_info = _inspect_zarr(record.intensity_source_zarr, load_values=False)
    except Exception as exc:
        _add(
            issues,
            "FAIL",
            "intensity_source_exists",
            f"Complete intensity source is unavailable: {exc}",
        )

    if record.mask_zarr is not None:
        try:
            mask_info = _inspect_zarr(record.mask_zarr, load_values=False)
        except Exception as exc:
            _add(issues, "FAIL", "unet_mask_exists", f"Predicted U-Net mask is unavailable: {exc}")
    elif record.source_mode in {"filtered_unet", "raw_masked", "deconv_masked"}:
        _add(issues, "FAIL", "unet_mask_recorded", "Required U-Net mask path is missing")

    if record.raw_reference_zarr is not None:
        try:
            raw_info = _inspect_zarr(record.raw_reference_zarr, load_values=False)
        except Exception as exc:
            _add(issues, "WARN", "raw_reference_exists", f"Cannot inspect raw reference: {exc}")
    else:
        _add(
            issues,
            "WARN",
            "raw_reference_resolved",
            "Raw reference OME-Zarr could not be resolved for dtype/metadata comparison",
        )

    if source_info:
        _check_expected_channels(
            source_info,
            record.dataset,
            issues,
            check_prefix="source",
        )
    if intensity_info:
        _check_expected_channels(
            intensity_info,
            record.dataset,
            issues,
            check_prefix="intensity_source",
        )

    if source_info and raw_info:
        _compare_named_axes(source_info, raw_info, ("y", "x"), issues, check_prefix="source_raw")
        _compare_channel_count(source_info, raw_info, issues, dataset=record.dataset, check_prefix="source_raw")
        _compare_yx_scale(source_info, raw_info, issues, check_prefix="source_raw")
        if record.source_mode in {"filtered_unet", "raw_unmasked", "raw_masked"}:
            if source_info["dtype"] != raw_info["dtype"]:
                _add(
                    issues,
                    "FAIL",
                    "source_raw_dtype",
                    f"Intensity-preserving source dtype changed: {source_info['dtype']} "
                    f"versus raw {raw_info['dtype']}",
                )
        elif record.source_mode == "deconv_masked" and source_info["dtype"] != "float32":
            _add(
                issues,
                "FAIL",
                "deconvolved_dtype",
                f"Deconvolved MIP is expected to be float32, received {source_info['dtype']}",
            )

    if source_info and intensity_info:
        # The deconvolved intensity source is a CZYX volume while the saved source is a CYX MIP.
        _compare_named_axes(source_info, intensity_info, ("y", "x"), issues, check_prefix="source_intensity")
        _compare_channel_count(source_info, intensity_info, issues, dataset=record.dataset, check_prefix="source_intensity")

    if record.output_zarr.is_dir():
        try:
            output_info = _inspect_zarr(record.output_zarr, load_values=True)
            if output_info["dtype"] != "float32":
                _add(
                    issues,
                    "FAIL",
                    "prepared_dtype",
                    f"Prepared input must be float32, received {output_info['dtype']}",
                )
            if not output_info.get("finite", False):
                _add(issues, "FAIL", "prepared_finite", "Prepared input contains NaN or infinite values")
            if output_info.get("minimum", -1.0) < 0.0 or output_info.get("maximum", 2.0) > 1.0:
                _add(
                    issues,
                    "FAIL",
                    "prepared_range",
                    f"Prepared range is {output_info.get('minimum')}..{output_info.get('maximum')}, expected [0,1]",
                )
            if output_info.get("nonzero", 0) == 0:
                _add(issues, "FAIL", "prepared_nonzero", "Prepared input is completely black")
            if source_info:
                _compare_named_axes(output_info, source_info, ("y", "x"), issues, check_prefix="prepared_source")
                _compare_channel_count(output_info, source_info, issues, dataset=record.dataset, check_prefix="prepared_source")
                _compare_yx_scale(output_info, source_info, issues, check_prefix="prepared_source")
            processing = output_info["attrs"].get("pft_processing", {})
            if processing.get("normalization_order") != "normalize_complete_source_then_apply_mask":
                _add(
                    issues,
                    "FAIL",
                    "prepared_normalization_order",
                    "Prepared input lacks the required normalize-then-mask provenance",
                )
            if processing.get("model_normalization_required") is not False:
                _add(
                    issues,
                    "FAIL",
                    "model_normalization_disabled",
                    "Prepared metadata must state that model-side normalization is disabled",
                )
            if record.dataset == "2d_time":
                channel_selection = processing.get("channel_selection", {})
                if channel_selection.get("policy") != "use_channel_0_only":
                    _add(
                        issues,
                        "FAIL",
                        "prepared_channel_policy",
                        "2d_time prepared input must record the use_channel_0_only policy",
                    )
                elif channel_selection.get("selected_channels") != [0]:
                    _add(
                        issues,
                        "FAIL",
                        "prepared_selected_channel",
                        "2d_time prepared input must record selected channel [0]",
                    )
        except Exception as exc:
            _add(issues, "FAIL", "prepared_readable", f"Cannot validate prepared input: {exc}")
    else:
        _add(
            issues,
            "WARN",
            "prepared_input_exists",
            f"Prepared input is not created yet: {record.output_zarr}",
        )

    expected_yx: tuple[int, int] | None = None
    reference_info = output_info or source_info
    if reference_info and "y" in reference_info["axes"] and "x" in reference_info["axes"]:
        expected_yx = (
            int(reference_info["shape"][reference_info["axes"].index("y")]),
            int(reference_info["shape"][reference_info["axes"].index("x")]),
        )
    training_mask_info = _inspect_training_mask(record.training_mask, expected_yx, issues)

    severities = [issue["severity"] for issue in issues]
    status = "FAIL" if "FAIL" in severities else ("WARN" if "WARN" in severities else "PASS")
    return {
        "dataset": record.dataset,
        "source_mode": record.source_mode,
        "sample_key": record.sample_key,
        "status": status,
        "source_zarr": str(record.source_zarr),
        "intensity_source_zarr": str(record.intensity_source_zarr),
        "mask_zarr": str(record.mask_zarr) if record.mask_zarr else "",
        "raw_reference_zarr": str(record.raw_reference_zarr) if record.raw_reference_zarr else "",
        "prepared_input_zarr": str(record.output_zarr),
        "training_mask": str(record.training_mask),
        "source_dtype": source_info["dtype"] if source_info else "",
        "stored_source_channels": _channel_count(source_info) if source_info else "",
        "effective_segmentation_channels": (
            _effective_channel_count(source_info, record.dataset) if source_info else ""
        ),
        "channel_policy": "use_channel_0_only" if record.dataset == "2d_time" else "retain_defined_channels",
        "raw_dtype": raw_info["dtype"] if raw_info else "",
        "prepared_dtype": output_info["dtype"] if output_info else "",
        "prepared_min": output_info.get("minimum", "") if output_info else "",
        "prepared_max": output_info.get("maximum", "") if output_info else "",
        "prepared_nonzero": output_info.get("nonzero", "") if output_info else "",
        "training_instances": (
            training_mask_info.get("positive_instance_ids", "") if training_mask_info else ""
        ),
        "issues": issues,
    }


def _write_reports(
    rows: list[dict[str, Any]],
    report_root: Path,
    *,
    visualization: dict[str, Any] | None,
) -> tuple[Path, Path, Path, Path]:
    """Write reports to persistent storage.

    Args:
        rows (list[dict[str, Any]]): Text value specifying rows.
        report_root (Path): Directory used for report.
        visualization (dict[str, Any] | None): Text value specifying visualization.

    Returns:
        tuple[Path, Path, Path, Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _write_reports(
        ...     rows="rows",
        ...     report_root=Path("path/to/resource"),
        ...     visualization="visualization",
        ... )
    """
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "segmentation_input_check.csv"
    json_path = report_root / "segmentation_input_check.json"
    txt_path = report_root / "segmentation_input_check.txt"
    html_path = report_root / "segmentation_input_check.html"

    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["issues"] = " | ".join(
            f"{issue['severity']}:{issue['check']}:{issue['message']}" for issue in row["issues"]
        )
        flat_rows.append(flat)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "PASS": sum(row["status"] == "PASS" for row in rows),
            "WARN": sum(row["status"] == "WARN" for row in rows),
            "FAIL": sum(row["status"] == "FAIL" for row in rows),
        },
        "visualization": visualization,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "PFT SEGMENTATION INPUT CHECK",
        "=" * 72,
        f"Generated UTC: {payload['generated_utc']}",
        f"PASS: {payload['counts']['PASS']}",
        f"WARN: {payload['counts']['WARN']}",
        f"FAIL: {payload['counts']['FAIL']}",
        "",
    ]
    if visualization is not None:
        lines.extend(
            [
                "RANDOM VISUAL CHECK",
                "-" * 72,
                f"Sample: {visualization['sample_key']}",
                f"Raw PNG: {visualization['raw_png']}",
                f"Normalized PNG: {visualization['normalized_png']}",
                (
                    "Saved normalized input available: "
                    f"{visualization['normalized_available']}"
                ),
                (
                    "In-memory normalized preview generated: "
                    f"{visualization.get('normalized_preview_generated', False)}"
                ),
                "",
            ]
        )
    for row in rows:
        lines.append(f"[{row['status']}] {row['sample_key']}")
        if not row["issues"]:
            lines.append("  All checks passed")
        for issue in row["issues"]:
            lines.append(f"  {issue['severity']} {issue['check']}: {issue['message']}")
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    issue_rows: list[str] = []
    for row in rows:
        messages = "<br>".join(
            html.escape(
                f"{issue['severity']} {issue['check']}: {issue['message']}"
            )
            for issue in row["issues"]
        ) or "All checks passed"
        issue_rows.append(
            "<tr>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td>{html.escape(row['sample_key'])}</td>"
            f"<td>{messages}</td>"
            "</tr>"
        )

    visualization_html = "<p>Visualization was disabled.</p>"
    if visualization is not None:
        raw_name = Path(visualization["raw_png"]).name
        normalized_name = Path(visualization["normalized_png"]).name
        if visualization["normalized_available"]:
            normalized_note = (
                "The right image shows the saved normalized segmentation input without "
                "additional display normalization."
            )
        elif visualization.get("normalized_preview_generated"):
            normalized_note = (
                "The prepared OME-Zarr is not created yet. The right image was calculated "
                "in memory using the same normalize-then-mask procedure and was not saved as data."
            )
        else:
            normalized_note = "The normalized preview could not be generated."
        visualization_html = f"""
        <h2>Random visual check</h2>
        <p><strong>Sample:</strong> {html.escape(visualization['sample_key'])}</p>
        <p>The raw image is percentile-scaled only for visualization. Raw numerical values are not changed. {html.escape(normalized_note)}</p>
        <div class="images">
          <figure>
            <img src="{html.escape(raw_name)}" alt="Raw sample preview">
            <figcaption>Raw/original source</figcaption>
          </figure>
          <figure>
            <img src="{html.escape(normalized_name)}" alt="Normalized sample preview">
            <figcaption>Prepared normalized segmentation input</figcaption>
          </figure>
        </div>
        """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PFT segmentation input check</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; line-height: 1.4; }}
h1, h2 {{ margin-bottom: 0.4rem; }}
.summary {{ display: flex; gap: 1rem; margin: 1rem 0; }}
.summary span {{ border: 1px solid #bbb; border-radius: 6px; padding: 0.5rem 0.8rem; }}
.images {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; align-items: start; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; border: 1px solid #bbb; }}
figcaption {{ margin-top: 0.4rem; font-weight: bold; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ border: 1px solid #bbb; padding: 0.45rem; text-align: left; vertical-align: top; }}
th {{ background: #eee; }}
@media (max-width: 900px) {{ .images {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>PFT segmentation input check</h1>
<p><strong>Generated UTC:</strong> {html.escape(payload['generated_utc'])}</p>
<div class="summary">
  <span>PASS: {payload['counts']['PASS']}</span>
  <span>WARN: {payload['counts']['WARN']}</span>
  <span>FAIL: {payload['counts']['FAIL']}</span>
</div>
{visualization_html}
<h2>Per-sample checks</h2>
<table>
<thead><tr><th>Status</th><th>Sample</th><th>Issues</th></tr></thead>
<tbody>{''.join(issue_rows)}</tbody>
</table>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return csv_path, json_path, txt_path, html_path


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Check segmentation sources, prepared inputs, metadata, and manual masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--filter-root", type=Path, default=None)
    parser.add_argument("--unet-root", type=Path, default=None)
    parser.add_argument("--mip-root", type=Path, default=None)
    parser.add_argument("--sample", default=None, help="Check one exact sample key")
    parser.add_argument(
        "--preview-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random preview-sample selection",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip the random raw-versus-normalized PNG and HTML visualization",
    )
    args = parser.parse_args()

    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is not valid for {dataset!r}: {valid_modes}")

    records = discover_segmentation_sources(
        PROJECT_ROOT,
        dataset,
        source_mode,
        filter_root=args.filter_root,
        unet_root=args.unet_root,
        mip_root=args.mip_root,
    )
    if args.sample:
        records = [record for record in records if record.sample_key == args.sample]
        if not records:
            parser.error(f"Unknown sample key: {args.sample}")

    print("\nChecking segmentation workflow inputs")
    print(f"Checker version: {CHECKER_VERSION}")
    print(f"Script path:     {SCRIPT_FILE}")
    print("=" * 72)
    print(f"Dataset:       {dataset}")
    print(f"Source mode:   {source_mode}")
    print(f"Samples:       {len(records)}")
    if dataset == "2d_time":
        print("Channel policy: use C=0 only; additional channels are ignored")
    else:
        print("Channel policy: retain the dataset-defined numerical channels")
    print("Mask root:     results/training_files/segmentation")

    rows = []
    for index, record in enumerate(records, start=1):
        row = audit_record(record)
        rows.append(row)
        print(f"[{index:03d}/{len(records):03d}] {row['status']:4s} {record.sample_key}")
        for issue in row["issues"]:
            if issue["severity"] in {"FAIL", "WARN"}:
                print(f"    {issue['severity']}: {issue['message']}")

    report_root = PROJECT_ROOT / "results" / "training_files" / "segmentation" / "checks" / dataset / source_mode
    visualization: dict[str, Any] | None = None
    if not args.no_visualization:
        try:
            visualization = _create_random_visualization(
                records,
                report_root,
                dataset=dataset,
                seed=args.preview_seed,
            )
            print("\nRandom visual check:")
            print(f"  sample:      {visualization['sample_key']}")
            print(f"  raw image:   {visualization['raw_png']}")
            print(f"  normalized:  {visualization['normalized_png']}")
        except Exception as exc:
            print(f"\nWARN: Random visualization could not be created: {exc}")

    for row in rows:
        row["preview_selected"] = bool(
            visualization is not None
            and row["sample_key"] == visualization["sample_key"]
        )

    csv_path, json_path, txt_path, html_path = _write_reports(
        rows,
        report_root,
        visualization=visualization,
    )
    failures = sum(row["status"] == "FAIL" for row in rows)
    warnings = sum(row["status"] == "WARN" for row in rows)
    print("\nReports:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    print(f"  {html_path}")
    print(f"\nResult: {failures} failure(s), {warnings} warning sample(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
