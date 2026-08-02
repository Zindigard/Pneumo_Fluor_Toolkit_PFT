r"""Validate all files required for thesis-aligned 2D U-Net training and SNR.

The checker compares the canonical raw OME-Zarr stores, intensity-preserving
local-threshold outputs, and hand-labelled masks converted to binary foreground.
After inference, it also verifies that the saved enhanced OME-Zarr preserves
inside-mask values, retains exactly 2% of outside-mask intensity by default,
keeps the source dtype and axes, and is therefore non-normalized. It also writes
QC figures showing the stored filtered values and the exact P1-P99.8 normalized
tensor that will enter the U-Net.

Exit code 0 means all selected datasets are ready for training. Exit code 1
means at least one raw image, filtered image, or reference mask is missing or
incompatible.


Examples
--------
Show all command-line parameters:

    python scripts/segmentation/check_unet_2d_inputs.py --help

Validate all canonical HADA inputs and verify 98% outside-mask depletion:

    python scripts/segmentation/check_unet_2d_inputs.py \
        --dataset 2d_time \
        --preview-count 2 \
        --expected-outside-mask-depletion 0.98
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import html
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

_SCRIPT = Path(__file__).resolve()


def _project_root() -> Path:
    """Return project root for the supplied inputs.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _project_root()
    """
    for candidate in (_SCRIPT.parent, *_SCRIPT.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate project root containing scripts and src/PFT")


PROJECT_ROOT = _project_root()
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, normalize_axes  # noqa: E402
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (  # noqa: E402
    DATASETS_2D,
    default_filtered_root,
    find_mask_file,
    normalize_image01,
    ome_zarr_to_hwc_frames_2d,
    read_reference_mask_2d,
)


@dataclass(frozen=True)
class CheckRow:
    """Store validated configuration or result data for check row."""
    dataset: str
    sample: str
    status: str
    raw_zarr: str
    filtered_zarr: str
    mask_path: str
    raw_exists: bool
    filtered_exists: bool
    mask_exists: bool
    axes_match: bool
    shape_match: bool
    dtype_match: bool
    retained_values_exact: bool
    filtered_storage_normalized: str
    frame_count_raw: int
    frame_count_filtered: int
    channel_count: int
    foreground_pixels: int
    background_pixels: int
    positive_source_labels: int
    snr_pre_inference_ready: bool
    predicted_mask_exists: bool
    foreground_output_exists: bool
    foreground_output_axes_shape_match: bool
    foreground_output_dtype_match: bool
    outside_mask_depletion_verified: bool
    foreground_output_non_normalized: bool
    snr_post_inference_ready: bool
    preview_png: str
    issues: str


def _discover(root: Path) -> dict[str, Path]:
    """Discover the requested operation in the configured project structure.

    Args:
        root (Path): Root directory used to resolve relative project paths.

    Returns:
        dict[str, Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _discover(root=Path("path/to/resource"))
    """
    if not root.is_dir():
        return {}
    return {
        path.parent.name: path
        for path in sorted(root.glob("*/image.ome.zarr"))
        if path.is_dir()
    }


def _read_array(path: Path) -> tuple[np.ndarray, str]:
    """Read array from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Example:
        >>> result = _read_array(path=Path("path/to/resource"))
    """
    array, axes = load_ome_zarr(path, level=0, as_numpy=True)
    return np.asarray(array), normalize_axes(axes)




def _broadcast_prediction_to_input(
    prediction: np.ndarray,
    prediction_axes: str,
    input_axes: str,
    input_shape: tuple[int, ...],
) -> np.ndarray:
    """Broadcast a YX or TYX prediction to the complete input-array shape.

    Args:
        prediction (np.ndarray): Array containing prediction.
        prediction_axes (str): Text value specifying prediction axes.
        input_axes (str): Text value specifying input axes.
        input_shape (tuple[int, ...]): Numerical value controlling input shape.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _broadcast_prediction_to_input(
        ...     prediction=image_array,
        ...     prediction_axes="prediction_axes",
        ...     input_axes="input_axes",
        ...     input_shape=1,
        ... )
    """
    prediction_axes = normalize_axes(prediction_axes)
    input_axes = normalize_axes(input_axes)
    if any(axis not in input_axes for axis in prediction_axes):
        raise ValueError(
            f"Prediction axes {prediction_axes} are incompatible with input axes {input_axes}"
        )

    ordered_prediction_axes = [axis for axis in input_axes if axis in prediction_axes]
    permutation = [prediction_axes.index(axis) for axis in ordered_prediction_axes]
    ordered = np.transpose(prediction, axes=permutation) if permutation else prediction

    reshape: list[int] = []
    ordered_index = 0
    for axis, input_size in zip(input_axes, input_shape):
        if axis in prediction_axes:
            prediction_size = int(ordered.shape[ordered_index])
            if prediction_size != int(input_size):
                raise ValueError(
                    f"Prediction/input size mismatch for axis {axis}: "
                    f"{prediction_size} != {input_size}"
                )
            reshape.append(prediction_size)
            ordered_index += 1
        else:
            reshape.append(1)
    return np.broadcast_to(ordered.reshape(tuple(reshape)), input_shape)


def _expected_background_suppressed_array(
    filtered: np.ndarray,
    input_axes: str,
    prediction: np.ndarray,
    prediction_axes: str,
    depletion: float,
) -> np.ndarray:
    """Reproduce the documented non-normalized post-processing operation.

    Args:
        filtered (np.ndarray): Array containing filtered.
        input_axes (str): Text value specifying input axes.
        prediction (np.ndarray): Array containing prediction.
        prediction_axes (str): Text value specifying prediction axes.
        depletion (float): Numerical value controlling depletion.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _expected_background_suppressed_array(
        ...     filtered=image_array,
        ...     input_axes="input_axes",
        ...     prediction=image_array,
        ...     prediction_axes="prediction_axes",
        ...     depletion=0.5,
        ... )
    """
    if not 0.0 <= depletion <= 1.0:
        raise ValueError("depletion must be in [0,1]")
    broadcast = _broadcast_prediction_to_input(
        prediction > 0, prediction_axes, input_axes, filtered.shape
    )
    residual = 1.0 - float(depletion)
    expected = np.where(broadcast, filtered.astype(np.float64), filtered * residual)
    if np.issubdtype(filtered.dtype, np.integer):
        limits = np.iinfo(filtered.dtype)
        expected = np.clip(np.rint(expected), limits.min, limits.max)
    return expected.astype(filtered.dtype, copy=False)


def _storage_normalization_state(raw: np.ndarray, filtered: np.ndarray, exact: bool) -> str:
    """Describe whether normalization appears to have been applied before saving.

    Args:
        raw (np.ndarray): Array containing raw.
        filtered (np.ndarray): Array containing filtered.
        exact (bool): Boolean flag controlling exact.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _storage_normalization_state(
        ...     raw=image_array,
        ...     filtered=image_array,
        ...     exact=True,
        ... )
    """
    if not exact:
        return "UNKNOWN: retained values differ from raw"
    if raw.dtype != filtered.dtype:
        return "UNKNOWN: dtype differs from raw"
    if np.issubdtype(filtered.dtype, np.integer):
        return "NO: integer dtype and retained values equal raw"
    finite = filtered[np.isfinite(filtered)]
    if finite.size and float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0:
        return "AMBIGUOUS: float data are within [0,1], but retained values equal raw"
    return "NO: retained values and dtype equal raw"


# Fluorescence-style display map for the single-channel 2d_time/HADA dataset.
# This affects PNG previews only; quantitative arrays are never recoloured or changed.
_TIME_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "pft_time_blue", [(0.0, 0.0, 0.0), (0.0, 0.25, 1.0)]
)


def _display(
    hwc: np.ndarray,
    *,
    dataset: str,
    already_normalized: bool = False,
) -> tuple[np.ndarray, str | LinearSegmentedColormap | None]:
    """Prepare one image for the HTML/PNG input-readiness preview.

    The returned image is display-normalized only. The stored raw and filtered
    arrays are not modified. For ``2d_time``, the single HADA channel is shown
    with a black-to-blue fluorescence map. For ``2d_wga_dapi``, channel 0 is
    displayed as DAPI/blue and channel 1 as WGA/green.

    Args:
        hwc (np.ndarray): Array containing hwc.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        already_normalized (bool): Boolean flag controlling already normalized. Defaults to ``False``.

    Returns:
        tuple[np.ndarray, str | LinearSegmentedColormap | None]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _display(hwc=image_array, dataset="2d_time")
    """
    image = hwc if already_normalized else normalize_image01(hwc, "percentile")

    if dataset == "2d_time":
        if image.shape[-1] != 1:
            raise ValueError(
                f"2d_time preview expects one channel, received shape {image.shape}"
            )
        return image[..., 0], _TIME_BLUE_CMAP

    if image.shape[-1] == 1:
        return image[..., 0], "gray"
    if image.shape[-1] == 2:
        rgb = np.stack(
            [np.zeros_like(image[..., 0]), image[..., 1], image[..., 0]], axis=-1
        )
        return rgb, None
    return image[..., :3], None


def _save_preview(
    raw_frame: np.ndarray,
    filtered_frame: np.ndarray,
    mask: np.ndarray,
    path: Path,
    dataset: str,
    sample: str,
) -> Path:
    """Save preview to persistent storage.

    Args:
        raw_frame (np.ndarray): Array containing raw frame.
        filtered_frame (np.ndarray): Array containing filtered frame.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        path (Path): Filesystem path to the required input or output resource.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample (str): Text value specifying sample.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _save_preview(
        ...     raw_frame=image_array,
        ...     filtered_frame=image_array,
        ...     mask=image_array,
        ...     path=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     sample="sample",
        ... )
    """
    normalized_input = normalize_image01(filtered_frame, "percentile")
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.3), dpi=160)

    raw_display, raw_cmap = _display(raw_frame, dataset=dataset)
    filtered_display, filtered_cmap = _display(filtered_frame, dataset=dataset)
    normalized_display, normalized_cmap = _display(
        normalized_input, dataset=dataset, already_normalized=True
    )

    if dataset == "2d_time":
        raw_title = "Raw HADA image\noriginal values; blue display only"
        filtered_title = (
            "Local-threshold filtered HADA\noriginal values; blue display only"
        )
        normalized_title = (
            "Normalized U-Net input\nP1-P99.8 → [0,1]; blue display only"
        )
    else:
        raw_title = "Raw image\nDAPI blue + WGA green; display only"
        filtered_title = (
            "Local-threshold filtered image\noriginal values; display normalized"
        )
        normalized_title = "Normalized U-Net input\nP1-P99.8 per channel → [0,1]"

    panels: list[
        tuple[str, np.ndarray, str | LinearSegmentedColormap | None]
    ] = [
        (raw_title, raw_display, raw_cmap),
        (filtered_title, filtered_display, filtered_cmap),
        (normalized_title, normalized_display, normalized_cmap),
        ("Reference foreground mask\npositive labels converted to foreground", mask, "gray"),
    ]
    for axis, (title, image, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle(f"U-Net input readiness: {dataset} | {sample}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _write_csv(path: Path, rows: list[CheckRow]) -> None:
    """Write CSV data to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (list[CheckRow]): Value specifying rows for the operation.

    Example:
        >>> _write_csv(path=Path("path/to/resource"), rows=[])
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CheckRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_html(path: Path, rows: list[CheckRow], roots: dict[str, dict[str, str]]) -> None:
    """Write html to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (list[CheckRow]): Value specifying rows for the operation.
        roots (dict[str, dict[str, str]]): Text value specifying roots.

    Example:
        >>> _write_html(
        ...     path=Path("path/to/resource"),
        ...     rows=[],
        ...     roots="roots",
        ... )
    """
    failures = sum(row.status == "FAIL" for row in rows)
    status = "PASS" if failures == 0 else "FAIL"
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>PFT 2D U-Net input readiness</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;line-height:1.4}",
        "table{border-collapse:collapse;width:100%;font-size:12px}",
        "th,td{border:1px solid #ccc;padding:6px;vertical-align:top}",
        "th{background:#eee;position:sticky;top:0}.PASS{background:#e8f5e9}",
        ".FAIL{background:#ffebee}img{max-width:100%;height:auto;border:1px solid #bbb}",
        "code{background:#f4f4f4;padding:2px 4px}</style></head><body>",
        f"<h1>PFT 2D U-Net input readiness: {status}</h1>",
        f"<p>Generated UTC: {html.escape(datetime.now(timezone.utc).isoformat())}</p>",
        "<p><strong>Normalization decision:</strong> keep local-threshold OME-Zarr files in their original intensity scale. The training and inference code normalizes each complete image independently per channel by mapping P1 and P99.8 to [0,1]. Do not save a second normalized training dataset.</p>",
        "<p><strong>SNR prerequisites:</strong> raw image, foreground mask, and filtered image are required before training. After inference, <code>pred_mask.ome.zarr</code> and <code>foreground_filtered.ome.zarr</code> complete the before/after SNR workflow. The checker verifies that the saved enhanced image is not normalized, that inside-mask values are unchanged, and that outside-mask values retain the configured residual intensity (2% for 98% depletion).</p>",
    ]
    for dataset, mapping in roots.items():
        lines.append(f"<h2>{html.escape(dataset)}</h2><ul>")
        for key, value in mapping.items():
            lines.append(f"<li>{html.escape(key)}: <code>{html.escape(value)}</code></li>")
        lines.append("</ul>")

    preview_rows = [row for row in rows if row.preview_png]
    if preview_rows:
        lines.append("<h2>Representative input previews</h2>")
        for row in preview_rows:
            relative = Path(row.preview_png).relative_to(path.parent).as_posix()
            lines.append(
                f"<h3>{html.escape(row.dataset)} | {html.escape(row.sample)}</h3>"
                f"<img src='{html.escape(relative)}' alt='U-Net input preview'>"
            )

    lines.append("<h2>Per-sample checks</h2><table><thead><tr>")
    columns = [
        "dataset", "sample", "status", "raw_exists", "filtered_exists",
        "mask_exists", "axes_match", "shape_match", "dtype_match",
        "retained_values_exact", "filtered_storage_normalized",
        "foreground_pixels", "background_pixels", "positive_source_labels",
        "snr_pre_inference_ready",
        "predicted_mask_exists", "foreground_output_exists",
        "foreground_output_axes_shape_match", "foreground_output_dtype_match",
        "outside_mask_depletion_verified", "foreground_output_non_normalized",
        "snr_post_inference_ready", "issues",
    ]
    lines.extend(f"<th>{html.escape(column)}</th>" for column in columns)
    lines.append("</tr></thead><tbody>")
    for row in rows:
        data = asdict(row)
        lines.append(f"<tr class='{row.status}'>")
        for column in columns:
            lines.append(f"<td>{html.escape(str(data[column]))}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table></body></html>")
    path.write_text("".join(lines), encoding="utf-8")


def check_dataset(
    dataset: str,
    *,
    raw_root: Path,
    filtered_root: Path,
    mask_root: Path,
    inference_root: Path,
    output_root: Path,
    preview_count: int,
    expected_outside_mask_depletion: float,
) -> tuple[list[CheckRow], dict[str, str]]:
    """Check dataset for validity and expected structure.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        raw_root (Path): Directory used for raw.
        filtered_root (Path): Directory used for filtered.
        mask_root (Path): Directory used for mask.
        inference_root (Path): Directory used for inference.
        output_root (Path): Directory used for output.
        preview_count (int): Number of preview used by the operation.
        expected_outside_mask_depletion (float): Numerical value controlling expected outside mask depletion.

    Returns:
        tuple[list[CheckRow], dict[str, str]]: Mapping containing the generated or resolved values.

    Example:
        >>> result = check_dataset(
        ...     dataset="2d_time",
        ...     raw_root=Path("path/to/resource"),
        ...     filtered_root=Path("path/to/resource"),
        ...     mask_root=Path("path/to/resource"),
        ...     inference_root=Path("path/to/resource"),
        ...     output_root=Path("path/to/resource"),
        ...     preview_count=1,
        ...     expected_outside_mask_depletion=0.5,
        ... )
    """
    raw = _discover(raw_root)
    filtered = _discover(filtered_root)
    mask_samples: dict[str, Path] = {}
    if mask_root.is_dir():
        for sample_dir in sorted(path for path in mask_root.iterdir() if path.is_dir()):
            mask_path = find_mask_file(sample_dir)
            if mask_path is not None:
                mask_samples[sample_dir.name] = mask_path

    all_samples = sorted(set(raw) | set(filtered) | set(mask_samples))
    rows: list[CheckRow] = []
    previews_written = 0

    if not all_samples:
        rows.append(
            CheckRow(
                dataset=dataset,
                sample="__DATASET__",
                status="FAIL",
                raw_zarr=str(raw_root),
                filtered_zarr=str(filtered_root),
                mask_path=str(mask_root),
                raw_exists=raw_root.is_dir(),
                filtered_exists=filtered_root.is_dir(),
                mask_exists=mask_root.is_dir(),
                axes_match=False,
                shape_match=False,
                dtype_match=False,
                retained_values_exact=False,
                filtered_storage_normalized="UNKNOWN",
                frame_count_raw=0,
                frame_count_filtered=0,
                channel_count=0,
                foreground_pixels=0,
                background_pixels=0,
                positive_source_labels=0,
                snr_pre_inference_ready=False,
                predicted_mask_exists=False,
                foreground_output_exists=False,
                foreground_output_axes_shape_match=False,
                foreground_output_dtype_match=False,
                outside_mask_depletion_verified=False,
                foreground_output_non_normalized=False,
                snr_post_inference_ready=False,
                preview_png="",
                issues="FAIL: no raw, filtered, or reference-mask samples were discovered",
            )
        )

    for sample in all_samples:
        raw_path = raw.get(sample, raw_root / sample / "image.ome.zarr")
        filtered_path = filtered.get(sample, filtered_root / sample / "image.ome.zarr")
        mask_path = mask_samples.get(sample, mask_root / sample / "mask.tif")
        failures: list[str] = []
        warnings: list[str] = []

        raw_exists = raw_path.is_dir()
        filtered_exists = filtered_path.is_dir()
        mask_exists = mask_path.is_file()
        if not raw_exists:
            failures.append("raw image missing")
        if not filtered_exists:
            failures.append("filtered image missing")
        if not mask_exists:
            failures.append("mask.tif missing")

        axes_match = shape_match = dtype_match = retained_exact = False
        frame_count_raw = frame_count_filtered = channel_count = 0
        foreground = background = positive_label_count = 0
        storage_normalization = "UNKNOWN"
        raw_frames: list[np.ndarray] = []
        filtered_frames: list[np.ndarray] = []
        raw_array: np.ndarray | None = None
        filtered_array: np.ndarray | None = None
        raw_axes = ""
        filtered_axes = ""
        mask: np.ndarray | None = None

        if raw_exists and filtered_exists:
            try:
                raw_array, raw_axes = _read_array(raw_path)
                filtered_array, filtered_axes = _read_array(filtered_path)
                axes_match = raw_axes == filtered_axes
                shape_match = raw_array.shape == filtered_array.shape
                dtype_match = raw_array.dtype == filtered_array.dtype
                if not axes_match:
                    failures.append(f"axes mismatch: raw={raw_axes}, filtered={filtered_axes}")
                if not shape_match:
                    failures.append(
                        f"shape mismatch: raw={raw_array.shape}, filtered={filtered_array.shape}"
                    )
                if not dtype_match:
                    failures.append(
                        f"dtype mismatch: raw={raw_array.dtype}, filtered={filtered_array.dtype}"
                    )
                if shape_match:
                    retained_exact = bool(
                        np.all((filtered_array == 0) | (filtered_array == raw_array))
                    )
                    if not retained_exact:
                        failures.append(
                            "filtered values are neither zero nor exact copies of raw values"
                        )
                storage_normalization = _storage_normalization_state(
                    raw_array, filtered_array, retained_exact
                )
                raw_frames = ome_zarr_to_hwc_frames_2d(raw_path, dataset=dataset)
                filtered_frames = ome_zarr_to_hwc_frames_2d(
                    filtered_path, dataset=dataset
                )
                frame_count_raw = len(raw_frames)
                frame_count_filtered = len(filtered_frames)
                channel_count = int(filtered_frames[0].shape[-1])
                if frame_count_raw != frame_count_filtered:
                    failures.append(
                        f"raw/filtered frame-count mismatch: {frame_count_raw}/{frame_count_filtered}"
                    )
                expected_channels = 1 if dataset == "2d_time" else 2
                if channel_count != expected_channels:
                    failures.append(
                        f"expected {expected_channels} channels, found {channel_count}"
                    )
            except Exception as exc:
                failures.append(f"image validation error: {type(exc).__name__}: {exc}")

        if mask_exists:
            try:
                mask, positive_label_count = read_reference_mask_2d(mask_path)
                foreground = int(np.count_nonzero(mask))
                background = int(mask.size - foreground)
                if foreground == 0:
                    failures.append("mask contains no foreground pixels")
                if background == 0:
                    failures.append("mask contains no background pixels")
                if filtered_frames and mask.shape != filtered_frames[0].shape[:2]:
                    failures.append(
                        f"mask/image YX mismatch: mask={mask.shape}, image={filtered_frames[0].shape[:2]}"
                    )
            except Exception as exc:
                failures.append(f"mask validation error: {type(exc).__name__}: {exc}")

        snr_pre_ready = bool(
            raw_exists
            and filtered_exists
            and mask_exists
            and foreground > 0
            and background > 1
            and shape_match
            and retained_exact
            and not failures
        )
        pred_path = inference_root / sample / "pred_mask.ome.zarr"
        foreground_path = inference_root / sample / "foreground_filtered.ome.zarr"
        pred_exists = pred_path.is_dir()
        foreground_exists = foreground_path.is_dir()
        output_axes_shape_match = False
        output_dtype_match = False
        depletion_verified = False
        output_non_normalized = False

        if pred_exists != foreground_exists:
            warnings.append(
                "post-inference output is incomplete: pred_mask.ome.zarr and "
                "foreground_filtered.ome.zarr must be present together"
            )

        if (
            pred_exists
            and foreground_exists
            and filtered_array is not None
            and filtered_axes
        ):
            try:
                prediction_array, prediction_axes = _read_array(pred_path)
                enhanced_array, enhanced_axes = _read_array(foreground_path)
                output_axes_shape_match = bool(
                    enhanced_axes == filtered_axes
                    and enhanced_array.shape == filtered_array.shape
                )
                output_dtype_match = enhanced_array.dtype == filtered_array.dtype
                if not output_axes_shape_match:
                    failures.append(
                        "saved enhanced OME-Zarr axes/shape do not match filtered input: "
                        f"filtered={filtered_axes}{filtered_array.shape}, "
                        f"enhanced={enhanced_axes}{enhanced_array.shape}"
                    )
                if not output_dtype_match:
                    failures.append(
                        "saved enhanced OME-Zarr dtype differs from filtered input: "
                        f"{filtered_array.dtype} != {enhanced_array.dtype}"
                    )

                if output_axes_shape_match:
                    expected_enhanced = _expected_background_suppressed_array(
                        filtered_array,
                        filtered_axes,
                        prediction_array,
                        prediction_axes,
                        expected_outside_mask_depletion,
                    )
                    if np.issubdtype(filtered_array.dtype, np.floating):
                        depletion_verified = bool(
                            np.allclose(
                                enhanced_array,
                                expected_enhanced,
                                rtol=1e-6,
                                atol=1e-7,
                                equal_nan=True,
                            )
                        )
                    else:
                        depletion_verified = bool(
                            np.array_equal(enhanced_array, expected_enhanced)
                        )
                    if not depletion_verified:
                        failures.append(
                            "saved enhanced values do not match the requested outside-mask "
                            f"depletion of {100.0 * expected_outside_mask_depletion:.1f}%"
                        )

                output_non_normalized = bool(
                    output_axes_shape_match
                    and output_dtype_match
                    and depletion_verified
                )
            except Exception as exc:
                failures.append(
                    f"post-inference output validation error: {type(exc).__name__}: {exc}"
                )

        snr_post_ready = bool(
            snr_pre_ready
            and pred_exists
            and foreground_exists
            and output_axes_shape_match
            and output_dtype_match
            and depletion_verified
            and output_non_normalized
        )

        preview = ""
        if (
            previews_written < preview_count
            and raw_frames
            and filtered_frames
            and mask is not None
            and mask.shape == filtered_frames[0].shape[:2]
        ):
            preview_path = output_root / "previews" / dataset / f"{sample}_unet_input.png"
            _save_preview(
                raw_frames[0], filtered_frames[0], mask, preview_path, dataset, sample
            )
            preview = str(preview_path)
            previews_written += 1

        status = "FAIL" if failures else "PASS"
        issues = " | ".join(
            [f"FAIL: {message}" for message in failures]
            + [f"WARN: {message}" for message in warnings]
        )
        rows.append(
            CheckRow(
                dataset=dataset,
                sample=sample,
                status=status,
                raw_zarr=str(raw_path),
                filtered_zarr=str(filtered_path),
                mask_path=str(mask_path),
                raw_exists=raw_exists,
                filtered_exists=filtered_exists,
                mask_exists=mask_exists,
                axes_match=axes_match,
                shape_match=shape_match,
                dtype_match=dtype_match,
                retained_values_exact=retained_exact,
                filtered_storage_normalized=storage_normalization,
                frame_count_raw=frame_count_raw,
                frame_count_filtered=frame_count_filtered,
                channel_count=channel_count,
                foreground_pixels=foreground,
                background_pixels=background,
                positive_source_labels=positive_label_count,
                snr_pre_inference_ready=snr_pre_ready,
                predicted_mask_exists=pred_exists,
                foreground_output_exists=foreground_exists,
                foreground_output_axes_shape_match=output_axes_shape_match,
                foreground_output_dtype_match=output_dtype_match,
                outside_mask_depletion_verified=depletion_verified,
                foreground_output_non_normalized=output_non_normalized,
                snr_post_inference_ready=snr_post_ready,
                preview_png=preview,
                issues=issues,
            )
        )

    roots = {
        "raw root": str(raw_root),
        "filtered root": str(filtered_root),
        "mask root": str(mask_root),
        "inference root": str(inference_root),
        "raw image count": str(len(raw)),
        "filtered image count": str(len(filtered)),
        "mask count": str(len(mask_samples)),
        "expected outside-mask depletion": str(expected_outside_mask_depletion),
        "expected outside-mask residual": str(1.0 - expected_outside_mask_depletion),
    }
    return rows, roots


def build_parser() -> argparse.ArgumentParser:
    """Build parser from the supplied inputs.

    Returns:
        argparse.ArgumentParser: Result produced by the operation.

    Example:
        >>> result = build_parser()
    """
    parser = argparse.ArgumentParser(
        description="Check 2D U-Net raw, local-threshold, mask, IoU, and SNR readiness."
    )
    parser.add_argument(
        "--dataset", choices=[*DATASETS_2D, "all"], default="all"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Override results/img/<dataset>; allowed only for one dataset.",
    )
    parser.add_argument(
        "--filtered-root",
        type=Path,
        help="Override the selected local-threshold root; allowed only for one dataset.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        help="Override results/training_files/U-net/<dataset>; allowed only for one dataset.",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        help="Override results/U-net/<dataset>; allowed only for one dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "U-net" / "input_check",
    )
    parser.add_argument("--preview-count", type=int, default=2)
    parser.add_argument(
        "--expected-outside-mask-depletion",
        type=float,
        default=0.98,
        help=(
            "Expected post-inference depletion outside the predicted mask. "
            "Default 0.98 verifies that 2%% remains."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the command-line workflow and return its process exit status.

    Args:
        argv (Sequence[str] | None): Optional command-line argument sequence. When omitted, arguments are read from ``sys.argv``. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.expected_outside_mask_depletion <= 1.0:
        raise SystemExit("--expected-outside-mask-depletion must be in [0,1]")
    datasets = list(DATASETS_2D) if args.dataset == "all" else [args.dataset]
    if len(datasets) > 1 and any(
        value is not None
        for value in (args.raw_root, args.filtered_root, args.mask_root, args.inference_root)
    ):
        raise SystemExit("Root overrides require selecting one dataset, not --dataset all.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[CheckRow] = []
    root_map: dict[str, dict[str, str]] = {}

    for dataset in datasets:
        raw_root = Path(args.raw_root or PROJECT_ROOT / "results" / "img" / dataset)
        filtered_root = Path(
            args.filtered_root or default_filtered_root(PROJECT_ROOT, dataset)
        )
        mask_root = Path(
            args.mask_root
            or PROJECT_ROOT / "results" / "training_files" / "U-net" / dataset
        )
        inference_root = Path(
            args.inference_root or PROJECT_ROOT / "results" / "U-net" / dataset
        )
        rows, roots = check_dataset(
            dataset,
            raw_root=raw_root,
            filtered_root=filtered_root,
            mask_root=mask_root,
            inference_root=inference_root,
            output_root=output_root,
            preview_count=max(0, args.preview_count),
            expected_outside_mask_depletion=args.expected_outside_mask_depletion,
        )
        all_rows.extend(rows)
        root_map[dataset] = roots

    csv_path = output_root / "unet_2d_input_readiness.csv"
    html_path = output_root / "UNET_2D_INPUT_READINESS.html"
    text_path = output_root / "UNET_2D_INPUT_READINESS.txt"
    _write_csv(csv_path, all_rows)
    _write_html(html_path, all_rows, root_map)

    failures = [row for row in all_rows if row.status == "FAIL"]
    status = "PASS" if not failures else "FAIL"
    lines = [
        "PFT 2D U-NET INPUT READINESS",
        "=" * 80,
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Status: {status}",
        f"Samples checked: {len(all_rows)}",
        f"Samples failed: {len(failures)}",
        "",
        "Decision: pass the intensity-preserving filtered OME-Zarr files to the U-Net.",
        "Do not pre-save normalized images. Training and inference apply P1-P99.8",
        "normalization per complete image and per channel immediately before prediction.",
        f"Expected outside-mask depletion after inference: {100.0 * args.expected_outside_mask_depletion:.1f}%.",
        f"Expected outside-mask residual intensity: {100.0 * (1.0 - args.expected_outside_mask_depletion):.1f}%.",
        "SNR after is valid only when the saved enhanced OME-Zarr passes value, dtype,",
        "axis, shape, and non-normalization checks.",
        "",
        f"CSV: {csv_path}",
        f"HTML with input previews: {html_path}",
    ]
    if failures:
        lines.append("\nBlocking failures:")
        lines.extend(f"- {row.dataset}/{row.sample}: {row.issues}" for row in failures)
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
