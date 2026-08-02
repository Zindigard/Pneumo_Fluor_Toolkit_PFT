"""
Provide command-line and programmatic utilities for normalize aligned cells.

Examples
--------
Show all command-line parameters:

    python scripts/statistics/normalize_aligned_cells.py --help

Representative execution:

    python scripts/statistics/normalize_aligned_cells.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --example
"""

from __future__ import annotations

"""Normalize PCA-aligned cells for one dataset or all supported datasets.

Normal mode reads ``pca_aligned`` outputs. Passing ``--example`` reads
``pca_aligned_example`` outputs. The same normalization calculations are used
in both modes. Every standardized cell has a longitudinal coordinate from 0 to
1, with geometric midcell at 0.5.
"""

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

SCRIPT_FILE = Path(__file__).resolve()


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
        find_project_root,
        normalization_output_path,
        pca_output_path,
        selected_datasets,
        source_mode_for,
    )

    return {
        "DATASETS": DATASETS,
        "find_project_root": find_project_root,
        "normalization_output_path": normalization_output_path,
        "pca_output_path": pca_output_path,
        "selected_datasets": selected_datasets,
        "source_mode_for": source_mode_for,
    }


COMMON = _import_common()


def import_display_helper(project_root: Path):
    """Return import display helper for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = import_display_helper(project_root=Path("path/to/resource"))
    """
    script_dir = project_root / "scripts" / "statistics"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from check_masks_for_statistics import image_to_rgb
    except ImportError:
        local_dir = SCRIPT_FILE.parent
        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))
        from check_masks_for_statistics import image_to_rgb
    return image_to_rgb


@dataclass(frozen=True)
class NormalizationResult:
    """Store validated configuration or result data for normalization result."""
    dataset: str
    source_mode: str
    run_mode: str
    annotation_id: str
    source_label: int
    source_cell_data: str
    target_width: int
    target_length: int
    margin: int
    source_height: int
    source_width: int
    standardized_foreground_pixels: int
    channel_count: int
    intensity_normalization: str
    output_dir: str


def _tight_crop(mask: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tight crop for the supplied inputs.

    Args:
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        image (np.ndarray): Input image array to process.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _tight_crop(mask=image_array, image=image_array)
    """
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        raise ValueError("PCA mask is empty")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    return mask[y0:y1, x0:x1], image[:, y0:y1, x0:x1]


def _resize_exact(array: np.ndarray, shape: tuple[int, int], order: int) -> np.ndarray:
    """Resize exact to the requested shape.

    Args:
        array (np.ndarray): Array containing array.
        shape (tuple[int, int]): Target or observed array shape.
        order (int): Numerical value controlling order.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _resize_exact(
        ...     array=image_array,
        ...     shape=1,
        ...     order=1,
        ... )
    """
    source_h, source_w = array.shape[-2:]
    target_h, target_w = shape
    zoom = (target_h / max(source_h, 1), target_w / max(source_w, 1))
    resized = ndi.zoom(array, zoom=zoom, order=order, mode="nearest", prefilter=order > 1)
    output = np.zeros(shape, dtype=resized.dtype)
    copy_h = min(target_h, resized.shape[0])
    copy_w = min(target_w, resized.shape[1])
    source_y0 = max(0, (resized.shape[0] - copy_h) // 2)
    source_x0 = max(0, (resized.shape[1] - copy_w) // 2)
    target_y0 = max(0, (target_h - copy_h) // 2)
    target_x0 = max(0, (target_w - copy_w) // 2)
    output[target_y0 : target_y0 + copy_h, target_x0 : target_x0 + copy_w] = resized[
        source_y0 : source_y0 + copy_h,
        source_x0 : source_x0 + copy_w,
    ]
    return output


def spatial_standardize(
    mask: np.ndarray,
    image: np.ndarray,
    target_width: int,
    target_length: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial standardize for the supplied inputs.

    Args:
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        image (np.ndarray): Input image array to process.
        target_width (int): Numerical value controlling target width.
        target_length (int): Numerical value controlling target length.
        margin (int): Numerical value controlling margin.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = spatial_standardize(
        ...     mask=image_array,
        ...     image=image_array,
        ...     target_width=1,
        ...     target_length=1,
        ...     margin=1,
        ... )
    """
    mask_crop, image_crop = _tight_crop(mask.astype(bool), image.astype(np.float32))
    content_h = target_width - 2 * margin
    content_w = target_length - 2 * margin
    if content_h < 1 or content_w < 1:
        raise ValueError("The selected margin leaves no valid target area")

    resized_mask = _resize_exact(mask_crop.astype(np.uint8), (content_h, content_w), order=0) > 0
    resized_channels = [
        _resize_exact(channel, (content_h, content_w), order=1).astype(np.float32)
        for channel in image_crop
    ]
    standardized_mask = np.zeros((target_width, target_length), dtype=bool)
    standardized_image = np.zeros(
        (image.shape[0], target_width, target_length), dtype=np.float32
    )
    standardized_mask[margin : margin + content_h, margin : margin + content_w] = resized_mask
    for channel_index, channel in enumerate(resized_channels):
        standardized_image[
            channel_index,
            margin : margin + content_h,
            margin : margin + content_w,
        ] = channel
    standardized_image[:, ~standardized_mask] = 0.0
    return standardized_mask, standardized_image


def normalize_within_cell(
    image: np.ndarray,
    mask: np.ndarray,
    method: str,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Normalize within cell using the configured procedure.

    Args:
        image (np.ndarray): Input image array to process.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        method (str): Text value specifying method.

    Returns:
        tuple[np.ndarray, list[dict[str, float]]]: Mapping containing the generated or resolved values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = normalize_within_cell(
        ...     image=image_array,
        ...     mask=image_array,
        ...     method="method",
        ... )
    """
    output = np.zeros_like(image, dtype=np.float32)
    parameters: list[dict[str, float]] = []
    for channel_index, channel in enumerate(image):
        values = channel[mask]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            parameters.append({"channel": channel_index, "low": 0.0, "high": 0.0})
            continue
        if method == "none":
            output[channel_index, mask] = channel[mask]
            parameters.append(
                {
                    "channel": channel_index,
                    "low": float(np.min(finite)),
                    "high": float(np.max(finite)),
                }
            )
            continue
        if method == "minmax":
            low, high = float(np.min(finite)), float(np.max(finite))
        elif method == "percentile":
            low, high = [float(value) for value in np.percentile(finite, (1.0, 99.0))]
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            normalized = np.zeros_like(channel, dtype=np.float32)
        else:
            normalized = np.clip((channel - low) / (high - low), 0.0, 1.0).astype(
                np.float32
            )
        output[channel_index, mask] = normalized[mask]
        parameters.append({"channel": channel_index, "low": low, "high": high})
    return output, parameters


def _save_cyx(path: Path, image: np.ndarray) -> None:
    """Save cyx to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        image (np.ndarray): Input image array to process.

    Example:
        >>> _save_cyx(path=Path("path/to/resource"), image=image_array)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, image.astype(np.float32), metadata={"axes": "CYX"})


def _annotate_normalized_axis(axis, array_shape: tuple[int, int]) -> None:
    """Return annotate normalized axis for the supplied inputs.

    Args:
        axis (Any): Array axis along which the operation is performed.
        array_shape (tuple[int, int]): Numerical value controlling array shape.

    Example:
        >>> _annotate_normalized_axis(axis=..., array_shape=1)
    """
    _height, width = array_shape
    mid_x = (width - 1) / 2.0
    axis.axvline(mid_x, color="yellow", linestyle="--", linewidth=1.0, alpha=0.9)
    text_box = dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5)
    axis.text(
        0.02,
        0.03,
        "0",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox=text_box,
    )
    axis.text(
        0.50,
        0.03,
        "0.5",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox=text_box,
    )
    axis.text(
        0.98,
        0.03,
        "1",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox=text_box,
    )
    axis.text(
        0.50,
        0.97,
        "midcell",
        transform=axis.transAxes,
        ha="center",
        va="top",
        color="yellow",
        fontsize=11,
        fontweight="bold",
        bbox=text_box,
    )


def save_normalization_figures(
    output_dir: Path,
    image_pca: np.ndarray,
    mask_pca: np.ndarray,
    image_standardized: np.ndarray,
    mask_standardized: np.ndarray,
    image_normalized: np.ndarray,
    image_to_rgb,
    title: str,
) -> None:
    """Save normalization figures to persistent storage.

    Args:
        output_dir (Path): Directory where generated resources are written.
        image_pca (np.ndarray): Array containing image principal-component analysis result.
        mask_pca (np.ndarray): Array containing mask principal-component analysis result.
        image_standardized (np.ndarray): Array containing image standardized.
        mask_standardized (np.ndarray): Array containing mask standardized.
        image_normalized (np.ndarray): Array containing image normalized.
        image_to_rgb (Any): Value specifying image to RGB representation for the operation.
        title (str): Title displayed on the generated figure or report section.

    Example:
        >>> save_normalization_figures(
        ...     output_dir=Path("path/to/resource"),
        ...     image_pca=image_array,
        ...     mask_pca=image_array,
        ...     image_standardized=image_array,
        ...     mask_standardized=image_array,
        ...     image_normalized=image_array,
        ...     image_to_rgb=...,
        ...     title="title",
        ... )
    """
    spatial_figure, spatial_axes = plt.subplots(
        2, 2, figsize=(9, 7), constrained_layout=True
    )
    spatial_axes[0, 0].imshow(mask_pca, cmap="gray", interpolation="nearest")
    spatial_axes[0, 0].set_title("PCA mask before spatial normalization")
    spatial_axes[0, 1].imshow(mask_standardized, cmap="gray", interpolation="nearest")
    spatial_axes[0, 1].set_title("Mask after spatial normalization")
    _annotate_normalized_axis(spatial_axes[0, 1], mask_standardized.shape)
    spatial_axes[1, 0].imshow(image_to_rgb(image_pca, mask_pca))
    spatial_axes[1, 0].set_title("PCA image before spatial normalization")
    spatial_axes[1, 1].imshow(image_to_rgb(image_standardized, mask_standardized))
    spatial_axes[1, 1].set_title("Image after spatial normalization")
    _annotate_normalized_axis(spatial_axes[1, 1], mask_standardized.shape)
    for axis in spatial_axes.ravel():
        axis.axis("off")
    spatial_figure.suptitle(title)
    spatial_figure.savefig(
        output_dir / "spatial_normalization_before_after.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(spatial_figure)

    intensity_figure, intensity_axes = plt.subplots(
        1, 2, figsize=(9, 4), constrained_layout=True
    )
    intensity_axes[0].imshow(image_to_rgb(image_standardized, mask_standardized))
    intensity_axes[0].set_title("Standardized absolute intensity")
    _annotate_normalized_axis(intensity_axes[0], mask_standardized.shape)
    intensity_axes[1].imshow(image_to_rgb(image_normalized, mask_standardized))
    intensity_axes[1].set_title("Within-cell normalized intensity")
    _annotate_normalized_axis(intensity_axes[1], mask_standardized.shape)
    for axis in intensity_axes:
        axis.axis("off")
    intensity_figure.suptitle(title)
    intensity_figure.savefig(
        output_dir / "intensity_normalization_before_after.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(intensity_figure)


def save_annotation_map(
    output_path: Path,
    weighted_sum: np.ndarray,
    weight: np.ndarray,
    image_to_rgb,
    title: str,
) -> None:
    """Save annotation map to persistent storage.

    Args:
        output_path (Path): Filesystem path where the generated result is written.
        weighted_sum (np.ndarray): Array containing weighted sum.
        weight (np.ndarray): Array containing weight.
        image_to_rgb (Any): Value specifying image to RGB representation for the operation.
        title (str): Title displayed on the generated figure or report section.

    Example:
        >>> save_annotation_map(
        ...     output_path=Path("path/to/resource"),
        ...     weighted_sum=image_array,
        ...     weight=image_array,
        ...     image_to_rgb=...,
        ...     title="title",
        ... )
    """
    if weighted_sum.size == 0 or not np.any(weight > 0):
        return
    average = weighted_sum / np.maximum(weight[None, ...], np.finfo(np.float32).eps)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.imshow(image_to_rgb(average, weight > 0))
    axis.set_title(title)
    _annotate_normalized_axis(axis, weight.shape)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write CSV data to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (list[dict[str, object]]): Text value specifying rows.

    Example:
        >>> _write_csv(path=Path("path/to/resource"), rows="rows")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def process_pca_root(
    *,
    project_root: Path,
    dataset: str,
    source_mode: str,
    run_mode: str,
    pca_root: Path,
    output_root: Path,
    target_length: int,
    target_width: int,
    margin: int,
    intensity_normalization: str,
    overwrite: bool,
) -> dict[str, object]:
    """Process principal-component analysis result root using the configured workflow.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        run_mode (str): Text value specifying run mode.
        pca_root (Path): Directory used for principal-component analysis result.
        output_root (Path): Directory used for output.
        target_length (int): Numerical value controlling target length.
        target_width (int): Numerical value controlling target width.
        margin (int): Numerical value controlling margin.
        intensity_normalization (str): Text value specifying intensity normalization.
        overwrite (bool): Whether an existing output may be replaced.

    Returns:
        dict[str, object]: Mapping containing the generated or resolved values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = process_pca_root(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     run_mode="run_mode",
        ...     pca_root=Path("path/to/resource"),
        ...     output_root=Path("path/to/resource"),
        ...     target_length=1,
        ...     target_width=1,
        ...     margin=1,
        ...     intensity_normalization="intensity_normalization",
        ...     overwrite=True,
        ... )
    """
    if not pca_root.exists():
        raise FileNotFoundError(f"PCA root does not exist: {pca_root}")
    _prepare_output(output_root, overwrite)
    image_to_rgb = import_display_helper(project_root)
    cell_files = sorted(pca_root.rglob("pca_cell_data.npz"), key=lambda p: str(p).lower())
    if not cell_files:
        raise FileNotFoundError(f"No pca_cell_data.npz files found under {pca_root}")

    rows: list[dict[str, object]] = []
    accumulators: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cell_file in cell_files:
        relative = cell_file.parent.relative_to(pca_root)
        annotation_relative = (
            relative.parent.parent if relative.parent.name == "cells" else relative.parent
        )
        annotation_id = annotation_relative.as_posix()
        cell_output = output_root / relative
        cell_output.mkdir(parents=True, exist_ok=True)

        with np.load(cell_file) as data:
            source_label = int(data["source_label"])
            mask_pca = np.asarray(data["mask_pca"], dtype=bool)
            image_pca = np.asarray(data["image_pca"], dtype=np.float32)
        if image_pca.ndim != 3 or mask_pca.ndim != 2:
            raise ValueError(f"Invalid PCA cell arrays in {cell_file}")
        if tuple(image_pca.shape[-2:]) != tuple(mask_pca.shape):
            raise ValueError(f"PCA image-mask shape mismatch in {cell_file}")

        standardized_mask, standardized_absolute = spatial_standardize(
            mask=mask_pca,
            image=image_pca,
            target_width=target_width,
            target_length=target_length,
            margin=margin,
        )
        standardized_normalized, parameters = normalize_within_cell(
            standardized_absolute,
            standardized_mask,
            method=intensity_normalization,
        )

        tiff.imwrite(cell_output / "mask_standardized.tif", standardized_mask.astype(np.uint8))
        _save_cyx(cell_output / "image_standardized_absolute.tif", standardized_absolute)
        _save_cyx(cell_output / "image_standardized_normalized.tif", standardized_normalized)
        np.savez_compressed(
            cell_output / "normalized_cell_data.npz",
            source_label=np.int64(source_label),
            mask_standardized=standardized_mask.astype(np.uint8),
            image_standardized_absolute=standardized_absolute.astype(np.float32),
            image_standardized_normalized=standardized_normalized.astype(np.float32),
        )

        result = NormalizationResult(
            dataset=dataset,
            source_mode=source_mode,
            run_mode=run_mode,
            annotation_id=annotation_id,
            source_label=source_label,
            source_cell_data=str(cell_file),
            target_width=target_width,
            target_length=target_length,
            margin=margin,
            source_height=int(mask_pca.shape[0]),
            source_width=int(mask_pca.shape[1]),
            standardized_foreground_pixels=int(np.count_nonzero(standardized_mask)),
            channel_count=int(image_pca.shape[0]),
            intensity_normalization=intensity_normalization,
            output_dir=str(cell_output),
        )
        (cell_output / "normalization_metadata.json").write_text(
            json.dumps(
                {
                    **asdict(result),
                    "intensity_parameters": parameters,
                    "coordinate_note": (
                        "The standardized longitudinal coordinate is 0 to 1, with "
                        "geometric midcell at 0.5. PCA does not assign biological pole identity."
                    ),
                    "quantitative_note": (
                        "image_standardized_absolute preserves transformed input values. "
                        "image_standardized_normalized is intended for localization plots, "
                        "not absolute fluorescence comparisons."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_normalization_figures(
            output_dir=cell_output,
            image_pca=image_pca,
            mask_pca=mask_pca,
            image_standardized=standardized_absolute,
            mask_standardized=standardized_mask,
            image_normalized=standardized_normalized,
            image_to_rgb=image_to_rgb,
            title=f"{annotation_id}, original label {source_label}",
        )
        rows.append(asdict(result))

        if annotation_id not in accumulators:
            accumulators[annotation_id] = (
                np.zeros_like(standardized_normalized, dtype=np.float64),
                np.zeros(standardized_mask.shape, dtype=np.float64),
            )
        weighted_sum, weight = accumulators[annotation_id]
        weighted_sum += standardized_normalized * standardized_mask[None, ...]
        weight += standardized_mask.astype(np.float64)

    for annotation_id, (weighted_sum, weight) in accumulators.items():
        annotation_output = output_root / Path(*annotation_id.split("/"))
        save_annotation_map(
            annotation_output / "standardized_cell_map_preview.png",
            weighted_sum=weighted_sum,
            weight=weight,
            image_to_rgb=image_to_rgb,
            title=f"Standardized-cell map preview: {annotation_id}",
        )
        np.savez_compressed(
            annotation_output / "standardized_cell_map_data.npz",
            weighted_sum=weighted_sum.astype(np.float32),
            valid_weight=weight.astype(np.float32),
        )

    _write_csv(output_root / "normalization_summary.csv", rows)
    summary = {
        "dataset": dataset,
        "source_mode": source_mode,
        "run_mode": run_mode,
        "pca_root": str(pca_root),
        "output_root": str(output_root),
        "n_cells": len(rows),
        "n_annotations": len(accumulators),
        "target_length": target_length,
        "target_width": target_width,
        "margin": margin,
        "intensity_normalization": intensity_normalization,
    }
    (output_root / "normalization_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[NORMALIZE] {dataset}: {len(rows)} cells")
    print(f"Normalization output: {output_root}")
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
            "Normalize PCA-aligned cells for one dataset or all datasets. "
            "Use --example to read example-mode PCA outputs."
        )
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--dataset",
        choices=("all", *COMMON["DATASETS"]),
        default="all",
    )
    parser.add_argument("--source-mode")
    parser.add_argument("--pca-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--target-length", type=int, default=90)
    parser.add_argument("--target-width", type=int, default=30)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument(
        "--intensity-normalization",
        choices=("none", "minmax", "percentile"),
        default="minmax",
    )
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
    if args.target_length < 3 or args.target_width < 3:
        raise ValueError("Target dimensions must be at least 3 pixels")
    if args.margin < 0:
        raise ValueError("--margin cannot be negative")
    if args.pca_root and args.dataset == "all":
        raise ValueError("An explicit --pca-root requires one concrete --dataset.")

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
        if not pca_root.exists():
            message = f"Missing PCA output for {dataset}: {pca_root}"
            if args.skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise FileNotFoundError(message)

        default_output = COMMON["normalization_output_path"](
            project_root, dataset, source_mode, args.example
        )
        if args.output_root:
            explicit_root = args.output_root.expanduser().resolve()
            output_root = (
                explicit_root / dataset / source_mode
                if len(datasets) > 1
                else explicit_root
            )
        else:
            output_root = default_output

        summaries.append(
            process_pca_root(
                project_root=project_root,
                dataset=dataset,
                source_mode=source_mode,
                run_mode="example" if args.example else "full",
                pca_root=pca_root,
                output_root=output_root,
                target_length=args.target_length,
                target_width=args.target_width,
                margin=args.margin,
                intensity_normalization=args.intensity_normalization,
                overwrite=args.overwrite,
            )
        )

    if not summaries:
        raise RuntimeError("No datasets were normalized.")

    batch_summary = {
        "requested_dataset": args.dataset,
        "run_mode": "example" if args.example else "full",
        "processed_datasets": [item["dataset"] for item in summaries],
        "summaries": summaries,
    }
    batch_path = (
        project_root
        / "results"
        / "statistics_preparation"
        / ("normalized_cells_example" if args.example else "normalized_cells")
        / "normalization_batch_summary.json"
    )
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"\nNormalization batch summary: {batch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
