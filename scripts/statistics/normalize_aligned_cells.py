from __future__ import annotations

"""Map PCA-aligned cells to a common two-dimensional coordinate system.

Spatial normalization rescales the aligned cell bounding box to a fixed width
and pole-to-pole length. The script saves both absolute transformed intensity
and optional within-cell min-max normalized intensity. The latter is intended
only for localization maps and profiles, not for absolute fluorescence
comparisons.
"""

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

SCRIPT_FILE = Path(__file__).resolve()


def find_project_root(start: Path | None = None) -> Path:
    current = (start or SCRIPT_FILE).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT project root; provide --project-root.")


def import_display_helper(project_root: Path):
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
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        raise ValueError("PCA mask is empty")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    return mask[y0:y1, x0:x1], image[:, y0:y1, x0:x1]


def _resize_exact(array: np.ndarray, shape: tuple[int, int], order: int) -> np.ndarray:
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
        source_y0 : source_y0 + copy_h, source_x0 : source_x0 + copy_w
    ]
    return output


def spatial_standardize(
    mask: np.ndarray,
    image: np.ndarray,
    target_width: int,
    target_length: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
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
                {"channel": channel_index, "low": float(np.min(finite)), "high": float(np.max(finite))}
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
            normalized = np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)
        output[channel_index, mask] = normalized[mask]
        parameters.append({"channel": channel_index, "low": low, "high": high})
    return output, parameters


def _save_cyx(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, image.astype(np.float32), metadata={"axes": "CYX"})


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
    spatial_figure, spatial_axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    spatial_axes[0, 0].imshow(mask_pca, cmap="gray", interpolation="nearest")
    spatial_axes[0, 0].set_title("PCA mask before spatial normalization")
    spatial_axes[0, 1].imshow(mask_standardized, cmap="gray", interpolation="nearest")
    spatial_axes[0, 1].set_title("Mask after spatial normalization")
    spatial_axes[1, 0].imshow(image_to_rgb(image_pca, mask_pca))
    spatial_axes[1, 0].set_title("PCA image before spatial normalization")
    spatial_axes[1, 1].imshow(image_to_rgb(image_standardized, mask_standardized))
    spatial_axes[1, 1].set_title("Image after spatial normalization")
    for axis in spatial_axes.ravel():
        axis.axis("off")
    spatial_figure.suptitle(title)
    spatial_figure.savefig(output_dir / "spatial_normalization_before_after.png", dpi=180)
    plt.close(spatial_figure)

    intensity_figure, intensity_axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    intensity_axes[0].imshow(image_to_rgb(image_standardized, mask_standardized))
    intensity_axes[0].set_title("Standardized absolute intensity")
    intensity_axes[1].imshow(image_to_rgb(image_normalized, mask_standardized))
    intensity_axes[1].set_title("Within-cell normalized intensity")
    for axis in intensity_axes:
        axis.axis("off")
    intensity_figure.suptitle(title)
    intensity_figure.savefig(output_dir / "intensity_normalization_before_after.png", dpi=180)
    plt.close(intensity_figure)


def save_annotation_map(
    output_path: Path,
    weighted_sum: np.ndarray,
    weight: np.ndarray,
    image_to_rgb,
    title: str,
) -> None:
    if weighted_sum.size == 0 or not np.any(weight > 0):
        return
    average = weighted_sum / np.maximum(weight[None, ...], np.finfo(np.float32).eps)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.imshow(image_to_rgb(average, weight > 0))
    axis.set_title(title)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize PCA-aligned cells to a common coordinate system.")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--pca-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--target-length", type=int, default=90)
    parser.add_argument("--target-width", type=int, default=30)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument(
        "--intensity-normalization",
        choices=("none", "minmax", "percentile"),
        default="minmax",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root
        else find_project_root()
    )
    pca_root = args.pca_root.expanduser().resolve()
    if args.output_root:
        output_root = args.output_root.expanduser().resolve()
    elif pca_root.parent.parent.name == "pca_aligned":
        output_root = (
            pca_root.parent.parent.parent
            / "normalized_cells"
            / pca_root.parent.name
            / pca_root.name
        )
    else:
        output_root = pca_root.parent / f"{pca_root.name}_normalized"
    if args.target_length < 3 or args.target_width < 3:
        raise ValueError("Target dimensions must be at least 3 pixels")
    if args.margin < 0:
        raise ValueError("--margin cannot be negative")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output folder is not empty: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_to_rgb = import_display_helper(project_root)
    cell_files = sorted(pca_root.rglob("pca_cell_data.npz"), key=lambda p: str(p).lower())
    if not cell_files:
        raise FileNotFoundError(f"No pca_cell_data.npz files found under {pca_root}")

    rows: list[dict[str, object]] = []
    accumulators: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cell_file in cell_files:
        relative = cell_file.parent.relative_to(pca_root)
        annotation_relative = relative.parent.parent if relative.parent.name == "cells" else relative.parent
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
            target_width=args.target_width,
            target_length=args.target_length,
            margin=args.margin,
        )
        standardized_normalized, parameters = normalize_within_cell(
            standardized_absolute,
            standardized_mask,
            method=args.intensity_normalization,
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
            annotation_id=annotation_id,
            source_label=source_label,
            source_cell_data=str(cell_file),
            target_width=args.target_width,
            target_length=args.target_length,
            margin=args.margin,
            source_height=int(mask_pca.shape[0]),
            source_width=int(mask_pca.shape[1]),
            standardized_foreground_pixels=int(np.count_nonzero(standardized_mask)),
            channel_count=int(image_pca.shape[0]),
            intensity_normalization=args.intensity_normalization,
            output_dir=str(cell_output),
        )
        (cell_output / "normalization_metadata.json").write_text(
            json.dumps(
                {
                    **asdict(result),
                    "intensity_parameters": parameters,
                    "quantitative_note": (
                        "image_standardized_absolute preserves the transformed input values. "
                        "image_standardized_normalized is intended only for localization "
                        "plots and must not be used for absolute fluorescence comparisons."
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
    (output_root / "normalization_run_summary.json").write_text(
        json.dumps(
            {
                "pca_root": str(pca_root),
                "output_root": str(output_root),
                "n_cells": len(rows),
                "target_length": args.target_length,
                "target_width": args.target_width,
                "margin": args.margin,
                "intensity_normalization": args.intensity_normalization,
                "interpretation": {
                    "spatial_normalization": (
                        "PCA-aligned cell bounding boxes are rescaled to a common "
                        "pole-to-pole length and transverse width."
                    ),
                    "within_cell_intensity_normalization": (
                        "Used only for localization patterns. Absolute transformed "
                        "intensity is saved separately."
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Normalized cells: {len(rows)}")
    print(f"Normalization output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
