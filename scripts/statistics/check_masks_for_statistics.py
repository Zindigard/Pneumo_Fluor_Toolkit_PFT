from __future__ import annotations

"""Validate instance masks and their corresponding images before PCA analysis.

The script is designed first for manual instance annotations stored under::

    results/training_files/segmentation/<dataset>/<source_mode>/...

It also accepts arbitrary mask and image roots, which allows the same validation
stage to be reused later for model predictions. Original positive label values
are preserved. Non-consecutive labels are reported but are not treated as an
error.
"""

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

SCRIPT_FILE = Path(__file__).resolve()
SUPPORTED_DATASETS = ("2d_time", "2d_wga_dapi", "3d_mip")
DEFAULT_SOURCE_MODES = {
    "2d_time": "filtered_unet",
    "2d_wga_dapi": "filtered_unet",
    "3d_mip": "deconv_masked",
}
MASK_NAMES = ("mask.tif", "mask.tiff", "labels.tif", "labels.tiff")


def find_project_root(start: Path | None = None) -> Path:
    current = (start or SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate the PFT project root. Run the script from inside the "
        "Pneumo_Fluor_Toolkit_PFT repository or provide --project-root."
    )


def configure_project_imports(project_root: Path) -> None:
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass(frozen=True)
class AnnotationPair:
    annotation_id: str
    sample_name: str
    annotation_type: str
    split: str
    mask_path: Path
    image_path: Path
    relative_output: Path


@dataclass(frozen=True)
class CellCheck:
    label: int
    area_pixels: int
    component_count: int
    touches_border: bool
    too_small: bool
    centroid_y: float
    centroid_x: float
    status: str


@dataclass(frozen=True)
class PairCheck:
    annotation_id: str
    sample_name: str
    annotation_type: str
    split: str
    mask_path: str
    image_path: str
    mask_shape: str
    image_shape: str
    image_axes: str
    mask_dtype: str
    n_instances: int
    min_label: int | None
    max_label: int | None
    labels_consecutive: bool
    missing_label_count: int
    n_small_instances: int
    n_border_instances: int
    n_fragmented_labels: int
    n_pca_eligible_instances: int
    overall_status: str
    messages: str
    overlay_path: str


def _safe_identifier(text: str) -> str:
    allowed = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_", "/"}:
            allowed.append(char)
        elif char in {" ", ".", ":", "+"}:
            allowed.append("_")
    return "".join(allowed).strip("_/") or "annotation"


def _annotation_kind(relative_parent: Path) -> tuple[str, str, str, Path]:
    parts = list(relative_parent.parts)
    if "crops" in parts:
        crop_index = parts.index("crops")
        sample_parts = parts[:crop_index]
        split = parts[crop_index + 1] if len(parts) > crop_index + 1 else "unknown"
        crop_id = parts[crop_index + 2] if len(parts) > crop_index + 2 else relative_parent.name
        sample_name = "/".join(sample_parts) or "unknown_sample"
        annotation_id = f"{sample_name}/crops/{split}/{crop_id}"
        relative_output = Path(*sample_parts, "crops", split, crop_id)
        return annotation_id, sample_name, "crop", split, relative_output

    sample_name = relative_parent.as_posix()
    annotation_id = f"{sample_name}/full"
    relative_output = relative_parent / "full"
    return annotation_id, sample_name, "full", "full", relative_output


def discover_annotation_pairs(
    mask_root: Path,
    image_root: Path,
    include: str = "all",
    sample_filters: Sequence[str] | None = None,
) -> list[AnnotationPair]:
    mask_root = Path(mask_root).resolve()
    image_root = Path(image_root).resolve()
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    selected_samples = {value.strip() for value in sample_filters or () if value.strip()}
    paths: list[Path] = []
    for name in MASK_NAMES:
        paths.extend(mask_root.rglob(name))

    pairs: list[AnnotationPair] = []
    seen: set[str] = set()
    for mask_path in sorted(paths, key=lambda p: str(p).lower()):
        key = str(mask_path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)

        try:
            relative_parent = mask_path.parent.relative_to(mask_root)
        except ValueError:
            continue

        annotation_id, sample_name, annotation_type, split, relative_output = _annotation_kind(
            relative_parent
        )
        if include != "all":
            if include == "full" and annotation_type != "full":
                continue
            if include in {"train", "validation"} and split != include:
                continue
        if selected_samples and not any(
            sample_name == sample or sample_name.endswith(sample) for sample in selected_samples
        ):
            continue

        if annotation_type == "crop":
            image_base = mask_path.parent
        else:
            image_base = image_root / relative_parent
        image_candidates = (
            image_base / "segmentation_input.ome.zarr",
            image_base / "image.ome.zarr",
            image_base / "segmentation_input.tif",
            image_base / "segmentation_input.tiff",
            image_base / "image.tif",
            image_base / "image.tiff",
        )
        image_path = next((candidate for candidate in image_candidates if candidate.exists()), image_candidates[0])

        pairs.append(
            AnnotationPair(
                annotation_id=annotation_id,
                sample_name=sample_name,
                annotation_type=annotation_type,
                split=split,
                mask_path=mask_path.resolve(),
                image_path=image_path.resolve(),
                relative_output=relative_output,
            )
        )
    return pairs


def load_label_mask(mask_path: Path) -> tuple[np.ndarray, str]:
    raw = np.asarray(tiff.imread(mask_path))
    squeezed = np.squeeze(raw)
    if squeezed.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask, received shape {raw.shape}")
    if not np.issubdtype(squeezed.dtype, np.integer):
        rounded = np.rint(squeezed)
        if not np.allclose(squeezed, rounded, rtol=0.0, atol=1e-6):
            raise ValueError("Mask contains non-integer label values")
        squeezed = rounded.astype(np.int64)
    if np.any(squeezed < 0):
        raise ValueError("Mask contains negative label values")
    return squeezed.astype(np.int64, copy=False), str(raw.dtype)


def _remove_singleton_nonspatial_axes(array: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = axes.lower()
    image = np.asarray(array)
    index = 0
    while index < len(axes):
        letter = axes[index]
        if letter not in {"c", "y", "x"}:
            if image.shape[index] != 1:
                raise ValueError(
                    f"Expected a single 2D field, but axis {letter!r} has size "
                    f"{image.shape[index]} for axes={axes!r}"
                )
            image = np.take(image, 0, axis=index)
            axes = axes[:index] + axes[index + 1 :]
            continue
        index += 1
    return image, axes


def load_image_cyx(image_path: Path, project_root: Path) -> tuple[np.ndarray, str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    if image_path.name.lower().endswith(".ome.zarr"):
        configure_project_imports(project_root)
        from PFT.core_prog_parts.decoder_omezar import load_ome_zarr

        array, axes = load_ome_zarr(image_path, level=0, as_numpy=True)
        image = np.asarray(array, dtype=np.float32)
        axes = str(axes).lower()
    elif image_path.suffix.lower() in {".tif", ".tiff"}:
        image = np.asarray(tiff.imread(image_path), dtype=np.float32)
        axes = "yx" if image.ndim == 2 else "cyx"
    else:
        raise ValueError(f"Unsupported image format: {image_path}")

    image, axes = _remove_singleton_nonspatial_axes(image, axes)
    if axes == "yx":
        image = image[None, ...]
        axes = "cyx"
    elif set(axes) == {"c", "y", "x"} and len(axes) == 3:
        image = np.transpose(image, [axes.index("c"), axes.index("y"), axes.index("x")])
        axes = "cyx"
    else:
        raise ValueError(f"Expected YX or CYX image after selection, received axes={axes}")

    if image.ndim != 3:
        raise ValueError(f"Expected CYX image, received shape {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains NaN or infinite values")
    return image.astype(np.float32, copy=False), axes


def _robust_display_channel(channel: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    values = channel[valid_mask] if valid_mask is not None and np.any(valid_mask) else channel.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(channel.shape, dtype=np.float32)
    low, high = np.percentile(values, (1.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        return np.zeros(channel.shape, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)


def image_to_rgb(image_cyx: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    channels = [_robust_display_channel(channel, valid_mask) for channel in image_cyx]
    if len(channels) == 1:
        return np.repeat(channels[0][..., None], 3, axis=2)
    if len(channels) == 2:
        # WGA-like channel in magenta and DAPI-like channel in cyan.
        first, second = channels[:2]
        return np.stack((np.maximum(first, second * 0.15), second, np.maximum(first, second)), axis=2)
    return np.stack(channels[:3], axis=2)


def _mask_boundaries(labels: np.ndarray) -> np.ndarray:
    foreground = labels > 0
    eroded = ndi.binary_erosion(foreground, structure=np.ones((3, 3), dtype=bool))
    outer = foreground & ~eroded
    differences = np.zeros(labels.shape, dtype=bool)
    differences[:-1, :] |= labels[:-1, :] != labels[1:, :]
    differences[1:, :] |= labels[1:, :] != labels[:-1, :]
    differences[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    differences[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return outer | (differences & foreground)


def save_overlay(
    image_cyx: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    max_label_text: int = 300,
) -> None:
    rgb = image_to_rgb(image_cyx)
    boundaries = _mask_boundaries(labels)
    overlay = rgb.copy()
    overlay[boundaries] = np.array([1.0, 0.85, 0.0], dtype=np.float32)

    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axis.imshow(overlay)
    axis.set_title(title)
    axis.axis("off")

    positive = [int(value) for value in np.unique(labels) if value > 0]
    if len(positive) <= max_label_text:
        for label_value in positive:
            yy, xx = np.nonzero(labels == label_value)
            if xx.size == 0:
                continue
            axis.text(
                float(xx.mean()),
                float(yy.mean()),
                str(label_value),
                fontsize=6,
                ha="center",
                va="center",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.5, "edgecolor": "none"},
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def inspect_cells(labels: np.ndarray, min_object_pixels: int) -> list[CellCheck]:
    height, width = labels.shape
    checks: list[CellCheck] = []
    for label_value in [int(value) for value in np.unique(labels) if value > 0]:
        binary = labels == label_value
        yy, xx = np.nonzero(binary)
        area = int(xx.size)
        _, components = ndi.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        touches_border = bool(
            np.any(yy == 0)
            or np.any(xx == 0)
            or np.any(yy == height - 1)
            or np.any(xx == width - 1)
        )
        too_small = area < min_object_pixels
        status_terms: list[str] = []
        if too_small:
            status_terms.append("too_small")
        if components != 1:
            status_terms.append("fragmented_label")
        if touches_border:
            status_terms.append("touches_border")
        status = "eligible" if not status_terms else ";".join(status_terms)
        checks.append(
            CellCheck(
                label=label_value,
                area_pixels=area,
                component_count=int(components),
                touches_border=touches_border,
                too_small=too_small,
                centroid_y=float(yy.mean()) if area else math.nan,
                centroid_x=float(xx.mean()) if area else math.nan,
                status=status,
            )
        )
    return checks


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows_list:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def validate_pair(
    pair: AnnotationPair,
    project_root: Path,
    output_dir: Path,
    min_object_pixels: int,
    max_label_text: int,
) -> tuple[PairCheck, list[CellCheck]]:
    messages: list[str] = []
    failures: list[str] = []
    warnings: list[str] = []
    information: list[str] = []
    labels: np.ndarray | None = None
    image: np.ndarray | None = None
    source_dtype = ""
    image_axes = ""

    try:
        labels, source_dtype = load_label_mask(pair.mask_path)
    except Exception as exc:
        failures.append(f"mask_error={exc}")

    try:
        image, image_axes = load_image_cyx(pair.image_path, project_root)
    except Exception as exc:
        failures.append(f"image_error={exc}")

    cells: list[CellCheck] = []
    positive: list[int] = []
    labels_consecutive = True
    missing_count = 0
    overlay_path = output_dir / "mask_image_overlay.png"

    if labels is not None:
        positive = [int(value) for value in np.unique(labels) if value > 0]
        if not positive:
            failures.append("mask_has_no_positive_instances")
        else:
            expected = set(range(min(positive), max(positive) + 1))
            missing = sorted(expected - set(positive))
            labels_consecutive = not missing
            missing_count = len(missing)
            if missing:
                information.append(
                    f"non_consecutive_labels_preserved;missing_count={len(missing)}"
                )
            cells = inspect_cells(labels, min_object_pixels=min_object_pixels)
            if any(cell.too_small for cell in cells):
                warnings.append("small_instances_present")
            if any(cell.component_count != 1 for cell in cells):
                warnings.append("one_or_more_labels_have_disconnected_components")
            if any(cell.touches_border for cell in cells):
                warnings.append("border_touching_instances_present")

    if labels is not None and image is not None:
        if tuple(image.shape[-2:]) != tuple(labels.shape):
            failures.append(
                f"shape_mismatch:image_yx={tuple(image.shape[-2:])};mask_yx={tuple(labels.shape)}"
            )
        else:
            save_overlay(
                image,
                labels,
                overlay_path,
                title=f"{pair.annotation_id}: image with original labels",
                max_label_text=max_label_text,
            )

    messages.extend(failures)
    messages.extend(warnings)
    messages.extend(information)
    overall_status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    pca_eligible = sum(
        1
        for cell in cells
        if not cell.too_small and cell.component_count == 1
    )
    result = PairCheck(
        annotation_id=pair.annotation_id,
        sample_name=pair.sample_name,
        annotation_type=pair.annotation_type,
        split=pair.split,
        mask_path=str(pair.mask_path),
        image_path=str(pair.image_path),
        mask_shape=str(tuple(labels.shape)) if labels is not None else "",
        image_shape=str(tuple(image.shape)) if image is not None else "",
        image_axes=image_axes,
        mask_dtype=source_dtype,
        n_instances=len(positive),
        min_label=min(positive) if positive else None,
        max_label=max(positive) if positive else None,
        labels_consecutive=labels_consecutive,
        missing_label_count=missing_count,
        n_small_instances=sum(cell.too_small for cell in cells),
        n_border_instances=sum(cell.touches_border for cell in cells),
        n_fragmented_labels=sum(cell.component_count != 1 for cell in cells),
        n_pca_eligible_instances=pca_eligible,
        overall_status=overall_status,
        messages=" | ".join(messages),
        overlay_path=str(overlay_path) if overlay_path.exists() else "",
    )
    return result, cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate training or predicted instance masks before PCA analysis."
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--source-mode")
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--include",
        choices=("all", "full", "train", "validation"),
        default="all",
    )
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--min-object-pixels", type=int, default=5)
    parser.add_argument("--max-overlay-labels", type=int, default=300)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root
        else find_project_root()
    )
    source_mode = args.source_mode or DEFAULT_SOURCE_MODES[args.dataset]
    mask_root = (
        args.mask_root.expanduser().resolve()
        if args.mask_root
        else project_root
        / "results"
        / "training_files"
        / "segmentation"
        / args.dataset
        / source_mode
    )
    image_root = (
        args.image_root.expanduser().resolve()
        if args.image_root
        else project_root
        / "results"
        / "segmentation_inputs"
        / args.dataset
        / source_mode
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else project_root
        / "results"
        / "statistics_preparation"
        / "mask_checks"
        / args.dataset
        / source_mode
    )

    if args.min_object_pixels < 1:
        raise ValueError("--min-object-pixels must be at least 1")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output folder is not empty: {output_root}. Use --overwrite to replace reports."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_annotation_pairs(
        mask_root=mask_root,
        image_root=image_root,
        include=args.include,
        sample_filters=args.sample,
    )
    if not pairs:
        raise FileNotFoundError(
            f"No mask annotations were found under {mask_root} for include={args.include!r}."
        )

    pair_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for pair in pairs:
        report_dir = output_root / pair.relative_output
        report_dir.mkdir(parents=True, exist_ok=True)
        result, cells = validate_pair(
            pair=pair,
            project_root=project_root,
            output_dir=report_dir,
            min_object_pixels=args.min_object_pixels,
            max_label_text=args.max_overlay_labels,
        )
        pair_rows.append(asdict(result))
        for cell in cells:
            row = {
                "annotation_id": pair.annotation_id,
                "sample_name": pair.sample_name,
                "annotation_type": pair.annotation_type,
                "split": pair.split,
                "mask_path": str(pair.mask_path),
                "image_path": str(pair.image_path),
            }
            row.update(asdict(cell))
            cell_rows.append(row)

        (report_dir / "check_report.json").write_text(
            json.dumps(
                {
                    "pair": asdict(result),
                    "cells": [asdict(cell) for cell in cells],
                    "label_policy": (
                        "Original positive label values are preserved. Label gaps are warnings, "
                        "not errors. A label with disconnected components is flagged because one "
                        "integer should represent one cell."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{result.overall_status}] {pair.annotation_id}: "
            f"instances={result.n_instances}, PCA-eligible={result.n_pca_eligible_instances}"
        )

    write_csv(output_root / "mask_check_summary.csv", pair_rows)
    write_csv(output_root / "mask_check_cells.csv", cell_rows)
    manifest_rows = [
        {
            "annotation_id": row["annotation_id"],
            "sample_name": row["sample_name"],
            "annotation_type": row["annotation_type"],
            "split": row["split"],
            "mask_path": row["mask_path"],
            "image_path": row["image_path"],
            "overall_status": row["overall_status"],
            "n_instances": row["n_instances"],
            "n_pca_eligible_instances": row["n_pca_eligible_instances"],
        }
        for row in pair_rows
    ]
    write_csv(output_root / "pca_input_manifest.csv", manifest_rows)

    counts = {status: sum(row["overall_status"] == status for row in pair_rows) for status in ("PASS", "WARN", "FAIL")}
    summary = {
        "project_root": str(project_root),
        "dataset": args.dataset,
        "source_mode": source_mode,
        "mask_root": str(mask_root),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "annotations_checked": len(pair_rows),
        "status_counts": counts,
        "total_instances": int(sum(int(row["n_instances"]) for row in pair_rows)),
        "total_pca_eligible_instances": int(
            sum(int(row["n_pca_eligible_instances"]) for row in pair_rows)
        ),
    }
    (output_root / "mask_check_global_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nReports: {output_root}")
    print(f"PCA manifest: {output_root / 'pca_input_manifest.csv'}")
    return 0 if counts["FAIL"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
