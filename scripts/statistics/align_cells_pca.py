from __future__ import annotations

"""Rotate every valid cell and its image channels using PCA.

The script consumes the validation manifest produced by
``check_masks_for_statistics.py``. Every original positive label value is
retained in filenames and metadata. The PCA eigenvector sign is not interpreted
as a biological pole identity; the aligned major axis is horizontal, but a
180-degree left-right ambiguity remains.
"""

import argparse
import csv
import json
import math
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


def find_project_root(start: Path | None = None) -> Path:
    current = (start or SCRIPT_FILE).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT project root; provide --project-root.")


def import_check_helpers(project_root: Path):
    script_dir = project_root / "scripts" / "statistics"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from check_masks_for_statistics import image_to_rgb, load_image_cyx, load_label_mask
    except ImportError:
        # This fallback supports testing the downloaded scripts before they are copied.
        local_dir = SCRIPT_FILE.parent
        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))
        from check_masks_for_statistics import image_to_rgb, load_image_cyx, load_label_mask
    return image_to_rgb, load_image_cyx, load_label_mask


@dataclass(frozen=True)
class PCAResult:
    annotation_id: str
    sample_name: str
    split: str
    source_label: int
    area_pixels: int
    component_count: int
    touches_border: bool
    centroid_y: float
    centroid_x: float
    pca_angle_deg: float
    applied_rotation_deg: float
    major_variance: float
    minor_variance: float
    anisotropy_ratio: float
    status: str
    cell_output_dir: str


def _safe_relative(annotation_id: str) -> Path:
    return Path(*[part for part in annotation_id.replace("\\", "/").split("/") if part])


def _tight_bounds(binary: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(binary)
    if xx.size == 0:
        raise ValueError("Cannot crop an empty cell mask")
    y0 = max(0, int(yy.min()) - padding)
    y1 = min(binary.shape[0], int(yy.max()) + 1 + padding)
    x0 = max(0, int(xx.min()) - padding)
    x1 = min(binary.shape[1], int(xx.max()) + 1 + padding)
    return y0, y1, x0, x1


def _crop_to_mask(
    mask: np.ndarray,
    image_cyx: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = _tight_bounds(mask, padding=padding)
    return mask[y0:y1, x0:x1], image_cyx[:, y0:y1, x0:x1]


def _pca(binary: np.ndarray) -> dict[str, float | bool]:
    yy, xx = np.nonzero(binary)
    area = int(xx.size)
    if area < 3:
        return {
            "valid": False,
            "angle_deg": 0.0,
            "major_variance": 0.0,
            "minor_variance": 0.0,
            "anisotropy_ratio": 1.0,
            "centroid_y": float(yy.mean()) if area else math.nan,
            "centroid_x": float(xx.mean()) if area else math.nan,
        }
    points = np.column_stack((xx.astype(np.float64), yy.astype(np.float64)))
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(area - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    major = float(max(values[order[0]], 0.0))
    minor = float(max(values[order[1]], 0.0))
    vector = vectors[:, order[0]]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    epsilon = np.finfo(np.float64).eps
    return {
        "valid": major > epsilon,
        "angle_deg": float(angle),
        "major_variance": major,
        "minor_variance": minor,
        "anisotropy_ratio": float(major / max(minor, epsilon)) if major > 0 else 1.0,
        "centroid_y": float(centroid[1]),
        "centroid_x": float(centroid[0]),
    }


def _rotate_cell(
    mask_crop: np.ndarray,
    image_crop: np.ndarray,
    angle_deg: float,
    output_padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotated_mask = ndi.rotate(
        mask_crop.astype(np.uint8),
        angle=float(angle_deg),
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) > 0
    rotated_channels = [
        ndi.rotate(
            channel.astype(np.float32),
            angle=float(angle_deg),
            reshape=True,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        for channel in image_crop
    ]
    rotated_image = np.stack(rotated_channels, axis=0).astype(np.float32)
    return _crop_to_mask(rotated_mask, rotated_image, padding=output_padding)


def _touches_border(binary: np.ndarray) -> bool:
    yy, xx = np.nonzero(binary)
    if xx.size == 0:
        return False
    height, width = binary.shape
    return bool(
        np.any(yy == 0)
        or np.any(xx == 0)
        or np.any(yy == height - 1)
        or np.any(xx == width - 1)
    )


def _save_tiff_cyx(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, image.astype(np.float32), metadata={"axes": "CYX"})


def _draw_pca_axis(axis, mask: np.ndarray, angle_deg: float) -> None:
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return
    cx, cy = float(xx.mean()), float(yy.mean())
    length = max(mask.shape) * 0.42
    radians = math.radians(angle_deg)
    dx = math.cos(radians) * length
    dy = math.sin(radians) * length
    axis.plot([cx - dx, cx + dx], [cy - dy, cy + dy], linewidth=1.5)


def save_before_after_figure(
    output_path: Path,
    mask_before: np.ndarray,
    image_before: np.ndarray,
    mask_after: np.ndarray,
    image_after: np.ndarray,
    angle_deg: float,
    title: str,
    image_to_rgb,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    axes[0, 0].imshow(mask_before, cmap="gray", interpolation="nearest")
    _draw_pca_axis(axes[0, 0], mask_before, angle_deg)
    axes[0, 0].set_title("Mask before PCA")
    axes[0, 1].imshow(mask_after, cmap="gray", interpolation="nearest")
    axes[0, 1].set_title("Mask after PCA")
    axes[1, 0].imshow(image_to_rgb(image_before, mask_before))
    axes[1, 0].set_title("Image before PCA")
    axes[1, 1].imshow(image_to_rgb(image_after, mask_after))
    axes[1, 1].set_title("Image after PCA")
    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_overview(
    output_path: Path,
    entries: list[tuple[int, np.ndarray, np.ndarray]],
    image_to_rgb,
    max_cells: int,
    title: str,
) -> None:
    selected = entries[:max_cells]
    if not selected:
        return
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(8, max(2.2, 2.2 * len(selected))), squeeze=False, constrained_layout=True
    )
    for row, (label_value, before, after) in enumerate(selected):
        axes[row, 0].imshow(image_to_rgb(before))
        axes[row, 0].set_title(f"Label {label_value}: before")
        axes[row, 1].imshow(image_to_rgb(after))
        axes[row, 1].set_title(f"Label {label_value}: after")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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
    parser = argparse.ArgumentParser(description="PCA-align every cell in validated instance masks.")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--min-object-pixels", type=int, default=5)
    parser.add_argument("--min-anisotropy", type=float, default=1.05)
    parser.add_argument("--input-padding", type=int, default=4)
    parser.add_argument("--output-padding", type=int, default=3)
    parser.add_argument("--exclude-border", action="store_true")
    parser.add_argument("--include-failed-pairs", action="store_true")
    parser.add_argument("--max-overview-cells", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root
        else find_project_root()
    )
    manifest = args.manifest.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else manifest.parent.parent.parent.parent / "pca_aligned" / manifest.parent.parent.name / manifest.parent.name
    )
    if args.min_object_pixels < 1:
        raise ValueError("--min-object-pixels must be at least 1")
    if args.min_anisotropy < 1.0:
        raise ValueError("--min-anisotropy must be at least 1.0")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output folder is not empty: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_to_rgb, load_image_cyx, load_label_mask = import_check_helpers(project_root)
    manifest_rows = _read_manifest(manifest)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest}")

    all_results: list[dict[str, object]] = []
    for row in manifest_rows:
        pair_status = row.get("overall_status", "")
        if pair_status == "FAIL" and not args.include_failed_pairs:
            print(f"[SKIP] {row.get('annotation_id')}: validation status FAIL")
            continue

        annotation_id = row["annotation_id"]
        sample_name = row.get("sample_name", "")
        split = row.get("split", "")
        mask_path = Path(row["mask_path"])
        image_path = Path(row["image_path"])
        labels, _ = load_label_mask(mask_path)
        image, _ = load_image_cyx(image_path, project_root)
        if tuple(labels.shape) != tuple(image.shape[-2:]):
            raise ValueError(
                f"Image-mask shape mismatch for {annotation_id}: "
                f"mask={labels.shape}, image={image.shape}"
            )

        annotation_output = output_root / _safe_relative(annotation_id)
        annotation_output.mkdir(parents=True, exist_ok=True)
        overview_entries: list[tuple[int, np.ndarray, np.ndarray]] = []

        for label_value in [int(value) for value in np.unique(labels) if value > 0]:
            binary = labels == label_value
            area = int(np.count_nonzero(binary))
            _, components = ndi.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
            pca = _pca(binary)
            border = _touches_border(binary)
            status_terms: list[str] = []
            should_rotate = True
            if area < args.min_object_pixels:
                status_terms.append("excluded_too_small")
                should_rotate = False
            if components != 1:
                status_terms.append("excluded_fragmented_label")
                should_rotate = False
            if args.exclude_border and border:
                status_terms.append("excluded_border")
                should_rotate = False
            if not bool(pca["valid"]):
                status_terms.append("excluded_invalid_pca")
                should_rotate = False
            if float(pca["anisotropy_ratio"]) < args.min_anisotropy:
                status_terms.append("not_rotated_low_anisotropy")
                should_rotate = False

            cell_dir = annotation_output / "cells" / f"label_{label_value:06d}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            mask_before, image_before = _crop_to_mask(binary, image, padding=args.input_padding)
            applied_angle = float(pca["angle_deg"]) if should_rotate else 0.0

            excluded = any(term.startswith("excluded_") for term in status_terms)
            if excluded:
                mask_after = mask_before.copy()
                image_after = image_before.copy()
            elif should_rotate:
                mask_after, image_after = _rotate_cell(
                    mask_before,
                    image_before,
                    angle_deg=applied_angle,
                    output_padding=args.output_padding,
                )
            else:
                mask_after, image_after = _crop_to_mask(
                    mask_before, image_before, padding=args.output_padding
                )

            tiff.imwrite(cell_dir / "mask_before_pca.tif", mask_before.astype(np.uint8))
            tiff.imwrite(cell_dir / "mask_after_pca.tif", mask_after.astype(np.uint8))
            _save_tiff_cyx(cell_dir / "image_before_pca.tif", image_before)
            _save_tiff_cyx(cell_dir / "image_after_pca.tif", image_after)
            np.savez_compressed(
                cell_dir / "pca_cell_data.npz",
                source_label=np.int64(label_value),
                mask_before=mask_before.astype(np.uint8),
                mask_pca=mask_after.astype(np.uint8),
                image_before=image_before.astype(np.float32),
                image_pca=image_after.astype(np.float32),
            )

            status = ";".join(status_terms) if status_terms else "aligned"
            result = PCAResult(
                annotation_id=annotation_id,
                sample_name=sample_name,
                split=split,
                source_label=label_value,
                area_pixels=area,
                component_count=int(components),
                touches_border=border,
                centroid_y=float(pca["centroid_y"]),
                centroid_x=float(pca["centroid_x"]),
                pca_angle_deg=float(pca["angle_deg"]),
                applied_rotation_deg=applied_angle,
                major_variance=float(pca["major_variance"]),
                minor_variance=float(pca["minor_variance"]),
                anisotropy_ratio=float(pca["anisotropy_ratio"]),
                status=status,
                cell_output_dir=str(cell_dir),
            )
            (cell_dir / "pca_metadata.json").write_text(
                json.dumps(
                    {
                        **asdict(result),
                        "source_mask": str(mask_path),
                        "source_image": str(image_path),
                        "orientation_note": (
                            "The major axis is horizontal after PCA. The sign of the "
                            "eigenvector remains arbitrary, so left and right are geometric "
                            "poles rather than assigned biological poles."
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            save_before_after_figure(
                cell_dir / "pca_before_after.png",
                mask_before,
                image_before,
                mask_after,
                image_after,
                angle_deg=float(pca["angle_deg"]),
                title=f"{annotation_id}, original label {label_value}",
                image_to_rgb=image_to_rgb,
            )
            overview_entries.append((label_value, image_before, image_after))
            all_results.append(asdict(result))

        save_overview(
            annotation_output / "pca_before_after_overview.png",
            overview_entries,
            image_to_rgb=image_to_rgb,
            max_cells=args.max_overview_cells,
            title=f"PCA alignment: {annotation_id}",
        )
        print(f"[PCA] {annotation_id}: {len(overview_entries)} labels processed")

    _write_csv(output_root / "pca_alignment_summary.csv", all_results)
    (output_root / "pca_run_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "output_root": str(output_root),
                "n_cell_records": len(all_results),
                "min_object_pixels": args.min_object_pixels,
                "min_anisotropy": args.min_anisotropy,
                "exclude_border": args.exclude_border,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nPCA output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
