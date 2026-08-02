r"""Provide command-line and programmatic utilities for align cells principal-component analysis result.

Examples
--------
Show all command-line parameters:

    python scripts/statistics/align_cells_pca.py --help

Representative 2D execution:

    python scripts/statistics/align_cells_pca.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --exclude-border \
        --overwrite

Representative 3D execution with a reproducible per-ROI cap:

    python scripts/statistics/align_cells_pca.py \
        --dataset 3d_mip \
        --source-mode deconv_masked \
        --exclude-border \
        --max-cells-per-roi-3d 100 \
        --selection-seed 1337 \
        --overwrite
"""

from __future__ import annotations

"""PCA-align cells for one dataset or all supported datasets.

Normal mode processes every eligible annotation from the mask-check manifest.
For ``3d_mip`` full-mode runs, a reproducible per-ROI cap can limit expensive
cell extraction while retaining every ROI. Passing ``--example`` changes only
the image selection: one low-cell full image is chosen per experimental
condition. The PCA calculations are identical for all selected cells.

The script preserves original instance-label values. PCA aligns the major axis
horizontally, but it does not assign biological left and right poles.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

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
        filter_manifest_rows,
        find_project_root,
        manifest_path,
        pca_output_path,
        read_csv,
        select_one_per_condition,
        selected_datasets,
        source_mode_for,
        write_csv,
    )

    return {
        "DATASETS": DATASETS,
        "filter_manifest_rows": filter_manifest_rows,
        "find_project_root": find_project_root,
        "manifest_path": manifest_path,
        "pca_output_path": pca_output_path,
        "read_csv": read_csv,
        "select_one_per_condition": select_one_per_condition,
        "selected_datasets": selected_datasets,
        "source_mode_for": source_mode_for,
        "write_csv": write_csv,
    }


COMMON = _import_common()


def import_check_helpers(project_root: Path):
    """Return import check helpers for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = import_check_helpers(project_root=Path("path/to/resource"))
    """
    script_dir = project_root / "scripts" / "statistics"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    try:
        from check_masks_for_statistics import image_to_rgb, load_image_cyx, load_label_mask
    except ImportError:
        local_dir = SCRIPT_FILE.parent
        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))
        from check_masks_for_statistics import image_to_rgb, load_image_cyx, load_label_mask
    return image_to_rgb, load_image_cyx, load_label_mask


@dataclass(frozen=True)
class PCAResult:
    """Store validated configuration or result data for pcaresult."""
    dataset: str
    source_mode: str
    run_mode: str
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
    selected_for_analysis: bool
    status: str
    cell_output_dir: str


def _safe_relative(annotation_id: str) -> Path:
    """Return safe relative for the supplied inputs.

    Args:
        annotation_id (str): Text value specifying annotation id.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _safe_relative(annotation_id="annotation_id")
    """
    return Path(*[part for part in annotation_id.replace("\\", "/").split("/") if part])



def _stable_roi_seed(base_seed: int, annotation_id: str) -> int:
    """Return a deterministic random seed for one ROI.

    The built-in Python hash is intentionally randomized between processes.
    A SHA-256 digest is therefore used so that the same annotation and base
    seed produce the same selected labels on every supported computer.

    Args:
        base_seed (int): User-defined base seed controlling reproducible sampling.
        annotation_id (str): Stable identifier of one image-mask annotation.

    Returns:
        int: Non-negative seed accepted by ``numpy.random.default_rng``.

    Example:
        >>> _stable_roi_seed(1337, "sample/full") == _stable_roi_seed(
        ...     1337, "sample/full"
        ... )
        True
    """
    digest = hashlib.sha256(annotation_id.encode("utf-8")).digest()
    annotation_value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((int(base_seed) + annotation_value) % (2**63 - 1))


def _select_labels_reproducibly(
    eligible_labels: Sequence[int],
    *,
    maximum: int | None,
    base_seed: int,
    annotation_id: str,
) -> set[int]:
    """Select at most ``maximum`` eligible labels without replacement.

    Sampling is performed independently for every ROI. When the number of
    eligible labels does not exceed the requested maximum, all labels are
    retained. A fixed base seed and stable annotation-specific seed make the
    selection reproducible across runs and computers.

    Args:
        eligible_labels (Sequence[int]): Valid instance-label values available in one ROI.
        maximum (int | None): Maximum retained labels. ``None`` disables the cap.
        base_seed (int): User-defined base seed for reproducible sampling.
        annotation_id (str): Stable identifier of the current ROI.

    Returns:
        set[int]: Selected instance-label values.

    Example:
        >>> selected = _select_labels_reproducibly(
        ...     list(range(1, 201)),
        ...     maximum=100,
        ...     base_seed=1337,
        ...     annotation_id="sample/full",
        ... )
        >>> len(selected)
        100
    """
    ordered = np.asarray(sorted({int(value) for value in eligible_labels}), dtype=np.int64)
    if maximum is None or ordered.size <= maximum:
        return {int(value) for value in ordered.tolist()}
    rng = np.random.default_rng(_stable_roi_seed(base_seed, annotation_id))
    selected = rng.choice(ordered, size=int(maximum), replace=False)
    return {int(value) for value in selected.tolist()}


def _tight_bounds(binary: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    """Return tight bounds for the supplied inputs.

    Args:
        binary (np.ndarray): Array containing binary.
        padding (int): Numerical value controlling padding.

    Returns:
        tuple[int, int, int, int]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _tight_bounds(binary=image_array, padding=1)
    """
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
    """Crop to mask to the requested region.

    Args:
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        image_cyx (np.ndarray): Array containing image cyx.
        padding (int): Numerical value controlling padding.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Example:
        >>> result = _crop_to_mask(
        ...     mask=image_array,
        ...     image_cyx=image_array,
        ...     padding=1,
        ... )
    """
    y0, y1, x0, x1 = _tight_bounds(mask, padding=padding)
    return mask[y0:y1, x0:x1], image_cyx[:, y0:y1, x0:x1]


def _pca(binary: np.ndarray) -> dict[str, float | bool]:
    """Return principal-component analysis result for the supplied inputs.

    Args:
        binary (np.ndarray): Array containing binary.

    Returns:
        dict[str, float | bool]: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _pca(binary=image_array)
    """
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
    """Return rotate cell for the supplied inputs.

    Args:
        mask_crop (np.ndarray): Array containing mask crop.
        image_crop (np.ndarray): Array containing image crop.
        angle_deg (float): Numerical value controlling angle deg.
        output_padding (int): Numerical value controlling output padding.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Example:
        >>> result = _rotate_cell(
        ...     mask_crop=image_array,
        ...     image_crop=image_array,
        ...     angle_deg=0.5,
        ...     output_padding=1,
        ... )
    """
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
    """Return touches border for the supplied inputs.

    Args:
        binary (np.ndarray): Array containing binary.

    Returns:
        bool: ``True`` when the requested condition is satisfied; otherwise ``False``.

    Example:
        >>> result = _touches_border(binary=image_array)
    """
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
    """Save TIFF data cyx to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        image (np.ndarray): Input image array to process.

    Example:
        >>> _save_tiff_cyx(path=Path("path/to/resource"), image=image_array)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, image.astype(np.float32), metadata={"axes": "CYX"})


def _draw_pca_axis(axis, mask: np.ndarray, angle_deg: float) -> None:
    """Draw principal-component analysis result axis on the target visualization.

    Args:
        axis (Any): Array axis along which the operation is performed.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        angle_deg (float): Numerical value controlling angle deg.

    Example:
        >>> _draw_pca_axis(
        ...     axis=...,
        ...     mask=image_array,
        ...     angle_deg=0.5,
        ... )
    """
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
    """Save before after figure to persistent storage.

    Args:
        output_path (Path): Filesystem path where the generated result is written.
        mask_before (np.ndarray): Array containing mask before.
        image_before (np.ndarray): Array containing image before.
        mask_after (np.ndarray): Array containing mask after.
        image_after (np.ndarray): Array containing image after.
        angle_deg (float): Numerical value controlling angle deg.
        title (str): Title displayed on the generated figure or report section.
        image_to_rgb (Any): Value specifying image to RGB representation for the operation.

    Example:
        >>> save_before_after_figure(
        ...     output_path=Path("path/to/resource"),
        ...     mask_before=image_array,
        ...     image_before=image_array,
        ...     mask_after=image_array,
        ...     image_after=image_array,
        ...     angle_deg=0.5,
        ...     title="title",
        ...     image_to_rgb=...,
        ... )
    """
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
    """Save overview to persistent storage.

    Args:
        output_path (Path): Filesystem path where the generated result is written.
        entries (list[tuple[int, np.ndarray, np.ndarray]]): Array containing entries.
        image_to_rgb (Any): Value specifying image to RGB representation for the operation.
        max_cells (int): Maximum permitted value of cells.
        title (str): Title displayed on the generated figure or report section.

    Example:
        >>> save_overview(
        ...     output_path=Path("path/to/resource"),
        ...     entries=image_array,
        ...     image_to_rgb=...,
        ...     max_cells=1,
        ...     title="title",
        ... )
    """
    selected = entries[:max_cells]
    if not selected:
        return
    figure, axes = plt.subplots(
        len(selected),
        2,
        figsize=(8, max(2.2, 2.2 * len(selected))),
        squeeze=False,
        constrained_layout=True,
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


def process_manifest(
    *,
    project_root: Path,
    dataset: str,
    source_mode: str,
    run_mode: str,
    manifest_rows: list[dict[str, object]],
    output_root: Path,
    min_object_pixels: int,
    min_anisotropy: float,
    input_padding: int,
    output_padding: int,
    exclude_border: bool,
    include_failed_pairs: bool,
    max_overview_cells: int,
    max_cells_per_roi: int | None,
    selection_seed: int,
    overwrite: bool,
) -> dict[str, object]:
    """PCA-align validated cells from every selected ROI.

    A per-ROI cap can be applied after structural eligibility checks. The cap
    affects only expensive cell extraction, rotation, normalization and
    downstream per-cell measurements. It does not change the original mask
    counts stored by ``check_masks_for_statistics.py``.

    Args:
        project_root (Path): Root directory of the PFT project.
        dataset (str): Dataset identifier, for example ``"2d_time"`` or ``"3d_mip"``.
        source_mode (str): Preprocessing source used to construct the images.
        run_mode (str): ``"full"`` or ``"example"``.
        manifest_rows (list[dict[str, object]]): Validated image-mask pairs.
        output_root (Path): Directory receiving PCA-aligned cell data.
        min_object_pixels (int): Minimum accepted instance area in pixels.
        min_anisotropy (float): Minimum anisotropy required for rotation.
        input_padding (int): Padding added before rotation.
        output_padding (int): Padding retained after rotation.
        exclude_border (bool): Exclude labels touching the image boundary.
        include_failed_pairs (bool): Permit manifest rows marked as failed.
        max_overview_cells (int): Maximum cells shown in each overview figure.
        max_cells_per_roi (int | None): Maximum valid cells retained per ROI. ``None`` keeps all.
        selection_seed (int): Base seed for reproducible per-ROI sampling.
        overwrite (bool): Replace an existing PCA output directory.

    Returns:
        dict[str, object]: Run-level summary.

    Example:
        >>> summary = process_manifest(
        ...     project_root=Path("project"),
        ...     dataset="3d_mip",
        ...     source_mode="deconv_masked",
        ...     run_mode="full",
        ...     manifest_rows=[],
        ...     output_root=Path("output"),
        ...     min_object_pixels=5,
        ...     min_anisotropy=1.05,
        ...     input_padding=4,
        ...     output_padding=3,
        ...     exclude_border=True,
        ...     include_failed_pairs=False,
        ...     max_overview_cells=20,
        ...     max_cells_per_roi=100,
        ...     selection_seed=1337,
        ...     overwrite=True,
        ... )
    """
    _prepare_output(output_root, overwrite)
    COMMON["write_csv"](output_root / "input_manifest_used.csv", manifest_rows)

    image_to_rgb, load_image_cyx, load_label_mask = import_check_helpers(project_root)
    all_results: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    processed_annotations = 0
    saved_cells = 0
    excluded_cells = 0
    not_selected_cells = 0
    total_labels_seen = 0
    total_valid_before_cap = 0

    for row in manifest_rows:
        pair_status = str(row.get("overall_status", ""))
        if pair_status == "FAIL" and not include_failed_pairs:
            print(f"[SKIP] {row.get('annotation_id')}: validation status FAIL")
            continue

        annotation_id = str(row["annotation_id"])
        sample_name = str(row.get("sample_name", ""))
        split = str(row.get("split", ""))
        mask_path = Path(str(row["mask_path"]))
        image_path = Path(str(row["image_path"]))
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

        label_evaluations: list[dict[str, object]] = []
        for label_value in [int(value) for value in np.unique(labels) if value > 0]:
            binary = labels == label_value
            area = int(np.count_nonzero(binary))
            _, components = ndi.label(
                binary,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            pca = _pca(binary)
            border = _touches_border(binary)
            status_terms: list[str] = []
            should_rotate = True
            if area < min_object_pixels:
                status_terms.append("excluded_too_small")
                should_rotate = False
            if components != 1:
                status_terms.append("excluded_fragmented_label")
                should_rotate = False
            if exclude_border and border:
                status_terms.append("excluded_border")
                should_rotate = False
            if not bool(pca["valid"]):
                status_terms.append("excluded_invalid_pca")
                should_rotate = False
            if float(pca["anisotropy_ratio"]) < min_anisotropy:
                status_terms.append("not_rotated_low_anisotropy")
                should_rotate = False

            excluded = any(term.startswith("excluded_") for term in status_terms)
            label_evaluations.append(
                {
                    "label_value": label_value,
                    "binary": binary,
                    "area": area,
                    "components": int(components),
                    "pca": pca,
                    "border": border,
                    "status_terms": status_terms,
                    "should_rotate": should_rotate,
                    "excluded": excluded,
                }
            )

        eligible_labels = [
            int(item["label_value"])
            for item in label_evaluations
            if not bool(item["excluded"])
        ]
        selected_labels = _select_labels_reproducibly(
            eligible_labels,
            maximum=max_cells_per_roi,
            base_seed=selection_seed,
            annotation_id=annotation_id,
        )
        n_unselected = len(eligible_labels) - len(selected_labels)
        total_labels_seen += len(label_evaluations)
        total_valid_before_cap += len(eligible_labels)
        not_selected_cells += n_unselected

        selection_rows.append(
            {
                "dataset": dataset,
                "source_mode": source_mode,
                "annotation_id": annotation_id,
                "sample_name": sample_name,
                "n_total_instance_labels": len(label_evaluations),
                "n_structurally_eligible_before_cap": len(eligible_labels),
                "n_selected_for_analysis": len(selected_labels),
                "n_not_selected_by_cap": n_unselected,
                "max_cells_per_roi": (
                    "" if max_cells_per_roi is None else int(max_cells_per_roi)
                ),
                "selection_seed": int(selection_seed),
                "roi_seed": _stable_roi_seed(selection_seed, annotation_id),
                "selection_method": (
                    "all eligible cells retained"
                    if max_cells_per_roi is None
                    or len(eligible_labels) <= max_cells_per_roi
                    else "reproducible random sampling without replacement"
                ),
                "selected_source_labels": ";".join(
                    str(value) for value in sorted(selected_labels)
                ),
            }
        )

        for item in label_evaluations:
            label_value = int(item["label_value"])
            binary = np.asarray(item["binary"], dtype=bool)
            area = int(item["area"])
            components = int(item["components"])
            pca = dict(item["pca"])
            border = bool(item["border"])
            status_terms = list(item["status_terms"])
            should_rotate = bool(item["should_rotate"])
            excluded = bool(item["excluded"])
            selected_for_analysis = not excluded and label_value in selected_labels

            if not excluded and not selected_for_analysis:
                status_terms.append("not_selected_per_roi_cap")
                result = PCAResult(
                    dataset=dataset,
                    source_mode=source_mode,
                    run_mode=run_mode,
                    annotation_id=annotation_id,
                    sample_name=sample_name,
                    split=split,
                    source_label=label_value,
                    area_pixels=area,
                    component_count=components,
                    touches_border=border,
                    centroid_y=float(pca["centroid_y"]),
                    centroid_x=float(pca["centroid_x"]),
                    pca_angle_deg=float(pca["angle_deg"]),
                    applied_rotation_deg=0.0,
                    major_variance=float(pca["major_variance"]),
                    minor_variance=float(pca["minor_variance"]),
                    anisotropy_ratio=float(pca["anisotropy_ratio"]),
                    selected_for_analysis=False,
                    status=";".join(status_terms),
                    cell_output_dir="",
                )
                all_results.append(asdict(result))
                continue

            cell_dir = annotation_output / "cells" / f"label_{label_value:06d}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            mask_before, image_before = _crop_to_mask(
                binary,
                image,
                padding=input_padding,
            )
            applied_angle = float(pca["angle_deg"]) if should_rotate else 0.0

            if excluded:
                mask_after = mask_before.copy()
                image_after = image_before.copy()
                excluded_cells += 1
            elif should_rotate:
                mask_after, image_after = _rotate_cell(
                    mask_before,
                    image_before,
                    angle_deg=applied_angle,
                    output_padding=output_padding,
                )
            else:
                mask_after, image_after = _crop_to_mask(
                    mask_before,
                    image_before,
                    padding=output_padding,
                )

            tiff.imwrite(
                cell_dir / "mask_before_pca.tif",
                mask_before.astype(np.uint8),
            )
            tiff.imwrite(
                cell_dir / "mask_after_pca.tif",
                mask_after.astype(np.uint8),
            )
            _save_tiff_cyx(cell_dir / "image_before_pca.tif", image_before)
            _save_tiff_cyx(cell_dir / "image_after_pca.tif", image_after)

            if selected_for_analysis:
                np.savez_compressed(
                    cell_dir / "pca_cell_data.npz",
                    source_label=np.int64(label_value),
                    mask_before=mask_before.astype(np.uint8),
                    mask_pca=mask_after.astype(np.uint8),
                    image_before=image_before.astype(np.float32),
                    image_pca=image_after.astype(np.float32),
                )
                saved_cells += 1

            status = ";".join(status_terms) if status_terms else "aligned"
            result = PCAResult(
                dataset=dataset,
                source_mode=source_mode,
                run_mode=run_mode,
                annotation_id=annotation_id,
                sample_name=sample_name,
                split=split,
                source_label=label_value,
                area_pixels=area,
                component_count=components,
                touches_border=border,
                centroid_y=float(pca["centroid_y"]),
                centroid_x=float(pca["centroid_x"]),
                pca_angle_deg=float(pca["angle_deg"]),
                applied_rotation_deg=applied_angle,
                major_variance=float(pca["major_variance"]),
                minor_variance=float(pca["minor_variance"]),
                anisotropy_ratio=float(pca["anisotropy_ratio"]),
                selected_for_analysis=selected_for_analysis,
                status=status,
                cell_output_dir=str(cell_dir),
            )
            (cell_dir / "pca_metadata.json").write_text(
                json.dumps(
                    {
                        **asdict(result),
                        "source_mask": str(mask_path),
                        "source_image": str(image_path),
                        "selection": {
                            "max_cells_per_roi": max_cells_per_roi,
                            "base_seed": int(selection_seed),
                            "roi_seed": _stable_roi_seed(
                                selection_seed,
                                annotation_id,
                            ),
                            "selected_for_analysis": selected_for_analysis,
                        },
                        "orientation_note": (
                            "The major axis is horizontal after PCA. The eigenvector "
                            "sign is arbitrary, so positions 0 and 1 remain geometric "
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
            max_cells=max_overview_cells,
            title=f"PCA alignment: {annotation_id}",
        )
        processed_annotations += 1
        print(
            f"[PCA] {dataset} | {annotation_id}: "
            f"selected={len(selected_labels)}/{len(eligible_labels)} eligible, "
            f"total_labels={len(label_evaluations)}"
        )

    COMMON["write_csv"](output_root / "pca_alignment_summary.csv", all_results)
    COMMON["write_csv"](
        output_root / "per_roi_cell_selection.csv",
        selection_rows,
    )
    summary = {
        "dataset": dataset,
        "source_mode": source_mode,
        "run_mode": run_mode,
        "output_root": str(output_root),
        "n_manifest_rows": len(manifest_rows),
        "n_processed_annotations": processed_annotations,
        "n_total_instance_labels": total_labels_seen,
        "n_structurally_eligible_before_cap": total_valid_before_cap,
        "n_saved_valid_cells": saved_cells,
        "n_not_selected_by_cap": not_selected_cells,
        "n_excluded_cells": excluded_cells,
        "max_cells_per_roi": max_cells_per_roi,
        "selection_seed": int(selection_seed),
        "selection_method": (
            "reproducible random sampling without replacement per ROI"
            if max_cells_per_roi is not None
            else "all structurally eligible cells"
        ),
        "min_object_pixels": min_object_pixels,
        "min_anisotropy": min_anisotropy,
        "exclude_border": exclude_border,
    }
    (output_root / "pca_run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"PCA output: {output_root}")
    print(f"Per-ROI selection report: {output_root / 'per_roi_cell_selection.csv'}")
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
            "PCA-align validated cells for one dataset or all datasets. "
            "Use --example to select one low-cell image per condition."
        )
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--dataset",
        choices=("all", *COMMON["DATASETS"]),
        default="all",
        help="Dataset to process; default: all.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional explicit manifest for backward compatibility. This requires "
            "one concrete --dataset and bypasses automatic manifest discovery."
        ),
    )
    parser.add_argument("--source-mode")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--preferred-min-cells", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=80)
    parser.add_argument(
        "--max-cells-per-roi-2d-time",
        type=int,
        default=100,
        help=(
            "Maximum structurally eligible cells retained from each 2d_time ROI "
            "in full mode. Use 0 to disable the cap. Default: 100."
        ),
    )
    parser.add_argument(
        "--max-cells-per-roi-3d",
        type=int,
        default=100,
        help=(
            "Maximum structurally eligible cells retained from each 3d_mip ROI "
            "in full mode. Use 0 to disable the cap. Default: 100."
        ),
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=1337,
        help="Base seed for reproducible per-ROI cell selection. Default: 1337.",
    )
    parser.add_argument("--exclude-time", type=int, action="append", default=[])
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument(
        "--include",
        choices=("full", "all", "train", "validation"),
        default="full",
        help="Annotations used in normal mode. Example mode always uses full images.",
    )
    parser.add_argument("--min-object-pixels", type=int, default=5)
    parser.add_argument("--min-anisotropy", type=float, default=1.05)
    parser.add_argument("--input-padding", type=int, default=4)
    parser.add_argument("--output-padding", type=int, default=3)
    parser.add_argument("--exclude-border", action="store_true")
    parser.add_argument("--include-failed-pairs", action="store_true")
    parser.add_argument("--max-overview-cells", type=int, default=20)
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
    if args.min_object_pixels < 1:
        raise ValueError("--min-object-pixels must be at least 1")
    if args.min_anisotropy < 1.0:
        raise ValueError("--min-anisotropy must be at least 1.0")
    if args.preferred_min_cells < 1:
        raise ValueError("--preferred-min-cells must be at least 1")
    if args.max_cells is not None and args.max_cells < 1:
        raise ValueError("--max-cells must be at least 1")
    if args.max_cells_per_roi_2d_time < 0:
        raise ValueError("--max-cells-per-roi-2d-time must be 0 or greater")
    if args.max_cells_per_roi_3d < 0:
        raise ValueError("--max-cells-per-roi-3d must be 0 or greater")
    if args.manifest and args.dataset == "all":
        raise ValueError("An explicit --manifest requires one concrete --dataset.")
    if args.example and args.include != "full":
        print("[INFO] --example uses full-image annotations; --include is ignored.")

    summaries: list[dict[str, object]] = []
    datasets = COMMON["selected_datasets"](args.dataset)
    for dataset in datasets:
        source_mode = COMMON["source_mode_for"](dataset, args.source_mode, args.dataset)
        source_manifest = (
            args.manifest.expanduser().resolve()
            if args.manifest
            else COMMON["manifest_path"](project_root, dataset, source_mode)
        )
        if not source_manifest.exists():
            message = (
                f"Missing mask-check manifest for {dataset}: {source_manifest}. "
                "Run check_masks_for_statistics.py first."
            )
            if args.skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise FileNotFoundError(message)

        raw_rows = COMMON["read_csv"](source_manifest)
        filtered = COMMON["filter_manifest_rows"](
            raw_rows,
            dataset=dataset,
            include="full" if args.example else args.include,
            excluded_times=set(args.exclude_time),
            sample_filters=args.sample,
            include_failed_pairs=args.include_failed_pairs,
        )
        if args.example:
            selected_rows, report = COMMON["select_one_per_condition"](
                filtered,
                dataset=dataset,
                preferred_min_cells=args.preferred_min_cells,
                max_cells=args.max_cells,
            )
            manifest_rows = selected_rows
        else:
            report = []
            manifest_rows = filtered

        if not manifest_rows:
            message = f"No eligible manifest rows for {dataset}."
            if args.skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise ValueError(message)

        default_output = COMMON["pca_output_path"](
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

        max_cells_per_roi: int | None = None
        if not args.example:
            if dataset == "2d_time" and args.max_cells_per_roi_2d_time > 0:
                max_cells_per_roi = int(args.max_cells_per_roi_2d_time)
            elif dataset == "3d_mip" and args.max_cells_per_roi_3d > 0:
                max_cells_per_roi = int(args.max_cells_per_roi_3d)

        if dataset in {"2d_time", "3d_mip"} and not args.example:
            if max_cells_per_roi is None:
                print(f"[INFO] {dataset} per-ROI cell cap is disabled.")
            else:
                print(
                    f"[INFO] {dataset} will retain at most "
                    f"{max_cells_per_roi} valid cells per ROI "
                    f"with seed {args.selection_seed}."
                )

        summary = process_manifest(
            project_root=project_root,
            dataset=dataset,
            source_mode=source_mode,
            run_mode="example" if args.example else "full",
            manifest_rows=manifest_rows,
            output_root=output_root,
            min_object_pixels=args.min_object_pixels,
            min_anisotropy=args.min_anisotropy,
            input_padding=args.input_padding,
            output_padding=args.output_padding,
            exclude_border=args.exclude_border,
            include_failed_pairs=args.include_failed_pairs,
            max_overview_cells=args.max_overview_cells,
            max_cells_per_roi=max_cells_per_roi,
            selection_seed=args.selection_seed,
            overwrite=args.overwrite,
        )
        if report:
            COMMON["write_csv"](output_root / "example_selection_report.csv", report)
            print("Selected one image per available condition:")
            for item in report:
                print(
                    f"  {item['condition']}: {item['sample_name']} "
                    f"({item['n_pca_eligible_cells']} cells)"
                )
        summaries.append(summary)

    if not summaries:
        raise RuntimeError("No datasets were processed.")

    batch_summary = {
        "requested_dataset": args.dataset,
        "run_mode": "example" if args.example else "full",
        "processed_datasets": [item["dataset"] for item in summaries],
        "max_cells_per_roi_3d": int(args.max_cells_per_roi_3d),
        "selection_seed": int(args.selection_seed),
        "summaries": summaries,
    }
    batch_path = (
        project_root
        / "results"
        / "statistics_preparation"
        / ("pca_aligned_example" if args.example else "pca_aligned")
        / "pca_batch_summary.json"
    )
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"\nPCA batch summary: {batch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
