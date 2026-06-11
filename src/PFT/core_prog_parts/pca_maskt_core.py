from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi


SUPPORTED_METHODS = ("cellpose", "omnipose", "stardist")
SUPPORTED_DATASETS = ("2d_time", "2d_wga_dapi", "3d")

SmallObjectPolicy = Literal["keep", "exclude"]
CollisionPolicy = Literal["keep_larger", "overwrite"]


@dataclass
class PCAMaskAlignmentConfig:
    """
    Configuration for PCA-based alignment of instance-label masks.

    """

    project_root: Path
    method: str
    dataset: str
    prediction_run_dir: Path
    sample_names: tuple[str, ...] | None = None
    output_subfolder: str = "pca_aligned"
    min_object_pixels: int = 5
    min_anisotropy_ratio: float = 1.05
    crop_padding: int = 4
    small_object_policy: SmallObjectPolicy = "keep"
    collision_policy: CollisionPolicy = "keep_larger"
    save_combined_mask: bool = True
    save_individual_cells: bool = True
    save_omezarr: bool = True
    overwrite: bool = False

    def validate(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.prediction_run_dir = Path(self.prediction_run_dir).expanduser().resolve()
        self.method = normalize_method(self.method)
        self.dataset = normalize_dataset(self.dataset)

        if not self.prediction_run_dir.exists():
            raise FileNotFoundError(
                f"Prediction run directory does not exist: {self.prediction_run_dir}"
            )
        if self.min_object_pixels < 1:
            raise ValueError("min_object_pixels must be at least 1")
        if self.min_anisotropy_ratio < 1.0:
            raise ValueError("min_anisotropy_ratio must be at least 1.0")
        if self.crop_padding < 0:
            raise ValueError("crop_padding cannot be negative")
        if self.small_object_policy not in {"keep", "exclude"}:
            raise ValueError(
                "small_object_policy must be either 'keep' or 'exclude'"
            )
        if self.collision_policy not in {"keep_larger", "overwrite"}:
            raise ValueError(
                "collision_policy must be either 'keep_larger' or 'overwrite'"
            )
        if not self.save_combined_mask and not self.save_individual_cells:
            raise ValueError(
                "At least one output must be enabled: combined mask or individual cells"
            )

        output_path = Path(self.output_subfolder)
        if (
            not self.output_subfolder.strip()
            or output_path.is_absolute()
            or len(output_path.parts) != 1
            or output_path.name in {".", ".."}
        ):
            raise ValueError(
                "output_subfolder must be a single folder name without path separators"
            )


def normalize_method(method: str) -> str:
    value = str(method).strip().lower().replace("-", "_")
    aliases = {
        "cell_pose": "cellpose",
        "omni": "omnipose",
        "star_dist": "stardist",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}. Choose one of {SUPPORTED_METHODS}."
        )
    return value


def normalize_dataset(dataset: str) -> str:
    value = str(dataset).strip().lower().replace("-", "_")
    aliases = {
        "2dtime": "2d_time",
        "time": "2d_time",
        "2d_wagi_dapi": "2d_wga_dapi",
        "wga_dapi": "2d_wga_dapi",
        "wagi_dapi": "2d_wga_dapi",
        "3d_sim": "3d",
        "3d_25d": "3d",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Choose one of {SUPPORTED_DATASETS}."
        )
    return value


def _prediction_root_candidates(
    project_root: Path,
    method: str,
    dataset: str,
) -> list[Path]:
    """Return portable candidate roots used by the existing segmentation scripts."""
    method = normalize_method(method)
    dataset = normalize_dataset(dataset)
    results = Path(project_root) / "results"

    candidates = [
        results / "segmentation_predictions" / method / dataset,
        results / f"segmentation_{method}" / dataset,
        results / f"{method}_segmentation" / dataset,
    ]

    if method == "cellpose":
        candidates.insert(0, results / "segmentation_cellpose" / dataset)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_mask_path(sample_dir: Path) -> Path | None:
    """Find the source label mask in one sample prediction directory."""
    sample_dir = Path(sample_dir)
    candidates = (
        sample_dir / "labels.tif",
        sample_dir / "labels.tiff",
        sample_dir / "mask.tif",
        sample_dir / "mask.tiff",
        sample_dir / "labels.ome.zarr",
        sample_dir / "mask.ome.zarr",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def list_mask_samples(prediction_run_dir: Path) -> list[tuple[str, Path, Path]]:
    """List ``(sample_name, sample_directory, mask_path)`` entries."""
    prediction_run_dir = Path(prediction_run_dir)
    if not prediction_run_dir.exists():
        return []

    entries: list[tuple[str, Path, Path]] = []
    for sample_dir in sorted(
        (path for path in prediction_run_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        mask_path = find_mask_path(sample_dir)
        if mask_path is not None:
            entries.append((sample_dir.name, sample_dir, mask_path))
    return entries


def discover_prediction_runs(
    project_root: Path,
    method: str,
    dataset: str,
) -> list[Path]:
    """Discover model/run directories without using computer-specific paths."""
    project_root = Path(project_root).expanduser().resolve()
    runs: list[Path] = []

    for base in _prediction_root_candidates(project_root, method, dataset):
        if not base.exists():
            continue

        if list_mask_samples(base):
            runs.append(base)

        for child in sorted(
            (path for path in base.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        ):
            if list_mask_samples(child):
                runs.append(child)

    deduplicated: list[Path] = []
    seen: set[str] = set()
    for run in runs:
        resolved = run.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduplicated.append(resolved)
    return deduplicated


def _load_label_mask(path: Path, dataset: str) -> tuple[np.ndarray, str]:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        labels = np.asarray(tiff.imread(path))
    elif path.name.lower().endswith(".ome.zarr"):
        try:
            from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
        except ImportError as exc:
            raise ImportError(
                "OME-Zarr loading requires PFT.core_prog_parts.decoder_omezar"
            ) from exc
        array, _axes = load_ome_zarr(path, level=0, as_numpy=False)
        labels = np.asarray(array)
    else:
        raise ValueError(f"Unsupported label-mask format: {path}")

    labels = np.squeeze(labels)
    dataset = normalize_dataset(dataset)
    expected_ndim = 3 if dataset == "3d" else 2
    if labels.ndim != expected_ndim:
        raise ValueError(
            f"Expected a {expected_ndim}D label mask for {dataset}, "
            f"but loaded shape {labels.shape} from {path}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        labels = np.rint(labels).astype(np.int64)
    if np.any(labels < 0):
        raise ValueError(f"Label masks cannot contain negative values: {path}")

    axes = "zyx" if labels.ndim == 3 else "yx"
    return labels.astype(np.int64, copy=False), axes


def _ensure_instance_labels(plane: np.ndarray) -> tuple[np.ndarray, bool]:
    """Convert a binary mask to connected-component labels when necessary."""
    plane = np.asarray(plane)
    positive = plane[plane > 0]
    if positive.size == 0:
        return np.zeros_like(plane, dtype=np.int64), False

    unique_positive = np.unique(positive)
    is_binary = unique_positive.size == 1 and int(unique_positive[0]) == 1
    if is_binary:
        connected, _count = ndi.label(plane > 0)
        return connected.astype(np.int64, copy=False), True
    return plane.astype(np.int64, copy=False), False


def _pca_orientation(binary: np.ndarray) -> dict[str, float | bool]:
    """Calculate the major PCA axis for one binary object in image coordinates."""
    yy, xx = np.nonzero(binary)
    area = int(xx.size)
    if area < 3:
        return {
            "pca_valid": False,
            "angle_deg": 0.0,
            "major_variance": 0.0,
            "minor_variance": 0.0,
            "anisotropy_ratio": 1.0,
            "centroid_y": float(yy.mean()) if area else math.nan,
            "centroid_x": float(xx.mean()) if area else math.nan,
        }

    points_xy = np.column_stack((xx.astype(np.float64), yy.astype(np.float64)))
    centroid_xy = points_xy.mean(axis=0)
    centered = points_xy - centroid_xy
    covariance = centered.T @ centered / max(area - 1, 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = float(max(eigenvalues[order[0]], 0.0))
    minor = float(max(eigenvalues[order[1]], 0.0))
    major_vector = eigenvectors[:, order[0]]

    angle = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0])))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0

    epsilon = np.finfo(np.float64).eps
    anisotropy = major / max(minor, epsilon) if major > 0.0 else 1.0
    return {
        "pca_valid": bool(major > epsilon),
        "angle_deg": float(angle),
        "major_variance": major,
        "minor_variance": minor,
        "anisotropy_ratio": float(anisotropy),
        "centroid_y": float(centroid_xy[1]),
        "centroid_x": float(centroid_xy[0]),
    }


def _tight_crop(binary: np.ndarray, padding: int = 0) -> np.ndarray:
    yy, xx = np.nonzero(binary)
    if xx.size == 0:
        return np.zeros((1, 1), dtype=bool)

    y0 = max(0, int(yy.min()) - padding)
    y1 = min(binary.shape[0], int(yy.max()) + 1 + padding)
    x0 = max(0, int(xx.min()) - padding)
    x1 = min(binary.shape[1], int(xx.max()) + 1 + padding)
    return np.asarray(binary[y0:y1, x0:x1], dtype=bool)


def _rotate_to_horizontal(
    binary: np.ndarray,
    angle_deg: float,
    padding: int,
) -> np.ndarray:
    """
    Rotate one object so its major PCA axis becomes horizontal.

    """
    source = _tight_crop(binary, padding=max(1, padding))
    rotated = ndi.rotate(
        source.astype(np.uint8),
        angle=float(angle_deg),
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ) > 0
    return _tight_crop(rotated, padding=padding)


def _paste_centered(
    destination: np.ndarray,
    cell_crop: np.ndarray,
    label_value: int,
    centroid_y: float,
    centroid_x: float,
    collision_policy: CollisionPolicy,
) -> tuple[int, int, int]:
    """Paste an aligned cell around its original centroid.

    Returns ``(placed_pixels, collision_pixels, clipped_pixels)``.
    """
    height, width = cell_crop.shape
    y0 = int(round(centroid_y - (height - 1) / 2.0))
    x0 = int(round(centroid_x - (width - 1) / 2.0))
    y1 = y0 + height
    x1 = x0 + width

    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(destination.shape[0], y1)
    dst_x1 = min(destination.shape[1], x1)

    total_pixels = int(np.count_nonzero(cell_crop))
    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return 0, 0, total_pixels

    src_y0 = dst_y0 - y0
    src_x0 = dst_x0 - x0
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    crop_view = cell_crop[src_y0:src_y1, src_x0:src_x1]
    dest_view = destination[dst_y0:dst_y1, dst_x0:dst_x1]

    in_bounds_pixels = int(np.count_nonzero(crop_view))
    clipped_pixels = max(0, total_pixels - in_bounds_pixels)
    collisions = crop_view & (dest_view != 0)
    collision_pixels = int(np.count_nonzero(collisions))

    if collision_policy == "keep_larger":
        writable = crop_view & (dest_view == 0)
        dest_view[writable] = int(label_value)
        placed_pixels = int(np.count_nonzero(writable))
    else:
        dest_view[crop_view] = int(label_value)
        placed_pixels = in_bounds_pixels

    return placed_pixels, collision_pixels, clipped_pixels


def _label_dtype(labels: np.ndarray) -> np.dtype:
    maximum = int(np.max(labels)) if labels.size else 0
    if maximum <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _save_omezarr(path: Path, labels: np.ndarray, axes: str) -> None:
    try:
        from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
    except ImportError as exc:
        raise ImportError(
            "OME-Zarr saving requires PFT.core_prog_parts.omezarr_utils"
        ) from exc

    save_ome_zarr(
        path,
        labels,
        axes=axes,
        meta=None,
        overwrite=True,
        pyramid_3d=("z" in axes),
    )


def _serializable_config(cfg: PCAMaskAlignmentConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["project_root"] = str(cfg.project_root)
    data["prediction_run_dir"] = str(cfg.prediction_run_dir)
    if cfg.sample_names is not None:
        data["sample_names"] = list(cfg.sample_names)
    return data


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _save_cell_crop(
    cells_root: Path,
    cell_crop: np.ndarray,
    label_value: int,
    z_index: int | None,
) -> Path:
    if z_index is None:
        out_dir = cells_root
        filename = f"cell_label_{label_value:06d}.tif"
    else:
        out_dir = cells_root / f"z{z_index:03d}"
        filename = f"cell_z{z_index:03d}_label_{label_value:06d}.tif"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    tiff.imwrite(path, cell_crop.astype(np.uint8), photometric="minisblack")
    return path


def _align_plane(
    source_plane: np.ndarray,
    cfg: PCAMaskAlignmentConfig,
    cells_root: Path | None,
    z_index: int | None,
) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    instance_plane, binary_was_relabelled = _ensure_instance_labels(source_plane)
    aligned = np.zeros(instance_plane.shape, dtype=np.int64)

    label_values = [int(value) for value in np.unique(instance_plane) if value != 0]
    objects: list[tuple[int, int]] = [
        (value, int(np.count_nonzero(instance_plane == value)))
        for value in label_values
    ]

    if cfg.collision_policy == "keep_larger":
        objects.sort(key=lambda item: (-item[1], item[0]))
    else:
        objects.sort(key=lambda item: item[0])

    records: list[dict[str, Any]] = []
    for label_value, area_pixels in objects:
        binary = instance_plane == label_value
        pca = _pca_orientation(binary)

        status = "aligned"
        applied_angle = float(pca["angle_deg"])
        should_rotate = True

        if area_pixels < cfg.min_object_pixels:
            should_rotate = False
            applied_angle = 0.0
            status = (
                "small_object_kept"
                if cfg.small_object_policy == "keep"
                else "small_object_excluded"
            )
        elif not bool(pca["pca_valid"]):
            should_rotate = False
            applied_angle = 0.0
            status = "invalid_pca_kept"
        elif float(pca["anisotropy_ratio"]) < cfg.min_anisotropy_ratio:
            should_rotate = False
            applied_angle = 0.0
            status = "low_anisotropy_kept"

        if status == "small_object_excluded":
            records.append(
                {
                    "z_index": "" if z_index is None else z_index,
                    "label": label_value,
                    "area_pixels": area_pixels,
                    "centroid_y": pca["centroid_y"],
                    "centroid_x": pca["centroid_x"],
                    "pca_angle_deg": pca["angle_deg"],
                    "applied_rotation_deg": 0.0,
                    "major_variance": pca["major_variance"],
                    "minor_variance": pca["minor_variance"],
                    "anisotropy_ratio": pca["anisotropy_ratio"],
                    "status": status,
                    "aligned_crop_path": "",
                    "placed_pixels": 0,
                    "collision_pixels": 0,
                    "clipped_pixels": 0,
                }
            )
            continue

        if should_rotate:
            cell_crop = _rotate_to_horizontal(
                binary,
                angle_deg=applied_angle,
                padding=cfg.crop_padding,
            )
        else:
            cell_crop = _tight_crop(binary, padding=cfg.crop_padding)

        crop_path = ""
        if cells_root is not None:
            crop_path = str(
                _save_cell_crop(
                    cells_root,
                    cell_crop,
                    label_value=label_value,
                    z_index=z_index,
                )
            )

        placed_pixels = collision_pixels = clipped_pixels = 0
        if cfg.save_combined_mask:
            placed_pixels, collision_pixels, clipped_pixels = _paste_centered(
                aligned,
                cell_crop,
                label_value=label_value,
                centroid_y=float(pca["centroid_y"]),
                centroid_x=float(pca["centroid_x"]),
                collision_policy=cfg.collision_policy,
            )

        records.append(
            {
                "z_index": "" if z_index is None else z_index,
                "label": label_value,
                "area_pixels": area_pixels,
                "centroid_y": pca["centroid_y"],
                "centroid_x": pca["centroid_x"],
                "pca_angle_deg": pca["angle_deg"],
                "applied_rotation_deg": applied_angle,
                "major_variance": pca["major_variance"],
                "minor_variance": pca["minor_variance"],
                "anisotropy_ratio": pca["anisotropy_ratio"],
                "status": status,
                "aligned_crop_path": crop_path,
                "placed_pixels": placed_pixels,
                "collision_pixels": collision_pixels,
                "clipped_pixels": clipped_pixels,
            }
        )

    return aligned, records, binary_was_relabelled


def align_mask_file(
    mask_path: Path,
    sample_dir: Path,
    cfg: PCAMaskAlignmentConfig,
) -> Path:
    """Align all instances in one 2D or slice-wise 3D label mask."""
    mask_path = Path(mask_path)
    sample_dir = Path(sample_dir)
    labels, axes = _load_label_mask(mask_path, cfg.dataset)

    output_dir = sample_dir / cfg.output_subfolder
    if output_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Enable overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_root = output_dir / "cells" if cfg.save_individual_cells else None
    records: list[dict[str, Any]] = []
    binary_relabelled = False

    if labels.ndim == 2:
        aligned, plane_records, relabelled = _align_plane(
            labels,
            cfg,
            cells_root=cells_root,
            z_index=None,
        )
        aligned_labels = aligned
        records.extend(plane_records)
        binary_relabelled = relabelled
    else:
        aligned_labels = np.zeros(labels.shape, dtype=np.int64)
        for z_index in range(labels.shape[0]):
            aligned_plane, plane_records, relabelled = _align_plane(
                labels[z_index],
                cfg,
                cells_root=cells_root,
                z_index=z_index,
            )
            aligned_labels[z_index] = aligned_plane
            records.extend(plane_records)
            binary_relabelled = binary_relabelled or relabelled

    aligned_labels = aligned_labels.astype(_label_dtype(aligned_labels), copy=False)

    tiff_path: Path | None = None
    zarr_path: Path | None = None
    if cfg.save_combined_mask:
        tiff_path = output_dir / "labels_pca_aligned.tif"
        tiff.imwrite(tiff_path, aligned_labels, photometric="minisblack")
        if cfg.save_omezarr:
            zarr_path = output_dir / "labels_pca_aligned.ome.zarr"
            _save_omezarr(zarr_path, aligned_labels, axes=axes)

    _write_records_csv(output_dir / "pca_alignment.csv", records)

    statuses: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        statuses[status] = statuses.get(status, 0) + 1

    summary = {
        "config": _serializable_config(cfg),
        "source_mask": str(mask_path),
        "source_shape": list(labels.shape),
        "source_dtype": str(labels.dtype),
        "output_axes": axes,
        "output_tiff": str(tiff_path) if tiff_path is not None else None,
        "output_omezarr": str(zarr_path) if zarr_path is not None else None,
        "individual_cells_directory": str(cells_root) if cells_root is not None else None,
        "binary_source_was_connected_component_labelled": binary_relabelled,
        "n_objects": len(records),
        "status_counts": statuses,
        "total_collision_pixels": int(
            sum(int(record["collision_pixels"]) for record in records)
        ),
        "total_clipped_pixels": int(
            sum(int(record["clipped_pixels"]) for record in records)
        ),
        "three_dimensional_processing": (
            "slice-wise 2D PCA; object identities are not linked across z"
            if labels.ndim == 3
            else None
        ),
        "orientation_note": (
            "PCA aligns the major axis horizontally. A mask alone cannot resolve "
            "the 180-degree left-right orientation ambiguity."
        ),
    }
    (output_dir / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return output_dir


def run_pca_alignment(cfg: PCAMaskAlignmentConfig) -> list[Path]:
    """Run PCA alignment for the configured prediction run and samples."""
    cfg.validate()
    available = list_mask_samples(cfg.prediction_run_dir)
    if not available:
        raise FileNotFoundError(
            "No sample folders containing labels.tif, labels.tiff, or "
            f"labels.ome.zarr were found under {cfg.prediction_run_dir}"
        )

    selected_names = set(cfg.sample_names) if cfg.sample_names is not None else None
    selected = [
        entry for entry in available
        if selected_names is None or entry[0] in selected_names
    ]

    if selected_names is not None:
        missing = sorted(selected_names - {entry[0] for entry in selected})
        if missing:
            raise FileNotFoundError(
                f"Selected samples were not found in the run directory: {missing}"
            )
    if not selected:
        raise RuntimeError("No mask samples were selected for PCA alignment")

    outputs: list[Path] = []
    for sample_name, sample_dir, mask_path in selected:
        output_dir = align_mask_file(mask_path, sample_dir, cfg)
        outputs.append(output_dir)
        print(
            f"[PCA alignment | {cfg.method} | {cfg.dataset}] "
            f"{sample_name} -> {output_dir}"
        )
    return outputs


def _run_for_dataset(
    cfg: PCAMaskAlignmentConfig,
    expected_dataset: str,
) -> list[Path]:
    actual = normalize_dataset(cfg.dataset)
    if actual != expected_dataset:
        raise ValueError(
            f"This entry function requires dataset={expected_dataset!r}, "
            f"but received {actual!r}"
        )
    return run_pca_alignment(cfg)


def run_pca_alignment_2d_time(
    cfg: PCAMaskAlignmentConfig,
) -> list[Path]:
    """Align Cellpose, Omnipose, or StarDist masks for the 2D time dataset."""
    return _run_for_dataset(cfg, "2d_time")


def run_pca_alignment_2d_wga_dapi(
    cfg: PCAMaskAlignmentConfig,
) -> list[Path]:
    """Align masks for the 2D WGA-DAPI dataset."""
    return _run_for_dataset(cfg, "2d_wga_dapi")


def run_pca_alignment_3d(
    cfg: PCAMaskAlignmentConfig,
) -> list[Path]:
    """Align each z-plane independently for the current slice-wise 3D masks."""
    return _run_for_dataset(cfg, "3d")


__all__ = [
    "PCAMaskAlignmentConfig",
    "SUPPORTED_METHODS",
    "SUPPORTED_DATASETS",
    "align_mask_file",
    "discover_prediction_runs",
    "find_mask_path",
    "list_mask_samples",
    "normalize_dataset",
    "normalize_method",
    "run_pca_alignment",
    "run_pca_alignment_2d_time",
    "run_pca_alignment_2d_wga_dapi",
    "run_pca_alignment_3d",
]
