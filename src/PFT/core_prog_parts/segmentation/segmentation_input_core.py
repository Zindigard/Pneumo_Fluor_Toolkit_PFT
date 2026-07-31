"""
Prepare and validate quantitative inputs for downstream instance segmentation.

Prepared outputs are float32 OME-Zarr arrays in [0, 1] and are stored below::

    results/segmentation_inputs/<dataset>/<source_mode>/<sample>/
        segmentation_input.ome.zarr

Manual instance masks are stored separately below::

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/mask.tif
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Literal, Sequence
import json

import numpy as np

from PFT.core_prog_parts.common_paths import find_project_root, normalize_dataset_name
from PFT.core_prog_parts.decoder_omezar import ensure_czyx, load_ome_zarr
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr

DatasetName = Literal["2d_time", "2d_wga_dapi", "3d_mip"]
SourceMode = Literal[
    "filtered_unet",
    "raw_unmasked",
    "raw_masked",
    "deconv_masked",
]

SUPPORTED_DATASETS: tuple[str, ...] = ("2d_time", "2d_wga_dapi", "3d_mip")
SOURCE_MODES_BY_DATASET: dict[str, tuple[str, ...]] = {
    "2d_time": ("filtered_unet",),
    "2d_wga_dapi": ("filtered_unet",),
    "3d_mip": ("raw_unmasked", "raw_masked", "deconv_masked"),
}
MIP_DIRECTORY_BY_MODE: dict[str, str] = {
    "raw_unmasked": "raw_mip",
    "raw_masked": "raw_mip_masked",
    "deconv_masked": "deconv_332_mip_masked",
}
DEFAULT_FILTER_LABEL: dict[str, str] = {
    "2d_time": "p97.5_k3f40_s370_k10f40",
    "2d_wga_dapi": "p98.0_k3f40_s370_k10f40",
}
DEFAULT_UNET_RUN: dict[str, str] = {
    "2d_time": "2d_time_v2_fixedval_zero_outside",
    "2d_wga_dapi": "2d_wga_dapi_v2_fixedval_zero_outside",
}


@dataclass(frozen=True)
class SegmentationSource:
    """Resolved files required to create one segmentation input."""

    dataset: str
    source_mode: str
    sample_key: str
    source_zarr: Path
    intensity_source_zarr: Path
    mask_zarr: Path | None
    raw_reference_zarr: Path | None
    output_zarr: Path
    training_mask: Path


@dataclass(frozen=True)
class PreparedSegmentationInput:
    """Summary of one successfully written segmentation input."""

    dataset: str
    source_mode: str
    sample_key: str
    source_zarr: Path
    intensity_source_zarr: Path
    mask_zarr: Path | None
    output_zarr: Path
    source_axes: str
    output_axes: str
    source_dtype: str
    intensity_source_dtype: str
    output_dtype: str
    output_shape: tuple[int, ...]
    output_min: float
    output_max: float
    nonzero_pixels: int
    mask_pixels: int | None
    created_utc: str


def _validate_dataset_mode(dataset: str, source_mode: str) -> tuple[str, str]:
    dataset = normalize_dataset_name(dataset)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported segmentation dataset: {dataset!r}")
    if source_mode not in SOURCE_MODES_BY_DATASET[dataset]:
        raise ValueError(
            f"Source mode {source_mode!r} is invalid for {dataset!r}; "
            f"expected one of {SOURCE_MODES_BY_DATASET[dataset]}"
        )
    return dataset, source_mode


def segmentation_input_root(
    project_root: Path,
    dataset: str,
    source_mode: str,
) -> Path:
    """Return and create the prepared-input root for one dataset and source."""
    dataset, source_mode = _validate_dataset_mode(dataset, source_mode)
    path = Path(project_root) / "results" / "segmentation_inputs" / dataset / source_mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def segmentation_mask_root(
    project_root: Path,
    dataset: str,
    source_mode: str,
) -> Path:
    """Return and create the requested manual instance-mask root."""
    dataset, source_mode = _validate_dataset_mode(dataset, source_mode)
    path = (
        Path(project_root)
        / "results"
        / "training_files"
        / "segmentation"
        / dataset
        / source_mode
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def segmentation_model_root(
    project_root: Path,
    family: str,
    dataset: str,
    source_mode: str,
) -> Path:
    """Return and create the canonical trained-model root."""
    dataset, source_mode = _validate_dataset_mode(dataset, source_mode)
    path = (
        Path(project_root)
        / "models"
        / "segmentation"
        / family.lower()
        / dataset
        / source_mode
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def segmentation_prediction_root(
    project_root: Path,
    family: str,
    dataset: str,
    source_mode: str,
    model_name: str,
) -> Path:
    """Return and create the canonical prediction root."""
    dataset, source_mode = _validate_dataset_mode(dataset, source_mode)
    path = (
        Path(project_root)
        / "results"
        / "segmentation_predictions"
        / family.lower()
        / dataset
        / source_mode
        / model_name
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _root_attrs(path: Path) -> dict[str, Any]:
    """Read JSON-compatible root attributes from one OME-Zarr store."""
    try:
        import zarr

        group = zarr.open_group(str(path), mode="r")
        return dict(group.attrs.asdict())
    except Exception:
        return {}


def _coordinate_scale(path: Path, level: int = 0) -> list[float] | None:
    """Read the OME-NGFF coordinate scale for one image level when available."""
    attrs = _root_attrs(path)
    try:
        datasets = attrs["multiscales"][0]["datasets"]
        transforms = datasets[level].get("coordinateTransformations", [])
        for transform in transforms:
            if transform.get("type") == "scale":
                return [float(value) for value in transform["scale"]]
    except Exception:
        return None
    return None


def _processing_attrs(path: Path) -> dict[str, Any]:
    attrs = _root_attrs(path)
    processing = attrs.get("pft_processing", {})
    return dict(processing) if isinstance(processing, dict) else {}


def _rebase_recorded_path(value: str | Path, project_root: Path) -> Path:
    """Resolve paths stored on another Windows workstation against this project."""
    candidate = Path(str(value)).expanduser()
    if candidate.exists():
        return candidate.resolve()

    windows_parts = list(PureWindowsPath(str(value)).parts)
    lower = [part.lower() for part in windows_parts]
    for anchor in ("results", "models", "scripts", "src"):
        if anchor in lower:
            index = lower.index(anchor)
            rebased = Path(project_root).joinpath(*windows_parts[index:])
            if rebased.exists():
                return rebased.resolve()

    posix_parts = list(candidate.parts)
    lower_posix = [part.lower() for part in posix_parts]
    for anchor in ("results", "models", "scripts", "src"):
        if anchor in lower_posix:
            index = lower_posix.index(anchor)
            rebased = Path(project_root).joinpath(*posix_parts[index:])
            if rebased.exists():
                return rebased.resolve()
    return candidate


def _latest_directory(candidates: Iterable[Path], preferred_name: str | None = None) -> Path:
    paths = [path for path in candidates if path.is_dir()]
    if not paths:
        raise FileNotFoundError("No matching processing directory was found")
    if preferred_name:
        for path in paths:
            if path.name == preferred_name:
                return path.resolve()
    paths.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return paths[0].resolve()


def resolve_filter_root(
    project_root: Path,
    dataset: str,
    explicit_root: Path | None = None,
) -> Path:
    """Resolve the intensity-preserving local-threshold output root."""
    dataset = normalize_dataset_name(dataset)
    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    base = (
        Path(project_root)
        / "results"
        / "Filters"
        / "Local_high_threshold"
        / dataset
        / "intensity_preserved"
    )
    return _latest_directory(base.glob("*"), DEFAULT_FILTER_LABEL.get(dataset))


def resolve_unet_root(
    project_root: Path,
    dataset: str,
    explicit_root: Path | None = None,
) -> Path:
    """Resolve a U-Net inference root containing ``pred_mask.ome.zarr`` files."""
    dataset = normalize_dataset_name(dataset)
    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    base = Path(project_root) / "results" / "U-net"
    candidates = [
        path
        for path in base.glob(f"{dataset}*")
        if path.is_dir() and any(path.glob("*/pred_mask.ome.zarr"))
    ]
    return _latest_directory(candidates, DEFAULT_UNET_RUN.get(dataset))


def _raw_2d_reference(project_root: Path, dataset: str, sample_key: str) -> Path | None:
    sample = Path(sample_key).name
    candidates = [
        Path(project_root) / "results" / "img" / dataset / sample / "image.ome.zarr",
        Path(project_root) / "results" / "img" / "2d_data" / dataset / sample / "image.ome.zarr",
    ]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    return None


def _raw_3d_reference_from_processing(
    source_zarr: Path,
    project_root: Path,
) -> Path | None:
    processing = _processing_attrs(source_zarr)
    raw = processing.get("raw_source_omezarr")
    if raw:
        path = _rebase_recorded_path(raw, project_root)
        if path.is_dir():
            return path.resolve()
    return None


def discover_segmentation_sources(
    project_root: Path | None,
    dataset: str,
    source_mode: str,
    *,
    filter_root: Path | None = None,
    unet_root: Path | None = None,
    mip_root: Path | None = None,
) -> list[SegmentationSource]:
    """Discover all source files required for one segmentation dataset.

    The returned list is deterministic and retains experiment subdirectories for
    3D MIP samples, preventing collisions between repeated sample names.
    """
    root = Path(project_root or find_project_root()).resolve()
    dataset, source_mode = _validate_dataset_mode(dataset, source_mode)
    output_root = segmentation_input_root(root, dataset, source_mode)
    mask_root = segmentation_mask_root(root, dataset, source_mode)
    records: list[SegmentationSource] = []

    if dataset in {"2d_time", "2d_wga_dapi"}:
        selected_filter_root = resolve_filter_root(root, dataset, filter_root)
        selected_unet_root = resolve_unet_root(root, dataset, unet_root)
        images = sorted(selected_filter_root.glob("*/image.ome.zarr"))
        if not images:
            raise FileNotFoundError(
                f"No filtered image.ome.zarr stores found below {selected_filter_root}"
            )
        for image in images:
            sample_key = image.parent.name
            mask = selected_unet_root / sample_key / "pred_mask.ome.zarr"
            records.append(
                SegmentationSource(
                    dataset=dataset,
                    source_mode=source_mode,
                    sample_key=sample_key,
                    source_zarr=image.resolve(),
                    intensity_source_zarr=image.resolve(),
                    mask_zarr=mask.resolve() if mask.is_dir() else mask,
                    raw_reference_zarr=_raw_2d_reference(root, dataset, sample_key),
                    output_zarr=output_root / sample_key / "segmentation_input.ome.zarr",
                    training_mask=mask_root / sample_key / "mask.tif",
                )
            )
        return records

    selected_mip_root = Path(mip_root).expanduser().resolve() if mip_root else (
        root / "results" / "mip_2d" / MIP_DIRECTORY_BY_MODE[source_mode]
    )
    if not selected_mip_root.is_dir():
        raise FileNotFoundError(selected_mip_root)
    images = sorted(selected_mip_root.glob("**/image.ome.zarr"))
    if not images:
        raise FileNotFoundError(f"No MIP image.ome.zarr stores found below {selected_mip_root}")

    for image in images:
        relative_parent = image.parent.relative_to(selected_mip_root)
        sample_key = relative_parent.as_posix()
        processing = _processing_attrs(image)
        intensity_source = image
        mask: Path | None = None
        if source_mode in {"raw_masked", "deconv_masked"}:
            recorded_intensity = processing.get("projection_source_omezarr")
            recorded_mask = processing.get("mask_omezarr")
            if recorded_intensity:
                intensity_source = _rebase_recorded_path(recorded_intensity, root)
            if recorded_mask:
                mask = _rebase_recorded_path(recorded_mask, root)
        records.append(
            SegmentationSource(
                dataset=dataset,
                source_mode=source_mode,
                sample_key=sample_key,
                source_zarr=image.resolve(),
                intensity_source_zarr=intensity_source,
                mask_zarr=mask,
                raw_reference_zarr=_raw_3d_reference_from_processing(image, root),
                output_zarr=output_root / relative_parent / "segmentation_input.ome.zarr",
                training_mask=mask_root / relative_parent / "mask.tif",
            )
        )
    return records


def _normalization_slices(shape: Sequence[int], axes: str) -> Iterable[tuple[Any, ...]]:
    """Yield one YX plane per frame and numerical channel."""
    axes = axes.lower()
    if len(shape) != len(axes):
        raise ValueError(f"Axes {axes!r} do not match shape {tuple(shape)}")
    if "y" not in axes or "x" not in axes:
        raise ValueError(f"Spatial YX axes are required, received {axes!r}")

    plane_axes = {axes.index("y"), axes.index("x")}
    iter_axes = [index for index in range(len(shape)) if index not in plane_axes]
    iter_shape = tuple(int(shape[index]) for index in iter_axes)
    coordinates = np.ndindex(iter_shape) if iter_shape else [()]
    for coordinate in coordinates:
        selection: list[Any] = [slice(None)] * len(shape)
        for axis_index, axis_coordinate in zip(iter_axes, coordinate):
            selection[axis_index] = int(axis_coordinate)
        yield tuple(selection)


def normalize_complete_image_per_channel(
    image: np.ndarray,
    axes: str,
    *,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Normalize every complete YX channel plane independently to [0, 1]."""
    if not 0.0 <= p_low < p_high <= 100.0:
        raise ValueError("Expected 0 <= p_low < p_high <= 100")
    source = np.asarray(image)
    if not np.issubdtype(source.dtype, np.number):
        raise TypeError(f"Expected numerical input, received {source.dtype}")
    if not np.all(np.isfinite(source)):
        raise ValueError("Intensity source contains NaN or infinite values")

    output = np.zeros(source.shape, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for plane_index, selection in enumerate(_normalization_slices(source.shape, axes)):
        plane = np.asarray(source[selection], dtype=np.float32)
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            raise ValueError(f"Normalization plane {plane_index} has no finite values")
        low = float(np.percentile(finite, p_low))
        high = float(np.percentile(finite, p_high))
        fallback = False
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
            fallback = True
        if high <= low:
            raise ValueError(
                f"Normalization plane {plane_index} has no usable intensity range: "
                f"low={low}, high={high}"
            )
        output[selection] = np.clip((plane - low) / (high - low), 0.0, 1.0)
        records.append(
            {
                "plane_index": plane_index,
                "selection": [str(item) for item in selection],
                "p_low": float(p_low),
                "p_high": float(p_high),
                "low": low,
                "high": high,
                "fallback_to_min_max": fallback,
            }
        )
    return output.astype(np.float32, copy=False), records


def _broadcast_mask(mask: np.ndarray, mask_axes: str, target_shape: Sequence[int], target_axes: str) -> np.ndarray:
    """Align a binary mask to an image axis order and broadcast over channels."""
    target_axes = target_axes.lower()
    mask_axes = mask_axes.lower()
    array = np.asarray(mask)
    if len(mask_axes) != array.ndim:
        raise ValueError(f"Mask axes {mask_axes!r} do not match shape {array.shape}")

    # Remove unsupported singleton dimensions, but never silently discard data.
    for axis in tuple(mask_axes):
        if axis not in target_axes:
            index = mask_axes.index(axis)
            if array.shape[index] != 1:
                raise ValueError(
                    f"Mask contains unsupported non-singleton axis {axis}={array.shape[index]}"
                )
            array = np.take(array, 0, axis=index)
            mask_axes = mask_axes[:index] + mask_axes[index + 1 :]

    current_axes = list(mask_axes)
    for axis in target_axes:
        if axis not in current_axes:
            array = np.expand_dims(array, axis=array.ndim)
            current_axes.append(axis)
    permutation = [current_axes.index(axis) for axis in target_axes]
    array = np.transpose(array, permutation)

    for axis_index, (mask_size, target_size) in enumerate(zip(array.shape, target_shape)):
        if mask_size not in {1, int(target_size)}:
            raise ValueError(
                f"Mask/image mismatch on axis {target_axes[axis_index]!r}: "
                f"mask={mask_size}, image={target_size}"
            )
    return np.broadcast_to(array > 0, tuple(int(value) for value in target_shape))


def create_segmentation_input(
    intensity_image: np.ndarray,
    intensity_axes: str,
    *,
    predicted_mask: np.ndarray | None = None,
    mask_axes: str | None = None,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create a validated float32 downstream segmentation input."""
    normalized, normalization_records = normalize_complete_image_per_channel(
        intensity_image,
        intensity_axes,
        p_low=p_low,
        p_high=p_high,
    )
    if predicted_mask is not None:
        if mask_axes is None:
            raise ValueError("mask_axes is required when predicted_mask is supplied")
        broadcast = _broadcast_mask(
            predicted_mask,
            mask_axes,
            normalized.shape,
            intensity_axes,
        )
        normalized = np.where(broadcast, normalized, 0.0).astype(np.float32)
        if np.any(normalized[~broadcast] != 0):
            raise RuntimeError("Nonzero values remain outside the predicted mask")

    if normalized.dtype != np.float32:
        raise RuntimeError(f"Prepared input must be float32, received {normalized.dtype}")
    if not np.all(np.isfinite(normalized)):
        raise RuntimeError("Prepared input contains NaN or infinite values")
    if float(np.min(normalized)) < 0.0 or float(np.max(normalized)) > 1.0:
        raise RuntimeError("Prepared input is outside [0, 1]")
    if np.count_nonzero(normalized) == 0:
        raise RuntimeError(
            "Prepared segmentation input is completely black. Check the intensity "
            "source and predicted foreground mask."
        )
    return normalized, normalization_records


def _load_array(path: Path, level: int = 0) -> tuple[np.ndarray, str]:
    if not Path(path).is_dir():
        raise FileNotFoundError(path)
    array, axes = load_ome_zarr(Path(path), level=level, as_numpy=True)
    return np.asarray(array), str(axes).lower()


def _load_projection_mip(path: Path) -> tuple[np.ndarray, str, str]:
    """Create an unmasked CYX MIP from a recorded CZYX projection source."""
    from PFT.core_prog_parts.segmentation.mip_3d_core import (
        maximum_intensity_projection_cyx,
    )

    if not Path(path).is_dir():
        raise FileNotFoundError(path)
    array, axes = load_ome_zarr(Path(path), level=0, as_numpy=False)
    array, axes = ensure_czyx(array, str(axes).lower())
    mip = maximum_intensity_projection_cyx(array)
    return np.asarray(mip), "cyx", str(np.dtype(array.dtype))


def _load_mip_mask(
    path: Path,
    *,
    target_slice_1based: int | None,
    expected_yx: tuple[int, int],
) -> tuple[np.ndarray, str]:
    mask, axes = _load_array(path)
    axes = axes.lower()
    if "c" in axes:
        index = axes.index("c")
        if mask.shape[index] != 1:
            raise ValueError(f"MIP mask must have one channel, received {mask.shape}, {axes}")
        mask = np.take(mask, 0, axis=index)
        axes = axes[:index] + axes[index + 1 :]
    if "t" in axes:
        index = axes.index("t")
        if mask.shape[index] != 1:
            raise ValueError(f"MIP mask must have one time point, received {mask.shape}, {axes}")
        mask = np.take(mask, 0, axis=index)
        axes = axes[:index] + axes[index + 1 :]
    if "z" in axes:
        if target_slice_1based is None:
            raise ValueError("MIP mask contains Z but no target slice is recorded")
        index = axes.index("z")
        z_index = int(target_slice_1based) - 1
        if not 0 <= z_index < mask.shape[index]:
            raise IndexError(
                f"Target Z{target_slice_1based} outside mask Z={mask.shape[index]}"
            )
        mask = np.take(mask, z_index, axis=index)
        axes = axes[:index] + axes[index + 1 :]
    if axes != "yx":
        raise ValueError(f"Expected final MIP mask axes yx, received {axes!r}")
    if tuple(mask.shape) != tuple(expected_yx):
        raise ValueError(f"MIP/mask shape mismatch: {expected_yx} versus {mask.shape}")
    return (mask > 0).astype(np.uint8), "yx"


def prepare_one_segmentation_input(
    record: SegmentationSource,
    *,
    overwrite: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> PreparedSegmentationInput:
    """Create and save one segmentation input from a resolved source record."""
    source_array, source_axes = _load_array(record.source_zarr)
    source_dtype = str(source_array.dtype)
    mask_array: np.ndarray | None = None
    mask_axes: str | None = None

    if record.dataset == "3d_mip" and record.source_mode in {"raw_masked", "deconv_masked"}:
        intensity_array, intensity_axes, intensity_source_dtype = _load_projection_mip(
            record.intensity_source_zarr
        )
        processing = _processing_attrs(record.source_zarr)
        target_slice = processing.get("target_slice_1based_for_qc_and_mask")
        if record.mask_zarr is None:
            raise FileNotFoundError(
                f"Masked MIP has no recorded mask_omezarr metadata: {record.source_zarr}"
            )
        mask_array, mask_axes = _load_mip_mask(
            record.mask_zarr,
            target_slice_1based=int(target_slice) if target_slice is not None else None,
            expected_yx=tuple(int(value) for value in intensity_array.shape[-2:]),
        )
        coordinate_scale = _coordinate_scale(record.source_zarr)
    else:
        intensity_array = source_array
        intensity_axes = source_axes
        intensity_source_dtype = str(intensity_array.dtype)
        coordinate_scale = _coordinate_scale(record.source_zarr)
        if record.mask_zarr is not None:
            mask_array, mask_axes = _load_array(record.mask_zarr)

    prepared, normalization_records = create_segmentation_input(
        intensity_array,
        intensity_axes,
        predicted_mask=mask_array,
        mask_axes=mask_axes,
        p_low=p_low,
        p_high=p_high,
    )

    processing = {
        "operation": "prepare_instance_segmentation_input",
        "dataset": record.dataset,
        "source_mode": record.source_mode,
        "sample_key": record.sample_key,
        "source_omezarr": str(record.source_zarr),
        "intensity_source_omezarr": str(record.intensity_source_zarr),
        "predicted_mask_omezarr": str(record.mask_zarr) if record.mask_zarr else None,
        "normalization_order": "normalize_complete_source_then_apply_mask",
        "normalization": {
            "method": "percentile_linear_clip",
            "p_low": float(p_low),
            "p_high": float(p_high),
            "independent_per_numerical_channel": True,
            "calculated_from_sparse_masked_result": False,
        },
        "outside_mask_value": 0.0 if record.mask_zarr else None,
        "stored_dtype": "float32",
        "stored_range": [0.0, 1.0],
        "model_normalization_required": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = save_ome_zarr(
        record.output_zarr,
        prepared,
        intensity_axes,
        overwrite=overwrite,
        pyramid_3d=False,
        pyramid_max_layer=0,
        coordinate_scale=coordinate_scale,
        extra_attrs={"pft_processing": processing},
    )
    report = {
        "record": asdict(record),
        "processing": processing,
        "normalization_planes": normalization_records,
        "source_axes": source_axes,
        "source_dtype": source_dtype,
        "intensity_axes": intensity_axes,
        "intensity_source_dtype": intensity_source_dtype,
        "output_shape": list(prepared.shape),
        "output_dtype": str(prepared.dtype),
        "output_min": float(np.min(prepared)),
        "output_max": float(np.max(prepared)),
        "output_nonzero": int(np.count_nonzero(prepared)),
        "mask_pixels": int(np.count_nonzero(mask_array)) if mask_array is not None else None,
    }
    report_path = Path(output).parent / "segmentation_input_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return PreparedSegmentationInput(
        dataset=record.dataset,
        source_mode=record.source_mode,
        sample_key=record.sample_key,
        source_zarr=record.source_zarr,
        intensity_source_zarr=record.intensity_source_zarr,
        mask_zarr=record.mask_zarr,
        output_zarr=Path(output),
        source_axes=source_axes,
        output_axes=intensity_axes,
        source_dtype=source_dtype,
        intensity_source_dtype=intensity_source_dtype,
        output_dtype=str(prepared.dtype),
        output_shape=tuple(int(value) for value in prepared.shape),
        output_min=float(np.min(prepared)),
        output_max=float(np.max(prepared)),
        nonzero_pixels=int(np.count_nonzero(prepared)),
        mask_pixels=int(np.count_nonzero(mask_array)) if mask_array is not None else None,
        created_utc=processing["created_utc"],
    )


def prepare_dataset_segmentation_inputs(
    project_root: Path | None,
    dataset: str,
    source_mode: str,
    *,
    filter_root: Path | None = None,
    unet_root: Path | None = None,
    mip_root: Path | None = None,
    overwrite: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.8,
    selected_sample_keys: set[str] | None = None,
) -> list[PreparedSegmentationInput]:
    """Prepare all or selected inputs for one independent dataset."""
    records = discover_segmentation_sources(
        project_root,
        dataset,
        source_mode,
        filter_root=filter_root,
        unet_root=unet_root,
        mip_root=mip_root,
    )
    if selected_sample_keys is not None:
        records = [record for record in records if record.sample_key in selected_sample_keys]
    if not records:
        raise RuntimeError("No segmentation source records were selected")

    outputs: list[PreparedSegmentationInput] = []
    for record in records:
        outputs.append(
            prepare_one_segmentation_input(
                record,
                overwrite=overwrite,
                p_low=p_low,
                p_high=p_high,
            )
        )
    return outputs


__all__ = [
    "DEFAULT_FILTER_LABEL",
    "DEFAULT_UNET_RUN",
    "MIP_DIRECTORY_BY_MODE",
    "PreparedSegmentationInput",
    "SOURCE_MODES_BY_DATASET",
    "SUPPORTED_DATASETS",
    "SegmentationSource",
    "create_segmentation_input",
    "discover_segmentation_sources",
    "normalize_complete_image_per_channel",
    "prepare_dataset_segmentation_inputs",
    "prepare_one_segmentation_input",
    "resolve_filter_root",
    "resolve_unet_root",
    "segmentation_input_root",
    "segmentation_mask_root",
    "segmentation_model_root",
    "segmentation_prediction_root",
]
