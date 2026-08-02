"""Prepare and validate quantitative inputs for downstream instance segmentation.

This module defines one preprocessing contract for Cellpose, Omnipose, and
StarDist. It supports the two independent 2D datasets and the 3D-to-2D MIP
products used by the PFT project.

The critical rule for U-Net-masked data is::

    segmentation_input = normalize(full_intensity_source) * foreground_mask

Percentiles are calculated independently for each numerical channel from the
complete, non-sparse intensity source. The mask is applied only after
normalization. A sparse ``foreground_filtered.ome.zarr`` or masked MIP is never
renormalized directly.

For ``2d_time``, numerical channel 0 is always selected before normalization;
any additional stored channel is ignored.

Prepared outputs are float32 OME-Zarr arrays in [0, 1] and are stored below::

    results/segmentation_inputs/<dataset>/<source_mode>/<sample>/
        segmentation_input.ome.zarr
        preview_raw.png
        preview_normalized.png
        preview_comparison.png

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
from PIL import Image

from PFT.core_prog_parts.common_paths import (
    find_project_root,
    normalize_dataset_name,
    project_relative_path,
    resolve_project_path,
)
from PFT.core_prog_parts.decoder_omezar import (
    ensure_czyx,
    load_ome_zarr,
    load_ome_zarr_direct,
    select_index_along_axis,
)
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
    preview_raw_png: Path
    preview_normalized_png: Path
    preview_comparison_png: Path
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
    """Validate dataset mode against the required constraints.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.

    Returns:
        tuple[str, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _validate_dataset_mode(dataset="2d_time", source_mode="original")
    """
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
    """Return and create the prepared-input root for one dataset and source.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = segmentation_input_root(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
    """
    dataset, source_mode = _validate_dataset_mode(dataset, source_mode)
    path = Path(project_root) / "results" / "segmentation_inputs" / dataset / source_mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def segmentation_mask_root(
    project_root: Path,
    dataset: str,
    source_mode: str,
) -> Path:
    """Return and create the requested manual instance-mask root.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = segmentation_mask_root(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
    """
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
    """Return and create the canonical trained-model root.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        family (str): Instance-segmentation model family to use, such as Cellpose, Omnipose, or StarDist.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = segmentation_model_root(
        ...     project_root=Path("path/to/resource"),
        ...     family="cellpose",
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
    """
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
    """Return and create the canonical prediction root.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        family (str): Instance-segmentation model family to use, such as Cellpose, Omnipose, or StarDist.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        model_name (str): Human-readable model name used in output paths and reports.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = segmentation_prediction_root(
        ...     project_root=Path("path/to/resource"),
        ...     family="cellpose",
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     model_name="model_name",
        ... )
    """
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
    """Read JSON-compatible root attributes from one OME-Zarr store.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _root_attrs(path=Path("path/to/resource"))
    """
    try:
        import zarr

        group = zarr.open_group(str(path), mode="r")
        return dict(group.attrs.asdict())
    except Exception:
        return {}


def _coordinate_scale(path: Path, level: int = 0) -> list[float] | None:
    """Read the OME-NGFF coordinate scale for one image level when available.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        level (int): Numerical value controlling level. Defaults to ``0``.

    Returns:
        list[float] | None: Collection containing the generated or selected values.

    Example:
        >>> result = _coordinate_scale(path=Path("path/to/resource"))
    """
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
    """Return processing attrs for the supplied inputs.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        dict[str, Any]: Mapping containing the generated or resolved values.

    Example:
        >>> result = _processing_attrs(path=Path("path/to/resource"))
    """
    attrs = _root_attrs(path)
    processing = attrs.get("pft_processing", {})
    return dict(processing) if isinstance(processing, dict) else {}


def _rebase_recorded_path(value: str | Path, project_root: Path) -> Path:
    """Resolve paths stored on another Windows workstation against this project.

    Args:
        value (str | Path): Value to validate, transform, store, or forward.
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _rebase_recorded_path(value="value", project_root=Path("path/to/resource"))
    """
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
    """Return latest directory for the supplied inputs.

    Args:
        candidates (Iterable[Path]): Filesystem path used for candidates.
        preferred_name (str | None): Text value specifying preferred name. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _latest_directory(candidates=Path("path/to/resource"))
    """
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
    """Resolve the intensity-preserving local-threshold output root.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        explicit_root (Path | None): Directory used for explicit. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = resolve_filter_root(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    dataset = normalize_dataset_name(dataset)
    if explicit_root is not None:
        root = resolve_project_path(explicit_root, project_root)
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
    """Resolve a U-Net inference root containing ``pred_mask.ome.zarr`` files.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        explicit_root (Path | None): Directory used for explicit. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = resolve_unet_root(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    dataset = normalize_dataset_name(dataset)
    if explicit_root is not None:
        root = resolve_project_path(explicit_root, project_root)
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
    """Return raw two-dimensional data reference for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        sample_key (str): Canonical relative identifier of a sample within the selected dataset and source mode.

    Returns:
        Path | None: Resolved or generated filesystem path.

    Example:
        >>> result = _raw_2d_reference(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     sample_key="sample_key",
        ... )
    """
    sample = Path(sample_key).name
    candidates = [
        Path(project_root) / "results" / "img" / dataset / sample / "image.ome.zarr",
        Path(project_root) / "results" / "img" / "2d_data" / dataset / sample / "image.ome.zarr",
    ]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    return None


def _resolve_processing_path(
    *,
    processing: dict[str, Any],
    source_zarr: Path,
    project_root: Path,
    absolute_key: str,
    project_relative_key: str,
    local_relative_key: str | None = None,
) -> Path | None:
    """Resolve one dependency recorded in MIP provenance metadata.

    Args:
        processing (dict[str, Any]): Text value specifying processing.
        source_zarr (Path): Filesystem path used for source Zarr.
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        absolute_key (str): Text value specifying absolute key.
        project_relative_key (str): Text value specifying project relative key.
        local_relative_key (str | None): Text value specifying local relative key. ``None`` selects the function's default behavior.

    Returns:
        Path | None: Resolved or generated filesystem path.

    Example:
        >>> result = _resolve_processing_path(
        ...     processing="processing",
        ...     source_zarr=Path("path/to/resource"),
        ...     project_root=Path("path/to/resource"),
        ...     absolute_key="absolute_key",
        ...     project_relative_key="project_relative_key",
        ... )
    """
    if local_relative_key:
        local_value = processing.get(local_relative_key)
        if local_value:
            local_path = (source_zarr.parent / str(local_value)).resolve()
            if local_path.is_dir():
                return local_path

    relative_value = processing.get(project_relative_key)
    if relative_value:
        relative_path = resolve_project_path(str(relative_value), project_root)
        if relative_path.is_dir():
            return relative_path

    absolute_value = processing.get(absolute_key)
    if absolute_value:
        absolute_path = _rebase_recorded_path(absolute_value, project_root)
        if absolute_path.is_dir():
            return absolute_path.resolve()
    return None


def _canonical_mip_dependencies(
    project_root: Path,
    sample_key: str,
    source_mode: str,
    processing: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    """Reconstruct canonical local dependencies for legacy MIP metadata.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        sample_key (str): Canonical relative identifier of a sample within the selected dataset and source mode.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        processing (dict[str, Any]): Text value specifying processing.

    Returns:
        tuple[Path | None, Path | None]: Resolved or generated filesystem path.

    Example:
        >>> result = _canonical_mip_dependencies(
        ...     project_root=Path("path/to/resource"),
        ...     sample_key="sample_key",
        ...     source_mode="original",
        ...     processing="processing",
        ... )
    """
    relative_sample = Path(*Path(sample_key).parts)
    raw_zarr = (
        project_root
        / "results"
        / "img"
        / "3d_data"
        / relative_sample
        / "image.ome.zarr"
    )

    intensity: Path | None
    if source_mode == "raw_masked":
        intensity = raw_zarr if raw_zarr.is_dir() else None
    elif source_mode == "deconv_masked":
        iterations = processing.get("deconvolution_iterations")
        if not isinstance(iterations, dict):
            iterations = {}
        blue = int(iterations.get("blue", 3))
        green = int(iterations.get("green", 3))
        red = int(iterations.get("red", 2))
        model = str(processing.get("deconvolution_model") or "BW")
        source_level = int(processing.get("source_level", 0))
        sample_name = relative_sample.name
        deconv_name = (
            f"{sample_name}__SK_RL__PSF{model}"
            f"__iterB{blue}_G{green}_R{red}__sourceL{source_level}"
        )
        candidate = (
            project_root
            / "results"
            / "deconv"
            / relative_sample.parent
            / deconv_name
            / "image.ome.zarr"
        )
        intensity = candidate if candidate.is_dir() else None
    else:
        intensity = None

    mask_candidate = (
        project_root
        / "results"
        / "U-net"
        / "3d_25d"
        / relative_sample
        / "pred_mask.ome.zarr"
    )
    mask = mask_candidate if mask_candidate.is_dir() else None
    return intensity, mask


def _raw_3d_reference_from_processing(
    source_zarr: Path,
    project_root: Path,
) -> Path | None:
    """Return raw three-dimensional data reference from processing for the supplied inputs.

    Args:
        source_zarr (Path): Filesystem path used for source Zarr.
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.

    Returns:
        Path | None: Resolved or generated filesystem path.

    Example:
        >>> result = _raw_3d_reference_from_processing(
        ...     source_zarr=Path("path/to/resource"),
        ...     project_root=Path("path/to/resource"),
        ... )
    """
    processing = _processing_attrs(source_zarr)
    path = _resolve_processing_path(
        processing=processing,
        source_zarr=source_zarr,
        project_root=project_root,
        absolute_key="raw_source_omezarr",
        project_relative_key="raw_source_project_relative",
    )
    return path.resolve() if path is not None else None


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

    Args:
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        filter_root (Path | None): Directory used for filter. ``None`` selects the function's default behavior.
        unet_root (Path | None): Directory used for U-Net result. ``None`` selects the function's default behavior.
        mip_root (Path | None): Directory used for maximum-intensity projection. ``None`` selects the function's default behavior.

    Returns:
        list[SegmentationSource]: Collection containing the generated or selected values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = discover_segmentation_sources(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
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

    selected_mip_root = (
        resolve_project_path(mip_root, root)
        if mip_root is not None
        else root / "results" / "mip_2d" / MIP_DIRECTORY_BY_MODE[source_mode]
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
            intensity_source = _resolve_processing_path(
                processing=processing,
                source_zarr=image,
                project_root=root,
                absolute_key="projection_source_omezarr",
                project_relative_key="projection_source_project_relative",
                local_relative_key="portable_unmasked_mip_relative",
            ) or image
            mask = _resolve_processing_path(
                processing=processing,
                source_zarr=image,
                project_root=root,
                absolute_key="mask_omezarr",
                project_relative_key="mask_project_relative",
                local_relative_key="portable_mask_relative",
            )

            canonical_intensity, canonical_mask = _canonical_mip_dependencies(
                root, sample_key, source_mode, processing
            )
            if intensity_source == image and canonical_intensity is not None:
                intensity_source = canonical_intensity
            if mask is None and canonical_mask is not None:
                mask = canonical_mask

            # Preserve the legacy recorded path in the error message when no
            # portable or canonical dependency can be found.
            if intensity_source == image and processing.get("projection_source_omezarr"):
                intensity_source = _rebase_recorded_path(
                    processing["projection_source_omezarr"], root
                )
            if mask is None and processing.get("mask_omezarr"):
                mask = _rebase_recorded_path(processing["mask_omezarr"], root)
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


def select_dataset_channels(
    image: Any,
    axes: str,
    dataset: str,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Apply the numerical-channel policy defined for one PFT dataset.

    ``2d_time`` contains HADA in numerical channel 0. Some source OME-Zarr
    stores also contain a second channel, but that channel is not part of the
    2d_time segmentation workflow. Therefore, channel 0 is selected and the C
    axis is removed before normalization and mask application.

    ``2d_wga_dapi`` retains its two numerical channels: DAPI at C=0 and WGA at
    C=1. MIP inputs are not modified by this helper.

    Args:
        image (Any): Input image array to process.
        axes (str): Axis specification describing the dimensional order of the image data.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        tuple[np.ndarray, str, dict[str, Any]]: Mapping containing the generated or resolved values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = select_dataset_channels(
        ...     image=image_array,
        ...     axes="axes",
        ...     dataset="2d_time",
        ... )
    """
    dataset = normalize_dataset_name(dataset)
    current_axes = str(axes).lower()
    array = image
    if len(current_axes) != int(array.ndim):
        raise ValueError(
            f"Axes {current_axes!r} do not match image shape {tuple(array.shape)}"
        )

    metadata: dict[str, Any] = {
        "dataset": dataset,
        "input_axes": current_axes,
        "input_shape": [int(value) for value in array.shape],
        "policy": "retain_all_numerical_channels",
        "selected_channels": None,
        "ignored_channels": [],
    }

    if dataset == "2d_time":
        metadata["policy"] = "use_channel_0_only"
        if "c" not in current_axes:
            metadata["selected_channels"] = [0]
            metadata["input_channel_count"] = 1
            metadata["output_axes"] = current_axes
            metadata["output_shape"] = [int(value) for value in array.shape]
            return np.asarray(array), current_axes, metadata

        channel_axis = current_axes.index("c")
        channel_count = int(array.shape[channel_axis])
        if channel_count < 1:
            raise ValueError("2d_time source contains no numerical channel")

        selection: list[Any] = [slice(None)] * int(array.ndim)
        selection[channel_axis] = 0
        selected = np.asarray(array[tuple(selection)])
        selected_axes = current_axes[:channel_axis] + current_axes[channel_axis + 1 :]
        metadata.update(
            {
                "input_channel_count": channel_count,
                "selected_channels": [0],
                "ignored_channels": list(range(1, channel_count)),
                "output_axes": selected_axes,
                "output_shape": [int(value) for value in selected.shape],
            }
        )
        return selected, selected_axes, metadata

    if dataset == "2d_wga_dapi":
        if "c" not in current_axes:
            raise ValueError(
                "2d_wga_dapi requires an explicit C axis with DAPI at C=0 and WGA at C=1"
            )
        channel_count = int(array.shape[current_axes.index("c")])
        if channel_count != 2:
            raise ValueError(
                f"2d_wga_dapi requires exactly 2 numerical channels, received {channel_count}"
            )
        metadata.update(
            {
                "input_channel_count": channel_count,
                "selected_channels": [0, 1],
                "output_axes": current_axes,
                "output_shape": [int(value) for value in array.shape],
            }
        )
        return np.asarray(array), current_axes, metadata

    metadata["output_axes"] = current_axes
    metadata["output_shape"] = [int(value) for value in array.shape]
    return np.asarray(array), current_axes, metadata


def _coordinate_scale_after_channel_selection(
    scale: Sequence[float] | None,
    input_axes: str,
    output_axes: str,
) -> list[float] | None:
    """Keep coordinate-scale metadata aligned after dropping a C axis.

    Args:
        scale (Sequence[float] | None): Multiplicative scale factor applied by the operation.
        input_axes (str): Text value specifying input axes.
        output_axes (str): Text value specifying output axes.

    Returns:
        list[float] | None: Collection containing the generated or selected values.

    Example:
        >>> result = _coordinate_scale_after_channel_selection(
        ...     scale=0.5,
        ...     input_axes="input_axes",
        ...     output_axes="output_axes",
        ... )
    """
    if scale is None:
        return None
    values = [float(value) for value in scale]
    input_axes = str(input_axes).lower()
    output_axes = str(output_axes).lower()
    if len(values) == len(output_axes):
        return values
    if len(values) == len(input_axes) and "c" in input_axes and "c" not in output_axes:
        channel_axis = input_axes.index("c")
        return values[:channel_axis] + values[channel_axis + 1 :]
    return None

def _normalization_slices(shape: Sequence[int], axes: str) -> Iterable[tuple[Any, ...]]:
    """Yield one YX plane per frame and numerical channel.

    Args:
        shape (Sequence[int]): Target or observed array shape.
        axes (str): Axis specification describing the dimensional order of the image data.

    Returns:
        Iterable[tuple[Any, ...]]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _normalization_slices(shape=1, axes="axes")
    """
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
    """Normalize every complete YX channel plane independently to [0, 1].

    Args:
        image (np.ndarray): Input image array to process.
        axes (str): Axis specification describing the dimensional order of the image data.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        tuple[np.ndarray, list[dict[str, Any]]]: Mapping containing the generated or resolved values.

    Raises:
        TypeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = normalize_complete_image_per_channel(image=image_array, axes="axes")
    """
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
    """Align a binary mask to an image axis order and broadcast over channels.

    Args:
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        mask_axes (str): Text value specifying mask axes.
        target_shape (Sequence[int]): Numerical value controlling target shape.
        target_axes (str): Text value specifying target axes.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _broadcast_mask(
        ...     mask=image_array,
        ...     mask_axes="mask_axes",
        ...     target_shape=1,
        ...     target_axes="target_axes",
        ... )
    """
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
    """Create a validated float32 downstream segmentation input.

    Args:
        intensity_image (np.ndarray): Array containing intensity image.
        intensity_axes (str): Text value specifying intensity axes.
        predicted_mask (np.ndarray | None): Array containing predicted mask. ``None`` selects the function's default behavior.
        mask_axes (str | None): Text value specifying mask axes. ``None`` selects the function's default behavior.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        tuple[np.ndarray, list[dict[str, Any]]]: Mapping containing the generated or resolved values.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = create_segmentation_input(intensity_image=image_array, intensity_axes="intensity_axes")
    """
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
    """Load array from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        level (int): Numerical value controlling level. Defaults to ``0``.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_array(path=Path("path/to/resource"))
    """
    if not Path(path).is_dir():
        raise FileNotFoundError(path)
    array, axes = load_ome_zarr(Path(path), level=level, as_numpy=True)
    return np.asarray(array), str(axes).lower()


def _declared_channel_count(path: Path) -> int | None:
    """Return the channel count declared by a derived OME-Zarr store.

    Legacy deconvolution products may contain a five-dimensional array whose
    three fluorescence channels were stored under the ``t`` axis while ``c``
    remained singleton. The metadata and the standard deconvolution directory
    tag provide independent evidence for repairing that representation.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        int | None: Computed numerical result.

    Example:
        >>> result = _declared_channel_count(path=Path("path/to/resource"))
    """
    candidates: list[int] = []

    try:
        import zarr

        root = zarr.open_group(str(path), mode="r")
        attrs = dict(root.attrs)

        omero = attrs.get("omero")
        if isinstance(omero, dict):
            channels = omero.get("channels")
            if isinstance(channels, list) and channels:
                candidates.append(len(channels))

        processing = attrs.get("pft_processing")
        if isinstance(processing, dict):
            for key in ("iterations_by_channel", "channel_psf_mapping", "channel_display_colors"):
                records = processing.get(key)
                if isinstance(records, list) and records:
                    candidates.append(len(records))
    except Exception:
        # Metadata evidence is optional. The path contract below remains
        # available for legacy stores that cannot be opened directly by zarr.
        pass

    directory_name = path.parent.name
    if "__iterB" in directory_name and "_G" in directory_name and "_R" in directory_name:
        candidates.append(3)

    positive = {int(value) for value in candidates if int(value) > 0}
    if len(positive) == 1:
        return positive.pop()
    return None


def _normalize_legacy_projection_axes(
    array: Any,
    axes: str,
    path: Path,
) -> tuple[Any, str]:
    """Normalize legacy time/channel layouts without losing image data.

    Supported repairs
    -----------------
    ``T=1``
        Remove the redundant singleton time axis.

    ``TCYX`` with shape ``(channels, z, y, x)``
        Some legacy PFT deconvolution stores contain the correct numerical
        CZYX array but expose the axis label TCYX. When the first dimension
        matches the independently declared fluorescence-channel count and the
        second dimension is a plausible Z stack, only the axis names are
        corrected. The numerical array is not transposed.

    singleton ``C`` with ``T=channels``
        Repair legacy five-dimensional layouts such as CZTYX with shape
        ``(1, z, channels, y, x)`` by removing singleton C and moving T to the
        canonical channel position.

    Genuine multi-time-point data remain an error because a temporal selection
    policy is required before a two-dimensional MIP can be defined.

    Args:
        array (Any): Value specifying array for the operation.
        axes (str): Axis specification describing the dimensional order of the image data.
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        tuple[Any, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _normalize_legacy_projection_axes(
        ...     array=image_array,
        ...     axes="axes",
        ...     path=Path("path/to/resource"),
        ... )
    """
    axes = str(axes).lower()
    if "t" not in axes:
        return array, axes

    time_axis = axes.index("t")
    time_count = int(array.shape[time_axis])
    if time_count == 1:
        return select_index_along_axis(array, axes, "t", 0)

    declared_channel_count = _declared_channel_count(path)

    # Legacy four-dimensional deconvolution output:
    #   declared axes: TCYX
    #   actual order:  CZYX
    #   shape example: (3, 40, 2560, 2560)
    # No transpose is required because the numerical order is already CZYX.
    if axes == "tcyx" and int(array.ndim) == 4:
        first_size = int(array.shape[0])
        second_size = int(array.shape[1])
        channel_evidence = (
            declared_channel_count == first_size
            if declared_channel_count is not None
            else 1 <= first_size <= 4
        )
        plausible_z_stack = second_size > max(first_size, 4)
        if channel_evidence and plausible_z_stack:
            return array, "czyx"

    channel_count = int(array.shape[axes.index("c")]) if "c" in axes else 1

    # Legacy five-dimensional output with a singleton C axis and fluorescence
    # channels incorrectly represented by T.
    if channel_count == 1 and declared_channel_count == time_count:
        if "c" in axes:
            array, axes = select_index_along_axis(array, axes, "c", 0)

        time_axis = axes.index("t")
        permutation = [time_axis] + [
            index for index in range(int(array.ndim)) if index != time_axis
        ]
        array = array.transpose(tuple(permutation))
        axes = "c" + "".join(axis for axis in axes if axis != "t")
        return array, axes

    raise ValueError(
        "MIP preparation found a multi-time-point projection source that "
        "cannot be repaired safely. "
        f"axes={axes!r}, shape={tuple(array.shape)}, T={time_count}, "
        f"C={channel_count}, declared_channels={declared_channel_count}, "
        f"path={path}. A genuine time series must be split or supplied with "
        "an explicit time-point selection policy."
    )


def _load_projection_mip(path: Path) -> tuple[np.ndarray, str, str]:
    """Load a portable CYX MIP or create one from a 3D projection source.

    Canonical three-dimensional inputs use ``CZYX``. The loader also supports
    legacy derived stores with a singleton time axis, legacy ``TCYX`` metadata
    over numerically CZYX arrays, and legacy ``CZTYX`` products in which the
    fluorescence channels were incorrectly represented by ``T``.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        tuple[np.ndarray, str, str]: Collection containing the generated or selected values.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_projection_mip(path=Path("path/to/resource"))
    """
    from PFT.core_prog_parts.segmentation.mip_3d_core import (
        maximum_intensity_projection_cyx,
    )

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    # Use direct Zarr access only for the 3D projection source. The standard
    # shared loader remains active for the working 2D preparation workflows.
    array, axes = load_ome_zarr_direct(path, level=0, as_numpy=False)
    axes = str(axes).lower()
    if len(axes) != int(array.ndim):
        raise ValueError(
            "OME-Zarr axes/shape mismatch for projection source: "
            f"axes={axes!r}, shape={tuple(array.shape)}, path={path}"
        )

    array, axes = _normalize_legacy_projection_axes(array, axes, path)
    source_dtype = str(np.dtype(array.dtype))

    if "z" in axes:
        array, axes = ensure_czyx(array, axes)
        mip = maximum_intensity_projection_cyx(array)
        return np.asarray(mip), "cyx", source_dtype

    if axes == "cyx":
        mip = np.asarray(array)
    elif axes == "yx":
        mip = np.asarray(array)[None, ...]
    else:
        raise ValueError(
            "Expected a CYX portable MIP or a CZYX 3D source after legacy-axis "
            f"normalization, received axes={axes!r}, shape={tuple(array.shape)}: {path}"
        )

    if mip.ndim != 3 or mip.shape[-2] < 1 or mip.shape[-1] < 1:
        raise ValueError(f"Invalid portable MIP shape {mip.shape}: {path}")
    if not np.isfinite(mip).all():
        raise ValueError(f"Portable MIP contains non-finite values: {path}")
    return mip, "cyx", source_dtype


def _load_mip_mask(
    path: Path,
    *,
    target_slice_1based: int | None,
    expected_yx: tuple[int, int],
) -> tuple[np.ndarray, str]:
    """Load maximum-intensity projection mask from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        target_slice_1based (int | None): Numerical value controlling target slice 1based.
        expected_yx (tuple[int, int]): Numerical value controlling expected yx.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Raises:
        IndexError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_mip_mask(
        ...     path=Path("path/to/resource"),
        ...     target_slice_1based=1,
        ...     expected_yx=1,
        ... )
    """
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



def _normalize_plane_for_preview(plane: np.ndarray) -> np.ndarray:
    """Convert one 2D numerical plane to uint8 for visual inspection only.

    Args:
        plane (np.ndarray): Array containing plane.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _normalize_plane_for_preview(plane=image_array)
    """
    array = np.asarray(plane, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.8))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


_RGB_DESTINATION_BY_NAME: dict[str, int] = {
    "red": 0,
    "green": 1,
    "blue": 2,
}


def _rgb_destination_from_hex(value: Any) -> int | None:
    """Return the dominant RGB destination encoded by an OME color value.

    Args:
        value (Any): Value to validate, transform, store, or forward.

    Returns:
        int | None: Computed numerical result.

    Example:
        >>> result = _rgb_destination_from_hex(value=...)
    """
    text = str(value or "").strip().lstrip("#")
    if len(text) == 8:
        text = text[:6]
    if len(text) != 6:
        return None
    try:
        red = int(text[0:2], 16)
        green = int(text[2:4], 16)
        blue = int(text[4:6], 16)
    except ValueError:
        return None
    values = (red, green, blue)
    maximum = max(values)
    if maximum <= 0 or values.count(maximum) != 1:
        return None
    return values.index(maximum)


def _preview_rgb_destinations(
    source_zarr: Path | None,
    channel_count: int,
) -> tuple[int, ...] | None:
    """Resolve numerical-channel to RGB-display mapping from OME-Zarr metadata.

    Numerical channels are never reordered. This helper only determines where
    each channel is placed in an RGB preview. The preferred source is the
    ``pft_processing.channel_display_colors`` record written during
    deconvolution. OME ``omero.channels[].color`` metadata is used as a fallback.

    Args:
        source_zarr (Path | None): Filesystem path used for source Zarr.
        channel_count (int): Number of channel used by the operation.

    Returns:
        tuple[int, ...] | None: Collection containing the generated or selected values.

    Example:
        >>> result = _preview_rgb_destinations(source_zarr=Path("path/to/resource"), channel_count=1)
    """
    if source_zarr is None or channel_count <= 0:
        return None

    try:
        import zarr

        root = zarr.open_group(str(Path(source_zarr)), mode="r")
        attrs = dict(root.attrs)
    except Exception:
        return None

    destinations: list[int | None] = [None] * int(channel_count)

    processing = attrs.get("pft_processing")
    if isinstance(processing, dict):
        records = processing.get("channel_display_colors")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                try:
                    channel_index = int(record.get("channel_index"))
                except (TypeError, ValueError):
                    continue
                color_name = str(record.get("display_color") or "").strip().lower()
                destination = _RGB_DESTINATION_BY_NAME.get(color_name)
                if 0 <= channel_index < channel_count and destination is not None:
                    destinations[channel_index] = destination

    if all(item is not None for item in destinations):
        return tuple(int(item) for item in destinations)

    omero = attrs.get("omero")
    channels = omero.get("channels") if isinstance(omero, dict) else None
    if isinstance(channels, list):
        for channel_index, record in enumerate(channels[:channel_count]):
            if destinations[channel_index] is not None or not isinstance(record, dict):
                continue
            destination = _rgb_destination_from_hex(record.get("color"))
            if destination is not None:
                destinations[channel_index] = destination

    if all(item is not None for item in destinations):
        return tuple(int(item) for item in destinations)
    return None


def _render_preview_rgb(
    image: np.ndarray,
    axes: str,
    dataset: str | None = None,
    rgb_destinations: Sequence[int] | None = None,
) -> np.ndarray:
    """Render one numerical image to RGB without changing channel order.

    Args:
        image (np.ndarray): Input image array to process.
        axes (str): Axis specification describing the dimensional order of the image data.
        dataset (str | None): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``. ``None`` selects the function's default behavior.
        rgb_destinations (Sequence[int] | None): Numerical value controlling RGB representation destinations. ``None`` selects the function's default behavior.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _render_preview_rgb(image=image_array, axes="axes")
    """
    array = np.asarray(image)
    axes = str(axes).lower()
    if array.ndim == 2 or axes == "yx":
        gray = _normalize_plane_for_preview(array)
        return np.stack([gray, gray, gray], axis=-1)

    if "c" not in axes:
        raise ValueError(
            f"Preview rendering expected YX or explicit C axis, received {axes!r}"
        )

    c_index = axes.index("c")
    moved = np.moveaxis(array, c_index, -1)
    channels = int(moved.shape[-1])

    if dataset == "2d_wga_dapi" and channels >= 2:
        blue = _normalize_plane_for_preview(moved[..., 0])
        green = _normalize_plane_for_preview(moved[..., 1])
        red = np.zeros_like(blue, dtype=np.uint8)
        return np.stack([red, green, blue], axis=-1)

    if channels == 1:
        gray = _normalize_plane_for_preview(moved[..., 0])
        return np.stack([gray, gray, gray], axis=-1)

    if rgb_destinations is not None:
        if len(rgb_destinations) < channels:
            raise ValueError(
                "RGB preview mapping is shorter than the numerical channel count: "
                f"mapping={tuple(rgb_destinations)}, channels={channels}"
            )
        output = np.zeros((*moved.shape[:2], 3), dtype=np.uint8)
        for channel_index in range(channels):
            destination = int(rgb_destinations[channel_index])
            if destination not in (0, 1, 2):
                raise ValueError(
                    f"Invalid RGB destination {destination} for channel {channel_index}"
                )
            rendered = _normalize_plane_for_preview(moved[..., channel_index])
            output[..., destination] = np.maximum(output[..., destination], rendered)
        return output

    # A missing metadata mapping must not silently imply that C0 is red.
    # Use a neutral grayscale preview until a valid color contract is available.
    gray = np.max(
        np.stack(
            [_normalize_plane_for_preview(moved[..., index]) for index in range(channels)],
            axis=-1,
        ),
        axis=-1,
    )
    return np.stack([gray, gray, gray], axis=-1)


def _save_preview_pngs(
    output_dir: Path,
    *,
    raw_image: np.ndarray,
    raw_axes: str,
    normalized_image: np.ndarray,
    normalized_axes: str,
    dataset: str,
    display_source_zarr: Path | None = None,
) -> tuple[Path, Path, Path, tuple[int, ...] | None]:
    """Save raw, normalized, and comparison previews with metadata colors.

    Args:
        output_dir (Path): Directory where generated resources are written.
        raw_image (np.ndarray): Array containing raw image.
        raw_axes (str): Text value specifying raw axes.
        normalized_image (np.ndarray): Array containing normalized image.
        normalized_axes (str): Text value specifying normalized axes.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        display_source_zarr (Path | None): Filesystem path used for display source Zarr. ``None`` selects the function's default behavior.

    Returns:
        tuple[Path, Path, Path, tuple[int, ...] | None]: Resolved or generated filesystem path.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _save_preview_pngs(
        ...     output_dir=Path("path/to/resource"),
        ...     raw_image=image_array,
        ...     raw_axes="raw_axes",
        ...     normalized_image=image_array,
        ...     normalized_axes="normalized_axes",
        ...     dataset="2d_time",
        ... )
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_array = np.asarray(raw_image)
    raw_axes_lower = str(raw_axes).lower()
    channel_count = (
        int(raw_array.shape[raw_axes_lower.index("c")])
        if "c" in raw_axes_lower
        else 1
    )
    destinations = _preview_rgb_destinations(display_source_zarr, channel_count)

    raw_rgb = _render_preview_rgb(
        raw_image,
        raw_axes,
        dataset=dataset,
        rgb_destinations=destinations,
    )
    normalized_rgb = _render_preview_rgb(
        normalized_image,
        normalized_axes,
        dataset=dataset,
        rgb_destinations=destinations,
    )
    if raw_rgb.shape[:2] != normalized_rgb.shape[:2]:
        raise ValueError(
            "Raw and normalized previews must share YX shape, received "
            f"{raw_rgb.shape[:2]} and {normalized_rgb.shape[:2]}"
        )
    divider = np.full((raw_rgb.shape[0], 16, 3), 255, dtype=np.uint8)
    comparison = np.concatenate([raw_rgb, divider, normalized_rgb], axis=1)

    raw_path = output_dir / "preview_raw.png"
    normalized_path = output_dir / "preview_normalized.png"
    comparison_path = output_dir / "preview_comparison.png"
    Image.fromarray(raw_rgb, mode="RGB").save(raw_path)
    Image.fromarray(normalized_rgb, mode="RGB").save(normalized_path)
    Image.fromarray(comparison, mode="RGB").save(comparison_path)
    return raw_path, normalized_path, comparison_path, destinations


def prepare_one_segmentation_input(
    record: SegmentationSource,
    *,
    overwrite: bool = True,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> PreparedSegmentationInput:
    """Create and save one segmentation input from a resolved source record.

    Args:
        record (SegmentationSource): Value specifying record for the operation.
        overwrite (bool): Whether an existing output may be replaced. Defaults to ``True``.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.

    Returns:
        PreparedSegmentationInput: Result produced by the operation.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = prepare_one_segmentation_input(record=...)
    """
    source_array, source_axes = _load_array(record.source_zarr)
    source_dtype = str(source_array.dtype)
    mask_array: np.ndarray | None = None
    mask_axes: str | None = None
    channel_selection: dict[str, Any] = {
        "dataset": record.dataset,
        "policy": "not_applied",
    }

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
        channel_selection = {
            "dataset": record.dataset,
            "policy": "retain_all_numerical_channels",
            "selected_channels": None,
            "ignored_channels": [],
            "input_axes": intensity_axes,
            "output_axes": intensity_axes,
            "input_shape": [int(value) for value in intensity_array.shape],
            "output_shape": [int(value) for value in intensity_array.shape],
        }
    else:
        intensity_array = source_array
        intensity_axes = source_axes
        intensity_source_dtype = str(intensity_array.dtype)
        original_intensity_axes = intensity_axes
        intensity_array, intensity_axes, channel_selection = select_dataset_channels(
            intensity_array,
            intensity_axes,
            record.dataset,
        )
        coordinate_scale = _coordinate_scale_after_channel_selection(
            _coordinate_scale(record.source_zarr),
            original_intensity_axes,
            intensity_axes,
        )
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
        "source_project_relative": project_relative_path(
            record.source_zarr, find_project_root(Path(__file__).resolve())
        ),
        "intensity_source_omezarr": str(record.intensity_source_zarr),
        "intensity_source_project_relative": project_relative_path(
            record.intensity_source_zarr, find_project_root(Path(__file__).resolve())
        ),
        "predicted_mask_omezarr": str(record.mask_zarr) if record.mask_zarr else None,
        "predicted_mask_project_relative": (
            project_relative_path(
                record.mask_zarr, find_project_root(Path(__file__).resolve())
            )
            if record.mask_zarr
            else None
        ),
        "normalization_order": "normalize_complete_source_then_apply_mask",
        "channel_selection": channel_selection,
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
    (
        preview_raw_png,
        preview_normalized_png,
        preview_comparison_png,
        preview_rgb_destinations,
    ) = _save_preview_pngs(
        Path(output).parent,
        raw_image=intensity_array,
        raw_axes=intensity_axes,
        normalized_image=prepared,
        normalized_axes=intensity_axes,
        dataset=record.dataset,
        display_source_zarr=record.intensity_source_zarr,
    )

    report = {
        "record": asdict(record),
        "processing": processing,
        "normalization_planes": normalization_records,
        "channel_selection": channel_selection,
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
        "preview_raw_png": str(preview_raw_png),
        "preview_normalized_png": str(preview_normalized_png),
        "preview_comparison_png": str(preview_comparison_png),
        "preview_rgb_destinations": (
            list(preview_rgb_destinations)
            if preview_rgb_destinations is not None
            else None
        ),
        "preview_rgb_destination_definition": {
            "0": "red",
            "1": "green",
            "2": "blue",
        },
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
        preview_raw_png=preview_raw_png,
        preview_normalized_png=preview_normalized_png,
        preview_comparison_png=preview_comparison_png,
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
    """Prepare all or selected inputs for one independent dataset.

    Args:
        project_root (Path | None): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        filter_root (Path | None): Directory used for filter. ``None`` selects the function's default behavior.
        unet_root (Path | None): Directory used for U-Net result. ``None`` selects the function's default behavior.
        mip_root (Path | None): Directory used for maximum-intensity projection. ``None`` selects the function's default behavior.
        overwrite (bool): Whether an existing output may be replaced. Defaults to ``True``.
        p_low (float): Numerical value controlling p low. Defaults to ``1.0``.
        p_high (float): Numerical value controlling p high. Defaults to ``99.8``.
        selected_sample_keys (set[str] | None): Optional set of canonical sample identifiers restricting the workflow to selected samples. ``None`` selects the function's default behavior.

    Returns:
        list[PreparedSegmentationInput]: Collection containing the generated or selected values.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = prepare_dataset_segmentation_inputs(
        ...     project_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ... )
    """
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
