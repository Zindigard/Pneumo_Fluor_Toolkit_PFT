"""
Shared definitions and metadata utilities for the two-dimensional N2V workflow.

The PFT project trains and applies Noise2Void only to the two-dimensional
microscopy datasets.  This module defines the authoritative model registry,
locates original OME-Zarr inputs below ``results/img``, reads source metadata,
and creates reproducible provenance records used by training, inference, and
validation scripts.

No function in this module performs intensity normalization.  Original level-0
OME-Zarr values are converted to ``float32`` only when they are supplied to a
Noise2Void model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from PFT.core_prog_parts.common_paths import find_project_root


@dataclass(frozen=True)
class N2VModelSpec:
    """Describe one supported N2V model and its source-channel interpretation.

    Attributes
    ----------
    key:
        Short command-line key used by training and inference scripts.
    dataset:
        Canonical dataset directory below ``results/img``.
    model_name:
        Folder name below ``models``.  This name is shared by training, model
        loading, and inference.
    variant:
        Output directory name used below ``results/N2V/<dataset>/<sample>``.
    channels:
        Original source-channel indices supplied to the model.
    channel_labels:
        Human-readable labels corresponding to ``channels``.
    joint:
        ``True`` when channels are supplied to one multi-channel model.
    description:
        Concise description displayed in terminal selection menus.
    """

    key: str
    dataset: str
    model_name: str
    variant: str
    channels: tuple[int, ...]
    channel_labels: tuple[str, ...]
    joint: bool
    description: str


MODEL_SPECS: dict[str, N2VModelSpec] = {
    "time": N2VModelSpec(
        key="time",
        dataset="2d_time",
        model_name="n2v_2d_time_raw",
        variant="time_ch0",
        channels=(0,),
        channel_labels=("HADA",),
        joint=False,
        description="2D time-lapse HADA, source channel 0",
    ),
    "dapi": N2VModelSpec(
        key="dapi",
        dataset="2d_wga_dapi",
        model_name="n2v_2d_wga_dapi_DAPI_blue_raw",
        variant="blue_only",
        channels=(0,),
        channel_labels=("DAPI",),
        joint=False,
        description="2D WGA-DAPI, DAPI/blue source channel 0",
    ),
    "wga": N2VModelSpec(
        key="wga",
        dataset="2d_wga_dapi",
        model_name="n2v_2d_wga_dapi_WGA_green_raw",
        variant="green_only",
        channels=(1,),
        channel_labels=("WGA",),
        joint=False,
        description="2D WGA-DAPI, WGA/green source channel 1",
    ),
    "joint": N2VModelSpec(
        key="joint",
        dataset="2d_wga_dapi",
        model_name="n2v_2d_wga_dapi_JOINT_DAPI+WGA_raw",
        variant="joint",
        channels=(0, 1),
        channel_labels=("DAPI", "WGA"),
        joint=True,
        description="2D WGA-DAPI, joint two-channel DAPI+WGA model",
    ),
}


def utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def get_model_spec(model_key: str) -> N2VModelSpec:
    """Return a model specification and reject unsupported model keys."""

    key = str(model_key).strip().lower()
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unknown N2V model key {model_key!r}. Choose one of: {choices}.") from exc


def model_key_from_dataset_and_loader_key(dataset: str, loader_key: str) -> str:
    """Map legacy toolkit loader keys to the authoritative model registry."""

    normalized = (str(dataset).strip().lower(), str(loader_key).strip())
    mapping = {
        ("2d_time", "blue"): "time",
        ("2d_time", "time_ch0"): "time",
        ("2d_wga_dapi", "blue_DAPI"): "dapi",
        ("2d_wga_dapi", "DAPI_blue"): "dapi",
        ("2d_wga_dapi", "green_WGA"): "wga",
        ("2d_wga_dapi", "WGA_green"): "wga",
        ("2d_wga_dapi", "JOINT_DAPI+WGA_raw"): "joint",
        ("2d_wga_dapi", "joint"): "joint",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise KeyError(f"Unsupported N2V dataset/key pair: {normalized}") from exc


def project_root(start: Path | None = None) -> Path:
    """Return the PFT repository root independently of the working directory."""

    return find_project_root(start or Path(__file__))


def original_dataset_root(spec: N2VModelSpec, root: Path | None = None) -> Path:
    """Return ``results/img/<dataset>`` for a model specification."""

    repo = root or project_root()
    return repo / "results" / "img" / spec.dataset


def models_root(root: Path | None = None) -> Path:
    """Return the repository model directory."""

    return (root or project_root()) / "models"


def n2v_results_root(root: Path | None = None) -> Path:
    """Return the root directory for N2V outputs and reports."""

    return (root or project_root()) / "results" / "N2V"


def is_ome_zarr_store(path: Path) -> bool:
    """Return ``True`` when a directory appears to be an OME-Zarr store."""

    return path.is_dir() and (
        path.name.lower().endswith(".ome.zarr")
        or (path / ".zgroup").exists()
        or (path / ".zattrs").exists()
        or (path / "zarr.json").exists()
    )


def discover_original_samples(spec: N2VModelSpec, root: Path | None = None) -> list[Path]:
    """Discover original sample OME-Zarr stores below ``results/img``.

    Only files named ``image.ome.zarr`` are accepted.  This intentionally
    excludes normalized TIFFs, RGB exports, previews, masks, and previous N2V
    outputs from the training and inference input set.
    """

    dataset_root = original_dataset_root(spec, root)
    if not dataset_root.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in dataset_root.rglob("image.ome.zarr")
        if is_ome_zarr_store(path)
    )


def sample_name_from_zarr(path: Path) -> str:
    """Return the biological sample directory name for an OME-Zarr path."""

    resolved = Path(path)
    if resolved.name.lower().endswith(".ome.zarr"):
        return resolved.parent.name
    return resolved.name


def choose_model_key_interactive(prompt: str = "Choose the N2V model") -> str:
    """Ask the user to select one supported model from a numbered terminal menu."""

    keys = list(MODEL_SPECS)
    print(prompt + ":")
    for index, key in enumerate(keys, start=1):
        spec = MODEL_SPECS[key]
        print(f"  {index}) {key:5s} | {spec.description}")
    while True:
        value = input("Enter model number or key: ").strip().lower()
        if value in MODEL_SPECS:
            return value
        try:
            index = int(value)
        except ValueError:
            index = -1
        if 1 <= index <= len(keys):
            return keys[index - 1]
        print("Invalid model selection.")


def _json_safe(value: Any) -> Any:
    """Convert nested metadata values to stable JSON-compatible objects."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    """Serialize metadata deterministically for hashing and equality checks."""

    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def metadata_sha256(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON metadata."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def root_attributes(zarr_path: Path) -> dict[str, Any]:
    """Read all root attributes from an OME-Zarr store as a plain dictionary."""

    try:
        import zarr
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise ImportError("Reading N2V OME-Zarr metadata requires zarr.") from exc

    group = zarr.open_group(str(Path(zarr_path)), mode="r")
    return _json_safe(dict(group.attrs))


def axes_from_attributes(attrs: Mapping[str, Any], ndim: int) -> str:
    """Extract the level-0 axis string from OME-NGFF metadata."""

    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales and isinstance(multiscales[0], Mapping):
        axes_value = multiscales[0].get("axes")
        if isinstance(axes_value, list):
            names: list[str] = []
            for item in axes_value:
                name = item.get("name") if isinstance(item, Mapping) else item
                names.append(str(name).lower())
            axes = "".join(names)
            if len(axes) == ndim:
                return axes
    pft_axes = attrs.get("pft_axes")
    if isinstance(pft_axes, str) and len(pft_axes) == ndim:
        return pft_axes.lower()
    fallback = {2: "yx", 3: "cyx", 4: "tcyx", 5: "tczyx"}
    return fallback.get(ndim, "".join(f"u{i}" for i in range(ndim)))


def level_zero_path(attrs: Mapping[str, Any]) -> str:
    """Return the level-0 array path declared by OME-NGFF metadata."""

    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales and isinstance(multiscales[0], Mapping):
        datasets = multiscales[0].get("datasets")
        if isinstance(datasets, list) and datasets and isinstance(datasets[0], Mapping):
            path = datasets[0].get("path")
            if path is not None:
                return str(path)
    return "0"


def source_array_properties(zarr_path: Path) -> dict[str, Any]:
    """Read level-0 shape, dtype, chunks, axes, attributes, and axis scales."""

    try:
        import zarr
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise ImportError("Reading N2V OME-Zarr inputs requires zarr.") from exc

    path = Path(zarr_path).resolve()
    group = zarr.open_group(str(path), mode="r")
    attrs = _json_safe(dict(group.attrs))
    array_path = level_zero_path(attrs)
    array = group[array_path]
    axes = axes_from_attributes(attrs, array.ndim)
    scale_by_axis = coordinate_scale_by_axis(attrs, axes)
    return {
        "zarr_path": str(path),
        "sample": sample_name_from_zarr(path),
        "array_path": array_path,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "chunks": [int(value) for value in array.chunks] if array.chunks else None,
        "axes": axes,
        "scale_by_axis": scale_by_axis,
        "channel_names": _json_safe(attrs.get("channel_names")),
        "attrs": attrs,
        "attrs_sha256": metadata_sha256(attrs),
    }


def coordinate_scale_by_axis(attrs: Mapping[str, Any], axes: str) -> dict[str, float]:
    """Return level-0 coordinate scales keyed by axis name.

    Missing non-spatial scales default to one.  Missing spatial scales are also
    represented as one so that the generated OME-Zarr remains valid, while the
    output checker reports any mismatch with the original metadata.
    """

    scale_by_axis = {axis: 1.0 for axis in axes}
    multiscales = attrs.get("multiscales")
    if not (isinstance(multiscales, list) and multiscales and isinstance(multiscales[0], Mapping)):
        return scale_by_axis
    datasets = multiscales[0].get("datasets")
    if not (isinstance(datasets, list) and datasets and isinstance(datasets[0], Mapping)):
        return scale_by_axis
    transformations = datasets[0].get("coordinateTransformations")
    if not isinstance(transformations, list):
        return scale_by_axis
    for transform in transformations:
        if not isinstance(transform, Mapping) or transform.get("type") != "scale":
            continue
        values = transform.get("scale")
        if isinstance(values, list) and len(values) == len(axes):
            for axis, value in zip(axes, values):
                try:
                    scale_by_axis[axis] = float(value)
                except (TypeError, ValueError):
                    scale_by_axis[axis] = 1.0
            break
    return scale_by_axis


def selected_channel_names(source_properties: Mapping[str, Any], channels: Sequence[int]) -> list[str]:
    """Return selected source channel labels with deterministic fallbacks."""

    names = source_properties.get("channel_names")
    if isinstance(names, list):
        selected: list[str] = []
        for channel in channels:
            if 0 <= int(channel) < len(names) and str(names[int(channel)]).strip():
                selected.append(str(names[int(channel)]))
            else:
                selected.append(f"channel_{int(channel)}")
        return selected
    return [f"channel_{int(channel)}" for channel in channels]


def source_scale_for_output_axes(
    source_properties: Mapping[str, Any], output_axes: str
) -> list[float]:
    """Map original coordinate scales to an output axis string."""

    scale_by_axis = source_properties.get("scale_by_axis")
    if not isinstance(scale_by_axis, Mapping):
        scale_by_axis = {}
    return [float(scale_by_axis.get(axis, 1.0)) for axis in output_axes]


def output_provenance_attributes(
    *,
    spec: N2VModelSpec,
    source_properties: Mapping[str, Any],
    source_frame_index: int,
    output_axes: str,
    output_shape: Sequence[int],
    output_dtype: str,
) -> dict[str, Any]:
    """Create complete metadata attributes for one N2V prediction output.

    The output stores both a direct copy of source attributes and a hash of that
    copy.  It also records the selected frame and channels, model identity,
    input/output normalization state, and prediction data properties.
    """

    source_attrs = _json_safe(source_properties.get("attrs", {}))
    source_path = str(source_properties.get("zarr_path", ""))
    source_pft_meta = source_attrs.get("pft_meta") if isinstance(source_attrs, Mapping) else None
    source_czi_path = source_attrs.get("source_path") if isinstance(source_attrs, Mapping) else None

    attrs: dict[str, Any] = {
        "pft_pipeline_stage": "n2v_denoising",
        "pft_processing": {
            "method": "Noise2Void",
            "model_key": spec.key,
            "model_name": spec.model_name,
            "dataset": spec.dataset,
            "variant": spec.variant,
            "generated_utc": utc_now_iso(),
            "input_normalization": "none",
            "output_normalization": "none",
            "prediction_dtype": str(output_dtype),
        },
        "pft_n2v_dataset": spec.dataset,
        "pft_n2v_sample": str(source_properties.get("sample", "")),
        "pft_n2v_variant": spec.variant,
        "pft_n2v_model_key": spec.key,
        "pft_n2v_model_name": spec.model_name,
        "pft_n2v_source_zarr": source_path,
        "pft_n2v_source_frame_index": int(source_frame_index),
        "pft_n2v_source_channels": [int(value) for value in spec.channels],
        "pft_n2v_source_channel_names": selected_channel_names(source_properties, spec.channels),
        "pft_n2v_source_axes": str(source_properties.get("axes", "")),
        "pft_n2v_source_shape": _json_safe(source_properties.get("shape")),
        "pft_n2v_source_dtype": str(source_properties.get("dtype", "")),
        "pft_n2v_source_chunks": _json_safe(source_properties.get("chunks")),
        "pft_n2v_source_attrs": source_attrs,
        "pft_n2v_source_attrs_sha256": metadata_sha256(source_attrs),
        "pft_n2v_input_normalization": "none",
        "pft_n2v_output_normalization": "none",
        "pft_n2v_output_is_display_normalized": False,
        "pft_n2v_output_axes": output_axes,
        "pft_n2v_output_shape": [int(value) for value in output_shape],
        "pft_n2v_output_dtype": str(output_dtype),
        "pft_n2v_generated_utc": utc_now_iso(),
    }

    if source_pft_meta is not None:
        attrs["pft_meta"] = _json_safe(source_pft_meta)
    if source_czi_path is not None:
        attrs["source_path"] = str(source_czi_path)
    attrs["channel_names"] = selected_channel_names(source_properties, spec.channels)
    return attrs


def model_spec_as_dict(spec: N2VModelSpec) -> dict[str, Any]:
    """Return a JSON-compatible model specification dictionary."""

    return _json_safe(asdict(spec))


def relative_or_absolute(path: Path, root: Path) -> str:
    """Represent a path relative to the repository when possible."""

    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def ensure_unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return resolved paths without duplicates while preserving sorted order."""

    return sorted({Path(path).resolve() for path in paths}, key=lambda item: str(item).lower())
