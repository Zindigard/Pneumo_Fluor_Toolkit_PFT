"""
Write microscopy arrays and CZI metadata as OME-NGFF OME-Zarr images.
The module determines canonical axis order, preserves level-0 pixel data,
stores physical coordinate scales and complete CZI metadata, and optionally
creates multiscale pyramids for 3D images.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import math
import numpy as np

try:
    from PFT.core_prog_parts.io import CziMeta
except Exception:
    class CziMeta:  # pragma: no cover - fallback for isolated documentation builds
        """Provide a minimal fallback metadata type for isolated documentation builds."""
        pass


_CANONICAL_TYPES = {
    "t": "time",
    "c": "channel",
    "z": "space",
    "y": "space",
    "x": "space",
}


def _json_safe(value: Any) -> Any:
    """Recursively convert metadata values into JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def metadata_as_json_dict(meta: CziMeta) -> dict[str, Any]:
    """Convert the complete ``CziMeta`` dataclass to a JSON-safe dictionary."""
    try:
        return _json_safe(asdict(meta))
    except Exception:
        return {"source_path": str(getattr(meta, "source_path", ""))}


def _axes_dicts(axes: str) -> list[dict[str, str]]:
    """Create OME-NGFF axis descriptors with axis type and micrometre units."""
    output: list[dict[str, str]] = []
    for axis in axes:
        descriptor = {"name": axis, "type": _CANONICAL_TYPES.get(axis, "unknown")}
        if axis in "zyx":
            descriptor["unit"] = "micrometer"
        output.append(descriptor)
    return output


def _scale_vector_from_meta_um(axes: str, meta: CziMeta) -> list[float]:
    """Build an axis-aligned coordinate scale vector from CZI physical sampling metadata."""
    sizes = {
        "x": getattr(meta, "pixel_size_um_x", None),
        "y": getattr(meta, "pixel_size_um_y", None),
        "z": getattr(meta, "pixel_size_um_z", None),
    }
    scale: list[float] = []
    for axis in axes:
        value = sizes.get(axis)
        scale.append(float(value) if value not in (None, 0) else 1.0)
    return scale


def scale_vector_from_meta_um(axes: str, meta: CziMeta) -> list[float]:
    """Expose physical scale-vector construction for validators and downstream readers."""
    return _scale_vector_from_meta_um(axes, meta)


def _axes_from_czi_header(arr: np.ndarray, meta: CziMeta) -> str | None:
    """Recover canonical squeezed axes from the original CZI header when unambiguous."""
    header_axes = getattr(meta, "axes", None)
    header_shape = getattr(meta, "header_shape", None)
    if not isinstance(header_axes, str) or not isinstance(header_shape, (tuple, list)):
        return None
    if len(header_axes) != len(header_shape):
        return None

    kept = [axis.lower() for axis, size in zip(header_axes, header_shape) if int(size) != 1]
    if len(kept) != arr.ndim:
        return None
    if any(axis not in _CANONICAL_TYPES for axis in kept):
        return None
    if len(set(kept)) != len(kept):
        return None
    if "y" not in kept or "x" not in kept:
        return None
    return "".join(kept)


def _infer_axes_for_squeezed(arr: np.ndarray, meta: CziMeta) -> str:
    """Infer supported OME axes for a squeezed array using header and metadata evidence."""
    from_header = _axes_from_czi_header(arr, meta)
    if from_header is not None:
        return from_header

    if arr.ndim == 2:
        return "yx"
    if arr.ndim < 2:
        raise ValueError(f"Invalid microscopy array ndim={arr.ndim}")

    non_spatial = arr.ndim - 2
    channel_names = getattr(meta, "channel_names", None)
    n_channels = len(channel_names) if isinstance(channel_names, list) else None

    channel_axis: int | None = None
    if n_channels:
        matches = [axis for axis in range(non_spatial) if arr.shape[axis] == n_channels]
        if len(matches) == 1:
            channel_axis = matches[0]

    if arr.ndim == 3:
        if channel_axis == 0 or (channel_axis is None and 2 <= arr.shape[0] <= 6):
            return "cyx"
        return "zyx"

    if arr.ndim == 4:
        if channel_axis == 0:
            return "czyx" if getattr(meta, "pixel_size_um_z", None) else "ctyx"
        if channel_axis == 1:
            return "tcyx"
        if channel_axis is None and 2 <= arr.shape[0] <= 6:
            return "czyx"
        return "tzyx"

    if arr.ndim == 5:
        if channel_axis in (0, 1):
            return "ctzyx" if channel_axis == 0 else "tczyx"
        return "tczyx"

    raise ValueError(
        f"Cannot infer OME axes for squeezed CZI array shape={arr.shape}. "
        "Preserve supported CZI axes T/C/Z/Y/X or provide explicit axes."
    )


def prepare_array_and_axes(arr: np.ndarray, meta: CziMeta) -> tuple[np.ndarray, str]:
    """Prepare the exact level-0 array and canonical OME axis order.
    
    The function transposes only when required to obtain ``T, C, Z, Y, X`` order
    and returns the prepared array together with its lowercase axis string.
    """
    prepared = np.asarray(arr)
    axes = _infer_axes_for_squeezed(prepared, meta)

    if len(axes) != prepared.ndim:
        raise ValueError(
            f"Inferred axes {axes!r} do not match array ndim={prepared.ndim}, "
            f"shape={prepared.shape}."
        )

    # OME-NGFF canonical ordering is time, channel, then spatial axes.
    canonical_axes = "".join(axis for axis in "tczyx" if axis in axes)
    if set(canonical_axes) != set(axes) or len(canonical_axes) != len(axes):
        raise ValueError(f"Unsupported or duplicated axes inferred for OME-Zarr: {axes!r}")
    if axes != canonical_axes:
        permutation = [axes.index(axis) for axis in canonical_axes]
        prepared = np.transpose(prepared, permutation)
        axes = canonical_axes

    return prepared, axes


def _scale_vector_for_level_um(
    axes: str,
    base_scale_um: list[float],
    *,
    level: int,
    downscale: int,
) -> list[float]:
    """Calculate the physical scale vector for one pyramid level.
    
    Only X and Y sampling are multiplied by the lateral downsampling factor; Z,
    channel, and time scales remain unchanged.
    """
    factor = downscale**level
    return [
        float(value) * factor if axis in ("y", "x") else float(value)
        for axis, value in zip(axes, base_scale_um)
    ]


def save_ome_zarr(
    out_zarr_dir: str | Path,
    arr: np.ndarray,
    axes: str,
    meta: CziMeta | None = None,
    *,
    overwrite: bool = True,
    chunks: tuple[int, ...] | None = None,
    pyramid_3d: bool = True,
    pyramid_max_layer: int = 2,
    pyramid_downscale: int = 2,
) -> Path:
    """Write an array and optional CZI metadata as an OME-NGFF image.
    
    The function creates level-0 data, optional 3D multiscale levels, coordinate
    transformations, channel and source attributes, and PFT-specific audit fields.
    It returns the generated OME-Zarr directory.
    """
    out_zarr_dir = Path(out_zarr_dir)
    arr = np.asarray(arr)
    axes = axes.lower()

    if len(axes) != arr.ndim:
        raise ValueError(f"axes {axes!r} does not match arr.ndim={arr.ndim}, shape={arr.shape}")
    if pyramid_max_layer < 0:
        raise ValueError("pyramid_max_layer must be zero or greater.")
    if pyramid_downscale < 2:
        raise ValueError("pyramid_downscale must be at least 2.")

    if overwrite and out_zarr_dir.exists():
        import shutil
        shutil.rmtree(out_zarr_dir)
    elif out_zarr_dir.exists():
        raise FileExistsError(f"OME-Zarr already exists: {out_zarr_dir}")

    out_zarr_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        import zarr
        from ome_zarr.io import parse_url
        from ome_zarr.scale import Scaler
        from ome_zarr.writer import write_image
    except Exception as exc:
        raise ImportError(
            "OME-Zarr export requires ome-zarr, zarr and numcodecs. "
            "Install the project dependencies from pyproject.toml."
        ) from exc

    store = parse_url(str(out_zarr_dir), mode="w").store
    root = zarr.group(store=store)

    use_pyramid = bool(pyramid_3d and "z" in axes and pyramid_max_layer >= 1)
    coordinate_transformations = None
    if meta is not None:
        base_scale = _scale_vector_from_meta_um(axes, meta)
        level_count = pyramid_max_layer + 1 if use_pyramid else 1
        coordinate_transformations = [
            [
                {
                    "type": "scale",
                    "scale": _scale_vector_for_level_um(
                        axes,
                        base_scale,
                        level=level,
                        downscale=pyramid_downscale,
                    ),
                }
            ]
            for level in range(level_count)
        ]

    scaler = Scaler(downscale=pyramid_downscale, max_layer=pyramid_max_layer) if use_pyramid else None

    write_image(
        image=arr,
        group=root,
        axes=_axes_dicts(axes),
        coordinate_transformations=coordinate_transformations,
        chunks=chunks,
        scaler=scaler,
    )

    if meta is not None:
        root.attrs["pft_meta"] = metadata_as_json_dict(meta)
        root.attrs["source_path"] = str(getattr(meta, "source_path", ""))
        root.attrs["channel_names"] = _json_safe(getattr(meta, "channel_names", None))

    root.attrs["pft_axes"] = axes
    root.attrs["pft_level0_shape"] = list(arr.shape)
    root.attrs["pft_level0_dtype"] = str(arr.dtype)
    root.attrs["pft_multiscale_enabled"] = use_pyramid
    root.attrs["pft_multiscale_reason"] = "3d_pyramid" if use_pyramid else "single_scale"
    root.attrs["pft_pyramid_max_layer"] = pyramid_max_layer if use_pyramid else 0
    root.attrs["pft_pyramid_downscale"] = pyramid_downscale

    return out_zarr_dir


def save_ome_zarr_next_to_outputs(
    out_dir: str | Path,
    arr: np.ndarray,
    meta: CziMeta,
    *,
    overwrite: bool = True,
    chunks: tuple[int, ...] | None = None,
    pyramid_3d: bool = True,
    pyramid_max_layer: int = 2,
    pyramid_downscale: int = 2,
) -> Path:
    """Prepare an array and write it as ``image.ome.zarr`` inside a sample output directory."""
    out_dir = Path(out_dir)
    prepared, axes = prepare_array_and_axes(arr, meta)
    return save_ome_zarr(
        out_dir / "image.ome.zarr",
        prepared,
        axes,
        meta=meta,
        overwrite=overwrite,
        chunks=chunks,
        pyramid_3d=pyramid_3d,
        pyramid_max_layer=pyramid_max_layer,
        pyramid_downscale=pyramid_downscale,
    )
