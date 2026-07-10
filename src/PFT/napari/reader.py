from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from PFT.core_prog_parts.io import load_czi
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr_next_to_outputs

LayerData = tuple[Any, dict, str]

SUPPORTED_FILE_SUFFIXES = {".czi", ".tif", ".tiff"}
SUPPORTED_ZARR_SUFFIXES = (".ome.zarr", ".zarr")


@dataclass(frozen=True)
class MinimalImageMeta:
    source_path: str
    axes: str | None
    header_shape: tuple[int, ...] | None
    squeezed_shape: tuple[int, ...] | None
    squeezed_ndim: int | None
    dtype: str | None
    intensity_min: float | None
    intensity_max: float | None
    pixel_size_um_x: float | None
    pixel_size_um_y: float | None
    pixel_size_um_z: float | None
    channel_names: list[str] | None
    channel_info: list[dict[str, Any]] | None
    raw_xml: str | None
    raw_xml_available: bool


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "setup.cfg").exists():
            return p
    return Path.cwd()


def sanitize_stem(name: str) -> str:
    bad = r'<>:"/\\|?*'
    return "".join(ch if ch not in bad else "_" for ch in name).strip()


def napari_zarr_dir_for_input(input_path: Path) -> Path:
    here = Path(__file__).resolve()
    root = find_project_root(here)
    base = root / "results" / "Napari_zarr"
    base.mkdir(parents=True, exist_ok=True)
    folder = base / sanitize_stem(input_path.stem)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def is_omezarr_path(path: str | Path) -> bool:
    return Path(path).name.lower().endswith(SUPPORTED_ZARR_SUFFIXES)


def resolve_omezarr_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if p.is_dir() and is_omezarr_path(p):
        return p
    if p.is_dir():
        hits = [h for h in p.rglob("image.ome.zarr") if h.is_dir()]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            shown = "\n".join(str(h) for h in hits[:10])
            raise ValueError(
                "Folder contains more than one image.ome.zarr. Please select one specific folder.\n"
                f"First matches:\n{shown}"
            )
    raise ValueError(f"OME-Zarr folder not found: {p}")


def pct_limits(image: Any, p_low: float = 1.0, p_high: float = 99.8) -> tuple[float, float] | None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    flat = image.reshape(-1)
    max_points = 2_000_000
    if flat.size > max_points:
        flat = flat[:: max(1, flat.size // max_points)]
    lo, hi = np.percentile(flat, [p_low, p_high])
    lo = float(lo)
    hi = float(hi if hi > lo else lo + 1.0)
    return lo, hi


def default_colormap(channel_index: int, channel_name: str | None = None) -> str:
    text = (channel_name or "").lower()
    if any(key in text for key in ("dapi", "hada", "405", "blue")):
        return "blue"
    if any(key in text for key in ("wga", "nada", "488", "green")):
        return "green"
    if any(key in text for key in ("tada", "561", "red")):
        return "red"
    fallback = ["blue", "green", "red", "magenta", "cyan", "yellow"]
    return fallback[channel_index % len(fallback)]


def get_channel_names(meta: Any, n_channels: int | None) -> list[str]:
    names = getattr(meta, "channel_names", None)
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    if n_channels is None:
        return ["raw"]
    return [f"C{idx}" for idx in range(n_channels)]


def detect_channel_axis(arr: Any, meta: Any | None = None) -> int | None:
    shape = tuple(arr.shape)
    ndim = len(shape)
    if ndim < 3:
        return None

    names = getattr(meta, "channel_names", None)
    if isinstance(names, list) and names:
        n_channels = len(names)
        matches = [i for i, size in enumerate(shape) if size == n_channels]
        if matches:
            if ndim == 3:
                return 0 if 0 in matches else 2 if 2 in matches else matches[0]
            if ndim == 4:
                return 0 if 0 in matches else 1 if 1 in matches else matches[0]
            if ndim == 5:
                return 1 if 1 in matches else 0 if 0 in matches else matches[0]

    if ndim == 3:
        if 1 < shape[0] <= 6:
            return 0
        if 1 < shape[-1] <= 6:
            return ndim - 1
        return None
    if ndim == 4:
        if 1 < shape[0] <= 6:
            return 0
        if 1 < shape[1] <= 6:
            return 1
        return None
    if ndim == 5:
        if 1 < shape[1] <= 6:
            return 1
        if 1 < shape[0] <= 6:
            return 0
    return None


def infer_axes(shape: tuple[int, ...], channel_axis: int | None) -> str:
    ndim = len(shape)
    if ndim == 2:
        return "yx"
    if ndim == 3:
        if channel_axis == 0:
            return "cyx"
        if channel_axis == 2:
            return "yxc"
        return "zyx"
    if ndim == 4:
        if channel_axis == 0:
            return "czyx"
        if channel_axis == 1:
            return "zcyx"
        return "tzyx"
    if ndim == 5:
        if channel_axis == 1:
            return "tczyx"
        return "tczyx"
    return "".join(f"d{i}" for i in range(ndim))


def axes_after_removing_channel(axes: str, channel_axis: int | None) -> str:
    if channel_axis is None or channel_axis >= len(axes):
        return axes
    return axes[:channel_axis] + axes[channel_axis + 1 :]


def scale_from_axes(axes: str, meta: Any | None) -> tuple[float, ...] | None:
    if not axes:
        return None
    px_x = getattr(meta, "pixel_size_um_x", None) if meta is not None else None
    px_y = getattr(meta, "pixel_size_um_y", None) if meta is not None else None
    px_z = getattr(meta, "pixel_size_um_z", None) if meta is not None else None
    out = []
    for axis in axes:
        if axis == "x":
            out.append(float(px_x) if px_x else 1.0)
        elif axis == "y":
            out.append(float(px_y) if px_y else 1.0)
        elif axis == "z":
            out.append(float(px_z) if px_z else 1.0)
        else:
            out.append(1.0)
    return tuple(out)


def maybe_as_lazy_zarr(array: Any) -> Any:
    try:
        import dask.array as da
    except Exception:
        return array
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return array
    return da.from_array(array, chunks=chunks, asarray=False)


def read_tiff(path: Path) -> tuple[np.ndarray, MinimalImageMeta]:
    import tifffile as tiff

    arr = np.asarray(tiff.imread(str(path)))
    channel_axis = detect_channel_axis(arr)
    axes = infer_axes(tuple(arr.shape), channel_axis)
    meta = MinimalImageMeta(
        source_path=str(path),
        axes=axes,
        header_shape=tuple(arr.shape),
        squeezed_shape=tuple(arr.shape),
        squeezed_ndim=int(arr.ndim),
        dtype=str(arr.dtype),
        intensity_min=float(arr.min()) if arr.size else None,
        intensity_max=float(arr.max()) if arr.size else None,
        pixel_size_um_x=None,
        pixel_size_um_y=None,
        pixel_size_um_z=None,
        channel_names=None,
        channel_info=None,
        raw_xml=None,
        raw_xml_available=False,
    )
    return arr, meta


def save_array_as_omezarr(path: Path, arr: np.ndarray, meta: Any) -> Path | None:
    out_dir = napari_zarr_dir_for_input(path)
    try:
        return save_ome_zarr_next_to_outputs(out_dir=out_dir, arr=arr, meta=meta, overwrite=True)
    except Exception as exc:
        print(f"[PFT napari reader] OME-Zarr export failed: {type(exc).__name__}: {exc}")
        return None


def open_omezarr_array(path: Path) -> tuple[Any, dict[str, Any]]:
    import zarr

    zarr_path = resolve_omezarr_path(path)
    root = zarr.open_group(str(zarr_path), mode="r")
    data_path = None
    multiscales = root.attrs.get("multiscales", None)
    if isinstance(multiscales, list) and multiscales:
        datasets = multiscales[0].get("datasets", [])
        if datasets:
            data_path = datasets[0].get("path", None)
    if data_path is None:
        data_path = "0"
    if data_path not in root:
        raise KeyError(f"Zarr array '{data_path}' not found in {zarr_path}")
    arr = maybe_as_lazy_zarr(root[data_path])
    attrs = dict(root.attrs)
    attrs["zarr_path"] = str(zarr_path)
    attrs["zarr_data_path"] = str(data_path)
    return arr, attrs


def make_layers_from_array(
    arr: Any,
    *,
    source_path: Path,
    meta: Any | None,
    source_format: str,
    ome_zarr_path: Path | None = None,
    zarr_attrs: dict[str, Any] | None = None,
) -> list[LayerData]:
    channel_axis = detect_channel_axis(arr, meta)
    axes = infer_axes(tuple(arr.shape), channel_axis)
    n_channels = int(arr.shape[channel_axis]) if channel_axis is not None else None
    channel_names = get_channel_names(meta, n_channels)

    common_metadata = {
        "pft_source_path": str(source_path),
        "pft_source_format": source_format,
        "pft_ome_zarr_path": str(ome_zarr_path) if ome_zarr_path is not None else None,
        "pft_axes": axes,
        "pft_channel_axis": channel_axis,
        "pft_channel_count": n_channels,
        "pft_original_shape": tuple(arr.shape),
        "pft_meta": meta,
        "pft_zarr_attrs": zarr_attrs,
        "pft_variant": "raw",
        "pft_is_normalized": False,
    }

    layers: list[LayerData] = []
    if channel_axis is None:
        layer_axes = axes
        scale = scale_from_axes(layer_axes, meta)
        kwargs: dict[str, Any] = {
            "name": f"PFT | raw | {source_path.stem}",
            "metadata": {**common_metadata, "pft_layer_axes": layer_axes, "pft_channel_name": source_path.stem},
        }
        if scale is not None and len(scale) == len(arr.shape):
            kwargs["scale"] = scale
        limits = pct_limits(arr)
        if limits is not None:
            kwargs["contrast_limits"] = limits
        layers.append((arr, kwargs, "image"))
        return layers

    layer_axes = axes_after_removing_channel(axes, channel_axis)
    scale = scale_from_axes(layer_axes, meta)
    for c in range(n_channels or 0):
        slicer = [slice(None)] * len(arr.shape)
        slicer[channel_axis] = c
        img = arr[tuple(slicer)]
        channel_name = channel_names[c] if c < len(channel_names) else f"C{c}"
        kwargs = {
            "name": f"PFT | raw | {channel_name}",
            "metadata": {
                **common_metadata,
                "pft_channel_index": c,
                "pft_channel_name": channel_name,
                "pft_layer_axes": layer_axes,
            },
            "colormap": default_colormap(c, channel_name),
            "blending": "additive",
        }
        if scale is not None and len(scale) == len(img.shape):
            kwargs["scale"] = scale
        limits = pct_limits(img)
        if limits is not None:
            kwargs["contrast_limits"] = limits
        layers.append((img, kwargs, "image"))
    return layers


def napari_get_reader(path: str | list[str]) -> Callable[[str], list[LayerData]] | None:
    if isinstance(path, list):
        if len(path) != 1:
            return None
        path = path[0]
    p = Path(path)
    lower_name = p.name.lower()
    if p.is_dir() and lower_name.endswith(SUPPORTED_ZARR_SUFFIXES):
        return read_image
    if p.is_dir() and (p / "image.ome.zarr").exists():
        return read_image
    if p.suffix.lower() in SUPPORTED_FILE_SUFFIXES:
        return read_image
    return None


def read_image(path: str) -> list[LayerData]:
    p = Path(path)
    lower_name = p.name.lower()
    if p.is_dir() and lower_name.endswith(SUPPORTED_ZARR_SUFFIXES):
        arr, attrs = open_omezarr_array(p)
        return make_layers_from_array(arr, source_path=p, meta=None, source_format="OME-Zarr", ome_zarr_path=resolve_omezarr_path(p), zarr_attrs=attrs)
    if p.is_dir() and (p / "image.ome.zarr").exists():
        zarr_path = resolve_omezarr_path(p)
        arr, attrs = open_omezarr_array(zarr_path)
        return make_layers_from_array(arr, source_path=p, meta=None, source_format="OME-Zarr parent folder", ome_zarr_path=zarr_path, zarr_attrs=attrs)

    suffix = p.suffix.lower()
    if suffix == ".czi":
        arr, meta = load_czi(p)
        arr = np.asarray(arr)
        zarr_path = save_array_as_omezarr(p, arr, meta)
        return make_layers_from_array(arr, source_path=p, meta=meta, source_format="CZI", ome_zarr_path=zarr_path)
    if suffix in {".tif", ".tiff"}:
        arr, meta = read_tiff(p)
        zarr_path = save_array_as_omezarr(p, arr, meta)
        return make_layers_from_array(arr, source_path=p, meta=meta, source_format="TIFF", ome_zarr_path=zarr_path)
    raise ValueError(f"Unsupported file format for PFT reader: {p}")
