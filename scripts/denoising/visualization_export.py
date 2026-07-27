"""
Unified visualization and image-export command for the PFT project.

The command reads OME-Zarr images produced by the PFT pipeline and can:

1. export selected two-dimensional planes or complete Z-series as normalized
   TIFF and PNG files;
2. create RGB composites and per-channel exports;
3. create collages from preview images;
4. compare a raw OME-Zarr image with a Noise2Void output using a three-panel
   figure containing RAW, N2V and an absolute-difference map displayed in red;
5. compare a raw image with any processed or deconvolved OME-Zarr image;
6. create panels across multiple deconvolution iterations; and
7. select or specify a crop and export the corresponding raw and normalized
   image region.

When ``--task`` is omitted, an interactive menu asks which output should be
created. Every task is also available through command-line arguments.

Difference images are computed from the numeric source
arrays before display normalization. A CSV manifest is written for every run so
that generated figures can be traced to their source files and parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tifffile
from PIL import Image, ImageOps


SCRIPT_PATH = Path(__file__).resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``scripts`` and ``src/PFT``.

    Parameters
    ----------
    start:
        File or directory from which the upward search should begin. The
        physical location of this script is used by default.

    Returns
    -------
    pathlib.Path
        Absolute repository-root path.

    Raises
    ------
    RuntimeError
        If the expected PFT repository layout cannot be found.
    """
    current = (start or SCRIPT_PATH).resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected 'scripts' and "
        "'src/PFT' in the same directory."
    )


PROJECT_ROOT = find_project_root()
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_IMAGE_ROOT = RESULTS_ROOT / "img"
DEFAULT_N2V_ROOT = RESULTS_ROOT / "N2V"
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "visualization_export"


@dataclass(frozen=True)
class OmeImage:
    """Read-only description of one OME-Zarr image level.

    Attributes
    ----------
    path:
        OME-Zarr directory.
    array:
        Zarr array for the selected multiscale level.
    axes:
        Axis string corresponding to ``array``.
    scale_by_axis:
        Physical scale values read from OME-NGFF coordinate transforms.
    channel_names:
        Channel labels stored by the PFT writer, when available.
    attrs:
        Root OME-Zarr attributes converted to a normal dictionary.
    level:
        Selected multiscale level index.
    level_path:
        Dataset path associated with the selected level.
    """

    path: Path
    array: Any
    axes: str
    scale_by_axis: dict[str, float]
    channel_names: list[str]
    attrs: dict[str, Any]
    level: int
    level_path: str


@dataclass(frozen=True)
class PlaneData:
    """Selected image plane represented in ``C, Y, X`` axis order."""

    cyx: np.ndarray
    channels: tuple[int, ...]
    time_index: int
    z_index: int


@dataclass
class ManifestRecord:
    """One generated output recorded in ``visualization_export_manifest.csv``."""

    task: str
    source_path: str
    comparison_path: str
    output_path: str
    output_kind: str
    level: int
    time_index: int
    z_index: int
    channels: str
    source_shape: str
    comparison_shape: str
    difference_mean: float | None = None
    difference_p99: float | None = None
    difference_max: float | None = None
    notes: str = ""


def require_zarr():
    """Import and return :mod:`zarr` with a project-specific error message."""
    try:
        import zarr
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "OME-Zarr visualization requires zarr. Activate the PFT environment "
            "and run 'python -m pip install -e .'."
        ) from exc
    return zarr


def is_ome_zarr_store(path: Path) -> bool:
    """Return ``True`` when *path* appears to be an OME-Zarr directory."""
    if not path.is_dir():
        return False
    if path.name.lower().endswith(".ome.zarr"):
        return True
    return any((path / marker).exists() for marker in (".zgroup", ".zattrs", "zarr.json"))


def discover_ome_zarr(root: Path) -> list[Path]:
    """Return unique OME-Zarr stores found recursively below *root*.

    Nested internal Zarr groups are excluded by retaining only top-level stores.
    """
    root = root.expanduser().resolve()
    if is_ome_zarr_store(root):
        return [root]
    if not root.exists():
        return []
    candidates = sorted(
        (path for path in root.rglob("*") if is_ome_zarr_store(path)),
        key=lambda path: (len(path.parts), str(path).lower()),
    )
    selected: list[Path] = []
    for candidate in candidates:
        if not any(parent in candidate.parents for parent in selected):
            selected.append(candidate)
    return selected


def _axes_string(value: Any, ndim: int) -> str:
    """Convert OME-NGFF axis metadata to a lowercase axis string."""
    if isinstance(value, str) and len(value) == ndim:
        return value.lower()
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name", ""))
            else:
                name = str(item)
            names.append(name.lower())
        joined = "".join(names)
        if len(joined) == ndim:
            return joined
    fallback = {2: "yx", 3: "cyx", 4: "czyx", 5: "tczyx"}
    return fallback.get(ndim, "".join(f"u{i}" for i in range(ndim)))


def _multiscales_info(attrs: dict[str, Any], level: int, ndim: int) -> tuple[str, str, dict[str, float]]:
    """Extract axes, array path and physical scale for one multiscale level."""
    multiscales = attrs.get("multiscales")
    axes = ""
    level_path = str(level)
    scale_by_axis: dict[str, float] = {}

    if isinstance(multiscales, list) and multiscales and isinstance(multiscales[0], dict):
        entry = multiscales[0]
        axes = _axes_string(entry.get("axes"), ndim)
        datasets = entry.get("datasets")
        if isinstance(datasets, list) and datasets:
            if level < 0 or level >= len(datasets):
                raise IndexError(
                    f"Requested pyramid level {level}, but the image contains {len(datasets)} levels."
                )
            dataset = datasets[level]
            if isinstance(dataset, dict):
                level_path = str(dataset.get("path", level))
                transformations = dataset.get("coordinateTransformations", [])
                if isinstance(transformations, list):
                    for transform in transformations:
                        if not isinstance(transform, dict) or transform.get("type") != "scale":
                            continue
                        scale = transform.get("scale")
                        if isinstance(scale, list) and len(scale) == len(axes):
                            for axis, value in zip(axes, scale):
                                try:
                                    scale_by_axis[axis] = float(value)
                                except (TypeError, ValueError):
                                    pass
                            break
    return axes, level_path, scale_by_axis


def open_ome_image(path: Path, level: int = 0) -> OmeImage:
    """Open one OME-Zarr level in read-only mode.

    Parameters
    ----------
    path:
        OME-Zarr directory.
    level:
        Zero-based multiscale level.

    Returns
    -------
    OmeImage
        Array, axes, channel labels, scale metadata and root attributes.
    """
    zarr = require_zarr()
    path = path.expanduser().resolve()
    if not is_ome_zarr_store(path):
        raise FileNotFoundError(f"OME-Zarr directory not found: {path}")

    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)

    provisional_path = str(level)
    provisional = root[provisional_path] if provisional_path in root else None
    provisional_ndim = int(getattr(provisional, "ndim", 0)) if provisional is not None else 0
    axes, level_path, scale_by_axis = _multiscales_info(attrs, level, provisional_ndim)
    if level_path not in root:
        if str(level) in root:
            level_path = str(level)
        else:
            raise KeyError(f"OME-Zarr level path '{level_path}' does not exist in {path}")
    array = root[level_path]
    if not axes or len(axes) != array.ndim:
        axes = _axes_string(attrs.get("pft_axes"), array.ndim)

    channel_names = attrs.get("channel_names")
    if not isinstance(channel_names, list):
        pft_meta = attrs.get("pft_meta")
        channel_names = pft_meta.get("channel_names") if isinstance(pft_meta, dict) else None
    if not isinstance(channel_names, list):
        channel_count = array.shape[axes.index("c")] if "c" in axes else 1
        channel_names = [f"channel_{index}" for index in range(channel_count)]
    channel_names = [str(name) for name in channel_names]

    return OmeImage(
        path=path,
        array=array,
        axes=axes,
        scale_by_axis=scale_by_axis,
        channel_names=channel_names,
        attrs=attrs,
        level=level,
        level_path=level_path,
    )


def parse_channels(text: str | None, channel_count: int) -> tuple[int, ...]:
    """Parse ``all`` or a comma-separated channel list into validated indices."""
    if text is None or text.strip().lower() in {"", "all", "*"}:
        return tuple(range(channel_count))
    result: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value >= channel_count:
            raise IndexError(f"Channel {value} is outside the valid range 0..{channel_count - 1}.")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("At least one channel must be selected.")
    return tuple(result)


def axis_size(image: OmeImage, axis: str) -> int:
    """Return the size of *axis*, or one when the axis is absent."""
    return int(image.array.shape[image.axes.index(axis)]) if axis in image.axes else 1


def select_plane_cyx(
    image: OmeImage,
    *,
    time_index: int = 0,
    z_index: int = 0,
    channels: Sequence[int] | None = None,
) -> PlaneData:
    """Read one plane and return it in ``C, Y, X`` order.

    All unsupported non-spatial axes are fixed at index zero. Missing channel
    axes are inserted as a singleton dimension.
    """
    axes = image.axes
    if "y" not in axes or "x" not in axes:
        raise ValueError(f"OME-Zarr axes must contain Y and X; stored axes are '{axes}'.")

    channel_count = axis_size(image, "c")
    selected_channels = tuple(channels) if channels is not None else tuple(range(channel_count))
    if any(index < 0 or index >= channel_count for index in selected_channels):
        raise IndexError(f"Selected channels {selected_channels} are invalid for {channel_count} channels.")

    if time_index < 0 or time_index >= axis_size(image, "t"):
        raise IndexError(f"Time index {time_index} is outside the image range.")
    if z_index < 0 or z_index >= axis_size(image, "z"):
        raise IndexError(f"Z index {z_index} is outside the image range.")

    selector: list[Any] = []
    retained_axes: list[str] = []
    for axis in axes:
        if axis == "t":
            selector.append(time_index)
        elif axis == "z":
            selector.append(z_index)
        elif axis == "c":
            selector.append(list(selected_channels))
            retained_axes.append("c")
        elif axis in {"y", "x"}:
            selector.append(slice(None))
            retained_axes.append(axis)
        else:
            selector.append(0)

    data = np.asarray(image.array[tuple(selector)])
    retained = "".join(retained_axes)
    if "c" not in retained:
        data = data[None, ...]
        retained = "c" + retained

    permutation = [retained.index("c"), retained.index("y"), retained.index("x")]
    data = np.transpose(data, permutation)
    if data.ndim != 3:
        raise ValueError(f"Expected selected image in CYX form, received {data.shape}.")
    return PlaneData(
        cyx=data,
        channels=selected_channels,
        time_index=time_index,
        z_index=z_index,
    )


def percentile_limits(array: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    """Return finite percentile limits suitable for display normalization."""
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (p_low, p_high))
    low = float(low)
    high = float(high)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def normalize_with_limits(array: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map an array linearly to ``float32`` values in the interval ``[0, 1]``."""
    result = (np.asarray(array, dtype=np.float32) - low) / max(high - low, 1e-12)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_cyx(
    cyx: np.ndarray,
    *,
    p_low: float,
    p_high: float,
    reference_cyx: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Normalize each channel, optionally using limits derived from a reference.

    Using the raw image as the reference gives raw and processed panels the same
    intensity mapping and prevents visual differences caused only by independent
    contrast stretching.
    """
    source = np.asarray(reference_cyx if reference_cyx is not None else cyx)
    target = np.asarray(cyx)
    if source.shape[0] != target.shape[0]:
        raise ValueError("Reference and target images must contain the same channel count.")
    output = np.empty(target.shape, dtype=np.float32)
    limits: list[tuple[float, float]] = []
    for channel in range(target.shape[0]):
        low, high = percentile_limits(source[channel], p_low, p_high)
        output[channel] = normalize_with_limits(target[channel], low, high)
        limits.append((low, high))
    return output, limits


def compose_rgb(cyx01: np.ndarray, *, single_color: str = "gray") -> np.ndarray:
    """Compose normalized ``C, Y, X`` data into an RGB display image.

    Channel zero is mapped to blue and channel one to green, matching the PFT
    DAPI/WGA convention. Channel two is mapped to red. Additional channels are
    added equally to all RGB components. A single channel can be shown in gray,
    blue or green.
    """
    cyx01 = np.asarray(cyx01, dtype=np.float32)
    channels, height, width = cyx01.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    if channels == 1:
        plane = cyx01[0]
        if single_color == "blue":
            rgb[..., 2] = plane
        elif single_color == "green":
            rgb[..., 1] = plane
        elif single_color == "red":
            rgb[..., 0] = plane
        else:
            rgb[...] = plane[..., None]
        return np.clip(rgb, 0.0, 1.0)

    rgb[..., 2] = cyx01[0]
    rgb[..., 1] = cyx01[1]
    if channels >= 3:
        rgb[..., 0] = cyx01[2]
    if channels > 3:
        extra = np.mean(cyx01[3:], axis=0)
        rgb += extra[..., None] / 3.0
    return np.clip(rgb, 0.0, 1.0)


def difference_red_rgb(
    raw_cyx: np.ndarray,
    processed_cyx: np.ndarray,
    *,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return an absolute-difference map and its red RGB representation.

    The scalar difference is the mean absolute difference across selected
    channels. It is normalized only for display using the requested percentile.
    Numeric difference statistics remain based on the unnormalized values.
    """
    raw = np.asarray(raw_cyx, dtype=np.float32)
    processed = np.asarray(processed_cyx, dtype=np.float32)
    if raw.shape != processed.shape:
        raise ValueError(f"Comparison shape mismatch: raw={raw.shape}, processed={processed.shape}")
    difference = np.mean(np.abs(processed - raw), axis=0)
    vmax = float(np.percentile(difference, percentile)) if difference.size else 1.0
    vmax = max(vmax, 1e-12)
    normalized = np.clip(difference / vmax, 0.0, 1.0)
    rgb = np.zeros((*difference.shape, 3), dtype=np.float32)
    rgb[..., 0] = normalized
    return difference, rgb, vmax


def _matplotlib_pyplot(gui: bool = False):
    """Return Matplotlib pyplot with an interactive or non-interactive backend."""
    import matplotlib

    if not gui:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def add_scalebar(
    axis: Any,
    *,
    pixel_size_um_x: float | None,
    image_shape_yx: tuple[int, int],
    bar_um: float,
) -> None:
    """Draw a white scale bar when valid X pixel spacing is available."""
    if pixel_size_um_x is None or pixel_size_um_x <= 0 or bar_um <= 0:
        return
    height, width = image_shape_yx
    bar_pixels = max(2, int(round(bar_um / pixel_size_um_x)))
    bar_pixels = min(bar_pixels, max(2, width // 2))
    margin = max(10, int(round(min(height, width) * 0.03)))
    y = height - margin
    x1 = width - margin
    x0 = x1 - bar_pixels
    axis.plot([x0, x1], [y, y], color="white", linewidth=4, solid_capstyle="butt")
    axis.text(
        (x0 + x1) / 2,
        y - max(6, margin // 3),
        f"{bar_um:g} µm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=9,
    )


def requested_formats(value: str) -> tuple[str, ...]:
    """Convert ``png``, ``tiff`` or ``both`` to concrete filename suffixes."""
    value = value.lower()
    if value == "both":
        return ("png", "tiff")
    if value in {"png", "tiff", "tif"}:
        return ("tiff" if value in {"tiff", "tif"} else "png",)
    raise ValueError(f"Unsupported output format: {value}")


def save_rgb_array(rgb01: np.ndarray, path: Path) -> None:
    """Save an RGB float image as PNG or 16-bit RGB TIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(rgb01, dtype=np.float32), 0.0, 1.0)
    if path.suffix.lower() == ".png":
        Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB").save(path)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(path, np.round(rgb * 65535.0).astype(np.uint16), photometric="rgb")
    else:
        raise ValueError(f"Unsupported image suffix: {path.suffix}")


def save_gray_array(gray01: np.ndarray, path: Path) -> None:
    """Save a normalized grayscale image as PNG or unsigned 16-bit TIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = np.clip(np.asarray(gray01, dtype=np.float32), 0.0, 1.0)
    if path.suffix.lower() == ".png":
        Image.fromarray(np.round(gray * 255.0).astype(np.uint8), mode="L").save(path)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(path, np.round(gray * 65535.0).astype(np.uint16))
    else:
        raise ValueError(f"Unsupported image suffix: {path.suffix}")


def save_three_panel(
    raw_rgb: np.ndarray,
    processed_rgb: np.ndarray,
    difference_rgb: np.ndarray,
    *,
    output_base: Path,
    formats: Sequence[str],
    raw_title: str,
    processed_title: str,
    difference_title: str,
    pixel_size_um_x: float | None,
    scale_bar_um: float,
    show_scalebar: bool,
    dpi: int,
) -> list[Path]:
    """Save a three-panel raw/processed/red-difference comparison figure."""
    plt = _matplotlib_pyplot(gui=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = (raw_rgb, processed_rgb, difference_rgb)
    titles = (raw_title, processed_title, difference_title)
    for index, (axis, panel, title) in enumerate(zip(axes, panels, titles)):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
        if show_scalebar and index < 2:
            add_scalebar(
                axis,
                pixel_size_um_x=pixel_size_um_x,
                image_shape_yx=(panel.shape[0], panel.shape[1]),
                bar_um=scale_bar_um,
            )

    output_paths: list[Path] = []
    for suffix in formats:
        path = output_base.with_suffix(".tiff" if suffix == "tiff" else ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi)
        output_paths.append(path)
    plt.close(figure)
    return output_paths


def write_manifest(records: Sequence[ManifestRecord], output_dir: Path) -> Path:
    """Write generated-output provenance to a CSV manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "visualization_export_manifest.csv"
    fieldnames = list(ManifestRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


def sample_name_from_zarr(path: Path) -> str:
    """Return a stable sample name from a standard ``sample/image.ome.zarr`` path."""
    return path.parent.name if path.name.lower() == "image.ome.zarr" else path.stem.replace(".ome", "")


def _safe_relative_name(path: Path, root: Path | None = None) -> str:
    """Create a filesystem-safe name from a path, optionally relative to a root."""
    try:
        value = path.resolve().relative_to(root.resolve()) if root is not None else path.name
    except ValueError:
        value = path.name
    text = "__".join(value.parts) if isinstance(value, Path) else str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "image"


def export_one_plane(
    image: OmeImage,
    plane: PlaneData,
    *,
    output_dir: Path,
    view: str,
    formats: Sequence[str],
    p_low: float,
    p_high: float,
    single_color: str,
) -> list[ManifestRecord]:
    """Export one selected plane as per-channel and/or RGB files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized, _ = normalize_cyx(plane.cyx, p_low=p_low, p_high=p_high)
    stem = f"t{plane.time_index:03d}_z{plane.z_index:04d}"
    records: list[ManifestRecord] = []

    if view in {"normalized", "both"}:
        for local_index, original_index in enumerate(plane.channels):
            channel_name = (
                image.channel_names[original_index]
                if original_index < len(image.channel_names)
                else f"channel_{original_index}"
            )
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", channel_name)
            for suffix in formats:
                path = output_dir / f"{stem}_c{original_index}_{safe_name}_norm.{suffix}"
                save_gray_array(normalized[local_index], path)
                records.append(
                    ManifestRecord(
                        task="export",
                        source_path=str(image.path),
                        comparison_path="",
                        output_path=str(path),
                        output_kind="normalized_channel",
                        level=image.level,
                        time_index=plane.time_index,
                        z_index=plane.z_index,
                        channels=str(original_index),
                        source_shape=str(tuple(plane.cyx.shape)),
                        comparison_shape="",
                    )
                )

    if view in {"rgb", "both"}:
        rgb = compose_rgb(normalized, single_color=single_color)
        for suffix in formats:
            path = output_dir / f"{stem}_rgb_norm.{suffix}"
            save_rgb_array(rgb, path)
            records.append(
                ManifestRecord(
                    task="export",
                    source_path=str(image.path),
                    comparison_path="",
                    output_path=str(path),
                    output_kind="normalized_rgb",
                    level=image.level,
                    time_index=plane.time_index,
                    z_index=plane.z_index,
                    channels=",".join(map(str, plane.channels)),
                    source_shape=str(tuple(plane.cyx.shape)),
                    comparison_shape="",
                )
            )

    if view == "raw":
        for local_index, original_index in enumerate(plane.channels):
            path = output_dir / f"{stem}_c{original_index}_raw.tiff"
            tifffile.imwrite(path, plane.cyx[local_index])
            records.append(
                ManifestRecord(
                    task="export",
                    source_path=str(image.path),
                    comparison_path="",
                    output_path=str(path),
                    output_kind="raw_channel",
                    level=image.level,
                    time_index=plane.time_index,
                    z_index=plane.z_index,
                    channels=str(original_index),
                    source_shape=str(tuple(plane.cyx.shape)),
                    comparison_shape="",
                    notes="Raw numeric pixels; TIFF only.",
                )
            )
    return records


def run_export_task(args: argparse.Namespace) -> list[ManifestRecord]:
    """Execute OME-Zarr-to-TIFF/PNG export for one image or a recursive batch."""
    input_path = Path(args.input).expanduser() if args.input else DEFAULT_IMAGE_ROOT
    stores = discover_ome_zarr(input_path)
    if not stores:
        raise FileNotFoundError(f"No OME-Zarr images found under: {input_path}")
    if not args.all and len(stores) > 1:
        stores = [stores[0]]

    formats = requested_formats(args.format)
    all_records: list[ManifestRecord] = []
    for store in stores:
        image = open_ome_image(store, level=args.level)
        channels = parse_channels(args.channels, axis_size(image, "c"))
        time_indices = range(axis_size(image, "t")) if args.all_time else [args.time]
        z_indices = range(axis_size(image, "z")) if args.all_z else [args.z]
        sample_out = Path(args.output) if args.output and len(stores) == 1 else (
            Path(args.output) / _safe_relative_name(store, input_path)
            if args.output
            else DEFAULT_OUTPUT_ROOT / "export" / _safe_relative_name(store, input_path)
        )
        for time_index in time_indices:
            for z_index in z_indices:
                plane = select_plane_cyx(
                    image,
                    time_index=int(time_index),
                    z_index=int(z_index),
                    channels=channels,
                )
                all_records.extend(
                    export_one_plane(
                        image,
                        plane,
                        output_dir=sample_out,
                        view=args.view,
                        formats=formats,
                        p_low=args.p_low,
                        p_high=args.p_high,
                        single_color=args.single_color,
                    )
                )
        print(f"[EXPORT] {store} -> {sample_out}")
    return all_records


def _read_provenance_value(attrs: dict[str, Any], key: str) -> Any:
    """Read a provenance attribute from root attributes or ``pft_processing``."""
    if key in attrs:
        return attrs[key]
    processing = attrs.get("pft_processing")
    if isinstance(processing, dict):
        return processing.get(key)
    return None


def infer_raw_from_n2v(n2v_path: Path) -> tuple[Path, int, tuple[int, ...], str]:
    """Infer raw OME-Zarr path, frame and channels for an N2V output.

    Provenance attributes are preferred. Standard PFT output-path inference is
    used for legacy N2V images.
    """
    n2v_image = open_ome_image(n2v_path, level=0)
    attrs = n2v_image.attrs
    source_value = (
        _read_provenance_value(attrs, "pft_n2v_source_ome_zarr")
        or _read_provenance_value(attrs, "pft_n2v_source_zarr")
        or _read_provenance_value(attrs, "source_ome_zarr")
    )
    frame_value = _read_provenance_value(attrs, "pft_n2v_source_frame_index")
    channel_value = _read_provenance_value(attrs, "pft_n2v_source_channels")
    variant_value = (
        _read_provenance_value(attrs, "pft_n2v_variant")
        or _read_provenance_value(attrs, "pft_n2v_mode")
    )

    if source_value:
        source_path = Path(str(source_value)).expanduser()
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
    else:
        try:
            relative = n2v_path.resolve().relative_to(DEFAULT_N2V_ROOT.resolve())
            dataset, sample, variant = relative.parts[0], relative.parts[1], relative.parts[2]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "Cannot infer raw image from N2V path. Supply --raw explicitly or "
                "use the standard results/N2V/<dataset>/<sample>/<variant>/denoised/image.ome.zarr layout."
            ) from exc
        source_path = DEFAULT_IMAGE_ROOT / dataset / sample / "image.ome.zarr"
        variant_value = variant_value or variant

    frame_index = int(frame_value) if frame_value is not None else 0
    if isinstance(channel_value, (list, tuple)):
        channels = tuple(int(value) for value in channel_value)
    elif isinstance(channel_value, str) and channel_value.strip():
        channels = tuple(int(value.strip()) for value in channel_value.split(","))
    else:
        variant = str(variant_value or n2v_path.parent.parent.name)
        if variant in {"green_only"}:
            channels = (1,)
        elif variant in {"joint"}:
            channels = (0, 1)
        else:
            channels = (0,)
    return source_path.resolve(), frame_index, channels, str(variant_value or "unknown")


def compare_pair(
    raw_path: Path,
    processed_path: Path,
    *,
    task: str,
    output_dir: Path,
    level: int,
    raw_time: int,
    raw_z: int,
    processed_time: int,
    processed_z: int,
    channels: Sequence[int] | None,
    p_low: float,
    p_high: float,
    diff_percentile: float,
    single_color: str,
    formats: Sequence[str],
    raw_label: str,
    processed_label: str,
    scale_bar_um: float,
    show_scalebar: bool,
    dpi: int,
    save_individuals: bool,
) -> list[ManifestRecord]:
    """Create one raw/processed/red-difference comparison and manifest records."""
    raw_image = open_ome_image(raw_path, level=level)
    processed_image = open_ome_image(processed_path, level=level)

    raw_channels = tuple(channels) if channels is not None else tuple(range(axis_size(raw_image, "c")))
    processed_channel_count = axis_size(processed_image, "c")
    if processed_channel_count == len(raw_channels):
        processed_channels = tuple(range(processed_channel_count))
    elif processed_channel_count > max(raw_channels, default=0):
        processed_channels = raw_channels
    else:
        raise ValueError(
            f"Processed image has {processed_channel_count} channels but raw selection is {raw_channels}."
        )

    raw_plane = select_plane_cyx(
        raw_image,
        time_index=raw_time,
        z_index=raw_z,
        channels=raw_channels,
    )
    processed_plane = select_plane_cyx(
        processed_image,
        time_index=processed_time,
        z_index=processed_z,
        channels=processed_channels,
    )
    if raw_plane.cyx.shape != processed_plane.cyx.shape:
        raise ValueError(
            f"Selected comparison planes differ: raw={raw_plane.cyx.shape}, "
            f"processed={processed_plane.cyx.shape}."
        )

    raw_norm, _ = normalize_cyx(raw_plane.cyx, p_low=p_low, p_high=p_high)
    processed_norm, _ = normalize_cyx(
        processed_plane.cyx,
        p_low=p_low,
        p_high=p_high,
        reference_cyx=raw_plane.cyx,
    )
    raw_rgb = compose_rgb(raw_norm, single_color=single_color)
    processed_rgb = compose_rgb(processed_norm, single_color=single_color)
    difference, difference_rgb, difference_vmax = difference_red_rgb(
        raw_plane.cyx,
        processed_plane.cyx,
        percentile=diff_percentile,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_base = output_dir / "panel_raw_processed_difference_red"
    panel_paths = save_three_panel(
        raw_rgb,
        processed_rgb,
        difference_rgb,
        output_base=panel_base,
        formats=formats,
        raw_title=raw_label,
        processed_title=processed_label,
        difference_title=f"Absolute difference (red, p{diff_percentile:g} scale)",
        pixel_size_um_x=raw_image.scale_by_axis.get("x"),
        scale_bar_um=scale_bar_um,
        show_scalebar=show_scalebar,
        dpi=dpi,
    )

    records: list[ManifestRecord] = []
    common = dict(
        task=task,
        source_path=str(raw_path),
        comparison_path=str(processed_path),
        level=level,
        time_index=raw_time,
        z_index=raw_z,
        channels=",".join(map(str, raw_channels)),
        source_shape=str(tuple(raw_plane.cyx.shape)),
        comparison_shape=str(tuple(processed_plane.cyx.shape)),
        difference_mean=float(np.mean(difference)),
        difference_p99=float(np.percentile(difference, 99.0)),
        difference_max=float(np.max(difference)),
    )
    for path in panel_paths:
        records.append(
            ManifestRecord(
                output_path=str(path),
                output_kind="three_panel_red_difference",
                notes=f"Difference display vmax={difference_vmax:.8g}",
                **common,
            )
        )

    if save_individuals:
        for suffix in formats:
            extension = ".tiff" if suffix == "tiff" else ".png"
            raw_output = output_dir / f"raw_display{extension}"
            processed_output = output_dir / f"processed_display{extension}"
            difference_output = output_dir / f"difference_red{extension}"
            save_rgb_array(raw_rgb, raw_output)
            save_rgb_array(processed_rgb, processed_output)
            save_rgb_array(difference_rgb, difference_output)
            for path, kind in (
                (raw_output, "raw_display"),
                (processed_output, "processed_display"),
                (difference_output, "difference_red"),
            ):
                records.append(
                    ManifestRecord(
                        output_path=str(path),
                        output_kind=kind,
                        notes=f"Difference display vmax={difference_vmax:.8g}",
                        **common,
                    )
                )
        tifffile.imwrite(output_dir / "difference_numeric_float32.tiff", difference.astype(np.float32))
        records.append(
            ManifestRecord(
                output_path=str(output_dir / "difference_numeric_float32.tiff"),
                output_kind="difference_numeric_float32",
                notes="Unnormalized mean absolute difference across selected channels.",
                **common,
            )
        )
    return records


def run_n2v_compare_task(args: argparse.Namespace) -> list[ManifestRecord]:
    """Create raw-versus-N2V three-panel comparisons for one pair or a batch."""
    formats = requested_formats(args.format)
    processed_paths = discover_ome_zarr(Path(args.processed)) if args.processed else discover_ome_zarr(DEFAULT_N2V_ROOT)
    if not processed_paths:
        raise FileNotFoundError("No N2V OME-Zarr outputs were found.")
    if not args.all and len(processed_paths) > 1:
        processed_paths = [processed_paths[0]]

    records: list[ManifestRecord] = []
    for processed_path in processed_paths:
        if args.raw:
            raw_path = Path(args.raw).expanduser().resolve()
            raw_time = args.time
            channels = None
            variant = processed_path.parent.parent.name
        else:
            raw_path, inferred_frame, inferred_channels, variant = infer_raw_from_n2v(processed_path)
            raw_time = args.time if args.time_explicit else inferred_frame
            channels = (
                parse_channels(args.channels, axis_size(open_ome_image(raw_path), "c"))
                if args.channels_explicit
                else inferred_channels
            )
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw OME-Zarr corresponding to {processed_path} was not found: {raw_path}")
        sample = sample_name_from_zarr(raw_path)
        output_dir = (
            Path(args.output)
            if args.output and len(processed_paths) == 1
            else (Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / "n2v_comparison") / sample / variant
        )
        records.extend(
            compare_pair(
                raw_path,
                processed_path,
                task="n2v_compare",
                output_dir=output_dir,
                level=args.level,
                raw_time=raw_time,
                raw_z=args.z,
                processed_time=0,
                processed_z=0,
                channels=channels,
                p_low=args.p_low,
                p_high=args.p_high,
                diff_percentile=args.diff_percentile,
                single_color=args.single_color,
                formats=formats,
                raw_label="RAW",
                processed_label="N2V",
                scale_bar_um=args.scale_bar_um,
                show_scalebar=not args.no_scalebar,
                dpi=args.dpi,
                save_individuals=args.save_individuals,
            )
        )
        print(f"[N2V COMPARE] {sample} | {variant} -> {output_dir}")
    return records


def run_generic_compare_task(args: argparse.Namespace) -> list[ManifestRecord]:
    """Compare two explicitly supplied OME-Zarr images."""
    if not args.raw or not args.processed:
        raise ValueError("The compare task requires both --raw and --processed.")
    raw_path = Path(args.raw).expanduser().resolve()
    processed_path = Path(args.processed).expanduser().resolve()
    raw_image = open_ome_image(raw_path, level=args.level)
    channels = parse_channels(args.channels, axis_size(raw_image, "c"))
    output_dir = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_ROOT / "comparison" / f"{sample_name_from_zarr(raw_path)}__{sample_name_from_zarr(processed_path)}"
    )
    return compare_pair(
        raw_path,
        processed_path,
        task="compare",
        output_dir=output_dir,
        level=args.level,
        raw_time=args.time,
        raw_z=args.z,
        processed_time=args.processed_time,
        processed_z=args.processed_z,
        channels=channels,
        p_low=args.p_low,
        p_high=args.p_high,
        diff_percentile=args.diff_percentile,
        single_color=args.single_color,
        formats=requested_formats(args.format),
        raw_label=args.raw_label,
        processed_label=args.processed_label,
        scale_bar_um=args.scale_bar_um,
        show_scalebar=not args.no_scalebar,
        dpi=args.dpi,
        save_individuals=args.save_individuals,
    )


def pick_preview(folder: Path, pattern: str | None = None) -> Path | None:
    """Return the preferred preview image within one sample directory."""
    preferred = ["preview_norm.png", "preview_raw.png", "preview.png", "image_norm16_rgb.tif"]
    if pattern:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    for name in preferred:
        path = folder / name
        if path.exists():
            return path
    for glob_pattern in ("preview*.png", "*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
        matches = sorted(folder.glob(glob_pattern))
        if matches:
            return matches[0]
    return None


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image as RGB, compositing transparency onto black."""
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (0, 0, 0))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def create_collage_image(images: Sequence[Image.Image], *, columns: int, padding: int) -> Image.Image:
    """Create a white-background collage using equally sized image cells."""
    if not images:
        raise ValueError("At least one image is required for a collage.")
    columns = max(1, columns)
    cell_width = min(image.width for image in images)
    cell_height = min(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * padding,
            rows * cell_height + (rows + 1) * padding,
        ),
        (255, 255, 255),
    )
    for index, image in enumerate(images):
        tile = ImageOps.contain(image, (cell_width, cell_height))
        cell = Image.new("RGB", (cell_width, cell_height), (0, 0, 0))
        cell.paste(tile, ((cell_width - tile.width) // 2, (cell_height - tile.height) // 2))
        row, column = divmod(index, columns)
        x = padding + column * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        canvas.paste(cell, (x, y))
    return canvas


def run_collage_task(args: argparse.Namespace) -> list[ManifestRecord]:
    """Create one collage from sample preview images."""
    input_root = Path(args.input_root or args.input or DEFAULT_IMAGE_ROOT).expanduser().resolve()
    folders = sorted(path for path in input_root.iterdir() if path.is_dir())
    if args.group_key:
        folders = [folder for folder in folders if args.group_key.lower() in folder.name.lower()]
    items: list[tuple[Path, Path]] = []
    for folder in folders:
        preview = pick_preview(folder, args.pattern)
        if preview is not None:
            items.append((folder, preview))
    if not items:
        raise FileNotFoundError(f"No preview images found under: {input_root}")

    collage = create_collage_image(
        [load_rgb_image(path) for _, path in items],
        columns=args.columns,
        padding=args.padding,
    )
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / "collage"
    output_dir.mkdir(parents=True, exist_ok=True)
    group = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.group_key or input_root.name)
    output_path = output_dir / f"collage_{group}.png"
    collage.save(output_path, dpi=(args.dpi, args.dpi))
    print(f"[COLLAGE] {len(items)} images -> {output_path}")
    return [
        ManifestRecord(
            task="collage",
            source_path=str(input_root),
            comparison_path="",
            output_path=str(output_path),
            output_kind="collage",
            level=0,
            time_index=0,
            z_index=0,
            channels="",
            source_shape=f"{len(items)} images",
            comparison_shape="",
            notes="; ".join(str(path) for _, path in items),
        )
    ]


def iteration_number(path: Path) -> int:
    """Return a numeric iteration label inferred from a path, or a large fallback."""
    matches = re.findall(r"(?:iter(?:ation)?[_-]?)?(\d+)", str(path), flags=re.IGNORECASE)
    return int(matches[-1]) if matches else 10**9


def run_iterations_task(args: argparse.Namespace) -> list[ManifestRecord]:
    """Create a horizontal panel containing raw and multiple processed iterations."""
    if not args.raw:
        raise ValueError("The iterations task requires --raw.")
    if not args.processed_root:
        raise ValueError("The iterations task requires --processed-root.")
    raw_path = Path(args.raw).expanduser().resolve()
    processed_paths = discover_ome_zarr(Path(args.processed_root))
    if not processed_paths:
        raise FileNotFoundError("No processed OME-Zarr images were found under --processed-root.")
    processed_paths.sort(key=lambda path: (iteration_number(path), str(path).lower()))

    raw_image = open_ome_image(raw_path, args.level)
    channels = parse_channels(args.channels, axis_size(raw_image, "c"))
    raw_plane = select_plane_cyx(raw_image, time_index=args.time, z_index=args.z, channels=channels)
    raw_norm, _ = normalize_cyx(raw_plane.cyx, p_low=args.p_low, p_high=args.p_high)
    panels = [compose_rgb(raw_norm, single_color=args.single_color)]
    titles = [args.raw_label]

    records: list[ManifestRecord] = []
    for processed_path in processed_paths:
        image = open_ome_image(processed_path, args.level)
        processed_channel_count = axis_size(image, "c")
        selected = tuple(range(processed_channel_count)) if processed_channel_count == len(channels) else channels
        plane = select_plane_cyx(
            image,
            time_index=args.processed_time,
            z_index=args.processed_z,
            channels=selected,
        )
        if plane.cyx.shape != raw_plane.cyx.shape:
            print(f"[SKIP] Shape mismatch: {processed_path} {plane.cyx.shape}")
            continue
        normalized, _ = normalize_cyx(
            plane.cyx,
            p_low=args.p_low,
            p_high=args.p_high,
            reference_cyx=raw_plane.cyx,
        )
        panels.append(compose_rgb(normalized, single_color=args.single_color))
        number = iteration_number(processed_path)
        titles.append(f"Iteration {number}" if number < 10**9 else processed_path.parent.name)

    plt = _matplotlib_pyplot(gui=False)
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), constrained_layout=True)
    axes_sequence = [axes] if len(panels) == 1 else list(axes)
    for index, (axis, panel, title) in enumerate(zip(axes_sequence, panels, titles)):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
        if not args.no_scalebar:
            add_scalebar(
                axis,
                pixel_size_um_x=raw_image.scale_by_axis.get("x"),
                image_shape_yx=(panel.shape[0], panel.shape[1]),
                bar_um=args.scale_bar_um,
            )

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / "iterations" / sample_name_from_zarr(raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in requested_formats(args.format):
        output_path = output_dir / f"iteration_panel.{suffix}"
        figure.savefig(output_path, dpi=args.dpi)
        records.append(
            ManifestRecord(
                task="iterations",
                source_path=str(raw_path),
                comparison_path=str(Path(args.processed_root)),
                output_path=str(output_path),
                output_kind="iteration_panel",
                level=args.level,
                time_index=args.time,
                z_index=args.z,
                channels=",".join(map(str, channels)),
                source_shape=str(tuple(raw_plane.cyx.shape)),
                comparison_shape=f"{len(panels) - 1} processed images",
            )
        )
    plt.close(figure)
    print(f"[ITERATIONS] {len(panels) - 1} processed images -> {output_dir}")
    return records


def parse_roi(value: str, shape_yx: tuple[int, int]) -> tuple[int, int, int, int]:
    """Parse ``y0,y1,x0,x1`` and clamp the crop to image bounds."""
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must use the form y0,y1,x0,x1.")
    y0, y1, x0, x1 = parts
    height, width = shape_yx
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    return y0, y1, x0, x1


class InteractiveCropper:
    """Matplotlib rectangle selector used by the interactive crop task."""

    def __init__(self, image_rgb: np.ndarray, title: str) -> None:
        """Create the crop window and register selection callbacks."""
        plt = _matplotlib_pyplot(gui=True)
        from matplotlib.widgets import RectangleSelector

        self.plt = plt
        self.roi: tuple[int, int, int, int] | None = None
        self.figure, self.axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
        self.axis.imshow(image_rgb)
        self.axis.set_title(title)
        self.axis.axis("off")
        self.selector = RectangleSelector(
            self.axis,
            self._on_select,
            useblit=True,
            button=[1],
            interactive=True,
            drag_from_anywhere=True,
        )
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_select(self, click: Any, release: Any) -> None:
        """Store the rectangle selected with the mouse."""
        if None in (click.xdata, click.ydata, release.xdata, release.ydata):
            return
        x0, x1 = sorted((int(math.floor(click.xdata)), int(math.ceil(release.xdata))))
        y0, y1 = sorted((int(math.floor(click.ydata)), int(math.ceil(release.ydata))))
        self.roi = (y0, y1, x0, x1)

    def _on_key(self, event: Any) -> None:
        """Confirm with Enter or cancel with Escape."""
        if event.key == "enter":
            self.plt.close(self.figure)
        elif event.key == "escape":
            self.roi = None
            self.plt.close(self.figure)

    def run(self) -> tuple[int, int, int, int] | None:
        """Display the crop window and return the selected rectangle."""
        print("Drag a rectangle, then press Enter. Press Escape to cancel.")
        self.plt.show()
        return self.roi


def run_crop_task(args: argparse.Namespace) -> list[ManifestRecord]:
    """Select or specify one crop and export raw and normalized crop files."""
    if not args.input:
        raise ValueError("The crop task requires --input.")
    image = open_ome_image(Path(args.input), args.level)
    channels = parse_channels(args.channels, axis_size(image, "c"))
    plane = select_plane_cyx(image, time_index=args.time, z_index=args.z, channels=channels)
    normalized, _ = normalize_cyx(plane.cyx, p_low=args.p_low, p_high=args.p_high)
    rgb = compose_rgb(normalized, single_color=args.single_color)

    if args.roi:
        roi = parse_roi(args.roi, (plane.cyx.shape[1], plane.cyx.shape[2]))
    else:
        selected = InteractiveCropper(rgb, f"{sample_name_from_zarr(image.path)}: select crop").run()
        if selected is None:
            raise RuntimeError("Crop selection was cancelled.")
        roi = parse_roi(",".join(map(str, selected)), (plane.cyx.shape[1], plane.cyx.shape[2]))

    y0, y1, x0, x1 = roi
    raw_crop = plane.cyx[:, y0:y1, x0:x1]
    normalized_crop = normalized[:, y0:y1, x0:x1]
    rgb_crop = compose_rgb(normalized_crop, single_color=args.single_color)
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / "crops" / sample_name_from_zarr(image.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"crop_y{y0}-{y1}_x{x0}-{x1}_raw_cyx.tiff"
    tifffile.imwrite(raw_path, raw_crop)
    records = [
        ManifestRecord(
            task="crop",
            source_path=str(image.path),
            comparison_path="",
            output_path=str(raw_path),
            output_kind="raw_crop_cyx",
            level=args.level,
            time_index=args.time,
            z_index=args.z,
            channels=",".join(map(str, channels)),
            source_shape=str(tuple(plane.cyx.shape)),
            comparison_shape=str(tuple(raw_crop.shape)),
            notes=f"ROI={roi}",
        )
    ]
    for suffix in requested_formats(args.format):
        path = output_dir / f"crop_y{y0}-{y1}_x{x0}-{x1}_rgb_norm.{suffix}"
        save_rgb_array(rgb_crop, path)
        records.append(
            ManifestRecord(
                task="crop",
                source_path=str(image.path),
                comparison_path="",
                output_path=str(path),
                output_kind="normalized_rgb_crop",
                level=args.level,
                time_index=args.time,
                z_index=args.z,
                channels=",".join(map(str, channels)),
                source_shape=str(tuple(plane.cyx.shape)),
                comparison_shape=str(tuple(raw_crop.shape)),
                notes=f"ROI={roi}",
            )
        )
    print(f"[CROP] ROI {roi} -> {output_dir}")
    return records


def choose(items: Sequence[str], prompt: str, default: int = 0) -> int:
    """Prompt for a zero-based menu choice."""
    print(f"\n{prompt}")
    for index, item in enumerate(items):
        marker = " (default)" if index == default else ""
        print(f"  [{index}] {item}{marker}")
    while True:
        value = input("Enter number: ").strip()
        if not value:
            return default
        try:
            index = int(value)
        except ValueError:
            print("Please enter an integer.")
            continue
        if 0 <= index < len(items):
            return index
        print(f"Please enter a value from 0 to {len(items) - 1}.")


def prompt_path(prompt: str, default: Path | None = None) -> Path:
    """Prompt for a filesystem path and expand the user directory marker."""
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return Path(value).expanduser() if value else Path(default) if default is not None else Path()


def interactive_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Collect missing task-specific values using a terminal menu."""
    tasks = [
        ("Export OME-Zarr planes to TIFF/PNG", "export"),
        ("Create preview collage", "collage"),
        ("Compare RAW and N2V with red difference", "n2v-compare"),
        ("Compare RAW and another processed OME-Zarr", "compare"),
        ("Create deconvolution-iteration panel", "iterations"),
        ("Select and export a crop", "crop"),
    ]
    task_index = choose([label for label, _ in tasks], "Select visualization/export task")
    args.task = tasks[task_index][1]

    if args.task == "export":
        args.input = str(prompt_path("OME-Zarr file or parent directory", DEFAULT_IMAGE_ROOT))
        args.all = choose(["One discovered image", "All discovered images"], "Processing scope") == 1
        args.view = ["normalized", "rgb", "both", "raw"][
            choose(
                ["Normalized per-channel images", "Normalized RGB", "Both normalized outputs", "Raw numeric TIFF"],
                "Select export content",
                default=2,
            )
        ]
        args.all_z = choose(["One Z plane", "All Z planes"], "Z selection") == 1
        if not args.all_z:
            args.z = int(input("Z index [0]: ").strip() or "0")
    elif args.task == "collage":
        args.input_root = str(prompt_path("Folder containing sample preview folders", DEFAULT_IMAGE_ROOT / "2d_wga_dapi"))
        args.group_key = input("Optional group-name filter [all]: ").strip() or None
    elif args.task == "n2v-compare":
        args.processed = str(prompt_path("N2V OME-Zarr or parent directory", DEFAULT_N2V_ROOT))
        args.all = choose(["One discovered N2V image", "All discovered N2V images"], "Processing scope") == 1
    elif args.task == "compare":
        args.raw = str(prompt_path("Raw OME-Zarr"))
        args.processed = str(prompt_path("Processed/deconvolved OME-Zarr"))
        args.processed_label = input("Processed-panel label [PROCESSED]: ").strip() or "PROCESSED"
    elif args.task == "iterations":
        args.raw = str(prompt_path("Raw OME-Zarr"))
        args.processed_root = str(prompt_path("Folder containing iteration OME-Zarr images"))
    elif args.task == "crop":
        args.input = str(prompt_path("OME-Zarr to crop"))
        args.roi = input("ROI y0,y1,x0,x1 [empty for mouse selection]: ").strip() or None
    return args


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for all visualization and export tasks."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified PFT visualization/export tool. Omit --task for an interactive menu."
        )
    )
    parser.add_argument(
        "--task",
        choices=("export", "collage", "n2v-compare", "compare", "iterations", "crop"),
        help="Output to generate. Omit for the interactive menu.",
    )
    parser.add_argument("--input", help="Input OME-Zarr file or input root.")
    parser.add_argument("--input-root", help="Input root used by the collage task.")
    parser.add_argument("--raw", help="Raw/source OME-Zarr path.")
    parser.add_argument("--processed", help="Processed/N2V OME-Zarr path or parent root.")
    parser.add_argument("--processed-root", help="Root containing processed iteration OME-Zarr images.")
    parser.add_argument("--output", help="Output directory. A results/visualization_export default is used otherwise.")
    parser.add_argument("--all", action="store_true", help="Process every discovered image instead of the first one.")
    parser.add_argument("--level", type=int, default=0, help="OME-Zarr pyramid level (default: 0).")
    parser.add_argument("--time", type=int, default=0, help="Raw/source time index (default: 0).")
    parser.add_argument("--processed-time", type=int, default=0, help="Processed-image time index (default: 0).")
    parser.add_argument("--z", type=int, default=0, help="Raw/source Z index (default: 0).")
    parser.add_argument("--processed-z", type=int, default=0, help="Processed-image Z index (default: 0).")
    parser.add_argument("--all-time", action="store_true", help="Export every time index.")
    parser.add_argument("--all-z", action="store_true", help="Export every Z index.")
    parser.add_argument("--channels", default="all", help="Channel indices, for example '0', '0,1' or 'all'.")
    parser.add_argument(
        "--view",
        choices=("normalized", "rgb", "both", "raw"),
        default="both",
        help="Content generated by the export task (default: both normalized channel and RGB files).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "tiff", "both"),
        default="both",
        help="Figure/export format (default: both).",
    )
    parser.add_argument("--p-low", type=float, default=1.0, help="Lower display percentile (default: 1).")
    parser.add_argument("--p-high", type=float, default=99.8, help="Upper display percentile (default: 99.8).")
    parser.add_argument(
        "--diff-percentile",
        type=float,
        default=99.9,
        help="Percentile mapped to maximum red in difference panels (default: 99.9).",
    )
    parser.add_argument(
        "--single-color",
        choices=("gray", "blue", "green", "red"),
        default="gray",
        help="Display colour for single-channel images (default: gray).",
    )
    parser.add_argument("--scale-bar-um", type=float, default=2.0, help="Scale-bar length in micrometres.")
    parser.add_argument("--no-scalebar", action="store_true", help="Do not draw scale bars.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution (default: 300 dpi).")
    parser.add_argument("--save-individuals", action="store_true", help="Also save separate raw, processed and difference images.")
    parser.add_argument("--raw-label", default="RAW", help="Title for the raw panel.")
    parser.add_argument("--processed-label", default="PROCESSED", help="Title for the processed panel.")
    parser.add_argument("--group-key", help="Folder-name substring used by the collage task.")
    parser.add_argument("--pattern", help="Optional preview-file glob used by the collage task.")
    parser.add_argument("--columns", type=int, default=2, help="Collage column count (default: 2).")
    parser.add_argument("--padding", type=int, default=18, help="Collage padding in pixels (default: 18).")
    parser.add_argument("--roi", help="Crop coordinates in y0,y1,x0,x1 form. Omit for mouse selection.")
    return parser


def validate_numeric_arguments(args: argparse.Namespace) -> None:
    """Validate shared percentile, scale and index parameters before processing."""
    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("Display percentiles must satisfy 0 <= p-low < p-high <= 100.")
    if not (0.0 < args.diff_percentile <= 100.0):
        raise ValueError("Difference percentile must be in the interval (0, 100].")
    if args.level < 0 or args.time < 0 or args.z < 0 or args.processed_time < 0 or args.processed_z < 0:
        raise ValueError("Level, time and Z indices must be non-negative.")
    if args.columns < 1 or args.padding < 0 or args.dpi < 1:
        raise ValueError("Columns and DPI must be positive; padding must be non-negative.")


def manifest_output_directory(args: argparse.Namespace, records: Sequence[ManifestRecord]) -> Path:
    """Resolve the directory in which the run-level manifest should be stored."""
    if args.output:
        return Path(args.output)
    if records:
        paths = [Path(record.output_path).parent for record in records]
        common = Path(paths[0])
        for path in paths[1:]:
            while common not in (path, *path.parents) and common != common.parent:
                common = common.parent
        return common
    return DEFAULT_OUTPUT_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected visualization/export task and write its CSV manifest."""
    parser = build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)
    args.time_explicit = "--time" in argv_list
    args.channels_explicit = "--channels" in argv_list
    if args.task is None:
        args = interactive_arguments(parser, args)
    validate_numeric_arguments(args)

    if args.task == "export":
        records = run_export_task(args)
    elif args.task == "collage":
        records = run_collage_task(args)
    elif args.task == "n2v-compare":
        records = run_n2v_compare_task(args)
    elif args.task == "compare":
        records = run_generic_compare_task(args)
    elif args.task == "iterations":
        records = run_iterations_task(args)
    elif args.task == "crop":
        records = run_crop_task(args)
    else:  # pragma: no cover - protected by argparse
        parser.error(f"Unsupported task: {args.task}")
        return 2

    manifest = write_manifest(records, manifest_output_directory(args, records))
    print("\nPFT visualization/export summary")
    print("=" * 72)
    print(f"Task: {args.task}")
    print(f"Generated files recorded: {len(records)}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
