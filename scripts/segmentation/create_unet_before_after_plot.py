"""
Create  original vs U-Net microscopy figures.

One 2d_time sample:

    python scripts/segmentation/create_unet_before_after_plot.py \
        --dataset 2d_time \
        --inference-root results/U-net/2d_time_v2_fixedval_dep998 \
        --sample WT_HADA_NHS_40min_ROI1_SIM \
        --view merged \
        --scale-bar-um 2 \
        --format both \
        --non-interactive

All WGA-DAPI samples:

    python scripts/segmentation/create_unet_before_after_plot.py \
        --dataset 2d_wga_dapi \
        --inference-root results/U-net/2d_wga_dapi_v2_fixedval_dep998 \
        --mode all \
        --view all \
        --scale-bar-um 2 \
        --format both \
        --non-interactive
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``scripts`` and ``src/PFT``."""
    current = (start or SCRIPT_PATH).resolve()
    search_start = current if current.is_dir() else current.parent
    for candidate in (search_start, *search_start.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the PFT project root. Expected sibling 'scripts' "
        "and 'src/PFT' directories."
    )


PROJECT_ROOT = find_project_root()
RESULTS_ROOT = PROJECT_ROOT / "results"
RAW_ROOT = RESULTS_ROOT / "img"
UNET_ROOT = RESULTS_ROOT / "U-net"


@dataclass(frozen=True)
class OmePlane:
    """Selected two-dimensional fluorescence plane and physical metadata."""

    cyx: np.ndarray
    axes: str
    pixel_size_um_x: float | None
    channel_names: tuple[str, ...]


def require_zarr():
    """Import zarr with an environment-specific error message."""
    try:
        import zarr
    except Exception as exc:
        raise ImportError(
            "This plotting script requires zarr. Activate the PFT environment "
            "and install the project dependencies."
        ) from exc
    return zarr


def is_ome_zarr(path: Path) -> bool:
    """Return True when a path appears to be an OME-Zarr store."""
    return path.is_dir() and (
        path.name.lower().endswith(".ome.zarr")
        or any((path / marker).exists() for marker in (".zgroup", ".zattrs", "zarr.json"))
    )


def axes_string(value: Any, ndim: int) -> str:
    """Convert OME-NGFF axes metadata into a lowercase axis string."""
    if isinstance(value, str) and len(value) == ndim:
        return value.lower()
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                names.append(str(item.get("name", "")).lower())
            else:
                names.append(str(item).lower())
        joined = "".join(names)
        if len(joined) == ndim:
            return joined
    return {2: "yx", 3: "cyx", 4: "czyx", 5: "tczyx"}.get(
        ndim, "".join(f"u{i}" for i in range(ndim))
    )


def read_level_metadata(
    attrs: dict[str, Any],
    *,
    level: int,
    provisional_ndim: int,
) -> tuple[str, str, dict[str, float]]:
    """Read axes, level path, and physical scales from OME-NGFF metadata."""
    axes = ""
    level_path = str(level)
    scales: dict[str, float] = {}

    multiscales = attrs.get("multiscales")
    if not (isinstance(multiscales, list) and multiscales and isinstance(multiscales[0], dict)):
        return axes, level_path, scales

    entry = multiscales[0]
    axes = axes_string(entry.get("axes"), provisional_ndim)
    datasets = entry.get("datasets")
    if not (isinstance(datasets, list) and datasets):
        return axes, level_path, scales
    if level < 0 or level >= len(datasets):
        raise IndexError(f"Requested pyramid level {level}, but only {len(datasets)} levels exist.")

    dataset = datasets[level]
    if not isinstance(dataset, dict):
        return axes, level_path, scales

    level_path = str(dataset.get("path", level))
    transforms = dataset.get("coordinateTransformations", [])
    if isinstance(transforms, list):
        for transform in transforms:
            if not isinstance(transform, dict) or transform.get("type") != "scale":
                continue
            scale = transform.get("scale")
            if isinstance(scale, list) and len(scale) == len(axes):
                for axis, value in zip(axes, scale):
                    try:
                        scales[axis] = float(value)
                    except (TypeError, ValueError):
                        pass
                break
    return axes, level_path, scales


def select_cyx(array: Any, axes: str, *, time_index: int, z_index: int) -> np.ndarray:
    """Read one plane and return it in C,Y,X order."""
    if "y" not in axes or "x" not in axes:
        raise ValueError(f"OME-Zarr axes must contain y and x; received '{axes}'.")

    selector: list[Any] = []
    retained_axes: list[str] = []

    for axis_index, axis in enumerate(axes):
        size = int(array.shape[axis_index])
        if axis == "t":
            if not 0 <= time_index < size:
                raise IndexError(f"Time index {time_index} outside 0..{size - 1}.")
            selector.append(time_index)
        elif axis == "z":
            if not 0 <= z_index < size:
                raise IndexError(f"Z index {z_index} outside 0..{size - 1}.")
            selector.append(z_index)
        elif axis in {"c", "y", "x"}:
            selector.append(slice(None))
            retained_axes.append(axis)
        else:
            selector.append(0)

    data = np.asarray(array[tuple(selector)])
    retained = "".join(retained_axes)

    if "c" not in retained:
        data = data[None, ...]
        retained = "c" + retained

    permutation = [retained.index("c"), retained.index("y"), retained.index("x")]
    cyx = np.transpose(data, permutation)
    if cyx.ndim != 3:
        raise ValueError(f"Expected a CYX plane, received shape {cyx.shape}.")
    return cyx


def open_ome_plane(
    path: Path,
    *,
    level: int,
    time_index: int,
    z_index: int,
) -> OmePlane:
    """Open one OME-Zarr image plane and its physical metadata."""
    zarr = require_zarr()
    path = path.expanduser().resolve()
    if not is_ome_zarr(path):
        raise FileNotFoundError(f"OME-Zarr store not found: {path}")

    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)

    provisional_path = str(level)
    provisional = root[provisional_path] if provisional_path in root else None
    provisional_ndim = int(getattr(provisional, "ndim", 0)) if provisional is not None else 0

    axes, level_path, scales = read_level_metadata(
        attrs,
        level=level,
        provisional_ndim=provisional_ndim,
    )
    if level_path not in root:
        if str(level) in root:
            level_path = str(level)
        else:
            raise KeyError(f"OME-Zarr level '{level_path}' does not exist in {path}")

    array = root[level_path]
    if not axes or len(axes) != array.ndim:
        axes = axes_string(attrs.get("pft_axes"), array.ndim)

    channel_names_value = attrs.get("channel_names")
    if not isinstance(channel_names_value, list):
        pft_meta = attrs.get("pft_meta")
        channel_names_value = (
            pft_meta.get("channel_names") if isinstance(pft_meta, dict) else None
        )
    channel_count = int(array.shape[axes.index("c")]) if "c" in axes else 1
    if not isinstance(channel_names_value, list):
        channel_names_value = [f"channel_{index}" for index in range(channel_count)]

    return OmePlane(
        cyx=select_cyx(array, axes, time_index=time_index, z_index=z_index),
        axes=axes,
        pixel_size_um_x=scales.get("x"),
        channel_names=tuple(str(name) for name in channel_names_value),
    )


def percentile_limits(array: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    """Return robust finite display limits."""
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


def normalize_pair_from_raw(
    raw_cyx: np.ndarray,
    processed_cyx: np.ndarray,
    *,
    p_low: float,
    p_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize raw and processed channels using limits derived only from raw."""
    raw = np.asarray(raw_cyx)
    processed = np.asarray(processed_cyx)
    if raw.shape != processed.shape:
        raise ValueError(f"Raw and processed shapes differ: {raw.shape} versus {processed.shape}.")

    raw_out = np.empty(raw.shape, dtype=np.float32)
    processed_out = np.empty(processed.shape, dtype=np.float32)

    for channel in range(raw.shape[0]):
        low, high = percentile_limits(raw[channel], p_low, p_high)
        denominator = max(high - low, 1e-12)
        raw_out[channel] = np.clip((raw[channel].astype(np.float32) - low) / denominator, 0, 1)
        processed_out[channel] = np.clip(
            (processed[channel].astype(np.float32) - low) / denominator, 0, 1
        )
    return raw_out, processed_out


def single_channel_rgb(channel_yx: np.ndarray, color: str) -> np.ndarray:
    """Map one normalized channel to blue or green RGB."""
    plane = np.clip(np.asarray(channel_yx, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros((*plane.shape, 3), dtype=np.float32)
    if color == "blue":
        rgb[..., 2] = plane
    elif color == "green":
        rgb[..., 1] = plane
    else:
        raise ValueError(f"Unsupported fluorescence display colour: {color}")
    return rgb


def merged_rgb(cyx01: np.ndarray, dataset: str) -> np.ndarray:
    """Compose the dataset-specific merged fluorescence display."""
    data = np.asarray(cyx01, dtype=np.float32)
    height, width = data.shape[-2:]
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    if dataset == "2d_time":
        rgb[..., 2] = data[0]
    elif dataset == "2d_wga_dapi":
        if data.shape[0] < 2:
            raise ValueError(
                "2d_wga_dapi requires at least two channels: DAPI channel 0 and WGA channel 1."
            )
        rgb[..., 2] = data[0]  # DAPI
        rgb[..., 1] = data[1]  # WGA
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return np.clip(rgb, 0.0, 1.0)


def add_scalebar(
    axis: Any,
    *,
    pixel_size_um_x: float,
    image_shape_yx: tuple[int, int],
    bar_um: float,
) -> None:
    """Draw a white micrometre scale bar with a black outline."""
    import matplotlib.patheffects as path_effects

    if pixel_size_um_x <= 0 or bar_um <= 0:
        raise ValueError("Pixel size and scale-bar length must be positive.")

    height, width = image_shape_yx
    bar_pixels = max(2, int(round(bar_um / pixel_size_um_x)))
    if bar_pixels >= width:
        raise ValueError(
            f"Requested {bar_um:g} µm scale bar requires {bar_pixels} pixels, "
            f"but the image width is only {width} pixels."
        )

    margin = max(10, int(round(min(height, width) * 0.04)))
    y = height - margin
    x1 = width - margin
    x0 = x1 - bar_pixels

    axis.plot([x0, x1], [y, y], color="black", linewidth=6, solid_capstyle="butt")
    axis.plot([x0, x1], [y, y], color="white", linewidth=3.5, solid_capstyle="butt")
    label = axis.text(
        (x0 + x1) / 2,
        y - max(7, margin // 3),
        f"{bar_um:g} µm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])


def safe_sample_name(value: str) -> str:
    """Return a filesystem-safe sample identifier."""
    return re.sub(r"[^A-Za-z0-9_.²-]+", "_", value).strip("_") or "sample"


def save_figure(
    figure: Any,
    output_base: Path,
    output_format: str,
    dpi: int,
) -> list[Path]:
    """Save a figure in PNG, PDF, or both formats."""
    suffixes = ("png", "pdf") if output_format == "both" else (output_format,)
    outputs: list[Path] = []
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        path = output_base.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    return outputs


def plot_sample(
    *,
    dataset: str,
    sample: str,
    raw_path: Path,
    processed_path: Path,
    output_root: Path,
    view: str,
    level: int,
    time_index: int,
    z_index: int,
    p_low: float,
    p_high: float,
    scale_bar_um: float,
    pixel_size_override: float | None,
    output_format: str,
    dpi: int,
    original_label: str,
    processed_label: str,
) -> tuple[list[Path], float]:
    """Create one publication-ready before/after figure."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    raw = open_ome_plane(
        raw_path,
        level=level,
        time_index=time_index,
        z_index=z_index,
    )
    processed = open_ome_plane(
        processed_path,
        level=level,
        time_index=time_index,
        z_index=z_index,
    )

    raw_norm, processed_norm = normalize_pair_from_raw(
        raw.cyx,
        processed.cyx,
        p_low=p_low,
        p_high=p_high,
    )

    pixel_size_um_x = (
        float(pixel_size_override)
        if pixel_size_override is not None
        else raw.pixel_size_um_x
    )
    if pixel_size_um_x is None or not math.isfinite(pixel_size_um_x) or pixel_size_um_x <= 0:
        raise ValueError(
            f"No valid X pixel size was found in the original OME-Zarr metadata for {sample}. "
            "Supply --pixel-size-um explicitly. No figure was saved."
        )

    if dataset == "2d_time":
        figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        panels = (
            merged_rgb(raw_norm, dataset),
            merged_rgb(processed_norm, dataset),
        )
        titles = (original_label, processed_label)
        for axis, image, title in zip(axes, panels, titles):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
            add_scalebar(
                axis,
                pixel_size_um_x=pixel_size_um_x,
                image_shape_yx=image.shape[:2],
                bar_um=scale_bar_um,
            )
    else:
        if raw_norm.shape[0] < 2:
            raise ValueError("The WGA-DAPI figure requires both DAPI and WGA channels.")

        if view == "merged":
            figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            panels = (
                merged_rgb(raw_norm, dataset),
                merged_rgb(processed_norm, dataset),
            )
            titles = (original_label, processed_label)
            for axis, image, title in zip(axes, panels, titles):
                axis.imshow(image)
                axis.set_title(title)
                axis.axis("off")
                add_scalebar(
                    axis,
                    pixel_size_um_x=pixel_size_um_x,
                    image_shape_yx=image.shape[:2],
                    bar_um=scale_bar_um,
                )
        elif view == "channels":
            figure, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
            rows = (
                ("DAPI", "blue", 0),
                ("WGA", "green", 1),
            )
            for row_index, (channel_label, color, channel_index) in enumerate(rows):
                images = (
                    single_channel_rgb(raw_norm[channel_index], color),
                    single_channel_rgb(processed_norm[channel_index], color),
                )
                titles = (
                    f"{original_label}: {channel_label}",
                    f"{processed_label}: {channel_label}",
                )
                for column_index, (image, title) in enumerate(zip(images, titles)):
                    axis = axes[row_index, column_index]
                    axis.imshow(image)
                    axis.set_title(title)
                    axis.axis("off")
                    add_scalebar(
                        axis,
                        pixel_size_um_x=pixel_size_um_x,
                        image_shape_yx=image.shape[:2],
                        bar_um=scale_bar_um,
                    )
        else:
            figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
            raw_panels = (
                single_channel_rgb(raw_norm[0], "blue"),
                single_channel_rgb(raw_norm[1], "green"),
                merged_rgb(raw_norm, dataset),
            )
            processed_panels = (
                single_channel_rgb(processed_norm[0], "blue"),
                single_channel_rgb(processed_norm[1], "green"),
                merged_rgb(processed_norm, dataset),
            )
            column_titles = ("DAPI", "WGA", "Merged")
            for row_index, (row_label, panels) in enumerate(
                ((original_label, raw_panels), (processed_label, processed_panels))
            ):
                for column_index, (column_label, image) in enumerate(zip(column_titles, panels)):
                    axis = axes[row_index, column_index]
                    axis.imshow(image)
                    axis.set_title(f"{row_label}: {column_label}")
                    axis.axis("off")
                    add_scalebar(
                        axis,
                        pixel_size_um_x=pixel_size_um_x,
                        image_shape_yx=image.shape[:2],
                        bar_um=scale_bar_um,
                    )

    figure.suptitle(sample, fontsize=13)
    output_base = output_root / safe_sample_name(sample) / "original_vs_unet_filtered"
    paths = save_figure(figure, output_base, output_format, dpi)
    plt.close(figure)
    return paths, pixel_size_um_x


def choose_number(title: str, options: Sequence[str]) -> int:
    """Prompt for one numbered item."""
    if not options:
        raise RuntimeError(f"No options are available for {title}.")
    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index:3d}. {option}")
    while True:
        value = input("Select number: ").strip()
        try:
            selected = int(value)
        except ValueError:
            print("Enter a valid number.")
            continue
        if 1 <= selected <= len(options):
            return selected - 1
        print(f"Select a value from 1 to {len(options)}.")


def discover_inference_roots(dataset: str) -> list[Path]:
    """Find U-Net inference roots containing completed sample outputs."""
    if not UNET_ROOT.exists():
        return []
    candidates: list[Path] = []
    for path in UNET_ROOT.iterdir():
        if not path.is_dir() or dataset not in path.name:
            continue
        if any(path.glob("*/foreground_filtered.ome.zarr")):
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def discover_samples(dataset: str, inference_root: Path) -> list[str]:
    """Return samples having both original and processed OME-Zarr images."""
    raw_dataset_root = RAW_ROOT / dataset
    samples: list[str] = []
    if not inference_root.exists():
        return samples
    for processed in inference_root.glob("*/foreground_filtered.ome.zarr"):
        sample = processed.parent.name
        raw = raw_dataset_root / sample / "image.ome.zarr"
        if is_ome_zarr(raw):
            samples.append(sample)
    return sorted(set(samples), key=str.casefold)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create original-versus-U-Net-filtered figures with physical scale bars."
    )
    parser.add_argument("--dataset", choices=("2d_time", "2d_wga_dapi"))
    parser.add_argument("--inference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--mode", choices=("selected", "all"), default="selected")
    parser.add_argument("--sample")
    parser.add_argument("--samples", nargs="*")
    parser.add_argument(
        "--view",
        choices=("merged", "channels", "all"),
        default="all",
        help="WGA-DAPI layout. For 2d_time, all choices produce the two-panel HADA figure.",
    )
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.8)
    parser.add_argument("--scale-bar-um", type=float, default=2.0)
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        help="Manual X pixel-size override in micrometres per pixel.",
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--original-label", default="Original")
    parser.add_argument("--processed-label", default="After filtering and U-Net")
    parser.add_argument("--non-interactive", action="store_true")
    return parser


def resolve_configuration(args: argparse.Namespace) -> tuple[str, Path, list[str], Path]:
    """Resolve dataset, inference root, samples, and output root."""
    dataset = args.dataset
    if dataset is None:
        if args.non_interactive:
            raise ValueError("--dataset is required in non-interactive mode.")
        datasets = ["2d_time", "2d_wga_dapi"]
        dataset = datasets[choose_number("Choose dataset", datasets)]

    inference_root = args.inference_root
    if inference_root is None:
        roots = discover_inference_roots(dataset)
        if not roots:
            raise FileNotFoundError(
                f"No completed U-Net inference roots were found under {UNET_ROOT} for {dataset}."
            )
        if args.non_interactive:
            raise ValueError("--inference-root is required in non-interactive mode.")
        labels = [str(path.relative_to(PROJECT_ROOT)) for path in roots]
        inference_root = roots[choose_number("Choose inference run", labels)]
    elif not inference_root.is_absolute():
        inference_root = PROJECT_ROOT / inference_root
    inference_root = inference_root.resolve()

    available = discover_samples(dataset, inference_root)
    if not available:
        raise FileNotFoundError(
            f"No matching original and foreground-filtered images were found in {inference_root}."
        )

    requested: list[str]
    if args.mode == "all":
        requested = available
    elif args.samples:
        requested = list(args.samples)
    elif args.sample:
        requested = [args.sample]
    else:
        if args.non_interactive:
            raise ValueError("--sample or --samples is required when --mode selected is used.")
        requested = [available[choose_number("Choose sample", available)]]

    missing = [sample for sample in requested if sample not in available]
    if missing:
        raise FileNotFoundError(
            "The following samples do not have both original and processed images: "
            + ", ".join(missing)
        )

    output_root = args.output_root
    if output_root is None:
        output_root = inference_root / "publication_before_after"
    elif not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    return dataset, inference_root, requested, output_root.resolve()


def main() -> int:
    """Run the plotting workflow."""
    parser = build_parser()
    args = parser.parse_args()

    if not (0 <= args.p_low < args.p_high <= 100):
        parser.error("Percentiles must satisfy 0 <= p-low < p-high <= 100.")
    if args.scale_bar_um <= 0 or args.dpi <= 0:
        parser.error("Scale-bar length and DPI must be positive.")
    if args.pixel_size_um is not None and args.pixel_size_um <= 0:
        parser.error("--pixel-size-um must be positive.")

    dataset, inference_root, samples, output_root = resolve_configuration(args)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | float]] = []
    for sample in samples:
        raw_path = RAW_ROOT / dataset / sample / "image.ome.zarr"
        processed_path = inference_root / sample / "foreground_filtered.ome.zarr"
        output_paths, pixel_size = plot_sample(
            dataset=dataset,
            sample=sample,
            raw_path=raw_path,
            processed_path=processed_path,
            output_root=output_root,
            view=args.view,
            level=args.level,
            time_index=args.time,
            z_index=args.z,
            p_low=args.p_low,
            p_high=args.p_high,
            scale_bar_um=args.scale_bar_um,
            pixel_size_override=args.pixel_size_um,
            output_format=args.format,
            dpi=args.dpi,
            original_label=args.original_label,
            processed_label=args.processed_label,
        )
        print(f"[OK] {sample}")
        for path in output_paths:
            print(f"     {path}")
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "raw_ome_zarr": str(raw_path.resolve()),
                    "processed_ome_zarr": str(processed_path.resolve()),
                    "output_path": str(path),
                    "pixel_size_um_x": pixel_size,
                    "scale_bar_um": args.scale_bar_um,
                    "display_p_low": args.p_low,
                    "display_p_high": args.p_high,
                    "view": args.view,
                }
            )

    manifest_path = output_root / "unet_before_after_figure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "dataset",
            "sample",
            "raw_ome_zarr",
            "processed_ome_zarr",
            "output_path",
            "pixel_size_um_x",
            "scale_bar_um",
            "display_p_low",
            "display_p_high",
            "view",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest: {manifest_path}")
    print(f"Figures created: {len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
