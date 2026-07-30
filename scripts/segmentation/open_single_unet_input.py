"""Open one selected 2D U-Net input image in napari.

The script supports four viewing modes for one individual sample:

``raw``
    Open the stored raw OME-Zarr image without numerical normalization.
``filtered``
    Open the stored intensity-preserving local-threshold OME-Zarr image.
``normalized``
    Open the temporary P1-P99.8 normalized local-threshold image used as the
    actual U-Net input. This normalized array is calculated in memory only and
    is not written to disk.
``comparison``
    Open both the stored filtered image and its exact normalized U-Net input as
    separate napari layers. The normalized layers are initially visible and the
    stored filtered layers are initially hidden, so they can be toggled for
    direct comparison.

Normalization policy
--------------------
For every complete 2D frame and every fluorescence channel independently, the
1st percentile is mapped to 0 and the 99.8th percentile is mapped to 1. Values
outside this interval are clipped to [0, 1]. This matches the 2D U-Net training
and inference preprocessing. It is never applied independently to patches.

Examples
--------
Interactive selection::

    python scripts/segmentation/open_single_unet_input.py

Open one named raw HADA sample::

    python scripts/segmentation/open_single_unet_input.py \
        --dataset 2d_time \
        --source raw \
        --sample WT_HADA_NHS_120min_ROI1_SIM

Open the normalized filtered image used by the U-Net::

    python scripts/segmentation/open_single_unet_input.py \
        --dataset 2d_time \
        --source normalized \
        --sample WT_HADA_NHS_5min_ROI1_SIM²

Open stored filtered and normalized versions together::

    python scripts/segmentation/open_single_unet_input.py \
        --dataset 2d_time \
        --source comparison \
        --sample WT_HADA_NHS_5min_ROI1_SIM²
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
DATASETS = ("2d_time", "2d_wga_dapi")
SOURCES = ("raw", "filtered", "normalized", "comparison")
EXPECTED_FILTER_PERCENTILE = {
    "2d_time": 97.5,
    "2d_wga_dapi": 98.0,
}
NORMALIZATION_LOW_PERCENTILE = 1.0
NORMALIZATION_HIGH_PERCENTILE = 99.8
NORMALIZATION_EPSILON = 1e-8


def find_project_root() -> Path:
    """Locate the repository root containing both ``scripts`` and ``src/PFT``."""
    for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the project root containing scripts and src/PFT")


PROJECT_ROOT = find_project_root()
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from PFT.core_prog_parts.decoder_omezar import (  # noqa: E402
    load_ome_zarr,
    move_yx_to_last,
    normalize_axes,
)


def raw_root(dataset: str) -> Path:
    """Return the canonical raw OME-Zarr root for one 2D dataset."""
    return PROJECT_ROOT / "results" / "img" / dataset


def filtered_root(dataset: str) -> Path:
    """Find the production intensity-preserving local-threshold result folder."""
    base = (
        PROJECT_ROOT
        / "results"
        / "Filters"
        / "Local_high_threshold"
        / dataset
        / "intensity_preserved"
    )
    percentile = EXPECTED_FILTER_PERCENTILE[dataset]
    preferred = base / f"p{percentile:.1f}_k3f40_s370_k10f40"
    if preferred.is_dir():
        return preferred

    candidates = sorted(
        path for path in base.glob(f"p{percentile:.1f}_*") if path.is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No p{percentile:.1f} local-threshold folder found under: {base}"
        )
    raise RuntimeError(
        "Several matching local-threshold folders were found. "
        "Use --root to select one explicitly:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def input_root(dataset: str, source: str, override: Path | None) -> Path:
    """Resolve the OME-Zarr root used by the selected viewing mode."""
    if override is not None:
        return override.expanduser().resolve()
    if source == "raw":
        return raw_root(dataset)
    return filtered_root(dataset)


def discover_samples(root: Path) -> dict[str, Path]:
    """Return ``sample name -> image.ome.zarr`` mappings."""
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    samples = {
        path.parent.name: path
        for path in sorted(root.glob("*/image.ome.zarr"))
        if path.is_dir()
    }
    if not samples:
        raise FileNotFoundError(f"No */image.ome.zarr stores found under: {root}")
    return samples


def prompt_choice(label: str, options: Sequence[str]) -> str:
    """Ask the user to select one numbered terminal option."""
    print(f"\n{label}")
    for index, option in enumerate(options, start=1):
        print(f"  {index:>3}. {option}")
    while True:
        answer = input("Select number: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            print("Enter a valid number.")
            continue
        if 1 <= selected <= len(options):
            return options[selected - 1]
        print(f"Enter a number from 1 to {len(options)}.")


def remove_singleton_nonvisual_axes(
    array: np.ndarray,
    axes: str,
) -> tuple[np.ndarray, str]:
    """Remove singleton axes other than T, C, Y and X before napari display."""
    result = np.asarray(array)
    result_axes = normalize_axes(axes)
    for axis in tuple(result_axes):
        if axis in {"t", "c", "y", "x"}:
            continue
        position = result_axes.index(axis)
        size = int(result.shape[position])
        if size != 1:
            raise ValueError(
                f"Unsupported non-singleton axis {axis}={size}; axes={result_axes}, "
                f"shape={result.shape}"
            )
        result = np.take(result, 0, axis=position)
        result_axes = result_axes[:position] + result_axes[position + 1 :]
    return result, result_axes


def normalize_for_unet(array: np.ndarray, axes: str) -> np.ndarray:
    """Return the exact P1-P99.8 normalized 2D U-Net input.

    The input axes must have Y and X in the final two positions. Every leading
    index combination therefore identifies one complete YX image plane, which
    corresponds to one time frame and one fluorescence channel when those axes
    are present. Each plane is independently converted to float32 and mapped to
    [0, 1]. This is equivalent to channel-wise normalization of every complete
    HWC frame in the training and inference code.
    """
    axes = normalize_axes(axes)
    if not axes.endswith("yx"):
        raise ValueError(
            f"normalize_for_unet requires YX as the final axes; axes={axes}, "
            f"shape={np.asarray(array).shape}"
        )

    source = np.asarray(array, dtype=np.float32)
    result = np.empty_like(source, dtype=np.float32)
    leading_shape = source.shape[:-2]
    leading_indices = np.ndindex(leading_shape) if leading_shape else [()]

    for index in leading_indices:
        plane = source[index] if index else source
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            normalized = np.zeros_like(plane, dtype=np.float32)
        else:
            low = float(np.percentile(plane, NORMALIZATION_LOW_PERCENTILE))
            high = float(np.percentile(plane, NORMALIZATION_HIGH_PERCENTILE))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                normalized = np.zeros_like(plane, dtype=np.float32)
            else:
                normalized = np.clip(
                    (plane - low) / (high - low + NORMALIZATION_EPSILON),
                    0.0,
                    1.0,
                ).astype(np.float32, copy=False)

        if index:
            result[index] = normalized
        else:
            result[...] = normalized

    return result


def channel_metadata(dataset: str, channel_count: int) -> tuple[list[str], list[str]]:
    """Return napari layer names and display colormaps for known datasets."""
    if dataset == "2d_time":
        names = ["HADA"] if channel_count == 1 else [f"HADA ch{n}" for n in range(channel_count)]
        return names, ["blue"] * channel_count

    if channel_count == 2:
        return ["DAPI", "WGA"], ["blue", "green"]
    return [f"channel {n}" for n in range(channel_count)], ["gray"] * channel_count


def array_statistics(array: np.ndarray) -> tuple[float, float]:
    """Return finite minimum and maximum values for terminal reporting."""
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def add_image_layers(
    viewer: object,
    array: np.ndarray,
    axes: str,
    dataset: str,
    layer_prefix: str,
    source_path: Path,
    source_type: str,
    normalized: bool,
    visible: bool = True,
) -> None:
    """Add one array to napari while retaining its channel organization."""
    metadata = {
        "source_path": str(source_path),
        "source_type": source_type,
        "axes": axes,
        "stored_dtype": str(array.dtype),
        "numerically_normalized": normalized,
    }
    if normalized:
        metadata.update(
            {
                "normalization": "percentile",
                "normalization_low_percentile": NORMALIZATION_LOW_PERCENTILE,
                "normalization_high_percentile": NORMALIZATION_HIGH_PERCENTILE,
                "normalization_scope": "complete 2D frame, independently per channel",
            }
        )

    if "c" in axes:
        channel_axis = axes.index("c")
        channel_count = int(array.shape[channel_axis])
        base_names, colormaps = channel_metadata(dataset, channel_count)
        layers = viewer.add_image(
            array,
            channel_axis=channel_axis,
            name=[f"{layer_prefix}: {name}" for name in base_names],
            colormap=colormaps,
            metadata=metadata,
        )
        for layer in layers:
            layer.visible = visible
    else:
        colormap = "blue" if dataset == "2d_time" else "gray"
        layer = viewer.add_image(
            array,
            name=f"{layer_prefix}: {source_path.parent.name}",
            colormap=colormap,
            metadata=metadata,
        )
        layer.visible = visible


def open_in_napari(zarr_path: Path, dataset: str, source: str, level: int) -> None:
    """Load one OME-Zarr store and display the requested representation."""
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError(
            "napari is not installed in the active environment. Install it or activate "
            "the PFT environment that contains napari."
        ) from exc

    stored, axes = load_ome_zarr(zarr_path, level=level, as_numpy=True)
    stored = np.asarray(stored)
    axes = normalize_axes(axes)
    stored, axes = remove_singleton_nonvisual_axes(stored, axes)
    stored, axes = move_yx_to_last(stored, axes)

    needs_normalized = source in {"normalized", "comparison"}
    normalized = normalize_for_unet(stored, axes) if needs_normalized else None

    stored_min, stored_max = array_statistics(stored)
    print("\nOpening one U-Net input image")
    print(f"  Path:   {zarr_path}")
    print(f"  Mode:   {source}")
    print(f"  Axes:   {axes}")
    print(f"  Shape:  {stored.shape}")
    print(f"  Stored dtype: {stored.dtype}")
    print(f"  Stored min:   {stored_min}")
    print(f"  Stored max:   {stored_max}")

    if normalized is not None:
        normalized_min, normalized_max = array_statistics(normalized)
        print("  U-Net normalization: complete frame/channel P1-P99.8 -> [0,1]")
        print(f"  Normalized dtype: {normalized.dtype}")
        print(f"  Normalized min:   {normalized_min}")
        print(f"  Normalized max:   {normalized_max}")
        print("  Normalized data are created in memory and are not saved.")
    else:
        print("  Numerical normalization: none")

    print("  Napari contrast controls affect display only.")
    viewer = napari.Viewer(title=f"{dataset} | {source} | {zarr_path.parent.name}")

    if source == "raw":
        add_image_layers(
            viewer,
            stored,
            axes,
            dataset,
            "raw",
            zarr_path,
            "raw",
            normalized=False,
        )
    elif source == "filtered":
        add_image_layers(
            viewer,
            stored,
            axes,
            dataset,
            "filtered original scale",
            zarr_path,
            "filtered",
            normalized=False,
        )
    elif source == "normalized":
        assert normalized is not None
        add_image_layers(
            viewer,
            normalized,
            axes,
            dataset,
            "normalized U-Net input",
            zarr_path,
            "filtered",
            normalized=True,
        )
    elif source == "comparison":
        assert normalized is not None
        add_image_layers(
            viewer,
            stored,
            axes,
            dataset,
            "filtered original scale",
            zarr_path,
            "filtered",
            normalized=False,
            visible=False,
        )
        add_image_layers(
            viewer,
            normalized,
            axes,
            dataset,
            "normalized U-Net input",
            zarr_path,
            "filtered",
            normalized=True,
            visible=True,
        )
    else:  # Defensive branch for programmatic use.
        raise ValueError(f"Unsupported source mode: {source}")

    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open one selected raw, filtered, or P1-P99.8 normalized 2D U-Net "
            "input image in napari."
        )
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument(
        "--source",
        choices=SOURCES,
        help=(
            "raw; filtered in original scale; normalized temporary U-Net input; "
            "or comparison with filtered and normalized layers"
        ),
    )
    parser.add_argument("--sample", help="Exact sample-folder name.")
    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "Optional custom root containing <sample>/image.ome.zarr folders. "
            "For normalized/comparison modes this must point to filtered data."
        ),
    )
    parser.add_argument("--level", type=int, default=0, help="OME-Zarr pyramid level.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching samples and exit without opening napari.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset = args.dataset or prompt_choice("Choose dataset", DATASETS)
    source = args.source or prompt_choice(
        "Choose image type",
        (
            "raw",
            "filtered",
            "normalized",
            "comparison",
        ),
    )
    root = input_root(dataset, source, args.root)
    samples = discover_samples(root)

    if args.list:
        print(f"\nAvailable {dataset}/{source} samples under:\n{root}\n")
        for name in samples:
            print(name)
        return 0

    if args.sample:
        sample = args.sample
        if sample not in samples:
            available = "\n  ".join(samples)
            raise FileNotFoundError(
                f"Sample not found: {sample}\nAvailable samples:\n  {available}"
            )
    else:
        sample = prompt_choice("Choose one sample", list(samples))

    open_in_napari(samples[sample], dataset, source, args.level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
