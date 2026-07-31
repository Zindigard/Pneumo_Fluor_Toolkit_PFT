"""
Create manual instance masks in Napari for prepared segmentation inputs.

* 0 = background
* 1 = first cell
* 2 = second cell
* ...

    results/segmentation_inputs/<dataset>/<source_mode>/<sample>/
        segmentation_input.ome.zarr


    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/
        mask.tif
        annotation_report.json
        annotation_overlay.png


    python scripts/segmentation/labeling_segmentation.py `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --sample "WT_HADA_NHS_20min_ROI1_SIM"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile as tiff
from skimage.segmentation import find_boundaries

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr  # noqa: E402
from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
    segmentation_input_root,
    segmentation_mask_root,
)


def _choose(title: str, values: Sequence[str]) -> str:
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    selected = int(input("Select number: ").strip())
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _list_inputs(root: Path) -> list[Path]:
    return sorted(root.glob("**/segmentation_input.ome.zarr"))


def _sample_key(path: Path, input_root: Path) -> str:
    return path.parent.relative_to(input_root).as_posix()


def _load_prepared_yxc(path: Path) -> tuple[np.ndarray, str]:
    array, axes = load_ome_zarr(path, level=0, as_numpy=True)
    image = np.asarray(array, dtype=np.float32)
    axes = str(axes).lower()
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Prepared input contains non-finite values: {path}")
    if image.size == 0 or float(np.min(image)) < 0.0 or float(np.max(image)) > 1.0:
        raise ValueError(f"Prepared input must be in [0,1]: {path}")

    # Remove only singleton non-spatial dimensions. Annotation is explicitly 2D.
    for axis in tuple(axes):
        if axis not in {"c", "y", "x"}:
            index = axes.index(axis)
            if image.shape[index] != 1:
                raise ValueError(
                    f"Manual instance labeling requires one 2D frame. "
                    f"Received axis {axis}={image.shape[index]} in {path}."
                )
            image = np.take(image, 0, axis=index)
            axes = axes[:index] + axes[index + 1 :]

    if axes == "yx":
        return image[..., None], axes
    if "c" not in axes or "y" not in axes or "x" not in axes:
        raise ValueError(f"Expected YX or CYX prepared input, received axes={axes}")
    permutation = [axes.index("y"), axes.index("x"), axes.index("c")]
    image = np.transpose(image, permutation)
    if image.ndim != 3:
        raise ValueError(f"Expected YXC image, received {image.shape}")
    return image.astype(np.float32, copy=False), axes


def _display_rgb(image_yxc: np.ndarray, dataset: str) -> np.ndarray:
    """Build a display-only RGB composite from the normalized numerical channels."""
    channels = image_yxc.shape[-1]
    rgb = np.zeros((*image_yxc.shape[:2], 3), dtype=np.float32)
    if dataset == "2d_time" or channels == 1:
        rgb[..., 2] = image_yxc[..., 0]
    elif dataset == "2d_wga_dapi":
        rgb[..., 2] = image_yxc[..., 0]  # DAPI blue
        rgb[..., 1] = image_yxc[..., min(1, channels - 1)]  # WGA green
    else:
        # 3D SIM MIP channel order used by the current workflow: blue, green, red.
        rgb[..., 2] = image_yxc[..., 0]
        if channels > 1:
            rgb[..., 1] = image_yxc[..., 1]
        if channels > 2:
            rgb[..., 0] = image_yxc[..., 2]
    maximum = float(np.max(rgb))
    if maximum > 1.0:
        rgb = rgb / maximum
    return np.clip(rgb, 0.0, 1.0)


def _load_existing_mask(mask_path: Path, expected_yx: tuple[int, int]) -> np.ndarray:
    if not mask_path.is_file():
        return np.zeros(expected_yx, dtype=np.uint16)
    mask = np.asarray(tiff.imread(mask_path))
    mask = np.squeeze(mask)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_yx):
        raise ValueError(
            f"Existing mask shape {mask.shape} does not match input YX {expected_yx}: {mask_path}"
        )
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError(f"Existing instance mask must contain non-negative integers: {mask_path}")
    return mask.astype(np.uint32 if np.max(mask) > 65535 else np.uint16, copy=False)


def _annotate_napari(
    image_yxc: np.ndarray,
    initial_mask: np.ndarray,
    *,
    dataset: str,
    title: str,
    brush_size: int,
) -> np.ndarray:
    import napari

    viewer = napari.Viewer(title=title)
    rgb = _display_rgb(image_yxc, dataset)
    viewer.add_image(rgb, name="normalized segmentation input (display composite)", rgb=True)

    if image_yxc.shape[-1] == 1:
        viewer.add_image(
            image_yxc[..., 0],
            name="channel 0",
            colormap="blue",
            contrast_limits=(0.0, 1.0),
            visible=False,
        )
    else:
        names = (
            ("DAPI", "blue"),
            ("WGA", "green"),
        ) if dataset == "2d_wga_dapi" else (
            ("blue channel", "blue"),
            ("green channel", "green"),
            ("red channel", "red"),
        )
        for channel in range(image_yxc.shape[-1]):
            name, colormap = names[min(channel, len(names) - 1)]
            viewer.add_image(
                image_yxc[..., channel],
                name=f"channel {channel}: {name}",
                colormap=colormap,
                contrast_limits=(0.0, 1.0),
                visible=False,
            )

    labels_layer = viewer.add_labels(
        initial_mask,
        name="cell instances: unique ID per cell",
        opacity=0.45,
    )
    labels_layer.brush_size = int(brush_size)
    labels_layer.selected_label = max(1, int(np.max(initial_mask)) + 1)

    print(
        "\nNapari instance-labeling instructions\n"
        "------------------------------------\n"
        "Assign every cell a unique positive integer label.\n"
        "Do not paint all cells with label 1.\n"
        "Keep background at label 0.\n"
        "Use the label picker or change Selected label before painting another cell.\n"
        "The displayed image is already normalized; do not export a separately normalized copy.\n"
        "Close the Napari window after completing or correcting the mask.\n"
    )
    napari.run()
    result = np.asarray(labels_layer.data)
    if result.ndim != 2 or tuple(result.shape) != tuple(image_yxc.shape[:2]):
        raise ValueError(f"Napari returned an invalid label shape: {result.shape}")
    if np.any(result < 0):
        raise ValueError("Napari mask contains negative labels")
    return result.astype(np.uint32 if np.max(result) > 65535 else np.uint16)


def _save_overlay(image_yxc: np.ndarray, mask: np.ndarray, output_png: Path, dataset: str) -> None:
    import matplotlib.pyplot as plt

    rgb = _display_rgb(image_yxc, dataset)
    overlay = np.array(rgb, copy=True)
    boundaries = find_boundaries(mask, mode="outer")
    overlay[boundaries] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_png, np.clip(overlay, 0.0, 1.0))


def annotate_one(
    input_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    dataset: str,
    source_mode: str,
    brush_size: int,
) -> Path:
    sample_key = _sample_key(input_path, input_root)
    image_yxc, source_axes = _load_prepared_yxc(input_path)
    sample_dir = output_root / Path(sample_key)
    sample_dir.mkdir(parents=True, exist_ok=True)
    mask_path = sample_dir / "mask.tif"
    initial = _load_existing_mask(mask_path, image_yxc.shape[:2])
    result = _annotate_napari(
        image_yxc,
        initial,
        dataset=dataset,
        title=f"PFT instance labeling | {dataset} | {source_mode} | {sample_key}",
        brush_size=brush_size,
    )

    positive_ids = np.unique(result[result > 0])
    if positive_ids.size == 0:
        raise RuntimeError("Annotation contains no positive cell instance")
    if positive_ids.size == 1 and int(positive_ids[0]) == 1:
        print(
            "WARNING: the mask contains only label 1. This is valid only when exactly one "
            "cell is present; otherwise assign a unique integer to every cell."
        )

    tiff.imwrite(mask_path, result, photometric="minisblack")
    overlay_path = sample_dir / "annotation_overlay.png"
    _save_overlay(image_yxc, result, overlay_path, dataset)
    report = {
        "dataset": dataset,
        "source_mode": source_mode,
        "sample_key": sample_key,
        "segmentation_input": str(input_path),
        "source_axes": source_axes,
        "input_shape_yxc": list(image_yxc.shape),
        "input_dtype": str(image_yxc.dtype),
        "input_range": [float(np.min(image_yxc)), float(np.max(image_yxc))],
        "mask_path": str(mask_path),
        "mask_dtype": str(result.dtype),
        "mask_shape_yx": list(result.shape),
        "instance_count": int(positive_ids.size),
        "maximum_label": int(np.max(result)),
        "background_label": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (sample_dir / "annotation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return mask_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manually create unique-cell instance masks in Napari.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--sample", default=None, help="Exact sample key")
    parser.add_argument("--next-missing", action="store_true")
    parser.add_argument("--brush-size", type=int, default=15)
    args = parser.parse_args()

    dataset = args.dataset or _choose("Choose independent dataset", SUPPORTED_DATASETS)
    valid_modes = SOURCE_MODES_BY_DATASET[dataset]
    source_mode = args.source_mode or _choose("Choose segmentation source", valid_modes)
    if source_mode not in valid_modes:
        parser.error(f"{source_mode!r} is not valid for {dataset!r}: {valid_modes}")

    input_root = segmentation_input_root(PROJECT_ROOT, dataset, source_mode)
    output_root = segmentation_mask_root(PROJECT_ROOT, dataset, source_mode)
    inputs = _list_inputs(input_root)
    if not inputs:
        raise FileNotFoundError(
            f"No prepared inputs found below {input_root}. Run prepare_segmentation_inputs.py first."
        )
    by_key = {_sample_key(path, input_root): path for path in inputs}

    if args.sample:
        if args.sample not in by_key:
            parser.error(f"Unknown sample key: {args.sample}")
        chosen_key = args.sample
    elif args.next_missing:
        missing = [
            key for key in by_key
            if not (output_root / Path(key) / "mask.tif").is_file()
        ]
        if not missing:
            print("All prepared inputs already have a manual instance mask.")
            return 0
        chosen_key = missing[0]
    else:
        labels = []
        keys = list(by_key)
        for key in keys:
            status = "mask exists" if (output_root / Path(key) / "mask.tif").is_file() else "missing"
            labels.append(f"{key} [{status}]")
        selected_label = _choose("Choose sample to annotate", labels)
        chosen_key = keys[labels.index(selected_label)]

    mask_path = annotate_one(
        by_key[chosen_key],
        input_root=input_root,
        output_root=output_root,
        dataset=dataset,
        source_mode=source_mode,
        brush_size=args.brush_size,
    )
    print(f"\nSaved manual instance mask:\n{mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
