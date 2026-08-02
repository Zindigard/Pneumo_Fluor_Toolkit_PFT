r"""Create manual Cellpose/Omnipose instance masks in Napari.

Two annotation modes are available.

``full_image``
    Opens the complete prepared segmentation input and saves one full-size
    instance mask. Use this mode only when every true cell in the image is
    annotated.

``crop_then_label``
    Opens the complete image first and asks the user to draw one rectangular
    crop. The selected crop is then opened in a second Napari window for
    instance labeling. The crop image and its mask are saved as an independent
    training or validation pair. Cells outside the selected crop are not part of
    the training pair and therefore are not interpreted as background.

Every bacterial cell must receive a unique positive integer label::

    0 = background
    1 = first cell
    2 = second cell
    ...

The displayed image is the prepared normalized float32 segmentation input. It
is not normalized again and the original OME-Zarr is never modified.

Full-image outputs
------------------

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/
        mask.tif
        annotation_report.json
        annotation_overlay.png

Crop outputs
------------

    results/training_files/segmentation/<dataset>/<source_mode>/<sample>/
        crops/<train|validation>/<crop_id>/
            segmentation_input.ome.zarr
            mask.tif
            annotation_report.json
            annotation_overlay.png
            crop_preview.png
        crops/crop_manifest.csv

Interactive use
---------------

    python scripts/segmentation/labeling_segmentation.py

Open a specific sample in crop mode
-----------------------------------

    python scripts/segmentation/labeling_segmentation.py `
        --annotation-mode crop_then_label `
        --crop-split train `
        --dataset 2d_time `
        --source-mode filtered_unet `
        --sample "WT_HADA_NHS_20min_ROI1_SIM"


Examples
--------
Interactive annotation selection:

    python scripts/segmentation/labeling_segmentation.py

Create a training crop for one prepared HADA sample:

    python scripts/segmentation/labeling_segmentation.py \
        --annotation-mode crop_then_label \
        --crop-split train \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --sample WT_HADA_NHS_40min_ROI1_SIM \
        --minimum-crop-size 64 \
        --brush-size 15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile as tiff
from skimage.segmentation import find_boundaries

SCRIPT_FILE = Path(__file__).resolve()
ANNOTATION_MODES: tuple[str, ...] = ("full_image", "crop_then_label")
CROP_SPLITS: tuple[str, ...] = ("train", "validation")


def _project_root() -> Path:
    """Return project root for the supplied inputs.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _project_root()
    """
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr  # noqa: E402
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr  # noqa: E402
from PFT.core_prog_parts.segmentation.segmentation_input_core import (  # noqa: E402
    SOURCE_MODES_BY_DATASET,
    SUPPORTED_DATASETS,
    segmentation_input_root,
    segmentation_mask_root,
)


def _choose(title: str, values: Sequence[str]) -> str:
    """Choose the requested operation according to the configured criteria.

    Args:
        title (str): Title displayed on the generated figure or report section.
        values (Sequence[str]): Text value specifying values.

    Returns:
        str: Generated or resolved text value.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _choose(title="title", values="values")
    """
    if not values:
        raise ValueError(f"No values are available for: {title}")
    print(f"\n{title}")
    for index, value in enumerate(values, start=1):
        print(f"  [{index}] {value}")
    selected = int(input("Select number: ").strip())
    if not 1 <= selected <= len(values):
        raise ValueError(f"Selection must be between 1 and {len(values)}")
    return values[selected - 1]


def _list_inputs(root: Path) -> list[Path]:
    """List inputs available in the configured project structure.

    Args:
        root (Path): Root directory used to resolve relative project paths.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _list_inputs(root=Path("path/to/resource"))
    """
    return sorted(root.glob("**/segmentation_input.ome.zarr"))


def _sample_key(path: Path, input_root: Path) -> str:
    """Return sample key for the supplied inputs.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        input_root (Path): Directory used for input.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _sample_key(path=Path("path/to/resource"), input_root=Path("path/to/resource"))
    """
    return path.parent.relative_to(input_root).as_posix()


def _load_prepared_yxc(path: Path) -> tuple[np.ndarray, str]:
    """Load prepared yxc from persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_prepared_yxc(path=Path("path/to/resource"))
    """
    array, axes = load_ome_zarr(path, level=0, as_numpy=True)
    image = np.asarray(array, dtype=np.float32)
    axes = str(axes).lower()
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Prepared input contains non-finite values: {path}")
    if image.size == 0 or float(np.min(image)) < 0.0 or float(np.max(image)) > 1.0:
        raise ValueError(f"Prepared input must be finite and in [0,1]: {path}")

    # Remove only singleton non-spatial dimensions. Annotation is explicitly 2D.
    for axis in tuple(axes):
        if axis not in {"c", "y", "x"}:
            index = axes.index(axis)
            if image.shape[index] != 1:
                raise ValueError(
                    "Manual instance labeling requires one 2D frame. "
                    f"Received axis {axis}={image.shape[index]} in {path}."
                )
            image = np.take(image, 0, axis=index)
            axes = axes[:index] + axes[index + 1 :]

    if axes == "yx":
        return image[..., None].astype(np.float32, copy=False), axes
    if "c" not in axes or "y" not in axes or "x" not in axes:
        raise ValueError(f"Expected YX or CYX prepared input, received axes={axes}")
    permutation = [axes.index("y"), axes.index("x"), axes.index("c")]
    image = np.transpose(image, permutation)
    if image.ndim != 3:
        raise ValueError(f"Expected YXC image, received {image.shape}")
    return image.astype(np.float32, copy=False), axes


def _display_rgb(image_yxc: np.ndarray, dataset: str) -> np.ndarray:
    """Build a display-only RGB composite from normalized numerical channels.

    Args:
        image_yxc (np.ndarray): Array containing image yxc.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = _display_rgb(image_yxc=image_array, dataset="2d_time")
    """
    channels = image_yxc.shape[-1]
    rgb = np.zeros((*image_yxc.shape[:2], 3), dtype=np.float32)
    if dataset == "2d_time" or channels == 1:
        rgb[..., 2] = image_yxc[..., 0]
    elif dataset == "2d_wga_dapi":
        rgb[..., 2] = image_yxc[..., 0]  # DAPI blue
        rgb[..., 1] = image_yxc[..., min(1, channels - 1)]  # WGA green
    else:
        # Current 3D SIM MIP display order: blue, green, red.
        rgb[..., 2] = image_yxc[..., 0]
        if channels > 1:
            rgb[..., 1] = image_yxc[..., 1]
        if channels > 2:
            rgb[..., 0] = image_yxc[..., 2]
    return np.clip(rgb, 0.0, 1.0)


def _load_existing_mask(mask_path: Path, expected_yx: tuple[int, int]) -> np.ndarray:
    """Load existing mask from persistent storage.

    Args:
        mask_path (Path): Filesystem path associated with mask.
        expected_yx (tuple[int, int]): Numerical value controlling expected yx.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _load_existing_mask(mask_path=Path("path/to/resource"), expected_yx=1)
    """
    if not mask_path.is_file():
        return np.zeros(expected_yx, dtype=np.uint16)
    mask = np.asarray(tiff.imread(mask_path))
    mask = np.squeeze(mask)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_yx):
        raise ValueError(
            f"Existing mask shape {mask.shape} does not match input YX {expected_yx}: "
            f"{mask_path}"
        )
    if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError(
            f"Existing instance mask must contain non-negative integers: {mask_path}"
        )
    return mask.astype(np.uint32 if np.max(mask) > 65535 else np.uint16, copy=False)


def _add_channel_layers(viewer: Any, image_yxc: np.ndarray, dataset: str) -> None:
    """Add hidden numerical channel layers for detailed inspection.

    Args:
        viewer (Any): Napari viewer instance associated with the current graphical operation.
        image_yxc (np.ndarray): Array containing image yxc.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Example:
        >>> _add_channel_layers(
        ...     viewer=...,
        ...     image_yxc=image_array,
        ...     dataset="2d_time",
        ... )
    """
    if image_yxc.shape[-1] == 1:
        viewer.add_image(
            image_yxc[..., 0],
            name="channel 0",
            colormap="blue",
            contrast_limits=(0.0, 1.0),
            visible=False,
        )
        return

    names = (
        (("DAPI", "blue"), ("WGA", "green"))
        if dataset == "2d_wga_dapi"
        else (("blue channel", "blue"), ("green channel", "green"), ("red channel", "red"))
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


def _select_crop_napari(
    image_yxc: np.ndarray,
    *,
    dataset: str,
    title: str,
    minimum_size: int,
) -> tuple[int, int, int, int]:
    """Let the user draw one rectangle and return integer YX crop bounds.

    Args:
        image_yxc (np.ndarray): Array containing image yxc.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        title (str): Title displayed on the generated figure or report section.
        minimum_size (int): Size parameter controlling minimum.

    Returns:
        tuple[int, int, int, int]: Collection containing the generated or selected values.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _select_crop_napari(
        ...     image_yxc=image_array,
        ...     dataset="2d_time",
        ...     title="title",
        ...     minimum_size=1,
        ... )
    """
    import napari

    viewer = napari.Viewer(title=title)
    viewer.add_image(
        _display_rgb(image_yxc, dataset),
        name="complete normalized segmentation input",
        rgb=True,
    )
    _add_channel_layers(viewer, image_yxc, dataset)
    crop_layer = viewer.add_shapes(
        name="DRAW ONE RECTANGULAR CROP",
        shape_type="rectangle",
        edge_color="yellow",
        face_color=[1.0, 1.0, 0.0, 0.08],
        edge_width=3,
    )
    crop_layer.mode = "add_rectangle"

    print(
        "\nCrop-selection instructions\n"
        "---------------------------\n"
        "Draw one rectangular crop around the region that you intend to annotate.\n"
        "Choose a region containing representative cells and avoid cutting through cells "
        "where possible.\n"
        "Only the last rectangle is used when more than one rectangle is present.\n"
        "Close the Napari window after drawing the rectangle.\n"
    )
    napari.run()

    if len(crop_layer.data) == 0:
        raise RuntimeError("No crop rectangle was drawn")
    if len(crop_layer.data) > 1:
        print("WARNING: multiple crop rectangles were found; the last rectangle is used.")

    vertices = np.asarray(crop_layer.data[-1], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise ValueError(f"Invalid Napari rectangle coordinates: {vertices.shape}")

    height, width = image_yxc.shape[:2]
    y0 = max(0, int(np.floor(np.min(vertices[:, 0]))))
    y1 = min(height, int(np.ceil(np.max(vertices[:, 0]))))
    x0 = max(0, int(np.floor(np.min(vertices[:, 1]))))
    x1 = min(width, int(np.ceil(np.max(vertices[:, 1]))))
    if y1 - y0 < minimum_size or x1 - x0 < minimum_size:
        raise ValueError(
            f"Selected crop is too small: {(y1 - y0, x1 - x0)}. "
            f"Minimum size is {minimum_size} x {minimum_size} pixels."
        )
    return y0, y1, x0, x1


def _annotate_napari(
    image_yxc: np.ndarray,
    initial_mask: np.ndarray,
    *,
    dataset: str,
    title: str,
    brush_size: int,
    annotation_scope: str,
) -> np.ndarray:
    """Return annotate napari for the supplied inputs.

    Args:
        image_yxc (np.ndarray): Array containing image yxc.
        initial_mask (np.ndarray): Array containing initial mask.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        title (str): Title displayed on the generated figure or report section.
        brush_size (int): Size parameter controlling brush.
        annotation_scope (str): Text value specifying annotation scope.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _annotate_napari(
        ...     image_yxc=image_array,
        ...     initial_mask=image_array,
        ...     dataset="2d_time",
        ...     title="title",
        ...     brush_size=1,
        ...     annotation_scope="annotation_scope",
        ... )
    """
    import napari

    viewer = napari.Viewer(title=title)
    viewer.add_image(
        _display_rgb(image_yxc, dataset),
        name="normalized segmentation input (display composite)",
        rgb=True,
    )
    _add_channel_layers(viewer, image_yxc, dataset)

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
        f"Annotation scope: {annotation_scope}\n"
        "Assign every true cell in this annotation scope a unique positive integer label.\n"
        "Paint the complete visible cell area, not only its fluorescent contour.\n"
        "Do not paint debris or biologically ambiguous signal.\n"
        "Do not paint all cells with label 1. Keep background at label 0.\n"
        "Close the Napari window after completing or correcting the mask.\n"
    )
    napari.run()
    result = np.asarray(labels_layer.data)
    if result.ndim != 2 or tuple(result.shape) != tuple(image_yxc.shape[:2]):
        raise ValueError(f"Napari returned an invalid label shape: {result.shape}")
    if np.any(result < 0):
        raise ValueError("Napari mask contains negative labels")
    return result.astype(np.uint32 if np.max(result) > 65535 else np.uint16)


def _save_overlay(
    image_yxc: np.ndarray,
    mask: np.ndarray,
    output_png: Path,
    dataset: str,
) -> None:
    """Save overlay to persistent storage.

    Args:
        image_yxc (np.ndarray): Array containing image yxc.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        output_png (Path): Filesystem path used for output PNG image.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Example:
        >>> _save_overlay(
        ...     image_yxc=image_array,
        ...     mask=image_array,
        ...     output_png=Path("path/to/resource"),
        ...     dataset="2d_time",
        ... )
    """
    import matplotlib.pyplot as plt

    rgb = _display_rgb(image_yxc, dataset)
    overlay = np.array(rgb, copy=True)
    boundaries = find_boundaries(mask, mode="outer")
    overlay[boundaries] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_png, np.clip(overlay, 0.0, 1.0))


def _save_crop_preview(image_yxc: np.ndarray, output_png: Path, dataset: str) -> None:
    """Save crop preview to persistent storage.

    Args:
        image_yxc (np.ndarray): Array containing image yxc.
        output_png (Path): Filesystem path used for output PNG image.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Example:
        >>> _save_crop_preview(
        ...     image_yxc=image_array,
        ...     output_png=Path("path/to/resource"),
        ...     dataset="2d_time",
        ... )
    """
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_png, _display_rgb(image_yxc, dataset))


def _validate_annotation(result: np.ndarray) -> np.ndarray:
    """Validate annotation against the required constraints.

    Args:
        result (np.ndarray): Array containing result.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _validate_annotation(result=image_array)
    """
    positive_ids = np.unique(result[result > 0])
    if positive_ids.size == 0:
        raise RuntimeError("Annotation contains no positive cell instance")
    if positive_ids.size == 1 and int(positive_ids[0]) == 1:
        print(
            "WARNING: the mask contains only label 1. This is valid only when exactly "
            "one cell is present; otherwise assign a unique integer to every cell."
        )
    return positive_ids


def _save_crop_input(
    crop_yxc: np.ndarray,
    output_zarr: Path,
    *,
    dataset: str,
    source_mode: str,
    sample_key: str,
    split: str,
    crop_id: str,
    bounds_yx: tuple[int, int, int, int],
    source_input: Path,
) -> Path:
    """Save crop input to persistent storage.

    Args:
        crop_yxc (np.ndarray): Array containing crop yxc.
        output_zarr (Path): Filesystem path used for output Zarr.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        sample_key (str): Canonical relative identifier of a sample within the selected dataset and source mode.
        split (str): Text value specifying split.
        crop_id (str): Text value specifying crop id.
        bounds_yx (tuple[int, int, int, int]): Numerical value controlling bounds yx.
        source_input (Path): Filesystem path used for source input.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _save_crop_input(
        ...     crop_yxc=image_array,
        ...     output_zarr=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     sample_key="sample_key",
        ...     split="split",
        ...     crop_id="crop_id",
        ...     bounds_yx=1,
        ...     source_input=Path("path/to/resource"),
        ... )
    """
    if crop_yxc.shape[-1] == 1:
        output_array = crop_yxc[..., 0].astype(np.float32, copy=False)
        output_axes = "yx"
    else:
        output_array = np.moveaxis(crop_yxc, -1, 0).astype(np.float32, copy=False)
        output_axes = "cyx"

    y0, y1, x0, x1 = bounds_yx
    processing = {
        "operation": "manual_annotation_crop",
        "dataset": dataset,
        "source_mode": source_mode,
        "sample_key": sample_key,
        "split": split,
        "crop_id": crop_id,
        "source_segmentation_input": str(source_input),
        "crop_bounds_yx": {
            "y_start_inclusive": y0,
            "y_end_exclusive": y1,
            "x_start_inclusive": x0,
            "x_end_exclusive": x1,
        },
        "stored_dtype": "float32",
        "stored_range": [0.0, 1.0],
        "additional_normalization_applied": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return Path(
        save_ome_zarr(
            output_zarr,
            output_array,
            output_axes,
            overwrite=True,
            pyramid_3d=False,
            pyramid_max_layer=0,
            extra_attrs={"pft_processing": processing},
        )
    )


def _next_crop_id(split_root: Path) -> str:
    """Return next crop id for the supplied inputs.

    Args:
        split_root (Path): Directory used for split.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = _next_crop_id(split_root=Path("path/to/resource"))
    """
    existing_numbers: list[int] = []
    for path in split_root.glob("crop_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("crop_")
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    return f"crop_{max(existing_numbers, default=0) + 1:03d}"


def _existing_crop_choices(sample_dir: Path) -> list[tuple[str, Path, str]]:
    """Return existing crop choices for the supplied inputs.

    Args:
        sample_dir (Path): Directory used for sample.

    Returns:
        list[tuple[str, Path, str]]: Resolved or generated filesystem path.

    Example:
        >>> result = _existing_crop_choices(sample_dir=Path("path/to/resource"))
    """
    choices: list[tuple[str, Path, str]] = []
    crops_root = sample_dir / "crops"
    for split in CROP_SPLITS:
        for crop_dir in sorted((crops_root / split).glob("crop_*")):
            input_zarr = crop_dir / "segmentation_input.ome.zarr"
            if input_zarr.is_dir():
                choices.append((f"{split}/{crop_dir.name}", crop_dir, split))
    return choices


def _write_crop_manifest(sample_dir: Path, row: dict[str, Any]) -> Path:
    """Write crop manifest to persistent storage.

    Args:
        sample_dir (Path): Directory used for sample.
        row (dict[str, Any]): Text value specifying row.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _write_crop_manifest(sample_dir=Path("path/to/resource"), row="row")
    """
    manifest = sample_dir / "crops" / "crop_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if manifest.is_file():
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    serialized = {key: str(value) for key, value in row.items()}
    identity = (serialized["split"], serialized["crop_id"])
    rows = [
        old for old in rows
        if (old.get("split", ""), old.get("crop_id", "")) != identity
    ]
    rows.append(serialized)
    fieldnames = list(serialized.keys())
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def annotate_full_image(
    input_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    dataset: str,
    source_mode: str,
    brush_size: int,
) -> Path:
    """Return annotate full image for the supplied inputs.

    Args:
        input_path (Path): Filesystem path to the input resource.
        input_root (Path): Directory used for input.
        output_root (Path): Directory used for output.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        brush_size (int): Size parameter controlling brush.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = annotate_full_image(
        ...     input_path=Path("path/to/resource"),
        ...     input_root=Path("path/to/resource"),
        ...     output_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     brush_size=1,
        ... )
    """
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
        title=f"PFT full-image labeling | {dataset} | {sample_key}",
        brush_size=brush_size,
        annotation_scope="complete image; annotate every true cell in the image",
    )
    positive_ids = _validate_annotation(result)

    tiff.imwrite(mask_path, result, photometric="minisblack")
    overlay_path = sample_dir / "annotation_overlay.png"
    _save_overlay(image_yxc, result, overlay_path, dataset)
    report = {
        "annotation_mode": "full_image",
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
        "all_true_cells_in_scope_must_be_annotated": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (sample_dir / "annotation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return mask_path


def annotate_crop(
    input_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    dataset: str,
    source_mode: str,
    brush_size: int,
    split: str,
    crop_id: str | None,
    minimum_crop_size: int,
) -> Path:
    """Return annotate crop for the supplied inputs.

    Args:
        input_path (Path): Filesystem path to the input resource.
        input_root (Path): Directory used for input.
        output_root (Path): Directory used for output.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        source_mode (str): Preprocessing source used to construct the input, such as the original image, a filtered image, or a U-Net-masked image.
        brush_size (int): Size parameter controlling brush.
        split (str): Text value specifying split.
        crop_id (str | None): Text value specifying crop id.
        minimum_crop_size (int): Size parameter controlling minimum crop.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = annotate_crop(
        ...     input_path=Path("path/to/resource"),
        ...     input_root=Path("path/to/resource"),
        ...     output_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ...     source_mode="original",
        ...     brush_size=1,
        ...     split="split",
        ...     crop_id="crop_id",
        ...     minimum_crop_size=1,
        ... )
    """
    sample_key = _sample_key(input_path, input_root)
    sample_dir = output_root / Path(sample_key)
    sample_dir.mkdir(parents=True, exist_ok=True)
    full_image_yxc, source_axes = _load_prepared_yxc(input_path)

    selected_crop_dir: Path | None = None
    selected_split = split
    existing = _existing_crop_choices(sample_dir)

    if crop_id:
        matches = [
            (label, directory, existing_split)
            for label, directory, existing_split in existing
            if directory.name == crop_id and existing_split == split
        ]
        if not matches:
            raise FileNotFoundError(
                f"Existing crop {split}/{crop_id} was not found below {sample_dir / 'crops'}"
            )
        selected_crop_dir = matches[0][1]
    elif existing:
        choices = ["create_new_crop"] + [label for label, _, _ in existing]
        action = _choose("Create a new crop or reopen an existing crop", choices)
        if action != "create_new_crop":
            match = next(item for item in existing if item[0] == action)
            selected_crop_dir = match[1]
            selected_split = match[2]

    if selected_crop_dir is not None:
        crop_input_path = selected_crop_dir / "segmentation_input.ome.zarr"
        crop_yxc, crop_source_axes = _load_prepared_yxc(crop_input_path)
        report_path = selected_crop_dir / "annotation_report.json"
        old_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        coordinates = old_report.get("crop_bounds_yx", {})
        y0 = int(coordinates.get("y_start_inclusive", 0))
        y1 = int(coordinates.get("y_end_exclusive", crop_yxc.shape[0]))
        x0 = int(coordinates.get("x_start_inclusive", 0))
        x1 = int(coordinates.get("x_end_exclusive", crop_yxc.shape[1]))
        selected_crop_id = selected_crop_dir.name
    else:
        y0, y1, x0, x1 = _select_crop_napari(
            full_image_yxc,
            dataset=dataset,
            title=f"PFT crop selection | {dataset} | {sample_key}",
            minimum_size=minimum_crop_size,
        )
        crop_yxc = full_image_yxc[y0:y1, x0:x1, :].copy()
        split_root = sample_dir / "crops" / selected_split
        selected_crop_id = crop_id or _next_crop_id(split_root)
        selected_crop_dir = split_root / selected_crop_id
        selected_crop_dir.mkdir(parents=True, exist_ok=True)
        crop_input_path = _save_crop_input(
            crop_yxc,
            selected_crop_dir / "segmentation_input.ome.zarr",
            dataset=dataset,
            source_mode=source_mode,
            sample_key=sample_key,
            split=selected_split,
            crop_id=selected_crop_id,
            bounds_yx=(y0, y1, x0, x1),
            source_input=input_path,
        )
        crop_source_axes = "yx" if crop_yxc.shape[-1] == 1 else "cyx"
        _save_crop_preview(crop_yxc, selected_crop_dir / "crop_preview.png", dataset)

    mask_path = selected_crop_dir / "mask.tif"
    initial = _load_existing_mask(mask_path, crop_yxc.shape[:2])
    result = _annotate_napari(
        crop_yxc,
        initial,
        dataset=dataset,
        title=(
            f"PFT crop labeling | {dataset} | {sample_key} | "
            f"{selected_split}/{selected_crop_id}"
        ),
        brush_size=brush_size,
        annotation_scope=(
            "selected crop only; annotate every true cell inside this crop. "
            "Cells outside the crop are excluded from the training pair"
        ),
    )
    positive_ids = _validate_annotation(result)

    tiff.imwrite(mask_path, result, photometric="minisblack")
    overlay_path = selected_crop_dir / "annotation_overlay.png"
    _save_overlay(crop_yxc, result, overlay_path, dataset)
    report = {
        "annotation_mode": "crop_then_label",
        "dataset": dataset,
        "source_mode": source_mode,
        "sample_key": sample_key,
        "split": selected_split,
        "crop_id": selected_crop_id,
        "source_segmentation_input": str(input_path),
        "crop_segmentation_input": str(crop_input_path),
        "source_axes": source_axes,
        "crop_source_axes": crop_source_axes,
        "source_shape_yxc": list(full_image_yxc.shape),
        "crop_shape_yxc": list(crop_yxc.shape),
        "crop_bounds_yx": {
            "y_start_inclusive": y0,
            "y_end_exclusive": y1,
            "x_start_inclusive": x0,
            "x_end_exclusive": x1,
        },
        "input_dtype": str(crop_yxc.dtype),
        "input_range": [float(np.min(crop_yxc)), float(np.max(crop_yxc))],
        "additional_normalization_applied": False,
        "mask_path": str(mask_path),
        "mask_dtype": str(result.dtype),
        "mask_shape_yx": list(result.shape),
        "instance_count": int(positive_ids.size),
        "maximum_label": int(np.max(result)),
        "background_label": 0,
        "all_true_cells_in_scope_must_be_annotated": True,
        "cells_outside_crop_are_training_background": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = selected_crop_dir / "annotation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest = _write_crop_manifest(
        sample_dir,
        {
            "dataset": dataset,
            "source_mode": source_mode,
            "sample_key": sample_key,
            "split": selected_split,
            "crop_id": selected_crop_id,
            "y_start_inclusive": y0,
            "y_end_exclusive": y1,
            "x_start_inclusive": x0,
            "x_end_exclusive": x1,
            "height": y1 - y0,
            "width": x1 - x0,
            "channels": crop_yxc.shape[-1],
            "instance_count": int(positive_ids.size),
            "crop_input": crop_input_path,
            "mask": mask_path,
            "report": report_path,
            "updated_utc": report["created_utc"],
        },
    )
    print(f"Crop manifest: {manifest}")
    return mask_path


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create full-image or crop-based unique-cell instance masks in Napari."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotation-mode", choices=ANNOTATION_MODES)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--source-mode",
        choices=sorted({mode for modes in SOURCE_MODES_BY_DATASET.values() for mode in modes}),
    )
    parser.add_argument("--sample", default=None, help="Exact sample key")
    parser.add_argument(
        "--next-missing",
        action="store_true",
        help="Open the first sample without a full-image mask; full_image mode only",
    )
    parser.add_argument("--brush-size", type=int, default=15)
    parser.add_argument("--crop-split", choices=CROP_SPLITS)
    parser.add_argument(
        "--crop-id",
        default=None,
        help="Reopen an existing crop ID, for example crop_001",
    )
    parser.add_argument("--minimum-crop-size", type=int, default=64)
    args = parser.parse_args()

    annotation_mode = args.annotation_mode or _choose(
        "Choose annotation mode",
        ANNOTATION_MODES,
    )
    if annotation_mode == "crop_then_label" and args.next_missing:
        parser.error("--next-missing applies only to full_image mode")

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
            f"No prepared inputs found below {input_root}. "
            "Run prepare_segmentation_inputs.py first."
        )
    by_key = {_sample_key(path, input_root): path for path in inputs}

    if args.sample:
        if args.sample not in by_key:
            parser.error(f"Unknown sample key: {args.sample}")
        chosen_key = args.sample
    elif args.next_missing:
        missing = [
            key
            for key in by_key
            if not (output_root / Path(key) / "mask.tif").is_file()
        ]
        if not missing:
            print("All prepared inputs already have a full-image instance mask.")
            return 0
        chosen_key = missing[0]
    else:
        labels: list[str] = []
        keys = list(by_key)
        for key in keys:
            sample_dir = output_root / Path(key)
            full_status = "full mask exists" if (sample_dir / "mask.tif").is_file() else "no full mask"
            crop_count = len(_existing_crop_choices(sample_dir))
            labels.append(f"{key} [{full_status}; crops={crop_count}]")
        selected_label = _choose("Choose sample to annotate", labels)
        chosen_key = keys[labels.index(selected_label)]

    if annotation_mode == "full_image":
        mask_path = annotate_full_image(
            by_key[chosen_key],
            input_root=input_root,
            output_root=output_root,
            dataset=dataset,
            source_mode=source_mode,
            brush_size=args.brush_size,
        )
    else:
        crop_split = args.crop_split or _choose(
            "Assign a split to a newly created crop",
            CROP_SPLITS,
        )
        mask_path = annotate_crop(
            by_key[chosen_key],
            input_root=input_root,
            output_root=output_root,
            dataset=dataset,
            source_mode=source_mode,
            brush_size=args.brush_size,
            split=crop_split,
            crop_id=args.crop_id,
            minimum_crop_size=args.minimum_crop_size,
        )

    print(f"\nSaved manual instance mask:\n{mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
