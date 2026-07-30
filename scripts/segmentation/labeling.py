"""Create or correct the known missing 2D U-Net reference masks.

Purpose
-------
This terminal tool is restricted to the microscopy samples that were reported
as missing a hand-labelled ``mask.tif`` during the PFT 2D input-readiness
check. It lets the user select a dataset and sample, opens the corresponding
normalized image in Napari, and saves a strict binary mask in the standard
training-data location::

    results/training_files/U-net/<dataset>/<sample>/mask.tif

Mask semantics
--------------
``0`` represents background and ``1`` represents bacterial foreground. The
saved mask is therefore directly compatible with ``check_2d_inputs.py`` and
``check_2d.py`` for the ROI-based SNR calculation.

The script also saves visual quality-control files beside ``mask.tif``. Existing
masks can be reopened and corrected after explicit confirmation. For a
channel-first TIFF with shape ``(C, Y, X)``, channel index 0 is used as the
manual annotation image.

Run interactively
-----------------
    python scripts/segmentation/labeling.py

Run a specific sample
---------------------
    python scripts/segmentation/labeling.py \
        --dataset 2d_wga_dapi \
        --sample WT_CSP_NHS_ROI2_SIM
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np
import tifffile as tiff
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries


MISSING_MASK_SAMPLES: dict[str, tuple[str, ...]] = {
    "2d_time": (
        "WT_HADA_NHS_120min_ROI1_SIM",
        "WT_HADA_NHS_20min_ROI1_SIM",
        "WT_HADA_NHS_20min_ROI2_SIM",
        "WT_HADA_THY_120min_ROI1_SIM²",
    ),
    "2d_wga_dapi": (
        "WT_CSP_NHS_ROI2_SIM",
        "WT_CSP_noNHS_ROI2_SIM",
        "WT_CSP_noNHS_ROI4_SIM",
        "WT_noCSP_NHS_ROI3_SIM",
        "WT_THY_CSP_ROI2_SIM",
    ),
}

PREFERRED_IMAGE_NAMES: tuple[str, ...] = (
    "image_norm16_rgb.tif",
    "image_norm16_rgb.tiff",
    "image_norm16.tif",
    "image_norm16.tiff",
    "preview_norm.png",
    "preview_raw.png",
)

DEFAULT_3D_TARGET_SLICES_1BASED: tuple[int, ...] = (10, 12)
WAVELENGTH_TO_RGB: tuple[tuple[float, int, str], ...] = (
    (405.0, 2, "blue"),
    (488.0, 1, "green"),
    (561.0, 0, "red"),
)


@dataclass(frozen=True)
class Composite3D:
    """Selected and neighbouring RGB composites plus display metadata."""

    previous_rgb: np.ndarray
    target_rgb: np.ndarray
    next_rgb: np.ndarray
    channel_records: tuple[dict[str, Any], ...]
    normalization_records: tuple[dict[str, float], ...]
    image_shape_czyx: tuple[int, int, int, int]
    array_path: str


def find_project_root(start: Path) -> Path:
    """Find the PFT repository root by searching parent directories."""
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "results").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the PFT repository root. Expected a parent directory "
        "containing pyproject.toml and results/."
    )


def choose_number(
    title: str,
    options: Sequence[str],
    *,
    allow_exit: bool = True,
) -> int | None:
    """Display a numbered terminal menu and return the selected zero-based index."""
    if not options:
        raise RuntimeError(f"No options are available for: {title}")

    print(f"\n{title}")
    if allow_exit:
        print("  [0] Exit")
    for index, option in enumerate(options, start=1):
        print(f"  [{index}] {option}")

    while True:
        answer = input("Enter number: ").strip()
        if allow_exit and answer == "0":
            return None
        try:
            selected = int(answer)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if 1 <= selected <= len(options):
            return selected - 1
        print("Selection is outside the available range.")


def sample_image_dir(project_root: Path, dataset: str, sample: str) -> Path:
    """Return the processed-image directory for one dataset sample."""
    return project_root / "results" / "img" / dataset / sample


def sample_mask_dir(project_root: Path, dataset: str, sample: str) -> Path:
    """Return the standard U-Net mask directory for one dataset sample."""
    return (
        project_root
        / "results"
        / "training_files"
        / "U-net"
        / dataset
        / sample
    )


def mask_path_for_sample(project_root: Path, dataset: str, sample: str) -> Path:
    """Return the required binary mask path for one sample."""
    return sample_mask_dir(project_root, dataset, sample) / "mask.tif"


def status_for_sample(project_root: Path, dataset: str, sample: str) -> str:
    """Return a compact terminal status for the source image and mask."""
    image_dir = sample_image_dir(project_root, dataset, sample)
    mask_path = mask_path_for_sample(project_root, dataset, sample)

    if not image_dir.is_dir():
        return "SOURCE IMAGE FOLDER MISSING"
    if mask_path.exists():
        return "MASK EXISTS"
    return "MASK MISSING"


def list_candidate_images(sample_dir: Path) -> list[Path]:
    """Find suitable normalized TIFF or PNG files for manual annotation."""
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Processed sample directory not found: {sample_dir}")

    available = {
        path.name.lower(): path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png"}
    }

    candidates: list[Path] = []
    for name in PREFERRED_IMAGE_NAMES:
        found = available.get(name.lower())
        if found is not None:
            candidates.append(found)

    # Retain additional image files as a fallback, excluding metadata-like masks.
    for path in sorted(available.values(), key=lambda p: p.name.lower()):
        lower_name = path.name.lower()
        if path in candidates:
            continue
        if any(token in lower_name for token in ("mask", "overlay", "outline")):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No annotation image was found in {sample_dir}. Expected one of: "
            + ", ".join(PREFERRED_IMAGE_NAMES)
        )
    return candidates


def read_annotation_image(path: Path) -> np.ndarray:
    """Read an annotation image and reduce it to a supported 2D display.

    Supported layouts
    -----------------
    YX
        Returned unchanged as a grayscale annotation image.

    YXC with three or four channels
        Interpreted as an RGB/RGBA display image. Alpha is discarded.

    CYX
        Interpreted as a channel-first microscopy TIFF. Channel index 0 is
        selected for manual annotation. This is the expected layout for files
        such as ``image_norm16.tif`` with shape ``(2, 2048, 2048)``.

    Singleton dimensions are removed before applying these rules.
    """
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = np.asarray(tiff.imread(str(path)))
    else:
        image = np.asarray(iio.imread(path))

    image = np.squeeze(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        return image[..., :3]

    if (
        image.ndim == 3
        and 1 <= image.shape[0] <= 6
        and image.shape[1] > 6
        and image.shape[2] > 6
    ):
        print(
            f"Channel-first annotation image detected: shape={image.shape}. "
            "Using channel 0."
        )
        return np.asarray(image[0])

    raise ValueError(
        f"Unsupported annotation image shape {image.shape} for {path}. "
        "Expected YX, YXRGB, or channel-first CYX data."
    )


def image_spatial_shape(image: np.ndarray) -> tuple[int, int]:
    """Return the Y and X dimensions of a supported annotation image."""
    if image.ndim == 2:
        return int(image.shape[0]), int(image.shape[1])
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        return int(image.shape[0]), int(image.shape[1])
    raise ValueError(f"Unsupported annotation image shape: {image.shape}")


def load_existing_mask(mask_path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load an existing mask for correction or return an empty binary mask."""
    if not mask_path.exists():
        return np.zeros(expected_shape, dtype=np.uint8)

    mask = np.asarray(tiff.imread(str(mask_path)))
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.shape != expected_shape:
        raise ValueError(
            f"Existing mask shape {mask.shape} does not match image shape "
            f"{expected_shape}: {mask_path}"
        )
    return (mask > 0).astype(np.uint8)


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGB microscopy image to display-ready RGB uint8."""
    array = np.asarray(image)

    if array.ndim == 2:
        channels = [array, array, array]
        array = np.stack(channels, axis=-1)
    elif array.ndim == 3 and array.shape[-1] in {3, 4}:
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.dtype == np.uint8:
        return array.copy()

    values = array.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Annotation image contains no finite pixel values.")

    low, high = np.percentile(finite, [1.0, 99.8])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def annotate_with_napari(
    image: np.ndarray,
    initial_mask: np.ndarray,
    *,
    sample: str,
    brush_size: int,
) -> np.ndarray:
    """Open Napari and return the edited strict binary foreground mask."""
    import napari

    display = to_rgb_uint8(image)

    viewer = napari.Viewer(
        title=f"PFT U-Net mask annotation | {sample} | cells=1, background=0"
    )
    viewer.add_image(display, name="image", rgb=True)

    labels_layer = viewer.add_labels(
        initial_mask.astype(np.uint8),
        name="mask",
        opacity=0.45,
    )
    labels_layer.selected_label = 1
    labels_layer.brush_size = int(brush_size)

    print(
        "\nNapari instructions\n"
        "-------------------\n"
        "Paint bacterial foreground with label 1.\n"
        "Erase incorrect regions back to label 0.\n"
        "Close the Napari window when annotation is complete.\n"
    )

    napari.run()

    result = np.asarray(labels_layer.data)
    return (result > 0).astype(np.uint8)


def create_outline(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Create a dilated outer boundary for visual quality control."""
    boundaries = find_boundaries(mask.astype(bool), mode="outer")
    return binary_dilation(boundaries, disk(int(thickness)))


def create_overlay(image: np.ndarray, mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Create an RGB image with the mask boundary drawn in yellow."""
    rgb = to_rgb_uint8(image)
    outline = create_outline(mask, thickness=thickness)
    output = rgb.copy()
    output[outline] = np.array([255, 255, 0], dtype=np.uint8)
    return output


def save_triptych(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    *,
    thickness: int = 3,
) -> None:
    """Save an original, yellow-outline, and binary-mask quality-control panel."""
    import matplotlib.pyplot as plt

    rgb = to_rgb_uint8(image)
    overlay = create_overlay(image, mask, thickness=thickness)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[1].imshow(overlay)
    axes[1].set_title("Original + yellow outline")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Mask (0/1)")
    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def validate_binary_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> None:
    """Validate shape, binary values, foreground, and background content."""
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {expected_shape}."
        )

    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Mask must contain only 0 and 1. Found: {sorted(unique)}")

    foreground = int(np.count_nonzero(mask == 1))
    background = int(np.count_nonzero(mask == 0))
    if foreground < 1:
        raise ValueError("Mask contains no foreground pixels.")
    if background < 2:
        raise ValueError("Mask must contain at least two background pixels.")


def save_annotation_outputs(
    project_root: Path,
    dataset: str,
    sample: str,
    source_image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    outline_thickness: int,
) -> Path:
    """Save the binary mask and its visual quality-control files."""
    output_dir = sample_mask_dir(project_root, dataset, sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_mask = output_dir / "mask.tif"
    tiff.imwrite(output_mask, mask.astype(np.uint8))

    # Keep the existing training-folder convention.
    tiff.imwrite(output_dir / "image.tif", np.asarray(image))
    tiff.imwrite(output_dir / "mask_vis.tif", (mask * 255).astype(np.uint8))

    overlay = create_overlay(image, mask, thickness=outline_thickness)
    iio.imwrite(output_dir / "overlay_yellow_outline.png", overlay)
    save_triptych(
        image,
        mask,
        output_dir / "composite_triptych.png",
        thickness=outline_thickness,
    )

    foreground = int(np.count_nonzero(mask == 1))
    background = int(np.count_nonzero(mask == 0))
    record = (
        "PFT manual U-Net mask annotation\n"
        "================================\n"
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Dataset: {dataset}\n"
        f"Sample: {sample}\n"
        f"Source annotation image: {source_image_path}\n"
        f"Mask path: {output_mask}\n"
        f"Mask shape: {mask.shape}\n"
        f"Foreground pixels: {foreground}\n"
        f"Background pixels: {background}\n"
        f"Foreground fraction: {foreground / mask.size:.8f}\n"
        "Mask semantics: 0=background, 1=foreground\n"
    )
    (output_dir / "labeling_record.txt").write_text(record, encoding="utf-8")
    return output_mask


def choose_dataset_interactively(project_root: Path) -> str | None:
    """Let the user choose a dataset containing hard-coded missing samples."""
    labels = []
    datasets = list(MISSING_MASK_SAMPLES)
    for dataset in datasets:
        missing_count = sum(
            not mask_path_for_sample(project_root, dataset, sample).exists()
            for sample in MISSING_MASK_SAMPLES[dataset]
        )
        labels.append(
            f"{dataset} | {missing_count} still missing / "
            f"{len(MISSING_MASK_SAMPLES[dataset])} listed"
        )

    selected = choose_number("Choose dataset", labels)
    return None if selected is None else datasets[selected]


def choose_sample_interactively(
    project_root: Path,
    dataset: str,
    *,
    include_completed: bool,
) -> str | None:
    """Let the user choose one listed sample and show its current readiness status."""
    samples = list(MISSING_MASK_SAMPLES[dataset])
    if not include_completed:
        samples = [
            sample
            for sample in samples
            if not mask_path_for_sample(project_root, dataset, sample).exists()
        ]

    if not samples:
        print(f"\nAll listed masks for {dataset} already exist.")
        return None

    labels = [
        f"{sample} | {status_for_sample(project_root, dataset, sample)}"
        for sample in samples
    ]
    selected = choose_number(f"Choose sample in {dataset}", labels)
    return None if selected is None else samples[selected]


def choose_image_interactively(sample_dir: Path) -> Path:
    """Let the user choose an annotation image from one sample directory."""
    candidates = list_candidate_images(sample_dir)
    if len(candidates) == 1:
        print(f"Using annotation image: {candidates[0].name}")
        return candidates[0]

    selected = choose_number(
        "Choose image to annotate",
        [path.name for path in candidates],
    )
    if selected is None:
        raise KeyboardInterrupt("Image selection cancelled.")
    return candidates[selected]


def save_annotation_outputs(
    project_root: Path,
    dataset: str,
    sample: str,
    source_image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    outline_thickness: int,
) -> Path:
    """Save the binary mask and its visual quality-control files."""
    output_dir = sample_mask_dir(project_root, dataset, sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_mask = output_dir / "mask.tif"
    tiff.imwrite(output_mask, mask.astype(np.uint8))

    # Keep the existing training-folder convention.
    tiff.imwrite(output_dir / "image.tif", np.asarray(image))
    tiff.imwrite(output_dir / "mask_vis.tif", (mask * 255).astype(np.uint8))

    overlay = create_overlay(image, mask, thickness=outline_thickness)
    iio.imwrite(output_dir / "overlay_yellow_outline.png", overlay)
    save_triptych(
        image,
        mask,
        output_dir / "composite_triptych.png",
        thickness=outline_thickness,
    )

    foreground = int(np.count_nonzero(mask == 1))
    background = int(np.count_nonzero(mask == 0))
    record = (
        "PFT manual U-Net mask annotation\n"
        "================================\n"
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Dataset: {dataset}\n"
        f"Sample: {sample}\n"
        f"Source annotation image: {source_image_path}\n"
        f"Mask path: {output_mask}\n"
        f"Mask shape: {mask.shape}\n"
        f"Foreground pixels: {foreground}\n"
        f"Background pixels: {background}\n"
        f"Foreground fraction: {foreground / mask.size:.8f}\n"
        "Mask semantics: 0=background, 1=foreground\n"
    )
    (output_dir / "labeling_record.txt").write_text(record, encoding="utf-8")
    return output_mask


def choose_dataset_interactively(project_root: Path) -> str | None:
    """Let the user choose a dataset containing hard-coded missing samples."""
    labels = []
    datasets = list(MISSING_MASK_SAMPLES)
    for dataset in datasets:
        missing_count = sum(
            not mask_path_for_sample(project_root, dataset, sample).exists()
            for sample in MISSING_MASK_SAMPLES[dataset]
        )
        labels.append(
            f"{dataset} | {missing_count} still missing / "
            f"{len(MISSING_MASK_SAMPLES[dataset])} listed"
        )

    selected = choose_number("Choose dataset", labels)
    return None if selected is None else datasets[selected]


def choose_sample_interactively(
    project_root: Path,
    dataset: str,
    *,
    include_completed: bool,
) -> str | None:
    """Let the user choose one listed sample and show its current readiness status."""
    samples = list(MISSING_MASK_SAMPLES[dataset])
    if not include_completed:
        samples = [
            sample
            for sample in samples
            if not mask_path_for_sample(project_root, dataset, sample).exists()
        ]

    if not samples:
        print(f"\nAll listed masks for {dataset} already exist.")
        return None

    labels = [
        f"{sample} | {status_for_sample(project_root, dataset, sample)}"
        for sample in samples
    ]
    selected = choose_number(f"Choose sample in {dataset}", labels)
    return None if selected is None else samples[selected]


def choose_image_interactively(sample_dir: Path) -> Path:
    """Let the user choose an annotation image from one sample directory."""
    candidates = list_candidate_images(sample_dir)
    if len(candidates) == 1:
        print(f"Using annotation image: {candidates[0].name}")
        return candidates[0]

    selected = choose_number(
        "Choose image to annotate",
        [path.name for path in candidates],
    )
    if selected is None:
        raise KeyboardInterrupt("Image selection cancelled.")
    return candidates[selected]


def save_annotation_outputs_2d(
    project_root: Path,
    dataset: str,
    zarr_path: Path,
    source_image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    outline_thickness: int,
) -> Path:
    output_dir = sample_mask_dir(project_root, dataset, sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mask = output_dir / "mask.tif"
    tiff.imwrite(output_mask, mask.astype(np.uint8))
    tiff.imwrite(output_dir / "image.tif", np.asarray(image))
    tiff.imwrite(output_dir / "mask_vis.tif", (mask * 255).astype(np.uint8))
    iio.imwrite(
        output_dir / "overlay_yellow_outline.png",
        create_overlay(image, mask, thickness=outline_thickness),
    )
    save_triptych(
        image,
        mask,
        output_dir / "composite_triptych.png",
        thickness=outline_thickness,
    )
    foreground = int(np.count_nonzero(mask))
    record = (
        "PFT manual 2D U-Net mask annotation\n"
        "====================================\n"
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Dataset: {dataset}\n"
        f"Sample: {sample}\n"
        f"Source annotation image: {source_image_path}\n"
        f"Mask path: {output_mask}\n"
        f"Mask shape: {mask.shape}\n"
        f"Foreground pixels: {foreground}\n"
        f"Background pixels: {mask.size - foreground}\n"
        f"Foreground fraction: {foreground / mask.size:.8f}\n"
        "Mask semantics: 0=background, 1=foreground\n"
    )
    (output_dir / "labeling_record.txt").write_text(record, encoding="utf-8")
    return output_mask
def confirm_existing_mask(mask_path: Path) -> bool:
    """Ask whether an existing mask should be reopened and overwritten."""
    if not mask_path.exists():
        return True
    response = input(
        f"\nMask already exists:\n{mask_path}\n"
        "Open it for correction and overwrite it after saving? [y/N]: "
    ).strip().lower()
    return response in {"y", "yes"}


def process_one_sample(
    project_root: Path,
    dataset: str,
    sample: str,
    *,
    brush_size: int,
    outline_thickness: int,
) -> Path | None:
    """Open one hard-coded sample in Napari and save its binary U-Net mask."""
    if dataset not in MISSING_MASK_SAMPLES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if sample not in MISSING_MASK_SAMPLES[dataset]:
        raise ValueError(
            f"Sample is not in the hard-coded missing-mask list: {dataset}/{sample}"
        )

    source_dir = sample_image_dir(project_root, dataset, sample)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Processed source folder does not exist: {source_dir}")

    output_mask = mask_path_for_sample(project_root, dataset, sample)
    if not confirm_existing_mask(output_mask):
        print("Mask correction cancelled.")
        return None

    source_image_path = choose_image_interactively(source_dir)
    image = read_annotation_image(source_image_path)
    expected_shape = image_spatial_shape(image)
    initial_mask = load_existing_mask(output_mask, expected_shape)

    edited_mask = annotate_with_napari(
        image,
        initial_mask,
        sample=sample,
        brush_size=brush_size,
    )
    validate_binary_mask(edited_mask, expected_shape)

    saved = save_annotation_outputs(
        project_root,
        dataset,
        sample,
        source_image_path,
        image,
        edited_mask,
        outline_thickness=outline_thickness,
    )
    print(f"\n2D mask saved: {saved}")
    return saved


# ---------------------------------------------------------------------------
# New 3D sparse 2.5D workflow
# ---------------------------------------------------------------------------


def find_3d_zarrs(project_root: Path) -> list[Path]:
    """Find all source 3D image.ome.zarr stores."""
    image_root = project_root / "results" / "img" / "3d_data"
    if not image_root.is_dir():
        raise FileNotFoundError(f"3D image root does not exist: {image_root}")
    stores = sorted(path for path in image_root.rglob("image.ome.zarr") if path.is_dir())
    if not stores:
        raise FileNotFoundError(f"No image.ome.zarr stores were found under {image_root}")
    return stores


def sample_name_for_zarr(zarr_path: Path) -> str:
    """Return the leaf sample folder name for display."""
    return zarr_path.parent.name


def relative_volume_path_3d(project_root: Path, zarr_path: Path) -> Path:
    """Return the collision-free experiment/sample path below results/img/3d_data."""
    image_root = (project_root / "results" / "img" / "3d_data").resolve()
    sample_dir = Path(zarr_path).resolve().parent
    try:
        relative = sample_dir.relative_to(image_root)
    except ValueError as error:
        raise ValueError(
            f"OME-Zarr is outside the configured 3D image root {image_root}: {zarr_path}"
        ) from error
    if len(relative.parts) < 2:
        raise ValueError(
            "3D annotation expects <experiment>/<sample>/image.ome.zarr below "
            f"{image_root}; received {relative}"
        )
    return relative


def volume_key_3d(project_root: Path, zarr_path: Path) -> str:
    """Return a unique experiment/sample identifier for display and splitting."""
    return relative_volume_path_3d(project_root, zarr_path).as_posix()


def mask_dir_3d(project_root: Path, zarr_path: Path) -> Path:
    return (
        project_root
        / "results"
        / "training_files"
        / "U-net"
        / "3d_25d"
        / relative_volume_path_3d(project_root, zarr_path)
    )


def mask_path_3d(project_root: Path, zarr_path: Path, slice_1based: int) -> Path:
    return mask_dir_3d(project_root, zarr_path) / f"z{slice_1based:03d}_mask.tif"


def completed_target_count(project_root: Path, zarr_path: Path) -> int:
    return sum(
        mask_path_3d(project_root, zarr_path, z).exists()
        for z in DEFAULT_3D_TARGET_SLICES_1BASED
    )


def relative_zarr_label(project_root: Path, zarr_path: Path) -> str:
    return volume_key_3d(project_root, zarr_path)


def choose_zarr_3d_interactively(
    project_root: Path,
    *,
    include_completed: bool,
) -> Path | None:
    stores = find_3d_zarrs(project_root)
    visible: list[Path] = []
    labels: list[str] = []
    for path in stores:
        completed = completed_target_count(project_root, path)
        if not include_completed and completed == len(DEFAULT_3D_TARGET_SLICES_1BASED):
            continue
        visible.append(path)
        labels.append(
            f"{relative_zarr_label(project_root, path)} | "
            f"{completed}/{len(DEFAULT_3D_TARGET_SLICES_1BASED)} target masks"
        )
    if not visible:
        print("\nAll 3D stacks have masks for Z10 and Z12.")
        return None
    selected = choose_number("Choose a 3D OME-Zarr stack", labels)
    return None if selected is None else visible[selected]


def choose_target_slice_interactively(
    project_root: Path,
    zarr_path: Path,
    *,
    include_completed: bool,
) -> int | None:
    slices = list(DEFAULT_3D_TARGET_SLICES_1BASED)
    if not include_completed:
        slices = [z for z in slices if not mask_path_3d(project_root, zarr_path, z).exists()]
    if not slices:
        print(f"\nAll target masks already exist for {volume_key_3d(project_root, zarr_path)}.")
        return None
    labels = []
    for z in slices:
        state = "MASK EXISTS" if mask_path_3d(project_root, zarr_path, z).exists() else "MASK MISSING"
        labels.append(f"Z{z} | input context Z{z-1}, Z{z}, Z{z+1} | {state}")
    selected = choose_number("Choose the middle Z-slice to annotate", labels)
    return None if selected is None else slices[selected]


def _resolve_rgb_assignment(wavelength_nm: float) -> tuple[int, str]:
    """Map wavelength to the nearest fixed RGB display channel."""
    wavelength, rgb_index, colour = min(
        WAVELENGTH_TO_RGB,
        key=lambda item: abs(float(wavelength_nm) - item[0]),
    )
    if abs(float(wavelength_nm) - wavelength) > 35.0:
        raise ValueError(
            f"Wavelength {wavelength_nm} nm cannot be assigned safely to the "
            "405/488/561 nm blue/green/red convention."
        )
    return rgb_index, colour


def _normalize_channel_context(
    channel_zyx: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Normalize three context planes together for consistent display."""
    values = np.asarray(channel_zyx, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Channel context contains no finite values.")
    low, high = np.percentile(finite, [1.0, 99.8])
    low = float(low)
    high = float(high)
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return normalized, low, high


def load_3d_composite_context(
    zarr_path: Path,
    *,
    level: int,
    slice_1based: int,
) -> Composite3D:
    """Load Z-1/Z/Z+1 only and construct wavelength-aware RGB composites."""
    import zarr

    from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
    from PFT.core_prog_parts.denoising.metadata_3d import resolve_channel_optics

    zarr_path = Path(zarr_path)
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"OME-Zarr store does not exist: {zarr_path}")

    meta = extract_ome_zarr_meta_for_compare(zarr_path, level=level)
    axes = str(meta.get("axes") or "").lower()
    if axes != "czyx":
        raise ValueError(
            f"3D annotation requires stored CZYX axes; received {axes!r}. "
            "Run check_3d_inputs.py first."
        )
    array_path = str(meta.get("array_path"))
    root = zarr.open_group(str(zarr_path), mode="r")
    array = root[array_path]
    if array.ndim != 4:
        raise ValueError(f"Expected CZYX data, received shape {array.shape}")

    c_count, z_count, y_count, x_count = (int(v) for v in array.shape)
    if slice_1based < 2 or slice_1based > z_count - 1:
        raise ValueError(
            f"Target Z{slice_1based} requires complete Z-1/Z/Z+1 context, "
            f"but the stack contains Z1..Z{z_count}."
        )
    z_index = slice_1based - 1

    # C, three context Z planes, Y, X. The complete volume is never loaded.
    context = np.asarray(array[:, z_index - 1 : z_index + 2, :, :], dtype=np.float32)
    if context.shape != (c_count, 3, y_count, x_count):
        raise ValueError(f"Unexpected context shape: {context.shape}")
    if not np.isfinite(context).all():
        raise ValueError(f"Selected context contains non-finite data: {zarr_path}")

    optics = resolve_channel_optics(zarr_path, level=level)
    if len(optics) != c_count:
        raise ValueError(
            f"Channel metadata count {len(optics)} does not match C={c_count}."
        )

    composites = np.zeros((3, y_count, x_count, 3), dtype=np.float32)
    channel_records: list[dict[str, Any]] = []
    normalization_records: list[dict[str, float]] = []
    used_rgb: set[int] = set()

    for channel in optics:
        rgb_index, colour = _resolve_rgb_assignment(channel.wavelength_nm)
        if rgb_index in used_rgb:
            raise ValueError(
                "More than one microscopy channel maps to the same RGB colour; "
                "the composite assignment is ambiguous."
            )
        used_rgb.add(rgb_index)
        normalized, low, high = _normalize_channel_context(context[channel.index])
        composites[..., rgb_index] = normalized
        channel_records.append(
            {
                "channel_index": int(channel.index),
                "channel_name": str(channel.name),
                "wavelength_nm": float(channel.wavelength_nm),
                "wavelength_source": str(channel.wavelength_source),
                "display_colour": colour,
                "rgb_index": int(rgb_index),
            }
        )
        normalization_records.append(
            {
                "channel_index": int(channel.index),
                "display_low_percentile_1": low,
                "display_high_percentile_99_8": high,
            }
        )

    composites_u8 = np.round(np.clip(composites, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Composite3D(
        previous_rgb=composites_u8[0],
        target_rgb=composites_u8[1],
        next_rgb=composites_u8[2],
        channel_records=tuple(channel_records),
        normalization_records=tuple(normalization_records),
        image_shape_czyx=(c_count, z_count, y_count, x_count),
        array_path=array_path,
    )


def save_annotation_outputs_3d(
    project_root: Path,
    zarr_path: Path,
    slice_1based: int,
    level: int,
    composite: Composite3D,
    mask: np.ndarray,
    *,
    outline_thickness: int,
) -> Path:
    """Save one sparse 2.5D target mask and slice-specific QC files."""
    sample = volume_key_3d(project_root, zarr_path)
    output_dir = mask_dir_3d(project_root, zarr_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"z{slice_1based:03d}"
    output_mask = output_dir / f"{prefix}_mask.tif"

    # The training target must remain strict 0/1 uint8.
    tiff.imwrite(output_mask, mask.astype(np.uint8))
    tiff.imwrite(output_dir / f"{prefix}_composite_rgb.tif", composite.target_rgb)
    tiff.imwrite(output_dir / f"{prefix}_mask_vis.tif", (mask * 255).astype(np.uint8))
    iio.imwrite(
        output_dir / f"{prefix}_overlay_yellow_outline.png",
        create_overlay(composite.target_rgb, mask, thickness=outline_thickness),
    )
    save_triptych(
        composite.target_rgb,
        mask,
        output_dir / f"{prefix}_composite_triptych.png",
        thickness=outline_thickness,
        original_title=f"Three-colour composite Z{slice_1based}",
    )

    foreground = int(np.count_nonzero(mask))
    channel_lines = "\n".join(
        "  C{channel_index} {channel_name}: {wavelength_nm:g} nm -> "
        "{display_colour} ({wavelength_source})".format(**record)
        for record in composite.channel_records
    )
    normalization_lines = "\n".join(
        "  C{channel_index}: p1={display_low_percentile_1:.8g}, "
        "p99.8={display_high_percentile_99_8:.8g}".format(**record)
        for record in composite.normalization_records
    )
    record = (
        "PFT manual sparse 2.5D U-Net annotation\n"
        "========================================\n"
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Source OME-Zarr: {zarr_path}\n"
        f"Source array path: {composite.array_path}\n"
        f"Pyramid level: {level}\n"
        f"Source shape CZYX: {composite.image_shape_czyx}\n"
        f"Sample: {sample}\n"
        f"Target slice, one-based: Z{slice_1based}\n"
        f"Target array index, zero-based: {slice_1based - 1}\n"
        f"Input context: Z{slice_1based - 1}, Z{slice_1based}, Z{slice_1based + 1}\n"
        f"Mask path: {output_mask}\n"
        f"Mask shape: {mask.shape}\n"
        f"Foreground pixels: {foreground}\n"
        f"Background pixels: {mask.size - foreground}\n"
        f"Foreground fraction: {foreground / mask.size:.8f}\n"
        "Mask semantics: 0=background, 1=foreground\n"
        "Channel-to-colour mapping:\n"
        f"{channel_lines}\n"
        "Display normalization computed jointly across Z-1/Z/Z+1:\n"
        f"{normalization_lines}\n"
        "The same context-wise percentile normalization and wavelength-to-RGB mapping are used during training and validation; only the uint8 conversion is preview-specific.\n"
    )
    (output_dir / f"{prefix}_labeling_record.txt").write_text(record, encoding="utf-8")
    return output_mask


def process_one_sample_3d(
    project_root: Path,
    zarr_path: Path,
    slice_1based: int,
    *,
    level: int,
    brush_size: int,
    outline_thickness: int,
) -> Path | None:
    """Open one OME-Zarr target slice in napari and save its binary mask."""
    zarr_path = Path(zarr_path).resolve()
    sample = volume_key_3d(project_root, zarr_path)

    output_mask = mask_path_3d(project_root, zarr_path, slice_1based)
    if not confirm_existing_mask(output_mask):
        print("Mask correction cancelled.")
        return None

    print("\nLoading only the selected Z-1/Z/Z+1 context...")
    composite = load_3d_composite_context(
        zarr_path,
        level=level,
        slice_1based=slice_1based,
    )
    expected_shape = image_spatial_shape(composite.target_rgb)
    initial_mask = load_existing_mask(output_mask, expected_shape)

    mapping = ", ".join(
        f"C{record['channel_index']} {record['wavelength_nm']:g}nm={record['display_colour']}"
        for record in composite.channel_records
    )
    print(f"Stack: {zarr_path}")
    print(f"Target: Z{slice_1based}; context Z{slice_1based-1}/Z{slice_1based}/Z{slice_1based+1}")
    print(f"Composite mapping: {mapping}")

    edited_mask = annotate_with_napari(
        composite.target_rgb,
        initial_mask,
        title=f"PFT 2.5D annotation | {sample} | target Z{slice_1based}",
        brush_size=brush_size,
        previous_image=composite.previous_rgb,
        next_image=composite.next_rgb,
    )
    validate_binary_mask(edited_mask, expected_shape)
    saved = save_annotation_outputs_3d(
        project_root,
        zarr_path,
        slice_1based,
        level,
        composite,
        edited_mask,
        outline_thickness=outline_thickness,
    )

    foreground = int(np.count_nonzero(edited_mask))
    print("\nMask saved successfully")
    print("-----------------------")
    print(f"Dataset: {dataset}")
    print(f"Sample: {sample}")
    print(f"Mask: {saved}")
    print(f"Shape: {edited_mask.shape}")
    print(f"Foreground fraction: {foreground / edited_mask.size:.4%}")
    return saved


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create or correct the nine known missing 2D U-Net masks and save "
            "them under results/training_files/U-net."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(MISSING_MASK_SAMPLES),
        help="Dataset containing the missing mask.",
    )
    parser.add_argument(
        "--sample",
        help="Exact listed 2D sample name. Requires --dataset in 2D mode.",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        help="Specific 3D image.ome.zarr. Applies only to --mode 3d.",
    )
    parser.add_argument(
        "--slice",
        dest="slice_1based",
        type=int,
        help=(
            "One-based middle Z-slice for 3D annotation. Standard targets are "
            "10 and 12."
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="OME-Zarr pyramid level for 3D annotation. Default: 0.",
        help="Exact hard-coded sample name. Requires --dataset.",
    )
    parser.add_argument(
        "--brush-size",
        type=int,
        default=20,
        help="Initial Napari labels-brush size in pixels. Default: 20.",
    )
    parser.add_argument(
        "--outline-thickness",
        type=int,
        default=3,
        help="Yellow quality-control outline thickness. Default: 3.",
    )
    parser.add_argument(
        "--show-completed",
        action="store_true",
        help="Show listed samples whose mask.tif already exists.",
    )
    return parser


def main() -> None:
    """Run command-selected or interactive missing-mask annotation."""
    args = build_parser().parse_args()
    project_root = find_project_root(Path(__file__))

    if args.sample and not args.dataset:
        raise SystemExit("--sample requires --dataset.")

    if args.dataset and args.sample:
        process_one_sample(
            project_root,
            args.dataset,
            args.sample,
            brush_size=args.brush_size,
            outline_thickness=args.outline_thickness,
        )
        return

    while True:
        dataset = args.dataset or choose_dataset_interactively(project_root)
        if dataset is None:
            return

        sample = choose_sample_interactively(
            project_root,
            dataset,
            include_completed=args.show_completed,
        )
        if sample is None:
            if args.dataset:
                return
            continue

        try:
            process_one_sample(
                project_root,
                dataset,
                sample,
                brush_size=args.brush_size,
                outline_thickness=args.outline_thickness,
            )
        except Exception as exc:
            print(f"\nERROR: {type(exc).__name__}: {exc}")

        if args.dataset:
            again = input(
                f"\nAnnotate another listed sample in {dataset}? [Y/n]: "
            ).strip().lower()
        else:
            again = input("\nAnnotate another missing mask? [Y/n]: ").strip().lower()

        if again in {"n", "no"}:
            return


if __name__ == "__main__":
    main()
