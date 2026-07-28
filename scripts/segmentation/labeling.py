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
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np
import tifffile as tiff
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries


# Samples reported as missing ``mask.tif`` by ANALYSIS_INPUT_READINESS.txt.
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
