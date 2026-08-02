"""Provide command-line and programmatic utilities for intenisty."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.denoising.notch_filter import list_omezarr_images, _ensure_cyx, _to_numpy
EPS = 1e-12
from PFT.core_prog_parts import visualization as viz
from PFT.core_prog_parts.common_paths import filtered_img_root, find_project_root
from PFT.core_prog_parts.image_utils import extract_display_plane, normalize01_percentile as norm01_percentile, to_uint8_percentile

"Creates intensity maps and FFT images from the same images used in filtering, for display and analysis."

INTENSITY_RGB_CMAP = viz.INTENSITY_RGB_CMAP

_UNUSED = (
    "intensity_rgb_orange_peak",
    [
        (0.00, (0.0, 0.0, 1.0)),  
        (0.55, (0.0, 1.0, 0.0)),  
        (0.85, (1.0, 0.0, 0.0)),  
        (1.00, (1.0, 0.65, 0.0)),  
    ],
)

CURATED_TEST_STEMS: dict[str, list[str]] = {
    "2d_time": [
        "WT_HADA_NHS_5min_ROI1_SIM²",
        "WT_HADA_NHS_5min_ROI2_SIM²",
        "WT_HADA_NHS_5min_ROI3_SIM²",
        "WT_HADA_NHS_5min_ROI4_SIM",
        "WT_HADA_NHS_5min_ROI4_SIM²",
        "WT_HADA_NHS_20min_ROI1_SIM",
        "WT_HADA_NHS_20min_ROI1_SIM²",
        "WT_HADA_NHS_20min_ROI2_SIM",
        "WT_HADA_NHS_20min_ROI2_SIM²",
        "WT_HADA_NHS_20min_ROI3_SIM",
        "WT_HADA_THY_40min_ROI3_SIM²",
        "WT_HADA_THY_40min_ROI4_SIM²",
    ]
}




def analysis_root() -> Path:
    """Helper function used by this module.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = analysis_root()
    """
    out = find_project_root() / "results" / "analysis" / "intensity_fft"
    out.mkdir(parents=True, exist_ok=True)
    return out




def fft_log_magnitude(img2d: np.ndarray) -> np.ndarray:
    """Helper function used by this module.

    Args:
        img2d (np.ndarray): Array containing img2d.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = fft_log_magnitude(img2d=image_array)
    """
    x = np.asarray(img2d, dtype=np.float32)
    F = np.fft.fftshift(np.fft.fft2(x - float(np.mean(x))))
    mag = np.log1p(np.abs(F)).astype(np.float32)
    return norm01_percentile(mag, p_lo=1.0, p_hi=99.8)


def list_curated_test_images(dataset: str, zarrs: list[Path]) -> list[Path]:
    """List available inputs for this workflow.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        zarrs (list[Path]): Filesystem path used for zarrs.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = list_curated_test_images(dataset="2d_time", zarrs=Path("path/to/resource"))
    """
    want = CURATED_TEST_STEMS.get(dataset, [])
    by_stem = {p.parent.name: p for p in zarrs}
    return [by_stem[s] for s in want if s in by_stem]



def load_planes(zarr_path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load data and return the processed result.

    Args:
        zarr_path (Path): Filesystem path associated with Zarr.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Collection containing the generated or selected values.

    Example:
        >>> result = load_planes(zarr_path=Path("path/to/resource"), dataset="2d_time")
    """
    arr, axes = load_ome_zarr(zarr_path, level=0, as_numpy=False)
    x = _to_numpy(arr)
    x, axes = _ensure_cyx(x, axes)
    blue = extract_display_plane(x, axes, 0)
    green = extract_display_plane(x, axes, 1) if dataset == "2d_wga_dapi" and ("c" in axes and x.shape[axes.index("c")] > 1) else None
    return blue, green


def make_rgb(dataset: str, blue: np.ndarray, green: np.ndarray | None) -> np.ndarray:
    """Create and return the requested display or object.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        blue (np.ndarray): Array containing blue.
        green (np.ndarray | None): Array containing green.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = make_rgb(
        ...     dataset="2d_time",
        ...     blue=image_array,
        ...     green=image_array,
        ... )
    """
    b = to_uint8_percentile(blue)
    z = np.zeros_like(b, dtype=np.uint8)
    if dataset == "2d_wga_dapi" and green is not None:
        g = to_uint8_percentile(green)
        return np.stack([z, g, b], axis=-1)
    return np.stack([z, z, b], axis=-1)


def choose_source_zarr(dataset: str, stem: str, raw_zarr: Path) -> Path:
    """Ask the user to choose a workflow option.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        stem (str): Text value specifying stem.
        raw_zarr (Path): Filesystem path used for raw Zarr.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = choose_source_zarr(
        ...     dataset="2d_time",
        ...     stem="stem",
        ...     raw_zarr=Path("path/to/resource"),
        ... )
    """
    filtered_zarr = filtered_img_root() / dataset / stem / "image.ome.zarr"
    return filtered_zarr if filtered_zarr.exists() else raw_zarr


def save_single_channel_outputs(channel: np.ndarray, out_dir: Path, stem: str, channel_name: str) -> None:
    """Save generated outputs to disk.

    Args:
        channel (np.ndarray): Channel index or channel identifier selected for processing.
        out_dir (Path): Directory used for out.
        stem (str): Text value specifying stem.
        channel_name (str): Text value specifying channel name.

    Example:
        >>> save_single_channel_outputs(
        ...     channel=image_array,
        ...     out_dir=Path("path/to/resource"),
        ...     stem="stem",
        ...     channel_name="channel_name",
        ... )
    """
    viz.save_single_channel_outputs(channel, out_dir, stem, channel_name, cmap=INTENSITY_RGB_CMAP)


def save_rgb_overview(dataset: str, blue: np.ndarray, green: np.ndarray | None, out_dir: Path, stem: str) -> None:
    """Save generated outputs to disk.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        blue (np.ndarray): Array containing blue.
        green (np.ndarray | None): Array containing green.
        out_dir (Path): Directory used for out.
        stem (str): Text value specifying stem.

    Example:
        >>> save_rgb_overview(
        ...     dataset="2d_time",
        ...     blue=image_array,
        ...     green=image_array,
        ...     out_dir=Path("path/to/resource"),
        ...     stem="stem",
        ... )
    """
    rgb = make_rgb(dataset, blue, green)
    viz.save_rgb_overview(rgb, out_dir, stem, panel_title=f"{stem} normalized RGB")


def process_one(dataset: str, raw_zarr: Path) -> Path:
    """Helper function used by this module.

    Args:
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        raw_zarr (Path): Filesystem path used for raw Zarr.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = process_one(dataset="2d_time", raw_zarr=Path("path/to/resource"))
    """
    stem = raw_zarr.parent.name
    source_zarr = choose_source_zarr(dataset, stem, raw_zarr)
    out_dir = analysis_root() / dataset / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    blue, green = load_planes(source_zarr, dataset)
    save_rgb_overview(dataset, blue, green, out_dir, stem)
    save_single_channel_outputs(blue, out_dir, stem, "blue")
    if dataset == "2d_wga_dapi" and green is not None:
        save_single_channel_outputs(green, out_dir, stem, "green")

    with open(out_dir / "source_used.txt", "w", encoding="utf-8") as f:
        f.write(str(source_zarr) + "\n")

    return out_dir


def choose_dataset_interactive() -> str:
    """Ask the user to choose a workflow option.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = choose_dataset_interactive()
    """
    print("Choose dataset:")
    print("  1) 2d_time")
    print("  2) 2d_wga_dapi")
    while True:
        s = input("Choose number [1/2]: ").strip()
        if s == "1":
            return "2d_time"
        if s == "2":
            return "2d_wga_dapi"
        print("Please choose 1 or 2.")


def choose_mode_interactive(n: int) -> str:
    """Ask the user to choose a workflow option.

    Args:
        n (int): Numerical value controlling n.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = choose_mode_interactive(n=1)
    """
    print("Choose mode:")
    print("  1) One image")
    print("  2) Curated subset")
    print("  3) All images")
    while True:
        s = input("Choose number [1/2/3]: ").strip()
        if s in {"1", "2", "3"}:
            return s
        print("Please choose 1, 2 or 3.")


def choose_index_interactive(paths: list[Path]) -> int:
    """Ask the user to choose a workflow option.

    Args:
        paths (list[Path]): Filesystem path used for paths.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = choose_index_interactive(paths=Path("path/to/resource"))
    """
    for i, p in enumerate(paths):
        print(f"  [{i:02d}] {p.parent.name}")
    while True:
        s = input(f"Choose image index [0-{len(paths)-1}]: ").strip()
        try:
            idx = int(s)
        except ValueError:
            idx = -1
        if 0 <= idx < len(paths):
            return idx
        print("Invalid index.")


def main() -> None:
    """Helper function used by this module.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        IndexError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    ap = argparse.ArgumentParser(description="Create intensity maps and FFT images from the same images used in filtering.")
    ap.add_argument("--dataset", choices=["2d_time", "2d_wga_dapi"], required=False)
    ap.add_argument("--mode", choices=["one", "curated", "all"], required=False)
    ap.add_argument("--image-index", type=int, default=None)
    args = ap.parse_args()

    dataset = args.dataset or choose_dataset_interactive()
    zarrs = list_omezarr_images(dataset)
    if not zarrs:
        raise FileNotFoundError(f"No OME-Zarr images found for dataset: {dataset}")

    mode = args.mode
    if mode is None:
        mode_choice = choose_mode_interactive(len(zarrs))
        mode = {"1": "one", "2": "curated", "3": "all"}[mode_choice]

    if mode == "curated":
        selected = list_curated_test_images(dataset, zarrs)
        if not selected:
            raise FileNotFoundError(f"No curated images found for dataset: {dataset}")
    elif mode == "all":
        selected = zarrs
    else:
        if args.image_index is None:
            idx = choose_index_interactive(zarrs)
        else:
            idx = args.image_index
        if not (0 <= idx < len(zarrs)):
            raise IndexError(f"image-index must be between 0 and {len(zarrs)-1}")
        selected = [zarrs[idx]]

    print(f"Processing {len(selected)} image(s) from dataset: {dataset}")
    for p in selected:
        out_dir = process_one(dataset, p)
        print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
