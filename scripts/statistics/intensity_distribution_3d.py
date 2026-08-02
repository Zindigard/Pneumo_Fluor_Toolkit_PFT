r"""Provide command-line and programmatic utilities for intensity distribution three-dimensional data.

Examples
--------
Show all command-line parameters:

    python scripts/statistics/intensity_distribution_3d.py --help

Representative execution:

    python scripts/statistics/intensity_distribution_3d.py \
        --root_3d results/img/3d_data \
        --level 0 \
        --out_root results/example_output
"""

from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``.

    The lookup is based on this script's physical location and therefore does
    not depend on the current working directory. An explicit error is raised
    when the expected repository layout cannot be found.

    Args:
        start (_PFTPath | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        _PFTPath: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _pft_project_root()
    """
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))



import argparse
from pathlib import Path
import sys

import numpy as np
import tifffile as tiff

_THIS_FILE = Path(__file__).resolve()

from PFT.core_prog_parts.decoder_omezar import (
    extract_ome_zarr_meta_for_compare,
    load_ome_zarr_3d_czyx,
)

DEFAULT_LEVEL = 2
DEFAULT_RELATIVE_3D_ROOT = Path("results/img/3d_data")
DEFAULT_RELATIVE_OUT_ROOT = Path("results/examination")

CHANNEL_COLOR_BY_INDEX = {
    0: "blue",
    1: "green",
    2: "red",
}


def find_project_root(start: Path) -> Path:
    """Find project root by walking upward until src/PFT exists.

    Args:
        start (Path): Filesystem path used for start.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = find_project_root(start=Path("path/to/resource"))
    """
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "src" / "PFT").exists():
            return p
    raise FileNotFoundError(
        "Could not find project root. Expected a folder containing 'src/PFT'."
    )


def ensure_dir(path: Path) -> None:
    """Create a folder if it does not yet exist.

    Args:
        path (Path): Filesystem path to the required input or output resource.

    Example:
        >>> ensure_dir(path=Path("path/to/resource"))
    """
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    """Make names safe for Windows folder/file names.

    Args:
        name (str): Name used to identify the current object, resource, or output.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = sanitize_name(name="name")
    """
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in str(name))
    return out.strip().replace(" ", "_")


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    """Min-max normalize one 2D slice to uint8 for display.

    Args:
        img (np.ndarray): Array containing img.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = normalize_to_u8(img=image_array)
    """
    arr = np.asarray(img, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out, 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def colorize_single_channel(gray_u8: np.ndarray, color_name: str) -> np.ndarray:
    """Place grayscale data into one RGB display channel.

    Args:
        gray_u8 (np.ndarray): Array containing gray u8.
        color_name (str): Text value specifying color name.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = colorize_single_channel(gray_u8=image_array, color_name="color_name")
    """
    rgb = np.zeros((*gray_u8.shape, 3), dtype=np.uint8)
    name = color_name.lower()
    if name == "blue":
        rgb[..., 2] = gray_u8
    elif name == "green":
        rgb[..., 1] = gray_u8
    elif name == "red":
        rgb[..., 0] = gray_u8
    else:
        rgb[...] = gray_u8[..., None]
    return rgb


def make_intensity_distribution_map(img: np.ndarray) -> np.ndarray:
    """Create an RGB intensity map for one 2D slice.

    0 stays black.
    Lowest nonzero intensities are blue, middle intensities are green,
    and the brightest intensities are red.

    Args:
        img (np.ndarray): Array containing img.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = make_intensity_distribution_map(img=image_array)
    """
    arr = np.asarray(img, dtype=np.float32)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)

    vmax = float(arr.max())
    if not np.isfinite(vmax) or vmax <= 0.0:
        return rgb

    x = np.clip(arr / vmax, 0.0, 1.0)
    nz = x > 0.0

    low = nz & (x <= 0.5)
    if np.any(low):
        t = x[low] / 0.5
        rgb[..., 2][low] = np.round(255.0 * (1.0 - t)).astype(np.uint8)
        rgb[..., 1][low] = np.round(255.0 * t).astype(np.uint8)

    high = x > 0.5
    if np.any(high):
        t = (x[high] - 0.5) / 0.5
        rgb[..., 1][high] = np.round(255.0 * (1.0 - t)).astype(np.uint8)
        rgb[..., 0][high] = np.round(255.0 * t).astype(np.uint8)

    return rgb


def make_normalized_composite(norm_by_channel: list[np.ndarray]) -> np.ndarray:
    """Create one RGB composite from normalized per-channel slices.

    Channel index 0 -> blue, 1 -> green, 2 -> red.
    Extra channels beyond 3 are ignored.

    Args:
        norm_by_channel (list[np.ndarray]): Array containing norm by channel.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = make_normalized_composite(norm_by_channel=image_array)
    """
    if not norm_by_channel:
        raise ValueError("No channels were provided for normalized composite.")

    y, x = norm_by_channel[0].shape
    rgb = np.zeros((y, x, 3), dtype=np.uint8)

    if len(norm_by_channel) > 0:
        rgb[..., 2] = norm_by_channel[0]
    if len(norm_by_channel) > 1:
        rgb[..., 1] = norm_by_channel[1]
    if len(norm_by_channel) > 2:
        rgb[..., 0] = norm_by_channel[2]

    return rgb


def make_intensity_composite(map_by_channel: list[np.ndarray]) -> np.ndarray:
    """Create one RGB composite from per-channel intensity-map RGB images.

    The three RGB maps are combined with a channel-wise maximum.

    Args:
        map_by_channel (list[np.ndarray]): Array containing map by channel.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = make_intensity_composite(map_by_channel=image_array)
    """
    if not map_by_channel:
        raise ValueError("No channels were provided for intensity composite.")

    stack = np.stack(map_by_channel, axis=0)
    return np.max(stack, axis=0).astype(np.uint8)


def save_per_channel_outputs(
    vol_czyx: np.ndarray,
    *,
    channel_names: list[str],
    dataset_out: Path,
) -> None:
    """Save per-slice, per-channel normalized and intensity-map TIFF files. Also save one per-slice 3-channel composite for normalized and intensity views.

    Args:
        vol_czyx (np.ndarray): Array containing vol czyx.
        channel_names (list[str]): Text value specifying channel names.
        dataset_out (Path): Filesystem path used for dataset out.

    Example:
        >>> save_per_channel_outputs(
        ...     vol_czyx=image_array,
        ...     channel_names="channel_names",
        ...     dataset_out=Path("path/to/resource"),
        ... )
    """
    per_channel_norm: list[Path] = []
    per_channel_map: list[Path] = []

    for c in range(vol_czyx.shape[0]):
        channel_name = str(channel_names[c])

        ch_root = dataset_out / f"C{c}_{sanitize_name(channel_name)}"
        norm_dir = ch_root / "normalized"
        map_dir = ch_root / "intensity_map"
        ensure_dir(norm_dir)
        ensure_dir(map_dir)

        per_channel_norm.append(norm_dir)
        per_channel_map.append(map_dir)

    composite_norm_dir = dataset_out / "composite" / "normalized"
    composite_map_dir = dataset_out / "composite" / "intensity_map"
    ensure_dir(composite_norm_dir)
    ensure_dir(composite_map_dir)

    z_count = vol_czyx.shape[1]

    for z in range(z_count):
        norm_slices: list[np.ndarray] = []
        map_slices: list[np.ndarray] = []

        for c in range(vol_czyx.shape[0]):
            channel_name = str(channel_names[c])
            display_color = CHANNEL_COLOR_BY_INDEX.get(c, "gray")
            raw_slice = np.asarray(vol_czyx[c, z])

            norm_u8 = normalize_to_u8(raw_slice)
            norm_rgb = colorize_single_channel(norm_u8, display_color)
            map_rgb = make_intensity_distribution_map(raw_slice)

            norm_slices.append(norm_u8)
            map_slices.append(map_rgb)

            base = f"z{z:03d}__{sanitize_name(channel_name)}"
            tiff.imwrite(
                str(per_channel_norm[c] / f"{base}__normalized_{display_color}.tif"),
                norm_rgb,
                photometric="rgb",
            )
            tiff.imwrite(
                str(per_channel_map[c] / f"{base}__intensity_map.tif"),
                map_rgb,
                photometric="rgb",
            )

        composite_norm = make_normalized_composite(norm_slices)
        composite_map = make_intensity_composite(map_slices)

        tiff.imwrite(
            str(composite_norm_dir / f"z{z:03d}__normalized_composite_rgb.tif"),
            composite_norm,
            photometric="rgb",
        )
        tiff.imwrite(
            str(composite_map_dir / f"z{z:03d}__intensity_map_composite_rgb.tif"),
            composite_map,
            photometric="rgb",
        )

        print(
            f"    saved z={z:03d} -> per-channel normalized / per-channel intensity_map / composites"
        )


def run_one_dataset(zarr_dir: Path, level: int, out_root: Path) -> Path:
    """Load one OME-Zarr volume and export per-slice TIFF views.

    Args:
        zarr_dir (Path): Directory used for Zarr.
        level (int): Numerical value controlling level.
        out_root (Path): Directory used for out.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = run_one_dataset(
        ...     zarr_dir=Path("path/to/resource"),
        ...     level=1,
        ...     out_root=Path("path/to/resource"),
        ... )
    """
    zarr_dir = zarr_dir.resolve()
    out_root = out_root.resolve()

    if not zarr_dir.exists():
        raise FileNotFoundError(f"OME-Zarr not found: {zarr_dir}")

    meta = extract_ome_zarr_meta_for_compare(zarr_dir, level=level)
    vol_czyx, axes = load_ome_zarr_3d_czyx(
        zarr_dir,
        level=level,
        time=0,
        channels=None,
        as_numpy=True,
    )

    if axes != "czyx" or vol_czyx.ndim != 4:
        raise RuntimeError(
            f"Expected decoded data in (C,Z,Y,X). Got axes={axes!r}, shape={vol_czyx.shape}."
        )

    dataset_name = zarr_dir.parent.name
    dataset_out = out_root / sanitize_name(dataset_name) / f"L{level}"
    ensure_dir(dataset_out)

    channel_names = meta.get("channel_names")
    if not isinstance(channel_names, list) or len(channel_names) < vol_czyx.shape[0]:
        channel_names = [f"channel_{i}" for i in range(vol_czyx.shape[0])]

    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"  Input zarr      : {zarr_dir}")
    print(f"  Level           : {level}")
    print(f"  Decoded shape   : {vol_czyx.shape} (C,Z,Y,X)")
    print(f"  Decoded dtype   : {vol_czyx.dtype}")
    print(f"  Stored axes/meta: {meta.get('axes')}")
    print(f"  Channel names   : {channel_names}")
    print(f"  Output root     : {dataset_out}")

    save_per_channel_outputs(
        vol_czyx,
        channel_names=[str(x) for x in channel_names],
        dataset_out=dataset_out,
    )

    return dataset_out


def find_all_omezarr_3d(root_3d: Path) -> list[Path]:
    """Find every image.ome.zarr under results/img/3d_data.

    Args:
        root_3d (Path): Filesystem path used for root three-dimensional data.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = find_all_omezarr_3d(root_3d=Path("path/to/resource"))
    """
    if not root_3d.exists():
        return []
    return sorted(root_3d.rglob("image.ome.zarr"))


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    project_root = find_project_root(Path(__file__).resolve())
    default_root_3d = project_root / DEFAULT_RELATIVE_3D_ROOT
    default_out_root = project_root / DEFAULT_RELATIVE_OUT_ROOT

    ap = argparse.ArgumentParser(
        description=(
            "Export per-slice normalized channel TIFFs, per-slice intensity maps, "
            "and per-slice 3-channel composite TIFFs for all 3D OME-Zarr datasets."
        )
    )
    ap.add_argument("--root_3d", type=Path, default=default_root_3d)
    ap.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    ap.add_argument("--out_root", type=Path, default=default_out_root)
    args = ap.parse_args()

    root_3d = Path(args.root_3d)
    out_root = Path(args.out_root)
    level = int(args.level)

    zarr_list = find_all_omezarr_3d(root_3d)
    if not zarr_list:
        raise SystemExit(f"No image.ome.zarr found under: {root_3d}")

    print("\n=== Batch 3D intensity export ===")
    print(f"3D root    : {root_3d}")
    print(f"Level      : {level}")
    print(f"Output root: {out_root}")
    print(f"Datasets   : {len(zarr_list)}")

    ok = 0
    failed: list[tuple[Path, str]] = []

    for zarr_dir in zarr_list:
        try:
            run_one_dataset(zarr_dir=zarr_dir, level=level, out_root=out_root)
            ok += 1
        except Exception as e:
            failed.append((zarr_dir, str(e)))
            print(f"\nFAILED: {zarr_dir}")
            print(f"  reason: {e}")

    print("\nDONE")
    print(f"Successful datasets: {ok}")
    print(f"Failed datasets    : {len(failed)}")

    if failed:
        print("\nFailed list:")
        for p, err in failed:
            print(f"  - {p}")
            print(f"    {err}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
