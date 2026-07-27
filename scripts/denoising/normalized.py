from __future__ import annotations
# Permit direct execution from a source checkout after scripts were grouped into subpackages.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()
for _pft_parent in [_PFT_SCRIPT_FILE.parent, *_PFT_SCRIPT_FILE.parents]:
    _pft_src = _pft_parent / "src"
    if (_pft_src / "PFT").exists():
        if str(_pft_src) not in _pft_sys.path:
            _pft_sys.path.insert(0, str(_pft_src))
        break

def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Locate the repository root independently of script nesting depth."""
    current = (start or _PFT_SCRIPT_FILE).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "PFT").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return _PFTPath.cwd()


from pathlib import Path
import sys
import csv

_THIS_FILE = Path(__file__).resolve()
for _p in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (_p / "src" / "PFT").exists():
        _SRC_DIR = _p / "src"
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        break

import argparse

import numpy as np
import tifffile
import zarr

from PFT.core_prog_parts.common_paths import ensure_dir


DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
DEFAULT_3D_ROOT = DEFAULT_PROJECT_ROOT / "results" / "img" / "3d_data"
DEFAULT_OUT_ROOT = DEFAULT_PROJECT_ROOT / "results" / "z_slices_export"
UINT16_MAX = 65535.0


def list_image_folders(root_3d: Path) -> list[Path]:
    return [p.parent for p in sorted(root_3d.rglob("image.ome.zarr"))]


def available_pyramid_levels(image_omezarr: Path) -> list[int]:
    root = zarr.open_group(str(image_omezarr), mode="r")
    ms = root.attrs.get("multiscales")
    if not ms or not isinstance(ms, list) or not ms[0].get("datasets"):
        return [0]
    ds = ms[0]["datasets"]
    return list(range(len(ds)))


def prompt_int(prompt: str, *, min_v: int | None = None, max_v: int | None = None, default: int | None = None) -> int:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            v = default
        else:
            try:
                v = int(s)
            except ValueError:
                print("Please enter an integer.")
                continue
        if min_v is not None and v < min_v:
            print(f"Value must be >= {min_v}")
            continue
        if max_v is not None and v > max_v:
            print(f"Value must be <= {max_v}")
            continue
        return v


def interactive_select_dataset(root_3d: Path) -> Path:
    folders = list_image_folders(root_3d)
    if not folders:
        raise SystemExit(f"No image.ome.zarr found under: {root_3d}")

    print("\nAvailable 3D datasets:")
    for i, f in enumerate(folders):
        rel = f.relative_to(root_3d) if root_3d in f.parents else f
        print(f"  [{i}] {rel}")

    idx = prompt_int(
        f"\nSelect dataset index [0-{len(folders)-1}]: ",
        min_v=0,
        max_v=len(folders) - 1,
    )
    in_zarr = folders[idx] / "image.ome.zarr"
    if not in_zarr.exists():
        raise SystemExit(f"Missing: {in_zarr}")
    return in_zarr


def _get_axes(zarr_path: Path) -> str:
    root = zarr.open_group(str(zarr_path), mode="r")
    ms = root.attrs.get("multiscales")
    if ms and isinstance(ms, list) and ms[0].get("axes"):
        axes = ms[0]["axes"]
        if isinstance(axes, list):
            return "".join(a["name"] if isinstance(a, dict) else str(a) for a in axes).lower()
        return str(axes).lower()
    return "czyx"


def _open_level_array(zarr_path: Path, level: int):
    root = zarr.open_group(str(zarr_path), mode="r")
    arr = root[str(level)]
    axes = _get_axes(zarr_path)
    return arr, axes


def _read_z_slice_cyx(arr, axes: str, z: int, t: int = 0) -> np.ndarray:
    axes = axes.lower()
    sel = []
    for ax in axes:
        if ax == "t":
            sel.append(t)
        elif ax == "c":
            sel.append(slice(None))
        elif ax == "z":
            sel.append(z)
        elif ax in ("y", "x"):
            sel.append(slice(None))
        else:
            sel.append(0)

    slab = np.asarray(arr[tuple(sel)])
    rem_axes = "".join(a for a in axes if a not in ("t", "z"))

    if rem_axes != "cyx":
        try:
            perm = [rem_axes.index("c"), rem_axes.index("y"), rem_axes.index("x")]
            slab = np.transpose(slab, perm)
        except ValueError as exc:
            raise ValueError(f"Unsupported remaining axes order '{rem_axes}' after slicing.") from exc
    return slab


def _scale_to_u16(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint16)
    x = (img.astype(np.float32) - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    return (x * UINT16_MAX).astype(np.uint16)


def _normalize_to_u16(img: np.ndarray) -> np.ndarray:
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    return _scale_to_u16(img, vmin, vmax)


def _channel_rgb_weights(ch: int) -> tuple[int, int, int]:
    palette = [
        (0, 0, 1),  # ch0 blue
        (0, 1, 0),  # ch1 green
        (1, 0, 0),  # ch2 red
        (1, 1, 0),  # ch3 yellow
        (1, 0, 1),  # ch4 magenta
        (0, 1, 1),  # ch5 cyan
    ]
    return palette[ch % len(palette)]


def _colorize_u16(img: np.ndarray, ch: int) -> np.ndarray:
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint16)
    r_w, g_w, b_w = _channel_rgb_weights(ch)
    if r_w:
        rgb[..., 0] = img
    if g_w:
        rgb[..., 1] = img
    if b_w:
        rgb[..., 2] = img
    return rgb


def _raw_stats(img: np.ndarray) -> dict[str, float]:
    raw_min = float(np.min(img))
    raw_max = float(np.max(img))
    raw_range = raw_max - raw_min
    used_percent_of_u16 = (raw_range / UINT16_MAX) * 100.0 if UINT16_MAX > 0 else 0.0
    stretch_factor = (UINT16_MAX / raw_range) if raw_range > 0 else 0.0
    normalization_gain_percent = (stretch_factor - 1.0) * 100.0 if stretch_factor > 0 else 0.0
    return {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_range": raw_range,
        "used_percent_of_u16": used_percent_of_u16,
        "normalization_gain_percent": normalization_gain_percent,
    }


def export_z_slices_raw_and_normalized_per_channel(
    *,
    input_zarr: Path,
    level: int,
    out_dir: Path,
) -> None:
    arr, axes = _open_level_array(input_zarr, level)

    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array for slice export. Got shape: {arr.shape}")

    c_count, z_count, y_count, x_count = arr.shape

    raw_dir = ensure_dir(out_dir / "raw")
    normalized_dir = ensure_dir(out_dir / "normalized")
    stats_dir = ensure_dir(out_dir / "stats")

    print("\n[EXPORT] Saving z-slices per channel")
    print(f"  input : {input_zarr}")
    print(f"  level : {level}")
    print(f"  shape : {arr.shape}")
    print(f"  axes  : {axes}")
    print(f"  raw   : {raw_dir}")
    print(f"  norm  : {normalized_dir}")
    print(f"  stats : {stats_dir}")
    print("  export: separate color TIFF for each channel and z-slice")
    print("  normalization: current per-slice, per-channel min-max to uint16")
    print("  statistics: raw_min, raw_max, raw_range, used_percent_of_u16, normalization_gain_percent")

    saved_raw = 0
    saved_norm = 0

    for ch in range(c_count):
        raw_ch_dir = ensure_dir(raw_dir / f"ch_{ch}")
        norm_ch_dir = ensure_dir(normalized_dir / f"ch_{ch}")
        stats_csv = stats_dir / f"ch_{ch}__raw_stats.csv"

        with stats_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "channel",
                    "z",
                    "raw_min",
                    "raw_max",
                    "raw_range",
                    "used_percent_of_u16",
                    "normalization_gain_percent",
                ],
            )
            writer.writeheader()

            for z in range(z_count):
                cyx = _read_z_slice_cyx(arr, axes, z)
                if cyx.shape != (c_count, y_count, x_count):
                    raise ValueError(
                        f"Unexpected slice shape at z={z}: {cyx.shape}, expected {(c_count, y_count, x_count)}"
                    )

                img = cyx[ch]
                norm_img = _normalize_to_u16(img)
                color_raw = _colorize_u16(img.astype(np.uint16, copy=False), ch)
                color_norm = _colorize_u16(norm_img, ch)

                raw_path = raw_ch_dir / f"ch{ch}_z{z:03d}__raw_color.tif"
                norm_path = norm_ch_dir / f"ch{ch}_z{z:03d}__normalized_u16_color.tif"

                tifffile.imwrite(str(raw_path), color_raw, photometric="rgb")
                tifffile.imwrite(str(norm_path), color_norm, photometric="rgb")
                saved_raw += 1
                saved_norm += 1

                row = {"channel": ch, "z": z}
                row.update(_raw_stats(img))
                writer.writerow(row)

    print(f"[EXPORT] Saved {saved_raw} raw color channel-slices and {saved_norm} normalized color channel-slices.")
    print(f"[EXPORT] Saved per-channel raw statistics CSV files in: {stats_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export all Z-slices from 3D OME-Zarr as raw and normalized per-channel color TIFFs plus raw statistics CSVs.")
    ap.add_argument("--root_3d", default=str(DEFAULT_3D_ROOT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--folder", default=None, help="Folder containing image.ome.zarr (skip selection)")
    ap.add_argument("--level", type=int, default=None, help="OME-Zarr pyramid level (default: 2 if available else 0)")
    args = ap.parse_args()

    root_3d = Path(args.root_3d)
    out_root = ensure_dir(Path(args.out_root))

    if args.folder:
        input_zarr = Path(args.folder) / "image.ome.zarr"
        if not input_zarr.exists():
            raise SystemExit(f"Missing: {input_zarr}")
    else:
        input_zarr = interactive_select_dataset(root_3d)

    levels = available_pyramid_levels(input_zarr)
    default_level = 2 if 2 in levels else 0
    if args.level is None:
        print(f"\nAvailable pyramid levels: {levels}")
        level = prompt_int(
            f"Choose level (default {default_level}): ",
            min_v=min(levels),
            max_v=max(levels),
            default=default_level,
        )
        if level not in levels:
            raise SystemExit(f"Invalid level {level}. Must be one of: {levels}")
    else:
        level = int(args.level)
        if level not in levels:
            raise SystemExit(f"Invalid level {level}. Must be one of: {levels}")

    dataset_name = input_zarr.parent.name
    export_dir = ensure_dir(out_root / dataset_name / f"level_{level}")

    export_z_slices_raw_and_normalized_per_channel(
        input_zarr=input_zarr,
        level=level,
        out_dir=export_dir,
    )

    print("\nDONE")
    print(f"  export dir: {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
