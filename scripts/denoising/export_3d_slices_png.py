r"""Export source 3D OME-Zarr Z-planes as normalized merged-RGB PNGs.

Each source volume has one configured target slice. The interactive menu and
batch summary display that target, while all 40 planes are exported for visual
inspection. Normalization is independent per slice and channel using P1-P99.8.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
import numpy as np
import zarr

SCRIPT_PATH = Path(__file__).resolve()
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.8
WAVELENGTH_TO_RGB: tuple[tuple[float, int, str], ...] = (
    (405.0, 2, "blue"),
    (488.0, 1, "green"),
    (561.0, 0, "red"),
)


def find_project_root() -> Path:
    for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("Cannot locate the PFT project root.")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare  # noqa: E402
from PFT.core_prog_parts.denoising.metadata_3d import ChannelOptics, resolve_channel_optics  # noqa: E402
from PFT.core_prog_parts.denoising.validation_3d import (  # noqa: E402
    relative_volume_path,
    target_slice_for_volume,
    volume_key,
)


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (Path.cwd() / expanded).resolve()


def discover_zarrs(search_root: Path) -> list[Path]:
    root = resolve_path(search_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Search root does not exist: {root}")
    stores = sorted(path.resolve() for path in root.rglob("image.ome.zarr") if path.is_dir())
    if not stores:
        raise FileNotFoundError(f"No source image.ome.zarr stores found below: {root}")
    return stores


def choose_zarr_interactively(stores: Sequence[Path], image_root: Path) -> Path:
    print("\nChoose a 3D OME-Zarr image")
    print("  [0] Exit")
    for index, path in enumerate(stores, start=1):
        target = target_slice_for_volume(path, image_root)
        print(f"  [{index}] {path.relative_to(PROJECT_ROOT)} | target Z{target}")
    while True:
        answer = input("Enter number: ").strip()
        if answer == "0":
            raise SystemExit(0)
        try:
            selected = int(answer)
        except ValueError:
            print("Enter a valid number.")
            continue
        if 1 <= selected <= len(stores):
            return stores[selected - 1]
        print(f"Enter a number from 0 to {len(stores)}.")


def rgb_assignment(wavelength_nm: float) -> tuple[int, str]:
    reference_nm, rgb_index, colour = min(
        WAVELENGTH_TO_RGB,
        key=lambda item: abs(float(wavelength_nm) - item[0]),
    )
    if abs(float(wavelength_nm) - reference_nm) > 35.0:
        raise ValueError(f"Unsupported channel wavelength: {wavelength_nm:g} nm")
    return rgb_index, colour


def normalize_plane(
    plane: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(plane, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Plane contains no finite pixels")
    low, high = (float(value) for value in np.percentile(finite, [low_percentile, high_percentile]))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32), low, high
    output = np.clip((values - low) / (high - low), 0.0, 1.0)
    output[~np.isfinite(output)] = 0.0
    return output.astype(np.float32, copy=False), low, high


def validate_channels(optics: Sequence[ChannelOptics], channel_count: int) -> None:
    if len(optics) != channel_count:
        raise ValueError(f"Channel metadata count {len(optics)} differs from C={channel_count}")
    assignments = [rgb_assignment(channel.wavelength_nm)[0] for channel in optics]
    if sorted(assignments) != [0, 1, 2]:
        raise ValueError(f"Expected unique red, green, and blue channels; resolved {assignments}")


def default_output_dir(zarr_path: Path, level: int) -> Path:
    return zarr_path.parent / f"image_normalized_png_slices_L{level}"


def clean_previous_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("z*_normalized_rgb.png", "normalization_values.csv", "export_record.txt"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def export_slices(
    zarr_path: Path,
    *,
    image_root: Path,
    level: int,
    output_dir: Path,
    expected_slices: int,
    low_percentile: float,
    high_percentile: float,
) -> dict[str, str]:
    target = target_slice_for_volume(zarr_path, image_root)
    meta = extract_ome_zarr_meta_for_compare(zarr_path, level=level)
    axes = str(meta.get("axes") or "").lower()
    if axes != "czyx":
        raise ValueError(f"Expected CZYX data, received axes={axes!r}")
    root = zarr.open_group(str(zarr_path), mode="r")
    array_path = str(meta.get("array_path") or "0")
    array = root[array_path]
    if array.ndim != 4:
        raise ValueError(f"Expected CZYX array, received shape={array.shape}")
    c_count, z_count, y_count, x_count = (int(value) for value in array.shape)
    if expected_slices > 0 and z_count != expected_slices:
        raise ValueError(f"Expected Z={expected_slices}, received Z={z_count}")
    if not 2 <= target <= z_count - 1:
        raise ValueError(f"Configured target Z{target} lacks complete context for Z={z_count}")

    optics = resolve_channel_optics(zarr_path, level=level)
    validate_channels(optics, c_count)
    assignments = {channel.index: rgb_assignment(channel.wavelength_nm) for channel in optics}
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(output_dir)
    normalization_rows: list[dict[str, object]] = []

    print(f"\n{volume_key(zarr_path, image_root)} | target Z{target}")
    print(f"Output: {output_dir}")
    for z_index in range(z_count):
        plane_cyx = np.asarray(array[:, z_index], dtype=np.float32)
        rgb = np.zeros((y_count, x_count, 3), dtype=np.float32)
        for channel in optics:
            rgb_index, colour = assignments[channel.index]
            normalized, low, high = normalize_plane(
                plane_cyx[channel.index],
                low_percentile=low_percentile,
                high_percentile=high_percentile,
            )
            rgb[..., rgb_index] = normalized
            normalization_rows.append({
                "slice_1based": z_index + 1,
                "is_configured_target": int(z_index + 1 == target),
                "channel_index": channel.index,
                "channel_name": channel.name,
                "wavelength_nm": channel.wavelength_nm,
                "display_colour": colour,
                "low_value": low,
                "high_value": high,
            })
        output_png = output_dir / f"z{z_index + 1:03d}_normalized_rgb.png"
        iio.imwrite(output_png, np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
        marker = " TARGET" if z_index + 1 == target else ""
        print(f"[{z_index + 1:02d}/{z_count:02d}] {output_png.name}{marker}")

    csv_path = output_dir / "normalization_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalization_rows[0]))
        writer.writeheader()
        writer.writerows(normalization_rows)

    record_path = output_dir / "export_record.txt"
    record_path.write_text(
        "\n".join([
            "PFT 3D normalized PNG export",
            "=" * 72,
            f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
            f"Source OME-Zarr: {zarr_path}",
            f"Source volume key: {volume_key(zarr_path, image_root)}",
            f"Configured target slice: Z{target}",
            f"2.5D target context: Z{target-1}/Z{target}/Z{target+1}",
            f"Source shape CZYX: {(c_count, z_count, y_count, x_count)}",
            f"Normalization: P{low_percentile:g}-P{high_percentile:g} independently per slice and channel",
            f"Output count: {z_count}",
        ]) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "sample": volume_key(zarr_path, image_root),
        "target_slice_1based": str(target),
        "input_omezarr": str(zarr_path),
        "output_directory": str(output_dir),
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export normalized merged-RGB PNGs for one or all mapped source 3D stacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zarr", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Export every mapped source image.ome.zarr")
    parser.add_argument("--search-root", type=Path, default=PROJECT_ROOT / "results" / "img" / "3d_data")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-slices", type=int, default=40)
    parser.add_argument("--low-percentile", type=float, default=LOW_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=HIGH_PERCENTILE)
    args = parser.parse_args()
    if args.zarr is not None and args.all:
        parser.error("Use either --zarr or --all, not both")
    if not 0 <= args.low_percentile < args.high_percentile <= 100:
        parser.error("Percentiles must satisfy 0 <= low < high <= 100")

    image_root = resolve_path(args.search_root)
    stores = discover_zarrs(image_root)
    if args.all:
        selected = stores
    elif args.zarr is not None:
        selected = [resolve_path(args.zarr)]
    else:
        selected = [choose_zarr_interactively(stores, image_root)]

    central_output = resolve_path(args.output_dir) if args.output_dir is not None else None
    rows: list[dict[str, str]] = []
    failures = 0
    for zarr_path in selected:
        try:
            output_dir = (
                central_output / relative_volume_path(zarr_path, image_root)
                if central_output is not None and len(selected) > 1
                else central_output or default_output_dir(zarr_path, args.level)
            )
            rows.append(export_slices(
                zarr_path,
                image_root=image_root,
                level=args.level,
                output_dir=output_dir,
                expected_slices=args.expected_slices,
                low_percentile=args.low_percentile,
                high_percentile=args.high_percentile,
            ))
        except Exception as error:
            failures += 1
            rows.append({
                "status": "FAIL",
                "sample": volume_key(zarr_path, image_root),
                "target_slice_1based": "",
                "input_omezarr": str(zarr_path),
                "output_directory": "",
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"FAIL: {zarr_path}: {type(error).__name__}: {error}")

    if len(selected) > 1:
        summary_root = central_output or image_root
        summary_root.mkdir(parents=True, exist_ok=True)
        summary_path = summary_root / f"normalized_png_export_summary_L{args.level}.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nBatch summary: {summary_path}")
    print(f"Completed: {len(selected) - failures}/{len(selected)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
