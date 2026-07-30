"""Validate 3D raw data and sparse 2.5D U-Net annotations before processing.

The default manually annotated target slices are Z10, Z24, and Z30. Each target is stored as one binary TIFF, for example
``results/training_files/U-net/3d_25d/<sample>/z010_mask.tif``. Only the middle
slice is annotated; neighbouring slices are read automatically during 2.5D
training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.denoising.validation_3d import (  # noqa: E402
    DEFAULT_TRAINING_SLICES_1BASED,
    check_3d_sample,
    write_readiness_report,
)


def _parse_slices(value: str) -> tuple[int, ...]:
    numbers = tuple(sorted({int(item) for item in value.replace(",", " ").split()}))
    if not numbers or any(number < 2 for number in numbers):
        raise ValueError("Training slices must be positive one-based indices with neighbours")
    return numbers


def _find_zarrs(root: Path) -> list[Path]:
    if root.name.endswith(".ome.zarr") and root.is_dir():
        return [root]
    return sorted(path for path in root.rglob("image.ome.zarr") if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check all 3D OME-Zarr and 2.5D U-Net annotation inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results" / "img" / "3d_data")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "training_files" / "U-net" / "3d_25d",
    )
    parser.add_argument(
        "--slices",
        default=",".join(str(value) for value in DEFAULT_TRAINING_SLICES_1BASED),
        help="One-based manually annotated target slices",
    )
    parser.add_argument(
        "--check-masks",
        action="store_true",
        help="Require Z10, Z24, and Z30 masks in the checked sample folders",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Check only samples that already have a folder under --mask-root; implies --check-masks",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Deprecated compatibility option; raw-data-only checking is now the default",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "noise_analysis" / "3d" / "input_check",
    )
    args = parser.parse_args()

    if args.raw_only and (args.check_masks or args.annotated_only):
        parser.error("--raw-only cannot be combined with --check-masks or --annotated-only")

    zarrs = _find_zarrs(args.root)
    if not zarrs:
        raise FileNotFoundError(f"No image.ome.zarr found under {args.root}")

    require_masks = bool(args.check_masks or args.annotated_only)
    if args.annotated_only:
        zarrs = [path for path in zarrs if (args.mask_root / path.parent.name).is_dir()]
        if not zarrs:
            raise FileNotFoundError(
                f"No annotated sample folders matching extracted OME-Zarr stores were found under {args.mask_root}"
            )

    slices = _parse_slices(args.slices)
    failures = 0
    for zarr_path in zarrs:
        report = check_3d_sample(
            zarr_path,
            level=args.level,
            mask_root=args.mask_root,
            training_slices_1based=slices,
            require_masks=require_masks,
        )
        text_path, json_path = write_readiness_report(report, args.out)
        status = "PASS" if report.passed else "FAIL"
        failures += int(not report.passed)
        print(f"[{status}] {report.sample} -> {text_path}")
        print(f"       JSON -> {json_path}")
    print(f"\nChecked {len(zarrs)} sample(s); failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
