r"""Validate 3D raw data and sparse 2.5D U-Net annotations before processing.

Each source volume has one manually selected target slice defined in the shared
3D target-slice map. The binary TIFF is stored as
``results/training_files/U-net/3d_25d/<experiment>/<sample>/zNNN_mask.tif``.
Neighbouring Z-1 and Z+1 slices are read automatically during 2.5D training.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/check_3d_inputs.py --help

Representative execution:

    python scripts/denoising/check_3d_inputs.py \
        --root results/img/3d_data \
        --level 0 \
        --mask-root results/training_files/U-net \
        --check-masks \
        --annotated-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_FILE = Path(__file__).resolve()


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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.denoising.validation_3d import (  # noqa: E402
    annotation_sample_dir,
    check_3d_sample,
    target_slice_for_volume,
    write_readiness_report,
)


def _find_zarrs(root: Path) -> list[Path]:
    """Find zarrs in the available data or project structure.

    Args:
        root (Path): Root directory used to resolve relative project paths.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = _find_zarrs(root=Path("path/to/resource"))
    """
    if root.name.endswith(".ome.zarr") and root.is_dir():
        return [root]
    return sorted(path for path in root.rglob("image.ome.zarr") if path.is_dir())


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
        "--check-masks",
        action="store_true",
        help="Check configured target masks; missing folders or masks produce warnings, not failures",
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
        zarrs = [path for path in zarrs if annotation_sample_dir(args.mask_root, path).is_dir()]
        if not zarrs:
            raise FileNotFoundError(
                f"No annotated sample folders matching extracted OME-Zarr stores were found under {args.mask_root}"
            )

    failures = 0
    warnings_count = 0
    for zarr_path in zarrs:
        target_slice = target_slice_for_volume(zarr_path)
        report = check_3d_sample(
            zarr_path,
            level=args.level,
            mask_root=args.mask_root,
            training_slices_1based=(target_slice,),
            require_masks=require_masks,
        )
        text_path, json_path = write_readiness_report(report, args.out)
        status = report.status
        failures += int(not report.passed)
        warnings_count += int(report.has_warnings)
        print(f"[{status}] {report.sample} -> {text_path}")
        print(f"       JSON -> {json_path}")
    print(
        f"\nChecked {len(zarrs)} sample(s); "
        f"samples_with_warnings={warnings_count}; failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
