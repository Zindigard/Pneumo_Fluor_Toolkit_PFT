"""
Create and validate metadata-matched PSFs for a 3D OME-Zarr image.

The script must know where ImageJ/Fiji and PSF Generator are installed. Pass
``--imagej-dir`` and ``--psf-creator-dir`` on a new computer. The latter can be
a directory containing ``PSF_Generator.jar`` or the JAR file itself.

Examples
--------
python scripts/denoising/create_psf.py --zarr D:/.../image.ome.zarr --level 2 \
    --models BW --imagej-dir D:/Fiji.app \
    --psf-creator-dir D:/PSFGenerator
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
    raise RuntimeError(f"Cannot locate PFT project root from {SCRIPT_FILE}")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.denoising.psf_creator import (  # noqa: E402
    _available_levels,
    _parse_models_arg,
    _prompt_level,
    generate_psfs_for_image,
)


def _find_first_image_omezarr(root: Path) -> Path:
    results = sorted(path for path in root.rglob("image.ome.zarr") if path.is_dir())
    if not results:
        raise FileNotFoundError(f"No image.ome.zarr found below {root}")
    return results[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or reuse metadata-matched, normalized 3D PSFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zarr", type=Path, default=None, help="Input image.ome.zarr")
    parser.add_argument(
        "--root-3d",
        type=Path,
        default=PROJECT_ROOT / "results" / "img" / "3d_data",
        help="Search root used when --zarr is omitted",
    )
    parser.add_argument("--models", default="BW", help="BW, GL, RW, or comma-separated values")
    parser.add_argument("--accuracy", default="Best", help="PSFGenerator accuracy setting")
    parser.add_argument("--level", type=int, default=None, help="OME-Zarr level; prompts when omitted")
    parser.add_argument(
        "--imagej-dir",
        type=Path,
        default=None,
        help="ESSENTIAL external path: ImageJ/Fiji installation folder containing Java and jars",
    )
    parser.add_argument(
        "--psf-creator-dir",
        type=Path,
        default=None,
        help="ESSENTIAL external path: PSF Generator folder or PSF_Generator.jar",
    )
    parser.add_argument("--no-reuse", action="store_true", help="Regenerate even when a suitable PSF exists")
    parser.add_argument("--quiet", action="store_true", help="Suppress Java output")
    args = parser.parse_args()

    zarr_dir = args.zarr or _find_first_image_omezarr(args.root_3d)
    zarr_dir = zarr_dir.expanduser().resolve()
    if not zarr_dir.is_dir():
        raise FileNotFoundError(zarr_dir)

    levels = _available_levels(zarr_dir)
    level = _prompt_level(levels, default=0) if args.level is None else int(args.level)
    if level not in levels:
        raise ValueError(f"Level {level} is unavailable; choose from {levels}")
    models = _parse_models_arg(args.models)

    print("\nPFT 3D PSF generation")
    print("=" * 72)
    print(f"OME-Zarr:          {zarr_dir}")
    print(f"Pyramid level:     {level}")
    print(f"Models:            {', '.join(models)}")
    print(f"ImageJ/Fiji:       {args.imagej_dir or 'project-managed default'}")
    print(f"PSF Generator:     {args.psf_creator_dir or 'ImageJ plugins/default'}")
    print("Existing PSF use:  validate and reuse" if not args.no_reuse else "Existing PSF use:  regenerate")

    outputs = generate_psfs_for_image(
        zarr_dir=zarr_dir,
        start_path=SCRIPT_FILE,
        level=level,
        models=models,
        accuracy=args.accuracy,
        quiet=args.quiet,
        imagej_dir=args.imagej_dir,
        psf_creator_dir=args.psf_creator_dir,
        reuse_existing=not args.no_reuse,
    )

    print("\nValidated PSFs")
    for (model, channel, selected_level), path in sorted(outputs.items()):
        print(f"  L{selected_level} | {model} | {channel} -> {path}")
    report_path = (
        PROJECT_ROOT / "results" / "psf" /
        f"psf_generation_report__{zarr_dir.parent.name}__L{level}.txt"
    )
    print(f"\nAudit report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
