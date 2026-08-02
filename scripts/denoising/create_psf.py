"""
Generate one reusable three-channel master PSF set for all 3D stacks.

The script reads level-0 metadata from one reference ``image.ome.zarr`` and
creates exactly three normalized PSFs, one for each channel wavelength. The
master PSFs are stored once in ``results/psf/master`` and are reused by every
compatible stack during deconvolution.

On a new computer, provide both paths explicitly:

* ``--imagej-dir``: ImageJ/Fiji installation folder containing Java and JARs;
* ``--psf-creator-dir``: PSF Generator folder or ``PSF_Generator.jar``.

The default wavelength set produces:

* ``psf_405nm_L0.tif`` for the blue channel;
* ``psf_488nm_L0.tif`` for the green channel;
* ``psf_561nm_L0.tif`` for the red channel;
* ``psf_master_metadata.json`` containing the reference microscope metadata.
u need to set location of imagej and plugin and they should exist.
$FIJI = "D:\Thesis\Pneumo_Fluor_Toolkit_PFT\.cache\fiji\Fiji"
$PY = (Get-Command python).Source
$PSFGEN = "D:\Thesis\Pneumo_Fluor_Toolkit_PFT\.cache\fiji\Fiji\plugins\PSF_Generator.jar"
$REF = (
    Get-ChildItem `
        "results\img\3d_data" `
        -Recurse `
        -Directory `
        -Filter "image.ome.zarr" |
    Sort-Object FullName |
    Select-Object -First 1
).FullName
plus create reference for master psf and then run the script with the following command

& $PY scripts\denoising\create_psf.py `
    --reference-zarr "$REF" `
    --model BW `
    --imagej-dir "$FIJI" `
    --psf-creator-dir "$PSFGEN"
master psf will be provided but if not use the above command to create master psf. 

Example
-------
python scripts/denoising/create_psf.py \
    --reference-zarr D:/.../image.ome.zarr \
    --model BW \
    --imagej-dir D:/Fiji.app \
    --psf-creator-dir D:/PSFGenerator
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
    raise RuntimeError(f"Cannot locate PFT project root from {SCRIPT_FILE}")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.denoising.psf_creator import (  # noqa: E402
    MASTER_LEVEL,
    _parse_models_arg,
    generate_or_reuse_master_psfs,
    master_psf_directory,
)


def _find_first_image_omezarr(root: Path) -> Path:
    """Find first image OME-Zarr in the available data or project structure.

    Args:
        root (Path): Root directory used to resolve relative project paths.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _find_first_image_omezarr(root=Path("path/to/resource"))
    """
    results = sorted(path for path in root.rglob("image.ome.zarr") if path.is_dir())
    if not results:
        raise FileNotFoundError(f"No image.ome.zarr found below {root}")
    return results[0]


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Generate or reuse one level-0 master PSF per channel wavelength.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference-zarr", "--zarr",
        dest="reference_zarr",
        type=Path,
        default=None,
        help="Reference level-0 image.ome.zarr used once to define the master PSF set",
    )
    parser.add_argument(
        "--root-3d",
        type=Path,
        default=PROJECT_ROOT / "results" / "img" / "3d_data",
        help="Search root used when --reference-zarr is omitted",
    )
    parser.add_argument(
        "--model",
        default="BW",
        choices=("BW", "GL", "RW"),
        help="Single active theoretical PSF model for all three master wavelengths",
    )
    parser.add_argument("--accuracy", default="Best", help="PSFGenerator accuracy setting")
    parser.add_argument(
        "--level",
        type=int,
        default=MASTER_LEVEL,
        help="Fixed master level. Only level 0 is accepted",
    )
    parser.add_argument(
        "--imagej-dir",
        type=Path,
        default=None,
        help="ESSENTIAL external path: ImageJ/Fiji installation folder containing Java and JARs",
    )
    parser.add_argument(
        "--psf-creator-dir",
        type=Path,
        default=None,
        help="ESSENTIAL external path: PSF Generator folder or PSF_Generator.jar",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Replace the complete master set after an intentional metadata or model change",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Regenerate all three PSFs even when the existing compatible files are valid",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress Java output")
    args = parser.parse_args()

    if args.level != MASTER_LEVEL:
        raise ValueError("The reusable master PSF set is defined only for OME-Zarr level 0")
    reference_zarr = args.reference_zarr or _find_first_image_omezarr(args.root_3d)
    reference_zarr = reference_zarr.expanduser().resolve()
    if not reference_zarr.is_dir():
        raise FileNotFoundError(reference_zarr)
    model = _parse_models_arg(args.model)[0]

    print("\nPFT reusable master PSF generation")
    print("=" * 72)
    print(f"Reference OME-Zarr: {reference_zarr}")
    print("Reference level:    0")
    print(f"PSF model:          {model}")
    print("Channel matching:   wavelength metadata")
    print("Display convention: 405 nm blue, 488 nm green, 561 nm red")
    print(f"ImageJ/Fiji:        {args.imagej_dir or 'project-managed default'}")
    print(f"PSF Generator:      {args.psf_creator_dir or 'ImageJ plugins/default'}")
    print(f"Master directory:   {master_psf_directory(PROJECT_ROOT)}")

    outputs = generate_or_reuse_master_psfs(
        reference_zarr=reference_zarr,
        start_path=SCRIPT_FILE,
        model=model,
        accuracy=args.accuracy,
        quiet=args.quiet,
        imagej_dir=args.imagej_dir,
        psf_creator_dir=args.psf_creator_dir,
        reuse_existing=not args.no_reuse,
        regenerate=args.regenerate,
    )

    print("\nValidated reusable master PSFs")
    for wavelength, path in sorted(outputs.items()):
        color = "blue" if wavelength < 450 else "green" if wavelength < 530 else "red"
        print(f"  {wavelength:g} nm | {color:5s} -> {path}")
    print(f"\nMaster metadata: {master_psf_directory(PROJECT_ROOT) / 'psf_master_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
