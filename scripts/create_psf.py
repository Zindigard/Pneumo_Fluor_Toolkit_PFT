"""
Run PSF generation for 3 models (1 per channel).

Outputs
-------
Configs: <project_root>\results\psf\generated\PSFGenerator_BW_<channel>_LambdaXXXnm.txt
PSFs   : <project_root>\results\psf\generated\psf_BW_<channel>_LambdaXXXnm.tif
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PFT.core_prog_parts.psf_creator import generate_psfs_for_image_all_models


DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
DEFAULT_3D_ROOT = DEFAULT_PROJECT_ROOT / "results" / "img" / "3d_data"


def _find_first_image_omezarr(root: Path) -> Path:
    hits = sorted(root.rglob("image.ome.zarr"))
    if not hits:
        raise FileNotFoundError(f"No image.ome.zarr found under: {root}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate PSF for BW model only (per channel).")
    ap.add_argument("--zarr", default=None, help="Path to image.ome.zarr. If omitted, first found under default 3D root is used.")
    ap.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT), help="Project root (default: D:\\Thesis\\Pneumo_Fluor_Toolkit_PFT)")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    if not project_root.exists():
        raise FileNotFoundError(project_root)

    if args.zarr:
        zarr_dir = Path(args.zarr)
    else:
        zarr_dir = _find_first_image_omezarr(project_root / "results" / "img" / "3d_data")

    if not zarr_dir.exists():
        raise FileNotFoundError(zarr_dir)

    # BW only
    outputs = generate_psfs_for_image_all_models(
        zarr_dir=zarr_dir,
        start_path=Path(__file__),
        models=("BW",),
        accuracy="Best",
        quiet=args.quiet,
        # Optional override (not needed if your defaults are correct):
        # channel_wavelength_nm={
        #     "TV1-T1-SR": 405.0,
        #     "TV1-T2-SR": 488.0,
        #     "TV1-T3-SR": 561.0,
        # },
    )

    print("Generated PSFs (BW):")
    for (model, ch), p in outputs.items():
        print(f"  {model} | {ch} -> {p}")

    # ------------------------------------------------------------
    # If later you want ALL 3 models at once, uncomment this block:
    #
    # outputs = generate_psfs_for_image_all_models(
    #     zarr_dir=zarr_dir,
    #     start_path=Path(__file__),
    #     models=("BW", "GL", "RW"),
    #     accuracy="Best",
    #     quiet=args.quiet,
    # )
    # ------------------------------------------------------------

    return 0


if __name__ == "__main__":
    raise SystemExit(main())