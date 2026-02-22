"""
Run PSF generation for 3 models (per channel).

"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
from PFT.core_prog_parts.psf_creator import generate_psfs_for_image_all_models


DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")


def _find_first_image_omezarr(root: Path) -> Path:
    hits = sorted(root.rglob("image.ome.zarr"))
    if not hits:
        raise FileNotFoundError(f"No image.ome.zarr found under: {root}")
    return hits[0]


def _parse_models_arg(s: str) -> Tuple[str, ...]:
    """
    Accepts:
      "all" -> ("BW","GL","RW")
      "BW" or "BW,GL" or "BW GL RW"
    """
    s = (s or "").strip()
    if not s or s.lower() == "all":
        return ("BW", "GL", "RW")

    parts = [p.strip().upper() for p in s.replace(",", " ").split() if p.strip()]
    allowed = {"BW", "GL", "RW"}
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Allowed: BW, GL, RW, or 'all'")
    # keep order but unique
    out = []
    for p in parts:
        if p not in out:
            out.append(p)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate PSFs for BW/GL/RW (per channel).")
    ap.add_argument(
        "--zarr",
        default=None,
        help="Path to image.ome.zarr. If omitted, first found under results/img/3d_data is used.",
    )
    ap.add_argument(
        "--project_root",
        default=str(DEFAULT_PROJECT_ROOT),
        help=r"Project root (default: D:\Thesis\Pneumo_Fluor_Toolkit_PFT)",
    )
    ap.add_argument("--models", default="all", help="Models to generate: all | BW | GL | RW | BW,GL | BW GL RW")
    ap.add_argument("--accuracy", default="Best", help="PSFGenerator accuracy (e.g. Best, Good, Fast)")
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

    models = _parse_models_arg(args.models)

    outputs = generate_psfs_for_image_all_models(
        zarr_dir=zarr_dir,
        start_path=Path(__file__),
        models=models,
        accuracy=args.accuracy,
        quiet=args.quiet,
        # Optional override:
        # channel_wavelength_nm={
        #     "TV1-T1-SR": 405.0,
        #     "TV1-T2-SR": 488.0,
        #     "TV1-T3-SR": 561.0,
        # },
    )

    print(f"Generated PSFs (models={models}):")
    for (model, ch), p in outputs.items():
        print(f"  {model} | {ch} -> {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())