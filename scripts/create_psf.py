from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import zarr
from PFT.core_prog_parts.psf_creator import generate_psfs_for_image
from PFT.core_prog_parts.psf_creator import _available_levels, _prompt_level


DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")


def _find_first_image_omezarr(root: Path) -> Path:
    hits = sorted(root.rglob("image.ome.zarr"))
    if not hits:
        raise FileNotFoundError(f"No image.ome.zarr found under: {root}")
    return hits[0]


def _parse_models_arg(s: str) -> Tuple[str, ...]:
    s = (s or "").strip()
    if not s or s.lower() == "all":
        return ("BW", "GL", "RW")

    parts = [p.strip().upper() for p in s.replace(",", " ").split() if p.strip()]
    allowed = {"BW", "GL", "RW"}
    bad = [p for p in parts if p not in allowed]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Allowed: BW, GL, RW, or 'all'")

    out = []
    for p in parts:
        if p not in out:
            out.append(p)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate PSFs (BW/GL/RW) for an OME-Zarr.")
    ap.add_argument("--zarr", default=None, help="Path to image.ome.zarr")
    ap.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--models", default="all", help="all | BW | GL | RW | BW,GL")
    ap.add_argument("--accuracy", default="Best")
    ap.add_argument("--level", type=int, default=None, help="OME-Zarr pyramid level (interactive if omitted)")
    ap.add_argument("--quiet", action="store_true")

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

    if args.level is None:
        levels = _available_levels(zarr_dir)
        level = _prompt_level(levels, default=2)
    else:
        level = args.level

    models = _parse_models_arg(args.models)

    print("\nPSF generation settings:")
    print(f"  zarr   : {zarr_dir}")
    print(f"  level  : {level}")
    print(f"  models : {models}")
    print(f"  accuracy: {args.accuracy}")

    outputs = generate_psfs_for_image(
        zarr_dir=zarr_dir,
        start_path=Path(__file__),
        level=level,
        models=models,
        accuracy=args.accuracy,
        quiet=args.quiet,
    )

    print("\nGenerated PSFs:")
    for (model, ch, lvl), p in outputs.items():
        print(f"  L{lvl} | {model} | {ch} -> {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())