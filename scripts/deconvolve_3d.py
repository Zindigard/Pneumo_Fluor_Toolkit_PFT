from __future__ import annotations
import argparse
from pathlib import Path

from PFT.core_prog_parts.Deconvolution_omezarr import (
    deconvolve_omezarr_3ch_to_omezarr,
    DEFAULT_CHANNEL_WAVELENGTH_NM,
)

DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
DEFAULT_3D_ROOT = DEFAULT_PROJECT_ROOT / "results" / "img" / "3d_data"
DEFAULT_OUT_ROOT = DEFAULT_PROJECT_ROOT / "results" / "deconv"


def list_image_folders(root_3d: Path) -> list[Path]:
    return [p.parent for p in sorted(root_3d.rglob("image.ome.zarr"))]


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


def prompt_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    opts_lower = [o.lower() for o in options]
    while True:
        s = input(prompt).strip().lower()
        if s == "" and default is not None:
            return default
        if s in opts_lower:
            return options[opts_lower.index(s)]
        print(f"Choose one of: {', '.join(options)}")


def interactive_select_dataset(root_3d: Path) -> Path:
    folders = list_image_folders(root_3d)
    if not folders:
        raise SystemExit(f"No image.ome.zarr found under: {root_3d}")

    print("\nAvailable 3D datasets:")
    for i, f in enumerate(folders):
        rel = f.relative_to(root_3d) if root_3d in f.parents else f
        print(f"  [{i}] {rel}")

    idx = prompt_int(f"\nSelect dataset index [0-{len(folders)-1}]: ", min_v=0, max_v=len(folders) - 1)
    in_zarr = folders[idx] / "image.ome.zarr"
    if not in_zarr.exists():
        raise SystemExit(f"Missing: {in_zarr}")
    return in_zarr


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive DL2 Richardson–Lucy deconvolution for 3D OME-Zarr.")
    ap.add_argument("--root_3d", default=str(DEFAULT_3D_ROOT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))

    ap.add_argument("--folder", default=None, help="Folder containing image.ome.zarr (skip selection)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--model", choices=["BW", "GL", "RW"], default=None)
    ap.add_argument("--background", type=float, default=None)

    args = ap.parse_args()

    root_3d = Path(args.root_3d)
    out_root = Path(args.out_root)

    if args.folder:
        in_zarr = Path(args.folder) / "image.ome.zarr"
        if not in_zarr.exists():
            raise SystemExit(f"Missing: {in_zarr}")
    else:
        in_zarr = interactive_select_dataset(root_3d)

    model = args.model or prompt_choice("Choose PSF model [BW/GL/RW] (default BW): ", ["BW", "GL", "RW"], default="BW")
    iters = args.iters if args.iters is not None else prompt_int("Iterations (default 15): ", min_v=1, max_v=500, default=15)
    background = args.background if args.background is not None else float(input("Background (default 0.0): ").strip() or "0.0")

    print("\nRun configuration:")
    print(f"  input zarr : {in_zarr}")
    print(f"  out root   : {out_root}")
    print(f"  PSF model  : {model}")
    print(f"  iters      : {iters}")
    print(f"  background : {background}")

    info = deconvolve_omezarr_3ch_to_omezarr(
        in_omezarr=in_zarr,
        out_root=out_root,
        model=model,
        iters=iters,
        background=background,
        channel_wavelength_nm=dict(DEFAULT_CHANNEL_WAVELENGTH_NM),
        overwrite=True,
    )

    print("\nDONE")
    print(f"  output zarr: {info.out_zarr}")
    print(f"  outdir     : {info.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())