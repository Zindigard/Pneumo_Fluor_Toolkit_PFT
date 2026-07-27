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

_THIS_FILE = Path(__file__).resolve()
for _p in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (_p / "src" / "PFT").exists():
        _SRC_DIR = _p / "src"
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        break

from PFT.core_prog_parts.common_paths import find_project_root as find_repo_root
import argparse
from pathlib import Path

import zarr

from PFT.core_prog_parts.denoising.Deconvolution_omezarr import (
    deconvolve_omezarr_3ch_to_omezarr,
    DEFAULT_CHANNEL_WAVELENGTH_NM,
)
"Applies deconvolution to 3D image data"


DEFAULT_PROJECT_ROOT = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
DEFAULT_3D_ROOT = DEFAULT_PROJECT_ROOT / "results" / "img" / "3d_data"
DEFAULT_OUT_ROOT = DEFAULT_PROJECT_ROOT / "results" / "deconv"


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
    ap.add_argument("--level", type=int, default=None, help="OME-Zarr pyramid level (default: 2 if available else 0)")
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

    levels = available_pyramid_levels(in_zarr)
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

    model = args.model or prompt_choice("Choose PSF model [BW/GL/RW] (default BW): ", ["BW", "GL", "RW"], default="BW")
    iters = args.iters if args.iters is not None else prompt_int("Iterations (default 15): ", min_v=1, max_v=500, default=15)
    background = args.background if args.background is not None else float(input("Background (default 0.0): ").strip() or "0.0")

    print("\nRun configuration:")
    print(f"  input zarr : {in_zarr}")
    print(f"  out root   : {out_root}")
    print(f"  level      : {level}")
    print(f"  PSF model  : {model}")
    print(f"  iters      : {iters}")
    print(f"  background : {background}")

    info = deconvolve_omezarr_3ch_to_omezarr(
        in_omezarr=in_zarr,
        out_root=out_root,
        model=model,
        iters=iters,
        background=background,
        level=level,
        channel_wavelength_nm=dict(DEFAULT_CHANNEL_WAVELENGTH_NM),
        overwrite=True,
    )

    print("\nDONE")
    print(f"  output zarr: {info.out_zarr}")
    print(f"  outdir     : {info.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
