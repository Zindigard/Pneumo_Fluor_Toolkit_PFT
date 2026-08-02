"""
Provide command-line and programmatic utilities for deconvolve three-dimensional data.

Examples
--------
Show all command-line parameters:

    python scripts/denoising/deconvolve_3d.py --help

Representative execution:

    python scripts/denoising/deconvolve_3d.py \
        --model BW \
        --folder 20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM \
        --root_3d results/img/3d_data \
        --level 0
"""

from __future__ import annotations

# Configure imports for direct execution from the repository source tree.
import sys as _pft_sys
from pathlib import Path as _PFTPath

_PFT_SCRIPT_FILE = _PFTPath(__file__).resolve()


def _pft_project_root(start: _PFTPath | None = None) -> _PFTPath:
    """Return the repository root containing both ``scripts`` and ``src/PFT``.

    The lookup is based on this script's physical location and therefore does
    not depend on the current working directory. An explicit error is raised
    when the expected repository layout cannot be found.

    Args:
        start (_PFTPath | None): Filesystem path used for start. ``None`` selects the function's default behavior.

    Returns:
        _PFTPath: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _pft_project_root()
    """
    current = (start or _PFT_SCRIPT_FILE).resolve()
    search_start = current if current.is_dir() else current.parent

    for candidate in (search_start, *search_start.parents):
        core_dir = candidate / "src" / "PFT" / "core_prog_parts"
        if (candidate / "scripts").is_dir() and core_dir.is_dir():
            return candidate

    raise RuntimeError(
        "Cannot locate the PFT repository root. Expected both "
        "'scripts' and 'src/PFT/core_prog_parts' in the same project folder. "
        f"Script location: {_PFT_SCRIPT_FILE}"
    )


_PFT_PROJECT_ROOT = _pft_project_root()
_PFT_SRC_DIR = _PFT_PROJECT_ROOT / "src"

if str(_PFT_SRC_DIR) not in _pft_sys.path:
    _pft_sys.path.insert(0, str(_PFT_SRC_DIR))



from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()

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
    """List image folders available in the configured project structure.

    Args:
        root_3d (Path): Filesystem path used for root three-dimensional data.

    Returns:
        list[Path]: Resolved or generated filesystem path.

    Example:
        >>> result = list_image_folders(root_3d=Path("path/to/resource"))
    """
    return [p.parent for p in sorted(root_3d.rglob("image.ome.zarr"))]


def available_pyramid_levels(image_omezarr: Path) -> list[int]:
    """Return available pyramid levels for the supplied inputs.

    Args:
        image_omezarr (Path): Filesystem path used for image OME-Zarr.

    Returns:
        list[int]: Collection containing the generated or selected values.

    Example:
        >>> result = available_pyramid_levels(image_omezarr=Path("path/to/resource"))
    """
    root = zarr.open_group(str(image_omezarr), mode="r")
    ms = root.attrs.get("multiscales")
    if not ms or not isinstance(ms, list) or not ms[0].get("datasets"):
        return [0]
    ds = ms[0]["datasets"]
    return list(range(len(ds)))


def prompt_int(prompt: str, *, min_v: int | None = None, max_v: int | None = None, default: int | None = None) -> int:
    """Return prompt int for the supplied inputs.

    Args:
        prompt (str): Text value specifying prompt.
        min_v (int | None): Minimum permitted value of v. ``None`` selects the function's default behavior.
        max_v (int | None): Maximum permitted value of v. ``None`` selects the function's default behavior.
        default (int | None): Numerical value controlling default. ``None`` selects the function's default behavior.

    Returns:
        int: Computed numerical result.

    Example:
        >>> result = prompt_int(prompt="prompt")
    """
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
    """Return prompt choice for the supplied inputs.

    Args:
        prompt (str): Text value specifying prompt.
        options (list[str]): Text value specifying options.
        default (str | None): Text value specifying default. ``None`` selects the function's default behavior.

    Returns:
        str: Generated or resolved text value.

    Example:
        >>> result = prompt_choice(prompt="prompt", options="options")
    """
    opts_lower = [o.lower() for o in options]
    while True:
        s = input(prompt).strip().lower()
        if s == "" and default is not None:
            return default
        if s in opts_lower:
            return options[opts_lower.index(s)]
        print(f"Choose one of: {', '.join(options)}")


def interactive_select_dataset(root_3d: Path) -> Path:
    """Return interactive select dataset for the supplied inputs.

    Args:
        root_3d (Path): Filesystem path used for root three-dimensional data.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = interactive_select_dataset(root_3d=Path("path/to/resource"))
    """
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
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Raises:
        SystemExit: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> exit_code = main()
    """
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
