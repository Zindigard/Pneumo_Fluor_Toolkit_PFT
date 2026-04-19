from __future__ import annotations

from pathlib import Path
import sys
import argparse
import zarr
from PFT.core_prog_parts.common_paths import find_project_root, ensure_dir
from PFT.core_prog_parts.deconvolution_no_fuji import (
    deconvolve_omezarr_3ch_to_omezarr_skimage,
    DEFAULT_CHANNEL_WAVELENGTH_NM,
)

"""
Interactive runner for scikit-image Richardson-Lucy deconvolution on 3D OME-Zarr.
"""


def list_image_folders(root_3d: Path) -> list[Path]:
    """List folders that contain image.ome.zarr."""
    return [p.parent for p in sorted(root_3d.rglob("image.ome.zarr"))]


def available_pyramid_levels(image_omezarr: Path) -> list[int]:
    """Read available OME-Zarr pyramid levels."""
    root = zarr.open_group(str(image_omezarr), mode="r")
    ms = root.attrs.get("multiscales")
    if not ms or not isinstance(ms, list) or not ms[0].get("datasets"):
        return [0]
    ds = ms[0]["datasets"]
    return list(range(len(ds)))


def prompt_int(
    prompt: str,
    *,
    min_v: int | None = None,
    max_v: int | None = None,
    default: int | None = None,
) -> int:
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


def prompt_float(prompt: str, *, default: float | None = None) -> float:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("Please enter a valid number.")


def prompt_choice(prompt: str, options: list[str], default: str | None = None) -> str:
    opts_lower = [o.lower() for o in options]
    while True:
        s = input(prompt).strip().lower()
        if s == "" and default is not None:
            return default
        if s in opts_lower:
            return options[opts_lower.index(s)]
        print(f"Choose one of: {', '.join(options)}")


def prompt_yesno(prompt: str, *, default: bool = False) -> bool:
    default_s = "y" if default else "n"
    while True:
        s = input(f"{prompt} [y/n] (default {default_s}): ").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please answer y or n.")


def interactive_select_dataset(root_3d: Path) -> Path:
    folders = list_image_folders(root_3d)
    if not folders:
        raise SystemExit(f"No image.ome.zarr found under: {root_3d}")

    print("\nAvailable 3D datasets:")
    for i, f in enumerate(folders):
        rel = f.relative_to(root_3d) if root_3d in f.parents else f
        print(f"  [{i}] {rel}")

    idx = prompt_int(
        f"\nSelect dataset index [0-{len(folders)-1}]: ",
        min_v=0,
        max_v=len(folders) - 1,
    )
    in_zarr = folders[idx] / "image.ome.zarr"
    if not in_zarr.exists():
        raise SystemExit(f"Missing: {in_zarr}")
    return in_zarr


def main() -> int:
    project_root = find_project_root(Path(__file__).resolve())
    default_3d_root = project_root / "results" / "img" / "3d_data"
    default_out_root = ensure_dir(project_root / "results" / "deconv")

    ap = argparse.ArgumentParser(
        description="Interactive scikit-image Richardson-Lucy deconvolution for 3D OME-Zarr."
    )
    ap.add_argument("--root_3d", default=str(default_3d_root))
    ap.add_argument("--out_root", default=str(default_out_root))

    ap.add_argument("--folder", default=None, help="Folder containing image.ome.zarr (skip selection)")
    ap.add_argument("--level", type=int, default=None, help="OME-Zarr pyramid level")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--model", choices=["BW", "GL", "RW"], default=None)
    ap.add_argument("--background", type=float, default=None)
    ap.add_argument("--clip", action="store_true", help="Pass clip=True to skimage RL")
    ap.add_argument("--filter_epsilon", type=float, default=None)

    args = ap.parse_args()

    root_3d = Path(args.root_3d)
    out_root = ensure_dir(Path(args.out_root))

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

    model = args.model or prompt_choice(
        "Choose PSF model [BW/GL/RW] (default BW): ",
        ["BW", "GL", "RW"],
        default="BW",
    )

    iters = args.iters if args.iters is not None else prompt_int(
        "Iterations (default 15): ",
        min_v=1,
        max_v=500,
        default=15,
    )

    background = args.background if args.background is not None else prompt_float(
        "Background (default 0.0): ",
        default=0.0,
    )

    clip = bool(args.clip)
    if args.clip is False:
        clip = prompt_yesno("Use clip=True for RL?", default=False)

    filter_epsilon = args.filter_epsilon
    if args.filter_epsilon is None:
        use_eps = prompt_yesno("Set filter_epsilon?", default=False)
        if use_eps:
            filter_epsilon = prompt_float("filter_epsilon value (example 1e-6): ", default=1e-6)

    print("\nRun configuration:")
    print(f"  project root     : {project_root}")
    print(f"  input zarr       : {in_zarr}")
    print(f"  out root         : {out_root}")
    print(f"  level            : {level}")
    print(f"  PSF model        : {model}")
    print(f"  iterations       : {iters}")
    print(f"  background       : {background}")
    print(f"  clip             : {clip}")
    print(f"  filter_epsilon   : {filter_epsilon}")

    info = deconvolve_omezarr_3ch_to_omezarr_skimage(
        in_omezarr=in_zarr,
        out_root=out_root,
        model=model,
        iters=iters,
        background=background,
        level=level,
        channel_wavelength_nm=dict(DEFAULT_CHANNEL_WAVELENGTH_NM),
        overwrite=True,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    print("\nDONE")
    print(f"  output zarr: {info.out_zarr}")
    print(f"  outdir     : {info.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())