"""Run validated 3D Richardson-Lucy deconvolution without Fiji.

This script reads level 0 of one CZYX OME-Zarr, validates the stack against the
single reusable three-wavelength master PSF set, performs channel-wise 3D
Richardson-Lucy deconvolution, and writes a new multiscale OME-Zarr. Channel-to-
PSF assignment uses wavelength metadata. Stored output values are raw float32
deconvolution values;
no 0-1 normalization is applied. ``clip`` is permanently disabled because
scikit-image clipping would destroy raw fluorescence intensity ranges.

Quality-control PNGs are created only for the planned 2.5D training slices:
Z10, Z24, and Z30. Each figure contains original, deconvolved, signed-
difference, and absolute-difference views. Richardson-Lucy iterations can be
set independently for the 405 nm blue, 488 nm green, and 561 nm red channels.
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
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.denoising.deconvolution_no_fuji import (  # noqa: E402
    DEFAULT_TRAINING_SLICES_1BASED,
    deconvolve_omezarr_3ch_to_omezarr_skimage,
)


def _parse_slices(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.replace(",", " ").split()}))
    if not result:
        raise ValueError("At least one QC slice is required")
    return result


def _find_zarrs(root: Path) -> list[Path]:
    if root.name.endswith(".ome.zarr") and root.is_dir():
        return [root]
    return sorted(path for path in root.rglob("image.ome.zarr") if path.is_dir())


def _select_zarr(root: Path) -> Path:
    items = _find_zarrs(root)
    if not items:
        raise FileNotFoundError(f"No image.ome.zarr found under {root}")
    if len(items) == 1:
        return items[0]
    print("\nAvailable 3D datasets")
    for index, item in enumerate(items):
        print(f"  [{index}] {item}")
    selected = int(input(f"Select dataset [0-{len(items)-1}]: ").strip())
    return items[selected]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run raw-intensity-preserving 3D Richardson-Lucy deconvolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zarr", type=Path, default=None, help="Input image.ome.zarr")
    parser.add_argument("--root-3d", type=Path, default=PROJECT_ROOT / "results" / "img" / "3d_data")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "deconv")
    parser.add_argument(
        "--level", type=int, default=0, choices=(0,),
        help="Fixed input pyramid level. The reusable master PSFs support only level 0",
    )
    parser.add_argument("--model", choices=("BW", "GL", "RW"), default="BW")
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Default Richardson-Lucy iteration count used for channels without an explicit override",
    )
    parser.add_argument("--iters-blue", type=int, default=None, help="Iteration count for the 405 nm blue channel")
    parser.add_argument("--iters-green", type=int, default=None, help="Iteration count for the 488 nm green channel")
    parser.add_argument("--iters-red", type=int, default=None, help="Iteration count for the 561 nm red channel")
    parser.add_argument("--background", type=float, default=0.0, help="Constant background subtracted before RL")
    parser.add_argument("--filter-epsilon", type=float, default=None)
    parser.add_argument("--pyramid-max-layer", type=int, default=2)
    parser.add_argument(
        "--qc-slices",
        default=",".join(str(value) for value in DEFAULT_TRAINING_SLICES_1BASED),
        help="One-based slices included in before/after/difference figures",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    input_zarr = args.zarr.expanduser().resolve() if args.zarr else _select_zarr(args.root_3d)
    slices = _parse_slices(args.qc_slices)
    channel_iterations = {
        color: value
        for color, value in (
            ("blue", args.iters_blue),
            ("green", args.iters_green),
            ("red", args.iters_red),
        )
        if value is not None
    }
    if any(value < 1 for value in channel_iterations.values()):
        raise ValueError("All per-channel iteration counts must be at least 1")
    print("\nPFT 3D Richardson-Lucy deconvolution")
    print("=" * 72)
    print(f"Input:              {input_zarr}")
    print(f"Input level:        {args.level}")
    print(f"PSF model:          {args.model}")
    print("PSF source:         results/psf/master (three reusable wavelength PSFs)")
    print("Channel mapping:    wavelength metadata; incompatible stacks stop with an error")
    print(f"Default iterations: {args.iters}")
    print(
        "Per-channel iters: "
        + (
            ", ".join(f"{color}={value}" for color, value in channel_iterations.items())
            if channel_iterations
            else "none; default used for all channels"
        )
    )
    print(f"Background:         {args.background}")
    print("Stored normalization: NONE")
    print("Stored dtype:       float32")
    print(f"Output pyramid:     levels 0-{args.pyramid_max_layer}")
    print(f"QC slices:          {slices}")

    result = deconvolve_omezarr_3ch_to_omezarr_skimage(
        in_omezarr=input_zarr,
        out_root=args.out_root,
        model=args.model,
        iters=args.iters,
        channel_iterations=channel_iterations,
        background=args.background,
        level=args.level,
        overwrite=not args.no_overwrite,
        clip=False,
        filter_epsilon=args.filter_epsilon,
        pyramid_max_layer=args.pyramid_max_layer,
        preview_slices_1based=slices,
    )
    print("\nCompleted")
    print(f"Preflight:      {result.preflight_report}")
    print(f"OME-Zarr:       {result.out_zarr}")
    print(f"Validation TXT: {result.report_txt}")
    print(f"Validation JSON:{result.report_json}")
    print(f"Statistics CSV: {result.stats_csv}")
    print(f"QC figures:     {result.preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
