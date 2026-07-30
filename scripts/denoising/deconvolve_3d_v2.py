"""Run validated 3D Richardson-Lucy deconvolution without Fiji.

With no ``--zarr`` argument, the script processes the fixed four-stack test
cohort, containing one source OME-Zarr from each acquisition directory. Each
stack uses its configured target slice for QC. The QC output contains raw
before/after and independently normalized before/after views for the merged RGB
image and all three wavelength-mapped channels; no difference image is saved.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
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
    deconvolve_omezarr_3ch_to_omezarr_skimage,
)
from PFT.core_prog_parts.denoising.validation_3d import (  # noqa: E402
    configured_test_zarrs,
    target_slice_for_volume,
    volume_key,
)


def _write_batch_summary(rows: list[dict[str, str]], out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "deconvolution_four_stack_test_summary.csv"
    fieldnames = [
        "generated_utc",
        "status",
        "sample",
        "target_slice_1based",
        "input_omezarr",
        "output_omezarr",
        "qc_directory",
        "validation_report",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw-intensity-preserving 3D Richardson-Lucy deconvolution. "
            "Without --zarr, one configured stack from each of four acquisition folders is processed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=None,
        help="Process one exact source image.ome.zarr instead of the fixed four-stack cohort",
    )
    parser.add_argument("--root-3d", type=Path, default=PROJECT_ROOT / "results" / "img" / "3d_data")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "deconv")
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=(0,),
        help="Fixed input pyramid level; reusable master PSFs support only level 0",
    )
    parser.add_argument("--model", choices=("BW", "GL", "RW"), default="BW")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--iters-blue", type=int, default=None)
    parser.add_argument("--iters-green", type=int, default=None)
    parser.add_argument("--iters-red", type=int, default=None)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--filter-epsilon", type=float, default=None)
    parser.add_argument("--pyramid-max-layer", type=int, default=2)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one stack fails; otherwise continue and record the failure",
    )
    args = parser.parse_args()

    root_3d = args.root_3d.expanduser().resolve()
    if args.zarr is not None:
        inputs = [args.zarr.expanduser().resolve()]
        if not inputs[0].is_dir() or inputs[0].name != "image.ome.zarr":
            raise FileNotFoundError(f"--zarr must point to image.ome.zarr: {inputs[0]}")
    else:
        inputs = configured_test_zarrs(root_3d)

    channel_iterations = {
        color: value
        for color, value in (
            ("blue", args.iters_blue),
            ("green", args.iters_green),
            ("red", args.iters_red),
        )
        if value is not None
    }
    if args.iters < 1 or any(value < 1 for value in channel_iterations.values()):
        raise ValueError("All Richardson-Lucy iteration counts must be at least 1")

    print("\nPFT 3D Richardson-Lucy deconvolution")
    print("=" * 72)
    print(f"Input mode:          {'single stack' if args.zarr else 'fixed four-stack cohort'}")
    print(f"Stacks:              {len(inputs)}")
    print(f"Input level:         {args.level}")
    print(f"PSF model:           {args.model}")
    print("PSF source:          results/psf/master")
    print("Stored normalization: NONE")
    print("Stored dtype:        float32")
    print("QC:                  raw before/after + normalized before/after; no differences")
    for index, path in enumerate(inputs, start=1):
        target = target_slice_for_volume(path, image_root=root_3d)
        print(f"  [{index}] {volume_key(path, root_3d)} | target Z{target}")

    rows: list[dict[str, str]] = []
    failures = 0
    for index, input_zarr in enumerate(inputs, start=1):
        sample = volume_key(input_zarr, root_3d)
        target = target_slice_for_volume(input_zarr, image_root=root_3d)
        print(f"\n[{index}/{len(inputs)}] {sample}")
        print(f"Target QC slice: Z{target}; context Z{target-1}/Z{target}/Z{target+1}")
        try:
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
                preview_slices_1based=(target,),
            )
            rows.append({
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "status": "PASS",
                "sample": sample,
                "target_slice_1based": str(target),
                "input_omezarr": str(input_zarr),
                "output_omezarr": str(result.out_zarr),
                "qc_directory": str(result.preview_dir),
                "validation_report": str(result.report_txt),
                "error": "",
            })
            print(f"PASS: {result.out_zarr}")
            print(f"QC:   {result.preview_dir}")
        except Exception as error:
            failures += 1
            rows.append({
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "status": "FAIL",
                "sample": sample,
                "target_slice_1based": str(target),
                "input_omezarr": str(input_zarr),
                "output_omezarr": "",
                "qc_directory": "",
                "validation_report": "",
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"FAIL: {type(error).__name__}: {error}")
            if args.stop_on_error:
                summary = _write_batch_summary(rows, args.out_root)
                print(f"Batch summary: {summary}")
                raise

    summary = _write_batch_summary(rows, args.out_root)
    print("\nDeconvolution batch completed")
    print(f"Passed: {len(inputs) - failures}/{len(inputs)}")
    print(f"Batch summary: {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
