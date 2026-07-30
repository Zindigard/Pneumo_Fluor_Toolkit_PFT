"""Infer Z10 and Z12 masks, combine them, and broadcast to the full Z-stack.

The trained merged-RGB 2.5D U-Net is evaluated only at Z10 and Z12. Their
binary masks are combined using a pixelwise maximum, equivalent to a logical
union. The resulting two-dimensional mask is then copied to every Z-slice and
saved as ``pred_mask.ome.zarr``. Fluorescence intensities are not modified.
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

from PFT.core_prog_parts.denoising.validation_3d import DEFAULT_TRAINING_SLICES_1BASED  # noqa: E402
from PFT.core_prog_parts.segmentation.unet_run_3d_25d_core import (  # noqa: E402
    UNet25DRunConfig,
    run_3d_25d_unet_masks,
)


def _parse_slices(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.replace(",", " ").split()}))
    if values != DEFAULT_TRAINING_SLICES_1BASED:
        raise ValueError(
            f"This workflow is fixed to inference slices {DEFAULT_TRAINING_SLICES_1BASED}; received {values}."
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infer Z10/Z12 masks, combine by pixelwise maximum, and broadcast to all Z-slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zarr", type=Path, required=True, help="Original image.ome.zarr used for U-Net inference")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "u_net_3d_25d" / "u_net_3d_25d_best.keras")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "U-net" / "3d_25d")
    parser.add_argument("--manual-mask-root", type=Path, default=PROJECT_ROOT / "results" / "training_files" / "U-net" / "3d_25d")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--inference-slices", default=",".join(map(str, DEFAULT_TRAINING_SLICES_1BASED)))
    parser.add_argument("--no-probability", action="store_true")
    args = parser.parse_args()

    config = UNet25DRunConfig(
        project_root=PROJECT_ROOT,
        input_zarr=args.zarr,
        model_path=args.model,
        output_root=args.out_root,
        manual_mask_root=args.manual_mask_root,
        level=args.level,
        patch=args.patch,
        channels=None,
        threshold=0.5,
        predict_batch_size=args.predict_batch_size,
        inference_slices_1based=_parse_slices(args.inference_slices),
        preview_slices_1based=_parse_slices(args.inference_slices),
        save_probability=not args.no_probability,
    )
    result = run_3d_25d_unet_masks(config)
    print("\nMerged-mask inference completed")
    print(f"Full ZYX mask OME-Zarr: {result.mask_zarr}")
    print(f"Probability OME-Zarr:   {result.probability_zarr}")
    print(f"Combined 2D mask:       {result.combined_mask_tif}")
    print(f"Source-slice products:  {result.source_slice_dir}")
    print(f"QC figures:             {result.preview_dir}")
    print(f"Report:                 {result.report_json}")
    print("Z10 and Z12 were inferred, combined by pixelwise maximum, and copied to all Z-slices.")
    print("No fluorescence intensities were modified in this step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
