"""Predict and save 2.5D U-Net foreground masks for a complete 3D volume.

The script reads the original OME-Zarr distribution used during training. It
saves only ``pred_mask.ome.zarr``, an optional probability OME-Zarr, and quality-
control figures for Z10, Z24, and Z30. It does not apply the
mask and does not save a background-attenuated fluorescence image.
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
    return tuple(sorted({int(item) for item in value.replace(",", " ").split()}))


def _parse_channels(value: str) -> tuple[int, ...] | None:
    return None if value.strip().lower() == "all" else tuple(int(item) for item in value.replace(",", " ").split())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mask-only 2.5D U-Net inference on an original 3D OME-Zarr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zarr", type=Path, required=True, help="Original image.ome.zarr used for U-Net inference")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "u_net_3d_25d" / "u_net_3d_25d_best.keras")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "U-net" / "3d_25d")
    parser.add_argument("--manual-mask-root", type=Path, default=PROJECT_ROOT / "results" / "training_files" / "U-net" / "3d_25d")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--channels", default="all")
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--qc-slices", default=",".join(map(str, DEFAULT_TRAINING_SLICES_1BASED)))
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
        channels=_parse_channels(args.channels),
        threshold=0.5,
        predict_batch_size=args.predict_batch_size,
        preview_slices_1based=_parse_slices(args.qc_slices),
        save_probability=not args.no_probability,
    )
    result = run_3d_25d_unet_masks(config)
    print("\nMask-only inference completed")
    print(f"Mask OME-Zarr:        {result.mask_zarr}")
    print(f"Probability OME-Zarr: {result.probability_zarr}")
    print(f"QC training slices:   {result.preview_dir}")
    print(f"Report:               {result.report_json}")
    print("No fluorescence intensities were modified in this step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
