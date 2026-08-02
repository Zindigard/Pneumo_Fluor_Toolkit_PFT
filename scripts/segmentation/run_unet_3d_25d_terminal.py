r"""Infer the configured per-volume target slice and broadcast its mask in Z.

Examples
--------
Show all options:

    python scripts/segmentation/run_unet_3d_25d_terminal.py --help

Run 2.5D U-Net inference for one 3D stack:

    python scripts/segmentation/run_unet_3d_25d_terminal.py \
        --zarr results/img/3d_data/20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM/image.ome.zarr \
        --model models/u_net_3d_25d/u_net_3d_25d_best.keras \
        --level 0 \
        --patch 256 \
        --predict-batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_FILE = Path(__file__).resolve()


def _project_root() -> Path:
    """Return project root for the supplied inputs.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _project_root()
    """
    for candidate in (SCRIPT_FILE.parent, *SCRIPT_FILE.parents):
        if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
            return candidate
    raise RuntimeError("PFT project root was not found")


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PFT.core_prog_parts.denoising.validation_3d import target_slice_for_volume  # noqa: E402
from PFT.core_prog_parts.segmentation.unet_run_3d_25d_core import (  # noqa: E402
    UNet25DRunConfig,
    run_3d_25d_unet_masks,
)


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Infer the configured target-slice mask and broadcast it to all Z-slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--zarr", type=Path, required=True, help="Source image.ome.zarr")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "u_net_3d_25d" / "u_net_3d_25d_best.keras")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "U-net" / "3d_25d")
    parser.add_argument("--manual-mask-root", type=Path, default=PROJECT_ROOT / "results" / "training_files" / "U-net" / "3d_25d")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--no-probability", action="store_true")
    args = parser.parse_args()

    input_zarr = args.zarr.expanduser().resolve()
    target = target_slice_for_volume(input_zarr)
    config = UNet25DRunConfig(
        project_root=PROJECT_ROOT,
        input_zarr=input_zarr,
        model_path=args.model,
        output_root=args.out_root,
        manual_mask_root=args.manual_mask_root,
        level=args.level,
        patch=args.patch,
        channels=None,
        threshold=0.5,
        predict_batch_size=args.predict_batch_size,
        save_probability=not args.no_probability,
    )
    result = run_3d_25d_unet_masks(config)
    print("\nConfigured target-slice mask inference completed")
    print(f"Target/context:          Z{target} from Z{target-1}/Z{target}/Z{target+1}")
    print(f"Full ZYX mask OME-Zarr: {result.mask_zarr}")
    print(f"Probability OME-Zarr:   {result.probability_zarr}")
    print(f"Target 2D mask:         {result.source_mask_tif}")
    print(f"Target-slice products:  {result.source_slice_dir}")
    print(f"QC figures:             {result.preview_dir}")
    print(f"Report:                 {result.report_json}")
    if result.reference_metrics is None:
        print(f"Reference metrics:      SKIPPED ({result.reference_metrics_status})")
    else:
        print(
            "Reference metrics:      "
            f"IoU={result.reference_metrics['iou']:.4f}; "
            f"Dice={result.reference_metrics['dice']:.4f}; "
            f"precision={result.reference_metrics['precision']:.4f}; "
            f"recall={result.reference_metrics['recall']:.4f}"
        )
    print("No fluorescence intensities were modified in this step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
