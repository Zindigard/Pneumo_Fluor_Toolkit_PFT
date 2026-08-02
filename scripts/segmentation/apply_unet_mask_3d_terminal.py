"""
Apply a saved 2.5D U-Net mask to a saved deconvolved OME-Zarr.

The configured per-volume target slice is used automatically for manual-mask
SNR evaluation. Foreground values are preserved and 2% of intensity remains
outside the predicted mask by default.

Examples
--------
Show all options:

    python scripts/segmentation/apply_unet_mask_3d_terminal.py --help

Apply complete outside-mask depletion to one deconvolved stack:

    python scripts/segmentation/apply_unet_mask_3d_terminal.py \
        --deconvolved-zarr results/deconv/20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM/image.ome.zarr \
        --mask-zarr results/U-net/3d_25d/20220218_dynamic/DpspA_THY_HADA_NADA_TADA_40min_ROI1_SIM/mask.ome.zarr \
        --outside-mask-depletion 1.0 \
        --pyramid-max-layer 2
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

from PFT.core_prog_parts.segmentation.mask_application_3d import (  # noqa: E402
    apply_saved_mask_to_deconvolution,
)


def main() -> int:
    """Execute the command-line workflow and return its process exit status.

    Returns:
        int: Computed numerical result.

    Example:
        >>> exit_code = main()
    """
    parser = argparse.ArgumentParser(
        description="Apply a saved 2.5D U-Net mask after 3D deconvolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--deconvolved-zarr", type=Path, required=True)
    parser.add_argument("--mask-zarr", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results" / "denoised_images" / "3d")
    parser.add_argument("--manual-mask-root", type=Path, default=PROJECT_ROOT / "results" / "training_files" / "U-net" / "3d_25d")
    parser.add_argument("--outside-mask-depletion", type=float, default=0.98)
    parser.add_argument("--pyramid-max-layer", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    result = apply_saved_mask_to_deconvolution(
        deconvolved_zarr=args.deconvolved_zarr,
        predicted_mask_zarr=args.mask_zarr,
        output_root=args.out_root,
        manual_mask_root=args.manual_mask_root,
        outside_mask_depletion=args.outside_mask_depletion,
        pyramid_max_layer=args.pyramid_max_layer,
        slices_1based=None,
        epsilon=args.epsilon,
        overwrite=not args.no_overwrite,
    )
    print("\nMask application completed")
    print(f"Final OME-Zarr:   {result.output_zarr}")
    print(f"Validation:       {result.validation_report}")
    print(f"SNR CSV:          {result.snr_csv}")
    print(f"SNR JSON:         {result.snr_json}")
    print(f"QC figures:       {result.preview_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
