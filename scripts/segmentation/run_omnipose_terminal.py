r"""Compatibility terminal entry point for Omnipose prediction.

This script delegates to :mod:`run_segmentation_model` and fixes the model
family to ``omnipose``. It consumes only prepared float32 [0, 1] inputs and
prevents a second normalization of sparse, masked images.

Examples
--------
Show all options:

    python scripts/segmentation/run_omnipose_terminal.py --help

Run Omnipose for one prepared sample:

    python scripts/segmentation/run_omnipose_terminal.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --mode one \
        --sample WT_HADA_NHS_40min_ROI1_SIM \
        --model bact_fluor_omni \
        --diameter none \
        --flow-threshold 0.4 \
        --mask-threshold 0.0 \
        --min-size 15
"""

from __future__ import annotations

from run_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="omnipose"))
