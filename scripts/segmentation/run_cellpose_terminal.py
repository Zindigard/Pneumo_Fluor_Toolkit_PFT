"""
Compatibility terminal entry point for Cellpose prediction.

This script delegates to :mod:`run_segmentation_model` and fixes the model
family to ``cellpose``. Inputs must first be prepared with
``prepare_segmentation_inputs.py``. Cellpose normalization is disabled because
the prepared OME-Zarr input is already normalized to float32 [0, 1].

Examples
--------
Show all options:

    python scripts/segmentation/run_cellpose_terminal.py --help

Run Cellpose-SAM for one prepared sample:

    python scripts/segmentation/run_cellpose_terminal.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --mode one \
        --sample WT_HADA_NHS_40min_ROI1_SIM \
        --model cpsam \
        --diameter none \
        --flow-threshold 0.4 \
        --cellprob-threshold 0.0 \
        --min-size 15
"""

from __future__ import annotations

from run_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="cellpose"))
