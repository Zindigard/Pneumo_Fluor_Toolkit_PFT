r"""Run pretrained Omnipose parameter screening against binary U-Net masks.

Examples
--------
Show all options:

    python scripts/segmentation/screen_omnipose_terminal.py --help

Screen two randomly selected complete-image annotations:

    python scripts/segmentation/screen_omnipose_terminal.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --model-name bact_fluor_omni \
        --annotation-source full-images-only \
        --annotation-split any \
        --sample-count 2 \
        --seed 1337 \
        --diameters none \
        --flow-thresholds 0.4 \
        --mask-thresholds 0.0 \
        --min-sizes 15 \
        --run-name bact_fluor_omni_default_random_2 \
        --save-labels
"""

from __future__ import annotations

from screen_pretrained_segmentation import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="omnipose"))
