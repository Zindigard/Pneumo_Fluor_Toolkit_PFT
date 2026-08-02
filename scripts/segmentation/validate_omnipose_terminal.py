r"""Terminal entry point for validation of a saved Omnipose model.

Examples
--------
Show all options:

    python scripts/segmentation/validate_omnipose_terminal.py --help

Validate a saved Omnipose model:

    python scripts/segmentation/validate_omnipose_terminal.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --model models/omnipose/omnipose_2d_time_50ep \
        --annotation-source all \
        --validation-policy combined \
        --validation-fraction 0.2 \
        --seed 1337 \
        --validation-diameter 0 \
        --validation-flow-threshold 0.4 \
        --validation-mask-threshold 0.0 \
        --validation-min-size 15
"""

from __future__ import annotations

from validate_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="omnipose"))
