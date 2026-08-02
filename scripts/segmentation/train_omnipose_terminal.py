r"""Compatibility terminal entry point for Omnipose training or fine-tuning.

The script delegates to :mod:`train_segmentation_model`, keeps every dataset
and source mode independent, and stores all models below
``models/segmentation/omnipose``.

Examples
--------
Show all options:

    python scripts/segmentation/train_omnipose_terminal.py --help

Fine-tune Omnipose for 50 epochs:

    python scripts/segmentation/train_omnipose_terminal.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --run-name omnipose_2d_time_50ep \
        --pretrained-model bact_fluor_omni \
        --annotation-source all \
        --epochs 50 \
        --batch-size 8 \
        --validation-policy combined \
        --validation-fraction 0.2 \
        --seed 1337
"""

from __future__ import annotations

from train_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="omnipose"))
