r"""Compatibility terminal entry point for Cellpose training or fine-tuning.

The script delegates to :mod:`train_segmentation_model`, stores models below
``models/segmentation/cellpose/<dataset>/<source_mode>/``, and requires manual
instance masks below ``results/training_files/segmentation``.

Examples
--------
Show all options:

    python scripts/segmentation/train_cellpose_terminal.py --help

Fine-tune Cellpose-SAM for 50 epochs:

    python scripts/segmentation/train_cellpose_terminal.py \
        --dataset 2d_time \
        --source-mode filtered_unet \
        --run-name cellpose_2d_time_50ep \
        --pretrained-model cpsam \
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
    raise SystemExit(main(default_family="cellpose"))
