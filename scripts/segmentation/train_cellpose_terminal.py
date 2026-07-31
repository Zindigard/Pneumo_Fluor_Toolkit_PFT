"""Compatibility terminal entry point for Cellpose training or fine-tuning.

The script delegates to :mod:`train_segmentation_model`, stores models below
``models/segmentation/cellpose/<dataset>/<source_mode>/``, and requires manual
instance masks below ``results/training_files/segmentation``.
"""

from __future__ import annotations

from train_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="cellpose"))
