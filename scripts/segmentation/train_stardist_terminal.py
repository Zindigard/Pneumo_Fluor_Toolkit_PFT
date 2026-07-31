"""Compatibility terminal entry point for StarDist training.

The script delegates to :mod:`train_segmentation_model` and stores each trained
model below ``models/segmentation/stardist/<dataset>/<source_mode>/``.
"""

from __future__ import annotations

from train_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="stardist"))
