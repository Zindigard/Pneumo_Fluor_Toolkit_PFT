"""
Compatibility terminal entry point for Omnipose training or fine-tuning.

The script delegates to :mod:`train_segmentation_model`, keeps every dataset
and source mode independent, and stores all models below
``models/segmentation/omnipose``.
"""

from __future__ import annotations

from train_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="omnipose"))
