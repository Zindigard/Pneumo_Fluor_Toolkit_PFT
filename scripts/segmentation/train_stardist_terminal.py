"""
Compatibility terminal entry point for StarDist training.

"""

from __future__ import annotations

from train_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="stardist"))
