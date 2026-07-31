"""
Compatibility terminal entry point for StarDist prediction.

This script delegates to :mod:`run_segmentation_model` and fixes the model
family to ``stardist``. 
"""

from __future__ import annotations

from run_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="stardist"))
