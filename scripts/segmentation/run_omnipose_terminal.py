"""Compatibility terminal entry point for Omnipose prediction.

This script delegates to :mod:`run_segmentation_model` and fixes the model
family to ``omnipose``. It consumes only prepared float32 [0, 1] inputs and
prevents a second normalization of sparse, masked images.
"""

from __future__ import annotations

from run_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="omnipose"))
