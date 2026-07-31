"""Compatibility terminal entry point for Cellpose prediction.

This script delegates to :mod:`run_segmentation_model` and fixes the model
family to ``cellpose``. Inputs must first be prepared with
``prepare_segmentation_inputs.py``. Cellpose normalization is disabled because
the prepared OME-Zarr input is already normalized to float32 [0, 1].
"""

from __future__ import annotations

from run_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="cellpose"))
