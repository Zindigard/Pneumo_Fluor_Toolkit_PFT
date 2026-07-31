"""
Compatibility terminal entry point for Cellpose training or fine-tuning.

"""

from __future__ import annotations

from train_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="cellpose"))
