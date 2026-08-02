"""
Terminal entry point for StarDist fine-tuning with PFT channel adaptation.

Input policy
------------
* 2d_time: one HADA channel, unchanged.
* 2d_wga_dapi: DAPI and WGA are merged to one numerical channel with a
  pixelwise maximum. The prepared files on disk are not modified.

Examples
--------
Show all options:

    python scripts/segmentation/train_stardist_terminal.py --help

Fine-tune StarDist for 50 epochs:

    python scripts/segmentation/train_stardist_terminal.py \
        --dataset 2d_wga_dapi \
        --source-mode filtered_unet \
        --run-name stardist_wga_dapi_50ep \
        --pretrained-model 2D_versatile_fluo \
        --annotation-source all \
        --epochs 50 \
        --validation-policy combined \
        --validation-fraction 0.2 \
        --seed 1337 \
        --n-rays 32 \
        --grid 2 \
        --patch-size 256 256
"""

from __future__ import annotations

import sys

from stardist_input_adapter import install_stardist_input_adapter, policy_description

# Install before importing train_segmentation_model so that both its preview
# collection and the core training collection use the same StarDist input.
install_stardist_input_adapter()

from train_segmentation_model import main  # noqa: E402


def _dataset_from_argv() -> str | None:
    """Return dataset from argv for the supplied inputs.

    Returns:
        str | None: Generated or resolved text value.

    Example:
        >>> result = _dataset_from_argv()
    """
    try:
        index = sys.argv.index("--dataset")
        return sys.argv[index + 1]
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    dataset = _dataset_from_argv()
    print(f"StarDist input policy: {policy_description(dataset)}")
    raise SystemExit(main(default_family="stardist"))
