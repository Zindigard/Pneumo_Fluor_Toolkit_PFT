"""
Run StarDist with the same channel policy used during fine-tuning.

Examples
--------
Show all options:

    python scripts/segmentation/run_stardist_terminal.py --help

Run StarDist for one WGA-DAPI sample:

    python scripts/segmentation/run_stardist_terminal.py \
        --dataset 2d_wga_dapi \
        --source-mode filtered_unet \
        --mode one \
        --sample WT_HADA_NHS_40min_ROI1_SIM \
        --model 2D_versatile_fluo \
        --prob-thresh 0.5 \
        --nms-thresh 0.4 \
        --min-size 15
"""

from __future__ import annotations

import sys

from stardist_input_adapter import install_stardist_input_adapter, policy_description

install_stardist_input_adapter()

from run_segmentation_model import main  # noqa: E402


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
