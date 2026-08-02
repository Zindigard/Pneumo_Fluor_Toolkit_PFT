r"""Validate a StarDist model using the training-consistent input policy.

Examples
--------
Show all options:

    python scripts/segmentation/validate_stardist_terminal.py --help

Validate a saved StarDist model:

    python scripts/segmentation/validate_stardist_terminal.py \
        --dataset 2d_wga_dapi \
        --source-mode filtered_unet \
        --model models/stardist/stardist_wga_dapi_50ep \
        --annotation-source all \
        --validation-policy combined \
        --validation-fraction 0.2 \
        --seed 1337 \
        --validation-prob-thresh 0.5 \
        --validation-nms-thresh 0.4 \
        --validation-min-size 15
"""

from __future__ import annotations

import sys

from stardist_input_adapter import install_stardist_input_adapter, policy_description

install_stardist_input_adapter()

from validate_segmentation_model import main  # noqa: E402


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
