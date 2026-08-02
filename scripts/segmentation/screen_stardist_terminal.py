r"""Screen StarDist predictions using the training-consistent input policy.

Examples
--------
Show all options:

    python scripts/segmentation/screen_stardist_terminal.py --help

Screen the pretrained StarDist model on validation annotations:

    python scripts/segmentation/screen_stardist_terminal.py \
        --dataset 2d_wga_dapi \
        --source-mode filtered_unet \
        --annotation-source crops-only \
        --annotation-split validation \
        --prob-thresholds 0.3 0.5 0.7 \
        --nms-thresholds 0.3 0.4 0.5 \
        --min-sizes 15 \
        --run-name stardist_validation_screen \
        --save-labels
"""

from __future__ import annotations

import sys

from stardist_input_adapter import install_stardist_input_adapter, policy_description

install_stardist_input_adapter()

from screen_pretrained_segmentation import main  # noqa: E402


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
