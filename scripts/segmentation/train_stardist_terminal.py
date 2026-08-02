"""Terminal entry point for StarDist fine-tuning with PFT channel adaptation.

Input policy
------------
* 2d_time: one HADA channel, unchanged.
* 2d_wga_dapi: DAPI and WGA are merged to one numerical channel with a
  pixelwise maximum. The prepared files on disk are not modified.
"""

from __future__ import annotations

import sys

from stardist_input_adapter import install_stardist_input_adapter, policy_description

# Install before importing train_segmentation_model so that both its preview
# collection and the core training collection use the same StarDist input.
install_stardist_input_adapter()

from train_segmentation_model import main  # noqa: E402


def _dataset_from_argv() -> str | None:
    try:
        index = sys.argv.index("--dataset")
        return sys.argv[index + 1]
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    dataset = _dataset_from_argv()
    print(f"StarDist input policy: {policy_description(dataset)}")
    raise SystemExit(main(default_family="stardist"))
