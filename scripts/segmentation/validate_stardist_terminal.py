"""Validate a StarDist model using the training-consistent input policy."""

from __future__ import annotations

import sys

from stardist_input_adapter import install_stardist_input_adapter, policy_description

install_stardist_input_adapter()

from validate_segmentation_model import main  # noqa: E402


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
