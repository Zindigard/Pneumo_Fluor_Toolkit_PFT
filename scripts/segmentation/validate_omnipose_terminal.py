"""Terminal entry point for validation of a saved Omnipose model."""

from __future__ import annotations

from validate_segmentation_model import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="omnipose"))
