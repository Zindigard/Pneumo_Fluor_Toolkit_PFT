"""Screen pretrained StarDist directly against manual integer instance masks."""

from __future__ import annotations

from screen_pretrained_segmentation import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="stardist"))
