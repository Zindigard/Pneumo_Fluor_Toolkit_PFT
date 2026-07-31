"""Run pretrained Cellpose-SAM parameter screening against binary U-Net masks."""

from __future__ import annotations

from screen_pretrained_segmentation import main


if __name__ == "__main__":
    raise SystemExit(main(default_family="cellpose"))
