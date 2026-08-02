r"""Compatibility entry point for the 2D U-Net inference CLI.

Examples
--------
Show all options:

    python scripts/segmentation/run_unet.py --help

Run the 2D U-Net for one sample:

    python scripts/segmentation/run_unet.py \
        --dataset 2d_time \
        --mode one \
        --input results/img/2d_time/WT_HADA_NHS_40min_ROI1_SIM/image.ome.zarr \
        --level 0 \
        --patch 256 \
        --threshold 0.5 \
        --outside-mask-depletion 0.98 \
        --non-interactive
"""

from __future__ import annotations

from pathlib import Path
import sys

_SCRIPT = Path(__file__).resolve()
for candidate in (_SCRIPT.parent, *_SCRIPT.parents):
    if (candidate / "scripts").is_dir() and (candidate / "src" / "PFT").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from scripts.segmentation.run_unet_terminal import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
