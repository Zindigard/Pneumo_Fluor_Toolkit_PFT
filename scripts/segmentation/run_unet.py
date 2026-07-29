"""Compatibility entry point for the 2D U-Net inference CLI."""

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
