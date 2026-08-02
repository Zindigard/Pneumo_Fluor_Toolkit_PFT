"""Provide command-line and programmatic utilities for U-Net result train two-dimensional data wga dapi core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import UNet2DTrainConfig, train_2d_binary_unet


@dataclass
class UNet2DWgaDapiTrainConfig(UNet2DTrainConfig):
    """Store validated configuration or result data for unet2 dwga dapi train config."""
    project_root: Path = find_project_root(Path(__file__).resolve())
    dataset: str = "2d_wga_dapi"


def train_2d_wga_dapi_unet(cfg: UNet2DWgaDapiTrainConfig | None = None) -> dict[str, Path]:
    """Train two-dimensional data wga dapi U-Net result using the supplied data and configuration.

    Args:
        cfg (UNet2DWgaDapiTrainConfig | None): Value specifying cfg for the operation. ``None`` selects the function's default behavior.

    Returns:
        dict[str, Path]: Resolved or generated filesystem path.

    Example:
        >>> result = train_2d_wga_dapi_unet()
    """
    cfg = cfg or UNet2DWgaDapiTrainConfig()
    cfg.dataset = "2d_wga_dapi"
    return train_2d_binary_unet(cfg)
