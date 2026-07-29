"""
Dataset-specific wrapper for two-channel WGA-DAPI U-Net training.

To tune training, create :class:`UNet2DWgaDapiTrainConfig` and change inherited
fields such as ``patch``, ``batch``, ``epochs``, ``lr``, ``base_filters``,
``dropout``, or foreground-aware sampling parameters. Refer to the inherited
configuration docstring for the expected effect of each value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import UNet2DTrainConfig, train_2d_binary_unet


@dataclass
class UNet2DWgaDapiTrainConfig(UNet2DTrainConfig):
    """WGA-DAPI defaults with all parameter guidance inherited from the base config."""

    project_root: Path = find_project_root(Path(__file__).resolve())
    dataset: str = "2d_wga_dapi"


def train_2d_wga_dapi_unet(cfg: UNet2DWgaDapiTrainConfig | None = None) -> dict[str, Path]:
    """Train the two-channel model using the shared validated training pipeline."""
    cfg = cfg or UNet2DWgaDapiTrainConfig()
    cfg.dataset = "2d_wga_dapi"
    return train_2d_binary_unet(cfg)
