"""Provide command-line and programmatic utilities for omnipose finetune two-dimensional data time core."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation.segmentation_model_io_core import (
    collect_training_pairs, extract_2d_image, extract_3d_volume, load_2d_mask, load_3d_mask,
    model_root, write_json,
)

DATASET = "2d_time"
FAMILY = "omnipose"


@dataclass
class OmniposeFineTuneConfig:
    """Store validated configuration or result data for omnipose fine tune config."""
    project_root: Path
    pretrained_model: str = "cyto2_omni"
    learning_rate: float = 0.1
    batch_size: int = 8
    n_epochs: int = 400
    diameter: float = 0.0
    use_gpu: bool = True
    channels: tuple[int, ...] | None = None

    def __post_init__(self):
        """Return post init for the supplied inputs.

        Example:
            >>> instance = OmniposeFineTuneConfig(...)
            >>> instance.__post_init__()
        """
        if self.channels is None:
            self.channels = (0, 1) if DATASET == "2d_wga_dapi" else (0,)

    def model_dir(self) -> Path:
        """Return model dir for the supplied inputs.

        Returns:
            Path: Resolved or generated filesystem path.

        Example:
            >>> instance = OmniposeFineTuneConfig(...)
            >>> result = instance.model_dir()
        """
        return model_root(self.project_root, FAMILY, DATASET)


def _prepare_folder(cfg: OmniposeFineTuneConfig) -> Path:
    """Prepare folder for downstream processing.

    Args:
        cfg (OmniposeFineTuneConfig): Value specifying cfg for the operation.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = _prepare_folder(cfg=config)
    """
    out = cfg.model_dir() / "prepared_training_data"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = collect_training_pairs(cfg.project_root, DATASET)
    written = 0
    for image_path, mask_path, sample in pairs:
        if DATASET == "3d":
            vol = extract_3d_volume(image_path, channels=cfg.channels)
            mask = load_3d_mask(mask_path)
            z_count = min(vol.shape[0], mask.shape[0])
            for z in range(z_count):
                tiff.imwrite(out / f"{sample}_z{z:03d}_img.tif", vol[z].astype(np.float32))
                tiff.imwrite(out / f"{sample}_z{z:03d}_masks.tif", mask[z].astype(np.int32))
                written += 1
        else:
            img = extract_2d_image(image_path, DATASET, channels=cfg.channels)
            mask = load_2d_mask(mask_path)
            tiff.imwrite(out / f"{sample}_img.tif", img.astype(np.float32))
            tiff.imwrite(out / f"{sample}_masks.tif", mask.astype(np.int32))
            written += 1
    if written == 0:
        raise RuntimeError(f"No Omnipose training files prepared for {DATASET}")
    return out


def finetune_omnipose_2d_time(cfg: OmniposeFineTuneConfig | None = None) -> Path:
    """Return finetune omnipose two-dimensional data time for the supplied inputs.

    Args:
        cfg (OmniposeFineTuneConfig | None): Value specifying cfg for the operation. ``None`` selects the function's default behavior.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = finetune_omnipose_2d_time()
    """
    if cfg is None:
        cfg = OmniposeFineTuneConfig(project_root=find_project_root())
    data_dir = _prepare_folder(cfg)
    out = cfg.model_dir()
    write_json(out / "training_config.json", cfg)
    cmd = [
        sys.executable, "-m", "omnipose", "--train",
        "--dir", str(data_dir),
        "--img_filter", "_img",
        "--mask_filter", "_masks",
        "--pretrained_model", str(cfg.pretrained_model),
        "--learning_rate", str(cfg.learning_rate),
        "--batch_size", str(cfg.batch_size),
        "--n_epochs", str(cfg.n_epochs),
        "--diameter", str(cfg.diameter),
    ]
    if cfg.use_gpu:
        cmd.append("--use_gpu")
    (out / "omnipose_train_command.txt").write_text(" ".join(cmd), encoding="utf-8")
    subprocess.run(cmd, check=True, cwd=str(out))
    candidates = sorted(out.glob("**/*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True)
    model_files = [p for p in candidates if p.is_file() and ("models" in str(p).lower() or p.suffix in (".pth", ".pt"))]
    model_path = model_files[0] if model_files else out
    print(f"[Omnipose {DATASET}] training finished: {model_path}")
    return model_path
