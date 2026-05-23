from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation_model_io_core import (
    extract_2d_image, extract_3d_volume, list_input_images, model_root, prediction_root,
    safe_sample_name, save_label_outputs, write_json,
)

DATASET = "2d_wga_dapi"
FAMILY = "omnipose"


def _default_model(project_root: Path) -> Path | None:
    root = model_root(project_root, FAMILY, DATASET)
    candidates = [p for p in sorted(root.glob("**/*"), key=lambda x: x.stat().st_mtime if x.is_file() else 0, reverse=True) if p.is_file()]
    for p in candidates:
        if p.suffix in (".pth", ".pt") or "models" in str(p).lower():
            return p
    return None


def _build_model(cfg: "OmniposeRunConfig"):
    try:
        from cellpose_omni import models
    except Exception:
        from cellpose import models
    model_path = Path(cfg.model_path) if cfg.model_path else _default_model(cfg.project_root)
    if model_path is not None:
        try:
            return models.CellposeModel(gpu=cfg.gpu, pretrained_model=str(model_path)), model_path
        except TypeError:
            return models.CellposeModel(gpu=cfg.gpu, pretrained_model=str(model_path), nchan=cfg.nchan), model_path
    try:
        return models.Cellpose(gpu=cfg.gpu, model_type=cfg.model_type), Path(cfg.model_type)
    except TypeError:
        return models.CellposeModel(gpu=cfg.gpu, model_type=cfg.model_type), Path(cfg.model_type)


def _eval_2d(model, img: np.ndarray, cfg: "OmniposeRunConfig") -> np.ndarray:
    res = model.eval(
        img,
        channels=cfg.cellpose_channels,
        diameter=cfg.diameter,
        mask_threshold=cfg.mask_threshold,
        flow_threshold=cfg.flow_threshold,
        min_size=cfg.min_size,
    )
    masks = res[0] if isinstance(res, tuple) else res
    return np.asarray(masks).astype(np.uint16)


@dataclass
class OmniposeRunConfig:
    project_root: Path
    model_path: Path | None = None
    model_type: str = "cyto2_omni"
    gpu: bool = True
    nchan: int = 1
    diameter: float | None = None
    mask_threshold: float = 0.0
    flow_threshold: float = 0.4
    min_size: int = 15
    cellpose_channels: tuple[int, int] = (0, 0)
    channels: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = (0, 1) if DATASET == "2d_wga_dapi" else (0,)
        self.nchan = len(self.channels)


def run_omnipose_2d_wga_dapi(cfg: OmniposeRunConfig | None = None) -> list[Path]:
    if cfg is None:
        cfg = OmniposeRunConfig(project_root=find_project_root())
    model, model_ref = _build_model(cfg)
    out_root = prediction_root(cfg.project_root, FAMILY, DATASET, Path(model_ref).stem)
    write_json(out_root / "run_config.json", cfg)
    outputs = []
    for image_path in list_input_images(cfg.project_root, DATASET):
        sample = safe_sample_name(image_path)
        if DATASET == "3d":
            vol = extract_3d_volume(image_path, channels=cfg.channels)
            labels = np.zeros(vol.shape[:3] if vol.ndim == 4 else vol.shape, dtype=np.uint16)
            for z in range(vol.shape[0]):
                labels[z] = _eval_2d(model, vol[z], cfg)
            zarr_path = save_label_outputs(labels, out_root / sample, axes="zyx")
        else:
            img = extract_2d_image(image_path, DATASET, channels=cfg.channels)
            labels = _eval_2d(model, img, cfg)
            zarr_path = save_label_outputs(labels, out_root / sample, axes="yx")
        outputs.append(zarr_path)
        print(f"[Omnipose {DATASET}] saved: {zarr_path}")
    return outputs
