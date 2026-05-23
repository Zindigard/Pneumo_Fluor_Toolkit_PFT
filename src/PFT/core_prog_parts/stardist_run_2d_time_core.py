from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation_model_io_core import (
    extract_2d_image, extract_3d_volume, list_input_images, model_root, prediction_root,
    safe_sample_name, save_label_outputs, write_json,
)

DATASET = "2d_time"
FAMILY = "stardist"


def _load_model(project_root: Path, model_dir: Path | None = None):
    from stardist.models import StarDist2D
    if model_dir is None:
        root = model_root(project_root, FAMILY, DATASET)
        model_dir = root / "model"
    model_dir = Path(model_dir)
    return StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent)), model_dir


@dataclass
class StarDistRunConfig:
    project_root: Path
    model_dir: Path | None = None
    prob_thresh: float | None = None
    nms_thresh: float | None = None
    channels: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = (0, 1) if DATASET == "2d_wga_dapi" else (0,)


def run_stardist_2d_time(cfg: StarDistRunConfig | None = None) -> list[Path]:
    if cfg is None:
        cfg = StarDistRunConfig(project_root=find_project_root())
    model, model_dir = _load_model(cfg.project_root, cfg.model_dir)
    out_root = prediction_root(cfg.project_root, FAMILY, DATASET, model_dir.name)
    write_json(out_root / "run_config.json", cfg)
    outputs = []
    for image_path in list_input_images(cfg.project_root, DATASET):
        sample = safe_sample_name(image_path)
        if DATASET == "3d":
            vol = extract_3d_volume(image_path, channels=cfg.channels)
            labels = np.zeros(vol.shape[:3] if vol.ndim == 4 else vol.shape, dtype=np.uint16)
            for z in range(vol.shape[0]):
                lab, _ = model.predict_instances(vol[z], prob_thresh=cfg.prob_thresh, nms_thresh=cfg.nms_thresh)
                labels[z] = lab.astype(np.uint16)
            zarr_path = save_label_outputs(labels, out_root / sample, axes="zyx")
        else:
            img = extract_2d_image(image_path, DATASET, channels=cfg.channels)
            labels, _ = model.predict_instances(img, prob_thresh=cfg.prob_thresh, nms_thresh=cfg.nms_thresh)
            zarr_path = save_label_outputs(labels.astype(np.uint16), out_root / sample, axes="yx")
        outputs.append(zarr_path)
        print(f"[StarDist {DATASET}] saved: {zarr_path}")
    return outputs
