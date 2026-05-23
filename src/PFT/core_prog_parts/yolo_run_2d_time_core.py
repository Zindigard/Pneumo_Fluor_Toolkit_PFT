from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation_model_io_core import (
    extract_2d_image, extract_3d_volume, list_input_images, model_root, prediction_root,
    safe_sample_name, save_label_outputs, to_uint8, write_json,
)

DATASET = "2d_time"
FAMILY = "yolo"


def _default_model(project_root: Path) -> Path:
    root = model_root(project_root, FAMILY, DATASET)
    for p in [root/"train"/"weights"/"best.pt", root/"train"/"weights"/"last.pt"]:
        if p.exists():
            return p
    candidates = sorted(root.glob("**/*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No YOLO .pt model found in {root}")


def _predict_2d(model, img: np.ndarray, conf: float, iou: float) -> np.ndarray:
    import cv2
    u8 = to_uint8(img)
    if u8.ndim == 2:
        pred_img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    else:
        rgb = u8[..., :3] if u8.shape[-1] >= 3 else np.repeat(u8[..., :1], 3, axis=-1)
        pred_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(pred_img, conf=conf, iou=iou, verbose=False)
    labels = np.zeros(u8.shape[:2], dtype=np.uint16)
    if not results:
        return labels
    r = results[0]
    if getattr(r, "masks", None) is None or r.masks is None:
        return labels
    masks = r.masks.data.cpu().numpy()
    for idx, m in enumerate(masks, start=1):
        m = cv2.resize(m.astype(np.uint8), (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_NEAREST)
        labels[m > 0] = idx
    return labels


@dataclass
class YOLORunConfig:
    project_root: Path
    model_path: Path | None = None
    conf: float = 0.25
    iou: float = 0.7
    channels: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = (0, 1) if DATASET == "2d_wga_dapi" else (0,)


def run_yolo_2d_time(cfg: YOLORunConfig | None = None) -> list[Path]:
    if cfg is None:
        cfg = YOLORunConfig(project_root=find_project_root())
    from ultralytics import YOLO
    model_path = Path(cfg.model_path) if cfg.model_path else _default_model(cfg.project_root)
    model = YOLO(str(model_path))
    out_root = prediction_root(cfg.project_root, FAMILY, DATASET, model_path.stem)
    write_json(out_root / "run_config.json", cfg)
    outputs = []
    for image_path in list_input_images(cfg.project_root, DATASET):
        sample = safe_sample_name(image_path)
        if DATASET == "3d":
            vol = extract_3d_volume(image_path, channels=cfg.channels)
            z_count = vol.shape[0]
            labels = np.zeros(vol.shape[:3] if vol.ndim == 4 else vol.shape, dtype=np.uint16)
            for z in range(z_count):
                labels[z] = _predict_2d(model, vol[z], cfg.conf, cfg.iou)
            zarr_path = save_label_outputs(labels, out_root / sample, axes="zyx")
        else:
            img = extract_2d_image(image_path, DATASET, channels=cfg.channels)
            labels = _predict_2d(model, img, cfg.conf, cfg.iou)
            zarr_path = save_label_outputs(labels, out_root / sample, axes="yx")
        outputs.append(zarr_path)
        print(f"[YOLO {DATASET}] saved: {zarr_path}")
    return outputs
