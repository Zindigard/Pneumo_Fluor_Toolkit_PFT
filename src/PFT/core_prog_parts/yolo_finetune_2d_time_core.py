from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff

from PFT.core_prog_parts.common_paths import find_project_root, normalize_dataset_name
from PFT.core_prog_parts.segmentation_model_io_core import (
    collect_training_pairs, extract_2d_image, extract_3d_volume, load_2d_mask, load_3d_mask,
    model_root, to_uint8, write_json,
)

DATASET = "2d_time"
FAMILY = "yolo"


def _mask_to_yolo_segments(mask: np.ndarray, min_area: int = 10) -> list[str]:
    import cv2
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"YOLO labels require 2D masks, got shape={m.shape}")
    if m.max() <= 1:
        n, cc = cv2.connectedComponents((m > 0).astype(np.uint8))
        labels = range(1, n)
        label_img = cc
    else:
        labels = [v for v in np.unique(m) if v != 0]
        label_img = m
    h, w = m.shape
    rows: list[str] = []
    for lab in labels:
        obj = (label_img == lab).astype(np.uint8)
        if int(obj.sum()) < min_area:
            continue
        contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area or len(cnt) < 3:
                continue
            cnt = cnt.reshape(-1, 2).astype(float)
            pts = []
            for x, y in cnt:
                pts.extend([max(0.0, min(1.0, x / max(w, 1))), max(0.0, min(1.0, y / max(h, 1)))])
            if len(pts) >= 6:
                rows.append("0 " + " ".join(f"{v:.6f}" for v in pts))
    return rows


def _save_yolo_item(img: np.ndarray, mask: np.ndarray, image_path: Path, label_path: Path) -> bool:
    import cv2
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _mask_to_yolo_segments(mask)
    if not rows:
        return False
    u8 = to_uint8(img)
    if u8.ndim == 2:
        bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    else:
        if u8.shape[-1] == 1:
            bgr = cv2.cvtColor(u8[..., 0], cv2.COLOR_GRAY2BGR)
        else:
            rgb = u8[..., :3] if u8.shape[-1] >= 3 else np.repeat(u8[..., :1], 3, axis=-1)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), bgr)
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return True


def _prepare_dataset(cfg: "YOLOFineTuneConfig") -> Path:
    pairs = collect_training_pairs(cfg.project_root, DATASET)
    out = cfg.model_dir() / "prepared_yolo_dataset"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    random.Random(cfg.seed).shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * cfg.val_fraction))) if len(pairs) > 1 else 0
    val_names = {sample for _, _, sample in pairs[:n_val]}
    kept = 0
    for image_path, mask_path, sample in pairs:
        split = "val" if sample in val_names else "train"
        if DATASET == "3d":
            vol = extract_3d_volume(image_path, channels=cfg.channels)
            mask = load_3d_mask(mask_path)
            if vol.ndim == 4:
                z_count = min(vol.shape[0], mask.shape[0])
                for z in range(z_count):
                    name = f"{sample}_z{z:03d}"
                    kept += int(_save_yolo_item(vol[z], mask[z], out/"images"/split/f"{name}.png", out/"labels"/split/f"{name}.txt"))
            else:
                z_count = min(vol.shape[0], mask.shape[0])
                for z in range(z_count):
                    name = f"{sample}_z{z:03d}"
                    kept += int(_save_yolo_item(vol[z], mask[z], out/"images"/split/f"{name}.png", out/"labels"/split/f"{name}.txt"))
        else:
            img = extract_2d_image(image_path, DATASET, channels=cfg.channels)
            mask = load_2d_mask(mask_path)
            kept += int(_save_yolo_item(img, mask, out/"images"/split/f"{sample}.png", out/"labels"/split/f"{sample}.txt"))
    if kept == 0:
        raise RuntimeError("No valid YOLO image/label pairs were prepared. Check masks and object sizes.")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['cell']\n",
        encoding="utf-8",
    )
    return data_yaml


@dataclass
class YOLOFineTuneConfig:
    project_root: Path
    base_model: str = "yolov8n-seg.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    workers: int = 0
    device: str | int | None = None
    val_fraction: float = 0.2
    seed: int = 42
    channels: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = (0, 1) if DATASET == "2d_wga_dapi" else (0,)

    def model_dir(self) -> Path:
        return model_root(self.project_root, FAMILY, DATASET)


def finetune_yolo_2d_time(cfg: YOLOFineTuneConfig | None = None) -> Path:
    if cfg is None:
        cfg = YOLOFineTuneConfig(project_root=find_project_root())
    from ultralytics import YOLO
    data_yaml = _prepare_dataset(cfg)
    out = cfg.model_dir()
    write_json(out / "training_config.json", cfg)
    model = YOLO(cfg.base_model)
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        workers=cfg.workers,
        project=str(out),
        name="train",
        exist_ok=True,
        device=cfg.device,
    )
    best = out / "train" / "weights" / "best.pt"
    if not best.exists():
        best = out / "train" / "weights" / "last.pt"
    print(f"[YOLO {DATASET}] model saved: {best}")
    return best
