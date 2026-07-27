from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from PFT.core_prog_parts.common_paths import find_project_root, ensure_dir
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, normalize_axes, move_yx_to_last
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (
    bce_dice_loss,
    dice_coef,
    dice_loss,
    iou_coef,
    normalize_image01,
    ome_zarr_to_hwc_2d,
)
from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import load_3d_image_czyx, make_25d_input_slice


@dataclass
class UNetRunConfig:
    project_root: Path = find_project_root(Path(__file__).resolve())
    dataset: str = "2d_time"  # 2d_time, 2d_wga_dapi, 3d_25d
    level: int = 0
    patch: int = 256
    threshold: float = 0.5
    normalize: str = "percentile"

    model_path: Path | None = None
    input_root: Path | None = None
    out_root: Path | None = None

    # 3D 2.5D options must match training.
    z_radius: int = 1
    channels: tuple[int, ...] | None = None


def find_omezarr_dirs(root: Path) -> list[Path]:
    return sorted([p for p in Path(root).rglob("*.ome.zarr") if p.is_dir()])


def default_model_path(project_root: Path, dataset: str) -> Path:
    if dataset == "3d_25d":
        return project_root / "models" / "u_net_3d_25d" / "u_net_3d_25d_best.keras"
    return project_root / "models" / f"u_net_{dataset}" / f"u_net_{dataset}_best.keras"


def default_input_root(project_root: Path, dataset: str) -> Path:
    if dataset == "3d_25d":
        return project_root / "results" / "img" / "3d_data"
    # Inference should use the same image family as training whenever possible.
    filtered = project_root / "results" / "img" / "filtered" / dataset
    if filtered.exists():
        return filtered
    return project_root / "results" / "img" / "2d_data" / dataset


def default_out_root(project_root: Path, dataset: str) -> Path:
    if dataset == "3d_25d":
        return project_root / "results" / "denoised_images" / "3d"
    return project_root / "results" / "U-net" / dataset


def load_unet_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_loss": dice_loss,
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
        },
    )


def pad_to_patch(img_hwc: np.ndarray, patch: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w, c = img_hwc.shape
    pad_h = (patch - (h % patch)) % patch
    pad_w = (patch - (w % patch)) % patch
    if pad_h == 0 and pad_w == 0:
        return img_hwc, (0, 0)
    return np.pad(img_hwc, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"), (pad_h, pad_w)


def predict_2d_tiled(model: tf.keras.Model, img_hwc: np.ndarray, cfg: UNetRunConfig) -> np.ndarray:
    img_norm = normalize_image01(img_hwc, cfg.normalize).astype(np.float32)
    img_pad, (pad_h, pad_w) = pad_to_patch(img_norm, cfg.patch)
    H, W, _ = img_pad.shape
    pred = np.zeros((H, W), dtype=np.float32)

    tiles = []
    coords = []
    for y0 in range(0, H, cfg.patch):
        for x0 in range(0, W, cfg.patch):
            tiles.append(img_pad[y0:y0 + cfg.patch, x0:x0 + cfg.patch, :])
            coords.append((y0, x0))

    batch = np.stack(tiles, axis=0).astype(np.float32)
    pred_batch = model.predict(batch, verbose=0)
    for k, (y0, x0) in enumerate(coords):
        pred[y0:y0 + cfg.patch, x0:x0 + cfg.patch] = pred_batch[k, ..., 0]

    if pad_h:
        pred = pred[:-pad_h, :]
    if pad_w:
        pred = pred[:, :-pad_w]
    return (pred >= cfg.threshold).astype(np.uint8)


def _sample_name_from_zarr(zarr_path: Path) -> str:
    return zarr_path.parent.name if zarr_path.name == "image.ome.zarr" else zarr_path.stem.replace(".ome", "")


def run_2d_time_unet_on_omezarr(zarr_path: Path, cfg: UNetRunConfig, model: tf.keras.Model | None = None) -> Path:
    cfg.dataset = "2d_time"
    cfg.model_path = Path(cfg.model_path or default_model_path(cfg.project_root, cfg.dataset))
    cfg.out_root = ensure_dir(Path(cfg.out_root or default_out_root(cfg.project_root, cfg.dataset)))
    model = model or load_unet_model(cfg.model_path)

    img = ome_zarr_to_hwc_2d(Path(zarr_path), dataset="2d_time", level=cfg.level)
    mask = predict_2d_tiled(model, img, cfg)

    out_dir = ensure_dir(cfg.out_root / _sample_name_from_zarr(Path(zarr_path)))
    out_zarr = out_dir / "pred_mask.ome.zarr"
    save_ome_zarr(out_zarr, mask.astype(np.uint8), "yx", meta=None, overwrite=True, pyramid_3d=False, pyramid_max_layer=0)
    return out_zarr


def run_2d_wga_dapi_unet_on_omezarr(zarr_path: Path, cfg: UNetRunConfig, model: tf.keras.Model | None = None) -> Path:
    cfg.dataset = "2d_wga_dapi"
    cfg.model_path = Path(cfg.model_path or default_model_path(cfg.project_root, cfg.dataset))
    cfg.out_root = ensure_dir(Path(cfg.out_root or default_out_root(cfg.project_root, cfg.dataset)))
    model = model or load_unet_model(cfg.model_path)

    img = ome_zarr_to_hwc_2d(Path(zarr_path), dataset="2d_wga_dapi", level=cfg.level)
    mask = predict_2d_tiled(model, img, cfg)

    out_dir = ensure_dir(cfg.out_root / _sample_name_from_zarr(Path(zarr_path)))
    out_zarr = out_dir / "pred_mask.ome.zarr"
    save_ome_zarr(out_zarr, mask.astype(np.uint8), "yx", meta=None, overwrite=True, pyramid_3d=False, pyramid_max_layer=0)
    return out_zarr


def run_3d_25d_unet_on_omezarr(zarr_path: Path, cfg: UNetRunConfig, model: tf.keras.Model | None = None) -> Path:
    cfg.dataset = "3d_25d"
    cfg.model_path = Path(cfg.model_path or default_model_path(cfg.project_root, cfg.dataset))
    cfg.out_root = ensure_dir(Path(cfg.out_root or default_out_root(cfg.project_root, cfg.dataset)))
    model = model or load_unet_model(cfg.model_path)

    img_czyx = load_3d_image_czyx(Path(zarr_path), level=cfg.level)
    _, z_total, y, x = img_czyx.shape
    pred_zyx = np.zeros((z_total, y, x), dtype=np.uint8)

    for z in range(z_total):
        plane_hwc = make_25d_input_slice(img_czyx, z, cfg)
        pred_zyx[z] = predict_2d_tiled(model, plane_hwc, cfg)

    out_dir = ensure_dir(cfg.out_root / _sample_name_from_zarr(Path(zarr_path)))
    out_zarr = out_dir / "pred_mask.ome.zarr"
    save_ome_zarr(
        out_zarr,
        pred_zyx.astype(np.uint8),
        "zyx",
        meta=None,
        overwrite=True,
        pyramid_3d=True,
        pyramid_max_layer=2,
        pyramid_downscale=2,
    )
    return out_zarr


def run_dataset(cfg: UNetRunConfig, *, process_all: bool = True, selected_zarrs: list[Path] | None = None) -> list[Path]:
    cfg.dataset = cfg.dataset.strip().lower()
    if cfg.dataset in {"3d", "3d_data", "3d_25d"}:
        cfg.dataset = "3d_25d"
    if cfg.dataset in {"2d_wagi_dapi", "2d_wga_dapi", "2d_dpa_wagi"}:
        cfg.dataset = "2d_wga_dapi"

    cfg.model_path = Path(cfg.model_path or default_model_path(cfg.project_root, cfg.dataset))
    cfg.input_root = Path(cfg.input_root or default_input_root(cfg.project_root, cfg.dataset))
    cfg.out_root = ensure_dir(Path(cfg.out_root or default_out_root(cfg.project_root, cfg.dataset)))

    model = load_unet_model(cfg.model_path)
    if selected_zarrs is not None:
        targets = [Path(p) for p in selected_zarrs]
    else:
        all_zarrs = find_omezarr_dirs(cfg.input_root)
        if not all_zarrs:
            raise RuntimeError(f"No OME-Zarr folders found under: {cfg.input_root}")
        targets = all_zarrs if process_all else [all_zarrs[0]]

    outputs = []
    for zarr_path in targets:
        if cfg.dataset == "2d_time":
            outputs.append(run_2d_time_unet_on_omezarr(zarr_path, cfg, model=model))
        elif cfg.dataset == "2d_wga_dapi":
            outputs.append(run_2d_wga_dapi_unet_on_omezarr(zarr_path, cfg, model=model))
        elif cfg.dataset == "3d_25d":
            outputs.append(run_3d_25d_unet_on_omezarr(zarr_path, cfg, model=model))
        else:
            raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    return outputs
