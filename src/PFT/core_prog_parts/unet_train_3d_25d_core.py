from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random

import numpy as np
import tifffile as tiff
import tensorflow as tf

from PFT.core_prog_parts.common_paths import find_project_root, ensure_dir
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, ensure_czyx
from PFT.core_prog_parts.unet_train_2d_time_core import (
    build_unet,
    bce_dice_loss,
    dice_coef,
    iou_coef,
    normalize_image01,
    save_training_graph,
)


@dataclass
class UNet25DTrainConfig:
    project_root: Path = find_project_root(Path(__file__).resolve())
    dataset: str = "3d_25d"
    level: int = 0

    image_root: Path | None = None
    mask_root: Path | None = None
    model_root: Path | None = None

    patch: int = 256
    batch: int = 4
    epochs: int = 50
    steps_per_epoch: int = 250
    val_steps: int = 50
    val_split: float = 0.2
    seed: int = 1337

    lr: float = 1e-3
    base_filters: int = 16
    dropout: float = 0.0
    normalize: str = "percentile"

    # 2.5D context: radius=1 means z-1, z, z+1.
    z_radius: int = 1
    channels: tuple[int, ...] | None = None  # None = all image channels.

    fg_fraction: float = 0.75
    fg_min_ratio: float = 0.02
    bg_max_ratio: float = 0.01
    max_tries: int = 100


def _default_image_root(project_root: Path) -> Path:
    return project_root / "results" / "img" / "3d_data"


def _default_mask_root(project_root: Path) -> Path:
    return project_root / "results" / "3d_label"


def _default_model_root(project_root: Path) -> Path:
    return project_root / "models" / "u_net_3d_25d"


def _find_ome_zarr_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.ome.zarr") if p.is_dir()])


def _sample_name_from_zarr(zarr_path: Path) -> str:
    # Usually: .../<sample>/image.ome.zarr
    return zarr_path.parent.name if zarr_path.name == "image.ome.zarr" else zarr_path.stem.replace(".ome", "")


def _find_3d_mask(mask_root: Path, sample: str) -> Path | None:
    direct = mask_root / sample
    candidates = []
    if direct.exists():
        candidates += [
            direct / "mask.tif",
            direct / "mask.tiff",
            direct / "labels.tif",
            direct / "label.tif",
            direct / "mask.ome.zarr",
            direct / "labels.ome.zarr",
        ]
        candidates += sorted(direct.glob("*.tif"))
        candidates += sorted(direct.glob("*.tiff"))
        candidates += sorted(direct.glob("*.ome.zarr"))
    candidates += [
        mask_root / f"{sample}.tif",
        mask_root / f"{sample}.tiff",
        mask_root / f"{sample}.ome.zarr",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def list_3d_training_pairs(cfg: UNet25DTrainConfig) -> list[tuple[Path, Path]]:
    image_root = Path(cfg.image_root or _default_image_root(cfg.project_root))
    mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root))
    if not image_root.exists():
        raise FileNotFoundError(f"3D image root does not exist: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"3D mask root does not exist: {mask_root}")

    pairs = []
    for zarr_path in _find_ome_zarr_dirs(image_root):
        sample = _sample_name_from_zarr(zarr_path)
        mask_path = _find_3d_mask(mask_root, sample)
        if mask_path is not None:
            pairs.append((zarr_path, mask_path))
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], cfg: UNet25DTrainConfig):
    rng = random.Random(cfg.seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    if len(pairs) == 1:
        return pairs, pairs
    n_val = max(1, int(round(len(pairs) * cfg.val_split)))
    return pairs[n_val:], pairs[:n_val]


def load_3d_image_czyx(zarr_path: Path, *, level: int = 0) -> np.ndarray:
    arr, axes = load_ome_zarr(zarr_path, level=level, as_numpy=True)
    arr, axes = ensure_czyx(arr, axes)
    return np.asarray(arr, dtype=np.float32)


def load_3d_mask_zyx(mask_path: Path, expected_zyx: tuple[int, int, int] | None = None) -> np.ndarray:
    if mask_path.suffix.lower() == ".zarr" or mask_path.name.endswith(".ome.zarr"):
        arr, axes = load_ome_zarr(mask_path, level=0, as_numpy=True)
        arr, axes = ensure_czyx(arr, axes)
        m = np.asarray(arr)
        # If mask has a C dimension, use first channel.
        m = m[0]
    else:
        m = tiff.imread(mask_path)
        m = np.asarray(m)
        # common cases: ZYX, CZYX, TZYX, YX single mask
        if m.ndim == 4:
            m = m[0]
        if m.ndim == 2 and expected_zyx is not None:
            z = expected_zyx[0]
            m = np.repeat(m[None, ...], z, axis=0)
        while m.ndim > 3:
            m = m[0]
    m = (m > 0).astype(np.uint8)
    if expected_zyx is not None and m.shape != expected_zyx:
        raise ValueError(f"3D mask shape mismatch for {mask_path}: mask={m.shape}, expected={expected_zyx}")
    return m


def make_25d_input_slice(img_czyx: np.ndarray, z: int, cfg: UNet25DTrainConfig) -> np.ndarray:
    c_total, z_total, _, _ = img_czyx.shape
    channels = cfg.channels if cfg.channels is not None else tuple(range(c_total))
    planes = []
    for dz in range(-cfg.z_radius, cfg.z_radius + 1):
        zz = int(np.clip(z + dz, 0, z_total - 1))
        for c in channels:
            if c < 0 or c >= c_total:
                raise IndexError(f"Requested channel {c}, but image has {c_total} channels.")
            planes.append(img_czyx[c, zz])
    return np.stack(planes, axis=-1).astype(np.float32, copy=False)


def _choose_patch(mask_yx: np.ndarray, cfg: UNet25DTrainConfig, rng: random.Random) -> tuple[int, int]:
    H, W = mask_yx.shape
    P = cfg.patch
    if H < P or W < P:
        raise ValueError(f"Patch {P} is larger than slice size {(H, W)}")
    want_fg = rng.random() < cfg.fg_fraction
    fg = np.argwhere(mask_yx > 0)
    best = None
    best_score = None
    for _ in range(cfg.max_tries):
        if want_fg and len(fg) > 0:
            cy, cx = fg[rng.randint(0, len(fg) - 1)]
            y0 = int(np.clip(cy - P // 2 + rng.randint(-P // 8, P // 8), 0, H - P))
            x0 = int(np.clip(cx - P // 2 + rng.randint(-P // 8, P // 8), 0, W - P))
        else:
            y0 = rng.randint(0, H - P)
            x0 = rng.randint(0, W - P)
        crop = mask_yx[y0:y0 + P, x0:x0 + P]
        ratio = float(crop.mean())
        if want_fg and ratio >= cfg.fg_min_ratio:
            return y0, x0
        if (not want_fg) and ratio <= cfg.bg_max_ratio:
            return y0, x0
        score = ratio if want_fg else -ratio
        if best is None or score > best_score:
            best = (y0, x0)
            best_score = score
    return best if best is not None else (rng.randint(0, H - P), rng.randint(0, W - P))


def make_25d_dataset(pairs: list[tuple[Path, Path]], cfg: UNet25DTrainConfig, *, training: bool):
    rng = random.Random(cfg.seed + (0 if training else 10000))
    first_img = load_3d_image_czyx(pairs[0][0], level=cfg.level)
    channels = len(cfg.channels if cfg.channels is not None else tuple(range(first_img.shape[0]))) * (2 * cfg.z_radius + 1)

    def gen():
        cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        while True:
            img_path, mask_path = pairs[rng.randint(0, len(pairs) - 1)]
            if img_path not in cache:
                img = load_3d_image_czyx(img_path, level=cfg.level)
                mask = load_3d_mask_zyx(mask_path, expected_zyx=img.shape[1:])
                cache[img_path] = (img, mask)
            img, mask = cache[img_path]
            z_total = mask.shape[0]
            fg_slices = np.where(mask.reshape(z_total, -1).sum(axis=1) > 0)[0]
            if len(fg_slices) > 0 and rng.random() < cfg.fg_fraction:
                z = int(fg_slices[rng.randint(0, len(fg_slices) - 1)])
            else:
                z = rng.randint(0, z_total - 1)
            plane = make_25d_input_slice(img, z, cfg)
            mask_yx = mask[z]
            y0, x0 = _choose_patch(mask_yx, cfg, rng)
            x = plane[y0:y0 + cfg.patch, x0:x0 + cfg.patch, :]
            y = mask_yx[y0:y0 + cfg.patch, x0:x0 + cfg.patch]
            x = normalize_image01(x, cfg.normalize).astype(np.float32)
            y = y.astype(np.float32)[..., None]
            yield x, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(cfg.patch, cfg.patch, channels), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg.patch, cfg.patch, 1), dtype=tf.float32),
        ),
    )
    if training:
        ds = ds.shuffle(128, seed=cfg.seed, reshuffle_each_iteration=True)
    return ds.batch(cfg.batch).prefetch(tf.data.AUTOTUNE), channels


def train_3d_25d_unet(cfg: UNet25DTrainConfig | None = None) -> dict[str, Path]:
    cfg = cfg or UNet25DTrainConfig()
    cfg.image_root = Path(cfg.image_root or _default_image_root(cfg.project_root))
    cfg.mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root))
    cfg.model_root = ensure_dir(Path(cfg.model_root or _default_model_root(cfg.project_root)))

    pairs = list_3d_training_pairs(cfg)
    if not pairs:
        raise RuntimeError(f"No 3D training pairs found. Images: {cfg.image_root} | Masks: {cfg.mask_root}")
    train_pairs, val_pairs = split_pairs(pairs, cfg)
    train_ds, channels = make_25d_dataset(train_pairs, cfg, training=True)
    val_ds, _ = make_25d_dataset(val_pairs, cfg, training=False)

    model = build_unet((cfg.patch, cfg.patch, channels), base_filters=cfg.base_filters, dropout=cfg.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef],
    )

    best_model = cfg.model_root / "u_net_3d_25d_best.keras"
    final_model = cfg.model_root / "u_net_3d_25d_final.keras"
    graph_png = cfg.model_root / "u_net_3d_25d_training_graph.png"
    summary_json = cfg.model_root / "u_net_3d_25d_training_summary.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_model), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(cfg.model_root / "u_net_3d_25d_history.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
    )
    model.save(final_model)
    save_training_graph(history, graph_png)

    summary = {
        "dataset": "3d_25d",
        "model_input_channels": channels,
        "z_radius": cfg.z_radius,
        "selected_image_channels": cfg.channels,
        "pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "image_root": str(cfg.image_root),
        "mask_root": str(cfg.mask_root),
        "patch": cfg.patch,
        "batch": cfg.batch,
        "epochs": cfg.epochs,
        "best_model": str(best_model),
        "final_model": str(final_model),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"best_model": best_model, "final_model": final_model, "graph": graph_png, "summary": summary_json}
