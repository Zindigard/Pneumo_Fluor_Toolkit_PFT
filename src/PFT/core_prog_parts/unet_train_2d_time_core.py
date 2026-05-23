from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Iterable

import numpy as np
import tifffile as tiff
import tensorflow as tf
import matplotlib.pyplot as plt

from PFT.core_prog_parts.common_paths import find_project_root, ensure_dir
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, normalize_axes, move_yx_to_last


@dataclass
class UNet2DTrainConfig:
    project_root: Path = find_project_root(Path(__file__).resolve())
    dataset: str = "2d_time"
    level: int = 0

    image_root: Path | None = None
    mask_root: Path | None = None
    model_root: Path | None = None

    patch: int = 256
    batch: int = 8
    epochs: int = 50
    steps_per_epoch: int = 300
    val_steps: int = 60
    val_split: float = 0.2
    seed: int = 1337

    lr: float = 1e-3
    base_filters: int = 16
    dropout: float = 0.0
    normalize: str = "percentile"

    fg_fraction: float = 0.75
    fg_min_ratio: float = 0.05
    bg_max_ratio: float = 0.02
    max_tries: int = 80


def _default_image_root(project_root: Path, dataset: str) -> Path:
    # U-Net training should normally use filtered/preprocessed images.
    return project_root / "results" / "img" / "filtered" / dataset


def _default_mask_root(project_root: Path, dataset: str) -> Path:
    return project_root / "results" / "training_files" / "U-net" / dataset


def _default_model_root(project_root: Path, dataset: str) -> Path:
    return project_root / "models" / f"u_net_{dataset}"


def normalize_image01(x: np.ndarray, mode: str = "percentile") -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if mode == "scale_uint16":
        return np.clip(x / 65535.0, 0.0, 1.0).astype(np.float32)
    if mode != "percentile":
        raise ValueError(f"Unknown normalization mode: {mode}")

    out = np.empty_like(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
        squeeze = True
    else:
        squeeze = False

    for c in range(x.shape[-1]):
        xc = x[..., c]
        lo = np.percentile(xc, 1.0)
        hi = np.percentile(xc, 99.8)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((xc - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return out[..., 0] if squeeze else out


def _as_numpy(a) -> np.ndarray:
    return np.asarray(a)


def _select_first_time_if_present(arr: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = normalize_axes(axes)
    if "t" not in axes:
        return arr, axes
    ax = axes.index("t")
    arr = np.take(arr, 0, axis=ax)
    axes = axes[:ax] + axes[ax + 1:]
    return arr, axes


def ome_zarr_to_hwc_2d(zarr_path: Path, *, dataset: str, level: int = 0) -> np.ndarray:
    arr, axes = load_ome_zarr(zarr_path, level=level, as_numpy=True)
    axes = normalize_axes(axes)
    arr, axes = _select_first_time_if_present(arr, axes)
    arr, axes = move_yx_to_last(arr, axes)

    if axes == "yx":
        img = arr[..., None]
    elif axes == "cyx":
        c_axis = axes.index("c")
        c_count = arr.shape[c_axis]
        if dataset == "2d_time":
            img = np.take(arr, 0, axis=c_axis)[..., None]
        elif dataset == "2d_wga_dapi":
            use_channels = [0, 1] if c_count > 1 else [0]
            planes = [np.take(arr, ci, axis=c_axis) for ci in use_channels]
            img = np.stack(planes, axis=-1)
        else:
            raise ValueError(f"Unsupported 2D dataset: {dataset}")
    else:
        raise ValueError(f"Unsupported 2D OME-Zarr axes '{axes}' for {zarr_path}")

    return img.astype(np.float32, copy=False)


def read_binary_mask_2d(mask_path: Path) -> np.ndarray:
    m = tiff.imread(mask_path)
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    return (m > 0).astype(np.uint8)


def find_mask_file(sample_dir: Path) -> Path | None:
    candidates = ["mask.tif", "mask.tiff", "labels.tif", "label.tif", "mask.png"]
    for name in candidates:
        p = sample_dir / name
        if p.exists():
            return p
    tifs = sorted(list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.tiff")))
    return tifs[0] if tifs else None


def list_2d_training_pairs(cfg: UNet2DTrainConfig) -> list[tuple[Path, Path]]:
    image_root = Path(cfg.image_root or _default_image_root(cfg.project_root, cfg.dataset))
    mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root, cfg.dataset))
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    pairs: list[tuple[Path, Path]] = []
    for sample_dir in sorted(mask_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        mask_path = find_mask_file(sample_dir)
        if mask_path is None:
            continue
        zarr_path = image_root / sample_dir.name / "image.ome.zarr"
        if zarr_path.exists():
            pairs.append((zarr_path, mask_path))
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]], cfg: UNet2DTrainConfig) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(cfg.seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    if len(pairs) == 1:
        return pairs, pairs
    n_val = max(1, int(round(len(pairs) * cfg.val_split)))
    return pairs[n_val:], pairs[:n_val]


def _random_crop(H: int, W: int, patch: int, rng: random.Random) -> tuple[int, int]:
    return rng.randint(0, H - patch), rng.randint(0, W - patch)


def sample_patch_2d(img: np.ndarray, mask: np.ndarray, cfg: UNet2DTrainConfig, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    H, W = mask.shape
    P = cfg.patch
    if H < P or W < P:
        raise ValueError(f"Patch {P} is larger than image/mask {(H, W)}")

    want_fg = rng.random() < cfg.fg_fraction
    fg = np.argwhere(mask > 0)
    best = None
    best_score = None

    for _ in range(cfg.max_tries):
        if want_fg and len(fg) > 0:
            cy, cx = fg[rng.randint(0, len(fg) - 1)]
            y0 = int(np.clip(cy - P // 2 + rng.randint(-P // 8, P // 8), 0, H - P))
            x0 = int(np.clip(cx - P // 2 + rng.randint(-P // 8, P // 8), 0, W - P))
        else:
            y0, x0 = _random_crop(H, W, P, rng)

        im = img[y0:y0 + P, x0:x0 + P, :]
        ma = mask[y0:y0 + P, x0:x0 + P]
        ratio = float(ma.mean())
        if want_fg and ratio >= cfg.fg_min_ratio:
            return im, ma
        if (not want_fg) and ratio <= cfg.bg_max_ratio:
            return im, ma

        score = ratio if want_fg else -ratio
        if best is None or score > best_score:
            best = (im, ma)
            best_score = score

    if best is not None:
        return best
    y0, x0 = _random_crop(H, W, P, rng)
    return img[y0:y0 + P, x0:x0 + P, :], mask[y0:y0 + P, x0:x0 + P]


def make_2d_dataset(pairs: list[tuple[Path, Path]], cfg: UNet2DTrainConfig, *, training: bool):
    rng = random.Random(cfg.seed + (0 if training else 10000))
    first_img = ome_zarr_to_hwc_2d(pairs[0][0], dataset=cfg.dataset, level=cfg.level)
    channels = first_img.shape[-1]

    def gen():
        while True:
            img_path, mask_path = pairs[rng.randint(0, len(pairs) - 1)]
            img = ome_zarr_to_hwc_2d(img_path, dataset=cfg.dataset, level=cfg.level)
            mask = read_binary_mask_2d(mask_path)
            if img.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Image/mask shape mismatch: {img_path} {img.shape[:2]} vs {mask_path} {mask.shape[:2]}")
            x, y = sample_patch_2d(img, mask, cfg, rng)
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
        ds = ds.shuffle(256, seed=cfg.seed, reshuffle_each_iteration=True)
    return ds.batch(cfg.batch).prefetch(tf.data.AUTOTUNE), channels


def conv_block(x, filters: int, dropout: float = 0.0):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_unet(input_shape: tuple[int, int, int], base_filters: int = 16, dropout: float = 0.0) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    c1 = conv_block(inputs, base_filters, dropout)
    p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1, base_filters * 2, dropout)
    p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2, base_filters * 4, dropout)
    p3 = tf.keras.layers.MaxPool2D()(c3)
    c4 = conv_block(p3, base_filters * 8, dropout)
    p4 = tf.keras.layers.MaxPool2D()(c4)
    bn = conv_block(p4, base_filters * 16, dropout)
    u4 = tf.keras.layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bn)
    c5 = conv_block(tf.keras.layers.Concatenate()([u4, c4]), base_filters * 8, dropout)
    u3 = tf.keras.layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c5)
    c6 = conv_block(tf.keras.layers.Concatenate()([u3, c3]), base_filters * 4, dropout)
    u2 = tf.keras.layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c6)
    c7 = conv_block(tf.keras.layers.Concatenate()([u2, c2]), base_filters * 2, dropout)
    u1 = tf.keras.layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c7)
    c8 = conv_block(tf.keras.layers.Concatenate()([u1, c1]), base_filters, dropout)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return tf.keras.Model(inputs, outputs, name="UNet2D")


def dice_coef(y_true, y_pred, eps: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * inter + eps) / (denom + eps))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * tf.reduce_mean(bce) + 0.5 * dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, eps: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + eps) / (union + eps))


def save_training_graph(history: tf.keras.callbacks.History, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    hist = history.history
    fig = plt.figure(figsize=(8, 5), dpi=180)
    ax = fig.add_subplot(1, 1, 1)
    for key in ["loss", "val_loss", "dice_coef", "val_dice_coef", "iou_coef", "val_iou_coef"]:
        if key in hist:
            ax.plot(hist[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_title("U-Net training history")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def train_2d_binary_unet(cfg: UNet2DTrainConfig) -> dict[str, Path]:
    cfg.dataset = cfg.dataset.strip().lower()
    cfg.image_root = Path(cfg.image_root or _default_image_root(cfg.project_root, cfg.dataset))
    cfg.mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root, cfg.dataset))
    cfg.model_root = ensure_dir(Path(cfg.model_root or _default_model_root(cfg.project_root, cfg.dataset)))

    pairs = list_2d_training_pairs(cfg)
    if not pairs:
        raise RuntimeError(f"No training pairs found. Images: {cfg.image_root} | Masks: {cfg.mask_root}")
    train_pairs, val_pairs = split_pairs(pairs, cfg)
    train_ds, channels = make_2d_dataset(train_pairs, cfg, training=True)
    val_ds, _ = make_2d_dataset(val_pairs, cfg, training=False)

    model = build_unet((cfg.patch, cfg.patch, channels), base_filters=cfg.base_filters, dropout=cfg.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef],
    )

    best_model = cfg.model_root / f"u_net_{cfg.dataset}_best.keras"
    final_model = cfg.model_root / f"u_net_{cfg.dataset}_final.keras"
    graph_png = cfg.model_root / f"u_net_{cfg.dataset}_training_graph.png"
    summary_json = cfg.model_root / f"u_net_{cfg.dataset}_training_summary.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_model), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(cfg.model_root / f"u_net_{cfg.dataset}_history.csv")),
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
        "dataset": cfg.dataset,
        "channels": channels,
        "pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "image_root": str(cfg.image_root),
        "mask_root": str(cfg.mask_root),
        "patch": cfg.patch,
        "batch": cfg.batch,
        "epochs": cfg.epochs,
        "steps_per_epoch": cfg.steps_per_epoch,
        "val_steps": cfg.val_steps,
        "best_model": str(best_model),
        "final_model": str(final_model),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"best_model": best_model, "final_model": final_model, "graph": graph_png, "summary": summary_json}


def train_2d_time_unet(cfg: UNet2DTrainConfig | None = None) -> dict[str, Path]:
    cfg = cfg or UNet2DTrainConfig()
    cfg.dataset = "2d_time"
    return train_2d_binary_unet(cfg)
