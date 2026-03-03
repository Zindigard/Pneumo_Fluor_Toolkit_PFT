import os
import json
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tifffile as tiff
import tensorflow as tf


@dataclass
class TrainConfig:
    repo_root: Path = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
    dataset: str = "2d_time"  # or 2d_wga_dapi

    data_root: Path = None

    out_root: Path = None

    patch: int = 256
    batch: int = 8
    epochs: int = 50
    lr: float = 1e-3

    steps_per_epoch: int = 300
    val_steps: int = 60

    # patch sampling balance
    fg_fraction: float = 0.8  # 80% foreground-containing crops, 20% background crops
    min_fg_ratio: float = 0.005  # >=0.5% of pixels are foreground to accept fg crop
    max_tries: int = 40

    val_split: float = 0.2
    seed: int = 1337

    normalize: str = "scale_uint16"

    base_filters: int = 32
    dropout: float = 0.0


def list_pairs(data_root: Path):
    pairs = []
    for sample_dir in sorted(data_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        img = sample_dir / "image.tif"
        msk = sample_dir / "mask.tif"
        if img.exists() and msk.exists():
            pairs.append((img, msk))
    return pairs


def read_image_mask_numpy(img_path: str, mask_path: str):
    img = tiff.imread(img_path)
    msk = tiff.imread(mask_path)

    msk = (msk > 0).astype(np.uint8)

    if img.ndim == 2:
        img = img[..., None]  # (H,W,1)
    elif img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3]  # keep RGB
    else:
        raise ValueError(f"Unsupported image shape {img.shape} in {img_path}")

    return img, msk


def normalize_crop_numpy(x: np.ndarray, mode: str):
    # x: (H,W,C)
    x = x.astype(np.float32)

    if mode == "scale_uint16":
        
        x = x / 65535.0
        x = np.clip(x, 0.0, 1.0)
        return x

    if mode == "percentile":
        out = np.empty_like(x, dtype=np.float32)
        for c in range(x.shape[-1]):
            xc = x[..., c]
            p1 = np.percentile(xc, 1)
            p2 = np.percentile(xc, 99.8)
            out[..., c] = (xc - p1) / (p2 - p1 + 1e-8)
        out = np.clip(out, 0.0, 1.0)
        return out

    raise ValueError(f"Unknown normalize mode: {mode}")


def random_crop_xy(H, W, patch):
    y0 = random.randint(0, H - patch)
    x0 = random.randint(0, W - patch)
    return y0, x0


def clamp_crop_center(cy, cx, H, W, patch):
    half = patch // 2
    y0 = int(np.clip(cy - half, 0, H - patch))
    x0 = int(np.clip(cx - half, 0, W - patch))
    return y0, x0


def sample_patch_numpy(img: np.ndarray, msk: np.ndarray, cfg: TrainConfig):
    """
    Foreground-aware sampling:
      - with prob cfg.fg_fraction -> try to sample a crop with enough foreground pixels
      - else -> sample a mostly-background crop
    """
    H, W = msk.shape
    P = cfg.patch

    want_fg = (random.random() < cfg.fg_fraction)
    fg_coords = None

    if want_fg:
        fg_coords = np.argwhere(msk > 0)

    for _ in range(cfg.max_tries):
        if want_fg and fg_coords is not None and len(fg_coords) > 0:
            # pick a random foreground pixel and jitter a bit
            cy, cx = fg_coords[random.randint(0, len(fg_coords) - 1)]
            cy = int(np.clip(cy + random.randint(-P // 4, P // 4), 0, H - 1))
            cx = int(np.clip(cx + random.randint(-P // 4, P // 4), 0, W - 1))
            y0, x0 = clamp_crop_center(cy, cx, H, W, P)
        else:
            y0, x0 = random_crop_xy(H, W, P)

        img_c = img[y0:y0 + P, x0:x0 + P, :]
        msk_c = msk[y0:y0 + P, x0:x0 + P]

        fg_ratio = float(msk_c.mean())

        if want_fg:
            if fg_ratio >= cfg.min_fg_ratio:
                return img_c, msk_c
        else:
            if fg_ratio < max(cfg.min_fg_ratio * 0.2, 0.001):
                return img_c, msk_c

    y0, x0 = random_crop_xy(H, W, P)
    return img[y0:y0 + P, x0:x0 + P, :], msk[y0:y0 + P, x0:x0 + P]



def make_dataset(pairs, cfg: TrainConfig, training: bool):
    rng = random.Random(cfg.seed + (0 if training else 999))

    def gen():
        
        while True:
            img_path, mask_path = pairs[rng.randint(0, len(pairs) - 1)]
            img, msk = read_image_mask_numpy(str(img_path), str(mask_path))

            
            img_c, msk_c = sample_patch_numpy(img, msk, cfg)

            img_c = normalize_crop_numpy(img_c, cfg.normalize)

            msk_c = msk_c.astype(np.float32)[..., None]

            yield img_c, msk_c

    
    first_img, _ = read_image_mask_numpy(str(pairs[0][0]), str(pairs[0][1]))
    C = first_img.shape[-1]
    P = cfg.patch

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(P, P, C), dtype=tf.float32),
            tf.TensorSpec(shape=(P, P, 1), dtype=tf.float32),
        ),
    )

    if training:
        ds = ds.shuffle(256, seed=cfg.seed, reshuffle_each_iteration=True)

    ds = ds.batch(cfg.batch).prefetch(tf.data.AUTOTUNE)
    return ds, C


def conv_block(x, filters, dropout=0.0):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_unet(input_shape, base_filters=32, dropout=0.0):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters, dropout=dropout)
    p1 = tf.keras.layers.MaxPool2D()(c1)

    c2 = conv_block(p1, base_filters * 2, dropout=dropout)
    p2 = tf.keras.layers.MaxPool2D()(c2)

    c3 = conv_block(p2, base_filters * 4, dropout=dropout)
    p3 = tf.keras.layers.MaxPool2D()(c3)

    c4 = conv_block(p3, base_filters * 8, dropout=dropout)
    p4 = tf.keras.layers.MaxPool2D()(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters * 16, dropout=dropout)

    # Decoder
    u4 = tf.keras.layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bn)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters * 8, dropout=dropout)

    u3 = tf.keras.layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters * 4, dropout=dropout)

    u2 = tf.keras.layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters * 2, dropout=dropout)

    u1 = tf.keras.layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters, dropout=dropout)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)

    return tf.keras.Model(inputs, outputs, name="UNet")


def dice_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + eps) / (denom + eps)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return 0.5 * bce + 0.5 * dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + eps) / (union + eps)
    return tf.reduce_mean(iou)


def save_prediction_previews(model, pairs, cfg: TrainConfig, out_dir: Path, n=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.sample(pairs, k=min(n, len(pairs)))

    for i, (img_path, mask_path) in enumerate(chosen, 1):
        img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
        img_c, msk_c = sample_patch_numpy(img, msk, cfg)
        img_c = normalize_crop_numpy(img_c, cfg.normalize)
        msk_c01 = (msk_c > 0).astype(np.uint8)

        pred = model.predict(img_c[None, ...], verbose=0)[0, ..., 0]
        pred01 = (pred > 0.5).astype(np.uint8)

        
        x = img_c
        if x.shape[-1] == 1:
            g = (np.clip(x[..., 0], 0, 1) * 255).astype(np.uint8)
            rgb = np.stack([g, g, g], axis=-1)
        else:
            rgb = (np.clip(x[..., :3], 0, 1) * 255).astype(np.uint8)

        out = rgb.copy()
        gt = msk_c01.astype(bool)
        pr = pred01.astype(bool)

        out[gt, 0] = 255
        out[gt, 1] = 255
        out[gt, 2] = 0

        out[pr, 0] = 0
        out[pr, 1] = 255
        out[pr, 2] = 255

        tiff.imwrite(str(out_dir / f"preview_{i:02d}.png"), out)


def parse_args(cfg: TrainConfig):
    import argparse

    p = argparse.ArgumentParser(description="Train U-Net (TF/Keras) on your annotated TIFFs.")
    p.add_argument("--dataset", default=cfg.dataset, choices=["2d_time", "2d_wga_dapi"])
    p.add_argument("--patch", type=int, default=cfg.patch)
    p.add_argument("--batch", type=int, default=cfg.batch)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)

    p.add_argument("--steps_per_epoch", type=int, default=cfg.steps_per_epoch)
    p.add_argument("--val_steps", type=int, default=cfg.val_steps)

    p.add_argument("--fg_fraction", type=float, default=cfg.fg_fraction)
    p.add_argument("--min_fg_ratio", type=float, default=cfg.min_fg_ratio)

    p.add_argument("--val_split", type=float, default=cfg.val_split)
    p.add_argument("--seed", type=int, default=cfg.seed)

    p.add_argument("--normalize", default=cfg.normalize, choices=["scale_uint16", "percentile"])

    p.add_argument("--base_filters", type=int, default=cfg.base_filters)
    p.add_argument("--dropout", type=float, default=cfg.dropout)

    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)

    cfg.data_root = cfg.repo_root / "results" / "training_files" / "U-net" / cfg.dataset
    cfg.out_root = cfg.repo_root / "results" / "training_files" / "U-net" / "_trained_models" / cfg.dataset
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    return cfg


def main():
    cfg = parse_args(TrainConfig())

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    pairs = list_pairs(cfg.data_root)
    if len(pairs) < 2:
        raise RuntimeError(
            f"Need at least 2 annotated samples in {cfg.data_root}. "
            f"Found {len(pairs)}."
        )

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds, C = make_dataset(train_pairs, cfg, training=True)
    val_ds, _ = make_dataset(val_pairs, cfg, training=False)

    model = build_unet(
        input_shape=(cfg.patch, cfg.patch, C),
        base_filters=cfg.base_filters,
        dropout=cfg.dropout,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef],
    )

    ckpt_path = cfg.out_root / "unet_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("\n=== DATA ===")
    print(f"Dataset: {cfg.dataset}")
    print(f"Annotated samples: {len(pairs)} (train {len(train_pairs)} / val {len(val_pairs)})")
    print(f"Patch: {cfg.patch}  Batch: {cfg.batch}")
    print(f"Foreground sampling: {cfg.fg_fraction*100:.0f}% fg, min_fg_ratio={cfg.min_fg_ratio}")
    print(f"Normalize: {cfg.normalize}")
    print(f"Output: {cfg.out_root}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    
    final_path = cfg.out_root / "unet_final.keras"
    model.save(str(final_path))

 
    hist_path = cfg.out_root / "history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    preview_dir = cfg.out_root / "previews"
    save_prediction_previews(model, val_pairs, cfg, preview_dir, n=6)

    print("\nSaved:")
    print(f"  Best model:  {ckpt_path}")
    print(f"  Final model: {final_path}")
    print(f"  History:     {hist_path}")
    print(f"  Previews:    {preview_dir}")


if __name__ == "__main__":
    main()