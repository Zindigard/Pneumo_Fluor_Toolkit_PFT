from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tifffile as tiff
import tensorflow as tf
import matplotlib.pyplot as plt

from PFT.core_prog_parts.decoder_omezar import load_ome_zarr
from PFT.core_prog_parts.notch_filter import _ensure_cyx, _to_numpy


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "src" / "PFT").exists():
            return p
    return Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    repo_root: Path = find_repo_root(Path(__file__).resolve())

    dataset: str = "2d_time"
    mask_root: Path | None = None
    filtered_root: Path | None = None
    models_root: Path | None = None
    run_root: Path | None = None

    patch: int = 256
    batch: int = 8
    epochs: int = 50
    lr: float = 1e-3

    steps_per_epoch: int = 300
    val_steps: int = 60

    fg_fraction: float = 0.8
    fg_min_ratio: float = 0.60
    bg_max_ratio: float = 0.10
    max_tries: int = 100

    val_split: float = 0.2
    seed: int = 1337

    normalize: str = "percentile"

    base_filters: int = 8
    dropout: float = 0.0

    preview_only: bool = False
    preview_n: int = 3


def yes_no_prompt(text, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        s = input(f"{text} {suffix} ").strip().lower()
        if s == "":
            return default
        if s in {"y", "yes"}:
            return True
        if s in {"n", "no"}:
            return False
        print("Please answer y or n.")


def choose_dataset_terminal(default="2d_time"):
    datasets = ["2d_time", "2d_wga_dapi"]
    print("\nChoose dataset to train:")
    for i, ds in enumerate(datasets, 1):
        tag = " (default)" if ds == default else ""
        print(f"  {i}) {ds}{tag}")

    s = input(f"Choose number or press Enter for [{default}]: ").strip()
    if s == "":
        return default
    try:
        k = int(s)
        if 1 <= k <= len(datasets):
            return datasets[k - 1]
    except ValueError:
        pass

    print(f"Invalid choice, using default: {default}")
    return default


def resolve_filtered_zarr(filtered_root: Path, dataset: str, sample: str) -> Path:
    cand = filtered_root / dataset / sample / "image.ome.zarr"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"Filtered OME-Zarr not found for sample '{sample}': {cand}. "
        f"Prepare thresholded/filtered inputs first."
    )


def list_pairs(mask_root: Path, filtered_root: Path, dataset: str):
    pairs: list[tuple[Path, Path]] = []
    ds_mask_root = mask_root / dataset
    if not ds_mask_root.exists():
        raise RuntimeError(f"Mask folder does not exist: {ds_mask_root}")

    for sample_dir in sorted(ds_mask_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        mask_path = sample_dir / "mask.tif"
        if not mask_path.exists():
            continue

        try:
            img_path = resolve_filtered_zarr(filtered_root, dataset, sample_dir.name)
        except FileNotFoundError:
            continue

        pairs.append((img_path, mask_path))
    return pairs


def _extract_display_plane(x: np.ndarray, axes: str, channel_index: int) -> np.ndarray:
    if "c" in axes:
        plane = np.take(x, indices=channel_index, axis=axes.index("c"))
    else:
        plane = x
    if "t" in axes and plane.ndim == 3:
        plane = plane[0]
    return np.asarray(plane, dtype=np.float32)


def read_image_mask_numpy(img_path: str, mask_path: str):
    arr, axes = load_ome_zarr(Path(img_path), level=0, as_numpy=False)
    img = _to_numpy(arr)
    img, axes = _ensure_cyx(img, axes)

    blue = _extract_display_plane(img, axes, 0)
    planes = [blue]

    if "c" in axes and img.shape[axes.index("c")] > 1:
        try:
            green = _extract_display_plane(img, axes, 1)
            planes.append(green)
        except Exception:
            pass

    img_hwc = np.stack(planes, axis=-1) if len(planes) > 1 else blue[..., None]

    msk = tiff.imread(mask_path)
    msk = (msk > 0).astype(np.uint8)

    if img_hwc.shape[:2] != msk.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch for {img_path} and {mask_path}: "
            f"{img_hwc.shape[:2]} vs {msk.shape[:2]}"
        )

    return img_hwc, msk


def normalize_crop_numpy(x: np.ndarray, mode: str):
    x = x.astype(np.float32)

    if mode == "scale_uint16":
        x = x / 65535.0
        return np.clip(x, 0.0, 1.0)

    if mode == "percentile":
        out = np.empty_like(x, dtype=np.float32)
        for c in range(x.shape[-1]):
            xc = x[..., c]
            p1 = np.percentile(xc, 1)
            p2 = np.percentile(xc, 99.8)
            out[..., c] = (xc - p1) / (p2 - p1 + 1e-8)
        return np.clip(out, 0.0, 1.0)

    raise ValueError(f"Unknown normalize mode: {mode}")


def image_to_rgb_uint8(img: np.ndarray, normalize_mode="percentile"):
    if img.ndim == 2:
        img = img[..., None]

    x = normalize_crop_numpy(img, normalize_mode)

    if x.shape[-1] == 1:
        b = (np.clip(x[..., 0], 0, 1) * 255).astype(np.uint8)
        z = np.zeros_like(b, dtype=np.uint8)
        return np.stack([z, z, b], axis=-1)

    if x.shape[-1] == 2:
        b = (np.clip(x[..., 0], 0, 1) * 255).astype(np.uint8)
        g = (np.clip(x[..., 1], 0, 1) * 255).astype(np.uint8)
        r = np.zeros_like(g, dtype=np.uint8)
        return np.dstack([r, g, b])

    rgb = (np.clip(x[..., :3], 0, 1) * 255).astype(np.uint8)
    return rgb


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
    H, W = msk.shape
    P = cfg.patch

    if H < P or W < P:
        raise ValueError(
            f"Patch size {P} is larger than image size {(H, W)}. "
            f"Reduce --patch or use larger images."
        )

    want_fg = (random.random() < cfg.fg_fraction)
    fg_coords = np.argwhere(msk > 0)

    best_img = None
    best_msk = None
    best_score = None

    for _ in range(cfg.max_tries):
        if want_fg and len(fg_coords) > 0:
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
            if fg_ratio >= cfg.fg_min_ratio:
                return img_c, msk_c
            score = fg_ratio
            if best_score is None or score > best_score:
                best_score = score
                best_img = img_c
                best_msk = msk_c
        else:
            if fg_ratio <= cfg.bg_max_ratio:
                return img_c, msk_c
            score = fg_ratio
            if best_score is None or score < best_score:
                best_score = score
                best_img = img_c
                best_msk = msk_c

    if best_img is not None:
        return best_img, best_msk

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


def build_unet(input_shape, base_filters=8, dropout=0.0):
    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(inputs, base_filters, dropout=dropout)
    p1 = tf.keras.layers.MaxPool2D()(c1)

    c2 = conv_block(p1, base_filters * 2, dropout=dropout)
    p2 = tf.keras.layers.MaxPool2D()(c2)

    c3 = conv_block(p2, base_filters * 4, dropout=dropout)
    p3 = tf.keras.layers.MaxPool2D()(c3)

    c4 = conv_block(p3, base_filters * 8, dropout=dropout)
    p4 = tf.keras.layers.MaxPool2D()(c4)

    bn = conv_block(p4, base_filters * 16, dropout=dropout)

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


def print_dataset_summary(pairs, cfg: TrainConfig, title="DATASET"):
    print(f"\n=== {title} ===")
    print(f"Dataset: {cfg.dataset}")
    print(f"Mask root: {cfg.mask_root}")
    print(f"Filtered root: {cfg.filtered_root}")
    print(f"Number of annotated mask / filtered-image pairs: {len(pairs)}")
    print("Samples:")
    for i, (img_path, mask_path) in enumerate(pairs, 1):
        img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
        print(
            f"  [{i:02d}] {mask_path.parent.name} | zarr: {img_path} | "
            f"image shape: {img.shape} | mask shape: {msk.shape}"
        )

    first_img, _ = read_image_mask_numpy(str(pairs[0][0]), str(pairs[0][1]))
    C = first_img.shape[-1]

    train_patches_per_epoch = cfg.steps_per_epoch * cfg.batch
    val_patches_per_epoch = cfg.val_steps * cfg.batch
    total_train_patches = train_patches_per_epoch * cfg.epochs
    total_val_patches = val_patches_per_epoch * cfg.epochs

    print("\nPatch / batch summary:")
    print(f"  Patch image shape: ({cfg.patch}, {cfg.patch}, {C})")
    print(f"  Patch mask shape: ({cfg.patch}, {cfg.patch}, 1)")
    print(f"  Batch size: {cfg.batch}")
    print(f"  Train patches per epoch: {train_patches_per_epoch}")
    print(f"  Val patches per epoch:   {val_patches_per_epoch}")
    print(f"  Total train patches over {cfg.epochs} epochs: {total_train_patches}")
    print(f"  Total val patches over {cfg.epochs} epochs:   {total_val_patches}")

    print("\nPatch sampling strategy:")
    print(f"  {cfg.fg_fraction * 100:.0f}% foreground patches (fg >= {cfg.fg_min_ratio:.2f})")
    print(f"  {(1 - cfg.fg_fraction) * 100:.0f}% background patches (fg <= {cfg.bg_max_ratio:.2f})")


def print_split_summary(train_pairs, val_pairs):
    print("\n=== TRAIN / VAL SPLIT ===")
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images:   {len(val_pairs)}")

    print("\nTrain samples:")
    for i, (img_path, mask_path) in enumerate(train_pairs, 1):
        img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
        print(f"  [T{i:02d}] {mask_path.parent.name} | zarr: {img_path.name} | image shape: {img.shape} | mask shape: {msk.shape}")

    print("\nValidation samples:")
    for i, (img_path, mask_path) in enumerate(val_pairs, 1):
        img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
        print(f"  [V{i:02d}] {mask_path.parent.name} | zarr: {img_path.name} | image shape: {img.shape} | mask shape: {msk.shape}")


def print_random_patch_examples(pairs, cfg: TrainConfig, n=5):
    print("\n=== RANDOM PATCH CHECK ===")
    for i in range(n):
        img_path, mask_path = random.choice(pairs)
        img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
        img_c, msk_c = sample_patch_numpy(img, msk, cfg)

        fg_ratio = float(msk_c.mean())
        bg_ratio = 1.0 - fg_ratio

        print(
            f"  Patch {i+1:02d} from {mask_path.parent.name} | zarr: {img_path.name} | "
            f"image shape: {img.shape} | mask shape: {msk.shape} | "
            f"patch image shape: {img_c.shape} | patch mask shape: {msk_c.shape} | "
            f"fg ratio: {fg_ratio:.3f} | bg ratio: {bg_ratio:.3f}"
        )


def save_run_summary(cfg: TrainConfig, pairs, train_pairs, val_pairs, C, out_path: Path):
    train_patches_per_epoch = cfg.steps_per_epoch * cfg.batch
    val_patches_per_epoch = cfg.val_steps * cfg.batch

    lines = []
    lines.append(f"Dataset: {cfg.dataset}")
    lines.append(f"Mask root: {cfg.mask_root}")
    lines.append(f"Filtered root: {cfg.filtered_root}")
    lines.append(f"Total annotated pairs: {len(pairs)}")
    lines.append(f"Train images: {len(train_pairs)}")
    lines.append(f"Val images: {len(val_pairs)}")
    lines.append(f"Patch image shape: ({cfg.patch}, {cfg.patch}, {C})")
    lines.append(f"Patch mask shape: ({cfg.patch}, {cfg.patch}, 1)")
    lines.append(f"Batch size: {cfg.batch}")
    lines.append(f"Epochs: {cfg.epochs}")
    lines.append(f"Steps per epoch: {cfg.steps_per_epoch}")
    lines.append(f"Val steps: {cfg.val_steps}")
    lines.append(f"Train patches per epoch: {train_patches_per_epoch}")
    lines.append(f"Val patches per epoch: {val_patches_per_epoch}")
    lines.append(f"Total training patches: {train_patches_per_epoch * cfg.epochs}")
    lines.append(f"Total validation patches: {val_patches_per_epoch * cfg.epochs}")
    lines.append(f"Normalization: {cfg.normalize}")
    lines.append(f"Foreground fraction: {cfg.fg_fraction}")
    lines.append(f"Foreground min ratio: {cfg.fg_min_ratio}")
    lines.append(f"Background max ratio: {cfg.bg_max_ratio}")
    lines.append(f"Base filters: {cfg.base_filters}")
    lines.append("")
    lines.append("All samples:")

    for i, (img_path, mask_path) in enumerate(pairs, 1):
        img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
        lines.append(
            f"  [{i:02d}] {mask_path.parent.name} | zarr: {img_path} | "
            f"image shape: {img.shape} | mask shape: {msk.shape}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_training_curves(history, out_png: Path, dataset_name: str):
    hist = history.history
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(15, 4.8), dpi=220)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, hist.get("loss", []), linewidth=2, label="Training loss")
    ax1.plot(epochs, hist.get("val_loss", []), linewidth=2, label="Validation loss")
    ax1.set_title(f"{dataset_name}: Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, hist.get("dice_coef", []), linewidth=2, label="Training Dice")
    ax2.plot(epochs, hist.get("val_dice_coef", []), linewidth=2, label="Validation Dice")
    ax2.set_title(f"{dataset_name}: Dice coefficient")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(epochs, hist.get("iou_coef", []), linewidth=2, label="Training IoU")
    ax3.plot(epochs, hist.get("val_iou_coef", []), linewidth=2, label="Validation IoU")
    ax3.set_title(f"{dataset_name}: Intersection over Union")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("IoU")
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(frameon=True)

    fig.suptitle(f"U-Net training history ({dataset_name})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_png.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def save_prediction_previews(model, pairs, cfg: TrainConfig, out_dir: Path, n=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.sample(pairs, k=min(n, len(pairs)))

    for i, (img_path, mask_path) in enumerate(chosen, 1):
        img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
        img_c, msk_c = sample_patch_numpy(img, msk, cfg)
        img_c_norm = normalize_crop_numpy(img_c, cfg.normalize)
        msk_c01 = (msk_c > 0).astype(np.uint8)

        pred = model.predict(img_c_norm[None, ...], verbose=0)[0, ..., 0]
        pred01 = (pred > 0.5).astype(np.uint8)

        rgb = image_to_rgb_uint8(img_c, normalize_mode=cfg.normalize)

        gt_overlay = rgb.copy()
        gt_overlay[msk_c01.astype(bool), 0] = 255
        gt_overlay[msk_c01.astype(bool), 1] = 255
        gt_overlay[msk_c01.astype(bool), 2] = 0

        pred_overlay = rgb.copy()
        pred_overlay[pred01.astype(bool), 0] = 255
        pred_overlay[pred01.astype(bool), 1] = 0
        pred_overlay[pred01.astype(bool), 2] = 255

        fig = plt.figure(figsize=(12, 4), dpi=180)

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(rgb)
        ax1.set_title("Patch")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(gt_overlay)
        ax2.set_title("Ground truth overlay")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(pred_overlay)
        ax3.set_title("Prediction overlay")
        ax3.axis("off")

        plt.tight_layout()
        fig.savefig(str(out_dir / f"preview_{i:02d}.png"), bbox_inches="tight")
        plt.close(fig)


def show_random_image_mask_patch(pairs, cfg: TrainConfig, save_path: Path | None = None):
    img_path, mask_path = random.choice(pairs)
    img, msk = read_image_mask_numpy(str(img_path), str(mask_path))
    img_c, msk_c = sample_patch_numpy(img, msk, cfg)

    full_rgb = image_to_rgb_uint8(img, normalize_mode=cfg.normalize)
    full_overlay = full_rgb.copy()
    full_overlay[msk.astype(bool), 0] = 255
    full_overlay[msk.astype(bool), 1] = 255
    full_overlay[msk.astype(bool), 2] = 0

    patch_rgb = image_to_rgb_uint8(img_c, normalize_mode=cfg.normalize)
    patch_overlay = patch_rgb.copy()
    patch_overlay[msk_c.astype(bool), 0] = 255
    patch_overlay[msk_c.astype(bool), 1] = 255
    patch_overlay[msk_c.astype(bool), 2] = 0

    fig = plt.figure(figsize=(16, 5), dpi=180)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(full_rgb)
    ax1.set_title(f"Random image\n{mask_path.parent.name}")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(full_overlay)
    ax2.set_title("Filtered image + mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(patch_overlay)
    ax3.set_title(f"Random patch\nfg ratio = {msk_c.mean():.2%}")
    ax3.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")

    plt.show()
    plt.close(fig)


def parse_args(cfg: TrainConfig):
    import argparse

    p = argparse.ArgumentParser(
        description="Train U-Net on filtered thresholded OME-Zarr images with masks from labeling."
    )
    p.add_argument("--dataset", default=None, choices=["2d_time", "2d_wga_dapi"])
    p.add_argument("--patch", type=int, default=cfg.patch)
    p.add_argument("--batch", type=int, default=cfg.batch)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--steps_per_epoch", type=int, default=cfg.steps_per_epoch)
    p.add_argument("--val_steps", type=int, default=cfg.val_steps)
    p.add_argument("--fg_fraction", type=float, default=cfg.fg_fraction)
    p.add_argument("--fg_min_ratio", type=float, default=cfg.fg_min_ratio)
    p.add_argument("--bg_max_ratio", type=float, default=cfg.bg_max_ratio)
    p.add_argument("--val_split", type=float, default=cfg.val_split)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--normalize", default=cfg.normalize, choices=["scale_uint16", "percentile"])
    p.add_argument("--base_filters", type=int, default=cfg.base_filters)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--preview_only", action="store_true")
    p.add_argument("--preview_n", type=int, default=cfg.preview_n)

    args = p.parse_args()

    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    if args.dataset is None:
        cfg.dataset = choose_dataset_terminal(default=cfg.dataset)

    cfg.mask_root = cfg.repo_root / "results" / "training_files" / "U-net"
    cfg.filtered_root = cfg.repo_root / "results" / "img" / "filtered"
    cfg.models_root = cfg.repo_root / "models"
    cfg.models_root.mkdir(parents=True, exist_ok=True)

    cfg.run_root = cfg.models_root / f"u_net_{cfg.dataset}"
    cfg.run_root.mkdir(parents=True, exist_ok=True)

    return cfg


def main():
    cfg = parse_args(TrainConfig())

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    pairs = list_pairs(cfg.mask_root, cfg.filtered_root, cfg.dataset)
    if len(pairs) < 2:
        raise RuntimeError(
            f"Need at least 2 annotated samples with masks in {cfg.mask_root / cfg.dataset} "
            f"and filtered OME-Zarr images in {cfg.filtered_root / cfg.dataset}. Found {len(pairs)}."
        )

    preview_dir = cfg.run_root / "preview_checks"
    preview_dir.mkdir(parents=True, exist_ok=True)

    print_dataset_summary(pairs, cfg, title="FULL DATASET")
    print_random_patch_examples(pairs, cfg, n=5)

    if cfg.preview_only:
        for i in range(cfg.preview_n):
            save_path = preview_dir / f"preview_check_{i+1:02d}.png"
            show_random_image_mask_patch(pairs, cfg, save_path=save_path)
        print("\nPreview-only mode finished.")
        print(f"Saved preview checks to: {preview_dir}")
        return

    do_preview = yes_no_prompt(
        "Do you want to inspect random filtered image + mask + random patch before training?",
        default=True,
    )
    if do_preview:
        save_path = preview_dir / "preview_check_before_training.png"
        show_random_image_mask_patch(pairs, cfg, save_path=save_path)
        print(f"Saved preview check to: {save_path}")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    if len(train_pairs) < 1:
        raise RuntimeError("Training split is empty. Add more annotated samples or reduce val_split.")

    print_split_summary(train_pairs, val_pairs)

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

    best_model_path = cfg.run_root / f"u_net_{cfg.dataset}_best.keras"
    final_model_path = cfg.run_root / f"u_net_{cfg.dataset}_final.keras"
    history_path = cfg.run_root / f"u_net_{cfg.dataset}_history.json"
    curves_path = cfg.run_root / f"u_net_{cfg.dataset}_training_curves.png"
    summary_txt = cfg.run_root / f"u_net_{cfg.dataset}_run_summary.txt"
    previews_out = cfg.run_root / "prediction_previews"

    save_run_summary(cfg, pairs, train_pairs, val_pairs, C, summary_txt)
    print(f"\nRun summary saved to: {summary_txt}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
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

    print("\n=== TRAINING SETTINGS ===")
    print(f"Dataset:              {cfg.dataset}")
    print(f"Mask root:            {cfg.mask_root / cfg.dataset}")
    print(f"Filtered root:        {cfg.filtered_root / cfg.dataset}")
    print(f"Patch size:           {cfg.patch}")
    print(f"Patch image shape:    ({cfg.patch}, {cfg.patch}, {C})")
    print(f"Patch mask shape:     ({cfg.patch}, {cfg.patch}, 1)")
    print(f"Batch size:           {cfg.batch}")
    print(f"Epochs:               {cfg.epochs}")
    print(f"Steps per epoch:      {cfg.steps_per_epoch}")
    print(f"Validation steps:     {cfg.val_steps}")
    print(f"Learning rate:        {cfg.lr}")
    print(f"Normalization:        {cfg.normalize}")
    print(f"Foreground fraction:  {cfg.fg_fraction:.2f}")
    print(f"Foreground min ratio: {cfg.fg_min_ratio:.2f}")
    print(f"Background max ratio: {cfg.bg_max_ratio:.2f}")
    print(f"Base filters:         {cfg.base_filters}")
    print(f"Train images:         {len(train_pairs)}")
    print(f"Validation images:    {len(val_pairs)}")
    print(f"Best model path:      {best_model_path}")
    print(f"Final model path:     {final_model_path}")
    print(f"Curves path:          {curves_path}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(final_model_path))

    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    save_training_curves(history, curves_path, cfg.dataset)
    save_prediction_previews(model, val_pairs, cfg, previews_out, n=6)

    print("\nSaved:")
    print(f"  Best model:       {best_model_path}")
    print(f"  Final model:      {final_model_path}")
    print(f"  History JSON:     {history_path}")
    print(f"  Training curves:  {curves_path}")
    print(f"  Training curves:  {curves_path.with_suffix('.pdf')}")
    print(f"  Run summary:      {summary_txt}")
    print(f"  Prediction views: {previews_out}")


if __name__ == "__main__":
    main()