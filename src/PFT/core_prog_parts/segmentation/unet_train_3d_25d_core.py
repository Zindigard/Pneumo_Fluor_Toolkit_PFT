"""Training and evaluation for the per-volume merged-RGB 2.5D U-Net.

Each source stack has one manually selected target slice in the shared mapping.
For a target Zn, the model input is the wavelength-mapped merged-RGB context
Z(n-1)/Zn/Z(n+1), concatenated as nine channels, and the target is one binary
foreground mask at Zn. Unannotated stacks are never treated as background.

Manual masks are stored as
``results/training_files/U-net/3d_25d/<experiment>/<sample>/zNNN_mask.tif``.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile as tiff
import zarr

from PFT.core_prog_parts.common_paths import ensure_dir, find_project_root
from PFT.core_prog_parts.decoder_omezar import extract_ome_zarr_meta_for_compare
from PFT.core_prog_parts.denoising.metadata_3d import resolve_channel_optics
from PFT.core_prog_parts.denoising.validation_3d import (
    TARGET_SLICE_BY_VOLUME_KEY,
    annotation_sample_dir,
    find_slice_mask,
    target_slice_for_volume,
    volume_key,
)
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (
    bce_dice_loss,
    build_unet,
    dice_coef,
    iou_coef,
    save_training_curves,
)


MODEL_CONTRACT = "pft_3d_25d_merged_rgb_per_volume_target_v1"


@dataclass
class UNet25DTrainConfig:
    """Configuration for 2.5D training and held-out annotation evaluation."""

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
    z_radius: int = 1
    channels: tuple[int, ...] | None = None
    threshold: float = 0.5
    predict_batch_size: int = 8

    fg_fraction: float = 0.75
    fg_min_ratio: float = 0.02
    bg_max_ratio: float = 0.01
    max_tries: int = 100


@dataclass(frozen=True)
class AnnotatedSlice:
    """One OME-Zarr, target Z index, manual mask, and RGB source mapping."""

    image_zarr: Path
    mask_tif: Path
    sample: str
    slice_1based: int
    z_index: int
    rgb_source_channels: tuple[int, int, int]


def _default_image_root(project_root: Path) -> Path:
    return project_root / "results" / "img" / "3d_data"


def _default_mask_root(project_root: Path) -> Path:
    return project_root / "results" / "training_files" / "U-net" / "3d_25d"


def _default_model_root(project_root: Path) -> Path:
    return project_root / "models" / "u_net_3d_25d"


def _find_image_zarrs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("image.ome.zarr") if path.is_dir())


def open_3d_image_czyx(zarr_path: Path, *, level: int = 0) -> zarr.Array:
    """Open one CZYX OME-Zarr level lazily without loading the volume."""
    zarr_path = Path(zarr_path)
    meta = extract_ome_zarr_meta_for_compare(zarr_path, level=level)
    axes = str(meta.get("axes") or "").lower()
    if axes != "czyx":
        raise ValueError(
            f"The memory-efficient 2.5D workflow requires stored CZYX axes; "
            f"received {axes!r} for {zarr_path}. Run check_3d_inputs.py first."
        )
    root = zarr.open_group(str(zarr_path), mode="r")
    array = root[str(meta["array_path"])]
    if array.ndim != 4:
        raise ValueError(f"Expected a four-dimensional CZYX array, got {array.shape}")
    return array


def load_3d_image_czyx(zarr_path: Path, *, level: int = 0) -> np.ndarray:
    """Load one complete CZYX image as float32. Prefer the lazy opener for training."""
    array = open_3d_image_czyx(zarr_path, level=level)
    output = np.asarray(array, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError(f"Image contains non-finite values: {zarr_path}")
    return output


def read_binary_slice_mask(mask_path: Path, expected_yx: tuple[int, int]) -> np.ndarray:
    """Read and strictly validate one binary target-slice mask."""
    mask = np.asarray(tiff.imread(str(mask_path)))
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D YX, got {mask.shape}: {mask_path}")
    if mask.shape != expected_yx:
        raise ValueError(f"Mask/image mismatch: mask={mask.shape}; expected={expected_yx}: {mask_path}")
    values = np.unique(mask)
    if not np.all(np.isin(values, (0, 1, 255))):
        raise ValueError(f"Mask must contain only 0/1 or 0/255, got {values[:20]}: {mask_path}")
    output = (mask > 0).astype(np.uint8)
    if not np.any(output) or np.all(output):
        raise ValueError(f"Mask must contain both foreground and background: {mask_path}")
    return output


RGB_REFERENCE_WAVELENGTHS_NM: dict[str, float] = {
    "red": 561.0,
    "green": 488.0,
    "blue": 405.0,
}


def resolve_rgb_source_channels(image_zarr: Path, *, level: int = 0) -> tuple[int, int, int]:
    """Return source channel indices in RGB order using wavelength metadata.

    The returned tuple is ``(red_source, green_source, blue_source)``. Every
    required colour must be represented exactly once. A tolerance of 35 nm is
    used only to accommodate minor metadata variation around 405/488/561 nm.
    """
    optics = resolve_channel_optics(image_zarr, level=level)
    assignments: dict[str, int] = {}
    for item in optics:
        colour, target = min(
            RGB_REFERENCE_WAVELENGTHS_NM.items(),
            key=lambda pair: abs(float(item.wavelength_nm) - pair[1]),
        )
        difference = abs(float(item.wavelength_nm) - target)
        if difference > 35.0:
            raise ValueError(
                f"Channel {item.index} ({item.name}) at {item.wavelength_nm:g} nm cannot be "
                "mapped safely to the 405/488/561 nm RGB convention."
            )
        if colour in assignments:
            raise ValueError(
                f"Multiple source channels map to {colour}: existing C{assignments[colour]}, "
                f"new C{item.index}."
            )
        assignments[colour] = int(item.index)
    missing = [colour for colour in ("red", "green", "blue") if colour not in assignments]
    if missing:
        raise ValueError(f"Missing wavelength channel(s) for merged RGB input: {missing}")
    return assignments["red"], assignments["green"], assignments["blue"]


def list_annotated_slices(cfg: UNet25DTrainConfig) -> list[AnnotatedSlice]:
    """Find all required sparse annotations without inventing unlabelled targets."""
    image_root = Path(cfg.image_root or _default_image_root(cfg.project_root))
    mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root))
    if not image_root.is_dir():
        raise FileNotFoundError(f"3D image root does not exist: {image_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"3D mask root does not exist: {mask_root}")

    entries: list[AnnotatedSlice] = []
    missing: list[str] = []
    selected_samples = 0
    for image_zarr in _find_image_zarrs(image_root):
        sample = volume_key(image_zarr, image_root)
        sample_mask_dir = annotation_sample_dir(mask_root, image_zarr, image_root)

        # A sample is selected for training only by creating its mask folder.
        # Unannotated stacks remain available for later inference and are ignored
        # here rather than being treated as incomplete training data.
        if not sample_mask_dir.is_dir():
            continue
        selected_samples += 1

        # Read only array geometry here; image voxels remain lazy.
        array = open_3d_image_czyx(image_zarr, level=cfg.level)
        if int(array.shape[0]) != 3:
            raise ValueError(
                f"Merged RGB training requires exactly three fluorescence channels; "
                f"received C={array.shape[0]} for {image_zarr}."
            )
        rgb_source_channels = resolve_rgb_source_channels(image_zarr, level=cfg.level)
        z_count = int(array.shape[1])
        slice_number = target_slice_for_volume(image_zarr, image_root)
        if slice_number < 2 or slice_number > z_count - 1:
            missing.append(
                f"{sample}: configured Z{slice_number} has no complete Z-1/Z/Z+1 context for Z={z_count}"
            )
            continue
        mask_path = find_slice_mask(sample_mask_dir, slice_number)
        if mask_path is None:
            missing.append(
                f"{sample}: missing configured target mask "
                f"{sample_mask_dir / f'z{slice_number:03d}_mask.tif'}"
            )
            continue
        entries.append(
            AnnotatedSlice(
                image_zarr,
                mask_path,
                sample,
                slice_number,
                slice_number - 1,
                rgb_source_channels,
            )
        )
    if selected_samples == 0:
        raise RuntimeError(
            f"No selected training samples were found under {mask_root}. "
            "Create one mask folder per selected stack."
        )
    if missing:
        raise FileNotFoundError(
            "Selected 2.5D training samples have incomplete annotations:\n"
            + "\n".join(f"- {item}" for item in missing)
        )
    if not entries:
        raise RuntimeError("No annotated 2.5D target slices were found")
    return entries


def split_annotated_slices(
    entries: Sequence[AnnotatedSlice], cfg: UNet25DTrainConfig
) -> tuple[list[AnnotatedSlice], list[AnnotatedSlice], str]:
    """Split by complete sample where possible, otherwise by spaced target slices."""
    rng = random.Random(cfg.seed)
    samples = sorted({entry.sample for entry in entries})
    if len(samples) >= 2:
        rng.shuffle(samples)
        validation_count = max(1, int(round(len(samples) * cfg.val_split)))
        validation_samples = set(samples[:validation_count])
        train = [entry for entry in entries if entry.sample not in validation_samples]
        validation = [entry for entry in entries if entry.sample in validation_samples]
        return train, validation, "complete-volume split"

    shuffled = list(entries)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * cfg.val_split)))
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    if not train:
        train = shuffled
    return train, validation, "single-volume slice split; biological independence is limited"


def _context_z_indices(img_czyx: Any, z: int, cfg: UNet25DTrainConfig) -> tuple[int, int, int]:
    """Return the complete Z-1/Z/Z+1 indices for one middle slice."""
    if cfg.z_radius != 1:
        raise ValueError("Merged-RGB 2.5D input requires z_radius=1")
    z_total = int(img_czyx.shape[1])
    if not 1 <= int(z) <= z_total - 2:
        raise IndexError(
            f"Middle Z index {z} does not have complete Z-1/Z/Z+1 context for Z={z_total}."
        )
    return int(z - 1), int(z), int(z + 1)


def _source_plane_keys(
    img_czyx: Any,
    z: int,
    cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
) -> list[tuple[int, int]]:
    """Return source plane keys in Z-major then RGB order."""
    if len(rgb_source_channels) != 3 or len(set(rgb_source_channels)) != 3:
        raise ValueError(f"Invalid RGB source-channel mapping: {rgb_source_channels}")
    if any(not 0 <= channel < int(img_czyx.shape[0]) for channel in rgb_source_channels):
        raise IndexError(
            f"RGB source mapping {rgb_source_channels} is incompatible with C={img_czyx.shape[0]}"
        )
    return [
        (source_channel, z_position)
        for z_position in _context_z_indices(img_czyx, z, cfg)
        for source_channel in rgb_source_channels
    ]


def make_25d_input_slice(
    img_czyx: Any,
    z: int,
    cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
) -> np.ndarray:
    """Read the complete Z-1/Z/Z+1 merged-RGB context in HWC order."""
    planes = [
        np.asarray(img_czyx[channel, z_position], dtype=np.float32)
        for channel, z_position in _source_plane_keys(
            img_czyx, z, cfg, rgb_source_channels
        )
    ]
    output = np.stack(planes, axis=-1).astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite values found in the merged-RGB 2.5D input for Z index {z}")
    return output


def make_25d_input_patch(
    img_czyx: Any,
    z: int,
    y0: int,
    x0: int,
    height: int,
    width: int,
    cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
) -> np.ndarray:
    """Read one merged-RGB 2.5D HWC patch from the OME-Zarr array."""
    planes = [
        np.asarray(
            img_czyx[channel, z_position, y0:y0 + height, x0:x0 + width],
            dtype=np.float32,
        )
        for channel, z_position in _source_plane_keys(
            img_czyx, z, cfg, rgb_source_channels
        )
    ]
    output = np.stack(planes, axis=-1).astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise ValueError(f"Non-finite values found in the merged-RGB 2.5D patch for Z index {z}")
    return output


def _normalization_limits(
    img_czyx: Any,
    z: int,
    cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
    cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Calculate context-wise limits matching the annotation RGB composites.

    For each wavelength channel, percentiles are calculated jointly across the
    three context planes. The same limits are then applied to that colour in
    Z-1, Z, and Z+1. This keeps training, validation, and annotation display
    construction consistent.
    """
    if cfg.normalize == "scale_uint16":
        return [(0.0, 65535.0)] * 9
    if cfg.normalize != "percentile":
        raise ValueError(f"Unknown normalization mode: {cfg.normalize}")
    cache = cache if cache is not None else {}
    z_positions = _context_z_indices(img_czyx, z, cfg)
    channel_limits: dict[int, tuple[float, float]] = {}
    for source_channel in rgb_source_channels:
        key = (int(source_channel), int(z))
        if key not in cache:
            context = np.asarray(
                img_czyx[source_channel, z_positions[0] : z_positions[-1] + 1],
                dtype=np.float32,
            )
            if not np.isfinite(context).all():
                raise ValueError(
                    f"Non-finite values in source channel {source_channel}, "
                    f"context Z{z_positions[0] + 1}..Z{z_positions[-1] + 1}"
                )
            low = float(np.percentile(context, 1.0))
            high = float(np.percentile(context, 99.8))
            cache[key] = (low, high)
        channel_limits[source_channel] = cache[key]
    return [
        channel_limits[source_channel]
        for _z_position in z_positions
        for source_channel in rgb_source_channels
    ]


def _normalize_with_limits(
    image_hwc: np.ndarray,
    limits: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Apply context-wise RGB normalization to a patch or complete input."""
    image = np.asarray(image_hwc, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != len(limits):
        raise ValueError(f"Normalization geometry mismatch: shape={image.shape}, limits={len(limits)}")
    output = np.empty_like(image, dtype=np.float32)
    for index, (low, high) in enumerate(limits):
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            output[..., index] = 0.0
        else:
            output[..., index] = np.clip(
                (image[..., index] - np.float32(low)) / np.float32(high - low + 1e-8),
                0.0,
                1.0,
            )
    return output


def make_merged_rgb_context_slice(
    img_czyx: Any,
    z: int,
    cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
    *,
    normalization_cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return normalized Z-1/Z/Z+1 RGB images as ``(3, Y, X, 3)``."""
    raw = make_25d_input_slice(img_czyx, z, cfg, rgb_source_channels)
    limits = _normalization_limits(
        img_czyx, z, cfg, rgb_source_channels, normalization_cache
    )
    normalized = _normalize_with_limits(raw, limits)
    return np.stack(
        (normalized[..., 0:3], normalized[..., 3:6], normalized[..., 6:9]),
        axis=0,
    )


def _choose_patch(mask_yx: np.ndarray, cfg: UNet25DTrainConfig, rng: random.Random) -> tuple[int, int]:
    height, width = mask_yx.shape
    patch = cfg.patch
    if height < patch or width < patch:
        raise ValueError(f"Patch {patch} exceeds image size {(height, width)}")
    want_foreground = rng.random() < cfg.fg_fraction
    foreground = np.argwhere(mask_yx > 0)
    best: tuple[int, int] | None = None
    best_score: float | None = None
    for _ in range(cfg.max_tries):
        if want_foreground and foreground.size:
            centre_y, centre_x = foreground[rng.randint(0, len(foreground) - 1)]
            y0 = int(np.clip(centre_y - patch // 2 + rng.randint(-patch // 8, patch // 8), 0, height - patch))
            x0 = int(np.clip(centre_x - patch // 2 + rng.randint(-patch // 8, patch // 8), 0, width - patch))
        else:
            y0 = rng.randint(0, height - patch)
            x0 = rng.randint(0, width - patch)
        ratio = float(mask_yx[y0:y0 + patch, x0:x0 + patch].mean())
        if want_foreground and ratio >= cfg.fg_min_ratio:
            return y0, x0
        if not want_foreground and ratio <= cfg.bg_max_ratio:
            return y0, x0
        score = ratio if want_foreground else -ratio
        if best is None or best_score is None or score > best_score:
            best, best_score = (y0, x0), score
    return best or (rng.randint(0, height - patch), rng.randint(0, width - patch))


def make_25d_dataset(
    entries: Sequence[AnnotatedSlice], cfg: UNet25DTrainConfig, *, training: bool
) -> tuple[tf.data.Dataset, int]:
    """Create an infinite patch dataset from annotated target slices only."""
    if not entries:
        raise ValueError("Cannot create a dataset from zero annotated slices")
    first_image = open_3d_image_czyx(entries[0].image_zarr, level=cfg.level)
    if int(first_image.shape[0]) != 3:
        raise ValueError(f"Merged RGB training requires C=3, received C={first_image.shape[0]}")
    model_channels = 9
    rng = random.Random(cfg.seed + (0 if training else 10000))

    def generator():
        image_cache: dict[Path, zarr.Array] = {}
        mask_cache: dict[Path, np.ndarray] = {}
        normalization_cache: dict[Path, dict[tuple[int, int], tuple[float, float]]] = {}
        while True:
            entry = entries[rng.randint(0, len(entries) - 1)]
            if entry.image_zarr not in image_cache:
                image_cache[entry.image_zarr] = open_3d_image_czyx(entry.image_zarr, level=cfg.level)
            image = image_cache[entry.image_zarr]
            if entry.mask_tif not in mask_cache:
                mask_cache[entry.mask_tif] = read_binary_slice_mask(entry.mask_tif, image.shape[-2:])
            mask = mask_cache[entry.mask_tif]
            y0, x0 = _choose_patch(mask, cfg, rng)
            x_patch = make_25d_input_patch(
                image,
                entry.z_index,
                y0,
                x0,
                cfg.patch,
                cfg.patch,
                cfg,
                entry.rgb_source_channels,
            )
            per_image_cache = normalization_cache.setdefault(entry.image_zarr, {})
            limits = _normalization_limits(
                image,
                entry.z_index,
                cfg,
                entry.rgb_source_channels,
                per_image_cache,
            )
            x_patch = _normalize_with_limits(x_patch, limits)
            y_patch = mask[y0:y0 + cfg.patch, x0:x0 + cfg.patch]
            yield x_patch.astype(np.float32, copy=False), y_patch[..., None].astype(np.float32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((cfg.patch, cfg.patch, model_channels), tf.float32),
            tf.TensorSpec((cfg.patch, cfg.patch, 1), tf.float32),
        ),
    )
    if training:
        dataset = dataset.shuffle(128, seed=cfg.seed, reshuffle_each_iteration=True)
    return dataset.batch(cfg.batch).prefetch(tf.data.AUTOTUNE), model_channels


def _pad_patch_to_size(patch_hwc: np.ndarray, patch_size: int) -> np.ndarray:
    pad_y = patch_size - patch_hwc.shape[0]
    pad_x = patch_size - patch_hwc.shape[1]
    if pad_y < 0 or pad_x < 0:
        raise ValueError(f"Patch exceeds requested size: {patch_hwc.shape} vs {patch_size}")
    if pad_y == 0 and pad_x == 0:
        return patch_hwc
    mode = "reflect" if patch_hwc.shape[0] > 1 and patch_hwc.shape[1] > 1 else "edge"
    return np.pad(patch_hwc, ((0, pad_y), (0, pad_x), (0, 0)), mode=mode)


def predict_25d_probability(
    model: tf.keras.Model,
    image_czyx: Any,
    z_index: int,
    cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
    *,
    normalization_cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> np.ndarray:
    """Predict one middle-slice probability map from merged-RGB context."""
    height = int(image_czyx.shape[-2])
    width = int(image_czyx.shape[-1])
    limits = _normalization_limits(
        image_czyx, z_index, cfg, rgb_source_channels, normalization_cache
    )
    probability = np.zeros((height, width), dtype=np.float32)
    tiles: list[np.ndarray] = []
    coordinates: list[tuple[int, int, int, int]] = []
    batch_size = max(1, cfg.predict_batch_size)

    def flush() -> None:
        if not tiles:
            return
        batch = np.stack(tiles).astype(np.float32, copy=False)
        predictions = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        for prediction, (y0, x0, tile_height, tile_width) in zip(predictions, coordinates):
            probability[y0:y0 + tile_height, x0:x0 + tile_width] = prediction[:tile_height, :tile_width, 0]
        tiles.clear()
        coordinates.clear()

    for y0 in range(0, height, cfg.patch):
        tile_height = min(cfg.patch, height - y0)
        for x0 in range(0, width, cfg.patch):
            tile_width = min(cfg.patch, width - x0)
            tile = make_25d_input_patch(
                image_czyx,
                z_index,
                y0,
                x0,
                tile_height,
                tile_width,
                cfg,
                rgb_source_channels,
            )
            tile = _normalize_with_limits(tile, limits)
            tiles.append(_pad_patch_to_size(tile, cfg.patch))
            coordinates.append((y0, x0, tile_height, tile_width))
            if len(tiles) >= batch_size:
                flush()
    flush()
    return probability


def _binary_metrics(reference: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-12) -> dict[str, float]:
    reference = reference.astype(bool)
    prediction = prediction.astype(bool)
    intersection = float(np.count_nonzero(reference & prediction))
    union = float(np.count_nonzero(reference | prediction))
    reference_count = float(np.count_nonzero(reference))
    prediction_count = float(np.count_nonzero(prediction))
    true_positive = intersection
    false_positive = float(np.count_nonzero(~reference & prediction))
    false_negative = float(np.count_nonzero(reference & ~prediction))
    return {
        "iou": intersection / (union + epsilon),
        "dice": 2.0 * intersection / (reference_count + prediction_count + epsilon),
        "precision": true_positive / (true_positive + false_positive + epsilon),
        "recall": true_positive / (true_positive + false_negative + epsilon),
    }


def _display_composite(
    image_czyx: Any,
    z_index: int,
    cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
    normalization_cache: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return the normalized merged RGB middle image used by the model."""
    context = make_merged_rgb_context_slice(
        image_czyx,
        z_index,
        cfg,
        rgb_source_channels,
        normalization_cache=normalization_cache,
    )
    return context[1]


def evaluate_annotated_slices(
    model: tf.keras.Model,
    entries: Sequence[AnnotatedSlice],
    cfg: UNet25DTrainConfig,
    output_dir: Path,
    *,
    partitions: Mapping[tuple[str, int], str] | None = None,
) -> tuple[Path, Path, Path]:
    """Evaluate and preview only the manually annotated training/validation slices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_cache: dict[Path, zarr.Array] = {}
    normalization_cache: dict[Path, dict[tuple[int, int], tuple[float, float]]] = {}
    rows: list[dict[str, object]] = []
    for entry in entries:
        if entry.image_zarr not in image_cache:
            image_cache[entry.image_zarr] = open_3d_image_czyx(entry.image_zarr, level=cfg.level)
        image = image_cache[entry.image_zarr]
        reference = read_binary_slice_mask(entry.mask_tif, image.shape[-2:])
        per_image_cache = normalization_cache.setdefault(entry.image_zarr, {})
        probability = predict_25d_probability(
            model,
            image,
            entry.z_index,
            cfg,
            entry.rgb_source_channels,
            normalization_cache=per_image_cache,
        )
        prediction = (probability >= cfg.threshold).astype(np.uint8)
        metrics = _binary_metrics(reference, prediction)
        partition = (partitions or {}).get((str(entry.image_zarr.resolve()), entry.slice_1based), "unspecified")
        rows.append({
            "sample": entry.sample,
            "partition": partition,
            "slice_1based": entry.slice_1based,
            "threshold": cfg.threshold,
            "foreground_reference": int(np.count_nonzero(reference)),
            "foreground_prediction": int(np.count_nonzero(prediction)),
            **metrics,
            "image_zarr": str(entry.image_zarr),
            "manual_mask": str(entry.mask_tif),
        })

        figure, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
        axes[0].imshow(_display_composite(image, entry.z_index, cfg, entry.rgb_source_channels, per_image_cache))
        axes[0].set_title("Merged RGB middle slice")
        axes[1].imshow(reference, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Manual mask")
        axes[2].imshow(probability, cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Foreground probability")
        axes[3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Predicted mask")
        overlay = _display_composite(image, entry.z_index, cfg, entry.rgb_source_channels, per_image_cache)
        axes[4].imshow(overlay)
        axes[4].contour(prediction, levels=[0.5], colors="white", linewidths=0.6)
        axes[4].contour(reference, levels=[0.5], colors="red", linewidths=0.6)
        axes[4].set_title(f"Overlay | IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}")
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"{entry.sample} | Z{entry.slice_1based:03d} | {partition}")
        safe_sample = entry.sample.replace("/", "__").replace("\\", "__")
        figure.savefig(output_dir / f"{safe_sample}__z{entry.slice_1based:03d}__mask_qc.png", dpi=160)
        plt.close(figure)

    csv_path = output_dir / "u_net_3d_25d_annotation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    def aggregate(selected: Sequence[dict[str, object]]) -> dict[str, float | int]:
        return {
            "count": len(selected),
            "mean_iou": float(np.mean([float(row["iou"]) for row in selected])),
            "mean_dice": float(np.mean([float(row["dice"]) for row in selected])),
            "mean_precision": float(np.mean([float(row["precision"]) for row in selected])),
            "mean_recall": float(np.mean([float(row["recall"]) for row in selected])),
        }

    partition_names = sorted({str(row["partition"]) for row in rows})
    by_partition = {
        name: aggregate([row for row in rows if row["partition"] == name])
        for name in partition_names
    }
    overall = aggregate(rows)
    summary = {
        "threshold": cfg.threshold,
        "annotated_slice_count": len(rows),
        **{key: value for key, value in overall.items() if key != "count"},
        "by_partition": by_partition,
        "rows": rows,
    }
    json_path = output_dir / "u_net_3d_25d_annotation_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text_path = output_dir / "u_net_3d_25d_annotation_metrics.txt"
    text_path.write_text(
        "\n".join([
            "PFT 2.5D U-Net annotated-slice evaluation",
            "=" * 72,
            f"Threshold: {cfg.threshold}",
            f"Annotated slices: {len(rows)}",
            f"Mean IoU: {summary['mean_iou']:.6f}",
            f"Mean Dice: {summary['mean_dice']:.6f}",
            f"Mean precision: {summary['mean_precision']:.6f}",
            f"Mean recall: {summary['mean_recall']:.6f}",
            "",
            *[
                f"{name}: count={values['count']}, IoU={values['mean_iou']:.6f}, "
                f"Dice={values['mean_dice']:.6f}, precision={values['mean_precision']:.6f}, "
                f"recall={values['mean_recall']:.6f}"
                for name, values in by_partition.items()
            ],
        ]) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, text_path


def train_3d_25d_unet(cfg: UNet25DTrainConfig | None = None) -> dict[str, Path]:
    """Train the 2.5D model, save models, and evaluate annotated slices."""
    cfg = cfg or UNet25DTrainConfig()
    cfg.image_root = Path(cfg.image_root or _default_image_root(cfg.project_root))
    cfg.mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root))
    cfg.model_root = ensure_dir(Path(cfg.model_root or _default_model_root(cfg.project_root)))
    if cfg.z_radius != 1:
        raise ValueError("This thesis workflow requires z_radius=1 (Z-1, Z, Z+1)")
    if cfg.channels is not None:
        raise ValueError(
            "Merged-RGB training always uses all three wavelength-mapped channels; "
            "do not provide a channel subset."
        )
    if cfg.threshold != 0.5:
        raise ValueError("The agreed binary threshold is fixed at 0.5")

    entries = list_annotated_slices(cfg)
    train_entries, validation_entries, split_note = split_annotated_slices(entries, cfg)
    training_dataset, model_channels = make_25d_dataset(train_entries, cfg, training=True)
    validation_dataset, validation_channels = make_25d_dataset(validation_entries, cfg, training=False)
    if model_channels != validation_channels:
        raise ValueError("Training and validation input-channel counts differ")

    model = build_unet((cfg.patch, cfg.patch, model_channels), base_filters=cfg.base_filters, dropout=cfg.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef],
    )
    best_model = cfg.model_root / "u_net_3d_25d_best.keras"
    final_model = cfg.model_root / "u_net_3d_25d_final.keras"
    history_csv = cfg.model_root / "u_net_3d_25d_history.csv"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_model), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(history_csv)),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ]
    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
    )
    model.save(final_model)
    curve_outputs = save_training_curves(history, cfg.model_root, cfg.dataset)

    best = tf.keras.models.load_model(
        str(best_model),
        custom_objects={"bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef, "iou_coef": iou_coef},
    )
    evaluation_dir = cfg.model_root / "evaluation_configured_target_slices"
    if evaluation_dir.exists():
        shutil.rmtree(evaluation_dir)
    partitions = {
        (str(entry.image_zarr.resolve()), entry.slice_1based): "train"
        for entry in train_entries
    }
    partitions.update({
        (str(entry.image_zarr.resolve()), entry.slice_1based): "validation"
        for entry in validation_entries
    })
    metrics_csv, metrics_json, metrics_txt = evaluate_annotated_slices(
        best, entries, cfg, evaluation_dir, partitions=partitions
    )

    summary = {
        "dataset": cfg.dataset,
        "model_contract": MODEL_CONTRACT,
        "input_definition": "Z-1/Z/Z+1 wavelength-mapped merged RGB images concatenated as 9 channels",
        "model_input_channels": model_channels,
        "z_radius": cfg.z_radius,
        "selected_image_channels": "all three, mapped by wavelength to RGB",
        "rgb_colour_mapping_nm": {"red": 561.0, "green": 488.0, "blue": 405.0},
        "training_target_policy": "one configured target slice per source volume",
        "target_slice_map": dict(TARGET_SLICE_BY_VOLUME_KEY),
        "training_targets_by_sample": {entry.sample: entry.slice_1based for entry in entries},
        "threshold": cfg.threshold,
        "annotation_count": len(entries),
        "train_annotation_count": len(train_entries),
        "validation_annotation_count": len(validation_entries),
        "split_note": split_note,
        "train_entries": [asdict(entry) for entry in train_entries],
        "validation_entries": [asdict(entry) for entry in validation_entries],
        "image_root": str(cfg.image_root),
        "mask_root": str(cfg.mask_root),
        "patch": cfg.patch,
        "batch": cfg.batch,
        "epochs_requested": cfg.epochs,
        "best_model": str(best_model),
        "final_model": str(final_model),
        "evaluation_metrics": str(metrics_json),
        "training_curves": {name: str(path) for name, path in curve_outputs.items()},
    }
    summary_path = cfg.model_root / "u_net_3d_25d_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {
        "best_model": best_model,
        "final_model": final_model,
        "history": history_csv,
        **curve_outputs,
        "summary": summary_path,
        "metrics_csv": metrics_csv,
        "metrics_json": metrics_json,
        "metrics_txt": metrics_txt,
        "evaluation_dir": evaluation_dir,
    }


__all__ = [
    "MODEL_CONTRACT",
    "AnnotatedSlice",
    "UNet25DTrainConfig",
    "evaluate_annotated_slices",
    "list_annotated_slices",
    "load_3d_image_czyx",
    "open_3d_image_czyx",
    "make_25d_input_patch",
    "make_25d_input_slice",
    "make_merged_rgb_context_slice",
    "predict_25d_probability",
    "resolve_rgb_source_channels",
    "read_binary_slice_mask",
    "train_3d_25d_unet",
]
