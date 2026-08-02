"""Core training implementation for 2D U-Net foreground/background detection.

This module is used for both supported 2D datasets:

``2d_time``
    One fluorescence channel per time frame.

``2d_wga_dapi``
    Two fluorescence channels, DAPI and WGA, processed together as one input.

The network performs semantic binary classification. Every pixel is assigned a
foreground probability and is compared with a hand-labelled reference mask.
Positive instance labels in a reference TIFF are converted to semantic
foreground by the rule ``mask > 0``.

Input policy
------------
The input is the intensity-preserving output of the local-threshold filter. Do
not save a separate normalized training dataset. The complete image is
normalized inside this module, independently per channel, before patches are
sampled. The default percentile normalization maps P1 to 0 and P99.8 to 1.
Training and inference must use the same normalization method.

Where to change parameters
--------------------------
The main adjustable values are collected in :class:`UNet2DTrainConfig`. They
can be changed through the terminal launcher or when constructing the config in
Python. The class documentation explains the expected effect of increasing or
decreasing every parameter.

Important consistency rules
---------------------------
* ``patch`` must be a multiple of 16 because the U-Net contains four pooling
  stages.
* Inference must use the same ``patch`` and ``normalize`` values as training.
* Changing ``base_filters`` changes the architecture. A model trained with one
  value cannot be loaded into a differently constructed architecture.
* Validation splitting is performed at sample level, not at patch level, to
  reduce information leakage between training and validation data.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile as tiff

from PFT.core_prog_parts.common_paths import ensure_dir, find_project_root
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, move_yx_to_last, normalize_axes


DATASETS_2D = ("2d_time", "2d_wga_dapi")
EXPECTED_LOCAL_THRESHOLD_PERCENTILE = {
    "2d_time": 97.5,
    "2d_wga_dapi": 98.0,
}


@dataclass
class UNet2DTrainConfig:
    """Configuration for thesis-aligned 2D semantic foreground training.

    Parameter guide
    ---------------
    project_root:
        Project directory used to construct all default paths. Changing it does
        not change learning behaviour.
    dataset:
        ``"2d_time"`` selects one-channel HADA data. ``"2d_wga_dapi"``
        selects two-channel WGA-DAPI data. Models are dataset-specific.
    level:
        OME-Zarr pyramid level. ``0`` uses full resolution. A higher level uses
        a downsampled image, reducing memory and training time but also removing
        fine spatial detail. Masks must correspond to the selected level.
    image_root, mask_root, model_root:
        Optional path overrides. They change where data are read or results are
        written, but do not change the network itself.

    patch:
        Training crop width and height in pixels. Larger patches provide more
        spatial context but use substantially more GPU memory and usually reduce
        the number of patches processed per second. Smaller patches require less
        memory but may omit complete cells or long structures. The value must be
        a positive multiple of 16 and must also be used during inference.
    batch:
        Number of patches processed in one optimizer update. Increasing it may
        improve throughput and produce smoother gradients, but requires more GPU
        memory. Decreasing it is the first action when an out-of-memory error
        occurs. Very small batches can make training noisier.
    epochs:
        Maximum passes through the configured training schedule. More epochs can
        improve convergence, but also increase runtime and overfitting risk.
        Select the best checkpoint from validation loss rather than assuming the
        final epoch is optimal.
    steps_per_epoch:
        Number of randomly sampled training batches per epoch. Increasing it
        exposes the model to more patches before the next epoch and increases
        runtime approximately proportionally.
    val_steps:
        Number of validation batches evaluated per epoch. More steps provide a
        more stable validation estimate but increase evaluation time.
    val_split:
        Fraction of complete samples reserved for validation. A larger value
        gives a more reliable validation estimate but leaves fewer samples for
        fitting. For a small dataset, 0.2 is a practical starting point.
    seed:
        Controls sample splitting and random patch sampling. Keeping it fixed
        improves reproducibility. Changing it can alter the exact split and
        therefore the measured validation curves.

    lr:
        Initial Adam learning rate. A larger value can learn faster but may make
        loss unstable or skip a good solution. A smaller value is more stable but
        may require more epochs. ``1e-3`` is the current starting value.
    base_filters:
        Number of filters in the first U-Net level. Each deeper level multiplies
        this value. Increasing it raises model capacity, memory use, model size,
        and training time. Decreasing it makes the model lighter but may reduce
        its ability to represent complex foreground patterns.
    dropout:
        Dropout probability applied after each convolutional block. Increasing
        it can reduce overfitting, but excessive dropout can slow convergence and
        underfit the data. ``0.0`` disables dropout. Values around 0.1 to 0.3 are
        reasonable experimental settings when validation loss worsens while
        training loss continues to improve.
    normalize:
        Intensity conversion used before patch extraction. ``"percentile"`` is
        recommended and maps P1 to 0 and P99.8 to 1 per channel. The optional
        ``"scale_uint16"`` mode divides by 65535 and is appropriate only when
        the input has a consistent uint16 intensity range. Inference must use the
        same mode as training.

    fg_fraction:
        Fraction of requested training patches that must contain labelled
        foreground. The default is 0.75: approximately 75% foreground-aware
        patches and 25% background patches. A value of 1.0 requests only
        foreground-containing patches, but this is normally not recommended
        because the network still needs examples of residual fluorescence and
        background texture. Increasing this value usually improves foreground
        recall; decreasing it strengthens background discrimination.
    fg_min_ratio:
        Minimum fraction of labelled foreground pixels required in a positive
        patch. The default is 0.01, corresponding to about 655 labelled pixels
        in a 256 x 256 patch. This lower requirement is appropriate for thin
        bacterial walls, septa, and sparse HADA signal. A value of 0.05 would
        require about 3277 positive pixels and may reject biologically useful
        sparse patches.
    bg_max_ratio:
        Maximum foreground fraction allowed in a requested background patch.
        The default 0.001 permits at most about 66 positive pixels in a 256 x
        256 patch. Set it to 0.0 to require strictly empty reference masks.
        Retaining a small number of background patches is important because the
        model must learn what should remain class 0.
    max_tries:
        Maximum attempts to locate a crop that satisfies the requested
        foreground/background ratio. Increasing it can improve adherence to the
        sampling rules but adds CPU preprocessing time. Lowering it is faster but
        causes more fallback crops when valid regions are rare.
    """

    # Dataset and path selection. These parameters do not directly change the
    # optimizer, but selecting another level or dataset changes the input data.
    project_root: Path = find_project_root(Path(__file__).resolve())
    dataset: str = "2d_time"
    level: int = 0

    image_root: Path | None = None
    mask_root: Path | None = None
    model_root: Path | None = None

    # Patch sampling and training duration.
    patch: int = 256
    batch: int = 4
    epochs: int = 30
    steps_per_epoch: int = 150
    val_steps: int = 30
    val_split: float = 0.2
    seed: int = 1337

    # Optimizer, architecture, regularization, and intensity preprocessing.
    lr: float = 1e-3
    base_filters: int = 16
    dropout: float = 0.0
    normalize: str = "percentile"

    # Class-aware crop sampling. These values help compensate for the typically
    # much larger background area in fluorescence images.
    fg_fraction: float = 0.75
    fg_min_ratio: float = 0.01
    bg_max_ratio: float = 0.001
    max_tries: int = 160

    # Input-pipeline and callback controls. Fixed validation patches remove the
    # large epoch-to-epoch metric fluctuations caused by resampling validation
    # crops. A small image cache reduces repeated OME-Zarr decoding on CPU.
    shuffle_buffer: int = 64
    cache_size: int = 6
    fixed_validation: bool = True
    reduce_lr_patience: int = 4
    early_stopping_patience: int = 8
    min_lr: float = 1e-6


def validate_train_config(cfg: UNet2DTrainConfig) -> None:
    """Validate parameters that affect U-Net dimensions and sampling.

    Args:
        cfg (UNet2DTrainConfig): Value specifying cfg for the operation.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> validate_train_config(cfg=config)
    """
    if cfg.patch <= 0 or cfg.patch % 16 != 0:
        raise ValueError("patch must be a positive multiple of 16 for four pooling stages")
    if cfg.batch <= 0 or cfg.epochs <= 0:
        raise ValueError("batch and epochs must be positive")
    if cfg.steps_per_epoch <= 0 or cfg.val_steps <= 0:
        raise ValueError("training and validation steps must be positive")
    if not 0.0 < cfg.val_split < 1.0:
        raise ValueError("val_split must be between zero and one")
    if not 0.0 <= cfg.fg_fraction <= 1.0:
        raise ValueError("fg_fraction must be in [0,1]")
    if not 0.0 <= cfg.bg_max_ratio <= cfg.fg_min_ratio <= 1.0:
        raise ValueError("require 0 <= bg_max_ratio <= fg_min_ratio <= 1")
    if cfg.max_tries <= 0:
        raise ValueError("max_tries must be positive")
    if cfg.lr <= 0:
        raise ValueError("learning rate must be positive")
    if cfg.shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive")
    if cfg.cache_size < 0:
        raise ValueError("cache_size must be non-negative")
    if cfg.reduce_lr_patience <= 0 or cfg.early_stopping_patience <= 0:
        raise ValueError("callback patience values must be positive")
    if cfg.early_stopping_patience <= cfg.reduce_lr_patience:
        raise ValueError("early_stopping_patience must exceed reduce_lr_patience")
    if cfg.min_lr <= 0 or cfg.min_lr > cfg.lr:
        raise ValueError("min_lr must be positive and not exceed the initial learning rate")


def _default_mask_root(project_root: Path, dataset: str) -> Path:
    """Return default mask root for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _default_mask_root(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    return project_root / "results" / "training_files" / "U-net" / dataset


def _default_model_root(project_root: Path, dataset: str) -> Path:
    """Return default model root for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _default_model_root(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    return project_root / "models" / f"u_net_{dataset}"


def _local_threshold_base(project_root: Path, dataset: str) -> Path:
    """Return local threshold base for the supplied inputs.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _local_threshold_base(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    return (
        project_root
        / "results"
        / "Filters"
        / "Local_high_threshold"
        / dataset
        / "intensity_preserved"
    )


def default_filtered_root(project_root: Path, dataset: str) -> Path:
    """Resolve the production local-threshold folder for one 2D dataset.

    The selected production settings are p97.5 for ``2d_time`` and p98.0 for
    ``2d_wga_dapi``. The remainder of the folder label is discovered so minor
    parameter-label changes do not break the training command.

    Args:
        project_root (Path): Root directory of the PFT project containing the results, models, scripts, and source-code directories.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        Path: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = default_filtered_root(project_root=Path("path/to/resource"), dataset="2d_time")
    """
    dataset = dataset.strip().lower()
    if dataset not in DATASETS_2D:
        raise ValueError(f"Unsupported 2D dataset: {dataset}")

    base = _local_threshold_base(project_root, dataset)
    percentile = EXPECTED_LOCAL_THRESHOLD_PERCENTILE[dataset]
    expected_label = base / f"p{percentile:.1f}_k3f40_s370_k10f40"
    if expected_label.is_dir():
        return expected_label

    expected_prefix = f"p{percentile:.1f}_"
    matches = sorted(p for p in base.glob(f"{expected_prefix}*") if p.is_dir())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        exact = [p for p in matches if (p / "local_threshold_dataset_summary.txt").exists()]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(
            f"Multiple local-threshold folders match {expected_prefix} under {base}. "
            "Pass --image-root explicitly."
        )

    # Return the expected base so a later error reports the useful location.
    return expected_label


def normalize_image01(x: np.ndarray, mode: str = "percentile") -> np.ndarray:
    """Normalize one complete HWC image channel-wise to [0, 1].

    Percentile normalization maps P1 to zero and P99.8 to one. It must be
    applied to the complete image before patch extraction, not independently
    to every patch. This keeps training and inference preprocessing identical.

    Args:
        x (np.ndarray): Horizontal coordinate or numerical input value used by the operation.
        mode (str): Text value specifying mode. Defaults to ``"percentile"``.

    Returns:
        np.ndarray: Array containing the processed result.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = normalize_image01(x=image_array)
    """
    x = np.asarray(x, dtype=np.float32)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Expected YX or YXC image, received shape={x.shape}")

    if mode == "scale_uint16":
        out = np.clip(x / 65535.0, 0.0, 1.0)
    elif mode == "percentile":
        out = np.empty_like(x, dtype=np.float32)
        for channel in range(x.shape[-1]):
            plane = x[..., channel]
            lo = float(np.percentile(plane, 1.0))
            hi = float(np.percentile(plane, 99.8))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                out[..., channel] = 0.0
            else:
                out[..., channel] = np.clip(
                    (plane - lo) / (hi - lo + 1e-8), 0.0, 1.0
                )
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return out[..., 0] if squeeze else out.astype(np.float32, copy=False)


def _select_axis(arr: np.ndarray, axes: str, axis: str, index: int) -> tuple[np.ndarray, str]:
    """Select axis according to the configured criteria.

    Args:
        arr (np.ndarray): Array containing arr.
        axes (str): Axis specification describing the dimensional order of the image data.
        axis (str): Array axis along which the operation is performed.
        index (int): Zero-based index of the selected element.

    Returns:
        tuple[np.ndarray, str]: Collection containing the generated or selected values.

    Example:
        >>> result = _select_axis(
        ...     arr=image_array,
        ...     axes="axes",
        ...     axis="axis",
        ...     index=1,
        ... )
    """
    position = axes.index(axis)
    arr = np.take(arr, int(index), axis=position)
    return arr, axes[:position] + axes[position + 1 :]


def ome_zarr_to_hwc_frames_2d(
    zarr_path: Path,
    *,
    dataset: str,
    level: int = 0,
) -> list[np.ndarray]:
    """Load every 2D frame from an OME-Zarr store as HWC float32 arrays.

    Args:
        zarr_path (Path): Filesystem path associated with Zarr.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        level (int): Numerical value controlling level. Defaults to ``0``.

    Returns:
        list[np.ndarray]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = ome_zarr_to_hwc_frames_2d(zarr_path=Path("path/to/resource"), dataset="2d_time")
    """
    dataset = dataset.strip().lower()
    if dataset not in DATASETS_2D:
        raise ValueError(f"Unsupported 2D dataset: {dataset}")

    arr, axes = load_ome_zarr(zarr_path, level=level, as_numpy=True)
    arr = np.asarray(arr)
    axes = normalize_axes(axes)
    arr, axes = move_yx_to_last(arr, axes)

    for axis in tuple(axes):
        if axis not in {"t", "c", "y", "x"}:
            size = int(arr.shape[axes.index(axis)])
            if size != 1:
                raise ValueError(
                    f"Unsupported non-singleton axis {axis}={size} in {zarr_path}"
                )
            arr, axes = _select_axis(arr, axes, axis, 0)

    time_count = int(arr.shape[axes.index("t")]) if "t" in axes else 1
    frames: list[np.ndarray] = []
    for time_index in range(time_count):
        frame = arr
        frame_axes = axes
        if "t" in frame_axes:
            frame, frame_axes = _select_axis(frame, frame_axes, "t", time_index)

        if frame_axes == "yx":
            hwc = frame[..., None]
        elif frame_axes == "cyx":
            channel_axis = frame_axes.index("c")
            channel_count = int(frame.shape[channel_axis])
            if dataset == "2d_time":
                hwc = np.take(frame, 0, axis=channel_axis)[..., None]
            else:
                if channel_count < 2:
                    raise ValueError(
                        f"2d_wga_dapi requires two channels, found {channel_count}: {zarr_path}"
                    )
                hwc = np.stack(
                    [
                        np.take(frame, 0, axis=channel_axis),
                        np.take(frame, 1, axis=channel_axis),
                    ],
                    axis=-1,
                )
        else:
            raise ValueError(
                f"Unsupported 2D OME-Zarr axes '{frame_axes}' for {zarr_path}"
            )
        frames.append(np.asarray(hwc, dtype=np.float32))
    return frames


def ome_zarr_to_hwc_2d(zarr_path: Path, *, dataset: str, level: int = 0) -> np.ndarray:
    """Backward-compatible loader returning the first available 2D frame.

    Args:
        zarr_path (Path): Filesystem path associated with Zarr.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.
        level (int): Numerical value controlling level. Defaults to ``0``.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = ome_zarr_to_hwc_2d(zarr_path=Path("path/to/resource"), dataset="2d_time")
    """
    return ome_zarr_to_hwc_frames_2d(zarr_path, dataset=dataset, level=level)[0]


def read_reference_mask_2d(mask_path: Path) -> tuple[np.ndarray, int]:
    """Read a 2D reference mask and convert every positive label to foreground.

    The source mask may already be binary or may contain positive instance
    labels. This follows the Methods definition ``M_bin = 1`` for ``M > 0``.
    Negative, non-finite, or non-2D masks are rejected.

    Args:
        mask_path (Path): Filesystem path associated with mask.

    Returns:
        tuple[np.ndarray, int]: Collection containing the generated or selected values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = read_reference_mask_2d(mask_path=Path("path/to/resource"))
    """
    source = np.asarray(tiff.imread(mask_path))
    source = np.squeeze(source)
    if source.ndim != 2:
        raise ValueError(
            f"Reference mask must be 2D after squeeze: {mask_path}, {source.shape}"
        )
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError(f"Reference mask must be numeric: {mask_path}, {source.dtype}")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"Reference mask contains non-finite values: {mask_path}")
    if np.any(source < 0):
        raise ValueError(f"Reference mask contains negative labels: {mask_path}")

    positive_labels = np.unique(source[source > 0])
    binary = (source > 0).astype(np.uint8)
    return binary, int(positive_labels.size)


def read_binary_mask_2d(mask_path: Path) -> np.ndarray:
    """Return the binary semantic foreground target derived from a source mask.

    Args:
        mask_path (Path): Filesystem path associated with mask.

    Returns:
        np.ndarray: Array containing the processed result.

    Example:
        >>> result = read_binary_mask_2d(mask_path=Path("path/to/resource"))
    """
    binary, _ = read_reference_mask_2d(mask_path)
    return binary


def find_mask_file(sample_dir: Path) -> Path | None:
    """Find the canonical numerical reference mask for one sample.

    Args:
        sample_dir (Path): Directory used for sample.

    Returns:
        Path | None: Resolved or generated filesystem path.

    Example:
        >>> result = find_mask_file(sample_dir=Path("path/to/resource"))
    """
    canonical = sample_dir / "mask.tif"
    if canonical.is_file():
        return canonical
    for name in ("mask.tiff", "labels.tif", "label.tif"):
        candidate = sample_dir / name
        if candidate.is_file():
            return candidate
    return None


def list_2d_training_pairs(cfg: UNet2DTrainConfig) -> list[tuple[Path, Path]]:
    """Pair filtered image stores and hand-labelled masks by sample directory.

    Args:
        cfg (UNet2DTrainConfig): Value specifying cfg for the operation.

    Returns:
        list[tuple[Path, Path]]: Resolved or generated filesystem path.

    Raises:
        FileNotFoundError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = list_2d_training_pairs(cfg=config)
    """
    image_root = Path(cfg.image_root or default_filtered_root(cfg.project_root, cfg.dataset))
    mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root, cfg.dataset))
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"Filtered image root does not exist: {image_root}. Run local-threshold filtering "
            "or pass --image-root."
        )
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    pairs: list[tuple[Path, Path]] = []
    for sample_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
        mask_path = find_mask_file(sample_dir)
        image_path = image_root / sample_dir.name / "image.ome.zarr"
        if mask_path is not None and image_path.is_dir():
            pairs.append((image_path, mask_path))
    return pairs


def validate_2d_training_pairs(
    pairs: list[tuple[Path, Path]], cfg: UNet2DTrainConfig
) -> list[dict[str, Any]]:
    """Validate shapes, channels, source masks, and foreground/background pixels.

    Args:
        pairs (list[tuple[Path, Path]]): Filesystem path used for pairs.
        cfg (UNet2DTrainConfig): Value specifying cfg for the operation.

    Returns:
        list[dict[str, Any]]: Mapping containing the generated or resolved values.

    Raises:
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = validate_2d_training_pairs(pairs=Path("path/to/resource"), cfg=config)
    """
    rows: list[dict[str, Any]] = []
    for image_path, mask_path in pairs:
        sample = image_path.parent.name
        status = "PASS"
        issue = ""
        shape = ""
        channels = 0
        frame_count = 0
        foreground = 0
        background = 0
        positive_label_count = 0
        try:
            frames = ome_zarr_to_hwc_frames_2d(
                image_path, dataset=cfg.dataset, level=cfg.level
            )
            mask, positive_label_count = read_reference_mask_2d(mask_path)
            frame_count = len(frames)
            channels = int(frames[0].shape[-1])
            shape = "x".join(str(v) for v in frames[0].shape)
            if any(frame.shape[:2] != mask.shape for frame in frames):
                raise ValueError(
                    f"Image/mask YX mismatch: frames={frames[0].shape[:2]}, mask={mask.shape}"
                )
            expected_channels = 1 if cfg.dataset == "2d_time" else 2
            if channels != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} input channels, found {channels}"
                )
            foreground = int(np.count_nonzero(mask))
            background = int(mask.size - foreground)
            if foreground == 0 or background == 0:
                raise ValueError(
                    f"Mask must contain foreground and background pixels: fg={foreground}, bg={background}"
                )
            if min(mask.shape) < cfg.patch:
                raise ValueError(
                    f"Patch {cfg.patch} exceeds mask dimensions {mask.shape}"
                )
        except Exception as exc:  # validation report should retain all samples
            status = "FAIL"
            issue = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "dataset": cfg.dataset,
                "sample": sample,
                "status": status,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "frame_count": frame_count,
                "image_shape_yxc": shape,
                "channels": channels,
                "foreground_pixels": foreground,
                "background_pixels": background,
                "positive_source_labels": positive_label_count,
                "issue": issue,
            }
        )
    return rows


def split_pairs(
    pairs: list[tuple[Path, Path]], cfg: UNet2DTrainConfig
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Create the fixed 80:20 image-pair split before patch extraction.

    Args:
        pairs (list[tuple[Path, Path]]): Filesystem path used for pairs.
        cfg (UNet2DTrainConfig): Value specifying cfg for the operation.

    Returns:
        tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = split_pairs(pairs=Path("path/to/resource"), cfg=config)
    """
    shuffled = list(pairs)
    random.Random(cfg.seed).shuffle(shuffled)
    if len(shuffled) < 2:
        raise RuntimeError("At least two image-mask pairs are required for an independent validation split.")
    n_val = max(1, int(round(len(shuffled) * cfg.val_split)))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1
    return shuffled[n_val:], shuffled[:n_val]


def _random_crop(height: int, width: int, patch: int, rng: random.Random) -> tuple[int, int]:
    """Return random crop for the supplied inputs.

    Args:
        height (int): Numerical value controlling height.
        width (int): Numerical value controlling width.
        patch (int): Numerical value controlling patch.
        rng (random.Random): Random-number generator used for reproducible sampling or augmentation.

    Returns:
        tuple[int, int]: Collection containing the generated or selected values.

    Example:
        >>> result = _random_crop(
        ...     height=1,
        ...     width=1,
        ...     patch=1,
        ...     rng=...,
        ... )
    """
    return rng.randint(0, height - patch), rng.randint(0, width - patch)


def sample_patch_2d(
    img: np.ndarray,
    mask: np.ndarray,
    cfg: UNet2DTrainConfig,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one class-aware patch without silently returning an empty positive crop.

    Approximately ``cfg.fg_fraction`` of requests are positive. A positive crop
    is centred near a labelled mask pixel and must satisfy ``fg_min_ratio``.
    The remaining requests are negative/background crops and must satisfy
    ``bg_max_ratio``. If the exact positive criterion cannot be reached after
    ``max_tries``, the best crop is accepted only when it still contains at
    least one labelled foreground pixel. Consequently, a requested positive
    patch can never silently become an all-zero target patch.

    Some negative patches are intentionally retained. Removing all negative
    examples would make false-positive foreground predictions more likely at
    inference time.

    Args:
        img (np.ndarray): Array containing img.
        mask (np.ndarray): Binary or labeled segmentation mask associated with the input image.
        cfg (UNet2DTrainConfig): Value specifying cfg for the operation.
        rng (random.Random): Random-number generator used for reproducible sampling or augmentation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = sample_patch_2d(
        ...     img=image_array,
        ...     mask=image_array,
        ...     cfg=config,
        ...     rng=...,
        ... )
    """
    height, width = mask.shape
    patch = cfg.patch
    if height < patch or width < patch:
        raise ValueError(f"Patch {patch} is larger than image/mask {(height, width)}")

    want_foreground = rng.random() < cfg.fg_fraction
    foreground_coordinates = np.argwhere(mask > 0)
    if want_foreground and len(foreground_coordinates) == 0:
        raise ValueError("Foreground patch requested, but the reference mask is empty")

    best: tuple[np.ndarray, np.ndarray] | None = None
    best_ratio = -1.0 if want_foreground else float("inf")

    for _ in range(cfg.max_tries):
        if want_foreground:
            cy, cx = foreground_coordinates[rng.randrange(len(foreground_coordinates))]
            y0 = int(
                np.clip(
                    cy - patch // 2 + rng.randint(-patch // 8, patch // 8),
                    0,
                    height - patch,
                )
            )
            x0 = int(
                np.clip(
                    cx - patch // 2 + rng.randint(-patch // 8, patch // 8),
                    0,
                    width - patch,
                )
            )
        else:
            y0, x0 = _random_crop(height, width, patch, rng)

        image_patch = img[y0 : y0 + patch, x0 : x0 + patch, :]
        mask_patch = mask[y0 : y0 + patch, x0 : x0 + patch]
        foreground_ratio = float(mask_patch.mean())

        if want_foreground:
            if foreground_ratio >= cfg.fg_min_ratio:
                return image_patch, mask_patch
            if foreground_ratio > best_ratio:
                best = image_patch, mask_patch
                best_ratio = foreground_ratio
        else:
            if foreground_ratio <= cfg.bg_max_ratio:
                return image_patch, mask_patch
            if foreground_ratio < best_ratio:
                best = image_patch, mask_patch
                best_ratio = foreground_ratio

    if best is None:
        raise RuntimeError("Patch sampler could not produce a candidate crop")

    if want_foreground and best_ratio <= 0.0:
        raise RuntimeError(
            "Positive patch sampling failed: every candidate had an all-zero mask"
        )

    # For rare geometries, return the best non-empty positive crop or the
    # cleanest available background crop rather than an unrelated random patch.
    return best


def make_2d_dataset(
    pairs: list[tuple[Path, Path]], cfg: UNet2DTrainConfig, *, training: bool
):
    """Build the training or validation TensorFlow dataset.

    Training patches remain stochastic. Validation patches are materialized once
    by default and then repeated unchanged for every epoch. This makes validation
    loss, Dice, and IoU directly comparable between epochs. A bounded LRU cache
    avoids reopening and normalizing the same OME-Zarr image for every patch.

    Args:
        pairs (list[tuple[Path, Path]]): Filesystem path used for pairs.
        cfg (UNet2DTrainConfig): Value specifying cfg for the operation.
        training (bool): Boolean flag controlling training.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = make_2d_dataset(
        ...     pairs=Path("path/to/resource"),
        ...     cfg=config,
        ...     training=True,
        ... )
    """
    rng = random.Random(cfg.seed + (0 if training else 10000))
    first_frame = ome_zarr_to_hwc_frames_2d(
        pairs[0][0], dataset=cfg.dataset, level=cfg.level
    )[0]
    channels = int(first_frame.shape[-1])
    cache: OrderedDict[tuple[str, str], tuple[list[np.ndarray], np.ndarray]] = OrderedDict()

    def load_pair(image_path: Path, mask_path: Path) -> tuple[list[np.ndarray], np.ndarray]:
        """Load pair from persistent storage.

        Args:
            image_path (Path): Filesystem path associated with image.
            mask_path (Path): Filesystem path associated with mask.

        Returns:
            tuple[list[np.ndarray], np.ndarray]: Collection containing the generated or selected values.

        Raises:
            ValueError: If the supplied inputs or runtime state violate the function's requirements.

        Example:
            >>> result = load_pair(image_path=Path("path/to/resource"), mask_path=Path("path/to/resource"))
        """
        key = (str(image_path), str(mask_path))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        frames = ome_zarr_to_hwc_frames_2d(
            image_path, dataset=cfg.dataset, level=cfg.level
        )
        normalized_frames = [normalize_image01(frame, cfg.normalize) for frame in frames]
        mask = read_binary_mask_2d(mask_path)
        for frame in normalized_frames:
            if frame.shape[:2] != mask.shape:
                raise ValueError(
                    f"Image/mask shape mismatch: {image_path} {frame.shape[:2]} vs "
                    f"{mask_path} {mask.shape}"
                )

        if cfg.cache_size > 0:
            cache[key] = (normalized_frames, mask)
            cache.move_to_end(key)
            while len(cache) > cfg.cache_size:
                cache.popitem(last=False)
        return normalized_frames, mask

    def sample_one() -> tuple[np.ndarray, np.ndarray]:
        """Return sample one for the supplied inputs.

        Returns:
            tuple[np.ndarray, np.ndarray]: Collection containing the generated or selected values.

        Example:
            >>> result = sample_one()
        """
        image_path, mask_path = pairs[rng.randrange(len(pairs))]
        frames, mask = load_pair(image_path, mask_path)
        image = frames[rng.randrange(len(frames))]
        x_patch, y_patch = sample_patch_2d(image, mask, cfg, rng)
        return x_patch.astype(np.float32), y_patch.astype(np.float32)[..., None]

    if not training and cfg.fixed_validation:
        patch_count = cfg.val_steps * cfg.batch
        x_values = np.empty(
            (patch_count, cfg.patch, cfg.patch, channels), dtype=np.float32
        )
        y_values = np.empty(
            (patch_count, cfg.patch, cfg.patch, 1), dtype=np.float32
        )
        for index in range(patch_count):
            x_values[index], y_values[index] = sample_one()
        dataset = tf.data.Dataset.from_tensor_slices((x_values, y_values))
        options = tf.data.Options()
        options.experimental_deterministic = True
        dataset = dataset.with_options(options).batch(
            cfg.batch, drop_remainder=True
        ).repeat()
        return dataset.prefetch(tf.data.AUTOTUNE), channels

    def generator():
        """Return generator for the supplied inputs.

        Returns:
            Any: Result produced by the operation.

        Example:
            >>> result = generator()
        """
        while True:
            yield sample_one()

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(cfg.patch, cfg.patch, channels), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(cfg.patch, cfg.patch, 1), dtype=tf.float32),
        ),
    )
    if training:
        dataset = dataset.shuffle(
            cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True
        )
    return dataset.batch(cfg.batch).prefetch(tf.data.AUTOTUNE), channels


def conv_block(x, filters: int, dropout: float = 0.0):
    """Two 3 x 3 convolutions, each followed by batch normalization and ReLU.

    Args:
        x (Any): Horizontal coordinate or numerical input value used by the operation.
        filters (int): Numerical value controlling filters.
        dropout (float): Numerical value controlling dropout. Defaults to ``0.0``.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = conv_block(x=..., filters=1)
    """
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int],
    base_filters: int = 16,
    dropout: float = 0.0,
) -> tf.keras.Model:
    """Build the four-stage 2D U-Net described in the Methods chapter.

    ``input_shape`` is normally ``(patch, patch, channels)``. ``base_filters``
    controls model width and therefore has the strongest effect on GPU memory
    after ``patch`` and ``batch``. ``dropout`` regularizes every convolutional
    block. Both parameters alter training behaviour, while ``base_filters`` also
    alters the saved model architecture.

    Args:
        input_shape (tuple[int, int, int]): Numerical value controlling input shape.
        base_filters (int): Numerical value controlling base filters. Defaults to ``16``.
        dropout (float): Numerical value controlling dropout. Defaults to ``0.0``.

    Returns:
        tf.keras.Model: Result produced by the operation.

    Example:
        >>> result = build_unet(input_shape=1)
    """
    inputs = tf.keras.Input(shape=input_shape)
    c1 = conv_block(inputs, base_filters, dropout)
    p1 = tf.keras.layers.MaxPool2D()(c1)
    c2 = conv_block(p1, base_filters * 2, dropout)
    p2 = tf.keras.layers.MaxPool2D()(c2)
    c3 = conv_block(p2, base_filters * 4, dropout)
    p3 = tf.keras.layers.MaxPool2D()(c3)
    c4 = conv_block(p3, base_filters * 8, dropout)
    p4 = tf.keras.layers.MaxPool2D()(c4)
    bottleneck = conv_block(p4, base_filters * 16, dropout)

    u4 = tf.keras.layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bottleneck)
    c5 = conv_block(tf.keras.layers.Concatenate()([u4, c4]), base_filters * 8, dropout)
    u3 = tf.keras.layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c5)
    c6 = conv_block(tf.keras.layers.Concatenate()([u3, c3]), base_filters * 4, dropout)
    u2 = tf.keras.layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c6)
    c7 = conv_block(tf.keras.layers.Concatenate()([u2, c2]), base_filters * 2, dropout)
    u1 = tf.keras.layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c7)
    c8 = conv_block(tf.keras.layers.Concatenate()([u1, c1]), base_filters, dropout)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return tf.keras.Model(inputs, outputs, name="UNet2DForeground")


def soft_dice_coef(y_true, y_pred, eps: float = 1e-6):
    """Return soft dice coef for the supplied inputs.

    Args:
        y_true (Any): Value specifying y true for the operation.
        y_pred (Any): Value specifying y pred for the operation.
        eps (float): Numerical value controlling eps. Defaults to ``1e-6``.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = soft_dice_coef(y_true=..., y_pred=...)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * intersection + eps) / (denominator + eps))


def dice_coef(y_true, y_pred, eps: float = 1e-6):
    """Binary Dice coefficient after thresholding probabilities at 0.5.

    Args:
        y_true (Any): Value specifying y true for the operation.
        y_pred (Any): Value specifying y pred for the operation.
        eps (float): Numerical value controlling eps. Defaults to ``1e-6``.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = dice_coef(y_true=..., y_pred=...)
    """
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * intersection + eps) / (denominator + eps))


def dice_loss(y_true, y_pred):
    """Differentiable soft Dice loss used in the combined training objective.

    Args:
        y_true (Any): Value specifying y true for the operation.
        y_pred (Any): Value specifying y pred for the operation.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = dice_loss(y_true=..., y_pred=...)
    """
    return 1.0 - soft_dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Return bce dice loss for the supplied inputs.

    Args:
        y_true (Any): Value specifying y true for the operation.
        y_pred (Any): Value specifying y pred for the operation.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = bce_dice_loss(y_true=..., y_pred=...)
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * tf.reduce_mean(bce) + 0.5 * dice_loss(y_true, y_pred)


def iou_coef(y_true, y_pred, eps: float = 1e-6):
    """Binary intersection over union after thresholding at 0.5.

    Args:
        y_true (Any): Value specifying y true for the operation.
        y_pred (Any): Value specifying y pred for the operation.
        eps (float): Numerical value controlling eps. Defaults to ``1e-6``.

    Returns:
        Any: Result produced by the operation.

    Example:
        >>> result = iou_coef(y_true=..., y_pred=...)
    """
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + eps) / (union + eps))


class LearningRateHistory(tf.keras.callbacks.Callback):
    """Add the effective optimizer learning rate to every history epoch."""

    def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]
        """Handle the epoch end callback event.

        Args:
            epoch (Any): Value specifying epoch for the operation.
            logs (Any): Value specifying logs for the operation. ``None`` selects the function's default behavior.

        Example:
            >>> instance = LearningRateHistory(...)
            >>> instance.on_epoch_end(epoch=...)
        """
        logs = logs if logs is not None else {}
        value = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        logs["learning_rate"] = float(value)


def _save_curve(
    history: dict[str, list[float]],
    train_key: str,
    validation_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> Path:
    """Save curve to persistent storage.

    Args:
        history (dict[str, list[float]]): Text value specifying history.
        train_key (str): Text value specifying train key.
        validation_key (str): Text value specifying validation key.
        ylabel (str): Text value specifying ylabel.
        title (str): Title displayed on the generated figure or report section.
        path (Path): Filesystem path to the required input or output resource.

    Returns:
        Path: Resolved or generated filesystem path.

    Example:
        >>> result = _save_curve(
        ...     history=0.5,
        ...     train_key="train_key",
        ...     validation_key="validation_key",
        ...     ylabel="ylabel",
        ...     title="title",
        ...     path=Path("path/to/resource"),
        ... )
    """
    figure = plt.figure(figsize=(7.2, 4.8), dpi=180)
    axis = figure.add_subplot(1, 1, 1)
    epochs = np.arange(1, len(history.get(train_key, [])) + 1)
    if train_key in history:
        axis.plot(epochs, history[train_key], label="training")
    if validation_key in history:
        axis.plot(epochs, history[validation_key], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def save_training_curves(history: tf.keras.callbacks.History, model_root: Path, dataset: str) -> dict[str, Path]:
    """Save separate thesis-ready loss, Dice, IoU, and learning-rate curves.

    Args:
        history (tf.keras.callbacks.History): Value specifying history for the operation.
        model_root (Path): Directory used for model.
        dataset (str): Dataset identifier that selects the supported acquisition and processing workflow, for example ``"2d_time"`` or ``"3d_data"``.

    Returns:
        dict[str, Path]: Resolved or generated filesystem path.

    Example:
        >>> result = save_training_curves(
        ...     history=...,
        ...     model_root=Path("path/to/resource"),
        ...     dataset="2d_time",
        ... )
    """
    model_root.mkdir(parents=True, exist_ok=True)
    values = {key: [float(v) for v in sequence] for key, sequence in history.history.items()}
    outputs = {
        "loss_curve": _save_curve(
            values,
            "loss",
            "val_loss",
            "BCE + soft Dice loss",
            f"U-Net loss: {dataset}",
            model_root / f"u_net_{dataset}_loss_curve.png",
        ),
        "dice_curve": _save_curve(
            values,
            "dice_coef",
            "val_dice_coef",
            "Binary Dice coefficient",
            f"U-Net Dice: {dataset}",
            model_root / f"u_net_{dataset}_dice_curve.png",
        ),
        "iou_curve": _save_curve(
            values,
            "iou_coef",
            "val_iou_coef",
            "Intersection over union",
            f"U-Net IoU: {dataset}",
            model_root / f"u_net_{dataset}_iou_curve.png",
        ),
    }

    if "learning_rate" in values:
        figure = plt.figure(figsize=(7.2, 4.8), dpi=180)
        axis = figure.add_subplot(1, 1, 1)
        epochs = np.arange(1, len(values["learning_rate"]) + 1)
        axis.plot(epochs, values["learning_rate"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Learning rate")
        axis.set_yscale("log")
        axis.set_title(f"U-Net learning rate: {dataset}")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        lr_path = model_root / f"u_net_{dataset}_learning_rate_curve.png"
        figure.savefig(lr_path)
        plt.close(figure)
        outputs["learning_rate_curve"] = lr_path

    history_json = model_root / f"u_net_{dataset}_history.json"
    history_json.write_text(json.dumps(values, indent=2), encoding="utf-8")
    outputs["history_json"] = history_json
    return outputs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV data to persistent storage.

    Args:
        path (Path): Filesystem path to the required input or output resource.
        rows (list[dict[str, Any]]): Text value specifying rows.

    Example:
        >>> _write_csv(path=Path("path/to/resource"), rows="rows")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        else:
            writer.writerow({"status": "no rows"})


def train_2d_binary_unet(cfg: UNet2DTrainConfig) -> dict[str, Path]:
    """Validate inputs, train one 2D U-Net, and save complete training evidence.

    Processing order
    ----------------
    1. Resolve paths and validate all configuration values.
    2. Pair each filtered OME-Zarr image with its reference mask.
    3. Validate shapes, channels, positive labels, and foreground/background.
    4. Split complete samples into training and validation subsets.
    5. Normalize complete images, then draw class-aware random patches.
    6. Train with Adam and the combined BCE plus soft-Dice objective.
    7. Save the best validation-loss model, final model, history, split list,
       architecture, and training curves.

    The learning-rate callback below halves the learning rate after six epochs
    without validation-loss improvement, down to 1e-6. Those callback constants
    may be changed in this function, but the initial learning rate is controlled
    by ``cfg.lr``.

    Args:
        cfg (UNet2DTrainConfig): Value specifying cfg for the operation.

    Returns:
        dict[str, Path]: Resolved or generated filesystem path.

    Raises:
        RuntimeError: If the supplied inputs or runtime state violate the function's requirements.
        ValueError: If the supplied inputs or runtime state violate the function's requirements.

    Example:
        >>> result = train_2d_binary_unet(cfg=config)
    """
    cfg.dataset = cfg.dataset.strip().lower()
    validate_train_config(cfg)
    if cfg.dataset not in DATASETS_2D:
        raise ValueError(f"Unsupported 2D dataset: {cfg.dataset}")
    cfg.image_root = Path(cfg.image_root or default_filtered_root(cfg.project_root, cfg.dataset))
    cfg.mask_root = Path(cfg.mask_root or _default_mask_root(cfg.project_root, cfg.dataset))
    cfg.model_root = ensure_dir(Path(cfg.model_root or _default_model_root(cfg.project_root, cfg.dataset)))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    pairs = list_2d_training_pairs(cfg)
    if not pairs:
        raise RuntimeError(
            f"No training pairs found. Images: {cfg.image_root} | Masks: {cfg.mask_root}"
        )

    validation_rows = validate_2d_training_pairs(pairs, cfg)
    validation_csv = cfg.model_root / f"u_net_{cfg.dataset}_input_validation.csv"
    _write_csv(validation_csv, validation_rows)
    failures = [row for row in validation_rows if row["status"] != "PASS"]
    if failures:
        examples = "; ".join(f"{row['sample']}: {row['issue']}" for row in failures[:5])
        raise RuntimeError(
            f"U-Net input validation failed for {len(failures)} sample(s). "
            f"See {validation_csv}. Examples: {examples}"
        )

    train_pairs, validation_pairs = split_pairs(pairs, cfg)
    split_rows = [
        {
            "dataset": cfg.dataset,
            "split": split,
            "sample": image_path.parent.name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        for split, subset in (("train", train_pairs), ("validation", validation_pairs))
        for image_path, mask_path in subset
    ]
    split_csv = cfg.model_root / f"u_net_{cfg.dataset}_split_manifest.csv"
    _write_csv(split_csv, split_rows)

    train_dataset, channels = make_2d_dataset(train_pairs, cfg, training=True)
    validation_dataset, _ = make_2d_dataset(validation_pairs, cfg, training=False)

    model = build_unet(
        (cfg.patch, cfg.patch, channels),
        base_filters=cfg.base_filters,
        dropout=cfg.dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef],
    )

    best_model = cfg.model_root / f"u_net_{cfg.dataset}_best.keras"
    final_model = cfg.model_root / f"u_net_{cfg.dataset}_final.keras"
    history_csv = cfg.model_root / f"u_net_{cfg.dataset}_history.csv"
    summary_json = cfg.model_root / f"u_net_{cfg.dataset}_training_summary.json"
    architecture_txt = cfg.model_root / f"u_net_{cfg.dataset}_architecture.txt"

    with architecture_txt.open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    # Validation loss selects the deployable checkpoint. ReduceLROnPlateau
    # lowers the step size when learning stalls. Increasing ``patience`` waits
    # longer before reducing the rate; a smaller ``factor`` makes a stronger
    # reduction; and a smaller ``min_lr`` permits finer late-stage updates.
    callbacks = [
        LearningRateHistory(),
        tf.keras.callbacks.ModelCheckpoint(
            str(best_model), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv)),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_steps=cfg.val_steps,
        callbacks=callbacks,
    )
    model.save(final_model)
    curve_outputs = save_training_curves(history, cfg.model_root, cfg.dataset)

    summary = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "channels": channels,
        "pair_count": len(pairs),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "normalization_scope": "complete image, independently per channel, before patch extraction",
        "normalization_percentiles": [1.0, 99.8],
        "validation_sampling": (
            "fixed patches reused every epoch"
            if cfg.fixed_validation
            else "stochastic patches resampled continuously"
        ),
        "fixed_validation_patch_count": (
            cfg.val_steps * cfg.batch if cfg.fixed_validation else None
        ),
        "best_model": str(best_model),
        "final_model": str(final_model),
        "history_csv": str(history_csv),
        "split_manifest": str(split_csv),
        "input_validation": str(validation_csv),
        "curves": {name: str(path) for name, path in curve_outputs.items()},
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "best_model": best_model,
        "final_model": final_model,
        "history_csv": history_csv,
        "summary": summary_json,
        "split_manifest": split_csv,
        "input_validation": validation_csv,
        "architecture": architecture_txt,
        **curve_outputs,
    }


def train_2d_time_unet(cfg: UNet2DTrainConfig | None = None) -> dict[str, Path]:
    """Train two-dimensional data time U-Net result using the supplied data and configuration.

    Args:
        cfg (UNet2DTrainConfig | None): Value specifying cfg for the operation. ``None`` selects the function's default behavior.

    Returns:
        dict[str, Path]: Resolved or generated filesystem path.

    Example:
        >>> result = train_2d_time_unet()
    """
    cfg = cfg or UNet2DTrainConfig()
    cfg.dataset = "2d_time"
    return train_2d_binary_unet(cfg)
