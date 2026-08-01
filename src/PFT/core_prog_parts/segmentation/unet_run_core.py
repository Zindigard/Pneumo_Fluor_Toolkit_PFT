"""Core inference and quantitative evaluation for the 2D U-Net.

The module loads a trained dataset-specific model, applies the same complete-
image normalization used during training, predicts foreground probabilities in
non-overlapping tiles, and reconstructs the original image dimensions. It can
save the binary mask, probability map, and a non-normalized background-
suppressed OME-Zarr image. Pixels inside the predicted foreground mask retain
their filtered intensities, whereas pixels outside the mask are attenuated by
98% by default and therefore retain 2% of their original intensity. 

The most important adjustable values are defined in :class:`UNetRunConfig`.
Inference is not a second training stage. Parameters such as ``threshold``,
``predict_batch_size``, and ``outside_mask_depletion`` can be changed without
retraining, while ``patch`` and ``normalize`` must remain consistent with the
trained model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tensorflow as tf

from PFT.core_prog_parts.common_paths import ensure_dir, find_project_root
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, normalize_axes
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (
    DATASETS_2D,
    bce_dice_loss,
    default_filtered_root,
    dice_coef,
    dice_loss,
    iou_coef,
    normalize_image01,
    ome_zarr_to_hwc_frames_2d,
    read_binary_mask_2d,
    soft_dice_coef,
)


BLUE_FLUORESCENCE_CMAP = LinearSegmentedColormap.from_list(
    "pft_blue_fluorescence",
    [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 1.0))],
)


@dataclass
class UNetRunConfig:
    """Configuration for 2D U-Net semantic foreground inference.

    Parameter guide
    ---------------
    project_root:
        Project directory used for default paths. It does not change predictions.
    dataset:
        Selects the matching model and expected channel layout. Use the same
        dataset type used during training.
    level:
        OME-Zarr pyramid level. Normally this must equal the training level.
        A higher level is faster and uses less memory, but changes image scale
        and can invalidate a model trained at full resolution.
    patch:
        Tile width and height. It must match the spatial input size of the saved
        model, normally 256. A different value usually causes an input-shape
        error and should not be used unless the model was trained accordingly.
    threshold:
        Converts probabilities into the final binary mask. Lowering it marks
        more pixels as foreground, usually increasing sensitivity/recall but
        also false positives. Raising it produces a stricter mask, usually
        increasing precision but missing weaker foreground regions. ``0.5`` is
        the neutral starting point. Threshold tuning should be evaluated on
        labelled validation data using IoU and Dice.
    normalize:
        Must match training. ``"percentile"`` maps P1 to 0 and P99.8 to 1 per
        channel. Changing normalization after training can shift the intensity
        distribution and substantially degrade predictions.
    predict_batch_size:
        Number of tiles predicted together. Increasing it improves GPU
        throughput until memory is saturated. Decrease it when inference causes
        an out-of-memory error. It should not materially change predictions.
    outside_mask_depletion:
        Fraction of the original filtered intensity removed outside the predicted
        mask. The default ``0.98`` removes 98% and leaves a residual intensity
        fraction of ``0.02``. Foreground pixels are not rescaled.

        Approximate effect of changing this value:

        * ``0.00``: no background attenuation; the saved image equals the
          filtered input.
        * ``0.90``: 90% attenuation; 10% of outside-mask intensity remains.
        * ``0.98``: requested default; 2% remains.
        * ``1.00``: complete removal; outside-mask pixels become zero.

        Increasing depletion generally raises measured SNR because the residual
        background mean and standard deviation decrease. However, it also makes
        false-negative mask errors more consequential because real signal that
        falls outside the predicted mask is strongly attenuated. Therefore, the
        value should be fixed before final quantitative analysis and reported in
        Methods. It changes post-processing only and does not require retraining.

    model_path, input_root, out_root, raw_root, mask_root:
        Optional path overrides. ``raw_root`` is used for before-filtering SNR,
        and ``mask_root`` supplies references for IoU, Dice, and ROI-SNR. Path
        changes do not alter the neural-network output for the same input.
    save_probability:
        Save the floating-point foreground probability map. Disabling it reduces
        disk use but removes the data needed for later threshold experiments.
    save_foreground_image:
        Save the non-normalized, background-suppressed OME-Zarr image. The input
        dtype and axis order are retained. Pixels inside the predicted mask keep
        their original filtered values.
        
    compute_metrics:
        Calculate IoU, Dice, and available ROI-SNR values when references exist.
        SNR before is calculated from the raw image and SNR after from the unmasked
        filtered input.
    strict_metrics:
        When ``False`` (default), samples without a hand-labelled reference mask
        are still processed and saved, but their quantitative metrics are marked
        as unavailable. When ``True``, a missing reference mask raises an error.
        The non-strict default permits complete-dataset inference while keeping
        metric coverage explicit in ``unet_2d_metrics_coverage.csv``.
    epsilon:
        Small numerical constant used only to prevent division by zero in
        metrics. It should remain very small and does not normally need tuning.
    """

    # Dataset, scale, and model-input compatibility.
    project_root: Path = find_project_root(Path(__file__).resolve())
    dataset: str = "2d_time"
    level: int = 0
    patch: int = 256
    threshold: float = 0.5
    normalize: str = "percentile"
    predict_batch_size: int = 8

    # Post-processing. A value of 0.98 leaves 2% outside the predicted mask.
    outside_mask_depletion: float = 0.98

    # Optional path overrides.
    model_path: Path | None = None
    input_root: Path | None = None
    out_root: Path | None = None
    raw_root: Path | None = None
    mask_root: Path | None = None

    # Output and evaluation controls.
    save_probability: bool = True
    save_foreground_image: bool = True
    compute_metrics: bool = True
    strict_metrics: bool = False
    epsilon: float = 1e-12


def validate_run_config(cfg: UNetRunConfig) -> None:
    """Validate inference parameters before loading the model or data."""
    if cfg.patch <= 0 or cfg.patch % 16 != 0:
        raise ValueError("patch must be a positive multiple of 16 for the trained U-Net")
    if not 0.0 <= cfg.threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if cfg.predict_batch_size <= 0:
        raise ValueError("predict_batch_size must be positive")
    if not 0.0 <= cfg.outside_mask_depletion <= 1.0:
        raise ValueError("outside_mask_depletion must be in [0,1]")
    if cfg.epsilon <= 0:
        raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class UNetInferenceOutput:
    dataset: str
    sample: str
    input_zarr: Path
    mask_zarr: Path
    probability_zarr: Path | None
    foreground_zarr: Path | None
    preview_png: Path
    metrics_json: Path | None


def find_omezarr_dirs(root: Path) -> list[Path]:
    """Return sample image.ome.zarr folders below a selected input root."""
    root = Path(root)
    direct = sorted(p for p in root.glob("*/image.ome.zarr") if p.is_dir())
    return direct if direct else sorted(p for p in root.rglob("*.ome.zarr") if p.is_dir())


def default_model_path(project_root: Path, dataset: str) -> Path:
    return project_root / "models" / f"u_net_{dataset}" / f"u_net_{dataset}_best.keras"


def default_input_root(project_root: Path, dataset: str) -> Path:
    return default_filtered_root(project_root, dataset)


def default_out_root(project_root: Path, dataset: str) -> Path:
    return project_root / "results" / "U-net" / dataset


def default_raw_root(project_root: Path, dataset: str) -> Path:
    return project_root / "results" / "img" / dataset


def default_mask_root(project_root: Path, dataset: str) -> Path:
    return project_root / "results" / "training_files" / "U-net" / dataset


def load_unet_model(model_path: Path) -> tf.keras.Model:
    """Load a trained U-Net with all custom loss and metric objects."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_loss": dice_loss,
            "soft_dice_coef": soft_dice_coef,
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
        },
    )


def pad_to_patch(img_hwc: np.ndarray, patch: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad an HWC image to exact multiples of the patch size."""
    height, width, _ = img_hwc.shape
    pad_h = (patch - height % patch) % patch
    pad_w = (patch - width % patch) % patch
    if pad_h == 0 and pad_w == 0:
        return img_hwc, (0, 0)
    padded = np.pad(img_hwc, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return padded, (pad_h, pad_w)


def predict_2d_tiled_probability(
    model: tf.keras.Model,
    img_hwc: np.ndarray,
    cfg: UNetRunConfig,
) -> np.ndarray:
    """Normalize a complete image, predict tiles, and reconstruct probabilities.

    Normalization occurs before tiling so each tile uses the same intensity
    reference. ``cfg.predict_batch_size`` controls only throughput and memory.
    ``cfg.patch`` must match the trained model input shape. The returned array is
    continuous in ``[0, 1]`` and can therefore be re-thresholded later without
    repeating neural-network inference.
    """
    image_normalized = normalize_image01(img_hwc, cfg.normalize).astype(np.float32)
    padded, (pad_h, pad_w) = pad_to_patch(image_normalized, cfg.patch)
    height, width, _ = padded.shape
    probability = np.zeros((height, width), dtype=np.float32)

    tiles: list[np.ndarray] = []
    coordinates: list[tuple[int, int]] = []
    for y0 in range(0, height, cfg.patch):
        for x0 in range(0, width, cfg.patch):
            tiles.append(padded[y0 : y0 + cfg.patch, x0 : x0 + cfg.patch, :])
            coordinates.append((y0, x0))

    for start in range(0, len(tiles), max(1, cfg.predict_batch_size)):
        batch_tiles = np.stack(
            tiles[start : start + max(1, cfg.predict_batch_size)], axis=0
        ).astype(np.float32)
        predictions = np.asarray(model.predict(batch_tiles, verbose=0))
        for local_index, prediction in enumerate(predictions):
            y0, x0 = coordinates[start + local_index]
            probability[y0 : y0 + cfg.patch, x0 : x0 + cfg.patch] = prediction[..., 0]

    if pad_h:
        probability = probability[:-pad_h, :]
    if pad_w:
        probability = probability[:, :-pad_w]
    return probability


def predict_2d_tiled(model: tf.keras.Model, img_hwc: np.ndarray, cfg: UNetRunConfig) -> np.ndarray:
    """Return the binary foreground mask at ``cfg.threshold``.

    Decrease the threshold to include weaker foreground probabilities; increase
    it to require stronger confidence. Threshold changes do not require
    retraining when a probability map is available.
    """
    return (predict_2d_tiled_probability(model, img_hwc, cfg) >= cfg.threshold).astype(np.uint8)


def _sample_name_from_zarr(zarr_path: Path) -> str:
    return zarr_path.parent.name if zarr_path.name == "image.ome.zarr" else zarr_path.stem.replace(".ome", "")


def _load_array_and_axes(path: Path, level: int) -> tuple[np.ndarray, str]:
    array, axes = load_ome_zarr(path, level=level, as_numpy=True)
    return np.asarray(array), normalize_axes(axes)


def _mask_stack_and_axes(masks: list[np.ndarray]) -> tuple[np.ndarray, str]:
    if len(masks) == 1:
        return masks[0].astype(np.uint8), "yx"
    return np.stack(masks, axis=0).astype(np.uint8), "tyx"


def _probability_stack_and_axes(probabilities: list[np.ndarray]) -> tuple[np.ndarray, str]:
    if len(probabilities) == 1:
        return probabilities[0].astype(np.float32), "yx"
    return np.stack(probabilities, axis=0).astype(np.float32), "tyx"


def _broadcast_masks_to_input(mask_stack: np.ndarray, input_axes: str, input_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast YX or TYX predictions over the channel axis of the input array."""
    input_axes = normalize_axes(input_axes)
    if "y" not in input_axes or "x" not in input_axes:
        raise ValueError(f"Input axes do not contain YX: {input_axes}")
    if mask_stack.ndim == 2:
        mask_stack = mask_stack[None, ...]

    time_count = mask_stack.shape[0]
    if "t" in input_axes:
        expected_time = int(input_shape[input_axes.index("t")])
        if time_count != expected_time:
            raise ValueError(
                f"Predicted frame count {time_count} does not match input T={expected_time}"
            )
    elif time_count != 1:
        raise ValueError("Multiple predicted frames cannot be applied to input without T axis")

    reshape: list[int] = []
    for axis, size in zip(input_axes, input_shape):
        if axis == "t":
            reshape.append(time_count)
        elif axis == "y":
            reshape.append(mask_stack.shape[-2])
        elif axis == "x":
            reshape.append(mask_stack.shape[-1])
        else:
            reshape.append(1)

    source = mask_stack if "t" in input_axes else mask_stack[0]
    if "t" not in input_axes:
        reshape = [value for axis, value in zip(input_axes, reshape)]
    return np.asarray(source).reshape(tuple(reshape))


def apply_foreground_mask_to_input(
    input_array: np.ndarray,
    input_axes: str,
    masks: list[np.ndarray],
    outside_mask_depletion: float = 0.98,
) -> np.ndarray:
    """Attenuate intensities outside the predicted foreground mask.

    Pixels inside the predicted mask are copied without rescaling. Pixels
    outside the mask are multiplied by ``1 - outside_mask_depletion``. With the
    default depletion of ``0.98``, 2% of the original filtered intensity remains
    outside the mask.

    The operation is performed on the original, non-normalized filtered array.
    Integer inputs are rounded to the nearest representable value and returned
    in the original integer dtype. Floating-point inputs retain their original
    floating dtype. Consequently, the saved OME-Zarr has the same axes and dtype
    as the filtered source rather than the temporary normalized U-Net tensor.

    Parameters
    ----------
    input_array:
        Original filtered image array loaded from OME-Zarr.
    input_axes:
        Axis order associated with ``input_array``.
    masks:
        One predicted binary YX mask per input frame.
    outside_mask_depletion:
        Removed fraction outside the mask. ``0`` leaves the image unchanged,
        ``0.98`` leaves 2%, and ``1`` sets outside-mask pixels to zero.
    """
    if not 0.0 <= outside_mask_depletion <= 1.0:
        raise ValueError("outside_mask_depletion must be in [0,1]")

    mask_stack = np.stack(masks, axis=0)
    broadcast = _broadcast_masks_to_input(mask_stack, input_axes, input_array.shape)
    residual_fraction = 1.0 - float(outside_mask_depletion)

    working = np.asarray(input_array, dtype=np.float64)
    attenuated = np.where(broadcast > 0, working, working * residual_fraction)

    source_dtype = np.asarray(input_array).dtype
    if np.issubdtype(source_dtype, np.integer):
        limits = np.iinfo(source_dtype)
        attenuated = np.rint(attenuated)
        attenuated = np.clip(attenuated, limits.min, limits.max)
    return attenuated.astype(source_dtype, copy=False)


def binary_iou(reference: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-12) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    intersection = int(np.count_nonzero(reference & prediction))
    union = int(np.count_nonzero(reference | prediction))
    return float((intersection + epsilon) / (union + epsilon))


def binary_dice(reference: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-12) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    intersection = int(np.count_nonzero(reference & prediction))
    denominator = int(np.count_nonzero(reference) + np.count_nonzero(prediction))
    return float((2 * intersection + epsilon) / (denominator + epsilon))


def roi_snr(
    image: np.ndarray,
    reference_mask: np.ndarray,
    epsilon: float = 1e-12,
    *,
    relative_sd_floor: float = 1e-8,
) -> float:
    """Compute ROI SNR and reject zero or numerically negligible background SD.

    Formula: ``(mean(signal) - mean(background)) / SD(background)``.
    The same hand-labelled reference mask must define both regions for every
    compared image. ``epsilon`` remains a numerical safeguard only; it must not
    turn a zero-variance background into an enormous finite SNR.
    """
    image = np.asarray(image, dtype=np.float64)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    if image.shape != reference_mask.shape:
        raise ValueError(
            f"ROI SNR image/mask shape mismatch: {image.shape} versus "
            f"{reference_mask.shape}"
        )

    finite = np.isfinite(image)
    signal = image[reference_mask & finite]
    background = image[(~reference_mask) & finite]
    if signal.size == 0:
        raise ValueError("ROI SNR requires at least one finite signal pixel")
    if background.size < 2:
        raise ValueError("ROI SNR requires at least two finite background pixels")

    signal_mean = float(np.mean(signal))
    background_mean = float(np.mean(background))
    background_sd = float(np.std(background, ddof=1))

    finite_values = image[finite]
    dynamic_range = float(np.ptp(finite_values)) if finite_values.size else 0.0
    numerical_scale = max(
        dynamic_range,
        abs(signal_mean),
        abs(background_mean),
        abs(signal_mean - background_mean),
        1.0,
    )
    sd_floor = max(float(epsilon), float(relative_sd_floor) * numerical_scale)
    if not np.isfinite(background_sd) or background_sd <= sd_floor:
        return float("nan")

    return float((signal_mean - background_mean) / background_sd)


def _normalized_hwc_with_reference(
    target_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> np.ndarray:
    """Normalize target channels using percentile limits from a reference image.

    This is used only for QC visualization. In particular, a completely
    zeroed background can make P99.8 of the sparse final image equal zero.
    Deriving limits from the filtered input prevents the valid foreground from
    being displayed as an entirely black panel.
    """
    target = np.asarray(target_hwc, dtype=np.float32)
    reference = np.asarray(reference_hwc, dtype=np.float32)
    if target.ndim == 2:
        target = target[..., None]
    if reference.ndim == 2:
        reference = reference[..., None]
    if target.shape != reference.shape:
        raise ValueError(
            f"Target/reference display shapes differ: {target.shape} versus {reference.shape}"
        )

    normalized = np.empty_like(target, dtype=np.float32)
    for channel in range(target.shape[-1]):
        reference_plane = reference[..., channel]
        lo = float(np.percentile(reference_plane, p_low))
        hi = float(np.percentile(reference_plane, p_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            finite = reference_plane[np.isfinite(reference_plane)]
            if finite.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
                if hi <= lo:
                    hi = lo + 1.0
        normalized[..., channel] = np.clip(
            (target[..., channel] - lo) / (hi - lo + 1e-8),
            0.0,
            1.0,
        )
    return normalized


def _display_from_normalized_hwc(normalized_hwc: np.ndarray) -> np.ndarray:
    """Compose an already normalized HWC fluorescence image for display."""
    normalized = np.asarray(normalized_hwc, dtype=np.float32)
    if normalized.ndim == 2:
        return normalized
    if normalized.shape[-1] == 1:
        return normalized[..., 0]
    if normalized.shape[-1] == 2:
        blue = normalized[..., 0]
        green = normalized[..., 1]
        red = np.zeros_like(blue)
        return np.stack([red, green, blue], axis=-1)
    return normalized[..., :3]


def _display_image(hwc: np.ndarray) -> np.ndarray:
    return _display_from_normalized_hwc(normalize_image01(hwc, "percentile"))


def _display_image_with_reference(
    target_hwc: np.ndarray,
    reference_hwc: np.ndarray,
) -> np.ndarray:
    """Display target with percentile limits derived from reference."""
    return _display_from_normalized_hwc(
        _normalized_hwc_with_reference(target_hwc, reference_hwc)
    )


def _save_preview(
    raw_hwc: np.ndarray | None,
    filtered_hwc: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    suppressed_hwc: np.ndarray,
    reference: np.ndarray | None,
    outside_mask_depletion: float,
    path: Path,
    sample: str,
) -> Path:
    """Save inference QC including the actual non-normalized final output.

    Display normalization is applied only for visualization. It does not alter
    the saved OME-Zarr or the arrays used for SNR calculation. For the one-channel
    2d_time dataset, every image, probability, and mask panel uses a black-to-blue
    fluorescence colour map. No grayscale or magma panel is used in that QC figure.
    """
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), dpi=160)
    depletion_percent = 100.0 * outside_mask_depletion
    residual_percent = 100.0 * (1.0 - outside_mask_depletion)
    filtered_display = _display_image(filtered_hwc)
    # Use the filtered input's percentile limits for the saved result. This is
    # essential when 100% outside-mask depletion makes almost all output pixels
    # exactly zero; independent P99.8 normalization can otherwise yield a fully
    # black but misleading QC panel.
    suppressed_display = _display_image_with_reference(
        suppressed_hwc,
        filtered_hwc,
    )
    residual_display = (
        suppressed_display[..., 0]
        if suppressed_display.ndim == 3
        else suppressed_display
    )
    residual_display = np.where(prediction > 0, 0.0, residual_display)

    single_channel = int(filtered_hwc.shape[-1]) == 1
    scalar_cmap = BLUE_FLUORESCENCE_CMAP if single_channel else "gray"
    probability_cmap = BLUE_FLUORESCENCE_CMAP if single_channel else "magma"

    panels: list[tuple[str, np.ndarray, object | None]] = [
        (
            "Raw image",
            _display_image(raw_hwc if raw_hwc is not None else filtered_hwc),
            scalar_cmap if single_channel else None,
        ),
        (
            "Filtered input, display normalized",
            filtered_display,
            scalar_cmap if single_channel else None,
        ),
        (
            "Exact normalized U-Net input",
            _display_image(normalize_image01(filtered_hwc, "percentile")),
            scalar_cmap if single_channel else None,
        ),
        ("Foreground probability", probability, probability_cmap),
        ("Predicted foreground mask", prediction, scalar_cmap),
        (
            f"Saved output: {depletion_percent:.1f}% outside-mask depletion",
            suppressed_display,
            scalar_cmap if single_channel else None,
        ),
        (
            "Reference mask" if reference is not None else "No reference mask",
            reference if reference is not None else np.zeros_like(prediction),
            scalar_cmap,
        ),
        (
            f"Residual outside mask: {residual_percent:.1f}%",
            residual_display,
            scalar_cmap,
        ),
    ]
    for axis, (title, image, cmap) in zip(axes.ravel(), panels):
        if single_channel:
            axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"2D U-Net inference QC: {sample}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def run_2d_unet_on_omezarr(
    zarr_path: Path,
    cfg: UNetRunConfig,
    model: tf.keras.Model | None = None,
) -> tuple[UNetInferenceOutput, list[dict[str, Any]]]:
    """Run one complete 2D sample and save the final background-suppressed image.

    Processing order is intentionally fixed:

    1. Normalize only the temporary network input and predict probabilities.
    2. Threshold probabilities to obtain binary semantic masks.
    3. Apply the masks to the original non-normalized filtered OME-Zarr array.
    4. Remove ``cfg.outside_mask_depletion`` of the intensity outside masks.
    5. Save the attenuated array as OME-Zarr with source axes and dtype.
    6. Calculate ROI-SNR from raw and unmasked filtered images using the same
       hand-labelled reference mask.

    The attenuated output is excluded from SNR because outside-mask suppression
    changes the background distribution and can create artificial SNR inflation.
    """
    zarr_path = Path(zarr_path)
    sample = _sample_name_from_zarr(zarr_path)
    model = model or load_unet_model(Path(cfg.model_path))

    filtered_frames = ome_zarr_to_hwc_frames_2d(
        zarr_path, dataset=cfg.dataset, level=cfg.level
    )
    probabilities = [predict_2d_tiled_probability(model, frame, cfg) for frame in filtered_frames]
    masks = [(probability >= cfg.threshold).astype(np.uint8) for probability in probabilities]

    input_array, input_axes = _load_array_and_axes(zarr_path, cfg.level)
    suppressed_array = apply_foreground_mask_to_input(
        input_array,
        input_axes,
        masks,
        outside_mask_depletion=cfg.outside_mask_depletion,
    )

    # Numerical integrity checks: the mask application must never alter values
    # inside the predicted foreground. With complete depletion, every value
    # outside the mask must be exactly zero. These checks verify the quantitative
    # array independently of any display normalization.
    mask_stack_for_check = np.stack(masks, axis=0)
    broadcast_for_check = _broadcast_masks_to_input(
        mask_stack_for_check,
        input_axes,
        input_array.shape,
    )
    inside_for_check = np.broadcast_to(
        broadcast_for_check > 0,
        input_array.shape,
    )
    if not np.array_equal(
        suppressed_array[inside_for_check],
        np.asarray(input_array)[inside_for_check],
    ):
        raise RuntimeError(
            f"Foreground-intensity integrity check failed for {sample}: "
            "values inside the predicted mask were modified."
        )
    if cfg.outside_mask_depletion == 1.0 and np.any(
        suppressed_array[~inside_for_check] != 0
    ):
        raise RuntimeError(
            f"Zero-background integrity check failed for {sample}: "
            "nonzero values remain outside the predicted mask."
        )

    sample_dir = ensure_dir(Path(cfg.out_root) / sample)
    mask_array, mask_axes = _mask_stack_and_axes(masks)
    probability_array, probability_axes = _probability_stack_and_axes(probabilities)

    mask_zarr = save_ome_zarr(
        sample_dir / "pred_mask.ome.zarr",
        mask_array,
        mask_axes,
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
        extra_attrs={
            "pft_processing": {
                "operation": "unet_semantic_foreground_prediction",
                "dataset": cfg.dataset,
                "source": str(zarr_path),
                "threshold": cfg.threshold,
                "normalization": "per-image per-channel percentile P1-P99.8",
            }
        },
    )

    probability_zarr: Path | None = None
    if cfg.save_probability:
        probability_zarr = save_ome_zarr(
            sample_dir / "foreground_probability.ome.zarr",
            probability_array,
            probability_axes,
            overwrite=True,
            pyramid_3d=False,
            pyramid_max_layer=0,
        )

    foreground_zarr: Path | None = None
    if cfg.save_foreground_image:
        foreground_zarr = save_ome_zarr(
            sample_dir / "foreground_filtered.ome.zarr",
            suppressed_array,
            input_axes,
            overwrite=True,
            pyramid_3d=False,
            pyramid_max_layer=0,
            extra_attrs={
                "pft_processing": {
                    "operation": "unet_background_attenuation",
                    "source_filtered": str(zarr_path),
                    "predicted_mask": str(mask_zarr),
                    "outside_mask_depletion_fraction": cfg.outside_mask_depletion,
                    "outside_mask_residual_fraction": 1.0 - cfg.outside_mask_depletion,
                    "inside_mask_intensity_scale": 1.0,
                    "normalization_applied_to_saved_intensities": False,
                    "saved_dtype": str(input_array.dtype),
                }
            },
        )

    raw_path = Path(cfg.raw_root) / sample / "image.ome.zarr"
    mask_path = Path(cfg.mask_root) / sample / "mask.tif"
    raw_frames = (
        ome_zarr_to_hwc_frames_2d(raw_path, dataset=cfg.dataset, level=cfg.level)
        if raw_path.is_dir()
        else []
    )
    reference = read_binary_mask_2d(mask_path) if mask_path.is_file() else None

    # Reload the saved output before SNR calculation. This guarantees that SNR
    # is calculated from the exact stored dtype and rounded integer values.
    if foreground_zarr is not None:
        suppressed_frames = ome_zarr_to_hwc_frames_2d(
            foreground_zarr,
            dataset=cfg.dataset,
            level=0,
        )
    else:
        # This branch is available only when metrics are disabled. It supports
        # QC previews without writing the optional output image.
        suppressed_frames = [
            apply_foreground_mask_to_input(
                filtered_frame,
                "yxc",
                [prediction],
                outside_mask_depletion=cfg.outside_mask_depletion,
            )
            for filtered_frame, prediction in zip(filtered_frames, masks)
        ]

    if len(suppressed_frames) != len(filtered_frames):
        raise ValueError(
            f"Saved/filtered frame-count mismatch for {sample}: "
            f"saved={len(suppressed_frames)}, filtered={len(filtered_frames)}"
        )

    metric_rows: list[dict[str, Any]] = []
    if cfg.compute_metrics:
        if reference is None:
            message = (
                f"Reference mask is missing for {sample}; inference outputs were saved, "
                "but IoU, Dice, and ROI-SNR metrics were skipped."
            )
            if cfg.strict_metrics:
                raise FileNotFoundError(f"{message} Expected: {mask_path}")
            print(f"[METRICS SKIPPED] {message}")
        else:
            if len(raw_frames) != len(filtered_frames):
                raise ValueError(
                    f"Raw/filtered frame-count mismatch for {sample}: raw={len(raw_frames)}, "
                    f"filtered={len(filtered_frames)}"
                )
            if reference.shape != filtered_frames[0].shape[:2]:
                raise ValueError(
                    f"Reference mask shape {reference.shape} does not match image YX "
                    f"{filtered_frames[0].shape[:2]} for {sample}"
                )

            for frame_index, (raw_frame, filtered_frame, suppressed_frame, prediction) in enumerate(
                zip(raw_frames, filtered_frames, suppressed_frames, masks)
            ):
                iou = binary_iou(reference, prediction, cfg.epsilon)
                dice = binary_dice(reference, prediction, cfg.epsilon)
                for channel_index in range(filtered_frame.shape[-1]):
                    snr_before = roi_snr(raw_frame[..., channel_index], reference, cfg.epsilon)
                    snr_after = roi_snr(
                        filtered_frame[..., channel_index], reference, cfg.epsilon
                    )
                    metric_rows.append(
                        {
                            "dataset": cfg.dataset,
                            "sample": sample,
                            "frame_index": frame_index,
                            "channel_index": channel_index,
                            "iou": iou,
                            "dice": dice,
                            "snr_before": snr_before,
                            "snr_after": snr_after,
                            "delta_snr": snr_after - snr_before,
                            "outside_mask_depletion": cfg.outside_mask_depletion,
                            "outside_mask_residual": 1.0 - cfg.outside_mask_depletion,
                            "raw_zarr": str(raw_path),
                            "filtered_zarr": str(zarr_path),
                            "reference_mask": str(mask_path),
                            "predicted_mask": str(mask_zarr),
                            "snr_before_source": str(raw_path),
                            "snr_after_source": str(zarr_path),
                            "snr_after_source_is_unmasked": True,
                            "snr_mask_source": str(mask_path),
                            "background_suppressed_output": (
                                str(foreground_zarr) if foreground_zarr is not None else ""
                            ),
                            # Retained for compatibility with earlier result readers.
                            "foreground_output": (
                                str(foreground_zarr) if foreground_zarr is not None else ""
                            ),
                        }
                    )

    preview_png = _save_preview(
        raw_frames[0] if raw_frames else None,
        filtered_frames[0],
        probabilities[0],
        masks[0],
        suppressed_frames[0],
        reference,
        cfg.outside_mask_depletion,
        sample_dir / "unet_inference_qc.png",
        sample,
    )

    metrics_json: Path | None = None
    if metric_rows:
        metrics_json = sample_dir / "unet_metrics.json"
        per_frame_overlap = {
            int(row["frame_index"]): {
                "frame_index": int(row["frame_index"]),
                "iou": float(row["iou"]),
                "dice": float(row["dice"]),
            }
            for row in metric_rows
        }
        overlap_rows = [per_frame_overlap[index] for index in sorted(per_frame_overlap)]
        iou_values = [float(row["iou"]) for row in overlap_rows]
        dice_values = [float(row["dice"]) for row in overlap_rows]
        before_values = [float(row["snr_before"]) for row in metric_rows]
        after_values = [float(row["snr_after"]) for row in metric_rows]
        iou_mean, iou_sd = _mean_std(iou_values)
        dice_mean, dice_sd = _mean_std(dice_values)
        before_mean, before_sd = _mean_std(before_values)
        after_mean, after_sd = _mean_std(after_values)
        metrics_json.write_text(
            json.dumps(
                {
                    "dataset": cfg.dataset,
                    "sample": sample,
                    "threshold": cfg.threshold,
                    "outside_mask_depletion": cfg.outside_mask_depletion,
                    "outside_mask_residual": 1.0 - cfg.outside_mask_depletion,
                    "snr_before_source": str(raw_path),
                    "snr_after_source": str(zarr_path),
                    "snr_after_source_is_unmasked": True,
                    "snr_reference_mask": str(mask_path),
                    "background_suppressed_output_excluded_from_snr": (
                        str(foreground_zarr) if foreground_zarr is not None else None
                    ),
                    "iou_mean": iou_mean,
                    "iou_standard_deviation": iou_sd,
                    "dice_mean": dice_mean,
                    "dice_standard_deviation": dice_sd,
                    "snr_before_mean": before_mean,
                    "snr_before_standard_deviation": before_sd,
                    "snr_after_mean": after_mean,
                    "snr_after_standard_deviation": after_sd,
                    "delta_snr": after_mean - before_mean,
                    "overlap_metrics_per_frame": overlap_rows,
                    "per_plane_channel_snr_metrics": metric_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    output = UNetInferenceOutput(
        dataset=cfg.dataset,
        sample=sample,
        input_zarr=zarr_path,
        mask_zarr=mask_zarr,
        probability_zarr=probability_zarr,
        foreground_zarr=foreground_zarr,
        preview_png=preview_png,
        metrics_json=metrics_json,
    )
    return output, metric_rows


def run_2d_time_unet_on_omezarr(
    zarr_path: Path, cfg: UNetRunConfig, model: tf.keras.Model | None = None
) -> Path:
    cfg.dataset = "2d_time"
    output, _ = run_2d_unet_on_omezarr(zarr_path, cfg, model=model)
    return output.mask_zarr


def run_2d_wga_dapi_unet_on_omezarr(
    zarr_path: Path, cfg: UNetRunConfig, model: tf.keras.Model | None = None
) -> Path:
    cfg.dataset = "2d_wga_dapi"
    output, _ = run_2d_unet_on_omezarr(zarr_path, cfg, model=model)
    return output.mask_zarr


def run_dataset(
    cfg: UNetRunConfig,
    *,
    process_all: bool = True,
    selected_zarrs: list[Path] | None = None,
) -> list[UNetInferenceOutput]:
    """Run one, selected, or all 2D samples and save dataset-level metrics."""
    cfg.dataset = cfg.dataset.strip().lower()
    validate_run_config(cfg)
    if cfg.dataset not in DATASETS_2D:
        raise ValueError(
            f"This updated workflow is restricted to 2D datasets: {DATASETS_2D}"
        )
    cfg.model_path = Path(cfg.model_path or default_model_path(cfg.project_root, cfg.dataset))
    cfg.input_root = Path(cfg.input_root or default_input_root(cfg.project_root, cfg.dataset))
    cfg.out_root = ensure_dir(Path(cfg.out_root or default_out_root(cfg.project_root, cfg.dataset)))
    cfg.raw_root = Path(cfg.raw_root or default_raw_root(cfg.project_root, cfg.dataset))
    cfg.mask_root = Path(cfg.mask_root or default_mask_root(cfg.project_root, cfg.dataset))

    model = load_unet_model(cfg.model_path)
    if selected_zarrs is not None:
        targets = [Path(path) for path in selected_zarrs]
    else:
        available = find_omezarr_dirs(cfg.input_root)
        if not available:
            raise RuntimeError(f"No OME-Zarr folders found under: {cfg.input_root}")
        targets = available if process_all else [available[0]]

    outputs: list[UNetInferenceOutput] = []
    all_metric_rows: list[dict[str, Any]] = []
    for zarr_path in targets:
        output, metric_rows = run_2d_unet_on_omezarr(zarr_path, cfg, model=model)
        outputs.append(output)
        all_metric_rows.extend(metric_rows)

    manifest_rows = [
        {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(output).items()
            }
        }
        for output in outputs
    ]
    _write_csv(cfg.out_root / "unet_2d_inference_manifest.csv", manifest_rows)

    metric_coverage_rows = [
        {
            "dataset": output.dataset,
            "sample": output.sample,
            "metrics_available": output.metrics_json is not None,
            "metrics_json": str(output.metrics_json) if output.metrics_json else "",
            "predicted_mask": str(output.mask_zarr),
            "foreground_filtered": str(output.foreground_zarr) if output.foreground_zarr else "",
            "qc_preview": str(output.preview_png),
        }
        for output in outputs
    ]
    _write_csv(cfg.out_root / "unet_2d_metrics_coverage.csv", metric_coverage_rows)

    if all_metric_rows:
        metrics_csv = cfg.out_root / "unet_2d_inference_metrics.csv"
        _write_csv(metrics_csv, all_metric_rows)
        overlap_by_frame = {
            (str(row["sample"]), int(row["frame_index"])): {
                "dataset": row["dataset"],
                "sample": row["sample"],
                "frame_index": int(row["frame_index"]),
                "iou": float(row["iou"]),
                "dice": float(row["dice"]),
                "reference_mask": row["reference_mask"],
                "predicted_mask": row["predicted_mask"],
            }
            for row in all_metric_rows
        }
        _write_csv(
            cfg.out_root / "unet_2d_overlap_per_frame.csv",
            [overlap_by_frame[key] for key in sorted(overlap_by_frame)],
        )

        sample_summaries: list[dict[str, Any]] = []
        for sample in sorted({str(row["sample"]) for row in all_metric_rows}):
            rows = [row for row in all_metric_rows if row["sample"] == sample]
            per_frame_overlap = {
                int(row["frame_index"]): (float(row["iou"]), float(row["dice"]))
                for row in rows
            }
            iou_mean, iou_sd = _mean_std(
                [values[0] for _, values in sorted(per_frame_overlap.items())]
            )
            dice_mean, dice_sd = _mean_std(
                [values[1] for _, values in sorted(per_frame_overlap.items())]
            )
            before_mean, before_sd = _mean_std([float(row["snr_before"]) for row in rows])
            after_mean, after_sd = _mean_std([float(row["snr_after"]) for row in rows])
            sample_summaries.append(
                {
                    "dataset": cfg.dataset,
                    "sample": sample,
                    "iou_mean": iou_mean,
                    "iou_standard_deviation": iou_sd,
                    "dice_mean": dice_mean,
                    "dice_standard_deviation": dice_sd,
                    "snr_before_mean": before_mean,
                    "snr_before_standard_deviation": before_sd,
                    "snr_after_mean": after_mean,
                    "snr_after_standard_deviation": after_sd,
                    "delta_snr": after_mean - before_mean,
                    "outside_mask_depletion": cfg.outside_mask_depletion,
                    "outside_mask_residual": 1.0 - cfg.outside_mask_depletion,
                }
            )
        summary_csv = cfg.out_root / "unet_2d_per_sample_summary.csv"
        _write_csv(summary_csv, sample_summaries)

        summary_lines = [
            "PFT 2D U-NET INFERENCE SUMMARY",
            "=" * 80,
            f"Dataset: {cfg.dataset}",
            f"Model: {cfg.model_path}",
            f"Input root: {cfg.input_root}",
            f"Samples processed: {len(outputs)}",
            f"Samples with quantitative metrics: {sum(output.metrics_json is not None for output in outputs)}",
            f"Samples without reference metrics: {sum(output.metrics_json is None for output in outputs)}",
            "Normalization: complete image, per channel, P1-P99.8 mapped to [0,1]",
            f"Prediction threshold: {cfg.threshold}",
            f"Outside-mask depletion: {100.0 * cfg.outside_mask_depletion:.1f}%",
            f"Outside-mask residual intensity: {100.0 * (1.0 - cfg.outside_mask_depletion):.1f}%",
            "Saved enhanced image: foreground_filtered.ome.zarr; source dtype and axes retained",
            "Saved image normalization: none; only temporary U-Net inputs are normalized",
            "SNR formula: (mean_signal - mean_background) / (background_sample_SD + epsilon)",
            "SNR before is calculated from the raw image.",
            "SNR after is calculated from the unmasked filtered input image.",
            "The same hand-labelled reference mask defines signal/background ROIs before and after.",
            "The background-suppressed U-Net output is excluded from SNR calculation.",
            "",
            "Sample | IoU mean | Dice mean | SNR raw | SNR filtered unmasked | Delta SNR",
        ]
        for row in sample_summaries:
            summary_lines.append(
                f"{row['sample']} | {row['iou_mean']:.4f} | {row['dice_mean']:.4f} | "
                f"{row['snr_before_mean']:.4f} | {row['snr_after_mean']:.4f} | "
                f"{row['delta_snr']:.4f}"
            )
        (cfg.out_root / "UNET_2D_INFERENCE_SUMMARY.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

    return outputs
