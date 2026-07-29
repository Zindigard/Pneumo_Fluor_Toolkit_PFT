"""
2.5D U-Net inference that saves probabilities and masks only.

"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PFT.core_prog_parts.common_paths import ensure_dir, find_project_root
from PFT.core_prog_parts.denoising.metadata_3d import (
    coordinate_scale_for_level,
    copyable_root_metadata,
)
from PFT.core_prog_parts.denoising.validation_3d import (
    DEFAULT_TRAINING_SLICES_1BASED,
    find_slice_mask,
)
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (
    bce_dice_loss,
    dice_coef,
    iou_coef,
)
from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import (
    UNet25DTrainConfig,
    open_3d_image_czyx,
    predict_25d_probability,
    read_binary_slice_mask,
)


@dataclass
class UNet25DRunConfig:
    """Configuration for mask-only 2.5D inference."""

    project_root: Path = find_project_root(Path(__file__).resolve())
    input_zarr: Path | None = None
    model_path: Path | None = None
    output_root: Path | None = None
    manual_mask_root: Path | None = None
    level: int = 0
    patch: int = 256
    channels: tuple[int, ...] | None = None
    normalize: str = "percentile"
    threshold: float = 0.5
    predict_batch_size: int = 8
    preview_slices_1based: tuple[int, ...] = DEFAULT_TRAINING_SLICES_1BASED
    save_probability: bool = True


@dataclass(frozen=True)
class UNet25DRunOutput:
    """Saved products from one mask-only inference run."""

    sample: str
    source_zarr: Path
    mask_zarr: Path
    probability_zarr: Path | None
    preview_dir: Path
    report_json: Path


def load_25d_model(model_path: Path) -> tf.keras.Model:
    """Load the 2.5D U-Net with custom loss and metric objects."""
    if not model_path.is_file():
        raise FileNotFoundError(f"2.5D model not found: {model_path}")
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
        },
    )


def _zyx_scale(input_zarr: Path, level: int) -> list[float]:
    full_scale = coordinate_scale_for_level(input_zarr, level=level)
    # The supported source layout is CZYX, so remove channel scale.
    if len(full_scale) != 4:
        raise ValueError(f"Expected CZYX coordinate scale, got {full_scale}")
    return [float(value) for value in full_scale[1:]]


def _display_composite(image_czyx: np.ndarray, z_index: int) -> np.ndarray:
    central = image_czyx[:, z_index]
    output = np.zeros((*central.shape[1:], 3), dtype=np.float32)
    destinations = (2, 1, 0)
    for channel in range(min(3, central.shape[0])):
        low, high = np.percentile(central[channel], (1.0, 99.8))
        if high <= low:
            high = low + 1.0
        output[..., destinations[channel]] = np.clip((central[channel] - low) / (high - low), 0, 1)
    return output


def _save_mask_previews(
    *,
    image_czyx: np.ndarray,
    probability_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    sample: str,
    output_dir: Path,
    slices_1based: Sequence[int],
    manual_mask_root: Path | None,
) -> list[Path]:
    """Save mask QC only for the manually planned target slices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for slice_number in slices_1based:
        z_index = slice_number - 1
        if not 0 <= z_index < mask_zyx.shape[0]:
            continue
        reference = None
        reference_path = None
        if manual_mask_root is not None:
            reference_path = find_slice_mask(manual_mask_root / sample, slice_number)
            if reference_path is not None:
                reference = read_binary_slice_mask(reference_path, mask_zyx.shape[-2:])
        columns = 5 if reference is not None else 4
        figure, axes = plt.subplots(1, columns, figsize=(4 * columns, 4), constrained_layout=True)
        axes[0].imshow(_display_composite(image_czyx, z_index))
        axes[0].set_title("Original 3-channel slice")
        axes[1].imshow(probability_zyx[z_index], cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("Foreground probability")
        axes[2].imshow(mask_zyx[z_index], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Predicted mask (0.5)")
        axes[3].imshow(_display_composite(image_czyx, z_index))
        axes[3].contour(mask_zyx[z_index], levels=[0.5], colors="white", linewidths=0.7)
        axes[3].set_title("Predicted-mask overlay")
        if reference is not None:
            axes[4].imshow(reference, cmap="gray", vmin=0, vmax=1)
            axes[4].contour(mask_zyx[z_index], levels=[0.5], colors="red", linewidths=0.6)
            axes[4].set_title("Manual mask + prediction contour")
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"{sample} | Z{slice_number:03d}")
        path = output_dir / f"{sample}__z{slice_number:03d}__predicted_mask_qc.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved.append(path)
    return saved


def run_3d_25d_unet_masks(cfg: UNet25DRunConfig) -> UNet25DRunOutput:
    """Predict all Z-slice masks and save no intensity-modified image."""
    if cfg.input_zarr is None:
        raise ValueError("input_zarr is required")
    input_zarr = Path(cfg.input_zarr).resolve()
    model_path = Path(cfg.model_path or cfg.project_root / "models" / "u_net_3d_25d" / "u_net_3d_25d_best.keras")
    output_root = Path(cfg.output_root or cfg.project_root / "results" / "U-net" / "3d_25d")
    if cfg.threshold != 0.5:
        raise ValueError("The agreed mask threshold is fixed at 0.5")
    image = open_3d_image_czyx(input_zarr, level=cfg.level)
    sample = input_zarr.parent.name
    model = load_25d_model(model_path)
    expected_channels = int(model.input_shape[-1])
    selected_count = len(cfg.channels) if cfg.channels is not None else image.shape[0]
    actual_channels = selected_count * 3
    if actual_channels != expected_channels:
        raise ValueError(
            f"Model expects {expected_channels} input channels, but inference produces {actual_channels}. "
            "Use the same fluorescence-channel selection as training."
        )

    training_cfg = UNet25DTrainConfig(
        project_root=cfg.project_root,
        level=cfg.level,
        patch=cfg.patch,
        channels=cfg.channels,
        normalize=cfg.normalize,
        threshold=cfg.threshold,
        predict_batch_size=cfg.predict_batch_size,
        z_radius=1,
    )
    probability = np.empty(tuple(int(value) for value in image.shape[1:]), dtype=np.float32)
    normalization_cache: dict[tuple[int, int], tuple[float, float]] = {}
    for z_index in range(int(image.shape[1])):
        print(f"[U-NET MASK] {sample}: Z{z_index + 1:03d}/{image.shape[1]:03d}")
        probability[z_index] = predict_25d_probability(
            model, image, z_index, training_cfg,
            normalization_cache=normalization_cache,
        )
    mask = (probability >= cfg.threshold).astype(np.uint8)

    sample_dir = ensure_dir(output_root / sample)
    retained = copyable_root_metadata(input_zarr)
    # The saved arrays are ZYX semantic products without a channel axis. Keep
    # source-channel metadata as provenance, not as misleading top-level image
    # channel metadata.
    source_channel_names = retained.pop("channel_names", None)
    source_omero = retained.pop("omero", None)
    processing = {
        "operation": "unet_3d_25d_semantic_foreground_prediction",
        "source_omezarr": str(input_zarr),
        "source_level": int(cfg.level),
        "model_path": str(model_path),
        "z_context": [-1, 0, 1],
        "selected_channels": list(cfg.channels) if cfg.channels is not None else "all",
        "threshold": 0.5,
        "normalization": cfg.normalize,
        "intensity_image_saved": False,
        "source_channel_names": source_channel_names,
        "source_omero": source_omero,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    mask_attrs = dict(retained)
    mask_attrs["pft_processing"] = processing
    mask_zarr = save_ome_zarr(
        sample_dir / "pred_mask.ome.zarr",
        mask,
        "zyx",
        overwrite=True,
        pyramid_3d=False,
        pyramid_max_layer=0,
        coordinate_scale=_zyx_scale(input_zarr, cfg.level),
        extra_attrs=mask_attrs,
    )

    probability_zarr: Path | None = None
    if cfg.save_probability:
        probability_attrs = dict(retained)
        probability_attrs["pft_processing"] = {**processing, "operation": "unet_3d_25d_foreground_probability"}
        probability_zarr = save_ome_zarr(
            sample_dir / "foreground_probability.ome.zarr",
            probability,
            "zyx",
            overwrite=True,
            pyramid_3d=False,
            pyramid_max_layer=0,
            coordinate_scale=_zyx_scale(input_zarr, cfg.level),
            extra_attrs=probability_attrs,
        )

    preview_dir = sample_dir / "qc_training_slices"
    previews = _save_mask_previews(
        image_czyx=image,
        probability_zyx=probability,
        mask_zyx=mask,
        sample=sample,
        output_dir=preview_dir,
        slices_1based=cfg.preview_slices_1based,
        manual_mask_root=Path(cfg.manual_mask_root) if cfg.manual_mask_root else None,
    )
    report = {
        "sample": sample,
        "source_omezarr": str(input_zarr),
        "source_level": cfg.level,
        "model_path": str(model_path),
        "mask_zarr": str(mask_zarr),
        "probability_zarr": str(probability_zarr) if probability_zarr else None,
        "mask_shape_zyx": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "threshold": 0.5,
        "foreground_voxels": int(np.count_nonzero(mask)),
        "background_voxels": int(mask.size - np.count_nonzero(mask)),
        "intensity_modification_performed": False,
        "preview_slices_1based": list(cfg.preview_slices_1based),
        "preview_files": [str(path) for path in previews],
        "configuration": asdict(cfg),
    }
    report_json = sample_dir / "u_net_3d_25d_mask_report.json"
    report_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return UNet25DRunOutput(sample, input_zarr, mask_zarr, probability_zarr, preview_dir, report_json)


__all__ = [
    "UNet25DRunConfig",
    "UNet25DRunOutput",
    "load_25d_model",
    "run_3d_25d_unet_masks",
]
