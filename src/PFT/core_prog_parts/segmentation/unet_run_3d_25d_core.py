"""Per-volume-target merged-RGB 2.5D U-Net inference.

For each source volume, the configured target Zn is inferred from the
wavelength-mapped merged-RGB context Z(n-1)/Zn/Z(n+1). The predicted 2D mask
is then broadcast through the full Z-stack and saved as a ZYX OME-Zarr mask.
"""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile as tiff

from PFT.core_prog_parts.common_paths import ensure_dir, find_project_root
from PFT.core_prog_parts.denoising.metadata_3d import (
    coordinate_scale_for_level,
    copyable_root_metadata,
)
from PFT.core_prog_parts.denoising.validation_3d import (
    TARGET_SLICE_BY_VOLUME_KEY,
    annotation_sample_dir,
    find_slice_mask,
    relative_volume_path,
    target_slice_for_volume,
    volume_key,
)
from PFT.core_prog_parts.omezarr_utils import save_ome_zarr
from PFT.core_prog_parts.segmentation.unet_train_2d_time_core import (
    bce_dice_loss,
    dice_coef,
    iou_coef,
)
from PFT.core_prog_parts.segmentation.unet_train_3d_25d_core import (
    MODEL_CONTRACT,
    UNet25DTrainConfig,
    make_merged_rgb_context_slice,
    open_3d_image_czyx,
    predict_25d_probability,
    read_binary_slice_mask,
    resolve_rgb_source_channels,
)



@dataclass
class UNet25DRunConfig:
    """Configuration for one configured target-slice inference."""

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
    save_probability: bool = True


@dataclass(frozen=True)
class UNet25DRunOutput:
    """Saved products from one configured target-slice inference run."""

    sample: str
    source_zarr: Path
    mask_zarr: Path
    probability_zarr: Path | None
    source_mask_tif: Path
    source_slice_dir: Path
    preview_dir: Path
    report_json: Path
    reference_mask_tif: Path | None
    reference_metrics_status: str
    reference_metrics: dict[str, float] | None



def load_25d_model(model_path: Path) -> tf.keras.Model:
    """Load the merged-RGB 2.5D U-Net with custom objects."""
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



def _validate_model_contract(model_path: Path) -> Path:
    """Require a model trained with the current per-volume target map."""
    summary_path = model_path.parent / "u_net_3d_25d_training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Model contract summary is missing: {summary_path}. Retrain the model "
            "with the per-volume-target merged-RGB training script before inference."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("model_contract") != MODEL_CONTRACT:
        raise ValueError(
            f"Model summary contract is {summary.get('model_contract')!r}, expected "
            f"{MODEL_CONTRACT!r}. Retrain with the current per-volume-target workflow."
        )
    saved_map = summary.get("target_slice_map")
    if saved_map != TARGET_SLICE_BY_VOLUME_KEY:
        raise ValueError(
            "The model target-slice map differs from the current project mapping. "
            "Retrain the model before inference."
        )
    if int(summary.get("model_input_channels", -1)) != 9:
        raise ValueError(
            "Model summary does not describe a nine-channel merged-RGB input: "
            f"{summary.get('model_input_channels')}"
        )
    return summary_path



def _zyx_scale(input_zarr: Path, level: int) -> list[float]:
    full_scale = coordinate_scale_for_level(input_zarr, level=level)
    if len(full_scale) != 4:
        raise ValueError(f"Expected CZYX coordinate scale, got {full_scale}")
    return [float(value) for value in full_scale[1:]]



def _yx_scale(input_zarr: Path, level: int) -> list[float]:
    return _zyx_scale(input_zarr, level)[1:]



def _validate_target_slice(slice_number: int, z_count: int) -> int:
    """Validate that the mapped target has complete Z-1/Z/Z+1 context."""
    slice_number = int(slice_number)
    if slice_number < 2 or slice_number > z_count - 1:
        raise ValueError(
            f"Configured Z{slice_number} lacks complete Z-1/Z/Z+1 context "
            f"for a stack with Z={z_count}."
        )
    return slice_number


def _binary_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Calculate binary overlap metrics when a manual reference mask exists."""
    reference_bool = reference.astype(bool)
    prediction_bool = prediction.astype(bool)
    intersection = float(np.count_nonzero(reference_bool & prediction_bool))
    union = float(np.count_nonzero(reference_bool | prediction_bool))
    reference_count = float(np.count_nonzero(reference_bool))
    prediction_count = float(np.count_nonzero(prediction_bool))
    false_positive = float(np.count_nonzero(~reference_bool & prediction_bool))
    false_negative = float(np.count_nonzero(reference_bool & ~prediction_bool))
    return {
        "iou": intersection / (union + epsilon),
        "dice": 2.0 * intersection / (reference_count + prediction_count + epsilon),
        "precision": intersection / (intersection + false_positive + epsilon),
        "recall": intersection / (intersection + false_negative + epsilon),
    }


def _load_optional_reference_mask(
    *,
    manual_mask_dir: Path | None,
    slice_number: int,
    expected_yx: tuple[int, int],
    sample: str,
) -> tuple[Path | None, np.ndarray | None, str]:
    """Load a reference mask if present; otherwise warn and skip metrics."""
    if manual_mask_dir is None:
        warnings.warn(
            f"{sample}: no manual-mask root was provided; reference metrics are skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, "skipped_no_manual_mask_root"

    reference_path = find_slice_mask(manual_mask_dir, slice_number)
    if reference_path is None:
        warnings.warn(
            f"{sample}: reference mask Z{slice_number:03d} is missing; inference continues "
            "and Dice/IoU/precision/recall are skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, "skipped_missing_reference_mask"

    reference = read_binary_slice_mask(reference_path, expected_yx)
    return reference_path, reference, "calculated"



def _save_source_slice_products(
    *,
    source_slice_dir: Path,
    slice_number: int,
    probability: np.ndarray,
    mask: np.ndarray,
) -> Path:
    """Save the target-slice probability map and strict binary mask."""
    source_slice_dir.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(
        source_slice_dir / f"z{slice_number:03d}_foreground_probability.tif",
        probability.astype(np.float32),
    )
    source_mask_tif = source_slice_dir / f"z{slice_number:03d}_pred_mask.tif"
    tiff.imwrite(source_mask_tif, mask.astype(np.uint8))
    return source_mask_tif



def _save_source_mask_preview(
    *,
    image_czyx,
    training_cfg: UNet25DTrainConfig,
    rgb_source_channels: tuple[int, int, int],
    slice_number: int,
    probability: np.ndarray,
    mask: np.ndarray,
    sample: str,
    output_dir: Path,
    reference: np.ndarray | None,
    reference_metrics: dict[str, float] | None,
) -> list[Path]:
    """Save the configured target-slice prediction quality-control figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization_cache: dict[tuple[int, int], tuple[float, float]] = {}
    context = make_merged_rgb_context_slice(
        image_czyx,
        slice_number - 1,
        training_cfg,
        rgb_source_channels,
        normalization_cache=normalization_cache,
    )
    merged_image = context[1]
    columns = 5 if reference is not None else 4
    figure, axes = plt.subplots(
        1,
        columns,
        figsize=(4 * columns, 4),
        constrained_layout=True,
    )
    axes[0].imshow(merged_image)
    axes[0].set_title(f"Merged RGB Z{slice_number}")
    axes[1].imshow(probability, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Foreground probability")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask (0.5)")
    axes[3].imshow(merged_image)
    axes[3].contour(mask, levels=[0.5], colors="white", linewidths=0.7)
    axes[3].set_title("Predicted-mask overlay")
    if reference is not None:
        axes[4].imshow(reference, cmap="gray", vmin=0, vmax=1)
        axes[4].contour(mask, levels=[0.5], colors="red", linewidths=0.6)
        axes[4].set_title(
            "Manual + prediction\n"
            f"IoU={reference_metrics['iou']:.3f}, Dice={reference_metrics['dice']:.3f}"
        )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    title = f"{sample} | source inference Z{slice_number:03d}"
    if reference is None:
        title += " | no reference mask: metrics skipped"
    figure.suptitle(title)
    safe_sample = sample.replace("/", "__").replace("\\", "__")
    path = output_dir / f"{safe_sample}__z{slice_number:03d}__predicted_mask_qc.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return [path]



def run_3d_25d_unet_masks(cfg: UNet25DRunConfig) -> UNet25DRunOutput:
    """Infer the configured target and broadcast its mask to the full Z-stack."""
    if cfg.input_zarr is None:
        raise ValueError("input_zarr is required")
    input_zarr = Path(cfg.input_zarr).resolve()
    model_path = Path(
        cfg.model_path
        or cfg.project_root / "models" / "u_net_3d_25d" / "u_net_3d_25d_best.keras"
    )
    output_root = Path(
        cfg.output_root or cfg.project_root / "results" / "U-net" / "3d_25d"
    )
    if cfg.threshold != 0.5:
        raise ValueError("The agreed mask threshold is fixed at 0.5")
    if cfg.channels is not None:
        raise ValueError(
            "Merged-RGB inference always uses all three wavelength-mapped channels."
        )

    image = open_3d_image_czyx(input_zarr, level=cfg.level)
    if int(image.shape[0]) != 3:
        raise ValueError(f"Merged RGB inference requires C=3, received C={image.shape[0]}")
    slice_number = _validate_target_slice(
        target_slice_for_volume(input_zarr),
        int(image.shape[1]),
    )
    relative_volume = relative_volume_path(input_zarr)
    sample = volume_key(input_zarr)
    model_summary_path = _validate_model_contract(model_path)
    model = load_25d_model(model_path)
    expected_channels = int(model.input_shape[-1])
    if expected_channels != 9:
        raise ValueError(
            f"Merged-RGB model must expect 9 channels, but model input is {expected_channels}. "
            "Retrain with the per-volume-target merged-RGB training code."
        )
    rgb_source_channels = resolve_rgb_source_channels(input_zarr, level=cfg.level)

    training_cfg = UNet25DTrainConfig(
        project_root=cfg.project_root,
        level=cfg.level,
        patch=cfg.patch,
        channels=None,
        normalize=cfg.normalize,
        threshold=cfg.threshold,
        predict_batch_size=cfg.predict_batch_size,
        z_radius=1,
    )
    print(
        f"[U-NET MASK] {sample}: infer Z{slice_number:03d} from "
        f"Z{slice_number - 1}/Z{slice_number}/Z{slice_number + 1} merged RGB"
    )
    probability = predict_25d_probability(
        model,
        image,
        slice_number - 1,
        training_cfg,
        rgb_source_channels,
        normalization_cache={},
    )
    mask = (probability >= cfg.threshold).astype(np.uint8)
    if not np.any(mask) or np.all(mask):
        raise ValueError("The predicted target-slice mask must contain both foreground and background.")

    z_count = int(image.shape[1])
    mask_zyx = np.broadcast_to(mask[None, ...], (z_count, *mask.shape)).copy()

    sample_dir = ensure_dir(output_root / relative_volume)
    stale_products = (
        sample_dir / "source_slice_predictions",
        sample_dir / "qc_source_slices_and_combination",
        sample_dir / "combined_foreground_probability.ome.zarr",
        sample_dir / "source_slice_prediction",
        sample_dir / "qc_z010_prediction",
        sample_dir / f"qc_z{slice_number:03d}_prediction",
    )
    for stale_path in stale_products:
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.exists():
            stale_path.unlink()
    source_slice_dir = ensure_dir(sample_dir / "source_slice_prediction")
    source_mask_tif = _save_source_slice_products(
        source_slice_dir=source_slice_dir,
        slice_number=slice_number,
        probability=probability,
        mask=mask,
    )

    retained = copyable_root_metadata(input_zarr)
    source_channel_names = retained.pop("channel_names", None)
    source_omero = retained.pop("omero", None)
    processing = {
        "operation": "unet_3d_25d_configured_target_foreground_mask",
        "source_omezarr": str(input_zarr),
        "source_level": int(cfg.level),
        "model_path": str(model_path),
        "model_summary_path": str(model_summary_path),
        "model_contract": MODEL_CONTRACT,
        "inference_slice_1based": int(slice_number),
        "inference_context_1based": [slice_number - 1, slice_number, slice_number + 1],
        "input_representation": (
            f"wavelength-mapped merged RGB at Z{slice_number - 1}/"
            f"Z{slice_number}/Z{slice_number + 1}"
        ),
        "rgb_source_channels": {
            "red": int(rgb_source_channels[0]),
            "green": int(rgb_source_channels[1]),
            "blue": int(rgb_source_channels[2]),
        },
        "rgb_wavelengths_nm": {"red": 561.0, "green": 488.0, "blue": 405.0},
        "source_mask_broadcast_to_all_z": True,
        "broadcast_z_count": z_count,
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
        mask_zyx,
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
        probability_attrs["pft_processing"] = {
            **processing,
            "operation": "unet_3d_25d_configured_target_probability_2d",
            "stored_axes": "yx",
            "broadcast_to_z": False,
        }
        probability_zarr = save_ome_zarr(
            sample_dir / f"z{slice_number:03d}_foreground_probability.ome.zarr",
            probability,
            "yx",
            overwrite=True,
            pyramid_3d=False,
            pyramid_max_layer=0,
            coordinate_scale=_yx_scale(input_zarr, cfg.level),
            extra_attrs=probability_attrs,
        )

    manual_mask_dir = (
        annotation_sample_dir(Path(cfg.manual_mask_root), input_zarr)
        if cfg.manual_mask_root
        else None
    )
    reference_mask_tif, reference, reference_metrics_status = _load_optional_reference_mask(
        manual_mask_dir=manual_mask_dir,
        slice_number=slice_number,
        expected_yx=mask.shape,
        sample=sample,
    )
    reference_metrics = (
        _binary_metrics(reference, mask)
        if reference is not None
        else None
    )
    preview_dir = sample_dir / f"qc_z{slice_number:03d}_prediction"
    previews = _save_source_mask_preview(
        image_czyx=image,
        training_cfg=training_cfg,
        rgb_source_channels=rgb_source_channels,
        slice_number=slice_number,
        probability=probability,
        mask=mask,
        sample=sample,
        output_dir=preview_dir,
        reference=reference,
        reference_metrics=reference_metrics,
    )

    foreground_pixels = int(np.count_nonzero(mask))
    report = {
        "sample": sample,
        "source_omezarr": str(input_zarr),
        "source_level": cfg.level,
        "model_path": str(model_path),
        "model_summary_path": str(model_summary_path),
        "model_contract": MODEL_CONTRACT,
        "mask_zarr": str(mask_zarr),
        "probability_zarr": str(probability_zarr) if probability_zarr else None,
        "source_mask_tif": str(source_mask_tif),
        "source_slice_dir": str(source_slice_dir),
        "inference_slice_1based": int(slice_number),
        "inference_context_1based": [slice_number - 1, slice_number, slice_number + 1],
        "foreground_pixels": foreground_pixels,
        "background_pixels": int(mask.size - foreground_pixels),
        "full_mask_shape_zyx": list(mask_zyx.shape),
        "full_mask_dtype": str(mask_zyx.dtype),
        "broadcast_to_all_z": True,
        "threshold": 0.5,
        "reference_mask_tif": str(reference_mask_tif) if reference_mask_tif else None,
        "reference_metrics_status": reference_metrics_status,
        "reference_metrics": reference_metrics,
        "intensity_modification_performed": False,
        "preview_files": [str(path) for path in previews],
        "configuration": asdict(cfg),
    }
    report_json = sample_dir / "u_net_3d_25d_mask_report.json"
    report_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return UNet25DRunOutput(
        sample=sample,
        source_zarr=input_zarr,
        mask_zarr=mask_zarr,
        probability_zarr=probability_zarr,
        source_mask_tif=source_mask_tif,
        source_slice_dir=source_slice_dir,
        preview_dir=preview_dir,
        report_json=report_json,
        reference_mask_tif=reference_mask_tif,
        reference_metrics_status=reference_metrics_status,
        reference_metrics=reference_metrics,
    )


__all__ = [
    "MODEL_CONTRACT",
    "UNet25DRunConfig",
    "UNet25DRunOutput",
    "load_25d_model",
    "run_3d_25d_unet_masks",
]
